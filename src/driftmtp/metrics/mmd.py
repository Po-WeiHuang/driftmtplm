"""Maximum Mean Discrepancy via `ignite.metrics.MaximumMeanDiscrepancy`.

https://docs.pytorch.org/ignite/generated/ignite.metrics.MaximumMeanDiscrepancy.html

Per user decision, this replaces an earlier from-scratch biased-MMD^2 estimator with the
pytorch-ignite implementation. New dependency: `pytorch-ignite==0.5.5`, recorded in
`third_party/mtp-lm-patch/conda_env.yml` and `requirements-lock.txt` (pure Python, only
requires `torch` and `packaging` -- no native compilation).

Two things confirmed correct as-is:
- ignite computes the UNBIASED MMD^2 U-statistic (excludes i==j self-pairs from the XX/YY
  sums) -- the correct form for this use.
- ignite's `compute()` returns `sqrt(clamp(MMD^2, min=0))`, i.e. MMD (not MMD^2) -- the
  quantity to log/plot (mean/std over training) is MMD.

Kernel bandwidth policy (`var`, i.e. sigma^2): ignite has no built-in auto-bandwidth, and a
naive per-call median-heuristic recompute would let the kernel scale silently drift across
training steps and across horizons, making a dropping MMD trend ambiguous between genuine
distributional convergence and a shifting measuring stick. Per user decision, the bandwidth
is instead:
  1. shared across ALL anchor horizons within a step (computed once from horizon-1 features
     only, the "first token"), so MMD is directly comparable horizon-to-horizon; and
  2. recomputed via `median_heuristic_bandwidth` on every diagnostic step while
     `step < warmup_steps`, then FROZEN at its last recomputed value for every step
     thereafter, so the MMD trend over training is measured against one fixed baseline
     after warmup -- mirroring the warmup-then-freeze pattern already used for lambda_R in
     claudedriftingplan.md's gradient-matching section.
`MMDBandwidthSchedule` implements this policy; `maximum_mean_discrepancy` itself just takes
whatever fixed bandwidth it's given, with no default, so the policy can't be silently
bypassed by an accidental per-call recompute.

TRAINING DIAGNOSTICS NO LONGER USE ignite (docs/geometry_diagnostics_report.md). Student and
teacher states are PAIRED (s_i, t_i are the same token), and ignite keeps the n paired terms
k(s_i, t_i) in its cross term, which biases MMD^2 downward; its sqrt(clamp(.)) then logs most
steps as exactly 0. `paired_maximum_mean_discrepancy` is the replacement: ignite's estimator
with those paired terms added back (Gretton et al. 2012, Lemma 6), in float32.
`maximum_mean_discrepancy` remains only for the eval pipeline (condrollouteval). `MMDBandwidthSchedule` is checkpointable (`state_dict`/`load_state_dict`).
"""

from typing import Dict, Optional

import torch
from ignite.metrics import MaximumMeanDiscrepancy as _IgniteMMD


@torch.no_grad()
def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix, (Nx, d) x (Ny, d) -> (Nx, Ny)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True).t()
    d2 = x2 + y2 - 2.0 * (x @ y.t())
    return d2.clamp_min(0.0)


@torch.no_grad()
def median_heuristic_bandwidth(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Median of pairwise squared distances within the pooled {x, y} set.

    Standard MMD bandwidth default (Gretton et al. 2012). Excludes the zero
    self-distances on the diagonal. Not meant to be called fresh on every
    diagnostic step directly -- see `MMDBandwidthSchedule`, which calls this
    only during warmup and freezes the result afterward.
    """
    pooled = torch.cat([x, y], dim=0)
    d2 = _pairwise_sq_dists(pooled, pooled)
    n = d2.shape[0]
    off_diag = d2[~torch.eye(n, dtype=torch.bool, device=d2.device)]
    return off_diag.median().clamp_min(1e-12)


@torch.no_grad()
def median_heuristic_bandwidth_from_sq_dists(
    d2_xx: torch.Tensor, d2_yy: torch.Tensor, d2_xy: torch.Tensor
) -> torch.Tensor:
    """`median_heuristic_bandwidth` from precomputed squared-distance blocks.

    The off-diagonal entries of the pooled {x, y} matrix are exactly the
    off-diagonal entries of XX and YY plus every XY entry twice (XY and its
    transpose YX), so this returns the same median without rebuilding the
    pooled (2N, 2N) matrix.
    """
    n_x, n_y = d2_xx.shape[0], d2_yy.shape[0]
    off_x = ~torch.eye(n_x, dtype=torch.bool, device=d2_xx.device)
    off_y = ~torch.eye(n_y, dtype=torch.bool, device=d2_yy.device)
    xy = d2_xy.reshape(-1)
    pooled = torch.cat([d2_xx[off_x], d2_yy[off_y], xy, xy])
    return pooled.median().clamp_min(1e-12)


class MMDBandwidthSchedule:
    """Tracks the single MMD kernel bandwidth shared across all anchor horizons.

    Recomputed via `median_heuristic_bandwidth` from horizon-1 (student, teacher)
    features on every `update()` call while `step < warmup_steps`; frozen at
    whatever value it held at the last such call for every step afterward.
    One instance should be created once (e.g. alongside the other running-metric
    trackers before the training loop) and reused across the whole run, so its
    frozen state persists.
    """

    def __init__(self, warmup_steps: int):
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        self.warmup_steps = warmup_steps
        self._value: Optional[torch.Tensor] = None

    def update(self, step: int, x_h1: torch.Tensor, y_h1: torch.Tensor) -> torch.Tensor:
        """Call once per diagnostic step with horizon-1's (student, teacher) features.

        Args:
            step: current training step/iteration count.
            x_h1: (N, d) student features at horizon 1 ("first token").
            y_h1: (N, d) teacher features at horizon 1, same `N` as `x_h1`.

        Returns:
            The bandwidth to use this step, for every horizon.
        """
        if self._value is None or step < self.warmup_steps:
            self._value = median_heuristic_bandwidth(x_h1, y_h1)
        return self._value

    def state_dict(self) -> Dict:
        value = None if self._value is None else self._value.detach().float().cpu()
        return {"warmup_steps": self.warmup_steps, "value": value}

    def load_state_dict(self, state: Dict) -> None:
        self._value = state["value"]

    def update_from_sq_dists(
        self, step: int, d2_xx: torch.Tensor, d2_yy: torch.Tensor, d2_xy: torch.Tensor
    ) -> torch.Tensor:
        """`update`, for a caller that already holds horizon-1's squared-distance blocks."""
        if self._value is None or step < self.warmup_steps:
            self._value = median_heuristic_bandwidth_from_sq_dists(d2_xx, d2_yy, d2_xy)
        return self._value

    @property
    def is_frozen(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError(
                "MMDBandwidthSchedule.update() must be called at least once before .value is read."
            )
        return self._value


@torch.no_grad()
def paired_maximum_mean_discrepancy(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: torch.Tensor,
) -> torch.Tensor:
    """ignite's MMD with the paired cross terms k(x_i, y_i) added back.

    x_i and y_i are PAIRED (student and teacher state of the same token). ignite's cross
    term XY = mean over ALL n^2 pairs includes the n paired kernels, which are large because
    paired states are close, and MMD^2 = XX + YY - 2 XY subtracts them. That biases MMD^2
    downward, so after ignite's clamp most steps log exactly 0. Here XY averages only the
    i != j pairs (normalised by n(n-1)), which adds the paired kernels back; XX and YY are
    ignite's own (self pairs excluded). This is Gretton et al. 2012, Lemma 6.

    Runs ignite's `update` on float32 copies (bf16 sums quantise the result) and reads its
    accumulators instead of calling `compute()`, which would also all-reduce across ranks.

    Returns sqrt(clamp(MMD^2, 0)), the same quantity ignite reports.
    """
    if x.dim() != 2 or x.shape != y.shape:
        raise ValueError(f"x, y must be matching (N, d); got {tuple(x.shape)}, {tuple(y.shape)}")
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least 2 paired samples")
    x32, y32 = x.float(), y.float()
    var = float(bandwidth)

    metric = _IgniteMMD(var=var, device=x.device)
    metric.update((x32, y32))
    # ignite 0.5.5 accumulators after one update: XX, YY (diagonal excluded) and XY (all n^2).
    xx, yy, xy_all = metric._xx_sum, metric._yy_sum, metric._xy_sum

    k_pair = torch.exp(-0.5 * (x32 - y32).square().sum(dim=-1) / var).sum()
    xy = (xy_all * (n * n) - k_pair) / (n * (n - 1))
    return (xx + yy - 2.0 * xy).clamp_min(0.0).sqrt()


@torch.no_grad()
def maximum_mean_discrepancy(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: torch.Tensor,
) -> torch.Tensor:
    """MMD between x and y via `ignite.metrics.MaximumMeanDiscrepancy`.

    Args:
        x: (N, d) student features.
        y: (N, d) teacher features. MUST have the same `N` as `x` --
            ignite's metric requires `x.shape == y.shape`. Guaranteed at the
            DriftMTP training call site by the shared `pred_pos_mask`
            slicing (see `features.extract_predictive_states`).
        bandwidth: the fixed squared-distance bandwidth (ignite's `var`) to
            use -- required, no default. Obtain it from a single shared
            `MMDBandwidthSchedule` rather than recomputing it per call; see
            module docstring.

    Returns:
        Scalar tensor: ignite's `compute()` output --
        `sqrt(clamp(unbiased MMD^2, min=0))` with a Gaussian RBF kernel of
        bandwidth `var`. Lower is better; `0` in expectation when `x` and
        `y` are drawn from the same distribution.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"x, y must be (N, d); got {tuple(x.shape)}, {tuple(y.shape)}")
    if x.shape != y.shape:
        raise ValueError(
            f"ignite.metrics.MaximumMeanDiscrepancy requires x.shape == y.shape, "
            f"got {tuple(x.shape)} vs {tuple(y.shape)}"
        )

    metric = _IgniteMMD(var=float(bandwidth), device=x.device)
    metric.update((x, y))
    return torch.tensor(metric.compute(), device=x.device, dtype=x.dtype)
