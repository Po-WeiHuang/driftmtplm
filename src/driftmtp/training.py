"""Phase 4 training-time integration: token drifting loss + stability logging.

Implements `claudedriftingplan.md`'s PHASE 4 -- Training Integration:

* `compute_drift_loss` wraps `features.extract_predictive_states` +
  `drifting.token_drifting_loss` into the one call the patched `pretrain.py`
  makes per iteration, mirroring how Phase 2's `validation.py` wraps the
  same primitives for the diagnostics path. Unlike the diagnostics path,
  the student extraction here is NOT detached -- `L_drift`'s gradient must
  reach the student parameters through `x = phi_S(u^S_{r,j})`.

* `DriftAccumulationTracker` implements the "Gradient-accumulation
  stability logging" note under PHASE 4: `S_j` and `lambda_{tau,j}` (and,
  by the same reasoning, per-temperature raw/normalized field RMS) are each
  a nonlinear function of the R-sized micro-batch pool, so their spread
  across the `gradient_accumulation_iters` micro-substeps of one optimizer
  step -- not just their mean -- is the concrete signal for whether R is
  large enough for stable kernel statistics at a given horizon. The spec
  explicitly requires this mean/std/ratio triplet for `S_j` and
  `lambda_{tau,j}`; this implementation applies the same triplet uniformly
  to raw/normalized field RMS too (a superset of the letter of the spec),
  since those are equally a nonlinear function of the same R-sized pool and
  a mean-only view would silently hide the same small-R instability the
  spec calls out `S_j`/`lambda_{tau,j}` for.

  Within one micro-substep, `token_drifting_loss` produces one `S_j` and
  one `lambda_{tau,j}` per active horizon `j` (see `drifting.py`). The
  tracker does NOT pool these across horizons into a single whole-window
  scalar: `k_toks` (the number of active horizons) is randomized per
  optimizer step, and higher horizons are expected to have inherently worse
  student/teacher distributional match, so "the average over all active
  horizons" would mix a different horizon population every time `k_toks`
  changes -- not a fair "is this improving" curve to read as iteration goes
  up. `DriftAccumulationTracker` therefore ONLY tracks a per-anchor-horizon
  breakdown -- using the same `features.anchor_horizons()` set already
  established for Phase 2's ECE/MMD/Sinkhorn diagnostics (absolute horizons
  `{1, 2, 3, mid, k_toks}`, capped at 5) -- so a fixed horizon's
  `S_j`/`lambda_{tau,j}`/field-RMS can be read as one continuous, comparable
  time series regardless of what `k_toks` was on any given step. `k_toks`
  itself is tracked with the same mean/std/ratio treatment for the same
  reason `S_j` and the rest are: it lets a reader correlate the two, and
  doubles as a self-check (constant `k_toks` within one window is a
  structural guarantee of the caller -- see below -- so a nonzero std here
  would mean that guarantee broke).

  A horizon that drops out of the anchor set (because a later window's
  `k_toks` no longer produces it) is actively pruned rather than left to go
  stale -- see `update()`'s docstring. Native `pretrain.py`'s own
  `running_pos_*[pos]` per-position metrics have exactly this staleness bug
  (a position stops being reached when `k_toks` shrinks, but its
  `RunningMean` is never reset, so `.compute()` silently keeps returning an
  old average from whatever `k_toks` regime last touched it); this tracker
  deliberately does not repeat that.
"""

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .drifting import DEFAULT_TEMPERATURES, DriftFieldResult, token_drift_field, token_drifting_loss
from .features import anchor_horizons, extract_predictive_states
from .horizon_weighting import horizon_weighted_sum


@dataclass
class DriftLossResult:
    loss: torch.Tensor  # scalar, grad-carrying through the student query states only
    per_horizon_results: List[DriftFieldResult]


def compute_drift_loss(
    student_hidden_full: torch.Tensor,
    teacher_hidden_full: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    eta: float = 1.0,
    use_column_norm: bool = True,
    exclude_self_positive: bool = True,
    tie_self_negative: bool = False,
    field_scales: Optional[torch.Tensor] = None,
    horizon_gamma: Optional[float] = None,
) -> DriftLossResult:
    """claudedriftingplan.md PHASE 4: `L_drift^token`, grad-carrying into the student.

    Args mirror `validation.compute_training_diagnostics`'s hidden-state
    args exactly (same `pred_pos_mask` slicing, same shape contract), so
    the two hooks can share one training step's forward passes -- see
    "Guaranteed length match between student and teacher" in
    claudedriftingplan.md.

    student_hidden_full: (data_bsz, S, d_h), NOT detached -- this is the
        live student forward pass; `extract_predictive_states` only slices,
        it does not stop gradients, so the returned `loss` back-propagates
        into the student through this tensor.
    teacher_hidden_full: (data_bsz, S, d_h), from the frozen teacher's
        `torch.no_grad()` forward pass. `token_drifting_loss` detaches it
        again internally regardless (belt-and-suspenders, per
        claudedriftingplan.md's "Teacher features ... MUST be detached").
    tie_self_negative, field_scales, horizon_gamma: see `drifting.token_drifting_loss`;
        `field_scales` normally comes from `DriftFieldScaleState.divisors(k_toks)`.
    """
    x = extract_predictive_states(
        student_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    )
    y_pos = extract_predictive_states(
        teacher_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    )
    loss, per_horizon_results = token_drifting_loss(
        x,
        y_pos,
        temperatures=temperatures,
        eta=eta,
        use_column_norm=use_column_norm,
        exclude_self_positive=exclude_self_positive,
        tie_self_negative=tie_self_negative,
        field_scales=field_scales,
        horizon_gamma=horizon_gamma,
    )
    return DriftLossResult(loss=loss, per_horizon_results=per_horizon_results)


# ---------------------------------------------------------------------------
# Full-batch drift field (claudefullbatchdriftplan.md): the field's population is the
# whole optimizer step (all gradient-accumulation micro-batches of every rank), not one
# micro-batch. V, S_j and lambda_{tau,j} are stop-gradient, so the full-batch drift
# gradient splits exactly by row: each micro-batch only needs its own rows of the
# full-batch V. Other ranks' rows enter only through V, so pooling ranks needs one
# all_gather of the detached states and no cross-rank backward.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftPopulationMode:
    full_batch: bool  # field estimated on the whole global batch (every micro-batch of every rank)
    two_pass: bool  # no-grad pass 1 needed: the drift term carries gradient and n_accum > 1
    log_only: bool  # full-batch field logged only (drift_weight == 0), no drift gradient


def drift_population_mode(
    drift_enabled: bool, drift_weight: float, n_accum: int, world_size: int
) -> DriftPopulationMode:
    """claudefullbatchdriftplan.md section 4. The population is the global batch, with no
    switch. Two passes are needed only when the drift term has a gradient and the step
    spans more than one micro-batch per rank; every other loss is separable per row and
    keeps the single pass. Ranks need no extra pass: with n_accum == 1 the micro-batch's
    graph is still alive when the other ranks' states arrive (pooled single pass).
    drift_weight == 0 needs no pass 1: the field is only logged, from the detached states
    cached during the normal pass."""
    if not drift_enabled:
        return DriftPopulationMode(full_batch=False, two_pass=False, log_only=False)
    full_batch = n_accum > 1 or world_size > 1
    return DriftPopulationMode(
        full_batch=full_batch,
        two_pass=n_accum > 1 and drift_weight != 0.0,
        log_only=full_batch and drift_weight == 0.0,
    )


@dataclass
class DriftPopulation:
    """The drift field's population: the step's predictive states of every rank (the
    global batch). Horizon j holds the rows of every rank whose k_toks > j, in rank order
    (k_toks differs across ranks under rand_rank_k_toks), with x and y rows paired. One
    rank's population is just its own states."""

    xs: List[torch.Tensor]  # per horizon j: (P_j, d_h) student states
    ys: List[torch.Tensor]  # per horizon j: (P_j, d_h) teacher states
    rank_rows: List[int]  # R_w, rows of each rank
    rank_k: List[int]  # k_toks of each rank
    rank: int  # this rank

    @classmethod
    def from_ranks(
        cls, x_ranks: List[torch.Tensor], y_ranks: List[torch.Tensor], rank: int
    ) -> "DriftPopulation":
        """x_ranks, y_ranks: per rank, its (R_w, k_w, d_h) states."""
        for x, y in zip(x_ranks, y_ranks):
            if x.dim() != 3 or y.shape != x.shape:
                raise ValueError(
                    f"x and y must both be (R, k_toks, d_h) and match; got "
                    f"{tuple(x.shape)} vs {tuple(y.shape)}"
                )

        def pool(states: List[torch.Tensor], j: int) -> torch.Tensor:
            parts = [s[:, j] for s in states if s.shape[1] > j]
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        k = max(x.shape[1] for x in x_ranks)
        return cls(
            xs=[pool(x_ranks, j) for j in range(k)],
            ys=[pool(y_ranks, j) for j in range(k)],
            rank_rows=[x.shape[0] for x in x_ranks],
            rank_k=[x.shape[1] for x in x_ranks],
            rank=rank,
        )

    @property
    def k(self) -> int:
        """Horizons in the population: max k_toks over ranks."""
        return len(self.xs)

    def rows(self, w: int, j: int) -> slice:
        """Rank w's rows inside horizon j (for j < rank_k[w])."""
        start = sum(r for r, k in zip(self.rank_rows[:w], self.rank_k[:w]) if k > j)
        return slice(start, start + self.rank_rows[w])


@torch.no_grad()
def gather_drift_population(
    all_gather: Callable[[torch.Tensor], torch.Tensor],
    rank: int,
    world_size: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> DriftPopulation:
    """Pool this rank's (R, k, d_h) step states with every other rank's.

    `all_gather` maps a tensor to its (world_size, ...) stack (`fabric.all_gather`). R and
    k_toks may differ across ranks, so ranks first exchange (R, k), pad to (max R, max k)
    for the one gather, and strip the padding (as `geometry.gather_predictive_states`, but
    keeping every horizon). Collective when world_size > 1: every rank must call this on
    the same iteration. world_size == 1 needs no communication.
    """
    if x.dim() != 3 or y.shape != x.shape:
        raise ValueError(
            f"x and y must both be (R, k_toks, d_h) and match; got {tuple(x.shape)} vs {tuple(y.shape)}"
        )
    if world_size == 1:
        return DriftPopulation.from_ranks([x], [y], rank=0)
    r, k = x.shape[0], x.shape[1]
    meta = all_gather(torch.tensor([r, k], device=x.device, dtype=torch.long))
    rank_rows, rank_k = meta[:, 0].tolist(), meta[:, 1].tolist()
    pair = x.new_zeros((2, max(rank_rows), max(rank_k), x.shape[2]))
    pair[0, :r, :k] = x
    pair[1, :r, :k] = y
    gathered = all_gather(pair)  # (world_size, 2, max R, max k, d_h)
    shapes = list(zip(rank_rows, rank_k))
    return DriftPopulation.from_ranks(
        [gathered[w, 0, :rw, :kw] for w, (rw, kw) in enumerate(shapes)],
        [gathered[w, 1, :rw, :kw] for w, (rw, kw) in enumerate(shapes)],
        rank=rank,
    )


@dataclass
class FullBatchDriftField:
    V: torch.Tensor  # (R, k, d_h) fp32, detached: V_multi of this rank's rows and horizons
    S_j: torch.Tensor  # (k,) fp32, detached (fp64 for fp64 inputs, here and in V)
    # (R,) mean_j mean_d (eta V)^2: each of this rank's rows' drift loss (sum_j w_j mean_d
    # (eta V)^2 with horizon weights)
    row_loss: torch.Tensor
    per_horizon_results: List[DriftFieldResult]  # every population horizon; diagnostics only
    population_row_loss: torch.Tensor  # (R_pop,) the same for every row of every rank

    @property
    def loss(self) -> torch.Tensor:
        """The full-batch drift loss, every row of the global batch weighted equally
        (= token_drifting_loss on the whole batch when k_toks is the same on every rank)."""
        return self.population_row_loss.mean()

    @property
    def population_rows(self) -> int:
        return self.population_row_loss.numel()


@torch.no_grad()
def compute_population_drift_field(
    population: DriftPopulation,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    eta: float = 1.0,
    use_column_norm: bool = True,
    exclude_self_positive: bool = True,
    tie_self_negative: bool = False,
    field_scales: Optional[torch.Tensor] = None,
    horizon_gamma: Optional[float] = None,
) -> FullBatchDriftField:
    """One drift field over the whole population (claudefullbatchdriftplan.md section 5,
    "full-batch field"): `token_drift_field` per horizon, same arguments as
    `token_drifting_loss`, computed in fp32 (the single-pass path uses the bf16 states as
    they come). Every rank computes the same field and keeps its own rows of V.

    field_scales: (population.k, n_tau), normally `DriftFieldScaleState.divisors(population.k)`.
    horizon_gamma: only changes the logged `row_loss` / `population_row_loss` (each row's
        horizons get `horizon_weights` over that row's own rank's k_toks, as its D_m
        does); V, S_j and the per-horizon results are the same with or without it.
    """
    k = population.k
    if field_scales is not None and field_scales.shape != (k, len(temperatures)):
        raise ValueError(
            f"field_scales must have shape ({k}, {len(temperatures)}), got {tuple(field_scales.shape)}"
        )
    # fp32, or fp64 when the inputs already are (the unit tests).
    dtype = torch.promote_types(population.xs[0].dtype, torch.float32)
    rank = population.rank
    k_local = population.rank_k[rank]
    V, S, sq, results = [], [], [], []
    for j in range(k):
        result = token_drift_field(
            population.xs[j].to(dtype),
            population.ys[j].to(dtype),
            temperatures=temperatures,
            use_column_norm=use_column_norm,
            exclude_self_positive=exclude_self_positive,
            tie_self_negative=tie_self_negative,
            field_scale=None if field_scales is None else field_scales[j],
        )
        if j < k_local:
            V.append(result.V_multi[population.rows(rank, j)])
        S.append(result.S_j)
        sq.append((eta * result.V_multi).pow(2).mean(dim=-1))
        # The loggers read only S_j, per_temperature and total_rms; dropping the (P_j, d_h)
        # tensors keeps one copy of the field (V) alive instead of three.
        results.append(replace(result, V_multi=None, x_normalized=None))
    V = torch.stack(V, dim=1)

    def over_horizons(per_horizon: torch.Tensor) -> torch.Tensor:
        # (rows, k_w) -> (rows,): the reduction over horizons that D_m uses.
        if horizon_gamma is None:
            return per_horizon.mean(dim=-1)
        return horizon_weighted_sum(per_horizon, horizon_gamma)

    row_loss = over_horizons((eta * V).pow(2).mean(dim=-1))
    # Each row averages its own rank's horizons, as that rank's D_m does.
    population_row_loss = torch.cat([
        over_horizons(torch.stack([sq[j][population.rows(w, j)] for j in range(k_w)], dim=1))
        for w, k_w in enumerate(population.rank_k)
    ])
    return FullBatchDriftField(
        V=V,
        S_j=torch.stack(S[:k_local]),
        row_loss=row_loss,
        per_horizon_results=results,
        population_row_loss=population_row_loss,
    )


@torch.no_grad()
def compute_full_batch_drift_field(
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    eta: float = 1.0,
    use_column_norm: bool = True,
    exclude_self_positive: bool = True,
    tie_self_negative: bool = False,
    field_scales: Optional[torch.Tensor] = None,
    horizon_gamma: Optional[float] = None,
) -> FullBatchDriftField:
    """`compute_population_drift_field` over one rank's step.

    x_all, y_all: (R_tot, k, d_h) detached student / teacher predictive states, the
        concatenation of every micro-batch's `extract_predictive_states` output.
    """
    return compute_population_drift_field(
        DriftPopulation.from_ranks([x_all], [y_all], rank=0),
        temperatures=temperatures,
        eta=eta,
        use_column_norm=use_column_norm,
        exclude_self_positive=exclude_self_positive,
        tie_self_negative=tie_self_negative,
        field_scales=field_scales,
        horizon_gamma=horizon_gamma,
    )


def drift_surrogate_loss(
    x: torch.Tensor,
    V_rows: torch.Tensor,
    S_j: torch.Tensor,
    eta: float = 1.0,
    horizon_gamma: Optional[float] = None,
) -> torch.Tensor:
    """D_m of claudefullbatchdriftplan.md section 5, pass 2: the drift loss of one
    micro-batch's rows against the full-batch field,

        x_hat = x / S_j,  target = sg(x_hat) + eta * V_rows,
        D_m = mean_j mean_r mean_d (x_hat - target)^2,

    the same reduction as `token_drifting_loss`. Its value is eta^2 mean (V_rows)^2 and its
    gradient is -2 eta V / (R_m k d_h S_j), exactly the full-batch field's, because the
    target is built from this pass's own x_hat (any pass 1 / pass 2 numerical difference
    does not enter). At least fp32 internally; the gradient flows back to `x` in its own
    dtype.

    With `horizon_gamma` (fixedkweightedplan_zh-TW.md) the mean over horizons becomes the
    weighted sum D_m = sum_j w_j mean_r mean_d (x_hat - target)^2, with w_j from
    `horizon_weights`, and the gradient of horizon j is -2 eta w_j V / (R_m d_h S_j):
    k w_j times the unweighted one.

    x: (R_m, k, d_h) live student states (grad-carrying). V_rows: (R_m, k, d_h), this
    micro-batch's rows of `FullBatchDriftField.V`. S_j: (k,).
    """
    if x.dim() != 3 or V_rows.shape != x.shape:
        raise ValueError(
            f"x and V_rows must both be (R, k_toks, d_h) and match; got "
            f"{tuple(x.shape)} vs {tuple(V_rows.shape)}"
        )
    if S_j.shape != (x.shape[1],):
        raise ValueError(f"S_j must have shape ({x.shape[1]},), got {tuple(S_j.shape)}")
    dtype = torch.promote_types(x.dtype, torch.float32)
    x_hat = x.to(dtype) / S_j.detach().to(dtype).view(1, -1, 1)
    target = x_hat.detach() + eta * V_rows.detach().to(dtype)
    if horizon_gamma is None:
        return (x_hat - target).pow(2).mean(dim=-1).mean()
    return horizon_weighted_sum((x_hat - target).pow(2).mean(dim=(0, 2)), horizon_gamma)


def _mean_std_ratio(values: Sequence[float]) -> Dict[str, float]:
    t = torch.tensor(list(values), dtype=torch.float64)
    mean = t.mean().item()
    std = t.std(unbiased=False).item() if t.numel() > 1 else 0.0
    ratio = mean / (std + 1e-12)
    return {"mean": mean, "std": std, "ratio": ratio}


class DriftAccumulationTracker:
    """Rolling window over one optimizer step's `gradient_accumulation_iters`
    micro-substeps, mirroring the `RunningMean(window=gradient_accumulation_iters)`
    idiom already used elsewhere in `pretrain.py` for the same "aggregate over
    the micro-substeps of one optimizer step, `.compute()` only at
    log_iter_interval-aligned iterations" pattern -- see claudedriftingplan.md
    PHASE 4's "Gradient-accumulation stability logging" note. Deliberately a
    plain Python/deque implementation (not `torchmetrics.RunningMean`)
    because it needs std, not just mean.
    """

    def __init__(self, temperatures: Sequence[float], window: int):
        self.temperatures = list(temperatures)
        self.window = window
        self._k_toks: deque = deque(maxlen=window)

        # Per-anchor-horizon breakdown (see module docstring). Populated lazily,
        # keyed by absolute horizon j -- a horizon only appears here once some
        # `update()` call's anchor_horizons(k_toks) includes it, and is pruned
        # entirely (not just cleared) the moment a later call's anchor set no
        # longer includes it, so `compute()` can never report stale data from a
        # past k_toks regime.
        self._anchor_s_j: Dict[int, deque] = {}
        self._anchor_lambda_tau: Dict[int, Dict[float, deque]] = {}
        self._anchor_raw_field_rms: Dict[int, Dict[float, deque]] = {}
        self._anchor_normalized_field_rms: Dict[int, Dict[float, deque]] = {}
        # Affinity masses: the kernel-saturation / self-anchor-starvation signal
        # PHASE 5 asks to check for. See `TemperatureFieldDiagnostics` in
        # drifting.py -- under use_column_norm=True these are the mass-split
        # prefactor on the field, so mass_neg (or mass_pos) collapsing toward 0
        # means the drift signal is being silenced by the prefactor rather than
        # by genuine student/teacher convergence. Identically 1.0 when
        # use_column_norm=False, in which case they are a config self-check.
        self._anchor_mass_pos: Dict[int, Dict[float, deque]] = {}
        self._anchor_mass_neg: Dict[int, Dict[float, deque]] = {}
        # Kernel-width diagnostics (claudedriftingplan.md > Kernel Temperature >
        # Effective neighbour count): N_eff ~1 means a temperature acts as a hard
        # nearest-neighbour selector, ~R means uniform weights. self_pos_weight is the
        # positive mass on the query's own paired teacher, identically 0.0 when
        # exclude_self_positive=True.
        self._anchor_n_eff_pos: Dict[int, Dict[float, deque]] = {}
        self._anchor_n_eff_neg: Dict[int, Dict[float, deque]] = {}
        self._anchor_self_pos_weight: Dict[int, Dict[float, deque]] = {}
        # claudetiedselfnegplan.md > Metrics: the scaling actually applied, the self-negative
        # weight, and the paired / unpaired / spurious decomposition of the Eq. 8 field,
        # keyed by wandb name. A diagnostic that is None (decomposition under column norm)
        # is not recorded, so its key simply does not appear.
        self._anchor_tau_extra: Dict[str, Dict[int, Dict[float, deque]]] = {
            key: {} for key in self._TAU_EXTRA_FIELDS
        }
        self._anchor_total_rms: Dict[int, deque] = {}

    # wandb key stem -> TemperatureFieldDiagnostics attribute.
    _TAU_EXTRA_FIELDS = {
        "lambda_used": "lambda_used",
        "self_neg_weight": "self_neg_weight",
        "field_paired_rms": "paired_rms",
        "field_unpaired_rms": "unpaired_rms",
        "field_spurious_rms": "spurious_rms",
        "field_cos_paired_spurious": "cos_paired_spurious",
    }

    def update(self, per_horizon_results: List[DriftFieldResult], k_toks: int) -> None:
        """Record one micro-substep's per-horizon results, broken out per
        anchor horizon (see `anchor_horizons`) -- no whole-window
        horizon-averaged scalar is tracked (see module docstring). `k_toks`
        is also recorded directly.

        `per_horizon_results[j - 1]` must be horizon `j` (1-indexed) -- the
        same convention `drifting.token_drifting_loss` and
        `features.extract_predictive_states` already use -- so
        `len(per_horizon_results)` must equal `k_toks`.
        """
        if not per_horizon_results:
            raise ValueError("per_horizon_results must be non-empty (k_toks >= 1)")
        if len(per_horizon_results) != k_toks:
            raise ValueError(
                f"len(per_horizon_results) ({len(per_horizon_results)}) must "
                f"equal k_toks ({k_toks})"
            )

        self._k_toks.append(k_toks)

        anchors = anchor_horizons(k_toks)
        self._prune_stale_anchor_horizons(anchors)
        for j in anchors:
            result = per_horizon_results[j - 1]
            self._anchor_s_j.setdefault(j, deque(maxlen=self.window)).append(
                result.S_j.item()
            )
            for tau in self.temperatures:
                diag = next(d for d in result.per_temperature if d.tau == tau)
                self._anchor_lambda_tau.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.lambda_tau)
                self._anchor_raw_field_rms.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.raw_field_rms)
                self._anchor_normalized_field_rms.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.normalized_field_rms)
                self._anchor_mass_pos.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.mass_pos)
                self._anchor_mass_neg.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.mass_neg)
                self._anchor_n_eff_pos.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.n_eff_pos)
                self._anchor_n_eff_neg.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.n_eff_neg)
                self._anchor_self_pos_weight.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.self_pos_weight)
                for key, attr in self._TAU_EXTRA_FIELDS.items():
                    value = getattr(diag, attr)
                    if value is not None:
                        self._anchor_tau_extra[key].setdefault(j, {}).setdefault(
                            tau, deque(maxlen=self.window)
                        ).append(value)
            self._anchor_total_rms.setdefault(j, deque(maxlen=self.window)).append(
                result.total_rms
            )

    def _prune_stale_anchor_horizons(self, current_anchors: List[int]) -> None:
        """Drop any previously-tracked anchor horizon that is NOT part of this
        call's anchor set, so a horizon that stops being an anchor (k_toks
        changed) can never silently report stale data at a later `compute()`.
        Horizons that remain anchors need no special handling here -- their
        bounded deques self-heal via normal maxlen eviction once a full
        window's worth of fresh appends has happened (see module docstring).
        """
        stale = set(self._anchor_s_j) - set(current_anchors)
        for j in stale:
            self._anchor_s_j.pop(j, None)
            self._anchor_lambda_tau.pop(j, None)
            self._anchor_raw_field_rms.pop(j, None)
            self._anchor_normalized_field_rms.pop(j, None)
            self._anchor_mass_pos.pop(j, None)
            self._anchor_mass_neg.pop(j, None)
            self._anchor_n_eff_pos.pop(j, None)
            self._anchor_n_eff_neg.pop(j, None)
            self._anchor_self_pos_weight.pop(j, None)
            for per_key in self._anchor_tau_extra.values():
                per_key.pop(j, None)
            self._anchor_total_rms.pop(j, None)

    def compute(self, prefix: str = "drift") -> Dict[str, float]:
        """Mean/std/ratio over whatever is currently in the window (up to the
        last `window` `update()` calls -- exactly one optimizer step's worth,
        provided `.update()` is called once per micro-substep and `.compute()`
        is only read at log_iter_interval-aligned iterations, both of which
        `pretrain.py` guarantees by construction)."""
        out: Dict[str, float] = {}

        def emit(name: str, values: Sequence[float]) -> None:
            for stat_name, value in _mean_std_ratio(values).items():
                out[f"{prefix}/{name}_{stat_name}"] = value

        emit("k_toks", self._k_toks)

        # Per-anchor-horizon breakdown: only horizons touched since the last
        # prune are present (see `_prune_stale_anchor_horizons`), so a horizon
        # absent from the current window's anchor set simply doesn't appear
        # this step -- sparse keys, not zero-filled or stale, matching Phase
        # 2's `validation.compute_training_diagnostics` convention for the
        # same underlying reason.
        for j in sorted(self._anchor_s_j):
            emit(f"S_j_h{j}", self._anchor_s_j[j])
        for j in sorted(self._anchor_lambda_tau):
            for tau in self.temperatures:
                if tau in self._anchor_lambda_tau[j]:
                    emit(f"lambda_tau_{tau}_h{j}", self._anchor_lambda_tau[j][tau])
        for j in sorted(self._anchor_raw_field_rms):
            for tau in self.temperatures:
                if tau in self._anchor_raw_field_rms[j]:
                    emit(f"raw_field_rms_tau_{tau}_h{j}", self._anchor_raw_field_rms[j][tau])
        for j in sorted(self._anchor_normalized_field_rms):
            for tau in self.temperatures:
                if tau in self._anchor_normalized_field_rms[j]:
                    emit(
                        f"normalized_field_rms_tau_{tau}_h{j}",
                        self._anchor_normalized_field_rms[j][tau],
                    )
        # Per-temperature loss contribution, eta^2 * ||V_tilde_tau||^2 / d_h in eta=1
        # units: the "drift loss per temperature" breakdown. Reads ~1.0 while a rung is
        # renormalized to unit scale, and decays toward 0 once its raw field falls below
        # eps and rms_field_normalize's (lambda_tau + eps) denominator damps it -- which
        # is the only mechanism by which the drift term can vanish at convergence. NOTE
        # the total drift loss is NOT the sum of these: the loss is the squared norm of
        # the SUMMED field, so it also carries the cross terms between rungs.
        for j in sorted(self._anchor_normalized_field_rms):
            for tau in self.temperatures:
                if tau in self._anchor_normalized_field_rms[j]:
                    emit(
                        f"loss_tau_{tau}_h{j}",
                        [v * v for v in self._anchor_normalized_field_rms[j][tau]],
                    )
        for j in sorted(self._anchor_mass_pos):
            for tau in self.temperatures:
                if tau in self._anchor_mass_pos[j]:
                    emit(f"mass_pos_tau_{tau}_h{j}", self._anchor_mass_pos[j][tau])
        for j in sorted(self._anchor_mass_neg):
            for tau in self.temperatures:
                if tau in self._anchor_mass_neg[j]:
                    emit(f"mass_neg_tau_{tau}_h{j}", self._anchor_mass_neg[j][tau])
        for j in sorted(self._anchor_n_eff_pos):
            for tau in self.temperatures:
                if tau in self._anchor_n_eff_pos[j]:
                    emit(f"n_eff_pos_tau_{tau}_h{j}", self._anchor_n_eff_pos[j][tau])
        for j in sorted(self._anchor_n_eff_neg):
            for tau in self.temperatures:
                if tau in self._anchor_n_eff_neg[j]:
                    emit(f"n_eff_neg_tau_{tau}_h{j}", self._anchor_n_eff_neg[j][tau])
        for j in sorted(self._anchor_self_pos_weight):
            for tau in self.temperatures:
                if tau in self._anchor_self_pos_weight[j]:
                    emit(f"self_pos_weight_tau_{tau}_h{j}", self._anchor_self_pos_weight[j][tau])
        for key, per_horizon in self._anchor_tau_extra.items():
            for j in sorted(per_horizon):
                for tau in self.temperatures:
                    if tau in per_horizon[j]:
                        emit(f"{key}_tau_{tau}_h{j}", per_horizon[j][tau])
        for j in sorted(self._anchor_total_rms):
            emit(f"field_total_rms_h{j}", self._anchor_total_rms[j])
        return out


# Warmup EMA decay for the frozen drift field scale, per observation; same time constant
# (10 observations) and rationale as SMOOTHL1_BETA_EMA_DECAY below. With the full-batch
# field (claudefullbatchdriftplan.md) there is one observation per optimizer step, so
# 10 optimizer steps; with n_accum == 1 the two coincide.
DRIFT_FIELD_SCALE_EMA_DECAY = 0.9


class DriftFieldScaleState:
    """Per-(horizon j, temperature tau) field scale lambda_{tau,j}: an EMA of the live batch
    RMS during LR warmup, then frozen (DDP-averaged over the ranks that observed each
    horizon). See claudetiedselfnegplan.md > Why the Field Scale Must Be Frozen.

    While not frozen, `divisors` is all NaN, so `rms_field_normalize` divides by the live
    lambda_tau and warmup is bit-identical to the unfrozen path. After the freeze each field
    is divided by the frozen value, so its normalized RMS is lambda_live / lambda_frozen and
    can fall as the student converges instead of being renormalized to 1 on every step.

    Same EMA / freeze / late-horizon / checkpoint policy as `SmoothL1BetaState`: seeded by
    the first observation; a horizon first seen after the freeze takes its live value once
    (rank-local) and is fixed from then on.
    """

    def __init__(
        self,
        k_max: int,
        temperatures: Sequence[float],
        device=None,
        ema_decay: float = DRIFT_FIELD_SCALE_EMA_DECAY,
    ):
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.ema_decay = ema_decay
        self.temperatures = list(temperatures)
        self.lam = torch.ones(k_max, len(self.temperatures), device=device)
        self.seen = torch.zeros(k_max, dtype=torch.bool, device=device)
        self.frozen = False

    def divisors(self, k_toks: int) -> torch.Tensor:
        """(k_toks, n_tau): the frozen lambda for frozen, seen horizons; NaN ("use live")
        everywhere else, and everywhere while not frozen."""
        if k_toks > self.lam.shape[0]:
            raise ValueError(f"k_toks ({k_toks}) exceeds k_max ({self.lam.shape[0]})")
        lam = self.lam[:k_toks]
        if not self.frozen:
            return torch.full_like(lam, float("nan"))
        return torch.where(self.seen[:k_toks, None], lam, torch.full_like(lam, float("nan")))

    @torch.no_grad()
    def observe(self, per_horizon_results: List[DriftFieldResult], k_toks: int) -> None:
        if len(per_horizon_results) != k_toks:
            raise ValueError(
                f"len(per_horizon_results) ({len(per_horizon_results)}) must equal k_toks ({k_toks})"
            )
        if k_toks > self.lam.shape[0]:
            raise ValueError(f"k_toks ({k_toks}) exceeds k_max ({self.lam.shape[0]})")
        d = self.ema_decay
        for j in range(k_toks):
            if self.frozen and self.seen[j]:
                continue
            by_tau = {diag.tau: diag.lambda_tau for diag in per_horizon_results[j].per_temperature}
            live = torch.tensor(
                [by_tau[tau] for tau in self.temperatures], dtype=self.lam.dtype, device=self.lam.device
            )
            # Unseen horizon (first warmup observation, or first after the freeze): take the
            # live value outright. Seen horizon during warmup: EMA update.
            self.lam[j] = torch.where(self.seen[j], d * self.lam[j] + (1.0 - d) * live, live)
            self.seen[j] = True

    @torch.no_grad()
    def freeze(self, fabric=None) -> None:
        """Freeze lambda. With `fabric` and world_size > 1, first replace each seen row by
        its mean over the ranks that observed horizon j. Collective: every rank must call
        this on the same iteration."""
        if fabric is not None and fabric.world_size > 1:
            seen_f = self.seen.float()
            lam_sum = fabric.all_reduce(self.lam * seen_f[:, None], reduce_op="sum")
            count = fabric.all_reduce(seen_f, reduce_op="sum")
            observed = count > 0
            self.lam = torch.where(
                observed[:, None], lam_sum / count.clamp_min(1.0)[:, None], self.lam
            )
            self.seen = observed
        self.frozen = True

    def metrics(self, prefix: str = "drift") -> Dict[str, float]:
        """`{prefix}/field_scale_frozen` and `{prefix}/lambda_frozen_tau_{tau}_h{j}` for every
        seen horizon (state, not a per-step statistic, so never stale)."""
        out: Dict[str, float] = {f"{prefix}/field_scale_frozen": float(self.frozen)}
        for idx in torch.nonzero(self.seen).flatten().tolist():
            for t, tau in enumerate(self.temperatures):
                out[f"{prefix}/lambda_frozen_tau_{tau}_h{idx + 1}"] = self.lam[idx, t].item()
        return out

    def state_dict(self) -> Dict:
        return {"lam": self.lam.clone(), "seen": self.seen.clone(), "frozen": self.frozen}

    def load_state_dict(self, state: Dict) -> None:
        self.lam = state["lam"].to(self.lam.device, self.lam.dtype)
        self.seen = state["seen"].to(self.seen.device, torch.bool)
        self.frozen = bool(state["frozen"])


# ---------------------------------------------------------------------------
# Smooth-L1 feature regulariser (EAGLE-style), see claudesmoothL1plan.md.
# Can be combined with the drift loss above (their weighted terms are summed):
# both regularise the same predictive states, but Smooth-L1 is paired
# (student r,j -> teacher r,j) with a per-horizon adaptive beta_j instead of an
# unpaired kernel field.
# ---------------------------------------------------------------------------

# Floor on beta_j. A zero beta (student == teacher exactly) would put a 0/0 in
# the L2 branch of torch.where, whose NaN leaks into the backward pass even
# when that branch is not selected.
SMOOTHL1_BETA_MIN = 1e-6

# Warmup EMA decay for beta_j, applied once per micro-step. Time constant
# 1/(1-decay) = 10 micro-steps: averages out single-batch quantile noise while
# tracking the error scale closely, so the frozen beta reflects the last few
# warmup steps. 0.0 means last batch only.
SMOOTHL1_BETA_EMA_DECAY = 0.9

# torch.quantile refuses inputs above 2**24 elements; fall back to kthvalue.
_QUANTILE_MAX_NUMEL = 2**24


@dataclass
class SmoothL1LossResult:
    loss: torch.Tensor  # scalar, grad through the student only
    per_horizon_loss: torch.Tensor  # (k,), detached, unweighted
    per_horizon_mse: torch.Tensor  # (k,), detached, for reference only
    per_horizon_frac_l1: torch.Tensor  # (k,), detached: fraction of elements with |e| >= beta_j


def extract_smoothl1_pair(
    student_hidden_full: torch.Tensor,
    teacher_hidden_full: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
):
    """(R, k, d_h) fp32 student states (grad-carrying) and detached teacher states.

    Sliced once per step and shared by `SmoothL1BetaState.observe` and
    `compute_smoothl1_loss`, so beta_j is estimated on exactly the errors the loss sees.
    """
    x = extract_predictive_states(
        student_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    ).float()
    y = extract_predictive_states(
        teacher_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    ).detach().float()
    return x, y


def compute_smoothl1_loss(
    x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, horizon_gamma: Optional[float] = None
) -> SmoothL1LossResult:
    """L_SL1 = mean_{r,j,d} SmoothL1_{beta_j}(x - sg(y)), with a separate beta_j per horizon.

    `F.smooth_l1_loss` only takes a scalar beta, hence the explicit `torch.where`.
    Mean over d_h, so the loss scale does not depend on model width.

    With `horizon_gamma` (fixedkweightedplan_zh-TW.md) the loss is sum_j w_j mean_{r,d}
    SmoothL1_{beta_j}, with w_j from `horizon_weights`. The per-horizon outputs stay
    unweighted.
    """
    k_toks = x.shape[1]
    err = x - y.detach()
    b = beta[:k_toks].detach().to(err).clamp_min(SMOOTHL1_BETA_MIN).view(1, k_toks, 1)
    abs_e = err.abs()
    is_l1 = abs_e >= b
    elem = torch.where(is_l1, abs_e - 0.5 * b, 0.5 * err.pow(2) / b)
    if horizon_gamma is None:
        loss = elem.mean()
    else:
        loss = horizon_weighted_sum(elem.mean(dim=(0, 2)), horizon_gamma)
    with torch.no_grad():
        return SmoothL1LossResult(
            loss=loss,
            per_horizon_loss=elem.detach().mean(dim=(0, 2)),
            per_horizon_mse=err.detach().pow(2).mean(dim=(0, 2)),
            per_horizon_frac_l1=is_l1.float().mean(dim=(0, 2)),
        )


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    flat = values.reshape(-1).float()
    if flat.numel() <= _QUANTILE_MAX_NUMEL:
        return torch.quantile(flat, q)
    k = min(max(1, math.ceil(q * flat.numel())), flat.numel())
    return flat.kthvalue(k).values


class SmoothL1BetaState:
    """Per-horizon beta_j from Q_q(|x_j - y_j|): during LR warmup, an EMA of the
    per-micro-step quantile; then frozen (DDP-averaged over the ranks that observed
    each horizon).

    The EMA is seeded: the first observation of horizon j sets beta_j = q_j outright,
    and later ones blend in as beta_j <- d * beta_j + (1 - d) * q_j. So beta_j never
    starts from the arbitrary initial value. The EMA is rank-local, like the batches it
    is fed.

    A horizon not seen before the freeze is estimated on its first observation (one
    batch, no EMA) and then fixed. That late estimate is rank-local: with random per-rank
    k_toks the other ranks may not see the horizon on the same step, so no collective can
    be issued.

    Implements `state_dict`/`load_state_dict`, so Fabric checkpoints it as part of the
    training `state` dict and a resumed run keeps its EMA and frozen beta.
    """

    def __init__(self, k_max: int, device=None, ema_decay: float = SMOOTHL1_BETA_EMA_DECAY):
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.ema_decay = ema_decay
        self.beta = torch.ones(k_max, device=device)
        self.seen = torch.zeros(k_max, dtype=torch.bool, device=device)
        self.frozen = False

    @torch.no_grad()
    def observe(self, err_abs: torch.Tensor, k_toks: int, q: float) -> None:
        if k_toks > self.beta.numel():
            raise ValueError(f"k_toks ({k_toks}) exceeds k_max ({self.beta.numel()})")
        d = self.ema_decay
        for j in range(k_toks):
            if self.frozen and self.seen[j]:
                continue
            q_j = _quantile(err_abs[:, j, :], q)
            # Unseen horizon (first warmup observation, or first after the freeze):
            # take q_j outright. Seen horizon during warmup: EMA update.
            new = torch.where(self.seen[j], d * self.beta[j] + (1.0 - d) * q_j, q_j)
            self.beta[j] = new.clamp_min(SMOOTHL1_BETA_MIN)
            self.seen[j] = True

    @torch.no_grad()
    def freeze(self, fabric=None) -> None:
        """Freeze beta. With `fabric` and world_size > 1, first replace each seen beta_j
        by its mean over the ranks that observed horizon j. Collective: every rank must
        call this on the same iteration."""
        if fabric is not None and fabric.world_size > 1:
            seen_f = self.seen.float()
            beta_sum = fabric.all_reduce(self.beta * seen_f, reduce_op="sum")
            count = fabric.all_reduce(seen_f, reduce_op="sum")
            observed = count > 0
            self.beta = torch.where(observed, beta_sum / count.clamp_min(1.0), self.beta)
            self.seen = observed
        self.frozen = True

    def state_dict(self) -> Dict:
        return {"beta": self.beta.clone(), "seen": self.seen.clone(), "frozen": self.frozen}

    def load_state_dict(self, state: Dict) -> None:
        # beta is the EMA itself, so older checkpoints (and the extra `ema`/`count` keys
        # of the bias-corrected variant) resume without conversion.
        self.beta = state["beta"].to(self.beta.device, self.beta.dtype)
        self.seen = state["seen"].to(self.seen.device, torch.bool)
        self.frozen = bool(state["frozen"])


class SmoothL1AccumulationTracker:
    """Window over one optimizer step's micro-substeps, same anchor-horizon and pruning
    policy as `DriftAccumulationTracker`. Emits one wandb section per horizon
    (`sl1_h{j}/...`) plus the pooled `sl1/pct_l1`."""

    _STATS = ("loss", "loss_weighted", "mse", "pct_l1", "pct_l2")

    def __init__(self, window: int, weight: float):
        self.window = window
        self.weight = weight
        self._pct_l1_pooled: deque = deque(maxlen=window)
        self._anchor: Dict[int, Dict[str, deque]] = {}

    def update(self, result: SmoothL1LossResult, k_toks: int) -> None:
        if result.per_horizon_loss.numel() != k_toks:
            raise ValueError(
                f"result has {result.per_horizon_loss.numel()} horizons, expected k_toks={k_toks}"
            )
        loss = result.per_horizon_loss.tolist()
        mse = result.per_horizon_mse.tolist()
        pct_l1 = (100.0 * result.per_horizon_frac_l1).tolist()
        self._pct_l1_pooled.append(sum(pct_l1) / k_toks)

        anchors = anchor_horizons(k_toks)
        for j in set(self._anchor) - set(anchors):
            del self._anchor[j]
        for j in anchors:
            values = {
                "loss": loss[j - 1],
                "loss_weighted": self.weight * loss[j - 1],
                "mse": mse[j - 1],
                "pct_l1": pct_l1[j - 1],
                "pct_l2": 100.0 - pct_l1[j - 1],
            }
            slots = self._anchor.setdefault(
                j, {s: deque(maxlen=self.window) for s in self._STATS}
            )
            for name, value in values.items():
                slots[name].append(value)

    def compute(self, beta_state: SmoothL1BetaState) -> Dict[str, float]:
        out: Dict[str, float] = {}

        def emit(key: str, values: Sequence[float]) -> None:
            stats = _mean_std_ratio(values)
            out[f"{key}_mean"] = stats["mean"]
            out[f"{key}_std"] = stats["std"]

        if self._pct_l1_pooled:
            emit("sl1/pct_l1", self._pct_l1_pooled)
        for j in sorted(self._anchor):
            for name in self._STATS:
                emit(f"sl1_h{j}/{name}", self._anchor[j][name])
        # beta for every horizon ever seen, not just anchors: it is state, not a
        # per-step statistic, so it never goes stale.
        for idx in torch.nonzero(beta_state.seen).flatten().tolist():
            out[f"sl1_h{idx + 1}/beta"] = beta_state.beta[idx].item()
        out["sl1/beta_frozen"] = float(beta_state.frozen)
        return out
