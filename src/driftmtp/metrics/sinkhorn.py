"""Entropic optimal transport (Sinkhorn) distance, Cuturi 2013.

Ground-cost convention matches `fwilliams/scalable-pytorch-sinkhorn`
(https://github.com/fwilliams/scalable-pytorch-sinkhorn/blob/main/sinkhorn.py)'s
`sinkhorn(x, y, p=2, ...)`: for `p=2` (the default, here and there), the ground cost between a
pair of points is the **plain Euclidean (L2) distance `||x-y||`, NOT the squared distance**
(fwilliams' `M_ij = ((x-y)**2).sum(dim=2) ** (1/2)`, confirmed by reading the actual source, not
guessed). That repo does **not** do any feature normalization (no z-score, no L2-normalizing
`x`/`y` to the unit sphere) — the entire scale reduction relative to a squared-cost convention
comes from that one exponent: squared distance grows like `O(d)` for `d`-dimensional features,
while plain Euclidean distance grows like `O(sqrt(d))`. This is very likely why literature
Sinkhorn numbers often land smaller than a squared-cost implementation's.

This module adopts fwilliams' `p`-norm ground-cost convention (via `torch.cdist`, which
computes exactly `M_ij = ||x_i - y_j||_p` for any `p`) WITHOUT vendoring their file or its
`pykeops` dependency — see `docs/phase2_diagnostics_report.md` §5 for why `pykeops` (JIT-compiled
CUDA/C++ kernels, a real risk on an HTCondor cluster, no benefit at our `R<=160` scale) was
declined. Everything else — uniform marginals, log-domain Sinkhorn-Knopp, no PCA / no additional
feature normalization, no subsampling — is unchanged and still documented here.

**`epsilon` default deliberately NOT changed to match fwilliams' `eps=1e-3`.** Their default is
calibrated for their actual use case (their README/examples: 3D point-cloud shape
correspondence, small-scale geometric coordinates), where typical inter-point distances are
already small. At our actual feature scale (raw, unnormalized LLM hidden states, `d_h` up to
2048), `eps=1e-3` relative to typical Euclidean distances (`O(sqrt(d_h))`, potentially dozens to
hundreds) would make the kernel logits enormous and the coupling collapse toward a near-hard
nearest-neighbor assignment — defeating the point of entropic smoothing, not just "small".
`sinkhorn_distance` keeps `0.1` (this module's original default) for its remaining callers, but
`0.1` is still near-hard at this scale, so the geometry diagnostics and the controlled-rollout
eval no longer use it.

**Scale-relative epsilon.** The optimal coupling is `P_ij = u_i exp(-C_ij / eps) v_j`, so for
one student the weight ratio of two teachers is `exp(-(C_ij - C_ik) / eps)`: what `eps` has to
be compared with is the spread of the costs, not their mean. `sinkhorn_epsilon_from_cost` sets
`eps = SINKHORN_EPS_STD_SCALE * std(C)`, the population std of the cost matrix (= the geometry
diagnostics' `d_st_std` when `C` is D_ST). On synthetic clustered states (N = 640-2000,
d = 1024) a scale of 1 left 1-12 teachers of effective weight per student, i.e. a soft but
still local coupling; 0.25 was already a hard assignment. `SinkhornEpsilonSchedule` re-derives
it from horizon 1 during warmup and then freezes it, exactly like `MMDBandwidthSchedule`, so a
Sinkhorn trend is measured with one fixed `eps`.

`sinkhorn_from_cost` is the raw entropic OT cost, not the debiased Sinkhorn divergence: it
still rewards a student that contracts toward its paired teachers. Use the energy distance
(`driftmtp.geometry.energy_distance_from_means`) to check for contraction, or the debiased
divergence (`entropic_ot_from_cost` + `sinkhorn_divergence_from_ot`, Feydy et al. 2019), which
needs two more Sinkhorn solves (on D_SS and D_TT) and is therefore computed only in the
controlled-rollout eval, not in the training diagnostics.

Also not carried over from fwilliams' version: non-uniform weights (`w_x`/`w_y` — unneeded, our
regions are always uniformly weighted) and the correspondence-index outputs
(`corrs_x_to_y`/`corrs_y_to_x` — unneeded, only the scalar cost is consumed downstream).
"""

import math
from typing import Dict, Optional, Tuple

import torch

# eps = SINKHORN_EPS_STD_SCALE * std(cost); see the module docstring.
SINKHORN_EPS_STD_SCALE = 1.0


@torch.no_grad()
def sinkhorn_epsilon_from_cost(
    cost: torch.Tensor, scale: float = SINKHORN_EPS_STD_SCALE
) -> torch.Tensor:
    """`scale * std(cost)`, population std over every entry, in at least float32."""
    return scale * cost.to(torch.promote_types(cost.dtype, torch.float32)).std(unbiased=False)


class SinkhornEpsilonSchedule:
    """The single Sinkhorn `eps` shared across all anchor horizons.

    Same policy as `MMDBandwidthSchedule`: re-derived from horizon 1 on every `update()`
    call while `step < warmup_steps`, frozen at its last value afterwards, and checkpointable
    so a resume restores it instead of re-estimating from one batch.
    """

    def __init__(self, warmup_steps: int, scale: float = SINKHORN_EPS_STD_SCALE):
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        if not scale > 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.warmup_steps = warmup_steps
        self.scale = scale
        self._value: Optional[torch.Tensor] = None

    def update(self, step: int, cost_std: torch.Tensor) -> torch.Tensor:
        """Call once per diagnostic step with horizon 1's cost std (`d_st_std`).

        Returns the `eps` to use this step, for every horizon.
        """
        if self._value is None or step < self.warmup_steps:
            self._value = (self.scale * cost_std.detach()).clamp_min(1e-12)
        return self._value

    def state_dict(self) -> Dict:
        value = None if self._value is None else self._value.detach().float().cpu()
        return {"warmup_steps": self.warmup_steps, "scale": self.scale, "value": value}

    def load_state_dict(self, state: Dict) -> None:
        self._value = state["value"]

    @property
    def value(self) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError(
                "SinkhornEpsilonSchedule.update() must be called at least once before .value is read."
            )
        return self._value


@torch.no_grad()
def sinkhorn_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 2,
    epsilon: float = 0.1,
    n_iters: int = 100,
    stop_threshold: float = 1e-6,
) -> torch.Tensor:
    """Entropically regularized p-Wasserstein-style OT cost between uniform empirical x, y.

    Log-domain Sinkhorn-Knopp iteration on the p-norm ground-cost matrix
    (`M_ij = ||x_i - y_j||_p`, via `torch.cdist`) with uniform marginals
    (`1/Nx`, `1/Ny`). Returns the transport cost `sum_ij P_ij * M_ij` at
    convergence (the "entropically regularized OT cost" reported in the
    DriftMTP research draft's Table 5/12) — the raw OT cost under the
    converged coupling, not the debiased Sinkhorn divergence, and not the
    outer `(.)^(1/p)` some p-Wasserstein-distance definitions apply to the
    aggregated cost (fwilliams' function doesn't apply that either — see
    module docstring).

    Args:
        x: (Nx, d) student features.
        y: (Ny, d) teacher features.
        p: which p-norm to use as the ground cost, matching fwilliams'
            `sinkhorn(..., p=2, ...)`. Default `2` -> Euclidean (L2)
            distance, NOT squared distance (see module docstring).
        epsilon: entropic regularization strength. Default `0.1`,
            deliberately NOT fwilliams' `1e-3` default — see module
            docstring for why that default doesn't transfer to our feature
            scale.
        n_iters: maximum Sinkhorn iterations.
        stop_threshold: stop early once the max abs change in the row dual
            potential between iterations drops below this.

    Returns:
        Scalar tensor, the entropic OT cost. Lower is better; `0` in the
        limit `x == y` and `epsilon -> 0`.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"x, y must be (N, d); got {tuple(x.shape)}, {tuple(y.shape)}")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"x, y must share feature dim; got {x.shape[-1]} vs {y.shape[-1]}")
    if p <= 0:
        raise ValueError(f"p must be > 0, got {p}")

    cost = torch.cdist(x, y, p=float(p))  # (Nx, Ny), M_ij = ||x_i - y_j||_p
    return sinkhorn_from_cost(cost, epsilon=epsilon, n_iters=n_iters, stop_threshold=stop_threshold)


@torch.no_grad()
def sinkhorn_from_cost(
    cost: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 100,
    stop_threshold: float = 1e-6,
) -> torch.Tensor:
    """`sinkhorn_distance` on a precomputed (Nx, Ny) ground-cost matrix.

    Lets a caller that already holds the Euclidean cross-distance matrix (e.g.
    `driftmtp.geometry`'s D_ST) reuse it instead of recomputing `torch.cdist`.
    """
    u, v = _sinkhorn_potentials(cost, epsilon, n_iters, stop_threshold)
    coupling = torch.exp(_dual_matrix(cost, u, v, epsilon))
    return (coupling * cost).sum()


def _dual_matrix(cost: torch.Tensor, u: torch.Tensor, v: torch.Tensor, epsilon: float) -> torch.Tensor:
    """`log P_ij = (u_i + v_j - C_ij) / eps`."""
    return (-cost + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon


def _sinkhorn_potentials(
    cost: torch.Tensor, epsilon: float, n_iters: int, stop_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-domain Sinkhorn-Knopp with uniform marginals; returns the potentials `(u, v)`.

    `u`, `v` are relative to the counting measure (`P_ij = exp((u_i + v_j - C_ij) / eps)`),
    not to `mu ⊗ nu` as in Feydy et al. 2019 eq. (9): `f_i = u_i - eps log mu_i`.
    """
    if cost.dim() != 2:
        raise ValueError(f"cost must be (Nx, Ny), got {tuple(cost.shape)}")
    nx, ny = cost.shape

    mu = torch.full((nx,), 1.0 / nx, device=cost.device, dtype=cost.dtype)
    nu = torch.full((ny,), 1.0 / ny, device=cost.device, dtype=cost.dtype)
    log_mu, log_nu = mu.log(), nu.log()

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    for _ in range(n_iters):
        u_prev = u
        u = epsilon * (log_mu - torch.logsumexp(_dual_matrix(cost, u, v, epsilon), dim=1)) + u
        v = epsilon * (log_nu - torch.logsumexp(_dual_matrix(cost, u, v, epsilon).t(), dim=1)) + v
        if (u - u_prev).abs().max().item() < stop_threshold:
            break
    return u, v


@torch.no_grad()
def entropic_ot_from_cost(
    cost: torch.Tensor,
    epsilon: float,
    n_iters: int = 100,
    stop_threshold: float = 1e-6,
) -> torch.Tensor:
    """The regularized OT_eps of Feydy et al. 2019 eq. (1), uniform marginals.

    `min_P <P, C> + eps KL(P | mu ⊗ nu)` — the full objective, not the sharp `<P, C>` that
    `sinkhorn_from_cost` returns. Evaluated as the dual objective, eq. (8), at the solver's
    potentials: `<mu, f> + <nu, g> - eps (sum_ij P_ij - 1)`. At convergence the last term is 0
    and this is Prop. 1's `<mu, f> + <nu, g>`; before it, the dual value is still a lower bound.
    """
    return sinkhorn_and_entropic_ot_from_cost(cost, epsilon, n_iters, stop_threshold)[1]


@torch.no_grad()
def sinkhorn_and_entropic_ot_from_cost(
    cost: torch.Tensor,
    epsilon: float,
    n_iters: int = 100,
    stop_threshold: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(sinkhorn_from_cost(cost), entropic_ot_from_cost(cost))` from ONE Sinkhorn solve.

    The first element is computed by exactly the same ops as `sinkhorn_from_cost`, so it is
    bitwise identical to it; the second reuses the same potentials and coupling.
    """
    u, v = _sinkhorn_potentials(cost, epsilon, n_iters, stop_threshold)
    coupling = torch.exp(_dual_matrix(cost, u, v, epsilon))
    sharp = (coupling * cost).sum()
    nx, ny = cost.shape
    # f = u - eps log(1/nx), g = v - eps log(1/ny); <mu, f> = mean(u) + eps log nx.
    dual = u.mean() + v.mean() + epsilon * (math.log(nx) + math.log(ny))
    return sharp, dual - epsilon * (coupling.sum() - 1.0)


@torch.no_grad()
def sinkhorn_divergence_from_ot(ot_st: torch.Tensor, ot_ss: torch.Tensor, ot_tt: torch.Tensor) -> torch.Tensor:
    """Debiased Sinkhorn divergence, Feydy et al. 2019 eq. (3).

    `S_eps(S, T) = OT_eps(S, T) - OT_eps(S, S) / 2 - OT_eps(T, T) / 2`, each term from
    `entropic_ot_from_cost` on D_ST, D_SS, D_TT (diagonals kept: the self-terms transport a
    population onto itself) with one shared eps. Theorem 1: `S_eps >= 0`, `= 0` iff the two
    populations match — so unlike the raw cost it does not reward a student that contracts
    toward its teachers. Limits (eq. 4): W1 as eps -> 0, half the (V-statistic) energy
    distance as eps -> inf.
    """
    return ot_st - 0.5 * ot_ss - 0.5 * ot_tt
