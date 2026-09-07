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
nearest-neighbor assignment — defeating the point of entropic smoothing, not just "small". Kept
at `0.1` (this module's original default) instead; recalibrating `epsilon` to the actual
observed hidden-state distance scale (mirroring `MMDBandwidthSchedule`'s warmup-then-freeze
treatment of the MMD kernel bandwidth) would be a reasonable follow-up, not done here.

Also not carried over from fwilliams' version: non-uniform weights (`w_x`/`w_y` — unneeded, our
regions are always uniformly weighted) and the correspondence-index outputs
(`corrs_x_to_y`/`corrs_y_to_x` — unneeded, only the scalar cost is consumed downstream).
"""

import torch


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

    nx, ny = x.shape[0], y.shape[0]
    cost = torch.cdist(x, y, p=float(p))  # (Nx, Ny), M_ij = ||x_i - y_j||_p

    mu = torch.full((nx,), 1.0 / nx, device=x.device, dtype=cost.dtype)
    nu = torch.full((ny,), 1.0 / ny, device=x.device, dtype=cost.dtype)
    log_mu, log_nu = mu.log(), nu.log()

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    def dual_matrix(u_: torch.Tensor, v_: torch.Tensor) -> torch.Tensor:
        return (-cost + u_.unsqueeze(1) + v_.unsqueeze(0)) / epsilon

    for _ in range(n_iters):
        u_prev = u
        u = epsilon * (log_mu - torch.logsumexp(dual_matrix(u, v), dim=1)) + u
        v = epsilon * (log_nu - torch.logsumexp(dual_matrix(u, v).t(), dim=1)) + v
        if (u - u_prev).abs().max().item() < stop_threshold:
            break

    coupling = torch.exp(dual_matrix(u, v))
    return (coupling * cost).sum()
