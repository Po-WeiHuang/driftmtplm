"""Feature-scale (S_j) and field-scale (lambda_{tau,j}) normalization for DriftMTP token drifting.

Implements the first and third normalization stages required by
`claudedriftingplan.md`'s "Drifting Normalization" section (feature scale S
-> field scale lambda_tau -> regularization scale lambda_drift): feature
scale here, and the RMS half of the multi-temperature field normalization.
The middle stage (effective temperature) is also here since it is a small,
independently testable scalar transform. The third stage
(lambda_drift, the training-time loss weight) is a Phase 4 hyperparameter,
not computed here.
"""

import math
from typing import Tuple

import torch


def self_pair_mask(num_regions: int, device=None) -> torch.Tensor:
    """(R, R) boolean mask, True on the diagonal (query r vs. sample l == r)."""
    return torch.eye(num_regions, dtype=torch.bool, device=device)


def pairwise_distances(
    x: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """L2 distances D+_{rl} = ||x_r - y_pos_l||, D-_{rl} = ||x_r - y_neg_l||.

    x, y_pos, y_neg: (R, d_h). Returns (dist_pos, dist_neg), each (R, R),
    unmasked -- the self term at dist_neg[r, r] is a real, finite distance.
    Self-exclusion happens later, in logit space (see `drifting.py`), not
    here, so this function has no notion of "valid" vs. "masked" entries.
    """
    if x.dim() != 2 or y_pos.shape != x.shape or y_neg.shape != x.shape:
        raise ValueError(
            f"x, y_pos, y_neg must all be (R, d_h) and match; got "
            f"{tuple(x.shape)}, {tuple(y_pos.shape)}, {tuple(y_neg.shape)}"
        )
    dist_pos = torch.cdist(x, y_pos, p=2)
    dist_neg = torch.cdist(x, y_neg, p=2)
    return dist_pos, dist_neg


def feature_scale(
    dist_pos: torch.Tensor,
    dist_neg: torch.Tensor,
    self_mask: torch.Tensor,
    d_h: int,
    exclude_self_positive: bool = True,
) -> torch.Tensor:
    """S_j = sg[ mean(D+_valid union D-_valid) / sqrt(d_h) ] (claudedriftingplan.md, "Feature Scale").

    D-_valid always excludes the self pair (`self_mask`).

    `exclude_self_positive` (default True) additionally drops D+[r, r], the
    query's OWN paired teacher token. This must track whatever
    `token_drift_field` does to `logit_pos`: a distance that no longer
    participates in the affinity should not set the coordinate scale either.
    As the student converges toward its teacher that entry tends to zero, so
    leaving it in biases S_j downward exactly when the student is doing well.

    Pass False to reproduce the original asymmetric construction, where the
    positive population was used in full. Returns a detached scalar tensor.
    """
    valid_neg = dist_neg[~self_mask]
    valid_pos = dist_pos[~self_mask] if exclude_self_positive else dist_pos
    pooled = torch.cat([valid_pos.reshape(-1), valid_neg.reshape(-1)])
    return (pooled.mean() / math.sqrt(d_h)).detach()


def effective_temperature(tau: float, d_h: int) -> float:
    """tilde_tau = tau * sqrt(d_h) (claudedriftingplan.md, "Kernel Temperature")."""
    return tau * math.sqrt(d_h)


def rms_field_normalize(
    V: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-coordinate RMS field normalization (claudedriftingplan.md,
    "Multi-Temperature Field Normalization"):

        lambda_{tau,j} = sg[ sqrt( mean_r( ||V_r||_2^2 / d_h ) ) ]
        V_tilde = V / (lambda_{tau,j} + eps)

    V: (R, d_h) raw field for one horizon/temperature. `eps`'s value is not
    specified by the research draft (only the symbol); `1e-6` is this
    implementation's own choice, small enough to be a no-op whenever
    `lambda_{tau,j}` is not degenerately close to zero.

    Returns (V_tilde, lambda_{tau,j}); lambda_{tau,j} is detached.
    """
    d_h = V.shape[-1]
    lam = torch.sqrt((V.pow(2).sum(dim=-1) / d_h).mean()).detach()
    V_tilde = V / (lam + eps)
    return V_tilde, lam