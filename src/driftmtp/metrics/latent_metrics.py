"""Per-horizon latent-space alignment diagnostics (MMD, Sinkhorn) at a chosen horizon set."""

from typing import Dict, List

import torch

from .mmd import MMDBandwidthSchedule, maximum_mean_discrepancy
from .sinkhorn import sinkhorn_distance


@torch.no_grad()
def per_horizon_latent_metrics(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    horizons: List[int],
    bandwidth_schedule: MMDBandwidthSchedule,
    step: int,
) -> Dict[str, float]:
    """MMD and Sinkhorn distance per horizon in `horizons`.

    Args:
        student_hidden: (R, k_toks, d_h).
        teacher_hidden: (R, k_toks, d_h) — same R and k_toks as
            `student_hidden` (guaranteed by shared `pred_pos_mask` slicing
            upstream, see `features.extract_predictive_states`).
        horizons: 1-indexed horizon numbers to evaluate, e.g. from
            `features.anchor_horizons(k_toks)`.
        bandwidth_schedule: shared `MMDBandwidthSchedule` instance (owned by
            the caller, persisted across training steps). Its `.update()` is
            called here once per call, using horizon-1 features (index 0 of
            `student_hidden`/`teacher_hidden`, always present regardless of
            whether `1` is in `horizons`) — the resulting single bandwidth
            is then used for every horizon's MMD this call, per the policy
            documented in `mmd.py`.
        step: current training step/iteration count, forwarded to
            `bandwidth_schedule.update()`.

    Returns:
        Flat dict: `{"mmd/h{j}": float, "sinkhorn/h{j}": float, ...}` for
        each `j` in `horizons`. `mmd/h{j}` is MMD (not MMD^2) from
        `maximum_mean_discrepancy`.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"student_hidden {tuple(student_hidden.shape)} and teacher_hidden "
            f"{tuple(teacher_hidden.shape)} must match exactly."
        )
    k_toks = student_hidden.shape[1]

    mmd_bandwidth = bandwidth_schedule.update(step, student_hidden[:, 0, :], teacher_hidden[:, 0, :])

    out: Dict[str, float] = {}
    for j in horizons:
        if not (1 <= j <= k_toks):
            raise ValueError(f"horizon {j} out of range for k_toks={k_toks}")
        idx = j - 1
        x_j = student_hidden[:, idx, :]
        y_j = teacher_hidden[:, idx, :]
        out[f"mmd/h{j}"] = maximum_mean_discrepancy(x_j, y_j, bandwidth=mmd_bandwidth).item()
        out[f"sinkhorn/h{j}"] = sinkhorn_distance(x_j, y_j).item()
    return out
