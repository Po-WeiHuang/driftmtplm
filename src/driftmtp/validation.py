"""Diagnostic-only training-time monitoring for DriftMTP (Phase 2).

Computes ECE (vs. hard teacher label), per-horizon MMD, and per-horizon
Sinkhorn distance from the student/teacher predictive states already
produced by one training step's forward passes. Does not touch gradients,
does not modify model parameters, and does not run any additional
student/teacher forward pass or dataloader.

See `claudedriftingplan.md` > Validation Metrics > "Decided Implementation
(Phase 2 scope)" for why this replaced an earlier standalone
validation-loop design: `generative_validate()` never reproduces the
training-time teacher-forced single-block pattern this needs, and the
training step already computes exactly that pattern every iteration, so
tapping it directly needs no second forward pass and no second dataloader.
"""

from typing import Dict

import torch

from .features import anchor_horizons, extract_predictive_states
from .geometry import pairwise_distance_matrices
from .metrics.calibration import expected_calibration_error
from .metrics.mmd import MMDBandwidthSchedule, paired_maximum_mean_discrepancy
from .metrics.sinkhorn import sinkhorn_distance


@torch.no_grad()
def compute_training_diagnostics(
    student_hidden_full: torch.Tensor,
    teacher_hidden_full: torch.Tensor,
    student_logits_full: torch.Tensor,
    hard_teacher_labels_full: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
    bandwidth_schedule: MMDBandwidthSchedule,
    step: int,
    prefix: str = "drift_diag",
) -> Dict[str, float]:
    """Diagnostic ECE/MMD/Sinkhorn at the anchor horizon set for one training step.

    Args:
        student_hidden_full: (data_bsz, S, d_h) — student
            `GPT.forward(..., return_hidden_states=True)` hidden states,
            NOT yet sliced by `pred_pos_mask`.
        teacher_hidden_full: (data_bsz, S, d_h) — teacher hidden states from
            the forward pass already run under `torch.no_grad()` with frozen
            parameters (detached by construction).
        student_logits_full: (R, k_toks, vocab) — the already-computed
            `soft_stud_preds` tensor (student's per-horizon logits).
        hard_teacher_labels_full: (R, k_toks) — the already-computed
            `hard_teach_preds` tensor.
        pred_pos_mask: (data_bsz, S) bool — the exact mask already used to
            build `soft_stud_preds` / `soft_teach_preds` in the training
            step; reused here so the resulting `(R, k_toks, d_h)`
            hidden-state tensors are guaranteed the same shape.
        tot_mask_regions: R.
        k_toks: number of active prediction horizons this step.
        bandwidth_schedule: shared `MMDBandwidthSchedule` instance, created
            once by the caller (e.g. alongside the other running-metric
            trackers before the training loop) and passed in on every
            diagnostic step so its warmup-then-freeze state persists across
            steps. See `mmd.py` for the policy this implements.
        step: current training step/iteration count, forwarded to
            `bandwidth_schedule`.
        prefix: metric key prefix (default `"drift_diag"`), kept separate
            from the training-loss metrics namespace.

    Returns:
        Flat `{str: float}` dict, safe to `.update()` into the existing
        training `metrics` dict and pass to `fabric.log_dict`. Every value
        is a detached Python float — no tensors, no autograd references are
        retained by the return value.
    """
    x = extract_predictive_states(
        student_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    ).detach()
    y_pos = extract_predictive_states(
        teacher_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    ).detach()

    horizons = anchor_horizons(k_toks)

    out: Dict[str, float] = {}
    for j in horizons:
        idx = j - 1
        out[f"{prefix}/ece/h{j}"] = expected_calibration_error(
            student_logits_full[:, idx, :], hard_teacher_labels_full[:, idx]
        ).item()

    # MMD: ignite's estimator with the paired (s_i, t_i) kernels added back, shared with
    # diag/ (docs/geometry_diagnostics_report.md); plain ignite logged ~0 on most steps. The
    # bandwidth comes from horizon 1 (always the first anchor) and is reused for every anchor.
    d_ss, d_tt, d_st = pairwise_distance_matrices(x[:, 0], y_pos[:, 0])
    bandwidth = bandwidth_schedule.update_from_sq_dists(
        step, d_ss.float().square(), d_tt.float().square(), d_st.float().square()
    )
    del d_ss, d_tt, d_st
    for j in horizons:
        out[f"{prefix}/mmd/h{j}"] = paired_maximum_mean_discrepancy(
            x[:, j - 1], y_pos[:, j - 1], bandwidth
        ).item()
        # Sinkhorn unchanged: raw states, cost = cdist in their own dtype.
        out[f"{prefix}/sinkhorn/h{j}"] = sinkhorn_distance(x[:, j - 1], y_pos[:, j - 1]).item()
    out[f"{prefix}/mmd_bandwidth"] = float(bandwidth)

    return out
