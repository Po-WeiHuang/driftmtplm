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
from .metrics.calibration import expected_calibration_error
from .metrics.latent_metrics import per_horizon_latent_metrics
from .metrics.mmd import MMDBandwidthSchedule


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

    latent = per_horizon_latent_metrics(x, y_pos, horizons, bandwidth_schedule, step)
    out.update({f"{prefix}/{key}": value for key, value in latent.items()})

    return out
