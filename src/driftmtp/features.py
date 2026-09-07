"""Extraction of student/teacher predictive-state tensors for DriftMTP.

Reuses the exact `pred_pos_mask` boolean tensor the upstream MTP training step
already uses to slice student/teacher logits (see
`third_party/mtp-lm/litgpt/pretrain.py`, `truncate_and_mask` /
`soft_stud_preds` / `soft_teach_preds`), which guarantees the extracted
hidden-state tensors are the same shape as the existing logit tensors — see
"Guaranteed length match between student and teacher" in
`claudedriftingplan.md` > Validation Metrics > Decided Implementation.
"""

import math
from typing import List

import torch


def extract_predictive_states(
    hidden_states: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
) -> torch.Tensor:
    """Slice (data_bsz, S, d_h) hidden states at pred_pos_mask into (R, k_toks, d_h).

    Args:
        hidden_states: (data_bsz, S, d_h) — post-`ln_f`, pre-`lm_head` hidden
            states, as returned by `GPT.forward(..., return_hidden_states=True)`.
        pred_pos_mask: (data_bsz, S) bool — the exact mask already used to
            slice `soft_stud_preds` / `soft_teach_preds` in the training step.
        tot_mask_regions: R, the number of active MTP regions this step
            (== `soft_stud_preds.shape[0]`).
        k_toks: number of predicted horizons this step
            (== `soft_stud_preds.shape[1]`).

    Returns:
        (R, k_toks, d_h) tensor. Index `j - 1` along dim 1 is prediction
        horizon `j` (1-indexed), matching the existing
        `soft_stud_preds`/`soft_teach_preds` convention exactly (both are
        produced by indexing the same boolean mask).
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be (data_bsz, S, d_h), got shape {tuple(hidden_states.shape)}"
        )
    if pred_pos_mask.dim() != 2 or pred_pos_mask.dtype != torch.bool:
        raise ValueError(
            f"pred_pos_mask must be a (data_bsz, S) bool tensor, got shape "
            f"{tuple(pred_pos_mask.shape)} dtype {pred_pos_mask.dtype}"
        )
    d_h = hidden_states.shape[-1]
    selected = hidden_states[pred_pos_mask]
    expected = tot_mask_regions * k_toks
    if selected.shape[0] != expected:
        raise ValueError(
            f"pred_pos_mask selected {selected.shape[0]} positions, expected "
            f"tot_mask_regions * k_toks = {tot_mask_regions} * {k_toks} = {expected}"
        )
    return selected.view(tot_mask_regions, k_toks, d_h)


def anchor_horizons(k_toks: int) -> List[int]:
    """First/second/third/middle/final predicted-token horizons, 1-indexed.

    For ``k_toks <= 5``, returns full coverage ``[1, ..., k_toks]`` — with
    that few horizons active, subsampling buys nothing and the plain
    ``{1, 2, 3, mid, k_toks}`` formula below would otherwise silently drop
    a horizon at ``k_toks == 5`` (``mid`` collides with the existing ``3``).
    `k_toks` is uniformly random over a range that can dip well below 5
    under the active curriculum (e.g. `randint(2, 16)` per step — see
    `claudedriftingplan.md` > Validation Metrics > Decided Implementation >
    "Anchor horizon set"), so this is not an edge case.

    For ``k_toks > 5``: ``mid = ceil(k_toks / 2)``;
    ``anchors = sorted({1, 2, 3, mid, k_toks} intersect [1, k_toks])``.
    """
    if k_toks < 1:
        raise ValueError(f"k_toks must be >= 1, got {k_toks}")
    if k_toks <= 5:
        return list(range(1, k_toks + 1))
    mid = math.ceil(k_toks / 2)
    candidates = {1, 2, 3, mid, k_toks}
    return sorted(h for h in candidates if 1 <= h <= k_toks)
