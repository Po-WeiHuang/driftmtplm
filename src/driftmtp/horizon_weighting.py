"""Horizon weights for the training losses (fixedkweightedplan_zh-TW.md, design section).

One optimizer step predicts k_toks horizons per region. Without weighting, every loss
(the MTP token loss, the drift loss, the Smooth-L1 loss) is mean-pooled over all
predicted positions. Every horizon has the same number of regions, so that is the
plain mean over horizons of the per-horizon loss l_j (l_j: mean over the regions of
horizon j).

With weighting the loss is sum_j w_j * l_j, with

    w_j = gamma^(j-1) / sum_{i=1..k_toks} gamma^(i-1),    j = 1..k_toks.

The weights are normalised over the k_toks of the current step, so they sum to 1 for
any k (random k needs nothing extra) and the weighted loss stays on the scale of the
mean-pooled one: drift_weight, smoothl1_weight and the learning rate keep their
meaning when the switch is flipped. gamma = 1 gives w_j = 1 / k_toks, i.e. mean pooling.

The geometric form is the one in EAGLE-3's training code (eagle/traineagle3/main.py:
`ploss_weight = [0.8 ** i for i in range(len(plosses))]`, combined as a plain sum; it
is not in the paper's text). The normalisation is this project's choice, not EAGLE-3's.
"""

from typing import Optional

import torch


def horizon_weights(
    k_toks: int,
    gamma: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(k_toks,) weights w_j = gamma^(j-1) / sum_i gamma^(i-1). Index j - 1 is horizon j
    (1-indexed), the convention of `features.extract_predictive_states`. Sums to 1.

    Computed in fp64 and cast, so the sum is 1 to the precision of `dtype`.
    """
    if k_toks < 1:
        raise ValueError(f"k_toks must be >= 1, got {k_toks}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    w = torch.pow(
        torch.tensor(gamma, dtype=torch.float64), torch.arange(k_toks, dtype=torch.float64)
    )
    return (w / w.sum()).to(device=device, dtype=dtype)


def horizon_weighted_sum(per_horizon: torch.Tensor, gamma: float) -> torch.Tensor:
    """sum_j w_j * per_horizon[..., j]: reduces the last dim (the k_toks horizons) with
    `horizon_weights(k_toks, gamma)`. At least fp32 (bf16 holds a weight to ~3 digits only);
    the gradient flows back to `per_horizon` in its own dtype."""
    dtype = torch.promote_types(per_horizon.dtype, torch.float32)
    w = horizon_weights(per_horizon.shape[-1], gamma, device=per_horizon.device, dtype=dtype)
    return (per_horizon.to(dtype) * w).sum(dim=-1)
