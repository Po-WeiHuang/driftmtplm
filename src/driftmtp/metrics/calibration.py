"""Expected Calibration Error (ECE), Eq. (94) of the DriftMTP research draft.

ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|, over M equal-width
confidence bins B_m.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Top-1 ECE between predicted confidence and correctness against `labels`.

    Args:
        logits: (N, vocab) unnormalized student logits.
        labels: (N,) int64 target token ids. In the DriftMTP training-time
            diagnostic this MUST be the teacher's hard prediction
            (`hard_teach_preds`), not dataset ground truth — see
            `claudedriftingplan.md` > Validation Metrics > Decided
            Implementation (Phase 2 scope) > "ECE target" for why.
        n_bins: number of equal-width confidence bins in [0, 1]. Default 15
            (standard ECE convention).

    Returns:
        Scalar tensor (float32), always in `[0, 1]`. Lower is better.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (N, vocab), got shape {tuple(logits.shape)}")
    if labels.dim() != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"labels must be (N,) matching logits.shape[0]={logits.shape[0]}, "
            f"got {tuple(labels.shape)}"
        )

    probs = F.softmax(logits.float(), dim=-1)
    confidences, predictions = probs.max(dim=-1)
    correctness = (predictions == labels).float()

    n = confidences.shape[0]
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device, dtype=torch.float32)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_acc = correctness[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece = ece + (bin_count.float() / n) * (bin_acc - bin_conf).abs()
    return ece
