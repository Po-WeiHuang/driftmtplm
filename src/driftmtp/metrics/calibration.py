"""Expected Calibration Error (ECE), Eq. (94) of the DriftMTP research draft.

ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|, over M equal-width
confidence bins B_m.

Two entry points:

- `expected_calibration_error(logits, labels)`: single-batch ECE, unchanged
  since Phase 2 — this is what the training-time diagnostic
  (`validation.py::compute_training_diagnostics`) calls, and its behavior
  MUST NOT change.
- `confidence_correctness` / `ece_from_pairs`: the same computation, split
  into an extraction step and a binning step, so a caller can accumulate
  `(confidence, correctness)` pairs across many batches/documents — e.g. the
  full held-out set for the controlled-rollout harness
  (`docs/controlled_rollout_plan.md` Phase 1) — and bin **once** over the
  pooled population, per `claduemetricworkprocedure.md` §6's pooling rule.
  `expected_calibration_error` is defined in terms of these two pieces, so
  the two code paths cannot drift apart.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def confidence_correctness(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-1 softmax confidence and correctness-against-`labels`, per example.

    Args:
        logits: (N, vocab) unnormalized logits.
        labels: (N,) int64 target token ids.

    Returns:
        `(confidences, correctness)`, each `(N,)` float32. `confidences` is
        the top-1 softmax probability; `correctness` is `1.0`/`0.0` for
        whether the argmax prediction equals `labels`.
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
    return confidences, correctness


@torch.no_grad()
def ece_from_pairs(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Bin a pre-computed `(confidence, correctness)` population into ECE.

    Callers accumulating pairs across many batches/documents (e.g. the
    controlled-rollout harness) should concatenate every pair into one pool
    first and call this **once** — calling `expected_calibration_error`
    once per batch and averaging the results is the forbidden
    `(1/D) * sum_d ECE_d` from `claduemetricworkprocedure.md` §6, not pooled
    ECE.

    Args:
        confidences: (N,) float top-1 confidences in `[0, 1]`.
        correctness: (N,) float/bool correctness labels (`1.0`/`0.0` or
            `True`/`False`), same `N` as `confidences`.
        n_bins: number of equal-width confidence bins in [0, 1]. Default 15
            (standard ECE convention).

    Returns:
        Scalar tensor (float32), always in `[0, 1]`. Lower is better.
    """
    if confidences.dim() != 1:
        raise ValueError(f"confidences must be (N,), got shape {tuple(confidences.shape)}")
    if correctness.dim() != 1 or correctness.shape[0] != confidences.shape[0]:
        raise ValueError(
            f"correctness must be (N,) matching confidences.shape[0]={confidences.shape[0]}, "
            f"got {tuple(correctness.shape)}"
        )

    confidences = confidences.float()
    correctness = correctness.float()

    n = confidences.shape[0]
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    ece = torch.zeros((), device=confidences.device, dtype=torch.float32)
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


@torch.no_grad()
def _bin_table(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 15,
) -> list[dict]:
    """Per-bin `(count, conf, acc, contribution)` for the binning above.

    Repeats `ece_from_pairs`'s bin construction -- same `linspace` boundaries,
    same half-open intervals, same closed last bin -- and returns one record
    per **non-empty** bin instead of their sum. `ece_from_pairs` itself is left
    untouched: the training-time diagnostic calls it through
    `expected_calibration_error` and must not change, so agreement is enforced
    by test rather than by sharing a code path.
    """
    confidences = confidences.float()
    correctness = correctness.float()

    n = confidences.shape[0]
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    table = []
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
        table.append(
            {
                "bin": i,
                "count": int(bin_count),
                "conf": float(bin_conf),
                "acc": float(bin_acc),
                "contribution": float((bin_count.float() / n) * (bin_acc - bin_conf).abs()),
            }
        )
    return table


@torch.no_grad()
def ece_bin_stats(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 15,
    thin_below: int = 30,
) -> dict:
    """How much of an ECE rests on how few samples. Diagnostic only.

    ECE is occupancy-weighted, so a bin holding one example still enters the
    sum with weight `1/N` — and its `|acc - conf|` can be as large as 1. The
    scalar hides that: two populations with the same ECE can differ entirely in
    whether the number came from the bulk or from a handful of stragglers.
    These four fields make the exposure visible **without changing any ECE**.

    Args:
        confidences: (N,) float confidences in `[0, 1]`, exactly as passed to
            `ece_from_pairs` — for the joint form that is `C[:, j]`, the
            per-region product, already multiplied before binning.
        correctness: (N,) float correctness labels, same `N`.
        n_bins: must match the `n_bins` the ECE was computed with.
        thin_below: a bin with fewer than this many samples counts as thin.

    Returns:
        `bins_populated` (int), `max_bin_weight` (float, largest `|B_m|/N`),
        `min_bin_count` (int, smallest non-empty bin), `thin_bin_share`
        (float, fraction of the ECE contributed by thin bins; `0.0` when the
        ECE is zero or the population is empty).
    """
    table = _bin_table(confidences, correctness, n_bins=n_bins)
    if not table:
        return {
            "bins_populated": 0,
            "max_bin_weight": 0.0,
            "min_bin_count": 0,
            "thin_bin_share": 0.0,
        }

    n = confidences.shape[0]
    total = sum(row["contribution"] for row in table)
    thin = sum(row["contribution"] for row in table if row["count"] < thin_below)
    return {
        "bins_populated": len(table),
        "max_bin_weight": max(row["count"] for row in table) / n,
        "min_bin_count": min(row["count"] for row in table),
        "thin_bin_share": (thin / total) if total > 0 else 0.0,
    }


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
    confidences, correctness = confidence_correctness(logits, labels)
    return ece_from_pairs(confidences, correctness, n_bins=n_bins)
