"""Global student-teacher hidden-state geometry diagnostics (distancdiagplan.md).

Monitoring only, and independent of the training objective: the same metrics, under the
same names, are computed for baseline MTP, MTP + Smooth-L1 and MTP + drifting. Everything
runs under `torch.no_grad()` on detached predictive states, and the return value holds
only Python floats, so nothing here can reach the loss, the gradients or the RNG state.

For every prediction horizon h, S_h / T_h are the student / teacher predictive states
selected by the training step's own `pred_pos_mask` (so exactly the positions the MTP loss
supervises; tail-padded regions are already dropped from that mask upstream), with the
original pairing (s_i, t_i) kept by row index. From the three Euclidean distance matrices
D_SS, D_TT, D_ST of that horizon, the module derives every metric in the plan, including
MMD (ignite's estimator with the paired cross terms added back, see
`metrics.mmd.paired_maximum_mean_discrepancy`; bandwidth from horizon 1's D^2) and Sinkhorn
(D_ST is its ground cost). Each matrix is built once per horizon, in the
hidden states' own dtype (bf16 under bf16-true training), and released before the next
horizon.

Keys: `{prefix}/h{h}/{metric}` for every horizon 1..k_toks this step. There is
deliberately no cross-horizon average: comparisons between setups are made at a fixed h.
"""

from typing import Callable, Dict, Optional, Tuple

import torch

from .features import extract_predictive_states
from .metrics.mmd import MMDBandwidthSchedule, paired_maximum_mean_discrepancy
from .metrics.sinkhorn import sinkhorn_from_cost

GEOMETRY_EPS = 1e-8

# A teacher whose nearest other teacher is closer than this fraction of d_tt_mean is
# treated as a duplicate. bf16 cdist cannot resolve distances that small: an exact
# duplicate measured as up to ~0.064 * mean distance (d = 512-2048, N = 160-640), and a
# true 0.05 * mean distance measured anywhere in [0, 1.6x]. At 0.1 * mean the relative
# error is within +-20%. Real teacher nearest-neighbour spacing is ~0.6 * mean.
GEOMETRY_DUPLICATE_REL_TOL = 0.1

# Per-horizon metric names, in logging order.
METRIC_NAMES = (
    "num_samples",
    "d_ss_mean",
    "d_ss_std",
    "d_tt_mean",
    "d_tt_std",
    "d_st_mean",
    "d_st_std",
    "pair_distance_mean",
    "pair_distance_std",
    "teacher_nn_distance_mean",
    "teacher_nn_distance_std",
    "pairing_gap_mean",
    "pairing_gap_std",
    "paired_teacher_is_nn_fraction",
    "paired_teacher_rank_mean",
    "paired_teacher_rank_median",
    "student_teacher_spread_ratio",
    "teacher_nn_internal_mean",
    "teacher_nn_internal_std",
    "teacher_duplicate_fraction",
    "pair_over_teacher_local_mean",
    "pair_over_teacher_local_std",
    "normalized_pairing_gap_mean",
    "normalized_pairing_gap_std",
    "mmd",
    "sinkhorn",
)


@torch.no_grad()
def pairwise_distance_matrices(
    s: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Euclidean D_SS, D_TT, D_ST for (N, d) student / teacher states, in their own dtype.

    Both populations are first shifted by their pooled mean. Distances are unchanged by a
    common shift, but cdist's |a|^2 + |b|^2 - 2ab form cancels badly in bf16 when the
    states share a large mean component: centering cut the paired-distance error on
    hidden-state-like data from ~0.5% to ~0.01%. (cdist's exact-difference mode has no
    bf16 kernel.) The D_SS / D_TT diagonals are set to exact zeros.
    """
    center = torch.cat([s, t]).mean(dim=0)
    s = s - center
    t = t - center
    d_ss = torch.cdist(s, s)
    d_tt = torch.cdist(t, t)
    d_st = torch.cdist(s, t)
    d_ss.fill_diagonal_(0.0)
    d_tt.fill_diagonal_(0.0)
    return d_ss, d_tt, d_st


def _mean_std(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return v.mean(), v.std(unbiased=False)


@torch.no_grad()
def metrics_from_distances(
    d_ss: torch.Tensor,
    d_tt: torch.Tensor,
    d_st: torch.Tensor,
    eps: float = GEOMETRY_EPS,
) -> Dict[str, torch.Tensor]:
    """Every per-horizon metric except `mmd` from one horizon's distance matrices
    (`mmd` runs ignite on the states themselves, see `compute_geometry_diagnostics`).

    Returns 0-dim tensors keyed by `METRIC_NAMES`, left on device so the caller can move
    every horizon's values to the host in one transfer. Distance-derived values keep the
    input dtype; the count-based ones (num_samples, NN fraction, rank) are float32 so
    integers above 256 stay exact. Stds are population stds (unbiased=False).
    """
    n = d_ss.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=d_ss.device)

    out: Dict[str, torch.Tensor] = {}
    out["num_samples"] = torch.tensor(float(n), dtype=torch.float32, device=d_ss.device)

    # D_SS / D_TT: zero diagonal excluded. D_ST: all N^2 cross pairs, diagonal included.
    out["d_ss_mean"], out["d_ss_std"] = _mean_std(d_ss[off])
    out["d_tt_mean"], out["d_tt_std"] = _mean_std(d_tt[off])
    out["d_st_mean"], out["d_st_std"] = _mean_std(d_st)

    # Paired distance is the D_ST diagonal only.
    d_pair = d_st.diagonal()
    out["pair_distance_mean"], out["pair_distance_std"] = _mean_std(d_pair)

    # Nearest teacher over ALL j, the paired teacher included, so G >= 0 by construction.
    d_nn = d_st.min(dim=1).values
    gap = d_pair - d_nn
    out["teacher_nn_distance_mean"], out["teacher_nn_distance_std"] = _mean_std(d_nn)
    out["pairing_gap_mean"], out["pairing_gap_std"] = _mean_std(gap)

    # r_i = 1 + #{j != i : D_ij < D_ii}; j = i never counts since D_ii < D_ii is false.
    # "Paired teacher is NN" is r_i == 1, i.e. G_i == 0: an exact tie with another teacher
    # counts in the pair's favour, consistent with the rank's strict inequality.
    rank = 1.0 + (d_st < d_pair[:, None]).sum(dim=1).float()
    out["paired_teacher_is_nn_fraction"] = (rank == 1.0).float().mean()
    out["paired_teacher_rank_mean"] = rank.mean()
    out["paired_teacher_rank_median"] = rank.quantile(0.5)

    out["student_teacher_spread_ratio"] = out["d_ss_mean"] / (out["d_tt_mean"] + eps)

    # Teacher local scale: nearest OTHER teacher.
    d_tt_nn = d_tt.masked_fill(~off, float("inf")).min(dim=1).values
    out["teacher_nn_internal_mean"], out["teacher_nn_internal_std"] = _mean_std(d_tt_nn)

    # A teacher with an (effectively) duplicate neighbour has no local scale, and dividing
    # by d_tt_nn + eps blows the ratios up to ~1e9. Such rows are excluded from the two
    # ratios and counted in teacher_duplicate_fraction. The tolerance is relative, not
    # == 0: in bf16 an exact duplicate does not come out as 0 (see GEOMETRY_DUPLICATE_REL_TOL).
    duplicate = d_tt_nn <= GEOMETRY_DUPLICATE_REL_TOL * out["d_tt_mean"]
    out["teacher_duplicate_fraction"] = duplicate.float().mean()
    keep = ~duplicate
    local = d_tt_nn[keep] + eps
    out["pair_over_teacher_local_mean"], out["pair_over_teacher_local_std"] = _mean_std(d_pair[keep] / local)
    out["normalized_pairing_gap_mean"], out["normalized_pairing_gap_std"] = _mean_std(gap[keep] / local)

    # Sinkhorn's ground cost is the Euclidean D_ST itself. The solver runs in float32: its
    # 100 dual updates in bf16 alone added ~1% error (vs ~0.02% in float32).
    out["sinkhorn"] = sinkhorn_from_cost(d_st.float())
    return out


@torch.no_grad()
def gather_predictive_states(
    all_gather: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool (R, k, d) student / teacher states from every rank, pairing preserved.

    `all_gather` maps a tensor to its (world_size, ...) stack (e.g. `fabric.all_gather`).
    R differs across ranks when tail-padded regions are dropped, and k_toks differs
    under rand_rank_k_toks, so ranks first exchange (R, k): every rank keeps the common
    horizons 1..min(k) and pads its rows to max(R) for the gather, and the padding is
    stripped afterwards. Collective: every rank must call this on the same iteration.
    """
    r, k = s.shape[0], s.shape[1]
    meta = all_gather(torch.tensor([r, k], device=s.device, dtype=torch.long))
    r_all = meta[:, 0].tolist()
    k_common = int(meta[:, 1].min())
    r_max = max(r_all)

    pair = torch.stack([s[:, :k_common], t[:, :k_common]])  # (2, R, k_common, d)
    if r < r_max:
        pad = pair.new_zeros((2, r_max - r) + tuple(pair.shape[2:]))
        pair = torch.cat([pair, pad], dim=1)
    gathered = all_gather(pair.contiguous())  # (world, 2, r_max, k_common, d)
    s_all = torch.cat([gathered[w, 0, :rw] for w, rw in enumerate(r_all)])
    t_all = torch.cat([gathered[w, 1, :rw] for w, rw in enumerate(r_all)])
    return s_all, t_all


@torch.no_grad()
def compute_geometry_diagnostics(
    student_hidden_full: torch.Tensor,
    teacher_hidden_full: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
    bandwidth_schedule: MMDBandwidthSchedule,
    step: int,
    gather_fn: Optional[Callable] = None,
    prefix: str = "diag",
    eps: float = GEOMETRY_EPS,
) -> Dict[str, float]:
    """Per-horizon geometry metrics for one training step.

    Args:
        student_hidden_full / teacher_hidden_full: (data_bsz, S, d_h) hidden states of
            this step's student and student-forced teacher passes.
        pred_pos_mask, tot_mask_regions, k_toks: the training step's own slicing, see
            `features.extract_predictive_states`.
        bandwidth_schedule: this hook's own `MMDBandwidthSchedule`, kept in the
            checkpointed training state. Updated from horizon 1's distance matrices and
            shared by every horizon.
        step: optimizer step, the schedule's warmup unit.
        gather_fn: optional `(s, t) -> (s_all, t_all)` that pools states across ranks
            (see `gather_predictive_states`); None keeps the rank-local population.

    Returns:
        Flat `{key: float}`; empty if fewer than 2 samples are available.
    """
    s = extract_predictive_states(student_hidden_full, pred_pos_mask, tot_mask_regions, k_toks).detach()
    t = extract_predictive_states(teacher_hidden_full, pred_pos_mask, tot_mask_regions, k_toks).detach()
    if gather_fn is not None:
        s, t = gather_fn(s, t)
    n, num_horizons = s.shape[0], s.shape[1]
    if n < 2:
        return {}

    per_horizon = []
    bandwidth = None
    for idx in range(num_horizons):
        d_ss, d_tt, d_st = pairwise_distance_matrices(s[:, idx], t[:, idx])
        if idx == 0:
            bandwidth = bandwidth_schedule.update_from_sq_dists(
                step, d_ss.square(), d_tt.square(), d_st.square()
            )
        metrics = metrics_from_distances(d_ss, d_tt, d_st, eps=eps)
        metrics["mmd"] = paired_maximum_mean_discrepancy(s[:, idx], t[:, idx], bandwidth)
        per_horizon.append(torch.stack([metrics[name].float() for name in METRIC_NAMES]))
        del d_ss, d_tt, d_st, metrics

    values = torch.stack(per_horizon).tolist()  # (num_horizons, num_metrics), one host sync
    del per_horizon, s, t

    out: Dict[str, float] = {
        f"{prefix}/num_horizons": float(num_horizons),
        f"{prefix}/mmd_bandwidth": float(bandwidth),
    }
    for idx, row in enumerate(values):
        for name, value in zip(METRIC_NAMES, row):
            out[f"{prefix}/h{idx + 1}/{name}"] = value
    return out
