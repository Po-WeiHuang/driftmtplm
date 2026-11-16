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
MMD (ignite's estimator with the paired cross terms added back, from D^2, see
`metrics.mmd.mmd_from_sq_dists(paired=True)`; bandwidth from horizon 1's D^2), Sinkhorn
(D_ST is its ground cost; eps from horizon 1's d_st_std, see
`metrics.sinkhorn.SinkhornEpsilonSchedule`) and the energy distance (from the three matrices'
means, see `energy_distance_from_means`). `stat_match*` (L_SM, arXiv:2003.04289 Eq. 4) is taken from the
states directly, see `statistics_matching`. The matrices are built once per horizon and step,
in float32 (`diagnostic_distances`), and passed as arguments to every metric -- and, when
both hooks run on the same micro-iteration and population, to `validation`'s `drift_diag/`
hook as well, which then builds none of its own.

Keys: `{prefix}/h{h}/{metric}` for every horizon 1..k_toks this step. There is
deliberately no cross-horizon average: comparisons between setups are made at a fixed h.
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

from .features import extract_predictive_states
from .metrics.mmd import MMDBandwidthSchedule, mmd_from_sq_dists
from .metrics.sinkhorn import SinkhornEpsilonSchedule, sinkhorn_from_cost

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
    "energy_distance",
    "energy_distance_normalized",
    "teacher_nn_internal_mean",
    "teacher_nn_internal_std",
    "teacher_duplicate_fraction",
    "pair_over_teacher_local_mean",
    "pair_over_teacher_local_std",
    "normalized_pairing_gap_mean",
    "normalized_pairing_gap_std",
    "mmd",
    "sinkhorn",
    "stat_match",
    "stat_match_mean_term",
    "stat_match_std_term",
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
    # Reduced in at least float32: a bf16 result keeps ~3 significant digits, and the
    # energy distance is a small difference of these means.
    v = v.to(torch.promote_types(v.dtype, torch.float32))
    return v.mean(), v.std(unbiased=False)


def energy_distance_from_means(d_st_mean, pair_distance_mean, d_ss_mean, d_tt_mean, n):
    """Energy distance (Szekely & Rizzo) from one horizon's already-computed distance means:

        ED = 2 E|S - T| - E|S - S'| - E|T - T'|

    d_ss_mean / d_tt_mean exclude the zero diagonals and d_st_mean averages all n^2 cross
    pairs, the n paired distances included. Those paired terms are taken back out,

        d_st_off = (n * d_st_mean - pair_distance_mean) / (n - 1),

    so every term is a U-statistic over distinct indices: the estimate is unbiased, does not
    use the pairing, and can come out slightly negative when the two populations match.
    ED >= 0 in expectation and is 0 only for identical distributions; the -E|S - S'| term
    makes a contracted (under-dispersed) student score worse, not better.
    """
    d_st_off = (n * d_st_mean - pair_distance_mean) / (n - 1)
    return 2 * d_st_off - d_ss_mean - d_tt_mean


@torch.no_grad()
def metrics_from_distances(
    d_ss: torch.Tensor,
    d_tt: torch.Tensor,
    d_st: torch.Tensor,
    eps: float = GEOMETRY_EPS,
) -> Dict[str, torch.Tensor]:
    """Every per-horizon metric except `mmd`, `sinkhorn` and `stat_match*` from one horizon's
    distance matrices (see `compute_geometry_diagnostics`: `mmd` and `stat_match*` use the
    states themselves, and `sinkhorn` needs the shared eps).

    Returns 0-dim tensors keyed by `METRIC_NAMES`, left on device so the caller can move
    every horizon's values to the host in one transfer. Means / stds are reduced in at least
    float32; the per-row ratios are formed in the input dtype. The count-based ones
    (num_samples, NN fraction, rank) are float32 so integers above 256 stay exact. Stds are
    population stds (unbiased=False).
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

    # Pairing-free and contraction-sensitive; from the means above, no new distances.
    out["energy_distance"] = energy_distance_from_means(
        out["d_st_mean"], out["pair_distance_mean"], out["d_ss_mean"], out["d_tt_mean"], n
    )
    out["energy_distance_normalized"] = out["energy_distance"] / (out["d_tt_mean"] + eps)

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
    return out


@torch.no_grad()
def statistics_matching(
    s: torch.Tensor, t: torch.Tensor, eps: float = GEOMETRY_EPS
) -> Dict[str, torch.Tensor]:
    """L_SM of Yang et al. 2020, "Knowledge distillation via adaptive instance
    normalization" (arXiv:2003.04289), Sec. 3.2, Eqs. (2)-(4), on (N, d) states.

    The paper's channels c are the d hidden dimensions and its H*W spatial positions are
    the N tokens of one horizon's ensemble, so mu^c / sigma^c are each dimension's mean
    and spread over the ensemble (Eq. 2; Eq. 3 with eps inside the sqrt, population var):
        L_SM = 1/d sum_c [(mu_T^c - mu_S^c)^2 + (sigma_T^c - sigma_S^c)^2]      (Eq. 4)
    Pairing-free: it compares the two ensembles' first and second moments only.
    `stat_match_mean_term` / `stat_match_std_term` are the two summands, each averaged
    over d, so they add up to `stat_match`. Runs in at least float32 (bf16 sums over N
    quantise the moments).
    """
    dtype = torch.promote_types(s.dtype, torch.float32)
    s, t = s.to(dtype), t.to(dtype)
    mu_s, mu_t = s.mean(dim=0), t.mean(dim=0)
    sigma_s = (s.var(dim=0, unbiased=False) + eps).sqrt()
    sigma_t = (t.var(dim=0, unbiased=False) + eps).sqrt()
    mean_term = (mu_t - mu_s).square().mean()
    std_term = (sigma_t - sigma_s).square().mean()
    return {
        "stat_match": mean_term + std_term,
        "stat_match_mean_term": mean_term,
        "stat_match_std_term": std_term,
    }


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


class DiagnosticDistances(NamedTuple):
    """One step's predictive states and every horizon's distance matrices, built once.

    `s`, `t`: (n, k, d_h) float32 states. `matrices[h - 1]` is horizon h's float32
    `(D_SS, D_TT, D_ST)` from `pairwise_distance_matrices`. Shared by the `diag/` and
    `drift_diag/` hooks of the same micro-iteration so neither rebuilds them.
    """

    s: torch.Tensor
    t: torch.Tensor
    matrices: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@torch.no_grad()
def diagnostic_distances(
    student_hidden_full: torch.Tensor,
    teacher_hidden_full: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
    gather_fn: Optional[Callable] = None,
) -> Optional[DiagnosticDistances]:
    """Extract (and optionally gather) the predictive states, then build D_SS / D_TT / D_ST
    once per horizon. None when fewer than 2 samples are available.

    float32, not the hidden states' bf16: MMD is built from these same matrices, and bf16's
    ~0.4% per-entry rounding swamps a small MMD^2 (see `metrics.mmd.mmd_from_sq_dists`).
    At the diagnostic population (n ~ 32 per rank) float32 costs nothing.
    """
    s = extract_predictive_states(student_hidden_full, pred_pos_mask, tot_mask_regions, k_toks).detach()
    t = extract_predictive_states(teacher_hidden_full, pred_pos_mask, tot_mask_regions, k_toks).detach()
    if gather_fn is not None:
        s, t = gather_fn(s, t)
    if s.shape[0] < 2:
        return None
    s, t = s.float(), t.float()
    matrices = [pairwise_distance_matrices(s[:, idx], t[:, idx]) for idx in range(s.shape[1])]
    return DiagnosticDistances(s, t, matrices)


@torch.no_grad()
def compute_geometry_diagnostics(
    student_hidden_full: torch.Tensor,
    teacher_hidden_full: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
    bandwidth_schedule: MMDBandwidthSchedule,
    sinkhorn_epsilon_schedule: SinkhornEpsilonSchedule,
    step: int,
    gather_fn: Optional[Callable] = None,
    prefix: str = "diag",
    eps: float = GEOMETRY_EPS,
    distances: Optional[DiagnosticDistances] = None,
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
        sinkhorn_epsilon_schedule: this hook's own `SinkhornEpsilonSchedule`, also
            checkpointed. Updated from horizon 1's d_st_std and shared by every horizon.
        step: optimizer step, both schedules' warmup unit.
        gather_fn: optional `(s, t) -> (s_all, t_all)` that pools states across ranks
            (see `gather_predictive_states`); None keeps the rank-local population.
        distances: this micro-iteration's `diagnostic_distances(...)`, already built for
            the SAME population (same gather_fn); None builds them here.

    Returns:
        Flat `{key: float}`; empty if fewer than 2 samples are available.
    """
    if distances is None:
        distances = diagnostic_distances(
            student_hidden_full, teacher_hidden_full, pred_pos_mask, tot_mask_regions, k_toks, gather_fn
        )
    if distances is None:
        return {}
    s, t = distances.s, distances.t
    num_horizons = s.shape[1]

    per_horizon = []
    bandwidth = None
    sinkhorn_eps = None
    # Every distance-based metric below takes horizon idx's three matrices as arguments; none
    # rebuilds them. MMD and the bandwidth use their squares.
    for idx, (d_ss, d_tt, d_st) in enumerate(distances.matrices):
        d2_ss, d2_tt, d2_st = d_ss.square(), d_tt.square(), d_st.square()
        if idx == 0:
            bandwidth = bandwidth_schedule.update_from_sq_dists(step, d2_ss, d2_tt, d2_st)
        metrics = metrics_from_distances(d_ss, d_tt, d_st, eps=eps)
        if idx == 0:
            sinkhorn_eps = sinkhorn_epsilon_schedule.update(step, metrics["d_st_std"])
        # Sinkhorn's ground cost is the Euclidean D_ST itself (float32, see diagnostic_distances).
        metrics["sinkhorn"] = sinkhorn_from_cost(d_st, epsilon=sinkhorn_eps)
        metrics["mmd"] = mmd_from_sq_dists(d2_ss, d2_tt, d2_st, bandwidth, paired=True)
        metrics.update(statistics_matching(s[:, idx], t[:, idx], eps=eps))
        per_horizon.append(torch.stack([metrics[name].float() for name in METRIC_NAMES]))
        del d2_ss, d2_tt, d2_st, metrics

    values = torch.stack(per_horizon).tolist()  # (num_horizons, num_metrics), one host sync
    del per_horizon, s, t

    out: Dict[str, float] = {
        f"{prefix}/num_horizons": float(num_horizons),
        f"{prefix}/mmd_bandwidth": float(bandwidth),
        f"{prefix}/sinkhorn_epsilon": float(sinkhorn_eps),
    }
    for idx, row in enumerate(values):
        for name, value in zip(METRIC_NAMES, row):
            out[f"{prefix}/h{idx + 1}/{name}"] = value
    return out
