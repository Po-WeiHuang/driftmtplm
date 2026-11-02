"""Algorithm-2 kernel normalization and the token drifting field/loss.

Implements `claudedriftingplan.md`'s "Algorithm-2 Kernel Normalization" and
"Token Drifting Loss" sections. The per-horizon primitives operate on
`(R, d_h)` tensors (one prediction horizon at a time); `token_drifting_loss`
loops over the `k_toks` horizon axis of `features.extract_predictive_states`'s
`(R, k_toks, d_h)` output.

## `use_column_norm`

`use_column_norm` (default `True`) selects between two DISTINCT field
formulations from the paper -- not a partial/simplified version of the same
one. See claudedriftingplan.md's "Column-Normalization Toggle" for the
authoritative spec.

* `use_column_norm=True` (default): **Algorithm 2** exactly. Positive and
  negative logits are CONCATENATED into one `(R, 2R)` matrix, then row-
  softmax (`A_row`, per-query) and column-softmax (`A_col`, per-sample
  across queries) are combined as `A = sqrt(A_row * A_col)`, split back into
  `A_pos`/`A_neg`, and reweighted by the opposite population's total mass
  (`drifting_weights`: `W_pos = A_pos * sum(A_neg)`, `W_neg = A_neg *
  sum(A_pos)`). Because `A_pos`/`A_neg` share one normalizer, `sum(A_pos) +
  sum(A_neg) = 1` per query, and this cross-population coupling is what
  makes the field self-damping as a query's population membership becomes
  ambiguous (part of how the field vanishes smoothly at the `p=q`
  equilibrium, Proposition 1).
* `use_column_norm=False`: the paper's **Eq. 8** formulation (the simpler,
  pre-Algorithm-2 field) -- `V_{p,q}(x) = V^+_p(x) - V^-_q(x)`, where each
  term is normalized by its OWN, INDEPENDENT partition function
  (`Z_p(x) = E_{y+~p}[k(x,y+)]`, `Z_q(x) = E_{y-~q}[k(x,y-)]`). This means
  `A_pos = softmax(logit_pos, dim=1)` and `A_neg = softmax(logit_neg, dim=1)`
  computed SEPARATELY (two independent softmax calls -- NOT concatenated
  first), each already summing to `1` on its own, with no cross-population
  coupling at all: the attraction and repulsion terms never see each
  other's scale. Since each already sums to `1`, `drifting_weights`'s
  cross-multiply would be a no-op here, so it's skipped entirely rather than
  applied-but-inert; `A_pos`/`A_neg` are used directly as the field weights.

Both branches reuse the identical self-negative masking, feature-scale
normalization `S_j`, effective temperature, and RMS field normalization --
only the affinity-construction (and, for Algorithm 2 only, the opposite-
population reweighting) step differs. `raw_drift_field` computes
`W_pos @ y_pos - W_neg @ y_neg` unchanged in both cases; this equals each
formulation's `V^+ - V^-` exactly because `sum(W_pos) == sum(W_neg)` in both
branches, so the implicit `-x` terms in each half cancel algebraically.

## `tie_self_negative` (Eq. 8 only)

See tied_self_negative_derivation.md and claudetiedselfnegplan.md. With the
self-positive kept (`exclude_self_positive=False`), write `a_i = A_pos[i, i]`,
`N_i = sum_{j!=i} A_neg[i, j] x_j` (other-student mean) and
`(1 - a_i) P_i = sum_{j!=i} A_pos[i, j] y_j` (other-teacher mass). The masked
Eq. 8 field splits exactly into

    V_i = a_i (y_i - x_i) + (1 - a_i)(P_i - N_i) + a_i (x_i - N_i)
          paired pull       unpaired drift          spurious push

The spurious push exists only because the self-positive has no matching
negative, and it does not vanish at student == teacher. `tie_self_negative`
gives the query's own detached student state the negative weight `a_i` and
scales the other negatives by `1 - a_i`:

    W_neg[i, i] = a_i,   W_neg[i, j] = (1 - a_i) A_neg[i, j]  (j != i)

Rows still sum to 1 (so the `-x` cancellation above still holds), the
self-negative displacement is `x_i - x_i = 0`, and the field becomes exactly
`paired + unpaired`, which is 0 at x == y.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from .normalization import (
    effective_temperature,
    feature_scale,
    pairwise_distances,
    rms_field_normalize,
    self_pair_mask,
)

DEFAULT_TEMPERATURES: Tuple[float, ...] = (0.02, 0.05, 0.2)


def algorithm2_affinity(
    logit_pos: torch.Tensor,
    logit_neg: torch.Tensor,
    use_column_norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Either Algorithm 2's affinity (steps 5-8, pre-reweighting) or Eq. 8's
    separately-normalized affinity.

    logit_pos, logit_neg: (R, R). `logit_neg` MUST already carry self-masking
    (e.g. `-inf` on the diagonal) -- this function does not mask anything
    itself.

    Returns (A_pos, A_neg), each (R, R):
    * `use_column_norm=True`: concatenate, row+column softmax, geometric
      mean, split -- the Algorithm-2 affinity BEFORE the opposite-population
      reweighting (`drifting_weights`, applied separately by the caller;
      each of `A_pos`, `A_neg` here sums to something in `[0, 1]` per query,
      jointly summing to `1`).
    * `use_column_norm=False`: Eq. 8 -- `softmax(logit_pos)` and
      `softmax(logit_neg)` computed independently (their own `Z_p`/`Z_q`),
      NOT concatenated first. Each already sums to `1` per query on its own
      -- the caller should use these directly as field weights, skipping
      `drifting_weights` entirely (it would be a no-op here).

    See the module docstring for the full derivation of why these are two
    genuinely different field formulations, not a partial/simplified
    version of the same one.
    """
    if logit_pos.shape != logit_neg.shape or logit_pos.dim() != 2 or logit_pos.shape[0] != logit_pos.shape[1]:
        raise ValueError(
            f"logit_pos and logit_neg must both be square (R, R) and match; "
            f"got {tuple(logit_pos.shape)} vs {tuple(logit_neg.shape)}"
        )

    if use_column_norm:
        num_regions = logit_pos.shape[0]
        logit = torch.cat([logit_pos, logit_neg], dim=1)  # (R, 2R), sample axis = dim 1
        a_row = torch.softmax(logit, dim=1)  # per-query (row) normalization
        a_col = torch.softmax(logit, dim=0)  # per-sample (column) normalization
        a = torch.sqrt(a_row * a_col)
        a_pos, a_neg = a[:, :num_regions], a[:, num_regions:]
    else:
        # Eq. 8: Z_p(x) and Z_q(x) are independent partition functions -- two
        # separate softmax calls, NOT one softmax over the concatenation.
        a_pos = torch.softmax(logit_pos, dim=1)
        a_neg = torch.softmax(logit_neg, dim=1)
    return a_pos, a_neg


def drifting_weights(
    a_pos: torch.Tensor, a_neg: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Algorithm-2 opposite-population weighting:
    W_pos = A_pos * sum_l(A_neg), W_neg = A_neg * sum_l(A_pos)."""
    w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)
    return w_pos, w_neg


def tied_negative_weights(a_pos: torch.Tensor, a_neg: torch.Tensor) -> torch.Tensor:
    """Eq. 8 negative weights with the self-negative tied to the self-positive:
    W_neg[i, i] = a_ii := A_pos[i, i], W_neg[i, j] = (1 - a_ii) A_neg[i, j] for j != i.

    `a_neg` must be the self-masked row softmax (zero diagonal, rows sum to 1), so each
    returned row sums to a_ii + (1 - a_ii) = 1. See the module docstring."""
    a_self = a_pos.diagonal()
    return (1 - a_self)[:, None] * a_neg + torch.diag_embed(a_self)


def raw_drift_field(
    w_pos: torch.Tensor, w_neg: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor
) -> torch.Tensor:
    """V = W_pos @ y_pos - W_neg @ y_neg."""
    return w_pos @ y_pos - w_neg @ y_neg


@torch.no_grad()
def field_decomposition(
    a_pos: torch.Tensor, a_neg: torch.Tensor, x: torch.Tensor, y_pos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(paired, unpaired, spurious), each (R, d_h), of the Eq. 8 field (module docstring):

        paired   = a_ii (y_i - x_i)
        unpaired = sum_{j!=i} A_pos[i, j] y_j - (1 - a_ii) N_i    (= (1 - a_ii)(P_i - N_i))
        spurious = a_ii (x_i - N_i)

    with N_i = (A_neg @ x)_i. The masked field is paired + unpaired + spurious and the tied
    field is paired + unpaired. `unpaired` uses the off-diagonal A_pos directly instead of
    (1 - a_ii) P_i, so a_ii -> 1 (underflow at small tau) never divides by zero.

    x is the detached student (the negative population); y_pos the teacher. Computed in
    fp32 for the diagnostics, never part of the graph."""
    a_pos, a_neg, x, y_pos = a_pos.float(), a_neg.float(), x.float(), y_pos.float()
    a_self = a_pos.diagonal()[:, None]
    n = a_neg @ x
    paired = a_self * (y_pos - x)
    unpaired = a_pos.diagonal_scatter(torch.zeros_like(a_self[:, 0])) @ y_pos - (1 - a_self) * n
    spurious = a_self * (x - n)
    return paired, unpaired, spurious


@dataclass
class TemperatureFieldDiagnostics:
    tau: float
    lambda_tau: float
    raw_field_rms: float
    normalized_field_rms: float
    mass_pos: float  # mean_i sum_j A_pos[i,j] -- always 1.0 when use_column_norm=False
    mass_neg: float  # mean_i sum_j A_neg[i,j] -- always 1.0 when use_column_norm=False
    # For use_column_norm=True, raw_field_rms ~= mass_pos*mass_neg*(true population gap), so a
    # shrinking raw_field_rms alongside mass_neg (or mass_pos) collapsing toward 0 is the
    # self-anchor-starvation signature flagged in claudedriftingplan.md's "Column-Normalization
    # Toggle" section -- the drift signal is being silenced by the mass-split prefactor, not by
    # genuine convergence of the teacher/student populations.
    # Effective neighbour count of each population's affinity row, renormalized to a
    # distribution p (claudedriftingplan.md > Kernel Temperature > Effective neighbour
    # count): N_eff = 1 / sum_l p_rl^2, averaged over queries. ~1 means hard nearest
    # neighbour (tau too small), ~R means uniform weights (tau too large).
    n_eff_pos: float = 0.0
    n_eff_neg: float = 0.0
    # mean_r p_pos[r, r]: positive mass on the query's own paired teacher. Identically 0.0
    # when exclude_self_positive=True; near 1 means the positive half of the field has
    # collapsed to per-token regression y_pos[r] - x[r].
    self_pos_weight: float = 0.0
    # The divisor actually applied to this rung's raw field: lambda_tau (live) unless a
    # frozen field scale was passed (claudetiedselfnegplan.md). lambda_tau stays the live
    # batch RMS either way.
    lambda_used: float = 0.0
    # mean_r W_neg[r, r]: 0.0 when the self-negative is masked, == self_pos_weight when
    # tie_self_negative=True. A config self-check.
    self_neg_weight: float = 0.0
    # RMS (raw_field_rms convention, pre-scale S_j-normalised units) of the three parts of
    # the Eq. 8 field, see `field_decomposition`. They do NOT add in RMS: paired and
    # spurious are anti-aligned near equilibrium, hence cos_paired_spurious (mean per-row
    # cosine). None under use_column_norm=True, where the decomposition does not hold.
    paired_rms: Optional[float] = None
    unpaired_rms: Optional[float] = None
    spurious_rms: Optional[float] = None
    cos_paired_spurious: Optional[float] = None


def _affinity_row_stats(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(mean_r N_eff, mean_r p[r, r]) of an (R, R) affinity, each row renormalized to sum
    to 1. `drifting_weights` only rescales whole rows, so this is the same before and
    after the Algorithm-2 reweighting.

    Under use_column_norm=True a row's geometric-mean affinity can underflow to all zeros
    at very small tau. Such a row has no distribution (and contributes nothing to the
    field), so it is left out of both means; NaN if every row is."""
    a = a.detach().float()
    row_mass = a.sum(dim=1)
    live = row_mass > 0
    p = a[live] / row_mass[live, None]
    n_eff = 1.0 / p.pow(2).sum(dim=1)
    return n_eff.mean(), p.gather(1, live.nonzero()).mean()


@dataclass
class DriftFieldResult:
    V_multi: torch.Tensor  # (R, d_h) multi-temperature-SUMMED field, stop-gradient-safe
    x_normalized: torch.Tensor  # (R, d_h) = x / S_j, grad-carrying through x
    S_j: torch.Tensor  # detached scalar
    per_temperature: List[TemperatureFieldDiagnostics] = field(default_factory=list)
    total_rms: float = 0.0  # RMS of V_multi, the summed scaled field that enters the target


def _field_rms(V: torch.Tensor) -> float:
    d_h = V.shape[-1]
    return torch.sqrt((V.detach().pow(2).sum(dim=-1) / d_h).mean()).item()


def _mean_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # Rows where either vector is zero (a_ii == 0) count as cosine 0.
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-12).mean().item()


def token_drift_field(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    use_column_norm: bool = True,
    eps: float = 1e-6,
    exclude_self_positive: bool = True,
    tie_self_negative: bool = False,
    field_scale: Optional[torch.Tensor] = None,
) -> DriftFieldResult:
    """Full per-horizon token drifting field: claudedriftingplan.md "Feature Scale"
    through "Multi-Temperature Field Normalization".

    x: (R, d_h) student query at one horizon, grad-carrying.
    y_pos: (R, d_h) teacher positive at the same horizon; detached
        internally regardless of its incoming `requires_grad` (spec:
        "Teacher features ... MUST be detached").
    use_column_norm: see module docstring.
    tie_self_negative: see module docstring. Eq. 8 only (raises with
        use_column_norm=True). With exclude_self_positive=True a_ii == 0, so it is a no-op.
    field_scale: optional (n_tau,) detached divisors replacing each rung's live
        lambda_tau; a NaN entry means "use the live value" (see `rms_field_normalize`).
    """
    if x.dim() != 2 or y_pos.shape != x.shape:
        raise ValueError(
            f"x and y_pos must both be (R, d_h) and match; got "
            f"{tuple(x.shape)} vs {tuple(y_pos.shape)}"
        )
    if tie_self_negative and use_column_norm:
        raise ValueError(
            "tie_self_negative is only defined for the Eq. 8 field (use_column_norm=False)."
        )
    if field_scale is not None and field_scale.shape != (len(temperatures),):
        raise ValueError(
            f"field_scale must have shape ({len(temperatures)},), got {tuple(field_scale.shape)}"
        )

    num_regions, d_h = x.shape
    y_pos = y_pos.detach()
    y_neg = x.detach()  # negative population: detached student states (self-interaction masked below)
    self_mask = self_pair_mask(num_regions, device=x.device)

    dist_pos, dist_neg = pairwise_distances(x, y_pos, y_neg)
    s_j = feature_scale(
        dist_pos, dist_neg, self_mask, d_h, exclude_self_positive=exclude_self_positive
    )

    x_n = x / s_j
    y_pos_n = y_pos / s_j
    y_neg_n = y_neg / s_j
    dist_pos_n = dist_pos / s_j
    dist_neg_n = dist_neg / s_j

    per_temp_fields = []
    diagnostics: List[TemperatureFieldDiagnostics] = []
    for t, tau in enumerate(temperatures):
        tilde_tau = effective_temperature(tau, d_h)
        # SYMMETRIC self-exclusion. The negative diagonal (the query's own
        # detached student state) has always been masked. With
        # exclude_self_positive=True the POSITIVE diagonal -- the query's own
        # paired teacher token -- is masked too.
        #
        # Why: y_pos[r] is the teacher state for the SAME token as the query
        # x[r], so as the student converges toward the teacher that pair's
        # distance goes to zero and, at low tau, its softmax weight goes to 1.
        # The positive mass then collapses onto a single self-pair while the
        # negative population still has its own self-pair removed. That
        # asymmetry leaves a systematic residual field that does not vanish at
        # equilibrium, so "student == teacher" is NOT a fixed point of the
        # original construction. Masking both restores the symmetry the
        # paper's unpaired setup has for free.
        logit_pos = -dist_pos_n / tilde_tau
        if exclude_self_positive:
            logit_pos = logit_pos.masked_fill(self_mask, float("-inf"))
        logit_neg = (-dist_neg_n / tilde_tau).masked_fill(self_mask, float("-inf"))

        a_pos, a_neg = algorithm2_affinity(logit_pos, logit_neg, use_column_norm=use_column_norm)
        mass_pos = a_pos.sum(dim=1)  # (R,); == 1.0 identically when use_column_norm=False
        mass_neg = a_neg.sum(dim=1)
        if use_column_norm:
            # Algorithm 2: reweight by the opposite population's total mass.
            w_pos, w_neg = drifting_weights(a_pos, a_neg)
        elif tie_self_negative:
            # Eq. 8 with the self-negative tied to the self-positive: both rows still sum
            # to 1, and the spurious a_ii (x_i - N_i) push is removed (module docstring).
            w_pos, w_neg = a_pos, tied_negative_weights(a_pos, a_neg)
        else:
            # Eq. 8: A_pos, A_neg already sum to 1 independently -- use them
            # directly as field weights (drifting_weights would be a no-op).
            w_pos, w_neg = a_pos, a_neg
        v_tau = raw_drift_field(w_pos, w_neg, y_pos_n, y_neg_n)
        scale_t = None if field_scale is None else field_scale[t]
        v_tau_tilde, lambda_tau = rms_field_normalize(v_tau, eps=eps, scale=scale_t)
        lambda_used = (
            lambda_tau
            if scale_t is None
            else torch.where(torch.isnan(scale_t), lambda_tau, scale_t.to(lambda_tau.dtype))
        )
        n_eff_pos, self_pos_weight = _affinity_row_stats(a_pos)
        n_eff_neg, _ = _affinity_row_stats(a_neg)

        decomposition = {}
        if not use_column_norm:
            paired, unpaired, spurious = field_decomposition(a_pos, a_neg, y_neg_n, y_pos_n)
            decomposition = dict(
                paired_rms=_field_rms(paired),
                unpaired_rms=_field_rms(unpaired),
                spurious_rms=_field_rms(spurious),
                cos_paired_spurious=_mean_row_cosine(paired, spurious),
            )

        per_temp_fields.append(v_tau_tilde)
        diagnostics.append(
            TemperatureFieldDiagnostics(
                tau=tau,
                lambda_tau=lambda_tau.item(),
                raw_field_rms=_field_rms(v_tau),
                normalized_field_rms=_field_rms(v_tau_tilde),
                mass_pos=mass_pos.mean().item(),
                mass_neg=mass_neg.mean().item(),
                n_eff_pos=n_eff_pos.item(),
                n_eff_neg=n_eff_neg.item(),
                self_pos_weight=self_pos_weight.item(),
                lambda_used=lambda_used.item(),
                self_neg_weight=w_neg.detach().diagonal().float().mean().item(),
                **decomposition,
            )
        )

    # SUM (not mean) over temperatures, matching lambertae/drifting's
    # `force_across_R = force_across_R + total_force_R / force_scale`. Summing keeps
    # each rung's contribution at its own scale, so a temperature whose raw field has
    # collapsed below `eps` (and is therefore damped toward zero by
    # rms_field_normalize's V / (lambda + eps)) simply drops out, instead of dividing
    # the two surviving rungs by n_tau. See TemperatureFieldDiagnostics.
    v_multi = torch.stack(per_temp_fields, dim=0).sum(dim=0)
    return DriftFieldResult(
        V_multi=v_multi,
        x_normalized=x_n,
        S_j=s_j,
        per_temperature=diagnostics,
        total_rms=_field_rms(v_multi),
    )


def token_drifting_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    eta: float = 1.0,
    use_column_norm: bool = True,
    eps: float = 1e-6,
    exclude_self_positive: bool = True,
    tie_self_negative: bool = False,
    field_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[DriftFieldResult]]:
    """claudedriftingplan.md "Token Drifting Loss": stopped-target token
    drifting loss, mean-pooled over the feature dimension d_h and averaged
    over regions R and horizons k_r.

    x: (R, k_toks, d_h) student predictive states (grad-carrying), e.g. from
        `features.extract_predictive_states` on the student hidden states.
    y_pos: (R, k_toks, d_h) teacher positive states, same shape.
    use_column_norm: see the module docstring / `token_drift_field`.
    tie_self_negative: see the module docstring / `token_drift_field`.
    field_scales: optional (k_toks, n_tau) detached divisors; row j-1 goes to horizon j
        as `token_drift_field`'s `field_scale` (NaN = live lambda_tau).

    Returns (loss, per_horizon_results): `loss` is a scalar tensor whose
    gradient reaches `x` only (teacher, negatives, S_j, lambda_{tau,j}, and
    the drift target are all stop-gradient by construction -- see
    `token_drift_field`); `per_horizon_results` is one `DriftFieldResult`
    per horizon, exposed for Phase 4 logging (S_j, per-temperature
    lambda_{tau,j} / field RMS).
    """
    if x.dim() != 3 or y_pos.shape != x.shape:
        raise ValueError(
            f"x and y_pos must both be (R, k_toks, d_h) and match; got "
            f"{tuple(x.shape)} vs {tuple(y_pos.shape)}"
        )

    k_toks = x.shape[1]
    if field_scales is not None and field_scales.shape != (k_toks, len(temperatures)):
        raise ValueError(
            f"field_scales must have shape ({k_toks}, {len(temperatures)}), got "
            f"{tuple(field_scales.shape)}"
        )
    per_horizon_losses = []
    per_horizon_results: List[DriftFieldResult] = []
    for j in range(k_toks):
        result = token_drift_field(
            x[:, j, :],
            y_pos[:, j, :],
            temperatures=temperatures,
            use_column_norm=use_column_norm,
            eps=eps,
            exclude_self_positive=exclude_self_positive,
            tie_self_negative=tie_self_negative,
            field_scale=None if field_scales is None else field_scales[j],
        )
        target = (result.x_normalized + eta * result.V_multi).detach()
        # Mean (not sum) over d_h, matching the reference implementation's
        # `jnp.mean(diff ** 2, axis=(-1, -2))` in lambertae/drifting's drift_loss.py.
        # Makes L_drift O(eta^2) ~ 0.6 rather than O(eta^2 * d_h) ~ 1e3, so the
        # magnitude is comparable to L_MTP and independent of hidden size, so a single
        # fixed lambda_drift calibrated once at the start of training stays meaningful.
        loss_j = (result.x_normalized - target).pow(2).mean(dim=-1).mean()
        per_horizon_losses.append(loss_j)
        per_horizon_results.append(result)

    loss = torch.stack(per_horizon_losses).mean()
    return loss, per_horizon_results