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
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

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


def raw_drift_field(
    w_pos: torch.Tensor, w_neg: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor
) -> torch.Tensor:
    """V = W_pos @ y_pos - W_neg @ y_neg."""
    return w_pos @ y_pos - w_neg @ y_neg


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


@dataclass
class DriftFieldResult:
    V_multi: torch.Tensor  # (R, d_h) multi-temperature-SUMMED field, stop-gradient-safe
    x_normalized: torch.Tensor  # (R, d_h) = x / S_j, grad-carrying through x
    S_j: torch.Tensor  # detached scalar
    per_temperature: List[TemperatureFieldDiagnostics] = field(default_factory=list)


def _field_rms(V: torch.Tensor) -> float:
    d_h = V.shape[-1]
    return torch.sqrt((V.detach().pow(2).sum(dim=-1) / d_h).mean()).item()


def token_drift_field(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    use_column_norm: bool = True,
    eps: float = 1e-6,
    exclude_self_positive: bool = True,
) -> DriftFieldResult:
    """Full per-horizon token drifting field: claudedriftingplan.md "Feature Scale"
    through "Multi-Temperature Field Normalization".

    x: (R, d_h) student query at one horizon, grad-carrying.
    y_pos: (R, d_h) teacher positive at the same horizon; detached
        internally regardless of its incoming `requires_grad` (spec:
        "Teacher features ... MUST be detached").
    use_column_norm: see module docstring.
    """
    if x.dim() != 2 or y_pos.shape != x.shape:
        raise ValueError(
            f"x and y_pos must both be (R, d_h) and match; got "
            f"{tuple(x.shape)} vs {tuple(y_pos.shape)}"
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
    for tau in temperatures:
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
        else:
            # Eq. 8: A_pos, A_neg already sum to 1 independently -- use them
            # directly as field weights (drifting_weights would be a no-op).
            w_pos, w_neg = a_pos, a_neg
        v_tau = raw_drift_field(w_pos, w_neg, y_pos_n, y_neg_n)
        v_tau_tilde, lambda_tau = rms_field_normalize(v_tau, eps=eps)

        per_temp_fields.append(v_tau_tilde)
        diagnostics.append(
            TemperatureFieldDiagnostics(
                tau=tau,
                lambda_tau=lambda_tau.item(),
                raw_field_rms=_field_rms(v_tau),
                normalized_field_rms=_field_rms(v_tau_tilde),
                mass_pos=mass_pos.mean().item(),
                mass_neg=mass_neg.mean().item(),
            )
        )

    # SUM (not mean) over temperatures, matching lambertae/drifting's
    # `force_across_R = force_across_R + total_force_R / force_scale`. Summing keeps
    # each rung's contribution at its own scale, so a temperature whose raw field has
    # collapsed below `eps` (and is therefore damped toward zero by
    # rms_field_normalize's V / (lambda + eps)) simply drops out, instead of dividing
    # the two surviving rungs by n_tau. See TemperatureFieldDiagnostics.
    v_multi = torch.stack(per_temp_fields, dim=0).sum(dim=0)
    return DriftFieldResult(V_multi=v_multi, x_normalized=x_n, S_j=s_j, per_temperature=diagnostics)


def token_drifting_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    eta: float = 1.0,
    use_column_norm: bool = True,
    eps: float = 1e-6,
    exclude_self_positive: bool = True,
) -> Tuple[torch.Tensor, List[DriftFieldResult]]:
    """claudedriftingplan.md "Token Drifting Loss": stopped-target token
    drifting loss, mean-pooled over the feature dimension d_h and averaged
    over regions R and horizons k_r.

    x: (R, k_toks, d_h) student predictive states (grad-carrying), e.g. from
        `features.extract_predictive_states` on the student hidden states.
    y_pos: (R, k_toks, d_h) teacher positive states, same shape.
    use_column_norm: see the module docstring / `token_drift_field`.

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