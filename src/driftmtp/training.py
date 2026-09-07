"""Phase 4 training-time integration: token drifting loss + stability logging.

Implements `claudedriftingplan.md`'s PHASE 4 -- Training Integration:

* `compute_drift_loss` wraps `features.extract_predictive_states` +
  `drifting.token_drifting_loss` into the one call the patched `pretrain.py`
  makes per iteration, mirroring how Phase 2's `validation.py` wraps the
  same primitives for the diagnostics path. Unlike the diagnostics path,
  the student extraction here is NOT detached -- `L_drift`'s gradient must
  reach the student parameters through `x = phi_S(u^S_{r,j})`.

* `DriftAccumulationTracker` implements the "Gradient-accumulation
  stability logging" note under PHASE 4: `S_j` and `lambda_{tau,j}` (and,
  by the same reasoning, per-temperature raw/normalized field RMS) are each
  a nonlinear function of the R-sized micro-batch pool, so their spread
  across the `gradient_accumulation_iters` micro-substeps of one optimizer
  step -- not just their mean -- is the concrete signal for whether R is
  large enough for stable kernel statistics at a given horizon. The spec
  explicitly requires this mean/std/ratio triplet for `S_j` and
  `lambda_{tau,j}`; this implementation applies the same triplet uniformly
  to raw/normalized field RMS too (a superset of the letter of the spec),
  since those are equally a nonlinear function of the same R-sized pool and
  a mean-only view would silently hide the same small-R instability the
  spec calls out `S_j`/`lambda_{tau,j}` for.

  Within one micro-substep, `token_drifting_loss` produces one `S_j` and
  one `lambda_{tau,j}` per active horizon `j` (see `drifting.py`). The
  tracker does NOT pool these across horizons into a single whole-window
  scalar: `k_toks` (the number of active horizons) is randomized per
  optimizer step, and higher horizons are expected to have inherently worse
  student/teacher distributional match, so "the average over all active
  horizons" would mix a different horizon population every time `k_toks`
  changes -- not a fair "is this improving" curve to read as iteration goes
  up. `DriftAccumulationTracker` therefore ONLY tracks a per-anchor-horizon
  breakdown -- using the same `features.anchor_horizons()` set already
  established for Phase 2's ECE/MMD/Sinkhorn diagnostics (absolute horizons
  `{1, 2, 3, mid, k_toks}`, capped at 5) -- so a fixed horizon's
  `S_j`/`lambda_{tau,j}`/field-RMS can be read as one continuous, comparable
  time series regardless of what `k_toks` was on any given step. `k_toks`
  itself is tracked with the same mean/std/ratio treatment for the same
  reason `S_j` and the rest are: it lets a reader correlate the two, and
  doubles as a self-check (constant `k_toks` within one window is a
  structural guarantee of the caller -- see below -- so a nonzero std here
  would mean that guarantee broke).

  A horizon that drops out of the anchor set (because a later window's
  `k_toks` no longer produces it) is actively pruned rather than left to go
  stale -- see `update()`'s docstring. Native `pretrain.py`'s own
  `running_pos_*[pos]` per-position metrics have exactly this staleness bug
  (a position stops being reached when `k_toks` shrinks, but its
  `RunningMean` is never reset, so `.compute()` silently keeps returning an
  old average from whatever `k_toks` regime last touched it); this tracker
  deliberately does not repeat that.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .drifting import DEFAULT_TEMPERATURES, DriftFieldResult, token_drifting_loss
from .features import anchor_horizons, extract_predictive_states


@dataclass
class DriftLossResult:
    loss: torch.Tensor  # scalar, grad-carrying through the student query states only
    per_horizon_results: List[DriftFieldResult]


def compute_drift_loss(
    student_hidden_full: torch.Tensor,
    teacher_hidden_full: torch.Tensor,
    pred_pos_mask: torch.Tensor,
    tot_mask_regions: int,
    k_toks: int,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    eta: float = 1.0,
    use_column_norm: bool = True,
    exclude_self_positive: bool = True,
) -> DriftLossResult:
    """claudedriftingplan.md PHASE 4: `L_drift^token`, grad-carrying into the student.

    Args mirror `validation.compute_training_diagnostics`'s hidden-state
    args exactly (same `pred_pos_mask` slicing, same shape contract), so
    the two hooks can share one training step's forward passes -- see
    "Guaranteed length match between student and teacher" in
    claudedriftingplan.md.

    student_hidden_full: (data_bsz, S, d_h), NOT detached -- this is the
        live student forward pass; `extract_predictive_states` only slices,
        it does not stop gradients, so the returned `loss` back-propagates
        into the student through this tensor.
    teacher_hidden_full: (data_bsz, S, d_h), from the frozen teacher's
        `torch.no_grad()` forward pass. `token_drifting_loss` detaches it
        again internally regardless (belt-and-suspenders, per
        claudedriftingplan.md's "Teacher features ... MUST be detached").
    """
    x = extract_predictive_states(
        student_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    )
    y_pos = extract_predictive_states(
        teacher_hidden_full, pred_pos_mask, tot_mask_regions, k_toks
    )
    loss, per_horizon_results = token_drifting_loss(
        x,
        y_pos,
        temperatures=temperatures,
        eta=eta,
        use_column_norm=use_column_norm,
        exclude_self_positive=exclude_self_positive,
    )
    return DriftLossResult(loss=loss, per_horizon_results=per_horizon_results)


def _mean_std_ratio(values: Sequence[float]) -> Dict[str, float]:
    t = torch.tensor(list(values), dtype=torch.float64)
    mean = t.mean().item()
    std = t.std(unbiased=False).item() if t.numel() > 1 else 0.0
    ratio = mean / (std + 1e-12)
    return {"mean": mean, "std": std, "ratio": ratio}


class DriftAccumulationTracker:
    """Rolling window over one optimizer step's `gradient_accumulation_iters`
    micro-substeps, mirroring the `RunningMean(window=gradient_accumulation_iters)`
    idiom already used elsewhere in `pretrain.py` for the same "aggregate over
    the micro-substeps of one optimizer step, `.compute()` only at
    log_iter_interval-aligned iterations" pattern -- see claudedriftingplan.md
    PHASE 4's "Gradient-accumulation stability logging" note. Deliberately a
    plain Python/deque implementation (not `torchmetrics.RunningMean`)
    because it needs std, not just mean.
    """

    def __init__(self, temperatures: Sequence[float], window: int):
        self.temperatures = list(temperatures)
        self.window = window
        self._k_toks: deque = deque(maxlen=window)

        # Per-anchor-horizon breakdown (see module docstring). Populated lazily,
        # keyed by absolute horizon j -- a horizon only appears here once some
        # `update()` call's anchor_horizons(k_toks) includes it, and is pruned
        # entirely (not just cleared) the moment a later call's anchor set no
        # longer includes it, so `compute()` can never report stale data from a
        # past k_toks regime.
        self._anchor_s_j: Dict[int, deque] = {}
        self._anchor_lambda_tau: Dict[int, Dict[float, deque]] = {}
        self._anchor_raw_field_rms: Dict[int, Dict[float, deque]] = {}
        self._anchor_normalized_field_rms: Dict[int, Dict[float, deque]] = {}
        # Affinity masses: the kernel-saturation / self-anchor-starvation signal
        # PHASE 5 asks to check for. See `TemperatureFieldDiagnostics` in
        # drifting.py -- under use_column_norm=True these are the mass-split
        # prefactor on the field, so mass_neg (or mass_pos) collapsing toward 0
        # means the drift signal is being silenced by the prefactor rather than
        # by genuine student/teacher convergence. Identically 1.0 when
        # use_column_norm=False, in which case they are a config self-check.
        self._anchor_mass_pos: Dict[int, Dict[float, deque]] = {}
        self._anchor_mass_neg: Dict[int, Dict[float, deque]] = {}

    def update(self, per_horizon_results: List[DriftFieldResult], k_toks: int) -> None:
        """Record one micro-substep's per-horizon results, broken out per
        anchor horizon (see `anchor_horizons`) -- no whole-window
        horizon-averaged scalar is tracked (see module docstring). `k_toks`
        is also recorded directly.

        `per_horizon_results[j - 1]` must be horizon `j` (1-indexed) -- the
        same convention `drifting.token_drifting_loss` and
        `features.extract_predictive_states` already use -- so
        `len(per_horizon_results)` must equal `k_toks`.
        """
        if not per_horizon_results:
            raise ValueError("per_horizon_results must be non-empty (k_toks >= 1)")
        if len(per_horizon_results) != k_toks:
            raise ValueError(
                f"len(per_horizon_results) ({len(per_horizon_results)}) must "
                f"equal k_toks ({k_toks})"
            )

        self._k_toks.append(k_toks)

        anchors = anchor_horizons(k_toks)
        self._prune_stale_anchor_horizons(anchors)
        for j in anchors:
            result = per_horizon_results[j - 1]
            self._anchor_s_j.setdefault(j, deque(maxlen=self.window)).append(
                result.S_j.item()
            )
            for tau in self.temperatures:
                diag = next(d for d in result.per_temperature if d.tau == tau)
                self._anchor_lambda_tau.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.lambda_tau)
                self._anchor_raw_field_rms.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.raw_field_rms)
                self._anchor_normalized_field_rms.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.normalized_field_rms)
                self._anchor_mass_pos.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.mass_pos)
                self._anchor_mass_neg.setdefault(j, {}).setdefault(
                    tau, deque(maxlen=self.window)
                ).append(diag.mass_neg)

    def _prune_stale_anchor_horizons(self, current_anchors: List[int]) -> None:
        """Drop any previously-tracked anchor horizon that is NOT part of this
        call's anchor set, so a horizon that stops being an anchor (k_toks
        changed) can never silently report stale data at a later `compute()`.
        Horizons that remain anchors need no special handling here -- their
        bounded deques self-heal via normal maxlen eviction once a full
        window's worth of fresh appends has happened (see module docstring).
        """
        stale = set(self._anchor_s_j) - set(current_anchors)
        for j in stale:
            self._anchor_s_j.pop(j, None)
            self._anchor_lambda_tau.pop(j, None)
            self._anchor_raw_field_rms.pop(j, None)
            self._anchor_normalized_field_rms.pop(j, None)
            self._anchor_mass_pos.pop(j, None)
            self._anchor_mass_neg.pop(j, None)

    def compute(self, prefix: str = "drift") -> Dict[str, float]:
        """Mean/std/ratio over whatever is currently in the window (up to the
        last `window` `update()` calls -- exactly one optimizer step's worth,
        provided `.update()` is called once per micro-substep and `.compute()`
        is only read at log_iter_interval-aligned iterations, both of which
        `pretrain.py` guarantees by construction)."""
        out: Dict[str, float] = {}

        def emit(name: str, values: Sequence[float]) -> None:
            for stat_name, value in _mean_std_ratio(values).items():
                out[f"{prefix}/{name}_{stat_name}"] = value

        emit("k_toks", self._k_toks)

        # Per-anchor-horizon breakdown: only horizons touched since the last
        # prune are present (see `_prune_stale_anchor_horizons`), so a horizon
        # absent from the current window's anchor set simply doesn't appear
        # this step -- sparse keys, not zero-filled or stale, matching Phase
        # 2's `validation.compute_training_diagnostics` convention for the
        # same underlying reason.
        for j in sorted(self._anchor_s_j):
            emit(f"S_j_h{j}", self._anchor_s_j[j])
        for j in sorted(self._anchor_lambda_tau):
            for tau in self.temperatures:
                if tau in self._anchor_lambda_tau[j]:
                    emit(f"lambda_tau_{tau}_h{j}", self._anchor_lambda_tau[j][tau])
        for j in sorted(self._anchor_raw_field_rms):
            for tau in self.temperatures:
                if tau in self._anchor_raw_field_rms[j]:
                    emit(f"raw_field_rms_tau_{tau}_h{j}", self._anchor_raw_field_rms[j][tau])
        for j in sorted(self._anchor_normalized_field_rms):
            for tau in self.temperatures:
                if tau in self._anchor_normalized_field_rms[j]:
                    emit(
                        f"normalized_field_rms_tau_{tau}_h{j}",
                        self._anchor_normalized_field_rms[j][tau],
                    )
        # Per-temperature loss contribution, eta^2 * ||V_tilde_tau||^2 / d_h in eta=1
        # units: the "drift loss per temperature" breakdown. Reads ~1.0 while a rung is
        # renormalized to unit scale, and decays toward 0 once its raw field falls below
        # eps and rms_field_normalize's (lambda_tau + eps) denominator damps it -- which
        # is the only mechanism by which the drift term can vanish at convergence. NOTE
        # the total drift loss is NOT the sum of these: the loss is the squared norm of
        # the SUMMED field, so it also carries the cross terms between rungs.
        for j in sorted(self._anchor_normalized_field_rms):
            for tau in self.temperatures:
                if tau in self._anchor_normalized_field_rms[j]:
                    emit(
                        f"loss_tau_{tau}_h{j}",
                        [v * v for v in self._anchor_normalized_field_rms[j][tau]],
                    )
        for j in sorted(self._anchor_mass_pos):
            for tau in self.temperatures:
                if tau in self._anchor_mass_pos[j]:
                    emit(f"mass_pos_tau_{tau}_h{j}", self._anchor_mass_pos[j][tau])
        for j in sorted(self._anchor_mass_neg):
            for tau in self.temperatures:
                if tau in self._anchor_mass_neg[j]:
                    emit(f"mass_neg_tau_{tau}_h{j}", self._anchor_mass_neg[j][tau])
        return out
