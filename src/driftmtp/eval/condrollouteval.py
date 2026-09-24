"""Controlled-rollout diagnostic harness — Phase 2 core.

See `docs/controlled_rollout_plan.md` for the design.

This is a **diagnostic, not a scorer**. Free rollout (`lm_eval`) measures
whether answers are correct; this harness never scores answers. It runs the
student and the teacher over held-out documents in *training's own MTP
layout* and reports, per horizon, `ece_stud_gt`, `ece_stud_teach`, `mmd`
and `sinkhorn` — each ECE in both a marginal and a joint (whole-block) form.

Phase 2 scope: load litgpt student/teacher checkpoints, load held-out
documents from the same `lm_eval` task free rollout evaluates on, lay them
out by calling training's own `truncate_and_mask`, run both forward passes,
and return per-region/per-horizon logits, hidden states and targets.

Phase 3 scope: reduce those outputs to pooled per-horizon metrics and write
`controlled_rollout_<timestamp>.json` next to `lm_eval`'s own output, so hop 3
(`push_lmeval_metrics_to_wandb.py`) can log both rollouts into one wandb run.

The layout is training's, not a reimplementation of it:
`build_batch_layout` calls `litgpt.pretrain.truncate_and_mask` itself and
`prepare_block_mask` calls `GPT.reconstruct_block_mask`, the same entry point
`pretrain.py:1292` uses. Real text is consumed `P` tokens at a time and
`K = k_toks - 1` mask slots are *inserted* after each `P`, giving
`[P real][K mask][P real][K mask] ...`. No real token is ever overwritten —
the `K` source positions a region reads ahead into reappear as the next
region's real prefix. `use_block_mask=True` throughout, so a real token
attends only to real tokens (a mask slot can never contaminate a later
region's prefix) and a mask slot attends only to real tokens plus its *own*
region's mask slots (a later MTP never sees an earlier MTP).

**Sequences are truncated exactly as in training**, to
`truncation_length` (default 160) destination slots, consuming
`num_toks_consumed = mask_region_ct * P + K` real tokens per document (136 at
the shipped config with `k_toks=7`). This is deliberate and load-bearing: the
student only ever performed MTP at RoPE positions below that bound, so mask
tokens, the interleaved block mask and the remapped RoPE were never exercised
further out. Running the diagnostic beyond it would be pure extrapolation and
would risk measuring positional drift instead of MTP behaviour.

Only two things differ from `pretrain.py`, both in `build_batch_layout`:

1. `offset = 0` — training rolls the offset across steps to vary where
   regions land; one deterministic pass is what a diagnostic wants.
2. Training's `pad_token_id` / `prelude_token_ids` prediction-masking is not
   used; short documents' out-of-range horizons are tracked in a separate
   `valid` mask instead, which keeps the `(tot_mask_regions, k_toks)` reshape
   fixed and gives per-horizon rather than per-region granularity.

`truncation_length`, `mask_region_ct` and `micro_batch_size` are all config
values matching training's `hparams.singleshot.*` / `hparams.train.*`.

The teacher receives the same layout with the mask slots filled in by the
student's hard predictions (`pretrain.py:1426-1438`), so it never sees a
mask token id and both models are positionally identical.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch


# ---------------------------------------------------------------------------
# Job 1 — load litgpt student/teacher checkpoints
# ---------------------------------------------------------------------------


def load_litgpt_model(checkpoint_dir: Union[str, Path], precision: Optional[str] = None):
    """Load one litgpt (lit-format) checkpoint for inference.

    Mirrors `third_party/mtp-lm/litgpt/generate/base_mtp.py::main`'s loading
    pattern minus the parts specific to autoregressive generation (no KV
    cache, no `torch.compile`) — this harness always does a single forward
    pass over a full sequence, never incremental decoding.

    The model is built with `use_block_mask=False` because `GPT.__init__`
    reads its block-mask geometry off `hparams`, which we don't have (and
    which training reads from its own config). `prepare_block_mask` turns it
    on per batch instead, with that batch's geometry.

    Args:
        checkpoint_dir: directory containing `model_config.yaml` and
            `lit_model.pth`, e.g. what `pretrain.py`'s `save_checkpoint`
            already writes.
        precision: Fabric precision string (e.g. `"bf16-true"`); `None` lets
            Fabric pick its default.

    Returns:
        The `GPT` module, in eval mode, all parameters frozen.
    """
    import lightning as L
    from litgpt.config import Config
    from litgpt.model import GPT
    from litgpt.utils import check_valid_checkpoint_dir, extend_checkpoint_dir, load_checkpoint

    checkpoint_dir = extend_checkpoint_dir(Path(checkpoint_dir))
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric = L.Fabric(devices=1, precision=precision)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    with fabric.init_module(empty_init=True):
        model = GPT(config, hparams=None, use_block_mask=False)
    model.eval()
    model = fabric.setup_module(model)
    load_checkpoint(fabric, model, checkpoint_dir / "lit_model.pth")
    model.requires_grad_(False)
    return model


def load_litgpt_tokenizer(checkpoint_dir: Union[str, Path]):
    """Load the tokenizer from the same checkpoint directory as `load_litgpt_model`."""
    from litgpt.tokenizer import Tokenizer

    return Tokenizer(Path(checkpoint_dir))


def resolve_mask_token_ids(
    tokenizer,
    pattern: str = "<|mtp_special_token_{i}|>",
    num_tokens: int = 32,
) -> List[int]:
    """Token ids for the position-numbered `<MTP>` mask tokens.

    Matches the exact convention `third_party/mtp-lm/litgpt/pretrain.py`'s
    `setup()` uses to resolve `hparams.singleshot.min_mask_id`/`max_mask_id`
    (lines 574-587): the mask token at local offset `i` within a masked
    span is `pattern.format(i=i)`. Index `0` of the returned list is what
    training calls `min_mask_id`.
    """
    return [tokenizer.token_to_id(pattern.format(i=i)) for i in range(num_tokens)]


# ---------------------------------------------------------------------------
# Job 2 — load the same held-out documents free rollout evaluates on
# ---------------------------------------------------------------------------

# Tasks verified to have a real, substantial reference-continuation field --
# see docs/eval_pipeline/phase2_condrollouteval.md for the per-task check this
# was built from. Do NOT add a task here without first verifying its raw
# reference field actually holds worked reasoning/output text, not just a
# short final-answer label: AIME25 (`answer` = a bare number, e.g. `"70"`),
# BBH (`target` = a bare label, e.g. `"False"`), and GPQA (a multiple-choice
# letter/text) all fail this -- not enough tokens to fill more than a region
# or two. IFEval has no reference field at all (it's graded by checking
# instruction-following rules, not by comparing to any ground truth).
# CNN/DailyMail's `highlights` field DOES hold real summary text and was NOT
# rejected on the same grounds as the others -- it's excluded here only
# because it hasn't been deliberately decided on yet.
_SUPPORTED_TASKS = frozenset({"gsm8k_cot_singleshot"})


def load_task_documents(
    task_name: str,
    reference_field: Optional[str] = "answer",
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load `(prompt_text, reference_text)` pairs from an `lm_eval` task.

    Uses the exact same task loader `lm_eval` itself uses, so these are the
    same documents free rollout evaluates on this task.

    Restricted to `_SUPPORTED_TASKS` -- controlled rollout needs a real
    reference continuation with actual reasoning/output text to fill enough
    regions. Most benchmark tasks' raw reference field is just a short
    final-answer label (or, for IFEval, doesn't exist at all); see
    `_SUPPORTED_TASKS`'s comment for what was actually checked. Raises
    `ValueError` for anything not on the list rather than silently handing
    back a short label as if it were a proper reference.

    Args:
        task_name: an `lm_eval` task name. Must be in `_SUPPORTED_TASKS`.
        reference_field: name of the raw document field holding the full
            ground-truth continuation text. Default `"answer"` matches
            gsm8k: its raw `answer` field is the full worked solution, while
            the task's own `doc_to_target` template extracts only the final
            numeric answer for scoring (e.g. `"18"`) — not usable as a
            token-by-token controlled-rollout reference. Pass a different
            field name for other tasks, or `None` to fall back to
            `doc_to_target` directly.
        limit: optional cap on the number of documents (for quick checks).

    Returns:
        A list of `{"prompt_text": str, "reference_text": str}` dicts.
    """
    if task_name not in _SUPPORTED_TASKS:
        raise ValueError(
            f"task {task_name!r} is not in the controlled-rollout allowlist "
            f"{sorted(_SUPPORTED_TASKS)}. Controlled rollout needs a real "
            f"reference continuation with actual reasoning/output text, not "
            f"just a short final-answer label -- verify the task's raw "
            f"reference field holds enough text before adding it to "
            f"_SUPPORTED_TASKS (see this function's module-level comment "
            f"and docs/eval_pipeline/phase2_condrollouteval.md)."
        )

    from lm_eval.tasks import TaskManager, get_task_dict

    task_dict = get_task_dict([task_name], TaskManager())
    task = task_dict[task_name]
    docs = list(task.test_docs()) if task.has_test_docs() else list(task.validation_docs())
    if limit is not None:
        docs = docs[:limit]

    documents = []
    for doc in docs:
        prompt_text = task.doc_to_text(doc)
        if reference_field is not None and reference_field in doc:
            reference_text = doc[reference_field]
        else:
            reference_text = task.doc_to_target(doc)
        documents.append({"prompt_text": str(prompt_text), "reference_text": str(reference_text)})
    return documents


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Job 3 — build training's layout, via training's own `truncate_and_mask`
# ---------------------------------------------------------------------------


@dataclass
class BatchLayout:
    """A batch of documents in exactly training's MTP layout.

    Every tensor is what `truncate_and_mask` returned, unmodified. `S` is
    `truncation_length` (the destination length, default 160) and `B` is how
    many documents were packed into this batch.
    """

    input_ids: torch.Tensor        # (B, S) the student's input
    target_ids: torch.Tensor       # (B, S) next-token targets
    mask_id_mask: torch.Tensor     # (B, S) True at mask slots
    pred_pos_mask: torch.Tensor    # (B, S) True at prediction positions
    prefix_pos_mask: torch.Tensor  # (B, S) True at non-prediction positions
    source_index: torch.Tensor     # (S,) true document position of each slot
    valid: torch.Tensor            # (B * mask_region_ct, k_toks) bool
    n_real: List[int]              # real tokens each document supplied
    k_toks: int
    truncation_length: int
    mask_region_ct: int
    region_width: int
    prefix_length: int             # P
    n_mask: int                    # K
    num_toks_consumed: int         # real tokens each document contributes
    offset: int                    # region-grid offset, pretrain.py:207-233

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def tot_mask_regions(self) -> int:
        """`data_bsz * mask_region_ct`, exactly as `pretrain.py:1268`."""
        return self.batch_size * self.mask_region_ct


def layout_geometry(k_toks: int, truncation_length: int = 160, mask_region_ct: int = 5):
    """`(K, P, region_width, num_toks_consumed)` for a given `k_toks`.

    Reproduces `pretrain.py:153-155` and `:204`:
    `K = k_toks - 1`, `P = truncation_length // mask_region_ct - K`,
    `region_width = P + K`, `num_toks_consumed = mask_region_ct * P + K`.

    `num_toks_consumed` is the number of *real document* tokens one sequence
    consumes — and, minus one, the **highest RoPE position the student ever
    saw during MTP training**. At the shipped config
    (`truncation_length=160`, `mask_region_ct=5`) and `k_toks=7` that is 135.
    This is the reason the harness truncates: see `build_batch_layout`.
    """
    if k_toks < 1:
        raise ValueError(f"k_toks must be >= 1, got {k_toks}")
    n_mask = k_toks - 1  # K
    region_width = truncation_length // mask_region_ct
    prefix_length = region_width - n_mask  # P
    if prefix_length < 1:
        raise ValueError(
            f"P = truncation_length // mask_region_ct - (k_toks - 1) = "
            f"{truncation_length} // {mask_region_ct} - {n_mask} = {prefix_length}, "
            f"must be >= 1 (litgpt/model.py:122 asserts the same)"
        )
    num_toks_consumed = mask_region_ct * prefix_length + n_mask
    return n_mask, prefix_length, region_width, num_toks_consumed


def destination_source_index(
    k_toks: int,
    truncation_length: int = 160,
    mask_region_ct: int = 5,
    offset: int = 0,
) -> torch.Tensor:
    """`(S,)` — the true document position each destination slot reads from.

    This is `truncate_and_mask`'s `final_indices` (`pretrain.py:159-233`),
    recomputed here because that function does the gather internally and never
    returns it. Needed for two things: deciding which horizons fell into a
    document's padding, and asserting in tests that this really does reproduce
    training's gather — which is checked directly against the tensors
    `truncate_and_mask` produced, at every supported offset.

    `offset` reproduces training's offset roll (`pretrain.py:207-233`), which
    slides the whole `[P real][K mask]` pattern along the document. See
    `build_batch_layout`.
    """
    n_mask, prefix_length, region_width, _ = layout_geometry(
        k_toks, truncation_length, mask_region_ct
    )
    if abs(offset) >= prefix_length:
        raise ValueError(
            f"abs(offset) must be < P ({prefix_length}), got {offset} "
            f"(litgpt/mtp.py:19 and pretrain.py:157 assert the same)"
        )
    S = truncation_length
    num_blocks = (S + prefix_length - 1) // prefix_length
    block_starts = torch.arange(0, num_blocks * prefix_length, prefix_length).unsqueeze(1)
    final_indices = (block_starts + torch.arange(region_width)).flatten()[:S]

    # pretrain.py:207-233, the "special roll slide thing for offset".
    if offset < 0:
        if n_mask == 0:
            old_start = final_indices[-1] + 1
        else:
            old_start = final_indices[-n_mask]
        final_indices = final_indices.roll(offset)
        final_indices[offset:] = torch.arange(old_start, old_start + (-offset))
        final_indices = final_indices + offset
    elif offset > 0:
        final_indices = final_indices.roll(offset)
        final_indices[0:offset] = torch.arange(-offset, 0)
        final_indices = final_indices + offset
    return final_indices


def build_batch_layout(
    documents: List[List[int]],
    k_toks: int,
    mask_token_ids: List[int],
    truncation_length: int = 160,
    mask_region_ct: int = 5,
    offset: int = 0,
    pad_id: int = 0,
) -> BatchLayout:
    """Lay out a batch of documents by calling training's `truncate_and_mask`.

    Not a reimplementation — this calls
    `litgpt.pretrain.truncate_and_mask` itself, so the layout is training's
    by construction and cannot drift from it.

    **Why the truncation.** Each sequence consumes only
    `num_toks_consumed` real tokens (136 at the shipped config with
    `k_toks=7`), so during training the student only ever performed MTP at
    RoPE positions `< num_toks_consumed`. Mask tokens, the interleaved block
    mask and the remapped RoPE were never exercised beyond that. Running the
    diagnostic further out would be pure extrapolation and would risk
    measuring positional drift rather than the MTP behaviour we want. So each
    document contributes its first `num_toks_consumed` tokens and no more.

    Documents shorter than that are padded to length and the horizons whose
    target landed in padding are marked `valid=False`. Training's own
    `pad_token_id` path is deliberately **not** used: it zeroes whole regions
    out of `pred_pos_mask`, which breaks the fixed
    `(tot_mask_regions, k_toks)` reshape (training compensates at
    `pretrain.py:1348`). Tracking validity separately keeps the reshape fixed
    and gives per-horizon rather than per-region granularity.

    Args:
        documents: tokenized `question + reference answer` per document, real
            tokens only. Length `B`; `B` should be `<= micro_batch_size`.
        k_toks: horizons per region.
        mask_token_ids: from `resolve_mask_token_ids`. Must be consecutive ids
            (training's `pretrain.py:250` computes them arithmetically from
            `min_mask_id`, so a gap would silently produce wrong tokens).
        truncation_length: destination length `S`, default 160 — training's
            `hparams.singleshot.truncation_length`.
        mask_region_ct: regions per sequence, default 5 — training's
            `hparams.singleshot.mask_region_ct`.
        offset: slides the whole `[P real][K mask]` pattern along the
            document, `pretrain.py:207-233`. Must satisfy `abs(offset) < P`.

            Training rolls this every step (`pretrain.py:1274-1288`;
            `roll_offsets` and `rand_rank_roll_offsets` both default to
            `True` in `args.py:315,318`), so over training the model sees
            every alignment of the region grid against the text. **Default
            `0` here**, which makes one deterministic, reproducible pass —
            but note that at `offset=0` every prediction lands at a document
            position that is a multiple of `P` (26, 52, 78, ... at the
            shipped config). Sweep it to cover the other alignments.
        pad_id: token id used to pad short documents.

    Returns:
        A `BatchLayout`.
    """
    from litgpt.pretrain import truncate_and_mask

    if not documents:
        raise ValueError("documents is empty")
    n_mask, prefix_length, region_width, num_toks_consumed = layout_geometry(
        k_toks, truncation_length, mask_region_ct
    )
    source_index = destination_source_index(
        k_toks, truncation_length, mask_region_ct, offset
    )
    if len(mask_token_ids) < n_mask:
        raise ValueError(
            f"need at least {n_mask} mask token ids for k_toks={k_toks}, got {len(mask_token_ids)}"
        )
    span = mask_token_ids[:n_mask] if n_mask else []
    if span and span != list(range(span[0], span[0] + n_mask)):
        raise ValueError(
            f"mask token ids must be consecutive; pretrain.py:250 derives them "
            f"arithmetically from min_mask_id. Got {span}."
        )

    # Each document must cover every source position the gather touches, plus
    # one more for the final target (target_ids is input_ids shifted by one).
    # truncate_and_mask asserts og_slen >= num_toks_consumed before applying
    # the offset roll (pretrain.py:202-205), so take whichever is larger.
    needed = max(num_toks_consumed, int(source_index.max()) + 1) + 1
    n_real: List[int] = []
    rows: List[List[int]] = []
    for doc in documents:
        if not doc:
            raise ValueError("documents contains an empty document")
        row = list(doc[:needed])
        n_real.append(len(row))
        rows.append(row + [pad_id] * (needed - len(row)))

    tokens = torch.tensor(rows, dtype=torch.long)
    input_ids = tokens[:, :num_toks_consumed].contiguous()
    target_ids = tokens[:, 1 : num_toks_consumed + 1].contiguous()

    prepared_input_ids, prepared_target_ids, mask_id_mask, pred_pos_mask, prefix_pos_mask = (
        truncate_and_mask(
            input_ids=input_ids,
            target_ids=target_ids,
            k_toks=k_toks,
            mask_id=None,
            truncation_length=truncation_length,
            mask_region_ct=mask_region_ct,
            offset=offset,
            pad_token_id=None,      # see docstring — validity tracked separately
            prelude_token_ids=None,
            min_mask_id=mask_token_ids[0],
            max_mask_id=span[-1] if span else mask_token_ids[0],
            # K == 0 means no mask slots at all; truncate_and_mask still
            # evaluates its mask-id expression (pretrain.py:250) and would then
            # call .max() on an empty selection (pretrain.py:253).
            skip_max_mask_id_check=(n_mask == 0),
        )
    )

    # A prediction position's target is real position source_index[d] + 1.
    target_real_index = source_index + 1
    real_len = torch.tensor(n_real, dtype=torch.long).unsqueeze(1)   # (B, 1)
    valid_all = target_real_index.unsqueeze(0) < real_len            # (B, S)
    valid = valid_all[pred_pos_mask].view(len(documents) * mask_region_ct, k_toks)

    return BatchLayout(
        input_ids=prepared_input_ids,
        target_ids=prepared_target_ids,
        mask_id_mask=mask_id_mask,
        pred_pos_mask=pred_pos_mask,
        prefix_pos_mask=prefix_pos_mask,
        source_index=source_index,
        valid=valid,
        n_real=n_real,
        k_toks=k_toks,
        truncation_length=truncation_length,
        mask_region_ct=mask_region_ct,
        region_width=region_width,
        prefix_length=prefix_length,
        n_mask=n_mask,
        num_toks_consumed=num_toks_consumed,
        offset=offset,
    )


def batch_documents(documents: List[Dict[str, str]], micro_batch_size: int = 32):
    """Yield `documents` in chunks of `micro_batch_size`.

    `micro_batch_size` is how many `question + answer` sequences are analysed
    in parallel in one forward pass — training's
    `hparams.train.micro_batch_size`, default 32. Because every sequence is
    truncated to the same `truncation_length`, batching needs no padding or
    attention-mask handling.
    """
    if micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")
    for start in range(0, len(documents), micro_batch_size):
        yield documents[start : start + micro_batch_size]


# ---------------------------------------------------------------------------
# Strategy — `conf_adapt`, mirroring free rollout's acceptance rule
# ---------------------------------------------------------------------------


@torch.no_grad()
def top1_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 softmax probability per position.

    Same quantity as `modeling_llama.py::_top1_confidence` (lines 845-852),
    which is what free rollout's `conf_adapt` thresholds on. Takes
    `(..., vocab)` and returns `(...)`.
    """
    return torch.softmax(logits, dim=-1).max(dim=-1).values


@torch.no_grad()
def conf_adapt_acceptance(confidence: torch.Tensor, threshold: float):
    """Which horizons `conf_adapt` would have accepted, per region.

    Reproduces `modeling_llama.py:889-911` exactly. Two details of that rule
    are easy to get wrong, and both change which samples end up pooled:

    1. **It accepts a contiguous leading run, not every horizon above the
       threshold.** With `[0.95, 0.92, 0.85, 0.97]` at threshold `0.9`, free
       rollout accepts horizons 1-2 and stops — horizon 4 is discarded even
       though `0.97 > 0.9`, because generation had already halted. Filtering
       each horizon independently would pool samples free rollout never
       produced.
    2. **Horizon 1 is always accepted**, even when it is below the threshold
       (`last_pos = 0` fallback, `:906-907`). Free rollout always emits at
       least one token per step, so the horizon-1 population is never
       filtered at all.

    Args:
        confidence: `(R, k_toks)` top-1 probabilities, from `top1_confidence`.
        threshold: the `conf_adapt` threshold, i.e. `strategy[1]`.

    Returns:
        `(accepted, effective_k)` — a `(R, k_toks)` bool mask, and the
        `(R,)` per-region accepted count. `effective_k` is the same quantity
        free rollout logs as `effective_k_values`
        (`modeling_llama.py:686`), so the two are directly comparable.
    """
    if confidence.dim() != 2:
        raise ValueError(f"confidence must be (R, k_toks), got {tuple(confidence.shape)}")
    k_toks = confidence.shape[1]

    below = confidence < threshold                       # (R, k_toks)
    any_below = below.any(dim=1)                         # (R,)
    first_below = below.int().argmax(dim=1)              # (R,), 0 when none below
    last_pos = torch.where(
        any_below,
        (first_below - 1).clamp(min=0),                  # :904-907, incl. the >= 0 fallback
        torch.full_like(first_below, k_toks - 1),        # :902-903, all above -> take everything
    )
    horizons = torch.arange(k_toks, device=confidence.device)
    accepted = horizons.unsqueeze(0) <= last_pos.unsqueeze(1)
    return accepted, last_pos + 1


@torch.no_grad()
def selected_mask(outputs: "ForwardOutputs", conf_adapt_threshold: Optional[float] = None):
    """`(R, k_toks)` bool — which horizons Phase 3 should actually pool.

    Always excludes horizons whose target fell in a short document's padding
    (`outputs.valid`). When `conf_adapt_threshold` is given, additionally
    excludes horizons `conf_adapt` would not have accepted, so the pooled
    population matches what free rollout at that threshold actually
    generates. `None` (the default) pools every valid horizon — the
    equivalent of free rollout's fixed-`k` strategy.
    """
    if conf_adapt_threshold is None:
        return outputs.valid
    accepted, _ = conf_adapt_acceptance(outputs.student_confidence, conf_adapt_threshold)
    return outputs.valid & accepted.to(outputs.valid.device)


# ---------------------------------------------------------------------------
# Job 4 — block mask + the two forward passes
# ---------------------------------------------------------------------------


def flex_attention_available() -> bool:
    """Whether this machine can actually *run* the block mask.

    The interleaved mask is a `BlockMask`, which only `flex_attention`
    consumes; litgpt falls back to plain SDPA otherwise, and SDPA rejects a
    `BlockMask` outright. litgpt gates this on
    `_SUPPORTS_FLEX_ATTENTION = torch.cuda.is_available() and
    torch.cuda.get_device_capability() >= (7, 5)`
    (`litgpt/attention_utils.py:82-85`), i.e. a Turing-or-newer GPU.

    This is the same constraint training runs under — `train_with_block_mask`
    is equally unusable on CPU — so it is a property of the mechanism, not of
    this harness. Building the layout works fine on CPU; only the forward
    pass needs the GPU.
    """
    try:
        from litgpt.attention_utils import _SUPPORTS_FLEX_ATTENTION
    except ImportError:
        return False
    return bool(_SUPPORTS_FLEX_ATTENTION)


def prepare_block_mask(model, layout: BatchLayout, device=None) -> None:
    """Install this batch's block mask and RoPE remap, as `pretrain.py:1292` does.

    One call to `GPT.reconstruct_block_mask` (`litgpt/model.py:96`) installs
    both halves of the mechanism:

    - the `interleaved_mtp_mask_mod_factory` attention rule
      (`litgpt/mtp.py`), under which a *real* token attends only to real
      tokens (so a mask slot can never contaminate a later region's prefix)
      and a *mask* slot attends only to real tokens plus its own region's
      mask slots (so a later MTP never sees an earlier MTP);
    - the `P`/`K`/`Ofs` that `construct_block_rope_feats` (`model.py:361`)
      reads on every forward, giving each real token its true document
      position and each mask slot the true position of the token it predicts.

    Every argument is training's, with `S = truncation_length` fixed, so the
    geometry is identical to training's, including `layout.offset` — the same
    value `build_batch_layout` gave to `truncate_and_mask`, so the attention
    mask, the RoPE remap and the token layout always agree on where the region
    grid sits.
    """
    model.reconstruct_block_mask(
        K=layout.n_mask,
        S=layout.truncation_length,
        B=layout.batch_size,
        mask_region_ct=layout.mask_region_ct,
        offset=layout.offset,
        bidirect_ss_attn=False,
        device=device,
    )
    model.use_block_mask = True


@dataclass
class ForwardOutputs:
    """One batch's raw per-region, per-horizon outputs.

    First two dims are always `(tot_mask_regions, k_toks)`, where
    `tot_mask_regions = batch_size * mask_region_ct` (`pretrain.py:1268`).
    `valid` is False for horizons whose target fell in a short document's
    padding — drop those before any metric.

    Logits are returned as-is and are large (`vocab` ~128k vs. `d_h` ~2k, a
    ~62x difference); Phase 3 reduces them to `(confidence, correctness)`
    immediately and discards them, per `claduemetricworkprocedure.md` §11.
    Hidden states are already on CPU.
    """

    student_logits: torch.Tensor      # (R, k_toks, vocab)
    student_confidence: torch.Tensor  # (R, k_toks), student's top-1 softmax prob
    student_hidden: torch.Tensor      # (R, k_toks, d_h), CPU
    teacher_logits: torch.Tensor      # (R, k_toks, vocab)
    teacher_hidden: torch.Tensor      # (R, k_toks, d_h), CPU
    student_hard_preds: torch.Tensor  # (R, k_toks), forced on the teacher
    target_ids: torch.Tensor          # (R, k_toks), true tokens
    valid: torch.Tensor               # (R, k_toks), bool


@torch.no_grad()
def run_controlled_forward(student, teacher, layout: BatchLayout, device=None) -> ForwardOutputs:
    """Student pass, then teacher pass forced with the student's own predictions.

    Follows `pretrain.py:1400-1438` exactly:

    1. Student forward; read logits/hidden at `pred_pos_mask`, reshaped to
       `(tot_mask_regions, k_toks, ...)` — `pretrain.py:1409`.
    2. `hard_stud_preds = argmax(...)`; the first `K` of each region's
       `k_toks` predictions are written into that region's `K` mask slots
       (`pretrain.py:1429`). The slot at within-region offset `P+i` receives
       horizon `i`'s prediction, which is precisely the token that slot
       stands in for.
    3. Teacher forward on that student-forced sequence, same block mask, same
       positions. The teacher therefore never sees a mask token id, and both
       models' outputs are positionally identical.

    Requires a Turing-or-newer GPU — see `flex_attention_available`.
    """
    if not flex_attention_available():
        raise RuntimeError(
            "Controlled rollout needs flex attention to apply the interleaved MTP "
            "block mask, which litgpt enables only on a CUDA GPU of compute "
            "capability >= 7.5 (litgpt/attention_utils.py:82-85). Without it the "
            "attention call silently falls back to plain SDPA, which rejects a "
            "BlockMask. Training has the same requirement for "
            "train_with_block_mask, so this is not specific to this harness. "
            "Run this on a GPU node."
        )

    if device is None:
        device = next(student.parameters()).device

    prepare_block_mask(student, layout, device=device)
    prepare_block_mask(teacher, layout, device=device)

    input_ids = layout.input_ids.to(device)
    pred_pos_mask = layout.pred_pos_mask.to(device)
    mask_id_mask = layout.mask_id_mask.to(device)
    R, k = layout.tot_mask_regions, layout.k_toks

    # 1. student prediction pass
    student_logits_full, student_hidden_full = student(input_ids, return_hidden_states=True)
    student_logits = student_logits_full[pred_pos_mask].view(R, k, -1)
    student_hidden = student_hidden_full[pred_pos_mask].view(R, k, -1)

    # 2. student-forced teacher pass
    student_hard_preds = student_logits.argmax(dim=-1)  # (R, k_toks)
    stud_forcing_input_ids = input_ids.clone()
    if layout.n_mask > 0:
        stud_forcing_input_ids[mask_id_mask] = student_hard_preds[:, : layout.n_mask].reshape(-1)

    teacher_logits_full, teacher_hidden_full = teacher(
        stud_forcing_input_ids, return_hidden_states=True
    )
    teacher_logits = teacher_logits_full[pred_pos_mask].view(R, k, -1)
    teacher_hidden = teacher_hidden_full[pred_pos_mask].view(R, k, -1)

    target_ids = layout.target_ids.to(device)[pred_pos_mask].view(R, k)

    return ForwardOutputs(
        student_logits=student_logits.detach(),
        student_confidence=top1_confidence(student_logits).detach(),
        student_hidden=student_hidden.detach().to("cpu"),
        teacher_logits=teacher_logits.detach(),
        teacher_hidden=teacher_hidden.detach().to("cpu"),
        student_hard_preds=student_hard_preds.detach(),
        target_ids=target_ids,
        valid=layout.valid.to(device),
    )


def process_batch(
    student,
    teacher,
    tokenizer,
    documents: List[Dict[str, str]],
    k_toks: int,
    mask_token_ids: List[int],
    truncation_length: int = 160,
    mask_region_ct: int = 5,
    offset: int = 0,
    pad_id: int = 0,
    device=None,
):
    """One batch of documents, start to finish: tokenize -> lay out -> both passes.

    Each document's prompt and reference answer are tokenized into one
    continuous real sequence (BOS on the prompt only, no EOS), then truncated
    to `num_toks_consumed` tokens. Regions tile across all of it, question
    included — this harness does not score answers, and the block mask means
    mask slots inside the question cannot affect anything, so there is
    nothing to exclude.

    Returns:
        `(layout, outputs)`.
    """
    token_ids = []
    for doc in documents:
        prompt_ids = tokenizer.encode(doc["prompt_text"], bos=True).tolist()
        reference_ids = tokenizer.encode(doc["reference_text"], bos=False, eos=False).tolist()
        token_ids.append(prompt_ids + reference_ids)

    layout = build_batch_layout(
        token_ids,
        k_toks,
        mask_token_ids,
        truncation_length=truncation_length,
        mask_region_ct=mask_region_ct,
        offset=offset,
        pad_id=pad_id,
    )
    outputs = run_controlled_forward(student, teacher, layout, device=device)
    return layout, outputs


# ---------------------------------------------------------------------------
# Phase 3 — reduce, pool, write
# ---------------------------------------------------------------------------
#
# Everything above produces, per batch, a `(R, k_toks)` view of one batch of
# documents. Everything below turns the whole dataset's worth of those into
# `controlled_rollout_<timestamp>.json`.
#
# Two rules govern the reduction, and both are load-bearing:
#
# 1. `selected_mask` gates EVERYTHING. A horizon the strategy did not accept
#    contributes neither its `(confidence, correctness)` pair nor its hidden
#    vector, and both models' hidden states drop together so the two
#    populations stay paired.
# 2. Logits never accumulate. `HorizonAccumulator.add` reduces them to
#    confidences and correctness flags and keeps nothing else, per
#    `claduemetricworkprocedure.md` §11 — logits are ~62x larger than hidden
#    states (vocab ~128k vs. d_h ~2k).


@torch.no_grad()
def _correctness(preds: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """`1.0`/`0.0` float for `preds == reference`, elementwise."""
    return (preds == reference).float()


class HorizonAccumulator:
    """Per-region, per-horizon material for every metric, across all batches.

    The region structure is deliberately **preserved** rather than flattened
    to one list per horizon. The joint ECE
    (`docs/controlled_rollout_plan.md` -> "ECE — the joint form") needs
    `C_{r,j} = prod_{i<=j} c_{r,i}` within a region, which a per-horizon
    flattening throws away.

    Held per batch, concatenated on `finalize()`:

    - `confidence`, `correct_gt`, `correct_teach` — `(R, k_toks)` float32
    - `selected` — `(R, k_toks)` bool, `selected_mask`'s verdict
    - `student_hidden`, `teacher_hidden` — `(R, k_toks, d_h)` float32, CPU

    Memory is dominated by the hidden states: at the shipped config
    (1319 documents x 5 regions, `k_toks=7`, `d_h=2048`) that is
    `6595 x 7 x 2048 x 4 B x 2 models` ~= 756 MB, which is why they are
    float32 on CPU and the logits are not kept at all.
    """

    def __init__(self, k_toks: int):
        if k_toks < 1:
            raise ValueError(f"k_toks must be >= 1, got {k_toks}")
        self.k_toks = k_toks
        self._confidence: List[torch.Tensor] = []
        self._correct_gt: List[torch.Tensor] = []
        self._correct_teach: List[torch.Tensor] = []
        self._selected: List[torch.Tensor] = []
        self._student_hidden: List[torch.Tensor] = []
        self._teacher_hidden: List[torch.Tensor] = []
        self._n_documents = 0

    @torch.no_grad()
    def add(self, outputs: "ForwardOutputs", selected: torch.Tensor, n_documents: int) -> None:
        """Reduce one batch's `ForwardOutputs` and keep only what metrics need.

        `selected` is `selected_mask(outputs, threshold)`'s `(R, k_toks)`
        verdict. Rows are kept whole — masking happens at reduce time — but
        the logits are reduced here and dropped.
        """
        if selected.shape != (outputs.student_confidence.shape[0], self.k_toks):
            raise ValueError(
                f"selected must be (R, k_toks)=({outputs.student_confidence.shape[0]}, "
                f"{self.k_toks}), got {tuple(selected.shape)}"
            )

        teacher_preds = outputs.teacher_logits.argmax(dim=-1)
        self._confidence.append(outputs.student_confidence.detach().float().cpu())
        self._correct_gt.append(
            _correctness(outputs.student_hard_preds, outputs.target_ids).cpu()
        )
        self._correct_teach.append(
            _correctness(outputs.student_hard_preds, teacher_preds).cpu()
        )
        self._selected.append(selected.detach().bool().cpu())
        self._student_hidden.append(outputs.student_hidden.detach().float().cpu())
        self._teacher_hidden.append(outputs.teacher_hidden.detach().float().cpu())
        self._n_documents += n_documents

    @property
    def n_documents(self) -> int:
        return self._n_documents

    @torch.no_grad()
    def finalize(self) -> Dict[str, torch.Tensor]:
        """Concatenate every batch into one `(R_total, ...)` tensor each."""
        if not self._confidence:
            raise RuntimeError("HorizonAccumulator.add() was never called — nothing to reduce.")
        return {
            "confidence": torch.cat(self._confidence, dim=0),
            "correct_gt": torch.cat(self._correct_gt, dim=0),
            "correct_teach": torch.cat(self._correct_teach, dim=0),
            "selected": torch.cat(self._selected, dim=0),
            "student_hidden": torch.cat(self._student_hidden, dim=0),
            "teacher_hidden": torch.cat(self._teacher_hidden, dim=0),
        }


@torch.no_grad()
def marginal_pairs(pooled: Dict[str, torch.Tensor], horizon: int, against: str = "gt"):
    """Horizon `j`'s own `(confidence, correctness)` pairs — the marginal ECE input.

    Args:
        pooled: `HorizonAccumulator.finalize()`'s output.
        horizon: 1-indexed horizon `j`.
        against: `"gt"` (true token) or `"teach"` (teacher's argmax).

    Returns:
        `(confidences, correctness)`, each `(n_j,)`, over the regions where
        horizon `j` itself was selected.
    """
    idx = _check_horizon(horizon, pooled["confidence"].shape[1])
    key = {"gt": "correct_gt", "teach": "correct_teach"}[against]
    keep = pooled["selected"][:, idx]
    return pooled["confidence"][keep, idx], pooled[key][keep, idx]


@torch.no_grad()
def cumulative_pairs(pooled: Dict[str, torch.Tensor], depth: int, against: str = "gt"):
    """The joint `(C_{:,j}, A_{:,j})` pairs at depth `j`.

    ```text
    C_{r,j} = prod_{i=1..j} c_{r,i}        A_{r,j} = prod_{i=1..j} a_{r,i}
    ```

    over the regions selected *through* depth `j` — i.e. where every horizon
    `1..j` survived `selected_mask`. Because `conf_adapt` accepts a
    contiguous leading run and padding invalidates a trailing one, that set
    is exactly `effective_k >= j`; the cumulative AND is computed explicitly
    anyway, so the result does not depend on that being true.

    `C` is the block probability the model itself asserts: MTP heads predict
    positions `t+1..t+k` conditionally independently given the same prefix,
    so there is no chain rule to apply and the product of the marginals *is*
    the joint. Calibrating it therefore tests that independence assumption —
    see `docs/controlled_rollout_plan.md` -> "ECE — the joint form".

    At `depth == 1` this returns exactly `marginal_pairs(..., 1, ...)`.
    """
    idx = _check_horizon(depth, pooled["confidence"].shape[1])
    key = {"gt": "correct_gt", "teach": "correct_teach"}[against]
    through = pooled["selected"][:, : idx + 1].all(dim=1)
    confidence = pooled["confidence"][:, : idx + 1].prod(dim=1)[through]
    correctness = pooled[key][:, : idx + 1].prod(dim=1)[through]
    return confidence, correctness


def _check_horizon(horizon: int, k_toks: int) -> int:
    if not (1 <= horizon <= k_toks):
        raise ValueError(f"horizon {horizon} out of range for k_toks={k_toks}")
    return horizon - 1


@torch.no_grad()
def _subsample_indices(n: int, max_pool_samples: Optional[int], generator) -> Optional[torch.Tensor]:
    """Seeded index draw, or `None` to use the whole population.

    `max_pool_samples` defaults to `None` everywhere — the full population is
    small enough to use whole (~6,595 per horizon at the shipped config, ~20 s
    and ~5.2 GB per horizon). It exists because both latent metrics are
    quadratic in `n`, so a larger held-out set, a larger `mask_region_ct`, or
    sweeping several `offset` alignments into one pool would need it.
    """
    if max_pool_samples is None or n <= max_pool_samples:
        return None
    return torch.randperm(n, generator=generator)[:max_pool_samples]


@torch.no_grad()
def horizon_features(
    pooled: Dict[str, torch.Tensor],
    horizon: int,
    max_pool_samples: Optional[int] = None,
    generator=None,
):
    """Horizon `j`'s paired `(student, teacher)` hidden states, `(n_j, d_h)` each.

    Student and teacher are indexed by the **same** rows — and, when
    subsampling, by the same draw. They are paired observations of one region,
    not two independent populations, and drawing them separately would destroy
    the pairing the whole diagnostic rests on.
    """
    idx = _check_horizon(horizon, pooled["confidence"].shape[1])
    keep = pooled["selected"][:, idx]
    x = pooled["student_hidden"][keep, idx, :]
    y = pooled["teacher_hidden"][keep, idx, :]
    draw = _subsample_indices(x.shape[0], max_pool_samples, generator)
    if draw is not None:
        x, y = x[draw], y[draw]
    return x, y


@torch.no_grad()
def effective_k(pooled: Dict[str, torch.Tensor]) -> torch.Tensor:
    """`(R,)` — how many horizons each region actually contributed.

    This is `selected.sum(dim=1)`: the count of horizons that survived both
    the strategy and the padding check. Under `conf_adapt` on a document long
    enough that nothing was padded it is exactly
    `conf_adapt_acceptance`'s `effective_k`, the quantity free rollout logs as
    `effective_k_values` (`modeling_llama.py:686`); padding can only lower it.
    """
    return pooled["selected"].sum(dim=1)


# ---------------------------------------------------------------------------
# Phase 3 — the per-horizon records and their aggregate
# ---------------------------------------------------------------------------


@torch.no_grad()
def per_horizon_records(
    pooled: Dict[str, torch.Tensor],
    n_bins: int = 15,
    max_pool_samples: Optional[int] = None,
    seed: int = 0,
    min_ece_samples: int = 300,
    min_latent_samples: int = 200,
    thin_bin_below: int = 30,
):
    """One record per horizon, plus the MMD bandwidth they all share.

    For each horizon `j = 1..k_toks` with at least one selected sample:

    | Field | Definition |
    |---|---|
    | `n_samples` | regions that survived `selected_mask` at this horizon |
    | `ece_stud_gt` / `ece_stud_teach` | marginal — `ece_from_pairs` on horizon `j` alone |
    | `ece_stud_gt_joint` / `ece_stud_teach_joint` | cumulative — `ece_from_pairs` on `(C_{:,j}, A_{:,j})` |
    | `mean_joint_confidence` | `C_{:,j}.mean()` — where this horizon's joint population actually sits |
    | `n_samples_joint` | regions selected *through* depth `j` |
    | `ece_*` / `ece_joint_*` | `bins_populated`, `max_bin_weight`, `min_bin_count`, `thin_bin_share` — occupancy of the binning each ECE above just did |
    | `mmd` / `sinkhorn` | full-population, one shared bandwidth |

    **Bandwidth.** One `MMDBandwidthSchedule(warmup_steps=1)`, updated once
    from horizon 1's features and frozen for every horizon after — so every
    horizon is measured with the same stick and a rising MMD means drift, not
    a moving kernel. Horizon 1 is the one horizon `conf_adapt` never filters
    (`modeling_llama.py:906-907`), so the bandwidth always comes from the
    full population. This mirrors `mmd.py`'s training-time policy.

    Horizons with no selected samples are **omitted**, never emitted as `0.0`
    or `NaN`.

    **Sample floors.** A thin horizon does not produce a noisy number, it
    produces a wrong one, so each metric is omitted below its floor rather
    than emitted and caveated. `n_samples` / `n_samples_joint` are always
    written, so a reader can see *why* a horizon carries no metric.

    `min_latent_samples=200` gates `mmd` and `sinkhorn` together — they are
    computed from the same `(x, y)`, so splitting the floor would report them
    on different populations. The binding constraint is MMD: the unbiased
    U-statistic goes negative on thin draws and `sqrt(clamp(·, min=0))` floors
    it, so a missing measurement reads as a *perfect* student/teacher match.
    Simulated at `d_h=2048`, that fired on 40% of draws at n=5, 10% at n=20,
    and never from n=50 up; separation of a drift-sized effect reaches
    Cohen's d ≈ 2.9 by n=200. Sinkhorn's own sampling noise is far smaller
    (its value moves ~3% between n=5 and n=2000), so 200 is generous for it.

    `min_ece_samples=300` gates the four ECE fields. ECE is occupancy-weighted
    — a sparse bin with a wild `|acc − conf|` enters with weight `n_b/n` — so
    its small-sample bias decays much faster than a bins-per-sample argument
    suggests: simulated against a known truth of 0.130, bias was +146% at
    n=5, +45% at n=50, +9% at n=200, +7% at n=300 and +3% at n=500. 300 is
    where the bias stops dominating. Note this floor buys an unbiased
    *value*, not per-horizon significance — separating two checkpoints ~19%
    apart in ECE needs n ≈ 1500 for d ≈ 2, so a single thin horizon's ECE gap
    should not be read as evidence on its own.

    `mean_joint_confidence` is a plain mean, unaffected by binning, and is
    emitted whenever the horizon has joint samples.

    **Bin occupancy.** The floor above is a *population* floor; there is no
    per-bin one, and `ece_from_pairs` does not impose one either — a bin with a
    single region still contributes `(1/N)·|acc − conf|`, which can be most of
    the value. `thin_bin_below` (default 30) sets what counts as a thin bin for
    `ece_thin_bin_share` / `ece_joint_thin_bin_share`, the fraction of each ECE
    that came from such bins. These fields change no ECE; they say how much to
    trust one.

    Returns:
        `(records, bandwidth)` — a list of dicts, and the float bandwidth
        (or `None` if no horizon had samples).
    """
    from ..metrics.calibration import ece_bin_stats, ece_from_pairs
    from ..metrics.mmd import MMDBandwidthSchedule, maximum_mean_discrepancy
    from ..metrics.sinkhorn import sinkhorn_distance

    k_toks = pooled["confidence"].shape[1]
    generator = torch.Generator().manual_seed(seed)

    schedule = MMDBandwidthSchedule(warmup_steps=1)
    bandwidth = None
    x1, y1 = horizon_features(pooled, 1, max_pool_samples, generator)
    if x1.shape[0] > 0:
        bandwidth = schedule.update(0, x1, y1)

    records = []
    for j in range(1, k_toks + 1):
        conf_gt, corr_gt = marginal_pairs(pooled, j, "gt")
        if conf_gt.shape[0] == 0:
            continue
        conf_te, corr_te = marginal_pairs(pooled, j, "teach")
        jconf_gt, jcorr_gt = cumulative_pairs(pooled, j, "gt")
        jconf_te, jcorr_te = cumulative_pairs(pooled, j, "teach")

        record = {
            "horizon": j,
            "n_samples": int(conf_gt.shape[0]),
            "n_samples_joint": int(jconf_gt.shape[0]),
        }
        # The `ece_*bin*` fields are diagnostics on the binning the ECE above
        # just performed, not a second metric: ECE is occupancy-weighted, so a
        # bin holding a handful of regions still enters with weight |B_m|/N and
        # can carry most of the value. They are computed on the `gt`
        # populations only -- `teach` bins the same confidences, so its
        # occupancy is identical and only the accuracies differ.
        if conf_gt.shape[0] >= min_ece_samples:
            record["ece_stud_gt"] = float(ece_from_pairs(conf_gt, corr_gt, n_bins=n_bins))
            record["ece_stud_teach"] = float(ece_from_pairs(conf_te, corr_te, n_bins=n_bins))
            for key, value in ece_bin_stats(
                conf_gt, corr_gt, n_bins=n_bins, thin_below=thin_bin_below
            ).items():
                record[f"ece_{key}"] = value
        if jconf_gt.shape[0] > 0:
            record["mean_joint_confidence"] = float(jconf_gt.mean())
        if jconf_gt.shape[0] >= min_ece_samples:
            record["ece_stud_gt_joint"] = float(ece_from_pairs(jconf_gt, jcorr_gt, n_bins=n_bins))
            record["ece_stud_teach_joint"] = float(
                ece_from_pairs(jconf_te, jcorr_te, n_bins=n_bins)
            )
            for key, value in ece_bin_stats(
                jconf_gt, jcorr_gt, n_bins=n_bins, thin_below=thin_bin_below
            ).items():
                record[f"ece_joint_{key}"] = value

        # Below min_latent_samples MMD's unbiased U-statistic goes negative
        # often enough that sqrt(clamp(., min=0)) reports a missing
        # measurement as a perfect match. Skip both rather than emit it.
        x, y = horizon_features(pooled, j, max_pool_samples, generator)
        if bandwidth is not None and x.shape[0] >= min_latent_samples:
            # The population the two metrics were actually computed on, which
            # is *not* n_samples once max_pool_samples caps a horizon.
            # aggregate_horizons weights by this, so a capped run weights
            # every horizon equally rather than by a count it never used.
            record["n_latent_samples"] = int(x.shape[0])
            record["mmd"] = float(maximum_mean_discrepancy(x, y, bandwidth=bandwidth))
            record["sinkhorn"] = float(sinkhorn_distance(x, y))
        records.append(record)

    return records, (float(bandwidth) if bandwidth is not None else None)


@torch.no_grad()
def aggregate_horizons(records: List[Dict], pooled: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """The across-horizon summary.

    | Key | Definition |
    |---|---|
    | `mmd/mean`, `sinkhorn/mean` | mean over horizons that have the metric, **weighted by `n_latent_samples`** |
    | `ece_stud_gt_joint`, `ece_stud_teach_joint` | the **deepest** horizon's joint ECE — the whole-block number, restated where a sweep can find it |
    | `avg_effective_k` | mean of `effective_k` over the regions that actually ran — those with `effective_k > 0` |

    **Why `effective_k > 0` and not every region.** `pooled` carries all
    `R = batch_size * mask_region_ct` regions the layout lays down, but a
    region whose mask block fell past the document's last real token
    contributes nothing: `selected` is all-`False` there and its
    `effective_k` is `0`. Dividing by `R` therefore mixes "how many horizons
    survived" with "how much of the 160-token window the documents filled",
    and the result is not `k_toks` even under `strategy=none`, where every
    region that ran keeps all `k` horizons by construction. On GSM8K at
    `truncation_length=160` that put the fixed-k summary at 0.847 / 1.718 /
    2.618 for `k_toks` 1 / 2 / 3 — an integer statistic reported as a
    document-length artifact, and identical across checkpoints because it
    never depended on the model.

    Restricting the denominator to regions that produced at least one MTP
    evaluation makes this the exact analogue of free rollout's
    `sum(effective_k_values) / len(effective_k_values)`
    (`modeling_llama.py:789`): the numerator counts every evaluation, the
    denominator counts every unit that ran — a step there, a region here.
    Nothing is dropped from the numerator, so a region clipped mid-block
    still contributes the horizons it did reach and the shortfall below
    `k_toks` is exactly the clipping rate rather than a length artifact
    (0.9963 at `k_toks=2`, 0.9893 at 3). `selected` is prefix-closed —
    `conf_adapt` never filters horizon 1 (`modeling_llama.py:906-907`) and
    padding truncates a suffix — so `(effective_k > 0).sum()` is exactly
    horizon 1's `n_samples`, making this equal to
    `sum(n_samples) / n_samples[0]` over `per_horizon`.

    **The latent means are sample-weighted.** Each horizon enters `mmd/mean`
    and `sinkhorn/mean` with weight `n_latent_samples` — the population its
    metric was actually computed on — so a horizon measured on more vectors
    carries more of the summary, and a thin one cannot swing it.

    Know what this costs. Under `conf_adapt` horizon 1 is never filtered and
    holds ~60% of the retained mass even after the sample floors, and horizon
    1 is where the student–teacher gap is *smallest* (GSM8K nodrift: 0.098 at
    h1 against 0.193 at h5). Weighting therefore pulls the summary toward the
    shallowest horizon — 0.161 unweighted against 0.126 weighted on that run
    — and damps the depth-wise growth the diagnostic exists to expose. The
    per-horizon records are where that growth is read; this scalar is for
    sweeps, and it ranks checkpoints the same way either way (nodrift >
    λ=0.0106 > λ=0.05, separation 32.3% unweighted, 34.7% weighted).

    Weighting by `n_latent_samples` rather than `n_samples` matters when
    `max_pool_samples` caps the population: every horizon is then measured on
    the same number of vectors and the weights are equal, which is correct —
    `n_samples` would weight by a count the metric never saw.

    ECE gets no mean over horizons and no cross-horizon pooling either: the
    `C_{:,j}` are nested transformations of the same regions, not independent
    populations, so averaging or concatenating them would double-count
    horizon 1 and under-count the deep block. The cumulative sequence *is*
    the summary; its last element is the headline.
    """
    out: Dict[str, float] = {}
    for key in ("mmd", "sinkhorn"):
        pairs = [(r[key], r["n_latent_samples"]) for r in records if key in r]
        total = sum(n for _, n in pairs)
        if total:
            out[f"{key}/mean"] = float(sum(v * n for v, n in pairs) / total)

    for key in ("ece_stud_gt_joint", "ece_stud_teach_joint"):
        deepest = [r for r in records if key in r]
        if deepest:
            out[key] = float(deepest[-1][key])
            out[f"{key}_depth"] = int(deepest[-1]["horizon"])

    # Regions that never ran carry effective_k == 0; they are padding, not a
    # model measurement. Omit the key rather than emit NaN if none ran at all
    # — _assert_finite refuses non-finite floats, and None is the only form
    # that survives to the notebook (see its docstring).
    ran = effective_k(pooled)
    ran = ran[ran > 0]
    if ran.numel():
        out["avg_effective_k"] = float(ran.float().mean())
    return out


# ---------------------------------------------------------------------------
# Phase 3 — write the file hop 3 reads
# ---------------------------------------------------------------------------


def _assert_finite(payload, path: str = "") -> None:
    """Raise on any non-finite float anywhere in the payload.

    A `NaN` or `Inf` reaching wandb is not visibly broken at hop 3 — it
    breaks two hops later, where `pull_wandb_eval_data.py` writes the summary
    dict with `str()` and the notebook's `ast.literal_eval` chokes on the bare
    token `nan`, silently discarding **the entire run's row**, every column of
    it (`docs/eval_pipeline/04_pull_wandb_eval_data.md` §3). `None` is the
    only safe form, so a metric that cannot be computed is omitted rather than
    emitted as `NaN`. Failing loudly here is the cheapest place to catch it.
    """
    import math

    if isinstance(payload, dict):
        for key, value in payload.items():
            _assert_finite(value, f"{path}.{key}" if path else str(key))
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            _assert_finite(value, f"{path}[{i}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError(
            f"non-finite value {payload!r} at {path!r} — refusing to write it. "
            f"A NaN here destroys the entire run's row at hop 5; omit the key "
            f"or emit null instead (see docs/eval_pipeline/04_pull_wandb_eval_data.md §3)."
        )


def build_results_payload(
    records: List[Dict],
    aggregate: Dict[str, float],
    *,
    task: str,
    k_toks: int,
    truncation_length: int,
    mask_region_ct: int,
    offset: int,
    strategy,
    n_documents: int,
    max_pool_samples: Optional[int],
    seed: int,
    mmd_bandwidth: Optional[float],
    student_checkpoint: str,
    teacher_checkpoint: str,
) -> Dict:
    """Assemble the JSON payload, and refuse to build a non-finite one.

    `per_horizon` is canonical; `aggregate` is derived from it. The naming of
    the *wandb* keys is deliberately not decided here — hop 3
    (`push_lmeval_metrics_to_wandb.py`) flattens this payload, which is where
    every other metric in the pipeline is named.
    """
    # Canonical `None` or `["conf_adapt", <threshold>]`. `list(strategy)` used
    # to sit in the dict below, which splits the *string* forms the sweep
    # actually passes into characters -- "conf_adapt+0.6" was recorded as
    # ['c','o','n','f','_','a','d','a','p','t','+','0','.','6'] -- and the
    # pusher copies this field into wandb config, where runs are grouped by
    # strategy.
    strategy_name, strategy_threshold = parse_strategy(strategy)

    payload = {
        "task": task,
        "k_toks": k_toks,
        "truncation_length": truncation_length,
        "mask_region_ct": mask_region_ct,
        "offset": offset,
        "strategy": None if strategy_name is None else [strategy_name, strategy_threshold],
        "n_documents": n_documents,
        "max_pool_samples": max_pool_samples,
        "seed": seed,
        "mmd_bandwidth": mmd_bandwidth,
        "student_checkpoint": str(student_checkpoint),
        "teacher_checkpoint": str(teacher_checkpoint),
        "per_horizon": records,
        "aggregate": aggregate,
    }
    _assert_finite(payload)
    return payload


def write_results(payload: Dict, out_dir: Union[str, Path], timestamp: Optional[str] = None) -> Path:
    """Write `controlled_rollout_<ISO timestamp>.json` into `out_dir`.

    The filename convention is `lm_eval`'s own: hop 3 finds this file with the
    same `rglob` + `sorted()[-1]` it already uses for `results_*.json`, and
    lexicographic order on an ISO timestamp is chronological order, so the
    newest wins.

    `out_dir` should be the **same** directory `lm_eval` wrote its
    `results_*.json` into — that shared directory is what lets one
    `push_lmeval_metrics_to_wandb.py` invocation log both rollouts into one
    wandb run.
    """
    import json
    from datetime import datetime, timezone

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    path = out_dir / f"controlled_rollout_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def parse_strategy(strategy):
    """`strategy` -> `(name, threshold)`, accepting every shape the sweep uses.

    Free rollout's `strategy` arrives as `None`, as the string `"none"` (the
    sweep's sentinel, `eval_reproduce_sweep.sh`), as `"conf_adapt+0.9"` (the
    `+`-joined CLI form), or already as `["conf_adapt", 0.9]`. Anything else
    raises rather than being silently treated as fixed-`k`, because quietly
    pooling every horizon when a threshold was intended would change which
    population the numbers describe without changing their names.
    """
    if strategy is None:
        return None, None
    if isinstance(strategy, str):
        if strategy.lower() in ("none", "null", ""):
            return None, None
        parts = strategy.split("+")
    elif isinstance(strategy, (list, tuple)):
        parts = list(strategy)
    else:
        raise ValueError(f"unrecognised strategy {strategy!r}")

    if len(parts) != 2 or str(parts[0]) != "conf_adapt":
        raise ValueError(
            f"unrecognised strategy {strategy!r}: expected None, 'none', "
            f"'conf_adapt+<threshold>' or ['conf_adapt', <threshold>]"
        )
    return "conf_adapt", float(parts[1])


# ---------------------------------------------------------------------------
# Phase 3 — the driver
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_controlled_rollout(
    student,
    teacher,
    tokenizer,
    documents: List[Dict[str, str]],
    *,
    task: str,
    k_toks: int,
    mask_token_ids: List[int],
    strategy=None,
    truncation_length: int = 160,
    mask_region_ct: int = 5,
    micro_batch_size: int = 32,
    offset: int = 0,
    pad_id: int = 0,
    n_bins: int = 15,
    min_ece_samples: int = 300,
    min_latent_samples: int = 200,
    max_pool_samples: Optional[int] = None,
    seed: int = 0,
    device=None,
    student_checkpoint: str = "",
    teacher_checkpoint: str = "",
    progress: bool = False,
) -> Dict:
    """The whole diagnostic: documents in, results payload out.

    ```text
    batch_documents -> process_batch -> selected_mask -> HorizonAccumulator.add
                                                                  |
                                                    per_horizon_records + aggregate
                                                                  |
                                                        the results payload
    ```

    Nothing is written to disk here — `write_results` does that — so this is
    callable from a test without a filesystem.

    Determinism: there is no sampling anywhere on the default path
    (`max_pool_samples=None`), so two runs over the same documents with the
    same checkpoints produce byte-identical output.
    """
    _, threshold = parse_strategy(strategy)
    accumulator = HorizonAccumulator(k_toks)

    for batch_index, batch in enumerate(batch_documents(documents, micro_batch_size)):
        _, outputs = process_batch(
            student,
            teacher,
            tokenizer,
            batch,
            k_toks,
            mask_token_ids,
            truncation_length=truncation_length,
            mask_region_ct=mask_region_ct,
            offset=offset,
            pad_id=pad_id,
            device=device,
        )
        accumulator.add(outputs, selected_mask(outputs, threshold), n_documents=len(batch))
        # Logits die with `outputs` at the end of this iteration; only the
        # reductions and the hidden states survive into the accumulator.
        del outputs
        if progress:
            print(f"  batch {batch_index + 1}: {accumulator.n_documents} documents", flush=True)

    pooled = accumulator.finalize()
    records, bandwidth = per_horizon_records(
        pooled,
        n_bins=n_bins,
        max_pool_samples=max_pool_samples,
        seed=seed,
        min_ece_samples=min_ece_samples,
        min_latent_samples=min_latent_samples,
    )
    aggregate = aggregate_horizons(records, pooled)

    return build_results_payload(
        records,
        aggregate,
        task=task,
        k_toks=k_toks,
        truncation_length=truncation_length,
        mask_region_ct=mask_region_ct,
        offset=offset,
        strategy=strategy,
        n_documents=accumulator.n_documents,
        max_pool_samples=max_pool_samples,
        seed=seed,
        mmd_bandwidth=bandwidth,
        student_checkpoint=student_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
    )


# ---------------------------------------------------------------------------
# Phase 4 — config resolution and CLI
# ---------------------------------------------------------------------------
#
# One YAML: the controlled-rollout block lives under `metadata.controlled_rollout`
# in the same `default_mtp.yaml` that `lm_eval run --config` reads.
#
# It has to be under `metadata` and not at the top level. `EvaluatorConfig` is a
# `@dataclass(slots=True)` built by `cls(**config)` (lm_eval
# `config/evaluate_config.py:219`), so ANY unrecognised top-level key raises
# `TypeError` and lm_eval never starts:
#
#     EvaluatorConfig(controlled_rollout={...})               -> TypeError
#     EvaluatorConfig(metadata={"controlled_rollout": {...}}) -> accepted
#
# `metadata` is a free-form dict field (`:190-193`) handed to tasks; a task that
# doesn't look for `controlled_rollout` ignores it. This module reads the file
# with plain `yaml.safe_load` and never constructs an `EvaluatorConfig` at all.


CONTROLLED_ROLLOUT_YAML_KEY = "controlled_rollout"

_CONFIG_DEFAULTS: Dict[str, object] = {
    "enabled": False,
    "student_checkpoint": None,
    "teacher_checkpoint": None,
    "task": None,
    "out_dir": None,
    "k_toks": None,
    "strategy": None,
    "truncation_length": 160,
    "mask_region_ct": 5,
    "micro_batch_size": 32,
    "offset": 0,
    "pad_id": 0,
    "n_bins": 15,
    "min_ece_samples": 300,
    "min_latent_samples": 200,
    "max_pool_samples": None,
    "seed": 0,
    "limit": None,
    "precision": None,
    "reference_field": "answer",
    "mask_token_pattern": "<|mtp_special_token_{i}|>",
}


def load_yaml_config(config_path: Union[str, Path]) -> Dict:
    """Read `default_mtp.yaml` and pull out what this harness needs.

    Returns a dict of controlled-rollout settings. `k_toks` and `strategy` are
    taken from **`gen_kwargs`**, not from the controlled block, so the two
    halves of a run cannot describe different models — there is only one place
    they are written.
    """
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: YAML root must be a mapping, got {type(raw).__name__}")

    metadata = raw.get("metadata") or {}
    block = dict(metadata.get(CONTROLLED_ROLLOUT_YAML_KEY) or {})

    gen_kwargs = raw.get("gen_kwargs") or {}
    if "k_toks" in gen_kwargs:
        block.setdefault("k_toks", gen_kwargs["k_toks"])
    if "strategy" in gen_kwargs:
        block.setdefault("strategy", gen_kwargs["strategy"])

    tasks = raw.get("tasks")
    if tasks and "task" not in block:
        block["task"] = tasks[0] if isinstance(tasks, list) else tasks
    if raw.get("output_path") and "out_dir" not in block:
        block["out_dir"] = raw["output_path"]

    unknown = set(block) - set(_CONFIG_DEFAULTS)
    if unknown:
        raise ValueError(
            f"{config_path}: unknown key(s) under metadata.{CONTROLLED_ROLLOUT_YAML_KEY}: "
            f"{sorted(unknown)}. Known keys: {sorted(_CONFIG_DEFAULTS)}"
        )
    return block


def resolve_config(yaml_block: Optional[Dict] = None, cli_overrides: Optional[Dict] = None) -> Dict:
    """Merge defaults < YAML < CLI, and validate the result.

    CLI wins because the sweep's `--gen_kwargs` override **replaces**
    `default_mtp.yaml`'s `gen_kwargs` mapping wholesale rather than merging into
    it (`EvaluatorConfig.from_cli` does a flat `dict.update`), so the YAML's
    `k_toks` is frequently *not* what the run actually used. The launcher
    therefore passes the same `$K_TOKS` / `$STRATEGY` shell variables to both
    `lm_eval` and this module.
    """
    config = dict(_CONFIG_DEFAULTS)
    config.update(yaml_block or {})
    config.update({k: v for k, v in (cli_overrides or {}).items() if v is not None})

    if not config["enabled"]:
        return config

    missing = [
        key
        for key in ("student_checkpoint", "teacher_checkpoint", "task", "out_dir", "k_toks")
        if config.get(key) in (None, "")
    ]
    if missing:
        raise ValueError(
            f"controlled rollout is enabled but {missing} not set. Supply them under "
            f"metadata.{CONTROLLED_ROLLOUT_YAML_KEY} in the YAML or as CLI flags."
        )

    config["k_toks"] = int(config["k_toks"])
    parse_strategy(config["strategy"])  # raise now, not after both models are loaded
    layout_geometry(config["k_toks"], config["truncation_length"], config["mask_region_ct"])
    return config


def main(argv: Optional[List[str]] = None) -> int:
    """`python -m driftmtp.eval.condrollouteval --config default_mtp.yaml ...`

    Exits 0 without doing anything when the controlled rollout is not enabled —
    either `enabled: false` or no `metadata.controlled_rollout` block at all —
    so the launcher needs no conditional around the call.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m driftmtp.eval.condrollouteval",
        description=(
            "Controlled-rollout diagnostic. Runs the student and a frozen teacher over "
            "held-out documents in training's MTP layout and writes "
            "controlled_rollout_<timestamp>.json next to lm_eval's own output."
        ),
    )
    parser.add_argument("--config", default=None,
                        help="default_mtp.yaml; settings are read from metadata.controlled_rollout")
    enable = parser.add_mutually_exclusive_group()
    enable.add_argument("--enabled", dest="enabled", action="store_true", default=None,
                        help="force on, overriding the YAML")
    enable.add_argument("--disabled", dest="enabled", action="store_false",
                        help="force off; exits 0 immediately")
    parser.add_argument("--student-checkpoint", "--student_checkpoint", dest="student_checkpoint",
                        help="litgpt checkpoint dir (lit_model.pth + model_config.yaml) -- NOT the HF dir")
    parser.add_argument("--teacher-checkpoint", "--teacher_checkpoint", dest="teacher_checkpoint",
                        help="frozen teacher, same format as --student-checkpoint")
    parser.add_argument("--task", help=f"lm_eval task name; must be in {sorted(_SUPPORTED_TASKS)}")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir",
                        help="the SAME directory lm_eval wrote results_*.json into")
    parser.add_argument("--k-toks", "--k_toks", dest="k_toks", type=int,
                        help="must equal free rollout's gen_kwargs.k_toks")
    parser.add_argument("--strategy",
                        help="'none', 'conf_adapt+0.9'; must equal free rollout's gen_kwargs.strategy")
    parser.add_argument("--truncation-length", "--truncation_length", dest="truncation_length", type=int)
    parser.add_argument("--mask-region-ct", "--mask_region_ct", dest="mask_region_ct", type=int)
    parser.add_argument("--micro-batch-size", "--micro_batch_size", dest="micro_batch_size", type=int)
    parser.add_argument("--offset", type=int, help="region-grid alignment; abs(offset) < P")
    parser.add_argument("--n-bins", "--n_bins", dest="n_bins", type=int, help="ECE bins (default 15)")
    parser.add_argument("--min-ece-samples", "--min_ece_samples", dest="min_ece_samples", type=int,
                        help="omit a horizon's ECE fields below this n (default 300)")
    parser.add_argument("--min-latent-samples", "--min_latent_samples", dest="min_latent_samples", type=int,
                        help="omit a horizon's MMD and Sinkhorn below this n (default 200)")
    parser.add_argument("--max-pool-samples", "--max_pool_samples", dest="max_pool_samples", type=int,
                        help="cap the per-horizon population for MMD/Sinkhorn; default is no cap")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, help="evaluate only the first N documents (debugging)")
    parser.add_argument("--precision", help="Fabric precision string, e.g. bf16-true")
    parser.add_argument("--reference-field", "--reference_field", dest="reference_field")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                        help="resolve and print the config, then exit without loading any model")
    args = parser.parse_args(argv)

    cli = {k: v for k, v in vars(args).items() if k not in ("config", "dry_run")}
    yaml_block = load_yaml_config(args.config) if args.config else {}
    config = resolve_config(yaml_block, cli)

    if not config["enabled"]:
        print("controlled rollout not enabled — nothing to do.")
        return 0

    if args.dry_run:
        print("resolved controlled-rollout config:")
        for key in sorted(config):
            print(f"  {key}: {config[key]!r}")
        return 0

    print(f"controlled rollout: task={config['task']} k_toks={config['k_toks']} "
          f"strategy={config['strategy']!r}", flush=True)

    student = load_litgpt_model(config["student_checkpoint"], precision=config["precision"])
    teacher = load_litgpt_model(config["teacher_checkpoint"], precision=config["precision"])
    tokenizer = load_litgpt_tokenizer(config["student_checkpoint"])
    mask_token_ids = resolve_mask_token_ids(
        tokenizer,
        pattern=config["mask_token_pattern"],
        num_tokens=max(config["k_toks"] - 1, 1),
    )
    documents = load_task_documents(
        config["task"], reference_field=config["reference_field"], limit=config["limit"]
    )
    print(f"loaded {len(documents)} documents", flush=True)

    payload = run_controlled_rollout(
        student,
        teacher,
        tokenizer,
        documents,
        task=config["task"],
        k_toks=config["k_toks"],
        mask_token_ids=mask_token_ids,
        strategy=config["strategy"],
        truncation_length=config["truncation_length"],
        mask_region_ct=config["mask_region_ct"],
        micro_batch_size=config["micro_batch_size"],
        offset=config["offset"],
        pad_id=config["pad_id"],
        n_bins=config["n_bins"],
        min_ece_samples=config["min_ece_samples"],
        min_latent_samples=config["min_latent_samples"],
        max_pool_samples=config["max_pool_samples"],
        seed=config["seed"],
        student_checkpoint=config["student_checkpoint"],
        teacher_checkpoint=config["teacher_checkpoint"],
        progress=True,
    )
    path = write_results(payload, config["out_dir"])
    print(f"wrote {path}")
    for record in payload["per_horizon"]:
        print(f"  h{record['horizon']}: n={record['n_samples']} "
              f"ece_gt={record['ece_stud_gt']:.4f} "
              f"mmd={record.get('mmd', float('nan')):.4f} "
              f"sinkhorn={record.get('sinkhorn', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
