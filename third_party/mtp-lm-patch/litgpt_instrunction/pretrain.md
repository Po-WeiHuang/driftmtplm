# `litgpt/pretrain.py` — Report

This file is the training entry point for **singleshot multi-token-prediction (MTP)
self-distillation pretraining**: a student `GPT` and a (weight-identical-at-init, frozen)
teacher `GPT` are trained together, where the student predicts a block of `k_toks` future
tokens at once and is supervised by the teacher's own predictions conditioned on the
student's guesses ("student-forced teacher feedback"), rather than by ground truth alone.
It builds directly on the block-mask/MTP machinery documented in
[model.md](model.md) (`GPT.reconstruct_block_mask`, `use_block_mask`, `CausalSelfAttention`'s
FlexAttention path) — this file is where that machinery is actually driven: it samples
`k_toks`/offsets each step, calls `model.reconstruct_block_mask(...)`, prepares batches
around it, and computes the distillation loss. At 2750 lines it is the largest file in the
codebase; this report covers it top-to-bottom, function by function, in source order.

---

## 1. Variable & argument glossary

As with [model.md](model.md), read this first — most names below recur across many
functions without being redefined locally.

### Core MTP quantities (shared with `model.py`, see model.md §1 for the attention-side view)

| Name | Meaning |
|---|---|
| `k_toks` | number of future tokens the student predicts in one shot for the *current* training step; **can vary step to step** under a curriculum (see `ConstantKTokCurriculum`/`PiecewiseKTokCurriculum` below) — this is the training-time analogue of `model.py`'s `K = k_toks - 1` |
| `K` | `k_toks - 1`, the number of dedicated MASK-token positions per region (matches `model.py`'s `K`) |
| `P` | prefix length per region, `= (S // mask_region_ct) - K`; recomputed whenever `k_toks` changes |
| `S` / `truncation_length` | total sequence length per training example after truncation (`hparams.singleshot.truncation_length`) |
| `mask_region_ct` | number of prefix+mask regions tiled across `S` (`hparams.singleshot.mask_region_ct`) |
| `rolling_offset` | per-step signed shift applied to region boundaries (see `hparams.singleshot.roll_offsets` below); passed as `offset` into both `truncate_and_mask` and `model.reconstruct_block_mask` so the data layout and the attention mask stay in sync |
| `tot_mask_regions` | `data_bsz * mask_region_ct` — total number of predicted regions in the current micro-batch, used to reshape flattened logits back into `(region, k_toks, vocab)` |
| `mask_id_mask` | boolean tensor (shape `(B, S)`) marking which input positions were overwritten with MASK-token ids (the last `K` positions of each region) |
| `pred_pos_mask` | boolean tensor marking which positions the loss is actually computed over — the last `K+1` positions of each region (one more than `mask_id_mask` because the position *just before* the first mask token also predicts into the masked span) |
| `prefix_pos_mask` | boolean tensor marking the ordinary "prefix" positions (`~pred_pos_mask`, modulo edge-case cleanup) — used for the optional auxiliary ground-truth prefix loss |

### `state` dict (created in `main`, threaded through `fit`)

| Key | Meaning |
|---|---|
| `state["model"]` | the student `GPT`, wrapped by `fabric.setup` (and optionally `torch.compile`) |
| `state["model_teacher"]` | the teacher `GPT` — same architecture, frozen gradients, periodically popped out of `state` before `fabric.save`/`fabric.load` calls since it's treated as a derived/side artifact rather than part of the resumable training state in some checkpoint paths (see §4, §5) |
| `state["optimizer"]` | optimizer for the student only — the teacher never receives gradients |
| `state["train_dataloader"]` | the (stateful) training dataloader, checkpointed for exact resume unless `train.ignore_dataloader_state_on_resume` |
| `state["iter_num"]` | micro-batch counter (increments every micro-batch, i.e. every gradient-accumulation sub-step) |
| `state["step_count"]` | optimizer-step counter (increments only when gradients are actually applied, i.e. once per `gradient_accumulation_iters` micro-batches) — this is what curricula, save intervals, and eval intervals are keyed on |

### Key `hparams.train` fields (`TrainArgs`, `litgpt/args.py`)

| Field | Meaning |
|---|---|
| `micro_batch_size` | per-device batch size per forward/backward |
| `global_batch_size` | target effective batch size; combined with `micro_batch_size`/world size to derive gradient accumulation steps (via `train.gradient_accumulation_iters(...)`, not defined in this file) |
| `max_tokens` | total token budget for the run — drives `max_iters` (see `fit`) instead of an epoch count |
| `max_seq_length` | the model's context length, applied to both `model` and `model_teacher` |
| `train_block_size` | how many raw input tokens the dataloader must supply per row; for block-mask training this is **derived automatically** in `main` from `P`/`K`/`mask_region_ct` (not user-set) so the dataloader only loads the minimum tokens actually consumed |
| `do_compile`, `compile_mode`, `dynamo_cache_size_limit` | `torch.compile` controls for both the model and the small `compile_utilities()` metric functions |
| `fabric_strategy`, `fsdp_*` | distributed strategy selection (DDP vs FSDP, hybrid-sharding device mesh, activation checkpointing) |
| `peak_lr`, `min_lr`, `lr_schedule`, `lr_warmup_steps`/`lr_warmup_fraction` | LR schedule inputs consumed by `get_lr` |
| `max_norm` | gradient-clipping threshold (required — `validate_args` errors if unset) |
| `save_interval`, `save_latest_interval`, `save_latest_ckpt`, `max_ckpts_to_keep`, `initial_save`, `final_save` | checkpointing cadence and retention |
| `tie_embeddings` | if true, ties `wte.weight` to `lm_head.weight` on both student and teacher |
| `ignore_dataloader_state_on_resume`, `ignore_extra_keys_in_init_ckpt` | resume-behavior overrides |

### Key `hparams.eval` fields (`EvalArgs`)

| Field | Meaning |
|---|---|
| `interval` | how many optimizer steps between validation runs |
| `max_iters` | how many validation batches to run per validation call |
| `initial_validation`, `final_validation` | whether to validate before the first step / after the last |

### Key `hparams.singleshot` fields (`SingleShotArgs`) — the MTP-specific configuration

| Field | Meaning |
|---|---|
| `truncation_length` | `S`, see above |
| `mask_region_ct` | see above |
| `k_toks` | curriculum spec for `k_toks` — an int, a `"start:value"`-style piecewise string (parsed by `parse_k_toks`), or already a curriculum object after `setup()` normalizes it into `ConstantKTokCurriculum`/`PiecewiseKTokCurriculum` |
| `lockstep_rand_k_toks` / `rand_rank_k_toks` | two mutually-exclusive alternatives to a fixed/scheduled `k_toks`: instead sample `k_toks` uniformly at random each step between `k_toks_min`/`k_toks_max` curricula — `lockstep` uses the same RNG seed on every rank (so all ranks pick the same `k_toks` that step), `rand_rank` seeds per-rank (via `state["step_count"] + fabric.global_rank * 12345678`) so ranks can diverge |
| `k_toks_min`, `k_toks_max` | curricula bounding the randomized-`k_toks` samplers above |
| `mask_id`, `min_mask_id`, `max_mask_id` | special-token id(s) used to overwrite masked positions in the input; either a single `mask_id` or a contiguous `[min_mask_id, max_mask_id]` range (position-wise-unique mask ids) |
| `mtp_special_token_pattern`, `num_mtp_special_tokens`, `pos_wise_unique_mtp_tok_ids` | alternate way to specify `mask_id`/`temp_sep_token_id_range` by formatting a token-string pattern (e.g. `"<|mtp_special_token_{i}|>"`) and resolving ids via the tokenizer, done once in `setup()` |
| `temp_sep_token_id_range` | a range of otherwise-unused special-token ids reserved for inserting a *visual* separator between rollout segments in logged generation text (see `generative_validate`) — not used in training itself |
| `bidirect_ss_attn` | whether MASK-token positions within a region attend to each other bidirectionally; passed to the **student's** `reconstruct_block_mask` only — the teacher always gets `bidirect_ss_attn=False` explicitly, everywhere this is called |
| `train_with_block_mask` | master switch: if false, training falls back to ordinary causal attention/masking (`mask_region_ct` must then be `1`, enforced in `validate_args`) |
| `roll_offsets`, `rand_rank_roll_offsets`, `lockstep_rand_roll_offsets` | analogous per-step-randomized-offset controls for `rolling_offset`, mirroring the `k_toks` randomization knobs; default (no randomization flags) is a deterministic sweep `rolling_offset = -(step_count % P)` |
| `beta` | weight on the student's own self-entropy term in `pt_ce_plus_ent_loss` |
| `hard_teacher_supervision`, `gt_teacher_supervision`, `hard_self_teacher_supervision` | three **mutually exclusive** (asserted, sum ≤ 1) supervision modes selecting what labels the loss is computed against — see §2's discussion of `pt_ce_plus_ent_loss` and the loss-selection block in `fit` |
| `sample_during_train` | if true, the student's predicted tokens fed to the teacher are sampled (multinomial) rather than taken via `argmax` |
| `supervise_prefix` / `supervise_prefix_only` | add (or replace the whole loss with) an auxiliary ground-truth next-token loss over the ordinary prefix positions |
| `last_region_loss_only` | when `mask_region_ct > 1`, restrict loss/metrics to only the last region per sequence (incompatible with `pqds.omit_tail_padding_in_loss`, asserted) |
| `rollout_multiplier` | at validation time, how many sequential single-shot rollout steps to chain (each producing `k_toks` more tokens, re-fed as the new mask-token block) — `1` means a single MTP prediction, `>1` chains several |
| `extra_train_metrics` | gates a large block of expensive additional per-position accuracy/confidence/loss metrics during training (see `fit`) |
| `topk_values`, `num_samples` | which top-k accuracy/confidence values to track, and how many stochastic SS (single-shot) samples to draw during generative validation |
| `extra_val_trunc_lengths`, `extra_val_k_toks_values` | additional truncation-length/`k_toks` values to also run generative validation at, beyond the training defaults |
| `multi_region_val_correction` | adjusts the validation truncation length to account for `mask_region_ct > 1` so validation sees a comparable prefix budget to training |
| `log_masks_and_inputs` | debug flag: dump a visualization of the block mask (via `attn_gym.visualize_attention_scores`) and a sample of prepared input/target/mask tensors whenever the mask configuration changes (detected via a hash of `model.block_mask_config`) |

### `hparams.pqds` fields (`PQDSArgs`) — only relevant when `data == "pqds"` (the Parquet streaming dataset)

| Field | Meaning |
|---|---|
| `pad_token_id` | padding/EOS-equivalent id; resolved from the tokenizer if unset |
| `omit_tail_padding_in_loss` | if true, positions after the first pad/EOS token in a sequence are excluded from `pred_pos_mask` (handled inside `truncate_and_mask`) |
| `prelude_token_ids` | an optional fixed subsequence (e.g. a chat-template preamble) — positions at or before its first occurrence are excluded from the loss, also handled inside `truncate_and_mask` |
| `estimate_token_counts`, `estimate_file_count` | whether/how to estimate total dataset token counts up front for logging |

### Loss-function terms (`pt_ce_plus_ent_loss`, see §2)

| Name | Meaning |
|---|---|
| `ce_teach_stud` | cross-entropy of the student's predictions against the teacher's distribution (or hard/GT labels) |
| `ent_teach` | entropy of the teacher's own predictive distribution (0 if teacher labels are already one-hot/hard) |
| `kl_teach_stud` | `ce_teach_stud - ent_teach`, i.e. actual KL divergence student‖teacher (reduces to `ce_teach_stud` when teacher labels are hard, since entropy of a one-hot distribution is 0) |
| `ent_stud` | entropy of the student's own predictive distribution, only computed if `beta > 0` |
| final `loss` | `kl_teach_stud + beta * ent_stud` (an entropy-regularized distillation loss) |

---

## 2. Top-level loss and metric helper functions (lines 72–439)

### `pt_ce_plus_ent_loss` (72–100)

The core self-distillation loss. Despite the name, it's implemented as a KL-divergence
isolated via cross-entropy identities rather than literally calling a softmax-KL function —
the inline comment explains the intent is to avoid extraneous computation. Two branches:
if `labels_teacher` is provided (hard/one-hot teacher labels — either genuinely hard teacher
predictions or ground-truth ids), the teacher's entropy is treated as exactly `0` and
`kl_teach_stud` reduces to plain cross-entropy. Otherwise, the teacher's own softmax
distribution is used as soft labels, and `kl_teach_stud = ce_teach_stud - ent_teach` is
computed properly. The student's own self-entropy `ent_stud` is only computed when
`beta > 0`, to skip the extra softmax/cross-entropy when it isn't needed. Returns
`(loss, ce_teach_stud, kl_teach_stud, ent_teach, ent_stud)` — all five values are logged as
separate metrics in the training loop.

### `batch_find_subarray` (103–128)

A batched subsequence-search utility (with a worked example commented below it, 130–137)
used to locate the first occurrence of `prelude_token_ids` in each row of a batch, via
`unfold`-based sliding-window comparison rather than a Python loop. Returns `-1` for rows
with no match.

### `truncate_and_mask` (140–335, "Begin Gemini :] version for multi region striding... and a
rewrite, and a rewrite...")

The single most important data-preparation function in the file, and the training-time twin
of `GPT.reconstruct_block_mask`/`construct_block_rope_feats` in `model.py` — it must stay
structurally consistent with those, since the whole scheme depends on the data layout and
the attention mask agreeing on where prefix/mask regions fall. Given raw `input_ids`/
`target_ids` of length `og_slen`, a `k_toks` value, and the same `truncation_length`/
`mask_region_ct`/`offset` parameters used to build the block mask, it:

1. Derives `K = k_toks - 1`, `P`, and `region_width = P + K` exactly as `model.py` does, and
   asserts `abs(offset) < P`.
2. Builds a flat index array (`final_indices`) tiling `region_width`-sized regions across the
   target length `S`, identical in spirit to `construct_block_rope_feats` in `model.py`, and
   two boolean masks over that same flattened region layout: `mask_id_mask` (`True` over the
   last `K` positions of each region — where MASK tokens will be written into the input) and
   `pred_pos_mask` (`True` over the last `K+1` positions — one wider, since the loss is
   computed starting from the position immediately preceding the first mask token, which
   predicts the first masked target).
3. Asserts the resulting indices don't exceed the original sequence (`final_indices.max()
   <= og_slen - 1`) and that the total consumed raw-token count exactly matches
   `mask_region_ct * P + K` — the same formula `main()` uses to set `train.train_block_size`,
   so this assertion is effectively a cross-check that the dataloader was configured
   correctly upstream.
4. Applies the same signed-`offset` "roll and patch" adjustment as
   `construct_block_rope_feats` (208–235), including the identical `K == 0` edge case for
   negative offsets, but here it additionally rolls `mask_id_mask` and `pred_pos_mask` in
   lockstep and zeroes out the wrapped-around edge positions in both.
5. Gathers `prepared_input_ids`/`prepared_target_ids` via `torch.index_select` over the
   final indices (237–242, again noting `index_select` over equivalent advanced indexing for
   performance, matching the same comment in `model.py`).
6. Overwrites the input at `mask_id_mask` positions with mask-token ids (249): if
   `min_mask_id` is set, each region gets a *position-wise unique* mask id
   (`(position_in_region) + min_mask_id`, wrapped via modulo `region_width`), otherwise every
   masked position gets the single flat `mask_id`. A follow-up assertion checks the inserted
   ids never exceed `max_mask_id`.
7. Sets `prefix_pos_mask = ~pred_pos_mask` as the normal case, then optionally narrows both
   `pred_pos_mask`/`mask_id_mask`/`prefix_pos_mask` further:
   - If `pad_token_id` is given (260–292): finds the first pad/EOS token per row in the
     *target* sequence, maps it to a region index, and zeroes out `pred_pos_mask`/
     `mask_id_mask`/`prefix_pos_mask` for every position in or after the region *following*
     that one — so no loss or prefix supervision is computed on regions that are mostly
     padding.
   - If `prelude_token_ids` is given (298–333): symmetric logic in the other direction —
     finds the first occurrence of the prelude subsequence via `batch_find_subarray`, maps it
     to a region index, and zeroes out the three masks for every position *at or before* the
     region containing that occurrence (e.g. to skip loss on a fixed chat-template preamble).

Returns `(prepared_input_ids, prepared_target_ids, mask_id_mask, pred_pos_mask,
prefix_pos_mask)`.

### `extend_w_mask` (338–350)

A much simpler helper used only during generative validation's rollout loop: appends
`k_toks - 1` fresh mask-token positions to the end of a sequence (either a flat `mask_id` or
a `min_mask_id`-based position-wise-unique range), used to set up the *next* single-shot
rollout step after consuming the current one's predictions.

### `topk_tok_accuracy` (353–373), `topk_tok_cu_confidence` (376–396), `ent_and_top1_confidence` (399–404), `nll_metric` (407–415)

A family of small vectorized metric functions, each computing per-position and
cumulative-average-over-position statistics without Python loops over the sequence
dimension (using `torch.cumsum` divided by an arange of running lengths):

- `topk_tok_accuracy`: for a set of predicted tokens and reference logits, computes whether
  each predicted token is within the top-`k` of the reference distribution, for each `k` in
  a list, returning both the flat accuracy and the position-wise cumulative accuracy curve.
- `topk_tok_cu_confidence`: analogous cumulative-average curve for "how much probability mass
  the reference distribution's top-`k` tokens carry."
- `ent_and_top1_confidence`: per-position entropy and top-1 token probability of a logits
  tensor.
- `nll_metric`: per-position and per-sequence (via cumulative-average) negative log
  likelihood of `tok_ids` under `logits` — used both for the auxiliary ground-truth prefix
  loss and for "forced teacher loss" metrics that ask the teacher to score various candidate
  continuations (see `generative_validate`).

### `compile_utilities` (418–439)

Wraps the metric functions above (plus `pt_ce_plus_ent_loss`) in `torch.compile`, rebinding
them into the module's global namespace via `globals().update(...)` so subsequent calls in
this file transparently use the compiled versions. `truncate_and_mask`/`extend_w_mask` are
present but commented out from compilation — presumably found not to compile cleanly or not
worth compiling given they run once per step outside the hot training-step path. Only called
when `hparams.train.do_compile` is set (see `main`).

---

## 3. `setup` (442–663) — CLI entry point and hyperparameter normalization

This is the function actually exposed to `jsonargparse`'s `CLI(setup)` at the bottom of the
file (§10), so its signature *is* the training script's full command-line/YAML interface.
Beyond the usual litgpt setup arguments (`model_name`/`model_config`, `out_dir`, `precision`,
`resume`, `data`, `optimizer`, `devices`, `seed`, ...), the fork-specific arguments are the
five dataclass bundles imported from `litgpt.args`: `train`, `eval`, `log`, `singleshot`,
`wandb`, plus `pqds` (only meaningful when `data == "pqds"`).

Key steps, in order:

1. Resolves `model_name`/`model_config` into a `Config`, supporting a `model_name="list"`
   escape hatch that prints all known config names and exits.
2. Injects `train.peak_lr` into the optimizer config's `lr` before hyperparameters are
   captured, then calls `capture_hparams()` (from `litgpt.utils`) to snapshot *all* function
   arguments — including the nested dataclasses — into a single `hparams` object, wrapped via
   `dict2attr` for convenient dotted access (`hparams.singleshot.k_toks`, etc.) while still
   supporting `.to_dict()` for logging.
3. **Curriculum/spec parsing** (522–555): several `hparams.singleshot` fields accept flexible
   CLI encodings (plain ints, YAML lists, or delimited strings) that need runtime parsing —
   `k_toks`, `k_toks_min`, `k_toks_max` via `parse_k_toks`, `topk_values` via
   `parse_topk_values`, `extra_val_trunc_lengths`/`extra_val_k_toks_values` via their
   respective parsers (all imported from `litgpt.args`). Each of `k_toks`/`k_toks_min`/
   `k_toks_max` is then normalized into a curriculum object — `PiecewiseKTokCurriculum` if a
   list of `(step, value)` pairs was given, `ConstantKTokCurriculum` if a plain int — with a
   comment flagging this two-phase approach (dataclass default, then override via
   YAML/CLI, then re-parse) as a workaround for not having found a cleaner way to hook into
   the dataclass init/override ordering.
4. **Special-token resolution** (567–587): `temp_sep_token_id_range` and `prelude_token_ids`
   are parsed from their CLI string encodings. If `mtp_special_token_pattern` is set (the
   alternate special-token specification method), it asserts `mask_id`/
   `temp_sep_token_id_range` were *not* separately set, formats `num_mtp_special_tokens`
   token strings from the pattern, and resolves them to ids via the tokenizer — splitting the
   resolved range into the ids actually used for masking (`mask_id` or
   `min_mask_id`/`max_mask_id`, depending on `pos_wise_unique_mtp_tok_ids`) versus the
   remainder reserved as `temp_sep_token_id_range` for validation-time visual separators.
5. Sets up a `WandbLogger` (the only supported logger — asserted) and resolves multi-node/
   multi-device settings from SLURM environment variables (`SLURM_NNODES`, `SLURM_NTASKS`)
   when present, falling back to the passed-in `devices`/`num_nodes`.
6. Picks a Fabric `strategy`: explicit DDP if `train.fabric_strategy == "ddp"`; FSDP
   (`FSDPStrategy` with `auto_wrap_policy={Block}` — i.e. FSDP shards at `Block` boundaries)
   if `train.fabric_strategy == "fsdp"` or multiple devices/nodes are in play, configured with
   hybrid sharding, an explicit device mesh (`num_nodes x devices` or an explicit
   `fsdp_device_mesh` override), optional activation checkpointing, and a distributed timeout;
   otherwise `"auto"` (single-device).
7. Constructs the `L.Fabric` object, checks NVLink connectivity when using multiple GPUs,
   launches it, logs the full hparams dict, and calls `main(...)` with everything threaded
   through.

---

## 4. `main` (666–890) — model/optimizer/data construction, one-time setup

Runs once per training job (as opposed to `fit`, which is the actual per-step loop).

- **Validates arguments** via `validate_args` (see §9) before doing anything expensive.
- **Builds both models** inside `fabric.init_module(empty_init=True)` (692–694): a student
  `GPT(config, hparams, use_block_mask=hparams.singleshot.train_with_block_mask,
  is_teacher=False)` and a teacher `GPT(..., is_teacher=True)` — two full, independently
  materialized models, not a single model with an EMA/copy trick. `initialize_weights` (§8)
  is then applied to both (GPT-NeoX-style init), embeddings are optionally tied
  (`train.tie_embeddings`), and `max_seq_length` is set on both from `train.max_seq_length`.
- **Derives `train_block_size` automatically for block-mask training** (710–723): if
  `train_with_block_mask` is on, the raw per-row token count the dataloader must supply is
  computed from `P`/`K`/`mask_region_ct` rather than left at the model's full context length
  — this is the "we don't need a full truncation length's worth of raw input" optimization
  the comment calls out. Under randomized `k_toks` (`lockstep_rand_k_toks`/
  `rand_rank_k_toks`), it conservatively sizes for the *minimum* `k_toks` value (since a
  smaller `k_toks` implies a larger `P`, i.e. more raw prefix tokens consumed), with an
  assertion that `train.max_seq_length == singleshot.truncation_length` in that mode — the
  comment warns this logic is "touchy, be careful."
- **Optional `torch.compile`** (729–747): compiles both the metric utility functions
  (`compile_utilities`) and both models, with an optional dynamo cache-size override and
  compile mode.
- **`fabric.setup(...)`** wraps both models for the chosen distributed strategy.
- **One-time block-mask visualization** (754–769): once the models are off the meta device,
  if `model.use_block_mask`, calls `attn_gym.visualize_attention_scores` with the model's own
  `mask_mod` to render and save a picture of the causal mask structure to
  `{out_dir}/mtp_masks/` — purely diagnostic, explicitly marked "unnecessary" in the comments,
  but useful for sanity-checking a new mask configuration.
- **Freezes the teacher**: iterates `model_teacher.parameters()` setting
  `requires_grad = False`.
- **Optimizer**: `instantiate_torch_optimizer` builds the optimizer over **student parameters
  only**, with `fused=True` on CUDA.
- **PQDS pad-token resolution** (780–789): if using the Parquet streaming dataset,
  `pqds.pad_token_id` is resolved from the tokenizer if not explicitly set (`pad_id` first,
  falling back to `eos_id`), then asserted non-`None`.
- **Dataloaders**: built via `get_dataloaders` (§7), with a follow-up special case (794–797)
  when `rollout_multiplier > 1`: the validation dataloader is rebuilt with extra sequence
  length (`k_toks.max_k_toks_value * (rollout_multiplier - 1)` more tokens) so there's enough
  ground truth available to score every chained rollout step during generative validation.
- **Initial checkpoint loading** (804–817): if `initial_checkpoint_dir` is given, both the
  student *and* the teacher are initialized from the same raw checkpoint file
  (`lit_model.pth`) — this is the "continued pretraining" path, distinct from `resume`. If
  `train.ignore_extra_keys_in_init_ckpt`, loading is done with `strict=False`.
- **State dict + resume**: builds the `state` dict described in §1's glossary, then calls
  `find_resume_path` and, if resuming, temporarily **pops `model_teacher` out of `state`**
  before `fabric.load(resume, state)` and puts it back afterward (834–843) — the teacher (and
  optionally the dataloader, if `ignore_dataloader_state_on_resume`) is deliberately excluded
  from Fabric's automatic state restoration and handled manually, presumably because the
  teacher's weights are meant to either come fresh from `initial_checkpoint_dir` or be
  reconstructed identically to the student rather than resumed from a possibly-differently-
  shaped checkpoint.
- **PyTorch 2.7/2.8 FSDP+compile workaround** (847–858): manually triggers
  `_root_pre_forward` on both models' underlying FSDP modules before training starts, working
  around a cited upstream PyTorch issue where lazy FSDP initialization doesn't play well with
  being called from inside a dynamo-compiled forward.
- Calls `fit(...)` (§5) to actually run training, then prints final throughput/memory
  summary statistics.

---

## 5. `fit` (893–1859) — the training loop

The largest function in the file. Structurally: one-time pre-loop setup (§5.1), then the
per-iteration training loop body (§5.2), interleaved with periodic validation (§5.3) and
checkpointing (§5.4), then a post-loop final save/validation (§5.5).

### 5.1 Pre-loop setup (907–1127)

- Optional **initial checkpoint save** and **initial validation** (911–1006) before any
  training happens, gated by `train.initial_save`/`eval.initial_validation` — initial
  validation runs the full `generative_validate` + `validate` pipeline (identical structure
  to the periodic validation block, see §5.3) and logs a wandb `Table` of prompt/generation
  columns plus flattened per-column statistics.
- **FLOPs measurement** (1016–1023): builds a throwaway `GPT` on the `"meta"` device (no real
  memory allocated) and calls Lightning's `measure_flops` on a dummy forward+loss to get a
  TFLOPs estimate for throughput logging, using `chunked_cross_entropy` as the dummy loss.
- **`max_iters` derivation** (1025–1034): `max_tokens` (from `TrainArgs`, per-device after
  dividing by world size) divided by tokens actually consumed per iteration
  (`micro_batch_size * train_block_size`) — this is a **token-budget-driven** stopping
  criterion rather than an epoch or fixed-step count (consistent with `validate_args`
  rejecting `train.epochs`/`train.max_steps` as unsupported).
- **Dataset size sanity-checking and epoch estimation** (1036–1076), including, for the PQDS
  dataset specifically, an optional expensive up-front token-count estimate
  (`_estimate_total_tokens`, only on rank 0) both for raw rows and for block-size-truncated
  rows.
- A cross-check (1059–1064) that the dataloader's actual per-batch token count matches the
  expected `tokens_per_iter` once corrected for the "one extra token for the right-shift"
  convention (the dataloader loads `train_block_size + 1` tokens per row so that
  `input_ids`/`target_ids` can be produced by a one-position shift without losing a training
  position).
- **Running-metric accumulators** (1084–1116): a battery of `torchmetrics.RunningMean`
  objects — one for training loss, grad norm, the four `pt_ce_plus_ent_loss` terms, and the
  auxiliary prefix loss unconditionally; and, gated by `hparams.singleshot.extra_train_metrics`,
  a much larger nested set of per-`top_k`-value and per-mask-position (`0..k_toks-1`) running
  means for accuracy, confidence, and student-forced-teacher loss, both raw-per-position and
  cumulative-over-position.
- Manual `gc.collect()`/`torch.cuda.empty_cache()`, then `warmup_iters` is computed via
  `train.warmup_iters(...)` (defined in `litgpt.args.TrainArgs`, not in this file).

### 5.2 Per-iteration loop body (1128–1660)

Iterates `train_iterator` (a `CycleIterator` wrapping the train dataloader, so it
transparently loops back to the start of the dataset — hence the manual `epoch` tracking used
for logging rather than relying on the dataloader itself to stop). Per iteration:

1. **Stop condition & LR** (1129–1139): breaks once `state["iter_num"] >= max_iters`;
   otherwise computes the LR for this iteration via `get_lr` (§8) and applies it to every
   optimizer param group.
2. **Batch split** (1141–1144): the raw batch (`train_data`, one extra token wider than
   `train_block_size` per the right-shift convention above) is split into `input_ids`/
   `target_ids` via a one-position shift.
3. **`k_toks` sampling** (1146–1173): implements the three mutually exclusive `k_toks`
   selection modes described in the glossary (`rand_rank_k_toks`, `lockstep_rand_k_toks`, or
   the plain curriculum `hparams.singleshot.k_toks.get_value(step_count)`). In the two random
   modes, after sampling `k_toks` it also eagerly recomputes `model.P`/
   `model.block_mask_config["P"]` (and mirrors onto `model_teacher`) since those are about to
   be used for offset math below, even before `reconstruct_block_mask` is called — the inline
   comment flags this ordering as "stale otherwise, be careful."
4. **Offset sampling** (1183–1200): if `train_with_block_mask` and `roll_offsets`, computes
   `rolling_offset` via one of the three offset-randomization modes (rank-random, lockstep-
   random, or the deterministic default sweep `-(step_count % P)`), mirroring the `k_toks`
   selection pattern.
5. **Rebuild the block mask for this step's `k_toks`/offset** (1204–1221): calls
   `model.reconstruct_block_mask(K=k_toks-1, S=truncation_length, B=data_bsz,
   mask_region_ct=..., offset=rolling_offset, bidirect_ss_attn=hparams.singleshot.bidirect_ss_attn)`
   and the equivalent call on `model_teacher` with `bidirect_ss_attn=False` always forced off.
   This is the direct call site into the `model.py` machinery documented in model.md §2 — note
   the mask is rebuilt from scratch **every single training step** (since `k_toks`/offset can
   both vary step to step), so `reconstruct_block_mask`'s cost is on the per-step hot path,
   not amortized.
6. **`truncate_and_mask`** (1223–1248): prepares `input_ids`/`target_ids`/the three masks
   using the function documented in §2, with `pad_token_id` only passed through when
   `hparams.pqds.omit_tail_padding_in_loss` is active.
7. **Consistency assertions & mask-change logging** (1250–1307): checks the actual predicted
   element count against the expected count (adjusting the expectation when tail-padding
   omission can legitimately shrink it), then computes a hash of `model.block_mask_config` and
   — only the first time a given configuration is seen (tracked in
   `model.old_settings_hashset`) and only if `hparams.singleshot.log_masks_and_inputs` — dumps
   a mask visualization and a sample of the prepared training tensors for debugging. A
   `fabric.barrier()` follows since rank-randomized offsets mean different ranks may hit a
   "new" hash at different times.
8. **Forward pass and distillation** (1311–1452), the algorithmic core, run inside
   `fabric.no_backward_sync(model, enabled=is_accumulating)` (skips the expensive gradient
   all-reduce on non-final accumulation micro-steps):
   - **Student pass**: `logits = model(input_ids)`, then gathered at `pred_pos_mask` and
     reshaped to `(tot_mask_regions, k_toks, vocab)` as `soft_stud_preds`; the complementary
     prefix logits/targets (`stud_prefix_logits`, `gt_prefix_ids`) are also extracted for the
     optional auxiliary prefix loss, and `gt_suffix_ids` for optional GT-supervision mode.
   - **Discretize the student's predictions** (1326–1330): either sampled
     (`sample_during_train`) or `argmax`'d into `hard_stud_preds`.
   - **Build the student-forcing sequence for the teacher** (1332–1337): clones `input_ids`
     and overwrites the mask positions with the student's own (hard) predictions — this is
     the "student-forced teacher feedback" step: the teacher is shown what the student
     actually guessed, not the ground truth, so its resulting distribution reflects "given
     that the student said X, what should have come next."
   - **Teacher forward pass** (1343–1347), skipped entirely if `gt_teacher_supervision` is
     on (no teacher forward needed when supervising against ground truth directly), otherwise
     run under `torch.no_grad()` to get `soft_teach_preds`/`hard_teach_preds`.
   - **`last_region_loss_only` reshaping** (1349–1362): if enabled, slices every one of the
     four soft/hard student/teacher prediction tensors down to just the last region per
     sequence (with an explicit comment noting this can't be done via in-place ops without
     breaking autograd, and that downstream code no longer needs `tot_mask_regions` to match).
   - **Extra per-step metrics** (1367–1406), gated by `extra_train_metrics`: computes
     student-forced-teacher NLL (`nll_metric`), and per-example top-k accuracy/confidence
     (`topk_tok_accuracy`, `topk_tok_cu_confidence`) via a Python loop over the batch of mask
     regions (not vectorized across regions, unlike the position dimension inside each metric
     function) — the averaged results feed the many running-mean accumulators from §5.1.
   - **Loss selection** (1409–1430): after asserting at most one supervision mode is active,
     picks between `hard_teacher_supervision` (student vs. teacher's hard argmax labels),
     `gt_teacher_supervision` (student vs. ground truth, teacher unused), and
     `hard_self_teacher_supervision` — a distinct third mode that first computes a
     `match_mask` of positions where the student's and teacher's hard predictions already
     agree, and supervises **only those matching positions** (using the student's own hard
     prediction as the label — i.e. a confirmation/self-distillation signal restricted to
     already-agreeing positions); if no positions match at all, the iteration's backward pass
     is skipped entirely (`continue`, after a defensive `optimizer.zero_grad()`). The
     unconditional fallback (no supervision flag set) is plain teacher-soft-label
     distillation.
   - **Auxiliary prefix loss** (1433–1448): computes `avg_gt_forced_prefix_loss` via
     `nll_metric` over the ordinary prefix positions whenever any exist, and either adds it
     to the main loss (`supervise_prefix`) or replaces the loss with it entirely
     (`supervise_prefix_only`).
   - **Backward** (1450–1452): `fabric.backward(loss / gradient_accumulation_iters)`.
9. **Post-backward bookkeeping** (1454–1480): updates all the running-mean accumulators with
   the current step's detached values (loss, the four `pt_ce_plus_ent_loss` terms, prefix
   loss, and — if `extra_train_metrics` — the large nested per-position/per-`k` metric set).
10. **Optimizer step** (1482–1494), only on the final accumulation micro-step
    (`not is_accumulating`): gradient clipping via `fabric.clip_gradients` (tracked in
    `running_grad_norm`), then `optimizer.step()`/`zero_grad()`, incrementing
    `state["step_count"]`.
11. **Periodic logging** (1496–1658): every `log_iter_interval` iterations, computes all the
    running-mean values, throughput metrics (`ThroughputMonitor`, using the measured FLOPs
    from §5.1), peak CUDA memory (then resets peak stats), and **three distinct
    tokens/sec figures** worth noting since they measure different things: `toks_per_sec_*`
    (every token the model processes in its forward pass, `micro_batch_size * max_seq_length`),
    `sup_toks_per_sec_*` (only tokens actually contributing to the loss, i.e.
    `pred_pos_mask.sum()`, corrected for `last_region_loss_only`), and
    `consumed_toks_per_sec_*` (raw tokens pulled from the dataloader, bounded by
    non-pad-token count when tail-padding omission is active) — the comment notes this last
    figure "can be much lower" since the dataloader only loads the minimum unique raw tokens
    the block-mask scheme actually needs. All of this plus extensive step-timing breakdown
    (`time_fwd`, `time_bwd`, `time_grad_clip`, etc., each individually timed via
    `time.perf_counter()` pairs sprinkled through the loop body) is assembled into a `metrics`
    dict and logged via `fabric.log_dict`.

### 5.3 Periodic validation (1660–1749)

Every `eval.interval` optimizer steps (and only on a non-accumulating iteration), runs the
full generative + loss validation sweep: for every combination of
`hparams.singleshot.extra_val_k_toks_values` and `extra_val_trunc_lengths` (each prepended
with a `None` sentinel meaning "use the training default"), calls `generative_validate` (§6)
and flattens its four return values (`gen_stats`, `gen_stats_lists`, `gen_losses`,
`gen_losses_lists`) into wandb-loggable dicts and `Table` objects — this block is
byte-for-byte structurally identical to the initial-validation block in §5.1 and the
final-validation block in §5.5 (all three were clearly written by copy-paste rather than
factored into a shared helper — worth knowing if you ever need to change this logic in more
than one place). Also runs plain teacher-forced `validate` (§5-adjacent, see below) for a
scalar val loss/perplexity.

### 5.4 Checkpointing (1751–1758)

Saves a step checkpoint when `step_count % save_interval == 0` and/or a "latest" checkpoint
when `step_count % save_latest_interval == 0`; if only the latter fires,
`is_latest_only_save=True` is passed through so `save_checkpoint` (§8) knows to delete the
per-step directory after copying it into `latest/`. As elsewhere, `model_teacher` is popped
out of `state` before the save call and restored after.

### 5.5 Post-loop (1760–1858)

After the loop exits: a final barrier, an optional final checkpoint (`train.final_save`), and
an optional final validation (`eval.final_validation`) — again structurally identical to
§5.1/§5.3.

---

## 6. `validate` (1862–1894) and `reset_model_kv_cache` (1897–1903)

`validate` is the plain (non-generative) validation loop: disables `use_block_mask` on the
student for the duration (so validation loss is computed under **ordinary causal
attention/masking**, not the MTP block mask — i.e. this measures standard next-token
teacher-forced loss, not MTP loss), runs up to `max_iters` batches of plain next-token
prediction with `chunked_cross_entropy`, averages, and restores `use_block_mask` before
returning. `reset_model_kv_cache` is a small helper (used repeatedly in `generative_validate`)
that clears and rebuilds a model's KV cache at a given batch size/max length, then explicitly
resets the cache buffers to zero via each block's `kv_cache.reset_parameters()`.

## 7. `generative_validate` (1906–2443) — the full generation/evaluation pipeline

The most elaborate function in the file, run once per `(val_k_toks, val_truncation_length)`
combination requested by the validation call sites. High-level structure:

- Temporarily extends both models' `max_seq_length` to accommodate `rollout_multiplier > 1`
  (chained rollouts need extra room), and updates the block-mask `K`/`P` to whatever
  `val_k_toks` is being evaluated at (defaulting to the curriculum's max value if not
  overridden).
- **Per validation batch, per individual row** (the comment at 1991–1992 notes this is done
  sequentially rather than batched — "batching would be nice" — i.e. a known
  performance-limiting simplification), for each row it produces and scores **four kinds of
  completions**:
  1. **SS (single-shot) rollout** (2010–2079): repeatedly calls `truncate_and_mask` +
     `model.reconstruct_block_mask` + a student forward pass to greedily fill in `k_toks` at a
     time, chaining `rollout_multiplier` times via `extend_w_mask` between rounds — this is
     the model's own MTP-style generation, and per-position entropy/top-1-confidence
     (`ent_and_top1_confidence`) is tracked along the way.
  2. **SS sampled rollout** (2081–2119), only if `hparams.singleshot.num_samples` is set:
     the same chained-rollout procedure but drawing `num_samples` independent stochastic
     (multinomial) trajectories per row instead of one greedy one.
  3. **AR (autoregressive) rollout, teacher and student** (2121–2182): block-masking is
     switched off on both models (`use_block_mask = False`) so they generate purely
     autoregressively one token at a time via `litgpt.generate.base.generate` (imported as
     `generate_fn`), using a properly reset/sized KV cache (`reset_model_kv_cache`), greedy
     decoding (`temperature=0.0`), and **no EOS stopping** (`eos_id=None`, with an explicit
     comment warning this is required because all distributed ranks must generate the same
     number of tokens in lockstep or the run will hang).
  4. **Forced-teacher scoring of all four completions** (2203–2238): stacks
     `{GT completion, AR teacher gen, AR student gen, SS student gen}`, runs each through the
     teacher once (`model_teacher(input_ids)`), and computes per-sequence and
     per-rollout-segment NLL against the teacher's own distribution via `nll_metric` — this
     produces the "forced teacher loss" metrics that answer "how plausible does the teacher
     find each of these four candidate continuations."
- **Visual separator insertion for logging** (2240–2279): since chained rollouts concatenate
  multiple `k_toks`-sized segments together, a spare special-token id from
  `temp_sep_token_id_range` (probing for one not already present in the generated text, to
  avoid collisions) is spliced in between rollout segments purely so the decoded text logged
  to wandb visually shows `<ss0>`, `<ss1>`, etc. segment boundaries.
- **Decoding and table assembly** (2280–2317): decodes prompt/GT/AR-teacher/AR-student/SS
  outputs (and SS samples) to strings, collects them into the `generations` list used to build
  the wandb generation table.
- **Cleanup and mask restoration** (2314–2350): clears both models' KV caches, restores
  `max_seq_length`, and — importantly — calls `model.reconstruct_block_mask`/
  `model_teacher.reconstruct_block_mask` once more at the end to **reset both models back to
  the base training configuration** (current curriculum `k_toks`, zero offset), since the
  per-row rollout loop above repeatedly overwrote `model.K`/`model.P`/the block mask itself
  for its own purposes and training must not silently continue with validation's leftover
  mask state.
- **Aggregate statistics** (2352–2434): for each of the five output columns (Prompt, GT
  Compl, AR Teach Gen, AR Stud Gen, SS Stud Gen, plus SS Stud Sample if sampling is on),
  computes repetition/diversity metrics (`compute_repetition_metrics`, §7.1) both overall and
  per rollout segment (`col_name kx{ri}`); computes mean/std/median/min/max/quantiles of the
  tracked entropy and top-1-confidence series for the SS output specifically; and aggregates
  the four forced-teacher-loss series (overall and per rollout segment) the same way. Returns
  `(generations, generation_stats_agg, generation_stats_lists, all_output_forced_teach_losses_agg, all_output_forced_teach_losses_lists)`
  — the `_agg` dicts are scalar summary stats for wandb metrics, the `_lists` dicts are the
  raw per-row values used to build wandb `Table`s for per-example inspection.

### 7.1 `compute_repetition_metrics` (2446–2470)

Thin wrapper around `measure_repetition_and_diversity` (imported from
`litgpt.repetition_diversity_tokens`, not defined in this file) applied per-row, then
aggregated into mean/std/median/min/max/quantiles per metric — the same
aggregation pattern used repeatedly throughout `generative_validate`.

---

## 8. Dataloading, scheduling, weight init, and checkpointing helpers (2473–2738)

### `get_dataloaders` (2473–2538)

Builds train/val dataloaders from either the custom `ParquetStream` streaming dataset
(`data == "pqds"`, using `torchdata`'s `StatefulDataLoader` so dataloader position survives a
resume, configured with sharding by `fabric.world_size`/`fabric.global_rank`) or a standard
litgpt `DataModule` (`data.connect(...)`/`prepare_data()`/`setup()`/
`train_dataloader()`/`val_dataloader()`).

### `get_lr` (2542–2559)

Standard linear-warmup-then-decay LR scheduler, supporting `"constant"` (flat after warmup)
or `"cosine"` (cosine decay to `min_lr`, asserted required for this mode) schedules; returns
`min_lr` unconditionally once `it > max_iters`.

### `ConstantKTokCurriculum` (2562–2573) and `PiecewiseKTokCurriculum` (2576–2603)

The two curriculum classes referenced throughout the glossary and `fit`. Both expose
`.get_value(current_step)`, `.max_k_toks_value`, and `.min_k_toks_value`, so calling code
doesn't need to branch on which curriculum type is in use. `ConstantKTokCurriculum` always
returns the same value. `PiecewiseKTokCurriculum` takes a list of `(start_step, value)`
tuples (assumed sorted) and returns whichever value's interval contains `current_step`,
treating the last entry's interval as open-ended (`[start, ∞)`); raises `RuntimeError` if no
interval matches, which per the logic given should be unreachable. Both implement
`__repr__` for readable logging of the curriculum shape.

### `initialize_weights` (2606–2625)

GPT-NeoX-style initialization (citing arXiv:2204.06745, adapted from the TinyLlama repo):
monkey-patches `reset_parameters` onto every `nn.Embedding`/`nn.Linear` submodule to use
`std = sqrt(2/5/n_embd)`, then does a second pass specifically overriding the output
projection (`proj`) of every `LLaMAMLP`/`CausalSelfAttention` module to a narrower
depth-scaled std (`1/sqrt(n_embd)/n_layer`) — the standard "scale down residual-stream output
projections by layer count" trick for stabilizing deep transformer training. Actually invokes
these patched `reset_parameters` methods only if **not** using FSDP
(`isinstance(fabric.strategy, FSDPStrategy)` check) — under FSDP, parameter materialization
and reset happens elsewhere in the FSDP wrapping path instead, so calling `reset_parameters`
here directly would either be redundant or operate on not-yet-real (meta) tensors.

### `save_checkpoint` (2627–2682)

Saves the full `state` dict via `fabric.save` (student model, optimizer, dataloader state,
counters — teacher is expected to already be popped out by the caller, per §4/§5.4), then
(rank 0 only) writes out hyperparameters (`save_hyperparameters(setup, ...)` — notably passes
the `setup` function itself so the saved hyperparameters reflect the actual CLI schema),
copies tokenizer config files if a `tokenizer_dir` was given, and saves the model `Config`.
If `train.save_latest_ckpt`, also maintains a rolling `latest/` directory: moves any existing
`latest/` to `latest_prev/`, copies the just-saved checkpoint directory to `latest/`, then
removes `latest_prev/` — a "safe-ish" copy-then-swap-then-cleanup ordering that avoids ever
leaving `latest/` in a fully-deleted state partway through, though `latest_prev/` is not kept
around as a real rollback point (removed immediately after the copy succeeds). If this was a
"latest-only" save (`is_latest_only_save`), the numbered per-step directory that was just
copied *from* is then deleted, keeping only the `latest/` copy. Finally calls `prune_ckpt_dirs`
to enforce `train.max_ckpts_to_keep`, wrapping the whole function in manual GC/CUDA-cache
clears before and after (checkpoint save is memory-intensive) and logging elapsed save time.

### `prune_ckpt_dirs` (2685–2698)

Rank-0-only: globs all `step-*` checkpoint directories, sorted (relies on zero-padded step
numbers in the directory name sorting correctly as strings — noted in a comment as an
assumption that would need revisiting if the naming scheme changed), and deletes the oldest
ones beyond `hparams.train.max_ckpts_to_keep`. A `None` value disables pruning entirely (keep
everything).

---

## 9. `validate_args` (2700–2738)

A pre-flight argument-consistency checker, called once at the top of `main`. Collects a list
of `issues` strings and raises a single `ValueError` joining all of them if any exist, rather
than failing on the first problem found — worth knowing if you're debugging a rejected config,
since the error message may list several independent problems at once. Checks include:
rejecting `train.max_steps`/`train.epochs`/`eval.max_new_tokens` as explicitly unsupported by
this script (token-budget-driven training instead, see §5.1); requiring `train.max_tokens`/
`train.max_norm` to be set; requiring `train.save_latest_interval` if `save_latest_ckpt` is
on; and several MTP-specific consistency rules cross-referenced in the glossary above
(`mask_region_ct` must be `1` when not using block-mask training; `roll_offsets`/
`log_masks_and_inputs` require `train_with_block_mask`; the various `rand_rank_*`/
`lockstep_rand_*` pairs are mutually exclusive).

---

## 10. CLI entry point (2741–2750)

Configures `jsonargparse` parsing settings (enables reading configs from URLs, enables
docstring-attribute parsing so the `setup()` docstring's `Arguments:` section populates
`--help` text), sets `torch.set_float32_matmul_precision("high")` globally, and hands the
whole `setup` function to `CLI(setup)` — meaning every parameter of `setup()` (and
transitively, every field of `TrainArgs`/`EvalArgs`/`SingleShotArgs`/`WandbArgs`/`PQDSArgs`/
`LogArgs`) becomes a CLI flag or YAML-config key automatically.

---

## 11. How a single training step actually flows (cross-reference)

Given how spread out the logic is, here's the direct call sequence for one training
iteration, tying together §5.2 and `model.py`'s machinery (model.md §8):

1. `fit` samples `k_toks` (curriculum or random) and `rolling_offset` (deterministic sweep or
   random) for this step.
2. `model.reconstruct_block_mask(...)` / `model_teacher.reconstruct_block_mask(...)` rebuild
   the FlexAttention block mask for both models (`model.py`, documented in model.md §2).
3. `truncate_and_mask(...)` (this file, §2) reshapes the raw batch to match that same
   region/offset layout and produces the input with mask tokens written in, plus
   `pred_pos_mask`/`prefix_pos_mask`.
4. `model(input_ids)` runs the student forward pass; internally, `GPT.forward` calls
   `construct_block_rope_feats` (model.md §2) to align RoPE positions with the same layout,
   and each `CausalSelfAttention` layer consults the block mask built in step 2 (model.md §4).
5. The student's hard predictions are spliced into a "student-forced" input and run through
   the frozen teacher (`model_teacher(...)`), which sees the same block-mask/RoPE alignment
   built for it in step 2.
6. `pt_ce_plus_ent_loss` (this file, §2) computes the distillation loss from student vs.
   teacher (or GT, or self-matched) predictions at the masked positions, optionally combined
   with the auxiliary ground-truth prefix loss.
7. Standard Fabric backward/accumulate/step/clip machinery follows.

If you're chasing a bug in the MTP training path, this is the thread to follow across both
files.

---

## 12. Worked example: how the sequence-length/masking parameters interact

The glossary in §1 lists `max_seq_length`, `train_block_size`, `truncation_length`, `P`, `K`,
`mask_region_ct`, `k_toks`/`k_toks_min`/`k_toks_max`, `mask_id`/`min_mask_id`/`max_mask_id`,
`roll_offsets`, and the validation-time `rollout_multiplier`/`multi_region_val_correction`/
`extra_val_trunc_lengths`/`extra_val_k_toks_values` largely as independent entries — this
section walks one concrete, small configuration through construction, a training step, and a
validation call, so you can see how they actually constrain each other. The index arithmetic
below was checked against a faithful pure-Python re-implementation of `truncate_and_mask`'s
logic (not hand-derived), so the numbers are exact, not illustrative approximations. It
reuses the same `P=4, K=2, S=12, mask_region_ct=2` numbers as [model.md](model.md)'s and
[mtp.md](mtp.md)'s worked examples for continuity.

### 12.1 Configuration

```
train.max_seq_length          = 12      # -> model.max_seq_length = model_teacher.max_seq_length = 12
eval.val_block_size            = None    # -> defaults to model.max_seq_length = 12
singleshot.truncation_length   = 12      # S
singleshot.mask_region_ct      = 2
singleshot.k_toks              = 3       # ConstantKTokCurriculum(3)  =>  max_k_toks_value = 3
singleshot.train_with_block_mask = True
singleshot.roll_offsets        = True    # deterministic sweep (no rand_rank/lockstep flag set)
singleshot.min_mask_id         = 100
singleshot.max_mask_id         = 101
```

### 12.2 Construction (`main`)

`GPT.__init__` always sizes the model's own `K`/`P` from the *base* curriculum's
**maximum**, regardless of what any given step will actually use:
`K = k_toks.max_k_toks_value - 1 = 3 - 1 = 2`, `S = 12`, `mask_region_ct = 2`, giving
`P = (12 // 2) - 2 = 4` (matches the config above by construction, since this example's
curriculum is constant). Then, in `main` (§4), because `train_with_block_mask` is on and
neither randomized-`k_toks` mode is active, `train.train_block_size` is derived from these
same `model.P`/`model.K` values:

```
train.train_block_size = mask_region_ct * model.P + model.K = 2*4 + 2 = 10
```

So the dataloader is configured to hand each row **10 raw tokens** (§1: `train_block_size`),
not the full `S = 12` — this is the "don't load a full truncation length's worth of raw
input" optimization from §4. `model.max_seq_length` stays `12` (that's the RoPE-cache/KV-cache
size, unrelated to how many raw tokens the dataloader supplies per row).

### 12.3 One training step, with `roll_offsets`

At step `state["step_count"] = s`, the deterministic offset sweep (§1, no rand-rank/lockstep
flag) gives `rolling_offset = -(s % P) = -(s % 4)`, cycling `0, -1, -2, -3, 0, -1, ...`. For
`s = 0` (`offset = 0`), `model.reconstruct_block_mask(K=2, S=12, mask_region_ct=2, offset=0,
...)` and `truncate_and_mask(k_toks=3, truncation_length=12, mask_region_ct=2, offset=0, ...)`
are called with matching parameters (this is the pairing §11 describes). Tracing
`truncate_and_mask`'s index arithmetic exactly, for a single row of 10 raw input tokens
`[t0..t9]`:

```
output position:   0   1   2   3   4   5   6   7   8   9  10  11
raw source index:  0   1   2   3   4   5   4   5   6   7   8   9
mask_id_mask:      .   .   .   .   M   M   .   .   .   .   M   M
pred_pos_mask:      .   .   .   P   P   P   .   .   .   P   P   P
```

Two things worth noting directly from this table:
- **Regions overlap in the *raw* input, not the prepared output.** Region 0 (output
  positions 0–5) and region 1 (output positions 6–11) both draw from raw indices `4` and `5`
  — region 0 uses them as its (to-be-masked) prediction targets, region 1 reuses the *same*
  raw tokens as ordinary prefix content. This overlap-by-`K` trick is exactly why only
  `mask_region_ct*P + K = 10` raw tokens are needed for `mask_region_ct*(P+K) = 12` prepared
  output positions, instead of `12` raw tokens with no reuse.
- **`pred_pos_mask` is one position wider than `mask_id_mask`** per region (output position 3
  and 9 are marked `P` but not `M`) — the position immediately before each mask block also
  contributes a loss term (it predicts into the first masked position), matching §1's
  glossary note.

For `s = 1` (`offset = -1`), the same trace shifts the region boundaries left by one output
position (`raw source index: 0 1 2 3 4 3 4 5 6 7 8 7`, using only 9 raw tokens this time —
still safely within the `train_block_size = 10` budget provisioned above). The budget of 10
was sized for `offset = 0` specifically (it's the maximum over one full `roll_offsets` cycle
in this configuration), so it comfortably covers every offset in the `0..-3` sweep here.

**Mask-id assignment**, using `min_mask_id = 100`/`max_mask_id = 101` (position-wise-unique
ids, one per slot in `K = 2`): the formula `(pos - offset) % region_width - P + min_mask_id`
evaluated at the four `mask_id_mask` positions (4, 5, 10, 11, with `offset=0`,
`region_width=6`, `P=4`) gives `100, 101, 100, 101` — i.e. the *first* mask slot of every
region always gets id `100`, the *second* always gets `101`, regardless of which region
it's in. This is what "position-wise-unique" means: unique **within a region**, reused
**across** regions.

### 12.4 A gotcha worth knowing: constant vs. varying `k_toks` with `mask_region_ct > 1`

Because `main` sizes `train.train_block_size` **once**, from the base curriculum's
*maximum* `k_toks` (§12.2), a `PiecewiseKTokCurriculum` that ever schedules a **smaller**
`k_toks` than its max runs into an under-provisioned dataloader whenever `mask_region_ct > 1`
— traced concretely: with the same `S=12, mask_region_ct=2`, dropping to `k_toks=2` (`K=1`)
at some later step gives `P = (12//2) - 1 = 5`, and `truncate_and_mask` then needs
`mask_region_ct*P + K = 2*5 + 1 = 11` raw tokens — one more than the `train_block_size = 10`
provisioned for the curriculum's max (`k_toks=3`). This is a direct consequence of
`consumed = mask_region_ct*(S//mask_region_ct) - K*(mask_region_ct - 1)`: whenever
`mask_region_ct > 1`, a *smaller* `K` (i.e. smaller `k_toks`) needs *more* raw tokens, not
fewer — so sizing off the max `k_toks` only happens to be safe when `mask_region_ct == 1`
(where `consumed = S` regardless of `K`) or when the curriculum never actually uses anything
below its stated max (e.g. `ConstantKTokCurriculum`, as in this example). If you use a
`k_toks` curriculum that decreases (or a `PiecewiseKTokCurriculum` with more than one
distinct value) together with `mask_region_ct > 1`, check this arithmetic for your specific
values.

**This is exactly why the two randomized-`k_toks` modes size differently.** When
`lockstep_rand_k_toks`/`rand_rank_k_toks` is on, `main` instead sizes
`train.train_block_size` from `k_toks_min` (§4: "the largest input tok count is determined by
the *min* `k` value") — i.e. from the *worst case* (smallest `k_toks`, largest `P`) over the
whole randomized range, not from the max. E.g. with `k_toks_min = 2`, `k_toks_max = 3` on
this same `S=12, mask_region_ct=2` setup: `min_K = 2 - 1 = 1`,
`corresp_P = (12//2) - 1 = 5`, `train.train_block_size = 2*5 + 1 = 11` — correctly
provisioned for *every* value `k_toks` might randomly take in `[2, 3]`, unlike the
plain-curriculum branch above.

### 12.5 Validation with `rollout_multiplier > 1`

Now consider a validation call with `singleshot.rollout_multiplier = 2`,
`singleshot.multi_region_val_correction = True` (the default), and
`singleshot.extra_val_k_toks_values = [2]`, `singleshot.extra_val_trunc_lengths = [16]` added
on top of the base config from §12.1.

**The validation sweep.** Every call site in `fit` (§5.1/§5.3/§5.5) builds
`val_k_toks_values = [None, 2]` and `val_lengths = [None, 16]` and calls
`generative_validate` once per combination (4 calls here), where `None` means "use the
default computed value" and a real number overrides it outright. Concretely:

| `(val_k_toks, val_len)` | resolved `val_k_toks` | resolved `val_truncation_length` |
|---|---|---|
| `(None, None)` | `3` (curriculum's `max_k_toks_value`) | via correction, see below: `10` |
| `(2, None)` | `2` (explicit override) | via correction: `11` |
| `(None, 16)` | `3` | `16` (explicit override, correction skipped entirely) |
| `(2, 16)` | `2` | `16` |

**How the `(None, None)`/`(2, None)` correction values are computed.** Before computing
`val_truncation_length`, `generative_validate` first re-derives `model.K`/`model.P` **fresh**
from whatever `val_k_toks` is in play for *this* call (`model.K = val_k_toks - 1`,
`model.P = (model.S // mask_region_ct) - (val_k_toks - 1)`) — this is the key difference from
§12.4's gotcha: it's recomputed per call rather than reused stale from construction. With
`multi_region_val_correction` on, `val_truncation_length = mask_region_ct * model.P + K`,
which is exactly the same formula as `train.train_block_size` in §12.2/§12.4 — for
`val_k_toks=3` this reproduces `10` (matching training's own budget), and for `val_k_toks=2`
it correctly produces `11` (matching the *worst-case* budget from §12.4's
`lockstep_rand_k_toks` sizing, not the under-provisioned `10`). Without
`multi_region_val_correction`, this would just fall back to the flat
`singleshot.truncation_length = 12` regardless of `val_k_toks` — usable, but not
budget-matched to what a `mask_region_ct`-region training step for that `val_k_toks` would
actually have consumed.

**One layout difference worth flagging**: the single-shot rollout's own internal
`truncate_and_mask` call inside `generative_validate` does **not** pass `mask_region_ct`
(it uses the function's default, `mask_region_ct=1`) — so regardless of the training-time
`mask_region_ct=2`, generative validation always prepares its rollout prompt as **one single
region** of width `val_truncation_length`, with `P = val_truncation_length - K` prefix tokens
and `K` mask tokens. `multi_region_val_correction`'s job is only to pick a
`val_truncation_length` that's *comparably sized* to what `mask_region_ct` training regions
would have consumed — it does not reproduce the multi-region tiling itself at validation
time.

**Where `rollout_multiplier` comes in.** For the `(None, None)` case
(`val_k_toks=3`, `val_truncation_length=10`), `rollout_multiplier=2` means the single-shot
rollout loop (`model.py`/`pretrain.py` §7) runs the `K=2`-token block-prediction step twice,
chaining the second round onto the first via `extend_w_mask`. The autoregressive (teacher/
student AR) comparison pass is sized to match: `eff_k_toks = val_k_toks * rollout_multiplier
= 3*2 = 6`, `ar_truncation_length = val_truncation_length + val_k_toks*(rollout_multiplier-1)
= 10 + 3*1 = 13` — i.e. the AR generation continues for `eff_k_toks=6` tokens total (two
`k_toks`-sized segments) so its output is directly comparable to the two chained SS rollout
segments. To have room for this, both models' `max_seq_length` is temporarily bumped up by
`adjustment = k_toks.max_k_toks_value*(rollout_multiplier-1) = 3*1 = 3` (`12 -> 15`) for the
duration of `generative_validate`, and — separately, once, back in `main` (§4) — the
validation *dataloader* itself is rebuilt with `eval.val_block_size + adjustment = 12+3 = 15`
so there's enough ground-truth continuation available to score every chained rollout segment,
not just the first.
