# `lm-evaluation-harness-mtp-lm` — Introduction & Guide

## 1. What this is

[`third_party/lm-evaluation-harness-mtp-lm`](../lm-evaluation-harness-mtp-lm) is a vendored clone of
[`jwkirchenbauer/lm-evaluation-harness-mtp-lm`](https://github.com/jwkirchenbauer/lm-evaluation-harness-mtp-lm),
a fork of [EleutherAI's `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) with the
minimum patches needed to drive **MTP (Multi-Token Prediction) generation** — the custom, non-autoregressive
`generate()` codepath implemented in [`third_party/mtp-lm`](../mtp-lm) (`litgpt/transformers_local/{llama,qwen3}/modeling_*.py`)
— through the standard harness evaluation loop, and to pipe MTP-specific inference stats (tokens/sec, average
accepted `k`, forward-eval count, ...) out to `outputs/**/*.jsonl` and W&B.

It is **not** a set of scoring tasks or metrics of its own — task definitions (`gsm8k`, `lambada_openai`, ...),
prompt templates, and metric implementations are all upstream EleutherAI code. The fork only touches:

| File | What it adds |
|---|---|
| [`lm_eval/models/huggingface.py`](../lm-evaluation-harness-mtp-lm/lm_eval/models/huggingface.py) | Accepts a `dict` return from `model.generate()` (instead of only a `Tensor`), unpacks it, decodes `token_ids` to text as usual, and re-attaches the MTP aux fields to the result for downstream logging. |
| [`lm_eval/evaluator.py`](../lm-evaluation-harness-mtp-lm/lm_eval/evaluator.py) | Strips the MTP aux fields back off each response *before* scoring (so `process_results` never sees them), and stashes them under `example["mtp_results"]` in the `log_samples` output for the JSONL dump. |
| [`lm_eval/models/sglang_causallms.py`](../lm-evaluation-harness-mtp-lm/lm_eval/models/sglang_causallms.py) | Equivalent plumbing for the sglang-server backend. |
| `lm_eval/_cli/*` (arg parsing) | List/`+`-separated CLI value handling for compound `gen_kwargs` like `strategy=["conf_adapt",0.9]` or `eos_id=128009+128001`. |

This directory (`lm-evaluation-harness-mtp-lm-patch`) is the sibling of [`third_party/mtp-lm-patch`](../mtp-lm-patch)
— i.e. where local, repo-specific patches/configs/scripts for the vendored harness are meant to live, mirroring
how `mtp-lm-patch` layers numbered `.patch` files and `config_hub/` overrides on top of `third_party/mtp-lm`. It is
currently empty aside from this guide.

The actual MTP `gen_kwargs` presets and `accelerate` configs referenced below live in
[`third_party/mtp-lm/config_hub/lm_eval/`](../mtp-lm/config_hub/lm_eval), e.g. `default_mtp.yaml`, `gsm8k_mtp.yaml`,
`gsm8k_mtp8_ca90.yaml`, `accelerate_config_1N.yaml`.

## 2. Quick guide

### Install

```bash
# (from an env with torch + third_party/mtp-lm already `pip install -e '.[all]'`-ed)
pip install -e third_party/lm-evaluation-harness-mtp-lm
```

This registers two equivalent console entry points, `lm-eval` and `lm_eval`, both defined in
[`pyproject.toml`](../lm-evaluation-harness-mtp-lm/pyproject.toml) as `lm_eval._cli.harness:cli_evaluate`
(also runnable as `python -m lm_eval`). The CLI has three subcommands (`lm_eval/_cli/{run,ls,validate}.py`):
`run` (evaluate), `ls` (list tasks), `validate` (sanity-check a task config). If you omit the subcommand
(e.g. `lm_eval --model hf ...`), `run` is inserted automatically for backward compatibility
(`lm_eval/_cli/harness.py:48-51`).

### Minimal non-MTP smoke test

```bash
lm_eval run --model hf --model_args pretrained=gpt2 --tasks lambada_openai --batch_size 8
```

### MTP run, single process

```bash
lm_eval run \
  --config third_party/mtp-lm/config_hub/lm_eval/gsm8k_mtp.yaml \
  --tasks gsm8k
```

`gsm8k_mtp.yaml` sets `model: hf`, `model_args.trust_remote_code: true`, `batch_size: 1`, and the MTP
`gen_kwargs` block (`do_mtp: true`, `k_toks`, `mask_id`, `strategy`, `eos_id`, `include_prompt`) described below.
CLI flags (`--tasks`, `--model_args`, `--gen_kwargs`, ...) override/merge on top of whatever `--config` sets —
see `lm_eval/config/evaluate_config.py`.

## 3. Where things happen

### 3a. Prompt formulation

All of this is stock upstream harness code (`lm_eval/api/task.py`, class `ConfigurableTask`), unmodified by the
MTP patch:

- **`ConfigurableTask.fewshot_context`** (`lm_eval/api/task.py:927`) builds the full prompt: system
  instruction/description, then `num_fewshot` few-shot Q/A turns sampled via `self.sampler` and rendered with
  `doc_to_text` / `doc_to_target` / `build_qa_turn` (~`task.py:975-1010`), then the eval doc's own question
  (without its answer). Can render as plain text or, with `--apply_chat_template`, through the tokenizer's chat
  template (see `HFLM.apply_chat_template`, `models/huggingface.py:1548`).
- **`ConfigurableTask.doc_to_text` / `doc_to_target` / `doc_to_choice`** (`task.py:1191`, nearby) resolve the
  per-task Jinja/format-string fields from the task's YAML (e.g. `lm_eval/tasks/gsm8k/gsm8k.yaml`).
- **`ConfigurableTask.construct_requests`** (`task.py:1353`) turns `(doc, ctx)` into `Instance` objects. For
  `generate_until` tasks (what MTP runs use) the arguments tuple is built at **`task.py:1394`**:
  `arguments = (ctx, deepcopy(self.config.generation_kwargs))` — this is the exact point where the task's
  (or CLI's) `gen_kwargs`, including `do_mtp`, get attached to every request.

### 3b. Model generation (tokenize → append MTP → forward → accept → decode + stats)

This is the part the fork actually touches, split between the harness (`HFLM`) and the model repo (`mtp-lm`).

| Step | File : line |
|---|---|
| **Tokenize** the batch of contexts | `HFLM.generate_until`, `tok_batch_encode()` call at [`lm_eval/models/huggingface.py:1456`](../lm-evaluation-harness-mtp-lm/lm_eval/models/huggingface.py#L1456) |
| **Dispatch to the model's `generate()`** | `HFLM._model_generate` at [`huggingface.py:970-1006`](../lm-evaluation-harness-mtp-lm/lm_eval/models/huggingface.py#L970) calls `self.model.generate(input_ids=context, ..., **generation_kwargs)`, where `generation_kwargs` still carries `do_mtp`, `k_toks`, `mask_id`, `strategy`, `eos_id`, `include_prompt`, `return_mtp_result_dict` straight from the task's `gen_kwargs`. |
| **Route standard-vs-MTP path** | Because the checkpoint's `config.json` was pushed with `trust_remote_code`-registered custom classes (`LlamaForCausalLM.register_for_auto_class(...)` in [`mtp-lm/litgpt/transformers_local/llama/modeling_llama.py:957-960`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L957)), `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` loads *that* class's overridden `.generate()` ([`modeling_llama.py:501`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L501)). It checks `do_mtp`, strictly validates/strips kwargs, and routes to `_mtp_generate`. (Qwen3 is the same, in `modeling_qwen3.py`.) |
| **Append MTP mask slots** | `_extend_w_mask` ([`modeling_llama.py:855`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L855)), called each loop iteration from inside `_mtp_generate` (`modeling_llama.py:665`) when `k_toks > 1`. Appends `k_toks - 1` placeholder/mask tokens (`mask_id`, or a range `[min_mask_id, max_mask_id)`) after the current sequence — these are the "extra slots" the model fills in during one forward pass, i.e. the actual multi-token-prediction mechanism. |
| **Run the forward pass** | `_mtp_next_tokens` ([`modeling_llama.py:877`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L877)) calls `outputs = model(**model_inputs)` — a single forward producing logits for all `k_toks` positions (prompt + mask slots) at once, using a KV cache carried across loop iterations (`model_kwargs["past_key_values"]`, cropped after each step). |
| **Accept: static vs. `conf_adapt`** | Same function, `modeling_llama.py:887-935`. `strategy is None` ("static") **always accepts all `k_toks`** predicted tokens via `argmax` (`:888`). `strategy[0] == "conf_adapt"` computes per-position top-1 softmax confidence (`_top1_confidence`, `:845`) and accepts only the **longest contiguous prefix whose confidence stays ≥ `strategy[1]` (the threshold)** (`:889-927`), falling back to sampling at `k=1` for the `conf_adapt_sample@1` variant. `strategy[0] == "random"` accepts a randomly-sampled `k` per step. |
| **Return decoded text + effective stats** | `_mtp_generate` assembles `mtp_result_dict` at [`modeling_llama.py:806-818`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L806) — `token_ids`, `tps` (tokens/sec), `avg_effective_k`, `effective_k_values` (per-step accepted count), `num_fwd_evals`, `t_prefill`, `t_gen` — **but only if `return_mtp_result_dict: true` is set in `gen_kwargs`**; otherwise it just returns the raw token tensor. Back in the harness, `HFLM.generate_until` (`huggingface.py:1479-1533`) detects the `dict` return, pops `token_ids`, decodes to text with `tok_decode` + `postprocess_generated_text` (`models/utils.py:726`), and re-attaches the aux MTP fields per-example. `evaluator.py:660-716` then strips those aux fields back off (so scoring never sees them) and files them under `example["mtp_results"]` in the logged-samples JSONL. |

**Note:** none of the checked-in presets (`default_mtp.yaml`, `gsm8k_mtp.yaml`, `gsm8k_mtp8_ca90.yaml` under
`mtp-lm/config_hub/lm_eval/`) set `return_mtp_result_dict: true` — only the example command in
`mtp-lm/README.md` does (`--gen_kwargs ...,return_mtp_result_dict=True,...`). Without it you'll still get MTP
generation and correct scoring, but no `tps`/`avg_effective_k` stats will be logged. Add
`return_mtp_result_dict: true` to the yaml (or `--gen_kwargs ...,return_mtp_result_dict=True`) if you want them.

### 3c. Scoring

Also stock upstream code:

- **`ConfigurableTask.process_results`** (`lm_eval/api/task.py:1441`, `generate_until` branch at `task.py:1564`)
  compares the decoded text against `doc_to_target(doc)` using each metric in `self._metric_fn_list` (e.g.
  `exact_match`, `acc`) — implementations in [`lm_eval/api/metrics.py`](../lm-evaluation-harness-mtp-lm/lm_eval/api/metrics.py)
  (`exact_match_fn`, `acc_fn`, `f1_score`, `bleu`, ... via `@register_metric`).
- Per-doc metric values accumulate in `task_output.sample_metrics` inside the main loop in `evaluator.py:719-720`.
- **`evaluator_utils.consolidate_results`** (`evaluator_utils.py:312`) aggregates those into final task-level
  numbers, and `lm_eval.utils.make_table` renders the summary table printed at the end of a run / saved to
  `results.json`.

## 4. Getting `accelerate launch` to work

`HFLM.__init__` always builds an `accelerate.Accelerator()` (`huggingface.py:133`), which auto-detects whether
it's running under the `accelerate` launcher via env vars the launcher sets — no code changes needed, just
launch it correctly. The canonical, working invocation (from `mtp-lm/README.md`, "Evaluation" section) is:

```bash
accelerate launch --config_file third_party/mtp-lm/config_hub/lm_eval/accelerate_config_1N.yaml -m lm_eval run \
  --config third_party/mtp-lm/config_hub/lm_eval/default_mtp.yaml \
  --model_args pretrained=$RUN_OUTPUT_DIR/$CKPT_SUBDIR,dtype=float32 \
  --tasks gsm8k_cot_singleshot \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs do_sample=False,do_mtp=True,include_prompt=True,return_mtp_result_dict=True,"until=Q:+</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>",mask_id=128259,eos_id=128009+128001,k_toks=1 \
  --output_path $EVAL_OUTPUT_DIR
```

Things that have to line up for this to actually work:

1. **`batch_size` must stay `1`.** `_mtp_generate` hard-asserts single-example generation
   (`modeling_llama.py:576-577`: `NotImplementedError("MTP generation currently only supports single-example
   generation (no batching)")`). This is fine — even ideal — with `accelerate launch`, because `accelerate`
   gives you *data parallelism* (each of the `num_processes` GPUs loads a full model copy and independently
   works through a disjoint slice of the eval set), not batching within a process. All the checked-in MTP yamls
   already set `batch_size: 1`.
2. **Use `accelerate_config_1N.yaml`, not `accelerate_config_fsdp_1N.yaml`.** The former is plain `MULTI_GPU`
   (data-parallel, one full copy per process) — this is the pattern MTP generation is written for. FSDP model
   sharding is a different, orthogonal use case (splitting one model too large for a single GPU across
   processes); the upstream harness README itself warns the basic data-parallel launch doesn't mix with FSDP
   sharding, and `_mtp_generate`'s manual KV-cache handling (`model_kwargs["past_key_values"].crop(...)`,
   `modeling_llama.py:747`) assumes a single process owns the whole cache. Only reach for `parallelize=True` /
   FSDP if the model genuinely doesn't fit on one GPU, and validate results carefully if you do.
3. **`trust_remote_code: true`** must be set (in the yaml's `model_args`, or `--model_args
   trust_remote_code=True`). The MTP checkpoints on the Hub ship their own `modeling_llama.py`/`modeling_qwen3.py`
   (via `register_for_auto_class`, see §3b) alongside the weights; `transformers` will only execute that shipped
   code — and therefore only honor `do_mtp=True` — with `trust_remote_code=True`. Without it you silently get
   vanilla `LlamaForCausalLM`/`Qwen3ForCausalLM`, and any `do_mtp`/`k_toks`/`strategy` kwargs are just ignored
   (or error, since vanilla HF `generate()` doesn't know them).
4. **`do_sample` must stay `False`.** The MTP path explicitly raises if `do_sample=True`
   (`modeling_llama.py:554-555`) — MTP generation here is greedy/confidence-thresholded, not sampled (except the
   `conf_adapt_sample@1` variant's final token, which is internal to the strategy, not `do_sample`).
5. **Prefer `--config <yaml>` over hand-encoding `gen_kwargs` on the CLI** for anything with nested lists (e.g.
   `strategy: ["conf_adapt", 0.9]`). The CLI parser does support compound values (`+`-joined lists, JSON-ish
   values — see `lm_eval/_cli/utils.py`), as the README example above shows for `until=...` and
   `eos_id=128009+128001`, but it's easy to mis-quote. The checked-in yamls under
   `mtp-lm/config_hub/lm_eval/` already encode `k_toks`/`strategy`/`mask_id`/`eos_id` correctly — just point
   `--config` at one and override only what changes per run (`--model_args`, `--tasks`, `--output_path`).
6. **Match `num_processes` in the accelerate config to the GPUs you actually have.**
   `accelerate_config_1N.yaml` defaults to `num_processes: 4` (tuned for a 4-GPU node); override with
   `accelerate launch --num_processes N --config_file ...` if you have a different GPU count, or edit the yaml.
7. **Install the harness fork into the same env you're launching from.** `accelerate launch -m lm_eval run ...`
   needs `lm_eval` importable in every spawned process — i.e. `pip install -e third_party/lm-evaluation-harness-mtp-lm`
   (and `pip install -e 'third_party/mtp-lm[all]'` for the model side) inside the active env, per
   `mtp-lm/install_torch_210_cuda_129_singleshot.sh`. Seeing `ModuleNotFoundError: No module named 'lm_eval'`
   under `accelerate launch` almost always means you're not in that env (e.g. running from system Python instead
   of the project `.venv`).

If you just want to confirm the launcher itself is wired up before worrying about MTP specifics, drop the
`--config`/MTP `gen_kwargs` and run the plain data-parallel smoke test first:

```bash
accelerate launch --config_file third_party/mtp-lm/config_hub/lm_eval/accelerate_config_1N.yaml -m lm_eval run \
  --model hf --model_args pretrained=gpt2 --tasks lambada_openai --batch_size 1
```

If that completes and reports `world_size > 1` in its logs, the accelerate/launch plumbing is fine and any
remaining issue is MTP-specific (points 1–5 above).
