# `litgpt/model.py` — Report

This file is the single-file definition of the decoder-only transformer used by this
project (litgpt/nanoGPT lineage). This fork extends the stock litgpt model with two things
layered on top of the base architecture: (1) a **FlexAttention block-mask scheme for
"singleshot" multi-token prediction (MTP)**, used to train the model to predict several
future tokens at once via self-distillation, and (2) **multi-head latent attention (MLA)**
as an alternative attention implementation. Line numbers below refer to the current version
of the file and are meant as a lookup aid while reading the source side by side with this
report.

---

## 1. Variable & argument glossary

Read this section first and refer back to it — the same names recur across nearly every
block below without being redefined each time.

### Shape/dimension symbols (used throughout, mostly in comments)

| Symbol | Meaning |
|---|---|
| `B` | batch size — how many independent rows are processed in one forward call |
| `T` | sequence length (time steps) processed in the current forward call — the number of token positions in one row |
| `C` | model embedding size (`config.n_embd`) |
| `C*` | attention-internal embedding size (may differ from `C` under GQA) |
| `hs` / `head_size` | size of a single attention head |
| `nh` / `n_head` | number of query heads |
| `n_query_groups` | number of key/value groups shared across query heads (GQA/MQA); equals `n_head` for plain multi-head attention, `1` for multi-query attention |



### `Config` fields (from `litgpt.config.Config`, read all over this file)

| Field | Meaning |
|---|---|
| `n_embd` | model hidden size |
| `n_layer` | number of transformer blocks |
| `n_head`, `n_query_groups`, `head_size` | attention sizing, see above |
| `padded_vocab_size` | vocabulary size padded for efficient matmuls |
| `block_size` | maximum context length the model supports |
| `bias`, `attn_bias`, `lm_head_bias` | whether various Linear layers carry a bias term |
| `norm_class`, `norm_eps` | normalization layer type (e.g. `RMSNorm`) and epsilon |
| `norm_1`, `norm_2`, `post_attention_norm`, `post_mlp_norm`, `shared_attention_norm` | flags controlling which normalization layers exist in a `Block` and whether the attention/MLP branches share one |
| `parallel_residual` | whether attention and MLP read from the same normalized input in parallel, vs. sequentially |
| `norm_qk`, `norm_qk_type` | whether to normalize Q/K projections, and at what point ("olmo2" = before head-splitting, "default" = after) |
| `rope_n_elem` | how many of the `head_size` dimensions receive rotary position embeddings |
| `rope_indices` | optional per-layer index into the RoPE cache, for architectures where different layers use different RoPE slices |
| `rope_adjustments` | dict of Llama-3.1/3.2-style frequency-scaling parameters (`factor`, `low_freq_factor`, `high_freq_factor`, `original_max_seq_len`) |
| `rope_condense_ratio`, `rope_base`, `rope_local_base_freq` | further RoPE frequency controls; `rope_local_base_freq` supports a separate local-attention RoPE base |
| `rotary_percentage` | fraction of head dim covered by RoPE (used to validate KV-cache sizing) |
| `scale_embeddings` | whether token embeddings are scaled by `sqrt(n_embd)` |
| `final_logit_softcapping`, `attention_logit_softcapping` | Gemma-2-style logit softcapping thresholds for the LM head and attention scores respectively |
| `attention_scores_scalar` | optional override for the attention scaling factor (defaults to `1/sqrt(head_size)`) |
| `sliding_window_size`, `sliding_window_indices` | sliding-window attention width and a per-layer bool list of which layers use it |
| `latent_attention` | if true, a `Block` uses `MultiheadLatentAttention` instead of `CausalSelfAttention` |
| `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `qk_head_dim`, `v_head_dim` | MLA-specific low-rank projection dimensions |
| `mlp_class`, `intermediate_size`, `gelu_approximate` | which MLP variant to instantiate and its sizing |
| `n_expert`, `n_expert_per_token`, `moe_intermediate_size` | Mixture-of-Experts routing config (`LLaMAMoE`) |

### `hparams` fields (passed into `GPT.__init__`, only consulted when `use_block_mask=True`)

These live under `hparams.singleshot` and `hparams.train` and configure the **MTP
block-mask** scheme specifically (not part of stock litgpt `Config`):

| Field | Meaning |
|---|---|
| `hparams.singleshot.k_toks.max_k_toks_value` | one more than `K` below — the number of MTP mask-token slots per region |
| `hparams.singleshot.truncation_length` | `S` below — the training sequence length the mask is built for |
| `hparams.singleshot.mask_region_ct` | number of repeated prefix+mask regions tiled across the sequence |
| `hparams.singleshot.bidirect_ss_attn` | whether mask tokens within a region can attend to each other bidirectionally (student only, see below) |
| `hparams.train.micro_batch_size` | `B` used to size the block mask |

### MTP block-mask internal variables (set in `GPT.reconstruct_block_mask`, consumed in `construct_block_rope_feats` and `CausalSelfAttention`)

The singleshot MTP scheme divides each training sequence into repeated **regions**, each
made of a block of ordinary "prefix" tokens followed by a block of special mask tokens the
model must predict several steps ahead:

| Variable | Meaning |
|---|---|
| `K` | number of mask-token positions per region (`= max_k_toks_value - 1`) |
| `S` | total truncated sequence length the mask covers |
| `mask_region_ct` | how many regions tile across `S` |
| `P` | prefix length per region, derived as `(S // mask_region_ct) - K` — must be positive (asserted) since it's the number of real tokens available to attend to |
| `offset` (`Ofs`) | signed shift applied to the region boundaries/RoPE positions; magnitude must be `< P` (asserted) |
| `B`, `H` | batch size and `n_query_groups`, needed to size the FlexAttention `BlockMask` |
| `bidirect_ss_attn` | if true, mask tokens in a region can see each other bidirectionally instead of strictly causally — intended for the student model only; the code prints a warning if a teacher is configured this way |
| `self.mask_mod` | the callable mask-predicate built by `interleaved_mtp_mask_mod_factory` (defined in `litgpt/mtp.py`, not this file) — kept as an attribute for external visualization/debugging |
| `self.block_mask` | a `functools.partial(create_block_mask, ...)` — deliberately left un-materialized until first use, so it can be built on whatever device the model ends up on |
| `self.block_mask_rope_feats` | `{"cos": ..., "sin": ...}`, the RoPE caches re-tiled to match the interleaved prefix/mask token layout (see §3, `construct_block_rope_feats`) |

**Tiny worked example.** Say `S = 12` (a 12-token training sequence), `mask_region_ct = 2`
(two regions), and `k_toks = 3` (so `K = k_toks - 1 = 2` mask-token slots per region). Then:

```
P = (S // mask_region_ct) - K = (12 // 2) - 2 = 4
region_width = P + K = 4 + 2 = 6
```

The 12 positions split into two 6-wide regions, each with 4 real "prefix" tokens followed by
2 mask-token slots:

```
position:  0  1  2  3  4  5  6  7  8  9 10 11
content:  [p  p  p  p  m  m][p  p  p  p  m  m]
region:    └──── region 0 ────┘└──── region 1 ────┘
                        ^prefix (P=4)  ^mask (K=2)
```

`p` = an ordinary prefix token attended to causally as usual; `m` = a position overwritten
with a MASK-token id, which the model must predict (this is what `mask_id_mask` marks in
`pretrain.py`'s `truncate_and_mask`). Each region's mask block can only "see" its own
region's prefix (plus, causally, everything before it) — that's the actual masking pattern
`interleaved_mtp_mask_mod` encodes, and it's why `P` must stay positive: with `K` too large
relative to `S`/`mask_region_ct`, a region would have no real prefix tokens left to predict
from at all. A nonzero `offset` just slides where these region boundaries fall along the
sequence (e.g. `offset = -1` shifts every region's start left by one position) rather than
changing `P`/`K`/`region_width` themselves — see `construct_block_rope_feats` above for how
that slide is implemented.

**Where do `B` and `T` fit into this picture, and how does `T` relate to `S`?** Continuing
the example: if this batch has `B = 2` rows, `reconstruct_block_mask` is called with
`B = 2` (from `hparams.train.micro_batch_size` in `pretrain.py`, see the `hparams` table
above) so the FlexAttention `BlockMask` is built for shape `(B=2, H, Q_LEN=12, KV_LEN=12)` —
i.e. the *same* 12-position region pattern drawn above is applied identically to both rows
in the batch; `B` here just says "how many copies of this mask, one per row," it doesn't
change `P`/`K`/`region_width` at all.

`S` and the runtime `T` (`= idx.size(1)` inside `GPT.forward`, see the shape-symbol table
above) are conceptually different things — `S` is a *construction-time parameter* you choose
when building the mask, `T` is the *actual width of the tensor* handed to `forward` — but in
practice **they must be equal, and nothing in this file enforces that automatically**: the
block mask is built for exactly `Q_LEN=KV_LEN=S`, so if you ever called
`model(idx)` with `idx.shape[1] != S`, the attention call would either error on a shape
mismatch or silently attend against the wrong region layout. This is why `pretrain.py`
always rebuilds the mask (`reconstruct_block_mask(S=..., ...)`) using the exact same length
it's about to feed into `forward` — for a normal training step that's
`hparams.singleshot.truncation_length` (§1's `S`), matching what `truncate_and_mask`
produced; during the rollout loop in `generative_validate`, `S` is instead re-set on every
rollout iteration to `ss_outputs.shape[0]` (the running generated sequence's *current*
length, which grows each rollout step) precisely so `S` keeps tracking `T` as the sequence
being generated gets longer. So: `B` mirrors the runtime batch size, and `S` must mirror the
runtime `T` — the model doesn't check this for you, the caller (`pretrain.py`) is
responsible for keeping them in lockstep every time it rebuilds the mask.

### Forward-pass / runtime arguments (recur in `GPT.forward`, `Block.forward`, `CausalSelfAttention.forward`)

| Name | Meaning |
|---|---|
| `idx` | input token ids, shape `(B, T)` |
| `input_pos` | if given, absolute positions of the current tokens for KV-cache decoding (shape `(T,)` or `(B, T)`); if `None`, the model is in full-sequence training mode with a plain causal mask |
| `input_pos_maxp1` | optional `max(input_pos) + 1`, lets the KV-cache read be sliced short for speed instead of always reading the full `max_seq_length` buffer |
| `lm_head_chunk_size` | if `> 0`, splits the final LM-head projection into chunks along the sequence dimension to cap peak autograd memory; returns a list of logit chunks instead of one tensor |
| `cos`, `sin` | RoPE caches, sliced/gathered per `input_pos` or per the block-mask's interleaved layout before being handed down into each `Block` |
| `mask` | either `None` (defaults to a causal mask inside SDPA), a dense boolean/float mask (KV-cache decoding, sliding window), or unused when a FlexAttention `block_mask` takes over instead |
| `is_teacher` | `GPT.__init__` flag distinguishing the teacher model (produces distillation targets) from the student; primarily affects the bidirectional-attention warning above |
| `use_block_mask` | `GPT`/`Block`/`CausalSelfAttention` flag that switches attention from standard SDPA/causal masking to the FlexAttention MTP block mask; propagated top-down via a property setter so it can be toggled after construction |

---

## 2. `GPT` (lines 28–441) — the top-level model

### `__init__` (29–70)

Constructs the standard litgpt module tree: a token embedding `wte`, an `n_layer`-deep
`ModuleList` of `Block`s, a final norm `ln_f`, and the `lm_head` projection to vocabulary
logits. Compared to stock litgpt, the constructor signature now also takes `hparams` (the
MTP/training hyperparameter bundle described in the glossary), plus `use_block_mask` and
`is_teacher`. If `use_block_mask` is set, the constructor calls `reconstruct_block_mask`
*before* building the block list, using the `hparams.singleshot`/`hparams.train` fields to
derive `K`, `S`, `B`, and the region/offset parameters. The resulting (still-unmaterialized)
mask object is then threaded into every `Block` at construction time via the `block_mask=`
argument, so all attention layers reference the same mask configuration from the start.

### `use_block_mask` property (72–84)

A property/setter pair rather than a plain attribute. Setting it walks every block's
`attn.use_block_mask` and updates it in place — this is what lets the model be switched
between block-mask (MTP) and ordinary causal attention *after* construction, e.g. to reuse
one model object for both a bidirectional-student and causal-teacher configuration without
rebuilding it. The setter guards against running before `self.transformer` exists (it is
invoked once during `__init__` itself, before the module dict is built).

### `prop_block_mask_partial` and `instantiate_block_mask` (85–95)

Two small helpers around mask materialization. `prop_block_mask_partial` pushes an
un-materialized `partial(create_block_mask, ...)` down into every layer — used when no
device is known yet. `instantiate_block_mask` does the opposite: given a device, it calls
the partial once to produce a concrete `BlockMask` tensor and assigns that same object to
every layer. In practice, per the comment in `scaled_dot_product_attention` (see §4), FSDP
makes sharing one mask object across layers difficult in the general case, so in the actual
training loop this materialization tends to happen lazily per-layer instead (see below).

### `reconstruct_block_mask` (96–154)

The core of the MTP masking scheme, and the piece most worth understanding if you're
modifying the singleshot training setup. Given `K`, `S`, `B`, `mask_region_ct`, an optional
`offset`, and `bidirect_ss_attn`, it first derives `P = (S // mask_region_ct) - K`, the
number of genuine prefix tokens available per region (asserted positive — a
misconfiguration here means a region has no real tokens to attend to). It stores all of
these as attributes on `self` (`self.K`, `self.P`, `self.S`, `self.Ofs`, `self.B`, `self.H`,
`self.mask_region_ct`, `self.bidirect_ss_attn`) so later code (notably
`construct_block_rope_feats`) can read them back without them being passed around
explicitly. It then builds `interleaved_mtp_mask_mod` via
`interleaved_mtp_mask_mod_factory(P, K, offset, bidirect_ss_attn=...)` — the actual
mask-predicate logic lives in `litgpt/mtp.py`, not in this file — and wraps
`create_block_mask` in a `functools.partial` bound to this mask-mod and the `B`/`H`/`Q_LEN`/
`KV_LEN` shape arguments, but *not* yet a device. If a `device` argument is supplied, it
materializes the mask immediately via `instantiate_block_mask`; otherwise it just propagates
the partial down via `prop_block_mask_partial`, deferring materialization to first use.

### `max_seq_length` property (157–186)

Lets inference use a shorter context than the model's trained `block_size`, to avoid
allocating unused RoPE-cache/KV-cache memory. Setting it validates against
`config.block_size`, and — on first use or whenever the value changes — recomputes the RoPE
`cos`/`sin` caches via `rope_cache()`. If an existing KV cache is now shorter than the new
`max_seq_length`, it prints a warning telling the caller to re-run `set_kv_cache` before the
next forward pass, rather than silently producing wrong results.

### `reset_parameters` and `_init_weights` (188–199)

`reset_parameters` simply rebuilds the RoPE cache (useful after moving the model to a new
device or dtype). `_init_weights`, meant to be used via `gpt.apply(gpt._init_weights)`,
applies the standard litgpt initialization: `Linear` and `Embedding` weights get
`normal_(mean=0, std=0.02)`, and `Linear` biases are zeroed.

### `forward` (201–302)

The main forward pass, and the place where the KV-cache/generation and
training/block-mask code paths diverge. It first validates `T <= max_seq_length`.

If `input_pos` is provided (generation/decoding mode), the RoPE `cos`/`sin` and the causal
`mask_cache` are gathered at the given positions via `batched_index_select`, with shape
handling for both a flat `(T,)` and a batched `(B, T)` `input_pos`. If `input_pos_maxp1` is
also given, the mask's last dimension is sliced down to just cover the positions actually
in use, which speeds up attention by avoiding wasted computation over the full
`max_seq_length` buffer — the docstring notes that deriving this value automatically from
`input_pos` would cause graph breaks and block `torch.compile`, so callers are expected to
pass it explicitly when they already know it.

If `input_pos` is `None` (training mode), the RoPE caches are simply sliced to the first `T`
positions. **This is where the fork's block-mask logic hooks in**: if `self.use_block_mask`
is true, `construct_block_rope_feats(cos, sin)` is called to re-tile these RoPE caches into
the interleaved prefix/mask-token layout the MTP scheme expects (see §3), and the resulting
`self.block_mask_rope_feats` values replace `cos`/`sin` for the rest of the forward pass.
The dense `mask` variable is left `None` in this branch — the causal structure is instead
encoded entirely inside the FlexAttention block mask consulted later, inside each attention
layer.

After computing `cos`/`sin`/`mask`, token embeddings are looked up (with an optional
`sqrt(n_embd)` scale-up per `config.scale_embeddings`), then each `Block` is run in
sequence — passing per-layer RoPE slices if `config.rope_indices` is set, otherwise the same
`cos`/`sin` to every layer. After the final norm, logits are computed through `lm_head`,
optionally passed through Gemma-2-style softcapping (`do_softcapping`), and optionally
chunked along the sequence dimension (`lm_head_chunk_size`) to reduce peak autograd memory —
in which case a list of logit chunks is returned instead of a single tensor.

### `from_name` (304–306)

Thin convenience constructor: `GPT.from_name("some-preset")` builds the model from a named
`Config` preset.

### `rope_cache` (308–344)

Builds the RoPE `cos`/`sin` caches for the model's current `max_seq_length`. Most of the
logic here is about interpreting `config.rope_adjustments`: with zero adjustment parameters
present, standard RoPE is used; with all four Llama-3.1/3.2 parameters present
(`factor`, `low_freq_factor`, `high_freq_factor`, `original_max_seq_len`), the full
frequency-smoothing adjustment is applied; with just `factor` present, simple linear RoPE
scaling is used; any other partial combination raises a `ValueError` naming the missing
parameters, since the code considers these parameters valid only in the "all or nothing"
combinations above.

### `construct_block_rope_feats` (348–403, "Begin Gemini :] efficient version")

A fork-specific addition (the comment credits Gemini for the implementation) that re-tiles
the flat, linearly-indexed RoPE `cos`/`sin` caches to match the interleaved
prefix(`P`)-then-mask(`K`) region layout used by the MTP block mask, so that RoPE position
indices line up correctly with the block-mask's notion of which token belongs to which
region. It computes region boundaries via `torch.arange(0, S, P)`, builds a flat index array
tiling `region_width = P + K` positions per region across the sequence, then clamps and
truncates it to length `S`. When `offset` is nonzero, it applies a "roll and patch" — for a
negative offset it rolls the index array and back-fills the wrapped-around tail with a fresh
run of positions starting just past the last real mask-token position (with a `K == 0`
special case, since indexing with `-0` would otherwise wrongly select the first element
instead of one-past-the-end); for a positive offset it rolls the other direction and fills
the front with negative placeholder positions. The final index array is used with
`torch.index_select` (rather than plain advanced indexing, per the inline comment, for
performance) to produce the tiled `cos`/`sin` tensors stored in
`self.block_mask_rope_feats`.

### `set_kv_cache` and `clear_kv_cache` (405–440)

`set_kv_cache` allocates a `KVCache` per block (via each attention layer's own
`build_kv_cache`, since `CausalSelfAttention` and `MultiheadLatentAttention` size their
caches differently) and, if needed, rebuilds the shared dense causal `mask_cache` used only
during generation — the comment notes this dense mask is deliberately only built when a KV
cache is in play, since passing an explicit `attn_mask` to SDPA otherwise disables its flash
implementation. `clear_kv_cache` tears both back down to `None`.

---

## 3. `Block` (443–522) — one transformer layer

`Block` composes one attention sub-layer and one MLP sub-layer with residual connections,
and its topology is configurable to reproduce several different published architectures.
`config.latent_attention` picks between `CausalSelfAttention` (the default, and the one
wired up for block-mask/MTP support) and `MultiheadLatentAttention` (DeepSeek-style MLA).
Beyond that, four independent norm flags (`norm_1`, `norm_2`, `post_attention_norm`,
`post_mlp_norm`) and a `shared_attention_norm` flag control which normalization layers exist
and whether the attention and MLP branches read from the same normalized input — the
constructor raises `NotImplementedError` for the combination
`shared_attention_norm=True` with `parallel_residual=False`, since no supported checkpoint
architecture uses it.

The `forward` docstring (489–508) includes an ASCII diagram of the two supported residual
topologies. In the **non-parallel** case, attention output is added back to the residual
stream first, and the MLP branch reads from *that* updated stream (sequential: attn, then
MLP on the attn output). In the **parallel** case, both attention and MLP branches read from
the (possibly shared) normalized input independently, and their outputs are summed together
into the residual stream at the end — this is the GPT-NeoX/Pythia-style parallel block used
by several of the supported model families. The actual `forward` implementation (510–522)
follows this diagram directly, applying `post_attention_norm`/`post_mlp_norm` (identity
unless configured) to each branch's output before it's added back into the residual stream.

---

## 4. `CausalSelfAttention` (525–768) — GQA/MQA attention with FlexAttention support

This is the primary attention implementation and the one that actually participates in the
MTP block-mask scheme.

### `__init__` (526–559)

Builds a single combined `qkv` projection sized for grouped-query attention
(`n_head + 2*n_query_groups` head-equivalents), plus an output projection `proj`. QK-norm is
optional and supports two placements depending on `config.norm_qk_type`: `"olmo2"` norms the
full concatenated query/key vectors before they're split into heads, while the default norms
per-head after splitting. Sliding-window attention is enabled per-block based on
`config.sliding_window_indices[block_idx]`. The fork-specific additions here are
`self._attention_call = _sdpa_or_flex_attention()` — a dispatcher resolved once at
construction time that decides whether standard SDPA or FlexAttention will actually run —
and the three block-mask handles (`self.block_mask`, `self.use_block_mask`,
`self.block_mask_config`) passed in from `GPT`/`Block` construction.

### `forward` (561–690)

Follows the standard litgpt attention recipe, with GQA/MQA handled via the diagram at
579–594: the combined `qkv` projection is split into query/key/value along the last
dimension, each reshaped to separate out the head dimension, and (for the default QK-norm
placement) optionally normalized. RoPE is applied to only the first `rope_n_elem` dimensions
of each head via `apply_rope`, with the remainder (if any) left untouched and concatenated
back on. If `input_pos` is given, the KV cache is read/written and optionally truncated to
`input_pos_maxp1`. Grouped queries are then expanded to match the query head count via
`repeat_interleave` — the inline note explains this step is required for flash attention
during training but can be skipped for pure multi-query attention (`n_query_groups == 1`)
during single-token decoding, since broadcasting handles it more cheaply there. If sliding
window attention is active for this layer, a bias mask combining the global causal triangle
with a lower sliding-window triangle is constructed (diagram at 663–671) and added to
whatever mask is already in play. Finally `scaled_dot_product_attention` computes the
attention output, which is reshaped back to `(B, T, C)` and passed through the output
projection.

### `scaled_dot_product_attention` (692–735)

Three distinct code paths, chosen based on config:

1. **Softcapped attention** (`config.attention_logit_softcapping` set): computed manually as
   `q @ k.mT`, scaled, passed through `do_softcapping`, causal-masked (building a fresh
   triangular mask if none was passed in), softmaxed in float32 for stability, then applied
   to `v`. This path can't use the fused SDPA kernel because of the softcap.
2. **Block-mask / FlexAttention path** (the fork's main addition): if `self.block_mask` is
   still an un-materialized `partial` (per the deferred-construction design in `GPT`
   described in §2), it's called now, for the first time, to build the concrete mask on
   whatever device `q` currently lives on. The inline comment explains this materializes a
   *local, per-layer* copy of the mask rather than a single object shared across all
   layers — the ideal would be one shared mask, but FSDP is noted as making that difficult in
   practice, with a link to a torchtune module that apparently works around the same issue.
   An assertion guards against the invalid state `use_block_mask=True` with no mask
   available. `pass_block_mask` is then computed as `use_block_mask and block_mask is not
   None`, and `self._attention_call` (the SDPA-or-flex dispatcher from `attention_utils`) is
   invoked with either the block mask or the ordinary `mask` argument depending on that flag.
3. **Plain SDPA fallback**: effectively the same call as path 2 but without ever touching
   the block-mask machinery, used implicitly whenever `use_block_mask` is false.

### `build_kv_cache` (737–757)

Sizes a `KVCache` for GQA: the value shape uses `n_query_groups` (not the full head count,
since KV heads are shared across query groups), and the key shape additionally accounts for
`rope_cache_length` when RoPE only covers part of the head dimension
(`rotary_percentage != 1.0`), raising a `TypeError` if that case arises without an explicit
`rope_cache_length` being supplied.

### `_load_from_state_dict` (759–768)

A backward-compatibility shim: older checkpoints stored a single combined `attn.weight`/
`attn.bias` in a different layout than the current `qkv.weight`/`qkv.bias`. If a legacy key
is found in the incoming state dict, it's popped and rewritten into the current layout via
`qkv_reassemble` before deferring to the normal `nn.Module` loading logic.

---

## 5. `MultiheadLatentAttention` (771–908) — DeepSeek-V2/V3-style MLA

An alternative attention implementation, selected via `config.latent_attention`, that
compresses queries and keys/values through low-rank bottlenecks before attention rather than
projecting them directly. Queries go through `q_a_proj` → `q_a_norm` (an `RMSNorm`) →
`q_b_proj`, producing per-head vectors that are then split into a RoPE'd slice (`q_rot`,
sized `qk_rope_head_dim`) and a non-positional slice (`q_pass`, sized `qk_nope_head_dim`).
Keys and values share a single compressed projection, `kv_a_proj_with_mqa`, which is split
into a shared low-rank part (further expanded per-head via `kv_a_norm` → `kv_b_proj` into
both the key's non-positional part and the value) and a RoPE'd key part (`k_rot`) computed
with only one head and then broadcast (`.expand(...)`) across all query heads — this
broadcast is the "multi-query" aspect referenced in the projection's name. RoPE is applied
to the two rotary slices exactly as in `CausalSelfAttention`, then the rotary and
non-rotary halves are concatenated back together for both `q` and `k`.

The rest of the method — KV-cache read/write, GQA head-count balancing via
`repeat_interleave`, reshaping, and the output projection — mirrors
`CausalSelfAttention.forward` closely. The key structural difference from
`CausalSelfAttention` is that **this class does not participate in the block-mask/
FlexAttention scheme at all** — its own `scaled_dot_product_attention` (871–890) only
implements the softcapped-manual-attention path and a plain `F.scaled_dot_product_attention`
fallback; there is no `block_mask`/`use_block_mask` handling here. `build_kv_cache` (892–908)
sizes the cache using MLA-specific dimensions (`qk_head_dim`, `v_head_dim`) and prints
warnings if a caller passes `rope_cache_length` or a non-1.0 `rotary_percentage`, since
neither has any effect under this attention scheme.

---

## 6. MLP variants (911–974)

Four MLP implementations are provided, selected via `config.mlp_class`, all operating
independently per-token (no cross-token mixing):

- **`GptNeoxMLP`** (911–922): the simplest variant — one up-projection `fc`, a GELU
  activation (with `config.gelu_approximate` controlling the approximation mode), and a
  down-projection `proj`.
- **`LLaMAMLP`** (925–938): the gated SwiGLU variant used by Llama-family models — two
  parallel up-projections `fc_1`/`fc_2`, combined as `silu(fc_1(x)) * fc_2(x)`, then
  projected back down.
- **`GemmaMLP`** (941–946): subclasses `LLaMAMLP` and only overrides `forward` to use GELU
  instead of SiLU as the gating activation, reusing the same two-projection structure.
- **`LLaMAMoE`** (949–974): a sparse mixture-of-experts MLP, each expert itself a
  `LLaMAMLP` with its own `moe_intermediate_size`. A linear `gate` produces per-token router
  logits; `torch.topk` selects `n_expert_per_token` experts per token, whose scores are
  softmaxed (in float32) to produce combination weights. Rather than a dense
  all-experts-all-tokens computation, it builds a boolean routing mask per expert, gathers
  only the tokens routed to that expert via `torch.where`, runs just those tokens through the
  expert, and scatter-adds the weighted result back — the docstring credits the Mistral
  reference MoE implementation for this approach.

---

## 7. Free functions / shared utilities (977–1235)

- **`build_rope_cache`** (977–1049): computes the RoPE `cos`/`sin` caches from base
  principles — inverse frequencies `theta = 1/base^(2i/n_elem)`, optionally adjusted per
  `extra_config` (the Llama-3.1/3.2 frequency-smoothing scheme threaded in from
  `GPT.rope_cache`, blending low- and high-frequency scaling based on wavelength relative to
  `original_max_seq_len`), then combined with position indices (`seq_idx`, itself scaled by
  `condense_ratio`) via an outer product. Handles the odd-`n_elem` edge case by *not*
  truncating when `n_elem == 1`, citing a specific upstream HuggingFace issue where
  truncating in that case breaks parity with HF's implementation. If
  `rope_local_base_freq` is given, a second local-frequency cache is computed and stacked
  alongside the global one, supporting architectures with separate local/global attention
  RoPE bases.
- **`batched_index_select`** (1052–1068) and **`batched_index_copy_`** (1071–1118): generic
  batched gather/scatter helpers used to index the RoPE and mask caches by a per-batch
  `input_pos` during KV-cache decoding. `batched_index_copy_` has three code paths: an MPS
  device path (since MPS lacks the needed batched `index_copy_` support and instead uses
  `scatter_`), a fast `t.index_copy_` path when the index isn't batched, and a slow Python
  `for` loop over the batch dimension as the fallback for batched indices on other devices —
  the inline comment notes this fallback exists because the batch and index dimensions can't
  generally be viewed together for the KV cache's memory layout.
- **`apply_rope`** (1121–1150): applies the standard "rotate half" RoPE transform,
  `x*cos + rotate_half(x)*sin`, with shape-broadcasting logic to align `cos`/`sin`'s
  `(B, T, head_size)` shape against `x`'s extra head dimension.
- **`do_softcapping`** (1153–1154): `tanh(x/thresh) * thresh`, the Gemma-2 logit-softcapping
  formula, used both for attention scores and (via `GPT.forward`) final LM-head logits.
- **`KVCache`** (1157–1204): a small `nn.Module` wrapping two non-persistent buffers `k`,
  `v` of shape `(batch_size, n_query_groups, max_seq_length, head_size)`. Its `forward`
  writes new `k`/`v` slices in at `input_pos` via `batched_index_copy_` and returns the full
  (batch-sliced) buffers; it also silently upcasts the buffer dtype to match incoming
  activations, to stay correct under autocast/AMP.
- **`build_mask_cache`** (1206–1208): builds the dense lower-triangular boolean causal mask
  used during KV-cache decoding.
- **`RMSNorm`** (1211–1235): standard RMSNorm, computed in float32 internally for numerical
  stability regardless of the input dtype, with an optional `add_unit_offset` mode
  (`weight` interpreted as `1 + weight`) matching Gemma's normalization convention.

---

## 8. How the MTP block-mask thread ties together

Since this is the main fork-specific addition layered onto stock litgpt, it's worth tracing
end-to-end separately from the block-by-block report above. Four places must stay in sync if
this scheme is ever modified:

1. **`GPT.reconstruct_block_mask`** (§2) computes `P`/`K`/region layout from `hparams` and
   builds the deferred `partial(create_block_mask, ...)`, using a `mask_mod` function
   imported from `litgpt/mtp.py` (not defined in this file).
2. **`GPT.construct_block_rope_feats`** (§2), called from `GPT.forward` during training when
   `use_block_mask` is on, re-tiles the RoPE `cos`/`sin` caches so position indices line up
   with the interleaved prefix/mask-token region layout the mask itself encodes.
3. **`Block.__init__`/`CausalSelfAttention.__init__`** (§3, §4) receive the same
   `block_mask`/`use_block_mask`/`block_mask_config` objects from `GPT.__init__`, so every
   layer shares the same mask *configuration* (though, per point 4, not always the same
   materialized mask *object*).
4. **`CausalSelfAttention.scaled_dot_product_attention`** (§4) is where the deferred
   `partial` mask actually gets materialized — lazily, the first time that layer runs
   attention — and where the run-time decision between FlexAttention and standard SDPA is
   made via `self._attention_call`.

If you're debugging the MTP training path, the `mask_mod` predicate logic itself
(`interleaved_mtp_mask_mod_factory`) lives in `litgpt/mtp.py`, not here — this file only
handles wiring the mask into the model and keeping RoPE positions consistent with it.
