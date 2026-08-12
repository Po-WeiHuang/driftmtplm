# `litgpt/mtp.py` — Report

This is a small (94-line) file with a single piece of real content:
`interleaved_mtp_mask_mod_factory`, the function that builds the **mask predicate** consumed
by `torch.nn.attention.flex_attention.create_block_mask`. This is the actual masking-logic
implementation referenced throughout [model.md](model.md) (`GPT.reconstruct_block_mask`,
§2) and [pretrain.md](pretrain.md) (`truncate_and_mask`, §2) but never shown in full there —
this report covers it on its own. It implements the mask-mod scheme for the paper cited in
the file's docstring, "Multi-Token Prediction" (arXiv:2507.11851).

A **mask-mod** in FlexAttention terms is just a function `(b, h, q, k) -> bool` — given a
batch index, head index, query position, and key position, it returns whether that
query/key pair is allowed to attend. FlexAttention calls it (conceptually) for every
`(q, k)` pair and compiles the result into an efficient block-sparse mask; this file is
where that per-pair boolean logic lives for the "singleshot" MTP interleaved-region scheme.

---

## 1. Variable & argument glossary

| Name | Meaning |
|---|---|
| `prefix_length` (aliased `P` inside the function) | number of ordinary "prefix" (next-token-prediction, NTP) positions per region — same `P` as in [model.md](model.md)'s MTP glossary |
| `K` | number of mask-token ("MTP") positions per region — same `K` as in model.md |
| `B` (local to this file, **not** batch size!) | `prefix_length + K`, the total width of one region — careful: this shadows the usual "batch size" meaning of `B` used everywhere else in this codebase; here it is purely a region-width constant, and the mask-mod's own `b` parameter (lowercase) is the actual batch index |
| `offset` (aliased `Ofs`) | signed shift of where region boundaries fall, identical in meaning to `offset`/`Ofs` in `model.py`'s `reconstruct_block_mask`/`construct_block_rope_feats` and `pretrain.py`'s `truncate_and_mask` — must satisfy `abs(offset) < P` (asserted) |
| `bidirect_ss_attn` (aliased `BD_MTP`) | if true, relaxes strict causality in two specific ways (see §3) — intended for the student model only, never the teacher (enforced by callers, not by this file) |
| `pos` | a token position index (0-indexed along the sequence), input to the inner `block_idx` helper |
| `block_idx(pos)` | returns which **region** (0, 1, 2, ...) `pos` belongs to if it's a mask-token position, or `-1` if `pos` is an ordinary prefix position — this is the core position→region classifier the whole mask depends on |
| `b`, `h`, `q`, `k` | the four arguments FlexAttention's mask-mod contract requires: batch index, head index, query position, key position. Only `q`/`k` are actually used here — `b`/`h` are accepted (positionally required by the FlexAttention API) but ignored, meaning **the same mask applies to every batch row and every head** |
| `qb`, `kb` | `block_idx(q)`, `block_idx(k)` — the region membership of the query and key positions being tested |
| `causal` | `k <= q`, ordinary autoregressive causality |
| `q_is_ntp`, `k_is_ntp` | whether the query/key position is an ordinary prefix (next-token-prediction) position, i.e. `qb == -1` / `kb == -1` |
| `same_block` | whether query and key fall in the *same* region (`qb == kb`) — only meaningful when both are mask-token positions, since prefix positions are always `-1` and would trivially satisfy this against each other otherwise (the mask logic never actually relies on `same_block` for two prefix positions — see §3) |
| `allow_ntp` | whether this `(q, k)` pair is allowed under the "prefix attends to prefix" rule (plus the optional bidirectional lookahead extension) |
| `allow_mtp` | whether this pair is allowed under the "mask-token attends to prefix-or-own-region" rule (plus the optional bidirectional-within-region extension) |
| `allow` | `allow_ntp | allow_mtp` — the final boolean returned by `mask_mod` |

---

## 2. `interleaved_mtp_mask_mod_factory` (7–92) — structure

This is a **factory function**: it takes the region-shape parameters once (`prefix_length`,
`K`, `offset`, `bidirect_ss_attn`) and returns a closure, `mask_mod`, that FlexAttention will
call repeatedly per query/key pair. This mirrors how it's actually used in `model.py`
(`GPT.reconstruct_block_mask` calls this factory once per mask reconstruction, then passes
the returned `mask_mod` into `create_block_mask`) — the shape parameters are baked into the
closure once via Python closure variables (`B`, `P`, `Ofs`, `BD_MTP`) rather than being
re-passed on every call, which matters for how cheaply FlexAttention can compile/cache the
resulting block mask.

### `block_idx(pos)` (21–49) — position → region classifier

This is the piece worth understanding first, since both branches of `mask_mod` are built
entirely out of calls to it. Conceptually, without any `offset`, positions repeat the
pattern `[prefix]*P + [mask]*K` every `B = P+K` positions (exactly the region layout
diagrammed in [model.md](model.md)'s MTP worked example), and `block_idx` answers "which
repetition of that pattern is this position in, if it's inside the mask part; otherwise
`-1`":

1. **Shift by the offset** (28): `pos_shifted = pos - Ofs`. Subtracting `Ofs` (rather than
   adding) means a *negative* `offset` (as used by `pretrain.py`'s deterministic per-step
   sweep, `rolling_offset = -(step_count % P)`) shifts region boundaries *later* in absolute
   position terms — consistent with the "roll" logic in `model.py`/`pretrain.py`'s own offset
   handling, which this function does not duplicate but must stay compatible with.
2. **Decompose into remainder and region index** (33–34): `off = pos_shifted % B` (position
   within the repeating pattern) and `win = pos_shifted // B` (which repetition, i.e. which
   region index, using floored division — so this also does the right thing for the negative
   `pos_shifted` values produced by a positive offset).
3. **Classify** (40–49): a position is prefix (`-1`) either if its remainder puts it in the
   first `P` slots of the pattern (`off < P`) *or* if `pos_shifted` itself went negative
   (`is_before_offset`) — the comment explains this second case: positions
   `[0, offset - 1]` (only reachable when `offset > 0`) get folded into the very first
   region's prefix rather than being given a spurious negative region index, since there's no
   real "region -1" to speak of. Otherwise, the position is a mask-token position and its
   region index is `win`.

### `mask_mod(b, h, q, k)` (51–90) — the actual attention predicate

Given `qb`/`kb` from `block_idx`, `causal`, and the is-prefix flags, the function computes
two independent "allow" conditions and returns their union:

- **`allow_ntp`** (75–80, the "prefix attends to prefix" rule): baseline
  `q_is_ntp & k_is_ntp & causal` — an ordinary prefix position attends causally to any
  earlier (or equal) prefix position, **from any region**, exactly as in plain
  next-token-prediction pretraining; it is never allowed to attend to a mask-token position
  at all (that's what makes prefix tokens behave like normal NTP training data, unaffected
  by the MTP mask tokens sitting after them). When `bidirect_ss_attn` is on, an extra
  disjunct is added (discussed in §3) letting the *last* prefix position of a region peek
  forward into its own region's upcoming mask block.
- **`allow_mtp`** (82–87, the "mask-token attends to prefix-or-own-region" rule): baseline
  `(~q_is_ntp) & causal & (k_is_ntp | same_block)` — a mask-token query position may attend
  causally to *any* earlier prefix position (from any region, including its own) or to any
  causally-earlier position within its *own* region (including other mask-token positions in
  the same region, still causally ordered by default). It may never attend to a *different*
  region's mask-token positions — that's the mechanism that keeps each region's MTP
  prediction independent of every other region's. When `bidirect_ss_attn` is on, a second
  extra disjunct (§3) makes attention *within* the region's mask block bidirectional instead
  of causal.
- **`allow = allow_ntp | allow_mtp`** (89): since `q_is_ntp` and `~q_is_ntp` are mutually
  exclusive, exactly one of the two base conditions is even eligible to fire for a given
  query, so this union just dispatches to whichever rule applies to the query's own
  position type.

---

## 3. The `bidirect_ss_attn` (`BD_MTP`) extension

This flag adds two specific relaxations on top of the strict-causal rules above, both
implemented via the extra `q_plus_1`-based terms at lines 62–72 and the `|` clauses at 80 and
87. It's gated everywhere by `BD_MTP` directly inside the boolean expressions (rather than an
`if` branching to two different functions), so when `bidirect_ss_attn=False` the extra terms
are computed as `False`-producing placeholders (71–72) and the `BD_MTP & ...` disjuncts
short-circuit away — the mask reduces exactly to the baseline rules from §2.

1. **Last-prefix-token lookahead** (extends `allow_ntp`, line 80): a query at the very last
   prefix position of a region would, under strict causality, never be allowed to see any
   token of the mask block that immediately follows it (mask-token positions are never
   `k_is_ntp`). This extension checks whether `q+1` is itself a mask-token position in the
   *same upcoming region* the query is about to hand off to (`qb_plus_1_same_block`, i.e.
   `block_idx(q+1) == kb`) and, if so, allows `q` to attend to that key too — letting the
   model's very last real-token representation before a region's mask block already start
   incorporating information from that block.
2. **Bidirectional mask-block attention** (extends `allow_mtp`, line 87): drops the `causal`
   requirement entirely for two mask-token positions in the *same* region — so instead of
   only being able to see earlier mask-token predictions in its own region, a query position
   can see *all* mask-token positions in its region, both earlier and later. This is what
   "bidirectional" refers to: within one region's block of `K` predicted tokens, they can all
   attend to each other freely, rather than being forced into a left-to-right causal chain
   the way ordinary autoregressive generation is.

As documented in [model.md](model.md), this flag is meant for the **student model only** —
`GPT.reconstruct_block_mask` prints a warning if a teacher model is ever configured with it
on, and every call site in `pretrain.py` explicitly passes `bidirect_ss_attn=False` when
reconstructing the teacher's mask.

---

## 4. Tiny worked example

Reusing the exact same numbers as [model.md](model.md)'s MTP worked example, for
continuity: `P = 4`, `K = 2` (so `k_toks = 3`), `offset = 0`, giving `B = P + K = 6` and a
12-position sequence made of two regions:

```
position:   0  1  2  3  4  5  6  7  8  9 10 11
content:   [p  p  p  p  m  m][p  p  p  p  m  m]
region:      └── region 0 ──┘  └── region 1 ──┘
block_idx:  -1 -1 -1 -1  0  0 -1 -1 -1 -1  1  1
```

(`block_idx` computed directly from §2's formula: positions 0–3 and 6–9 have
`off = pos % 6 < 4`, so they're prefix (`-1`); positions 4,5 have `off=4,5`, `win=0`, so
region `0`; positions 10,11 have `off=4,5`, `win=1`, so region `1`.)

**Strictly causal case (`bidirect_ss_attn=False`).** Reading off `allow_ntp`/`allow_mtp` for
a few representative queries:

- `q=3` (last prefix token of region 0, `qb=-1`): `allow_ntp` fires for any `k <= 3` that is
  also prefix — so `k ∈ {0,1,2,3}`. It can **not** see `k=4` or `k=5` (region 0's own mask
  tokens) at all, since `allow_ntp` requires `k_is_ntp` and `allow_mtp` requires `q` to
  *not* be prefix. Ordinary prefix tokens are entirely blind to any region's mask tokens,
  including their own region's.
- `q=4` (first mask token of region 0, `qb=0`): `allow_mtp` fires for `k_is_ntp & k<=4` (so
  all of `k ∈ {0,1,2,3}`) or `same_block & k<=4` (`k=4` itself). So `q=4` sees the full
  region-0 prefix plus itself — not `k=5` (fails causal), and not any region-1 position
  (region-1 prefix positions 6–9 all fail `k<=4`).
- `q=10` (first mask token of region 1, `qb=1`): `allow_mtp` fires for `k_is_ntp & k<=10`
  (all prefix positions from *both* regions: `{0,1,2,3,6,7,8,9}`) or `same_block & k<=10`
  (`k=10` itself). Notably it does **not** see `k=4` or `k=5` — region 0's mask tokens —
  since those fail `k_is_ntp` and fail `same_block` (`kb=0 != qb=1`). This is the key
  region-isolation property: region 1's predictions can see everything's ordinary prefix
  history, but never region 0's actual (predicted) mask-token content.
- `q=11` (second mask token of region 1): same as `q=10` plus `k=10` now also allowed
  (`same_block`, and now `k<=11` holds).

**With `bidirect_ss_attn=True`,** two things change for the same layout:

- `q=3` (last prefix token of region 0) gains visibility into `k=4` and `k=5`: since
  `block_idx(q+1) = block_idx(4) = 0`, which equals `kb=0` for both `k=4` and `k=5`, the
  extra `allow_ntp` disjunct fires — so the last prefix token can now "peek" at both upcoming
  mask-token positions of its own region, not just attend causally within the prefix.
- `q=4` (first mask token of region 0) gains visibility into `k=5` (previously blocked by
  `causal`): the extra `allow_mtp` disjunct drops the causal requirement for `same_block`
  pairs, so within region 0's 2-token mask block, position 4 and position 5 can now see each
  other in both directions — the block of predicted tokens becomes fully bidirectional among
  themselves, while remaining causal with respect to everything *outside* their own region.

This is exactly the mechanism [model.md](model.md) and [pretrain.md](pretrain.md) reference
without spelling out: `bidirect_ss_attn` doesn't change *which* regions or prefixes are
visible (that's still governed by `block_idx`/`same_block`/`k_is_ntp` the same way) — it only
loosens the *ordering* constraint (`causal`) among positions that were already going to be
in-scope for each other, and only in the two specific places described in §3.
