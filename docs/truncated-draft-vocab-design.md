# Wave 7 research: truncated draft vocabulary for the draft-mtp path

Researched 2026-08-17 against llama.cpp `master` = `4df29be4f`, the production patch stack
`0001..0006`, PR #26275 (`gh pr diff 26275`), and the two deployed GGUFs on this host.

**Verdict: GO, with one measurement gate first.** Draft logits for `--spec-type draft-mtp` go
through the *drafter's own* `output.weight` (Q4_0, 0.715 GB, 74% of every draft pass), so
truncating its rows removes real bytes on a bandwidth-bound workload. The mechanism needed
already exists at our pin — it is the EAGLE3 `d2t` scatter — so patch 0007 is a fourth copy of
an established upstream idiom, ~22 lines in one file, with **zero changes to
`common/speculative.cpp`** and no interaction with patches 0001/0006. It also reverts
independently of the model file, which is the strongest property in the design (5f).

Two honest caveats, in order of how much they move the number:

1. **Byte arithmetic gives a −13.5% cycle-traffic ceiling, but a draft pass is not purely
   bandwidth-bound.** Fixed per-pass overhead that truncation cannot touch may pull the
   realistic ceiling to ~7% before any acceptance loss. **Measure the per-draft-step overhead
   before doing the surgery** (5d.0) — one run, ~15 lines of throwaway instrumentation.
2. The remaining risk is not the plumbing; it is whether the chosen 40k token set covers what
   the drafter actually proposes on our traffic (5e).

Three corrections to the briefing are load-bearing and appear first.

---

## 0. Corrections to the briefing

### 0a. The greedy fast path (patches 0001/0006) does NOT apply to qwen3.8-27b

Patches 0001 and 0006 modify `struct common_speculative_impl_draft_dflash`
(master `common/speculative.cpp:908`). Production's qwen3.8-27b block runs
`--spec-type draft-mtp` (`k8s/workloads/apps/llama/configmap.yaml:115`), which instantiates
`common_speculative_impl_draft_mtp` (`common/speculative.cpp:1274`, dispatched at
`common/speculative.cpp:2492-2493`). These are two disjoint structs. The `greedy_fast_path`
flag added by 0001 and refined by 0006 lives at the DFlash sampling loop only and is never
reached from the MTP driver.

Patches 0001/0006 therefore apply to **muse-glimmer-30b** (the DFlash drafter), not to
qwen3.8. Every question in the briefing framed as "how does this interact with the greedy
fast path" has the same answer for qwen3.8: it does not.

Secondary consequence: production's sampler is `--temp 1.0 --top-k 20 --top-p 0.95`
(`configmap.yaml:113`), so even on the DFlash path the greedy fast path would be irrelevant
here. Temp-0 argmax ties are a non-issue for this wave.

### 0b. `is_mem_shared` is false for qwen35 in *both* drafter configurations

`common/speculative.cpp:2384` sets `cparams.ctx_other = ctx_tgt` unconditionally, which
reads as though `is_mem_shared` (`common/speculative.cpp:1367`,
`llama_get_ctx_other(ctx_dft) == ctx_tgt`) would be true for us. It is not.
`src/llama-context.cpp:141` resets `cparams.ctx_other = nullptr`, and only two arches
restore it:

- `src/llama-context.cpp:144-151` — `LLM_ARCH_GEMMA4_ASSISTANT`
- `src/llama-context.cpp:153-160` — `LLM_ARCH_EAGLE3` / `LLM_ARCH_DFLASH`, and only when the
  draft model carries neither `tok_embd` nor `output` of its own

Our drafter and target are both `arch=qwen35` (verified in the GGUF headers below), so
`ctx_other` stays `nullptr` and `is_mem_shared` is **false** in both the separate-drafter and
embedded-drafter configurations. `chain_heads` is also false in both, since
`n_layer_nextn == 1` for qwen35 (`src/models/qwen35.cpp:492` asserts exactly one MTP block).

This matters because `is_mem_shared` selects a different drafting *algorithm*, not just a
different byte count: it skips the catch-up decode (`common/speculative.cpp:1462`) and places
every draft token at the same position instead of `n_past + i + 1`
(`common/speculative.cpp:1666`). Both configurations take the same branch, so the
embedded-vs-separate comparison in section 2 is a pure byte comparison — no algorithmic
confound.

### 0c. The target model already loads the embedded MTP block, and it is dead weight

`common/common.cpp:1689` sets `mparams.load_mtp = true` whenever `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`
is in `params.speculative.types` — which it is, from `--spec-type draft-mtp`. That flag gates
the `blk.64.nextn.*` tensors at `src/models/qwen35.cpp:42`
(`mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0`), and it is applied to the **target** model load.

So today, with `--model-draft` supplied, `Qwen3.8-27B-Q4_K_L.gguf` loads its embedded MTP
block (425M Q4_0 params ≈ 0.24 GB) into VRAM, and never executes it: the target context is
`LLAMA_CONTEXT_TYPE_DEFAULT`, whose graph is selected at `src/models/qwen35.cpp:131`
(`graph_mtp` only for `LLM_GRAPH_TYPE_DECODER_MTP`), and the main pass explicitly skips MTP
layers (`src/models/qwen35.cpp:158`).

That is ~0.24 GB of reclaimable VRAM, not speed. Out of scope for this wave; filing it as a
separate observation.

---

## 1. Draft logits data flow for `--spec-type draft-mtp`

### Which tensor produces draft logits

**The drafter model's own `output.weight`.** The chain:

1. `common_speculative_init_result` loads the drafter as its own `llama_model` and builds a
   context with `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP`
   (`common/speculative.cpp:2371-2373`, `2391`, `2399`).
2. `LLAMA_CONTEXT_TYPE_MTP` maps to `LLM_GRAPH_TYPE_DECODER_MTP`
   (`src/llama-context.cpp:30`), and that graph type is used for **every** decode on that
   context (`src/llama-context.cpp:1816`, `2426`). The drafter context can never build the
   normal decoder graph.
3. `src/models/qwen35.cpp:131-133` routes `LLM_GRAPH_TYPE_DECODER_MTP` to
   `llama_model_qwen35::graph_mtp`.
4. The head is chosen at **`src/models/qwen35.cpp:637-638`**:

   ```cpp
   ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
   ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
   ```

   followed by `cur = build_lora_mm(head_w, cur, head_s);` (`:640`), `res->t_logits = cur` (`:643`).

5. Tensor dump of `mtp-Qwen3.8-27B-Q4_0.gguf` (18 tensors, `arch=qwen35`) shows
   **no `blk.64.nextn.shared_head_head.weight`** and no `blk.64.nextn.embed_tokens.weight`.
   Present nextn tensors: `eh_proj` (Q4_0), `enorm`, `hnorm`, `shared_head_norm` (all F32).

   So `head_w = model.output` = the drafter's own `output.weight`, **Q4_0, 5120×248320,
   0.715 GB**, read in full on every draft step.

Input embeddings take the parallel branch at `src/models/qwen35.cpp:521-524`:
`tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd`, then
`ggml_get_rows`. With no `nextn.embed_tokens`, that is the drafter's own `token_embd.weight`
(Q4_0, 0.715 GB), gathered by row — cheap, and **indexed by full-vocab target token id**
(see section 5a).

### Where draft sampling happens

Per-step, in `common_speculative_impl_draft_mtp::draft()`:

- `common/speculative.cpp:1614` — `common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true)`
- `common/speculative.cpp:1617` — `common_sampler_get_candidates(smpl, true)`
- `common/speculative.cpp:1626` — `const llama_token id = cur_p->data[0].id;` (argmax; the
  drafter always takes the top candidate regardless of the user's sampler)
- `common/speculative.cpp:1629` — `if (cur_p->data[0].p < params.p_min)` early stop.
  **Production has no `--spec-draft-p-min`**, so `p_min == 0` and this never fires; every
  cycle runs the full `n_max = 5` draft depth.

CPU or GPU: **GPU by default.** `common/common.h:331` — `bool backend_sampling = true; //
offload draft sampling to the backend (default: on)`. The MTP ctor acts on it at
`common/speculative.cpp:1350-1362`, attaching a `top_k(10)` `llama_sampler` chain to `ctx_dft`
via `llama_set_sampler`. Production does not pass `--no-spec-draft-backend-sampling`
(`configmap.yaml:101-118`), so the draft top-k runs on the GPU.

The host still calls `common_sampler_sample`, but with backend sampling active it takes the
short path: `common/sampling.cpp:607` calls `set_logits`, which at
`common/sampling.cpp:140-151` sizes the candidate array to the **backend's top-k count**
(10), not `n_vocab`; then `common/sampling.cpp:612-628` returns the backend-selected token
directly. So the host is *not* building a 248k-entry candidate array per draft step today.
(This is exactly why patch 0006 had to disable the DFlash greedy fast path under backend
sampling: `llama_get_logits_ith` is not a usable full-vocab row in that mode.)

### Where draft ids are consumed by the verify path

- `tools/server/server-context.cpp:2934-2941` — the server hands `&slot.spec_draft` to
  `common_speculative_get_draft_params(...)` as the output buffer, then calls
  `common_speculative_draft(spec.get())` (`:2952`).
- `tools/server/server-context.cpp:482-497` — drafted tokens are appended to the target
  batch alongside the sampled token.
- **`tools/server/server-context.cpp:3819`** — verification:
  `common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft)`.
- `tools/server/server-context.cpp:3880-3888` — acceptance counters, including the
  per-position histogram printed at `:614-634`.

---

## 2. Embedded vs separate drafter — answers the parked `PERFORMANCE.md` item

**Verdict: dropping `--model-draft` is a SPEED LOSS. Do not do it.**

When `--model-draft` is absent but the MTP type is active,
`common/speculative.cpp:2405-2412` takes the `else if (spec_mtp)` branch and builds the draft
context **on `model_tgt` itself**: `llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);`.
There is no second model. `model.output` in `graph_mtp` (`src/models/qwen35.cpp:637-640`) is
therefore the **target's** `output.weight`.

Tensor dumps (pure-Python GGUF header parse, `scratchpad/ggufinfo.py`):

| | `mtp-Qwen3.8-27B-Q4_0.gguf` | `Qwen3.8-27B-Q4_K_L.gguf` |
|---|---|---|
| arch | qwen35 | qwen35 |
| `output.weight` | **Q4_0, 0.715 GB** | **Q8_0, 1.351 GB** |
| `token_embd.weight` | Q4_0, 0.715 GB | Q8_0, 1.351 GB |
| `blk.64.nextn.eh_proj.weight` | Q4_0, 0.029 GB | Q4_0, 0.029 GB |
| `blk.64.nextn.shared_head_head` | **absent** | **absent** |
| `blk.64.nextn.embed_tokens` | **absent** | **absent** |
| Q4_0 param total | 2967M (= 2542M embeddings + 425M body) | 425M (the MTP block) |

The MTP *body* is byte-identical in both files: 425M Q4_0 params ≈ 0.24 GB. The algorithm is
identical too (section 0b). The only difference is the head, and the token_embd row gather.

Per draft pass:

| Configuration | body | head | total |
|---|---|---|---|
| Separate Q4_0 drafter (current) | 0.25 GB | 0.715 GB | **0.965 GB** |
| Embedded (drop `--model-draft`) | 0.25 GB | 1.351 GB | **1.601 GB** |

At `n_max = 5`, per verify+draft cycle (target verify pass 17.37 GB, from the background
report's dump):

| Configuration | cycle traffic | vs current |
|---|---|---|
| Current (separate Q4_0 drafter) | 17.37 + 5×0.965 = **22.20 GB** | — |
| Embedded | 17.37 + 5×1.601 = **25.37 GB** | **+14.3%** |

It buys ~1.68 GB of VRAM (the drafter file) at a cost of roughly 10–13% decode throughput on a
bandwidth-bound workload. `PERFORMANCE.md`'s "no speed change" is a prediction, and it is
wrong; the unverified risk flagged in the background report (section 3, "if the embedded path
computes draft logits through the *main* model's Q8_0 `output.weight`") is **confirmed**.

A further reason this closes the item permanently: the embedded head *is* the target's
`output.weight`, used by the main verify pass. It can never be truncated. Truncation
(section 5) is only available on the separate-drafter path. The two ideas are mutually
exclusive and truncation is worth far more.

---

## 3. PR #26275 (DSpark speculators, `9cd719af2`) — and why we do not need it

### How its d2t works

- **Storage**: a tensor named `d2t`, dtype **I64**, shape `[n_vocab_draft]`, holding
  **absolute target token ids**. The converter reads HF's delta-encoded `model.d2t` and adds
  `arange` to absolutize it: `data = data + np.arange(data.size, dtype=np.int64)`
  (`pr26275.diff:183`). It validates range (`:184-185`) and uniqueness (`:186-187`) and
  raises on violation. There is **no metadata KV key** for the mapping — the tensor's own
  `ne[0]` carries the draft vocab size. (`{arch}.sample_from_anchor`, added by the same PR at
  `:213`, is unrelated to d2t; it is a DSpark block-layout flag.)
- **Load**: `src/models/dflash.cpp`, `pr26275.diff:278-284` —
  `ml->get_tensor_meta("d2t")`; if present, `n_vocab_draft = d2t_meta->ne[0]` and
  `d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0)`. `output` is then created
  with `{ n_embd, n_vocab_draft }` (`:303`).
- **Remap**: **in the graph, immediately after the LM head matmul** (`:352-368`). It
  allocates a full `[1, n_vocab, n_outputs]` F32 tensor filled with `-INFINITY`, then
  `ggml_set_rows` scatters the reduced logits into the rows named by `d2t`, then reshapes back
  to `[n_vocab, n_outputs]`.
- **Sampling over the reduced head**: there is none. By the time any sampler sees the tensor,
  it is full-width `n_vocab` with `-inf` on unmapped rows. Every downstream consumer —
  `llama_get_logits_ith`, the CPU sampler chain, the backend sampler chain, the logits buffer
  sizing in `llama_context` — is unchanged, and draft ids come out already in target-vocab
  space.
- The PR also scatters the DSpark Markov-head *bias* the same way (`:337-345`), which is
  DSpark-specific and irrelevant to us.

### Is it reusable for draft-mtp?

**Not directly — but we do not need it.** The scatter in #26275 lives in `src/models/dflash.cpp`,
inside `llama_model_dflash::graph<false>::graph`, and its loader half is in
`llama_model_dflash::load_arch_tensors`. Neither is reachable from `arch=qwen35`.

The important finding is that **#26275 is not the origin of d2t — it is the third copy of it.**
The mechanism is already at our pin `4df29be4f`, shipped for EAGLE3:

- `src/llama-model.h:641` — `struct ggml_tensor * d2t = nullptr;  // draft to target vocabulary mapping`
- `src/llama-arch.h:649` / `src/llama-arch.cpp:641` — `LLM_TENSOR_D2T` → `"d2t"`
- `src/llama-arch.cpp:905` — `{LLM_TENSOR_D2T, {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_GET_ROWS}}`
- `gguf-py/gguf/constants.py:1128`, `:1874` — `MODEL_TENSOR.D2T` / `"d2t"`
- `src/models/eagle3.cpp:44-55` — the loader half
- `src/models/eagle3.cpp:318-331` — the graph scatter, textually identical to #26275's

So patch 0007 is a **fourth copy of an in-tree idiom into `src/models/qwen35.cpp`**, not a new
mechanism. Do not vendor #26275: it adds DSpark converter work and a `sample_from_anchor` flag
we have no use for, and its textual overlap with our patch 0006 is confined to the DFlash
`draft()` region, which patch 0007 does not touch at all. **Patch 0007 conflicts with nothing
in the 0001–0006 stack** — it is the only patch in the stack that touches `src/models/qwen35.cpp`.

---

## 4. Correctness under reduced draft support

### The accept/reject algorithm is not rejection sampling

llama.cpp does **not** implement standard speculative rejection sampling with residual
correction. `common_sampler_sample_and_accept_n` (`common/sampling.cpp:680-706`) does this:

```cpp
for (i = 0; i < draft.size(); i++) {
    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
    common_sampler_accept(gsmpl, id, true);
    result.push_back(id);
    if (draft[i] != id) {
        break;
    }
}
```

It samples from the **target's** distribution at every position and keeps the draft token only
when it exactly equals what the target sampled. The emitted token is the target's token in
every case — the draft token is never emitted; it is only compared.

**Therefore the output distribution is the target's by construction, not by a correction
argument.** The drafter's support cannot leak into outputs, because no drafter token is ever
an output token. This is a *stronger* guarantee than the residual-correction framing in the
background report (section 3), and it applies uniformly to our production sampler
(temp 1.0 / top-k 20 / top-p 0.95), to temp 0, and to any `p_min`.

Consequences that materially shrink the verification burden:

- **A wrong d2t mapping cannot corrupt output.** It can only collapse acceptance, which
  surfaces as a speed regression. The acceptance-per-position histogram
  (`tools/server/server-context.cpp:614-634`) is a cheaper and more sensitive bug detector
  than a KLD battery.
- **The greedy/temp-0 tie concern is moot** for qwen3.8, per section 0a: the fast path is
  DFlash-only.
- **`p_min` is irrelevant**: production has none, so the confidence early-stop at
  `common/speculative.cpp:1629` never fires, and no reduced-support probability is ever
  compared against a threshold.
- **The `-inf` rows are inert.** They are consumed only by the *draft* sampler (`top_k(10)` on
  `ctx_dft`), which is choosing among the drafter's own candidates. They never reach the
  target sampler.
- **The vocab-compatibility check is unaffected.** `common/speculative.cpp:101-120` compares
  `llama_vocab_n_tokens` and per-id token text — i.e. the *tokenizer* arrays, which the
  surgery leaves untouched at 248,320. It never inspects `output.weight`'s row count.

### The one real safety requirement

`ggml_set_rows` with an out-of-range index is an **out-of-bounds write**. #26275's converter
validates range and uniqueness (`pr26275.diff:184-187`) and `src/models/eagle3.cpp:324-325`
asserts dtype and length. Our surgery script must do the same. That check is non-optional —
unlike the quality battery, which is a nice-to-have.

Residual honest caveat: the *sequence* of tokens will differ from a no-speculation run,
because rejected positions consume extra RNG draws from the sampler. That is already true
today with any drafter and is a property of speculative decoding in llama.cpp, not of vocab
truncation. Distributional exactness holds; bit-identical replay against a non-speculative run
does not, and never did.

---

## 5. Design: patch 0007

### 5a. GGUF surgery on `mtp-Qwen3.8-27B-Q4_0.gguf`

**Truncate `output.weight` only. Do NOT truncate `token_embd.weight`.**

The reason is at `src/models/qwen35.cpp:521-524`: the MTP input embedding is
`ggml_get_rows(tok_embd_w, inp->tokens)`, and `inp->tokens` are the *real* batch token ids —
`dp.id_last` (the token the **target** just sampled, which can be any of the 248,320) and the
previously drafted ids (`common/speculative.cpp:1567`, `1666`). Truncating `token_embd` would
require remapping inputs through an inverse t2d, which is impossible for target-sampled tokens
outside the 40k set. Keep `token_embd.weight` at full 248,320 rows. It is a row gather, so it
costs ~2.9 KB per token, not 0.715 GB — there is nothing to win there anyway.

Truncation is a **lossless byte copy, no requantization**: a Q4_0 row of 5120 elements is
exactly 160 blocks of 32 (18 bytes each) = 2880 bytes, so selected rows copy whole.
40,000 rows × 2880 B = 115.2 MB, down from 715.2 MB.

Store the mapping as the existing `d2t` tensor — **reuse the EAGLE3/#26275 key, do not invent
a new one**: name `"d2t"`, dtype I64 (`GGML_TYPE_I64`), shape `[n_draft_vocab]`, values =
**absolute target token ids**, sorted ascending (not required by the code, but it makes the
file diffable and the uniqueness check trivial). 40,000 × 8 B = 320 KB.

Script requirements:

1. Parse the source GGUF header (the existing `scratchpad/ggufinfo.py` already does this).
2. Build the keep-set (see 5e), sort it, assert `0 <= id < 248320`, assert no duplicates,
   assert `len(set) == n_draft_vocab`.
3. Emit a new GGUF: identical KV block (tokenizer arrays untouched), tensor list with
   `output.weight` re-declared as `5120 × n_draft_vocab` and a new `d2t` I64 tensor appended.
4. Copy every unchanged tensor's bytes verbatim; copy the selected `output.weight` rows in
   keep-set order.
5. **Respect `general.alignment`** when computing tensor data offsets (default 32; read it
   from the source file rather than assuming).

Tooling note: `numpy` is not installed on this host, so `gguf-py`'s writer will not run as-is.
`pip` (26.1.2) and `venv` both work, so `python3 -m venv` + `pip install numpy` is the path of
least resistance; extending the existing pure-Python parser into a rewriter is the fallback and
avoids the dependency entirely.

Keep the original file. The fallback is a one-line configmap revert (5f).

### 5b. llama.cpp changes

**One file: `src/models/qwen35.cpp`. No changes to `common/speculative.cpp`, none to the
server, none to the sampler.**

*(i) Loader* — in `llama_model_qwen35::load_arch_tensors`, before the existing
`output` creation at `src/models/qwen35.cpp:48`, mirroring `src/models/eagle3.cpp:47-55`:

```cpp
int64_t n_vocab_draft = n_vocab;
const struct ggml_tensor * d2t_meta = ml.get_tensor_meta("d2t");   // NOTE: dot, not arrow — see below
if (d2t_meta) {
    n_vocab_draft = d2t_meta->ne[0];
    d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
    LLAMA_LOG_INFO("%s: QWEN35 using d2t mapping (draft_vocab_size = %lld)\n",
                   __func__, (long long) n_vocab_draft);
}
```

then change `src/models/qwen35.cpp:48` to use `n_vocab_draft` in place of `n_vocab`. The
target model has no `d2t` tensor, so `n_vocab_draft == n_vocab` there and its behaviour is
bit-identical. ~8 lines.

**Transcription trap — do NOT copy eagle3's `ml->` verbatim.** `src/models/eagle3.cpp:38`
leaves its `llama_model_loader &` parameter *unnamed*, so `ml` at `eagle3.cpp:47` resolves to
the **member pointer** `llama_model_loader * ml` (`src/llama-model.h:747`) and correctly uses
`ml->`. `src/models/qwen35.cpp:36` *names* its parameter `ml`, which shadows that member, so
inside `llama_model_qwen35::load_arch_tensors` you must write **`ml.get_tensor_meta("d2t")`**
with a dot — matching the existing `ml.get_weight(...)` and `ml.load_mtp` at
`src/models/qwen35.cpp:39-41`. This is the one line that will not compile if transcribed
loosely, and the arrow form fails with a confusing error.

*(ii) Graph* — in `llama_model_qwen35::graph_mtp`, immediately after
`cur = build_lora_mm(head_w, cur, head_s);` (`src/models/qwen35.cpp:640`) and before
`cb(cur, "result_output", -1)`, copy `src/models/eagle3.cpp:318-331` verbatim:

```cpp
if (model.d2t) {
    const int64_t n_draft_vocab = cur->ne[0];
    const int64_t n_outputs     = cur->ne[1];
    const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

    GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
    GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

    ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
    cur = ggml_set_rows(ctx0, logits,
            ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
            ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
    cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
}
```

~14 lines. **Total patch 0007: ~22 lines in one file.**

*(iii) One-line guard.* The normal qwen35 decoder graph also uses `model.output`
(`src/models/qwen35.cpp:223`) without a scatter. A truncated GGUF can never reach it (the
drafter is `mtp_only`, so its trunk tensors are `TENSOR_NOT_REQUIRED` at
`src/models/qwen35.cpp:40-41`, and its context is always `LLM_GRAPH_TYPE_DECODER_MTP` per
`src/llama-context.cpp:1816`, `2426`). Add `GGML_ASSERT(!model.d2t)` there anyway so the
invariant fails loudly rather than emitting silently-truncated logits.

*(iv) Interactions.*

- **Backend sampling (on by default):** no change needed. The scatter happens in-graph, so the
  backend `top_k(10)` chain on `ctx_dft` (`common/speculative.cpp:1350-1362`) sees a
  full-width row and returns target-vocab ids. The `-inf` rows lose top-k cleanly.
- **Patches 0001/0006:** no overlap. Different struct, different file (section 0a, section 3).
- **Cost of the scatter:** per draft step, `ggml_fill` writes 248,320 × 4 B = 0.993 MB and
  `ggml_set_rows` writes 40,000 × 4 B = 0.16 MB. At `n_max = 5` that is ~5.8 MB per cycle
  against a ~19.2 GB cycle — **0.03%**. Negligible; do not optimize it.

*(v) Deliberately deferred: the "sample in reduced space, remap on host" variant.* It would
avoid materializing the full-width row and could cut the draft sampler's work ~6×, but with
backend sampling on, `set_logits` already takes the top-k-candidates branch
(`common/sampling.cpp:140-151`) and the host never builds a 248k candidate array — so the win
is much smaller than it first appears, and it costs the "zero downstream changes" property
that makes v1 safe. Ship the scatter; revisit only if profiling shows draft sampling on the
critical path.

### 5c. Expected gain

Per draft pass: 0.965 GB → 0.25 (body) + 0.115 (40k head) = **0.365 GB**.

At `n_max = 5`:

| | cycle traffic | delta |
|---|---|---|
| Current | 17.37 + 5×0.965 = 22.20 GB | — |
| 40k draft head | 17.37 + 5×0.365 = 19.20 GB | **−13.5%** |

That is the ceiling, realized only if acceptance is unchanged. syv-ai lost ~10 points of
acceptance with a combined int8-drafter + truncated-vocab change, and lost acceptance directly
eats the saving because rejected draft positions are pure waste.

**But that ceiling is itself optimistic, because a draft pass is not purely bandwidth-bound.**
The background report measures ~660 GB/s effective, i.e. ~70% of the 3090's 936 GB/s peak, so
~30% of cycle time is not byte-moving — and the draft pass is the most exposed part of it. At
peak bandwidth, 0.965 GB is only ~1.03 ms of theoretical time, against which the per-pass
kernel launches, the `llama_decode` graph build/dispatch, the `h_row` memcpy
(`common/speculative.cpp:1615`, `:1668`), and the sampling round-trip are **fixed costs that
truncation cannot touch**.

Worked example: if the measured per-draft-step time is ~2 ms, then ~1 ms is fixed overhead,
and removing 0.6 GB buys ~30% of a draft step rather than the 62% the byte ratio implies —
pulling the ceiling from 13.5% down toward **~7% before any acceptance loss**. That is the
difference between "clearly worth ~250–400 LOC" and "possibly ~3%, reconsider."

**Measure this before doing the surgery** (see 5d.0). It is the single highest-value check
left in this wave and it costs one run.

Decision rule to agree on before benching: if acceptance falls more than a few points, the
change is net-negative and should be reverted rather than tuned.

### 5d. What must be measured

0. **FIRST, before any surgery: the fixed-overhead fraction of a draft step.** Compare the
   measured per-draft-step wall time against the 1.03 ms theoretical byte time (5c). If
   overhead dominates, the whole wave is worth ~3% and should be reconsidered.

   Patch 0001 already built the instrumentation pattern (`LLAMA_SPEC_PROF`,
   `spec_prof_counter`, the per-phase dump) but wired it **only into the DFlash driver** — the
   MTP driver has no equivalent. Two options: add the same two timers around
   `llama_decode(ctx_dft, batch)` (`common/speculative.cpp:1596`) and the sampling block
   (`common/speculative.cpp:1612-1630`), which is a ~15-line throwaway following 0001's
   existing shape; or infer it from the existing target-decode histogram plus total generation
   time and draft counts. The direct timers are worth the 15 lines — they also give you a
   before/after on the scatter's cost.

1. **Acceptance, per position.** `tools/server/server-context.cpp:614-634` already prints
   `draft acceptance`, `mean len`, and the per-position rates. This is the primary metric and
   the primary bug detector; a d2t error shows up here as acceptance near zero.
2. **Decode t/s matrix**, on the existing protocol: short, mid-1k, 7k, and the 54k agentic
   depth. The 54k point is the one that has decided prior waves (see the muse-glimmer
   draft-max 8 vs 15 result); do not ship on short+mid1k alone.
3. **Coverage rate on real traffic**: fraction of *target-sampled* tokens that fall inside the
   40k set. This is measurable offline from a calibration transcript and predicts the
   acceptance loss before any build.
4. **VRAM delta** (expect ~0.6 GB reclaimed) and a confirmation that the 131k context still
   fits.
5. **Quality**: a modest battery is worth running once as a bug detector, but per section 4 it
   is **not** a correctness gate — the output distribution is the target's by construction.
   Spend the effort on (1) and (3) instead.

### 5e. The main risk: choosing the 40k set

This, not the plumbing, is what determines whether the wave pays.

Acceptance requires the drafter's **argmax to exactly equal the token the target sampled** at
temp 1.0 / top-k 20 / top-p 0.95. So the set must cover what the target actually emits on
*our* traffic — English prose, code, and agentic tool-call JSON — not generic corpus
frequency. syv-ai calibrated on Danish/English/code and still lost ~10 points.

Recommended construction. The direct signal is **what the untruncated drafter currently
proposes**, not a corpus frequency ranking — truncation changes only the drafter's available
support, so the tokens that must survive are the ones its argmax actually picks today. That
signal is already in the tree: `common/speculative.cpp:1617-1623` logs the top-3 draft
candidates per step under `SPC_DBG`. So:

- Run the **current untruncated drafter** over a production-representative calibration corpus
  (agentic transcripts at depth, code, prose) with draft debug logging on, and take the union
  of (a) every token the drafter's argmax selected and (b) every token the target sampled.
  This *measures* coverage rather than predicting it, needs no new code, and directly produces
  the 5d.3 coverage number.
- Widen with a top-k union over the target's candidates if the measured union comes in well
  under 40k rows — spare rows are nearly free and only help.
- **Force-include every special and added token** — BOS/EOS, the chat-template turn
  delimiters, and the reasoning/thinking delimiters. If the drafter cannot propose the
  end-of-turn token, every turn ends in a guaranteed rejection, and that is a per-response
  cost that no frequency ranking will catch.
- Force-include all single-byte / byte-fallback tokens; they are cheap and prevent
  pathological failures on unusual input.
- Sweep the size (20k / 40k / 60k) only if 40k lands ambiguously. The traffic term is linear in
  rows, so 60k costs 57 MB more per draft pass — small enough that buying acceptance back with
  rows is usually the right trade.

### 5f. Risks and fallback

| Risk | Mitigation |
|---|---|
| Out-of-range/duplicate d2t → OOB write in `ggml_set_rows` | Validate range + uniqueness in the surgery script (mirrors `pr26275.diff:184-187`); the in-graph asserts at `eagle3.cpp:324-325` catch shape/dtype only |
| Acceptance collapse from a poorly chosen set | Measure coverage offline (5d.3) before building; per-position acceptance after |
| GGUF byte-layout error (alignment, offsets) | Model fails to load loudly; verify with a tensor re-dump before serving |
| Upstream drift on `qwen35.cpp` at the next rebase | 22 lines in one file, adjacent to a stable head-selection block; low conflict surface |
| Silent truncation if a d2t-carrying model ever runs the normal graph | `GGML_ASSERT(!model.d2t)` guard (5b.iii) |

**Fallback is a one-line configmap revert**: point `--model-draft` back at
`/models/mtp-Qwen3.8-27B-Q4_0.gguf`. Keep the original file on disk. The patched llama.cpp
binary is a no-op on an untruncated drafter (no `d2t` tensor → `n_vocab_draft == n_vocab` →
scatter skipped), so **the binary and the model can be reverted independently**. That is a
genuinely clean rollback and is the strongest argument for the in-graph scatter over any
design that touches `speculative.cpp`.

### 5g. Complexity estimate (honest)

| Component | LOC | Complexity |
|---|---|---|
| `src/models/qwen35.cpp` loader + graph + guard | ~23 | Low — copy of `eagle3.cpp:47-55` and `:318-331` |
| GGUF surgery script | ~150–250 | Medium — GGUF writing is fiddly; alignment/offsets are where it breaks |
| Calibration set builder | ~100 | Medium — needs a representative transcript corpus; this is the judgement-heavy part |
| Bench + acceptance battery | — | Reuses the existing matrix |

The C++ is the easy part. Budget the effort on 5e.

---

## 6. Summary of citations

| Claim | Evidence |
|---|---|
| Prod uses `draft-mtp`, sampler temp 1.0/top-k 20/top-p 0.95, `n_max 5`, no p-min | `k8s/workloads/apps/llama/configmap.yaml:101-118` |
| Patches 0001/0006 modify the DFlash impl, not MTP | `0001-...patch` hunks at `common_speculative_impl_draft_dflash`; master `common/speculative.cpp:908` vs `:1274` |
| MTP head = `nextn.shared_head_head` else `model.output` | `src/models/qwen35.cpp:637-640` |
| Drafter has no `shared_head_head`; `output.weight` is Q4_0 0.715 GB | GGUF dump, `mtp-Qwen3.8-27B-Q4_0.gguf`, 18 tensors |
| Embedded path builds ctx on `model_tgt` | `common/speculative.cpp:2405-2412` |
| Target `output.weight` is Q8_0 1.351 GB | GGUF dump, `Qwen3.8-27B-Q4_K_L.gguf` |
| `is_mem_shared` false for qwen35 | `src/llama-context.cpp:141`, `:144-160`; `common/speculative.cpp:1367` |
| Target loads dead MTP block | `common/common.cpp:1689`; `src/models/qwen35.cpp:42`, `:131`, `:158` |
| Backend draft sampling on by default | `common/common.h:331`; `common/speculative.cpp:1350-1362` |
| Backend sampling bypasses the 248k candidate array | `common/sampling.cpp:140-151`, `:607-628` |
| Draft argmax + p_min early stop | `common/speculative.cpp:1614`, `:1617`, `:1626`, `:1629` |
| Verification is sample-and-match, not rejection sampling | `common/sampling.cpp:680-706`; called at `tools/server/server-context.cpp:3819` |
| Acceptance histogram | `tools/server/server-context.cpp:614-634`, `:3880-3888` |
| d2t already at master for EAGLE3 | `src/llama-model.h:641`; `src/llama-arch.cpp:641`, `:905`; `src/models/eagle3.cpp:44-55`, `:318-331` |
| #26275 d2t: I64 absolute ids, validated, in-graph scatter | `pr26275.diff:183-187`, `:278-284`, `:303`, `:352-368` |
| Vocab-compat check inspects the tokenizer, not the head | `common/speculative.cpp:101-120` |
| MTP input embd gathers by full-vocab id | `src/models/qwen35.cpp:521-524`; `common/speculative.cpp:1567`, `:1666` |
| Draft top-3 candidates already logged (calibration source) | `common/speculative.cpp:1617-1623` |
| `ml` shadowing: dot in qwen35, arrow in eagle3 | `src/models/qwen35.cpp:36`, `:39-41` vs `src/models/eagle3.cpp:38`, `:47`; member at `src/llama-model.h:747` |
| `LLAMA_SPEC_PROF` pattern exists but is DFlash-only | patch `0001` hunks; MTP driver `common/speculative.cpp:1545-1697` has no timers |
