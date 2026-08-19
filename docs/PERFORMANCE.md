---
title: llama performance
---

# Making a 27B hybrid model decode faster on one RTX 3090

A one-day campaign (2026-08-16) to speed up `qwen3.8-27b` decode on a single RTX 3090, and
what shipped from it. The wins were **not** in the speculative-decoding config that looked like
the obvious lever. They were two CUDA kernel-dispatch decisions in llama.cpp tuned for other
shapes than ours — and the config change that mattered only worked *because* one of those was
fixed first.

Headline: **54k-token agentic decode went 49.8 → 66.8 t/s (+34%; 58.8 after v6, 66.8 after v7)**, which is the workload this
host actually serves. Other backends moved much less (muse-glimmer +5.7%, gemma4 and
qwen3.6-fusion unchanged), so read +18% as "the deep, code-grounded path", not as a blanket
number.

## Setup

**Hardware.** RTX 3090, 24 GB (24,564 MiB), sm_86 / cc 8.6, **82 SMs**, 370 W cap, driver
595.71.05; host 20 threads, 62 GB RAM. Being Ampere matters more than anything else here:
several llama.cpp fast paths for quantized KV are gated on `cc >= 89` (Ada) and do not exist on
this card.

**Serving.** Single-node k3s, one Flux-reconciled Deployment running
[`llama-swap`](https://github.com/mostlygeek/llama-swap) v230, which hot-swaps one
`llama-server` backend into VRAM per request — exactly one model resident at a time, selected
by the request's `"model"` field, `ttl: 3600` keeping it warm. Pod limits: 12 CPU / 32 GiB /
1 GPU. All four backends run at 131,072 context:

| model id | main GGUF | drafter | KV | draft-n | `env:` |
|---|---|---|---|---|---|
| `qwen3.8-27b` | bartowski `Qwen3.8-27B-Q4_K_L` | separate `mtp-Qwen3.8-27B-Q4_0` (MTP) | q4_0 | 4 | `GGML_CUDA_MMVQ_NE11_MAX=3` |
| `muse-glimmer-30b` | `muse-glimmer-30B-kquant-17gb` | `dflash-kquant` (DFlash) | q4_0 | 8 | `GGML_CUDA_MMVQ_NE11_MAX=3` |
| `hauhau-gemma4-26b-a4b` | Gemma4 26B-A4B QAT Q4_K_M (MoE) | `mtp-gemma-4-26B-A4B-it` | q8_0 | 4 | — |
| `qwen3.6-27b` | DavidAU Fable Fusion 711 Q4_K_S | embedded MTP head | q4_0 | 2 | — |

The tuning target is a hybrid Gated-DeltaNet model: **64 layers, only 16 holding standard
attention KV**, 24 Q heads / 4 KV heads (**GQA 6**), head dimension **256**. At q4_0 that is
18,432 B/token, which is why a 131k window fits at all. It also loads a Q8_0 vision projector.
Production flags (`configmap.yaml`):

```
/usr/local/bin/llama-server
  -m /models/Qwen3.8-27B-Q4_K_L.gguf
  --mmproj /models/mmproj-Qwen3.8-27B-Q8_0.gguf
  --host 0.0.0.0 --port ${PORT}
  --ctx-size 131072 --predict 131072
  --threads 8 --threads-batch 8 --parallel 1
  --batch-size 512 --ubatch-size 512
  --no-mmap -fa on --jinja
  --cache-type-k q4_0 --cache-type-v q4_0
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --repeat-penalty 1.0
  --model-draft /models/mtp-Qwen3.8-27B-Q4_0.gguf
  --spec-type draft-mtp --spec-draft-n-max 4
  -ngl 99 -ngld 99
```

**Runtime.** llama.cpp pinned at `62bf73d25c53b8161f8a22894d4f90c4aebbd7d0`, CUDA 12.8.1, built
for `sm_86` only, with four vendored patches globbed from `image/patches/`. 0001 and 0002
predate this campaign and target muse-glimmer's DFlash decode; **0003 and 0004 are what this
document is about.** (The Dockerfile header comment still lists only 0001/0002 — stale.)

**Per-backend `env:`.** llama-swap applies each model block's environment list to the
`llama-server` process it spawns. That is what makes patch 0004 shippable: its knob helps two
backends and hurts a third. Verify it is live, with the model loaded:

```bash
kubectl -n apps exec deploy/llama -c llama-swap -- sh -c \
  'tr "\0" "\n" < /proc/$(pgrep -x llama-server|head -1)/environ | grep MMVQ'
```

**Reading the numbers.** Almost everything below is a greedy bench (`temperature 0, top_k 1,
seed 42, cache_prompt true`), the only way to make arms comparable. Production samples at
`--temp 1.0 --top-k 20 --top-p 0.95`, where real acceptance — and therefore real t/s — is lower
for *every* arm, prod's included: the ranking carries, the absolute values do not. A
post-rollout prod smoke test measured 67 t/s. Payloads: `short` (~43–53 tokens),
`mid1k`/`mid1kprose` (~1k prompt, C++ source versus documentation prose), `long1024`/`prose7k`
(~7k), `deep` (54,440 tokens of llama.cpp sources — the real agentic workload).

## Baseline, and where the time went

Weights are ~17.5 GB, so at ~850 GB/s a no-speculation step should cost ~21 ms: **a ceiling of
about 48 t/s**. KV should be nearly free — 32,768 tokens of q4_0 across 16 layers is 604 MB,
**0.76 ms** at ~800 GB/s. Measured `tg64`:

| depth | t/s | ms/token | added vs d=0 |
|---|---|---|---|
| 0 | 39.54 ± 0.03 | 25.29 | — |
| 32,768 | 32.23 ± 0.08 | 31.03 | **+5.74** |
| 65,536 | 26.94 ± 0.05 | 37.12 | **+11.83** |

The depth penalty is **7.2× the prediction** and perfectly linear (+11.83 against 2 × 5.74 =
11.47), which rules out a fixed overhead. Three causes, on different code paths.

**1. The vector kernel reads KV once per Q head.** `ggml_cuda_get_best_fattn_kernel`
(`ggml/src/ggml-cuda/fattn.cu`), resolved for head_dim 256 / q4_0 K+V / cc 860 / GQA 6, sends
**batch 1 to the vector kernel** and **batch ≥ 2 to MMA_F16**; the Ada branch that would route
batch ≤ 2 to the vector kernel is gated `cc >= GGML_CUDA_CC_ADA_LOVELACE` and is dead here.
`flash_attn_ext_vec` gives every *Q head* its own block, so with GQA 6 the same KV head's K and
V are read and dequantized **six times per step**. That closes the gap: bandwidth predicts
+0.75 ms at 32k, ×6 gives **+4.53**, measured **+5.74**, the residual ~1.0 ms being dequant.
The kernel sustains **632 GB/s at 32k and 613 GB/s at 65k** against ~800 achievable — not slow,
just redundant, which decides the fix: share KV across the GQA group.

**2. Any batch > 1 bulk-dequantizes the whole cache.** MMA_F16 cannot read quantized KV.
`ggml_cuda_flash_attn_ext_get_alloc_size` marks `need_f16_K/V = true` unconditionally before
Ada, so `launch_fattn` (`fattn-common.cuh:1022-1084`) runs `to_fp16` over the **entire** q4_0 K
and V cache into a scratch buffer on **every FA call**, then reads it back. Measured later by
running the identical MMA kernel on f16 K/V, which skips the conversion by construction:
**543.6 µs per FA op, 67% of verify-path attention cost, 8.70 ms per verify step** across the
16 attention layers.

**3. The MMVQ → MMQ cliff sits three times too high.** Sustained forward-step cost at d=0:

| forward step (tokens) | ms/step | marginal ms per extra token |
|---|---|---|
| 1 | 23.48 | — |
| 2 | 26.37 | 2.89 |
| 4 | 39.14 | 6.57 |
| 8 | 62.35 | 6.24 |
| 12 | **40.13** | **−5.56** |
| 16 | 40.52 | 0.10 |

Two straight lines: **MMVQ `step(N) = 23.48 + 5.55·(N−1)`** for N ≤ 8, **MMQ
`step(N) = 35.43 + 0.39·N`** for N ≥ 12 — a 14× difference in marginal cost, and a 12-token
step **22.2 ms cheaper in wall time** than an 8-token one. They cross at **N ≈ 3.4**; llama.cpp
switches at 9, because `ggml_cuda_should_use_mmvq` ends in
`return ne11 <= MMVQ_MAX_BATCH_SIZE` with `#define MMVQ_MAX_BATCH_SIZE 8`
(`ggml/src/ggml-cuda/mmvq.cuh:3`), a flat constant for every NVIDIA GPU and quant type. The
only per-type tuning there is for AMD CDNA2, where **Q4_K is tuned to `ne11 <= 3`**: our model
is Q4_K and our measured crossover is 3.4. Verify batches of 4–8 tokens were overpaying
**24–38%**, which is what capped useful draft-n, not acceptance. (`GGML_CUDA_FORCE_MMQ` is not
the escape: it is a CMake `#ifdef`, not an env var, and even compiled in it only affects
`ggml_cuda_should_use_mmq`, reached *after* `should_use_mmvq` already returned true at
batch ≤ 8.)

## What shipped

### Patch 0003 — GQA-batched flash-attention vector kernel

Gives `flash_attn_ext_vec` the two-axis column split MMA already has: `ncols2` holds a whole GQA
group in registers while Q tokens stay on the grid, so one block serves the group and K/V are
read and dequantized once, staying quantized. Reusing `BEST_FATTN_KERNEL_VEC` means
`need_f16_K` is already false for q4_0, so the F16 scratch allocation disappears with no extra
code, and `ncols2 == 1` stays register- and instruction-identical to baseline.

Two register findings shaped it. `dst_meta[...] = make_float2(KQ_max[tid], KQ_sum[tid])` indexes
otherwise statically-indexed register arrays with a *runtime* `tid` — invisible at `ncols == 1`,
but at `ncols == 6` exactly the pattern that makes ptxas demote a whole array to local memory.
And fully unrolling the KQ loop spilled 116 bytes at `ncols2 >= 6`. An unrolled predicated store
plus `#pragma unroll 8` on the `i_KQ_0` loop gives **255 registers and zero spills** at every
head size and `ncols2`. SASS confirms the mechanism: 422 global loads per column at `ncols2=1`
against ~49 at `ncols2=6`.

**The SM-fill floor.** v5 regressed shallow depths, because the base grid shrinks 6× and
`parallel_blocks` cannot refill 82 SMs at short KV:

| llama-bench row | v5 patch vs base |
|---|---|
| tg64 @ d0 | **−2.4%** |
| tg64 @ d512 | **−2.3%** |
| tg64 @ d4096 | +0.1% |
| tg64 @ d32768 | **+9.8%** |
| tg64 @ d65536 | **+21.0%** |

With error bars of ±0.02–0.10 that is ~40σ. v6 adds a device-aware predicate,
`ntiles_dst * ntiles_KV >= nsm`, shared by the selector and the dispatcher so they cannot
disagree; on this GPU it activates at KV ≥ 5,376 and hands the shallow rows back to the stock
kernel. A device property, not a magic KV number.

**Why the gate is `Q->ne[1] <= 2`.** The first answer was wrong: a traffic model predicted the
vector kernel should win up to batch ~8. Op-level measurement (`test-backend-ops perf`, one
binary, env-switched, all 12 control rows matching within 1%) says otherwise:

| kv=54,016, µs/op | nb=3 | nb=4 | nb=5 | nb=6 | nb=8 |
|---|---|---|---|---|---|
| MMA_F16 | **812** | **812** | 831 | 833 | 838 |
| GQA vec | **713** | 935 | 1150 | 1372 | 1840 |

MMA's cost is **flat** in nb; the vector kernel's is **linear at ~222 µs per extra token**. The
reason is not bytes but achieved bandwidth: MMA's `cp.async` multi-stage pipeline streams at
**621 GB/s**, while the vector kernel at 255 registers and 128 threads/block runs at low
occupancy and is latency-bound at **280 GB/s**. Crossover `270 + 222·(nb−1) = 812` → **nb =
3.4**. So `nb ≤ 3` should win on the op — and at nb=3 it does, by 1.14× — but **end-to-end at
nb=3 measured per-step parity**, so that op win never reaches production. An independent line
agrees: an earlier llama-bench A/B with the cap forced to 4 showed batches 3 and 4 winning
nowhere and losing badly at depth (−0.3% and −9.2% at 65k). The shipped `≤ 2` is one notch
conservative rather than wrong.

**Validation.** KLD versus the unpatched kernel at `-ub 2`: mean **0.003010 ± 0.000096**, median
0.000469, same-top-p 98.009%, PPL ratio 0.999076 ± 0.001189 — numerics-level divergence from a
different reduction order, not a quality regression. `test-backend-ops -o FLASH_ATTN_EXT` passes
with zero failures on every tree it ran against (2,970 cases on the shipped v5 tree, 5,191 on
the upstream-master build, 2,945 on the wave-4 tree). Coverage was a real gap: upstream gates
quantized `type_KV` to `hsk in {64, 72}` and only reaches GQA 6 at `hsk == 256` with F16 K/V, so
**our exact shape (D=256, GQA 6, quantized KV) had zero upstream coverage.** The patch adds it,
plus fallback controls (no mask, ALiBi, sinks, logit softcap, permuted Q, `kv=1000`) that must
*not* change.

### Patch 0004 — env-gated MMVQ batch cap

Adds `GGML_CUDA_MMVQ_NE11_MAX`, capping the dense-path MMVQ batch threshold at runtime, off by
default. One incidental blocker: `ggml-cuda.cu:1882` carries
`static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE)` inside `ggml_cuda_mul_mat_id` (MoE
routing). That was dropped rather than lowering `MMVF_MAX_BATCH_SIZE` in lockstep, so the change
stays scoped to the quantized dense path and **MoE `mul_mat_id` keeps the stock threshold either
way.** Sanity gate at d=0, stock versus cap-3:

| N | stock (ms) | cap-3 (ms) | delta |
|---|---|---|---|
| 1–3 | 25.0 / 27.8 / 32.5 | 25.2 / 27.7 / 32.0 | ~0 (correctly scoped) |
| 4 | 39.06 | 35.78 | **−8.4%** |
| 5 | 43.14 | 36.21 | **−16.1%** |
| 6 | 49.46 | 36.41 | **−26.4%** |
| 8 | 61.46 | 36.84 | **−40.1%** |

N=4–8 collapse onto a flat ~36–37 ms line, matching the predicted MMQ curve. KLD versus stock at
`-ub 6`: **0.006325 ± 0.000258**.

**Why per-backend and not global.** A blanket cap of 3 cost `hauhau-gemma4-26b-a4b` **34%** of
its decode — its small dense matmuls genuinely prefer MMVQ at batch 4–8. That figure needs a
caveat the shipped comment does not carry: it was measured on a **hard-capped build** (153 vs
236 t/s on `short`). A later sweep of the **env-gated** build did not reproduce it, measuring
short +7.7% and mid1k −1.7% against a ~104–109 t/s baseline. The decision is unaffected — gemma
fails the ship rule either way, on the mid1k regression — but the "34%" belongs to the hard cap.

### Config: draft-n 4, and the coupling

Before the cap, `--spec-draft-n-max 4` was a real trade: better deep (52.75 vs 49.80), worse
short (58.37 vs 62.35). With the cap it wins everywhere. End-to-end on the threshold-3 tree,
greedy, 320 generated:

| arm | `deep` (54k) | `short` |
|---|---|---|
| n2, stock kernel (**old prod**) | 49.80 | 62.35 |
| **n4 + cap 3 (shipped)** | **58.55** | **68.50** |
| n4 + p-min 0.3 + cap 3 | 59.02 | 66.18 |
| n6 + cap 3 | 58.53 | 69.67 |

And at ~1k prompts, all arms measured in one block against the shipped kernel:

| arm | `mid1k` (code) | `mid1kprose` |
|---|---|---|
| v5 prod n2 | 64.01 | 62.69 |
| v6 n2 + cap | 64.11 | — |
| v6 n4, **no cap** | 60.77 | 59.82 |
| **v6 n4 + cap (shipped)** | **70.33** | **69.57** |

**The coupling is the load-bearing fact.** Same tree, same arm, cap off: 60.77/59.82 — *worse
than n2*. The env turns a −5% regression into a +10% win. If llama-swap's per-model `env:` is
ever dropped or mistyped while `--spec-draft-n-max 4` stays, the backend silently gets slower
than it was before, and nothing fails loudly. Never move one without the other.

**p-min shipped, then was removed.** v6 shipped `--spec-draft-n-max 4 --spec-draft-p-min 0.3` —
a combination that had **never been benchmarked**, since all the p-min evidence came from the
uncapped kernel. Measuring it afterwards (commit `ce6468d` dropped it) gave `mid1kprose`
**62.05 vs 69.57, −10.8%**, with non-overlapping rep ranges: the *best* combo rep (65.48) is
worse than the *worst* plain-n4 rep (69.52). On `mid1k` the combo means +3.6% but its reps span
66.20–75.84 and straddle the control. Plain n4 is also **bit-deterministic** at temp 0
(identical `draft_n`, t/s within 0.12% across four reps) while the p-min arm swings ±8%, because
early-stopping interacts with the MTP implementation's cross-request `pending_h` carryover.
`n-max 6` was rejected: no gain over 4, and 23,650 MiB against a 23,400 MiB ceiling.

### muse-glimmer: the same lever, a different mechanism

The cap was expected to be neutral for DFlash, whose natural verify batch (~16 tokens) sits
above the stock MMVQ cap. Wrong: `--spec-draft-n-max` hard-caps the verify batch at n-max+1,
*below* DFlash's proposal length, so most cycles land inside the crossover anyway.

| arm | short | mid1k | mix | VRAM |
|---|---|---|---|---|
| n15 stock (**old prod**) | 56.03 | 65.29 | 60.66 | 20,090 MiB |
| **n8 + cap 3 (shipped)** | **61.80** | **66.39** | **64.10 (+5.7%)** | 20,090 MiB |
| n15 + cap 3 (1 rep) | 59.61 | 67.09 | 63.35 | 20,090 MiB |

`qwen3.6-27b` keeps its stock config: acceptance decays much faster with n there
(.68 → .49 → .36), so n4+cap regresses −2.6% on mid1k and −4.6% on prose.

### Scorecard

| workload | before | after | delta |
|---|---|---|---|
| `qwen3.8` 54k agentic decode | 49.80 | 58.55 measured / 58.8 recorded | **+18%** |
| `qwen3.8` ~1k code | 64.01 | 70.33 | +9.9% |
| `qwen3.8` ~1k prose | 62.69 | 69.57 | +11.0% |
| `qwen3.8` short (~43 tok) | 62.35 | 68.50 | +9.9% |
| `qwen3.8` 54k, **no speculation** † | 28.2 | 33.3 | +17.9% |
| `muse-glimmer` mix | 60.66 | 64.10 | +5.7% |
| `gemma4`, `qwen3.6-fusion` | — | unchanged | — |

† Commit-record figure only — see the provenance note below.

VRAM: **23,318 MiB measured in production** with the shipped config loaded (bench peak 23,354
MiB), against 23,016 MiB before, on a 24,564 MiB card with a 23,400 MiB working ceiling.

**Provenance, where sources differ.** The shipped commit record (`configmap.yaml`,
`MIGRATION_LOG.md`) quotes 62.3→68 short, 62→74 at 7k code, 61.9→67.4 on 7k prose, 49.8→58.8
deep, and 28.2→33.3 no-spec. The preserved measurement reports contain the short and deep arms
(68.50 and 58.55) but **not** the two 7k arms, the 58.8 figure, or the no-spec pair — those were
run outside the reports that survive and exist only as campaign summaries. Where this document
had to choose it quotes the report row; the 7k and no-spec rows above have no surviving report
behind them and should be re-measured before anyone builds on them.

## Findings worth generalizing

**Content, not depth, decides draft-n economics.** At identical depth and generation length,
code-grounded generation ties or wins with longer drafts while prose loses ~7%. Acceptance is
indistinguishable at n-max 2 (0.749 vs 0.739) and separates only past draft position 2. The
2×2 on the pre-cap kernel: code −0.2% at 1k, +6.7% at 7k, +7.2% at 54k; prose −7.0% / −7.4% /
−6.5%, flat across depth. Depth scales the magnitude on the winning side; it never flips the
sign.

**`--spec-draft-p-min` inverts once verify is cheap.** It stops drafting when confidence drops
to save verify cost, but once the cap makes rejected positions nearly free it mostly forfeits
*accepted* tokens instead. It was never free anyway — stopping at position k has already paid
for the drafter decode that produced the rejected token, which is why p-min 0.9 drafts 1.68
tokens/cycle at 0.91 acceptance on deep and still runs 47.4 t/s against plain n2's 49.8.

**The MMVQ crossover is model-size dependent.** Upstream's flat 8 is wrong for a 27B dense Q4_K
matmul on this card but right for gemma4, whose smaller MoE matmuls genuinely prefer MMVQ at
batch 4–8. Any fix has to be per-model, which is why 0004 is an env knob.

**Mixed K/V cache types silently drop flash attention off the GPU.** `fattn.cu` has
`if (K->type != V->type) return BEST_FATTN_KERNEL_NONE` absent `GGML_CUDA_FA_ALL_QUANTS`. With
`q4_0` K and `q8_0` V, prompt processing collapses to **39%** of prod (521 vs 1319 t/s) and at
depth the run sat at 0% GPU utilisation on a CPU fallback. llama-bench still prints `fa = 1`
because it reports the *requested* flag, not the kernel chosen, so K at f16 with V at q4_0 to
save VRAM gets no warning.

**f16 and q8_0 KV do not help on Ampere.** q8_0 stays on the same 6×-redundant vector path with
1.9× the bytes, so it is *worse* than q4_0; f16 reaches the GQA-shared MMA path but is 8.4 GB
at 131k and does not fit alongside the drafter and projector. There is no shippable KV-type
interim — that arm was purely diagnostic.

**The default host-RAM prompt cache already solves agentic TTFT.** This build ships
`--cache-ram 8192`, `--cache-idle-slots true` and `--ctx-checkpoints 32` **on by default**, and
prod passes none of them. At 54k context:

| pattern | TTFT today | with the cache off |
|---|---|---|
| cold prefill | 55.8 s | — |
| repeat the same prompt | **0.26 s** | — |
| append a new turn (896 tok) | **1.44 s** | — |
| another client interleaves, then return | **0.57 s** | **57.0 s** |
| edit the tail and retry | **0.98 s** | **56.0 s** |

Three concurrent 50k conversations with zero shared prefix all stay hot (~1.1 s per revisit
against ~51 s cold) using 3.4 GiB of the 8 GiB default — room for about five. The pod's
*anonymous* RSS with the model resident is only **0.90 GiB** (weights are on the GPU), so a full
cache is ~8.9 GiB of a 32 GiB limit.

**Never lower `--ctx-checkpoints` on a hybrid model.** The recurrent DeltaNet state cannot be
truncated, so a prefix is reusable only if a checkpoint exists to roll back to; without one the
server hits `do_reset` and silently re-prefills from zero, turning every cache hit back into a
56-second prefill with no error. Raising it above 32 buys nothing — a real 55k conversation used
3 of the 32 slots.

**`/upstream/*` on llama-swap is proxied and loads the model.** A `GET /upstream/<model>/slots`
poll answers in ~300 µs when the model is resident and takes **10 s while cold-loading 23 GB**
when it is not, so an "is prod idle?" gate built on it wakes the very thing it checks — this
killed one vLLM startup by losing the VRAM race. Use `GET /running`, which llama-swap serves
itself and never proxies.

**llama-bench routes `-d` prefill through `-ub`.** `-ub 3 -d 32768` becomes 10,923 sequential
3-token steps and stalls, so batch size cannot be crossed with depth there; that cell has to
come from an end-to-end server run.

## Tried and rejected

- **llama.cpp master** (105 commits ahead): bit-identical output, −5.6% short / −1.5% long. The
  premise that our pin lacked backend sampling was wrong; the only new CLI flag in 105 commits
  is `--reasoning-effort`.
- **PR #27173, no env**: +2.1% on a 1024-token generation — the only config that beat pinned,
  but small, and an hours-old unreviewed PR. Re-evaluate after it merges.
- **`LLAMA_SPEC_CHAIN=1`** (same PR): −6% to −12% on the realistic prompt, acceptance
  .829 → .602, +368–462 MiB VRAM, and greedy output becomes **non-reproducible**.
  `LLAMA_SPEC_CHAIN_SUB=0` does not rescue it.
- **`GGML_CUDA_GRAPH_OPT=1`**: 64.41 vs 64.12 long, 62.28 vs 63.31 short — within noise.
- **vLLM W4A16 recipe** (`syv-ai/qwen38-27b-rtx3090`): the requant reproduced exactly, but at
  the recipe's own `--gpu-memory-utilization 0.90` context caps at **124,800 tokens, below
  prod's 131,072**, and cold start is **4–5 minutes** (112 s warm). It also needs a new ~8 GB
  image, and flashinfer JIT-compiles at first request, so the server health-checks green and
  then fails every request if `nvcc` is missing.
- **Option C, raising the vec gate to nb ≤ 8**: the vector kernel is latency-bound at 280 GB/s
  against MMA's 621 GB/s on the same shape, moving the crossover from a predicted 8.1 to 3.4.
- **Option B, incremental F16 scratch**: byte for byte an F16 shadow of the KV cache — 3.54 GB
  at 54k, **8.59 GB at 131k** — against ~1 GiB of headroom.
- **`--spec-draft-n-max 6`**: 23,650 MiB and worse throughput everywhere.
- **`--parallel 2`**: +620 MiB (23,942 MiB peak, over the ceiling), halves the per-client window
  to 65,536, saves 0.07–0.10 s per client switch.
- **`--slot-save-path`**: restore reports success (`n_restored` 54,468 in 251 ms) and the next
  request still re-prefills in 56.0 s with `cache_n = 0` — the save file carries no checkpoints
  and no draft state.
- **`--cache-reuse`**: force-zeroed at startup whenever `--mmproj` is loaded.
- **DFlash / EAGLE-3 / DSpark drafters**: open upstream bugs (DFlash acceptance stuck ~0.15 from
  a feature-extraction mismatch; EAGLE-3 asserts above ~700 tokens, i.e. every prompt we care
  about), and no trained checkpoint exists for this target.
- **Q8_0 MTP head**: **+3.16 GB** against the embedded baseline, not the +1.48 GB it appears to
  cost against the current file.
- **Embedded versus separate drafter**: the production Q4_K_L **already embeds an MTP head**
  (15 `blk.64.*` tensors) at the same Q4_0 quantization as the separate file, so dropping
  `--model-draft` is a potential ~1.6 GB VRAM saving with no speed change. Not acted on.

## Open items

### A2b — teach MMA to read q4_0 inline (Option A)

The remaining prize on the verify path, priced by two measurements rather than guessed. **A1**
ran the identical MMA kernel on f16 K/V — "MMA with the conversion pass deleted" — measuring the
conversion at **543.6 µs/op = 8.70 ms/step**, a hard floor on the saving, and clearing the risk
that killed Option C: MMA sustains **811 GB/s at kv=54,016 and 695 GB/s at kv=16,384**, so it
does *not* lose efficiency at the small byte count Option A would move. **A2a**, an isolated
load-stage microbenchmark whose f16 control validated within 8% of the real kernel, measured the
quantized load plus inline dequant at **174.9 µs against 251.7 µs** for today's f16 load: the
dequant consumes about a third of the bandwidth saving, not all of it, and the real MMA kernel
is ~92% load-bound at this shape.

Projection: **8.7–9.9 ms saved per verify step, +13% to +19% decode at depth.** Shared memory
fits at occupancy 2 for q4_0 (43,328 B against a 51,200 B budget) but **not for q8_0** (51,520
B, over by 320), and the design needs `D % 64 == 0` for `cp.async` alignment, so it does not
generalize to D=128 without a repack. Pre-flight register check clears at 255 registers with no
stack growth. If it lands it very likely **subsumes patch 0003** — MMA on f16 at nb=1 already
beats the GQA vector kernel (261.1 vs 271.2 µs) while moving 3.56× more bytes.

> **A2b result: SHIPPED as patch `0005` in `llama:cuda-swap-v7` (2026-08-16 ~23:45 UTC).**
> The inline-dequant MMA kernel beat its own projection. Op-level at kv=54,016, q4_0, single
> binary switched by `GGML_CUDA_FATTN_MMA_Q` (arm 0 = v6 kernel, arm 1/2 = new; every f16, q8_0
> and nb=1 control row 1.00×):
>
> | nb | v6 (µs/op) | new (µs/op) | speedup |
> |---|---|---|---|
> | 2 | 484.4 | 181.5 | 2.67× |
> | 3 | 813.1 | 160.5 | 5.07× |
> | 4 | 815.9 | 161.1 | 5.06× |
> | 5 | 831.6 | 224.0 | 3.71× |
> | 8 | 837.8 | 230.4 | 3.64× |
>
> Saved per verify step (×16 layers): 10.4–10.5 ms at nb=3/4, vs the 8.7–9.9 projection. The
> gate the data selected is nb ≥ 2 (not the planned ≥3): nb=2 is common in production and gains
> 2.67×, while nb=1 measured 10% *slower* on the new path (271 → 300 µs, the ~40 µs of inline
> dequant outweighs the vector kernel's single-column deficit), so patch 0003 stays for batch 1.
> End-to-end, qwen3.8-27b MTP n=4 on the 54k payload: **58.8 → 66.8 t/s (+13–16%)**; short and
> no-spec arms unchanged. KLD vs the F16-conversion path at `-ub 4`: mean 0.000000, max 0.000064,
> same-top-p 100% — numerically equivalent, as expected from dequantizing the same values in a
> different place. 2994/2994 FLASH_ATTN_EXT tests pass. Cumulative for the day on the primary
> workload: **49.8 → 66.8 t/s (+34%)**.

### Upstream submission of patch 0003

A branch is prepared and preserved at
`/data/buttercup_6tb/k3s/llama-models/upstream-pr/llama.cpp`
(`cuda-fattn-vec-gqa-small-batch`, three commits on master `4df29be4f`, rebase clean — no
upstream commit touched any `fattn` file between our pin and master). It passes 5,191
`FLASH_ATTN_EXT` cases with zero failures on the master build, and a duplicate search found no
competing PR.

**It has not been submitted, and submission is a manual step for a human.** `ggml-org/llama.cpp`
ships an `AGENTS.md` and `CONTRIBUTING.md` that explicitly forbid agent-created pull requests and
AI-written PR descriptions, on pain of a contributor ban, so the evidence pack contains
measurements and commands only.

### The gemma4 baseline discrepancy

Two sessions measured gemma4's `short` payload at **236 t/s** and **~104 t/s**, unexplained and
most likely a payload or generation-length difference. It affects no shipped decision, but it
means the "blanket cap costs gemma 34%" claim rests on a baseline that has not reproduced.
Re-check before ever enabling the env for that backend.

## Reproduction notes

These assume you are A/B-ing against a **live production service on the only GPU in the
machine**, the constraint that shaped the whole methodology.

- **Isolate the GPU.** Take an exclusive lock (`mkdir` is atomic) recording the owning **pid**,
  not just a name — an owner name cannot survive its holder's death, and one agent deadlocked
  against its own orphaned waiter for 15 minutes. Then idle-gate: prod slots
  `is_processing: false`, GPU under 5%, no new `chat/completions`, held two minutes. **Gate on
  `GET /running`, never `/upstream/*`.** Then `GET /unload` and confirm `memory.used < 1000 MiB`.
- **Run arms in containers**, using the exact production `cmd:` as the base flag set — including
  `--ctx-size 131072`, or the VRAM numbers mean nothing:

  ```bash
  docker run --rm --gpus all --ipc=host -p 5899:5899 \
    -v /data/buttercup_6tb/k3s/llama-models:/models:ro \
    llama:cuda-swap-v6 llama-server <prod flags> --spec-draft-n-max N
  ```

- **Fix payload and sampling**: `temperature 0, top_k 1, seed 42, cache_prompt true`, one warmup
  request to fill the prompt cache, then **2–4 measured reps**. Fixed-n speculative arms are
  bit-deterministic and agree to <0.2%; anything varying draft length is not, and needs four
  sequential reps whose min–max range still understates the true interval.
- **Compare per-step milliseconds, not raw t/s, across any numerics change.** A kernel change
  perturbs the logits, so two arms sample different token streams and different acceptance
  rates, and raw t/s then compares acceptance luck as much as kernel speed. Normalise with
  `steps = draft_n / n_max`; doing this turned an apparent −2.9% loss into a correctly
  attributed +5.7% one, and an apparent −1.5% loss into measured parity.
- **Measure sustained work.** A single small forward pass times the GPU's clock ramp out of
  idle: noise was **9–28%** on single-pass rows against **0.05–0.3%** sustained, and `-r 3` does
  not fix it because every repetition pays the same ramp. Log
  `clocks.sm,temperature.gpu,power.draw,memory.used` at 5 s, and interleave depth rows so
  thermal drift hits every arm equally. **Always include a same-block control** — one
  cross-block comparison here produced a 7% "regression" that never reproduced.
- **Quality-gate numerics** with `llama-perplexity --kl-divergence-base <base.dat>` at `-ub 2`
  (or `-ub 4`), generating a fresh base at the *same* ubatch. A base from a different ubatch
  exercises a different kernel path, and a run above the patched gate reports KLD exactly
  0.00000 at 100.000% same-top-p — which looks perfect and is vacuous. A same-top-p meaningfully
  below 100% is the evidence the run touched the kernel. Bar: mean KLD ≤ 0.01, against 0.0030
  (0003) and 0.0063 (0004) measured.
- **Verify VRAM on a clean run at the real context size**, never from `llama-bench` — it has no
  drafter, no projector, and sizes `n_ctx` to the test. The ceiling here is 23,400 MiB.

## Weight-byte traffic as a decode-cost model — corrected

An earlier working estimate put weight-byte traffic at **77% of decode step
cost**, used to predict how much a smaller quant should speed up decode. A
later dedicated decomposition (comparing measured per-step time deltas
against each quant's actual byte-size delta, with a same-family K-quant
control to hold dequant cost constant) found the real figure is **closer to
40%** — the 77% model correctly *orders* candidate quants by expected speed,
but overstates the *magnitude* of any byte-size win by roughly 2x. Two
independent arms converged on ~40%, so this isn't one quant type's dequant
cost skewing the estimate. Use ~40% as the working model for predicting a
future quant change's speed effect; see `docs/quant-selection.md` for the
full sweep this correction came from, including a 3.7bpw quant where the
model breaks down entirely (smaller weights, slower decode — dequant cost
overtook the bandwidth saving below 4 bits).

## DFlash2 and quant selection: see their own docs

Two more recent lines of investigation grew large enough for their own
files rather than fitting here:

- `docs/dflash2-findings.md` — evaluating `llama.cpp` PR #27342 (DFlash2)
  against this repo's MTP drafter: the MMVQ/MMQ small-batch crossover flips
  the verdict entirely, real-sampling and ngram-stacking caveats, and the
  two upstream bugs this surfaced.
- `docs/quant-selection.md` — the sweep behind moving prod's main quant to
  UD-Q4_K_XL, including a rejected 3.7bpw extreme.
- `docs/thermals-and-oc.md` — the fan-curve dead zone that was silently
  stopping cooling at high VRAM temperature, the NVML offset convention, and
  the shipped overclock.
