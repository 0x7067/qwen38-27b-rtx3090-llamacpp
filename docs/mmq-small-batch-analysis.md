# wave 6 — small-batch quantized GEMM for speculative verify (qwen3.8-27b)

Target from `muse-opt.md` §7.3/§9: at verify batch nb=5 the four extra columns cost **13.0 ms**
(mid1k) where the weights are re-read zero extra times. Prize sized at +18% to +33% decode.

Tree: `$SCRATCH/kernel/w6-tree` = prod `v9-tree` (master 4df29be4f + patches 0001-0005) plus
test-backend-ops perf/eval cases at qwen's shapes. Config arms `w6-B`, `w6-C` branch from it.

---

## M0 — measurement, and it relocates the defect

### 0.1 Two free source checks, done before any GPU time

Both changed the plan, so they are recorded first.

**(a) §9.3 item (4) — "a `ncols_dst=5` or `6` instantiation, a template argument plus a dispatch
line" — is a compile error.** The MMQ kernel is not templated on `ncols_dst`; that is a *runtime*
argument (`mmq_args::ncols_dst`, `mmq.cuh:1374`). The template parameter is `J`, the tile width, and
`mmq.cuh:211` carries `static_assert((J_) % 8 == 0, "bad J")`. The nsys label
`mul_mat_q<Q4_K, ncols_dst=8>` in §7.5 is `J=8`. J=8 is also the width of the Ampere int8 MMA
fragment (`m16n8k32`), so it is the hardware floor, not a tuning choice. §9.3 called this "the
highest value-per-effort experiment in the whole wave" — **it cannot be done as described.**

The underlying observation was still right, and §0.3 below quantifies it: cost really is
proportional to J, and at nb=5 three of eight computed columns are discarded. It just cannot be
fixed by narrowing the tile.

**(b) Patch 0002's re-tile never reached qwen's verify path.** 0002
(`0002-mmq-ampere-small-batch-tiles.patch`) changes exactly four `CASE` lines, all of them
**Q4_K and Q5_K at J=16**, `256,1,128,16` -> `128,1,64,16`. Qwen's verify runs at **J=8**. Every
J=8 entry for every type is stock `256, 1, 128, 8`:

```
mmq-config-ampere.cuh:36,41    Q4_0  256, 1, 128, 8
mmq-config-ampere.cuh:104,109  Q8_0  256, 1, 128, 8
mmq-config-ampere.cuh:157,162  Q4_K  256, 1, 128, 8
mmq-config-ampere.cuh:191,196  Q6_K  256, 1, 128, 8
```

So the wave-2 result that took Q4_K from 29% to 67% DRAM utilisation has never been applied to the
kernel qwen actually runs. That makes a config-table change, not a new kernel, the first thing to
try.

### 0.2 The type inventory in the brief and in §9.2 is wrong

§9.2 says "Q4_K (most layers)". The GGUF says otherwise. Parsed from
`Qwen3.8-27B-Q4_K_L.gguf` (pure-python reader, `$SCRATCH/ggufshapes.py`, no numpy in this env):

| tensor | m (out rows) | k (reduction) | type | per forward | MB/call |
|---|---|---|---|---|---|
| ffn_up, ffn_gate | 17408 | 5120 | **Q4_0** | 64 each | 50.1 |
| ffn_down | 5120 | 17408 | **Q4_0** | 64 | 50.1 |
| attn_qkv (gated-deltanet) | 10240 | 5120 | Q6_K | 48 | 43.0 |
| attn_gate (gated-deltanet) | 6144 | 5120 | Q4_K | 48 | 17.7 |
| ssm_out | 5120 | 6144 | Q8_0 | 48 | 33.4 |
| attn_q | 12288 | 5120 | Q4_0 | 16 | 35.4 |
| attn_k, attn_v | 1024 | 5120 | Q4_0 | 16 each | 2.9 |
| attn_output | 5120 | 6144 | Q4_0 | 16 | 17.7 |
| output head | 248320 | 5120 | Q8_0 | 1 | 1350.9 |

Total **16.44 GB** of weights per target forward, against the brief's 17.5 GB (the remainder is the
drafter's own gguf and the mmproj, neither of which is in the verify step). Share by bytes:
**Q4_0 61%**, Q8_0 17%, Q6_K 12%, Q4_K 5%. `qwen35.block_count = 65` = 48 gated-deltanet + 17
attention, where block 64 is the MTP head layer; the target forward is modelled as 64 layers
(48 GDN + 16 attention), which is the ±1 uncertainty in these sums.

**Any per-type work is Q4_0-first. A Q4_K-only change addresses 5% of the problem.**

### 0.3 The op-level sweep — MMQ's penalty is a fixed step, not a per-column cost

Added `MUL_MAT` cases at the nine shapes above for n = 1..9, to **both** `make_test_cases_eval()`
and `make_test_cases_perf()` (perf mode uses a separate list; the first build only patched eval and
measured nothing). Patch: `$SCRATCH/w6-patches/testcov-mulmat-qwen38-verify.patch`.

Two arms on one binary, using the vendored env cap to pick the kernel:
`GGML_CUDA_MMVQ_NE11_MAX=8` gives MMVQ for n=1..8; `=1` gives MMQ for n=2..9. Runner
`$SCRATCH/w6_block0.sh`, logs `$SCRATCH/kraw/w6/tbo_{V,Q}.log`, analysis `$SCRATCH/w6_parse.py`.
Both arms ran in one 7-minute GPU block (13:01-13:05 UTC).

us/run per call, MMVQ left, MMQ right:

| tensor | ty | V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8 | | Q1 | Q2 | Q4 | Q5 | Q8 | Q9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ffn_up | q4_0 | 60.8 | 61.2 | 63.0 | 71.9 | 78.2 | 91.3 | 103.9 | 116.5 | | 61.0 | 91.0 | 91.3 | 92.0 | 92.8 | 103.2 |
| ffn_down | q4_0 | 60.6 | 62.5 | 74.7 | 100.2 | 81.3 | 101.2 | 124.9 | 121.4 | | 60.7 | 91.4 | 93.1 | 92.2 | 93.8 | 105.1 |
| attn_qkv | q6_K | 58.0 | 59.4 | 68.4 | 78.0 | 86.8 | 98.2 | 109.1 | 121.5 | | 59.2 | 74.3 | 74.0 | 74.3 | 74.6 | 79.8 |
| attn_gate | q4_K | 24.8 | 28.5 | 35.9 | 43.6 | 49.9 | 58.7 | 67.2 | 75.0 | | 24.8 | 37.4 | 37.5 | 38.0 | 38.2 | 41.6 |
| ssm_out | q8_0 | 41.8 | 42.3 | 42.7 | 43.2 | 47.9 | 48.2 | 54.1 | 59.2 | | 41.9 | 57.1 | 57.2 | 58.0 | 58.1 | 60.1 |
| output head | q8_0 | 1523 | 1532 | 1540 | 1563 | 1631 | 1862 | 2041 | 2219 | | 1524 | 1825 | 1838 | 1842 | 1849 | 1855 |

**MMQ is flat from n=2 to n=8 and steps once.** ffn_up goes 61.0 -> 91.0 us the moment n leaves 1,
then moves 1.8 us across the next six columns. That is the J=8 tile computing eight columns
regardless of n: marginal per column **0.3 us**, fixed penalty **30 us**. n=9 steps again to 103 us,
which is the J=16 tile — cost is proportional to J, confirming the padding reading of §9.2.

**MMVQ is free for two extra columns and then turns linear.** ffn_up: 60.8 / 61.2 / 63.0 for
n=1/2/3 (0.7 us per extra column, i.e. hidden entirely behind the DRAM stall), then ~9-13 us per
column for n>=4. The break is where `calc_nwarps` drops 4 -> 2 and `calc_rows_per_block` goes 1 -> 2
(`mmvq.cu:372,478`).

### 0.4 Reconciliation against the model-level 13.0 ms — it closes

Summing each shape over its per-forward count:

| n | MMVQ (ms) | MMQ (ms) | prod routing (cap 3) | vs n=1 |
|---|---|---|---|---|
| 1 | 20.45 | 20.55 | 20.45 | +0.00 |
| 2 | 20.91 | 29.50 | 20.91 | +0.46 |
| 3 | 22.78 | 29.51 | 22.78 | +2.33 |
| 4 | 26.58 | 29.63 | 29.63 | +9.19 |
| **5** | **27.35** | **29.75** | **29.75** | **+9.31** |
| 6 | 31.76 | 29.76 | 29.76 | +9.32 |
| 8 | 39.56 | 30.01 | 30.01 | +9.56 |
| 9 | 32.58 | 32.82 | 32.82 | +12.37 |

Op-level marginal n=1 -> n=5 under prod routing: **9.31 ms**, against the model-level **13.0 ms**.
Same phenomenon, and the 1.4x gap is in the expected direction — these are isolated ops with warm
clocks and no competing work. The gate was "within roughly 2x"; it passes, so the defect is located
and M1 is justified.

Effective bandwidth at n=1 is **804 GB/s** across the whole forward's matmuls (16.44 GB / 20.45 ms),
against 692 GB/s measured in-model including attention, GDN and sampling. The n=1 path needs
nothing, exactly as §7.3 said.

Attribution of the 9.31 ms:

| tensor | ty | x | n=1 MMVQ | n=5 MMQ | delta | share |
|---|---|---|---|---|---|---|
| ffn_down | q4_0 | 64 | 60.6 | 92.2 | +2.02 ms | 21.7% |
| ffn_up | q4_0 | 64 | 60.8 | 92.0 | +2.00 ms | 21.5% |
| ffn_gate | q4_0 | 64 | 60.8 | 92.0 | +2.00 ms | 21.5% |
| attn_qkv | q6_K | 48 | 58.0 | 74.3 | +0.78 ms | 8.4% |
| ssm_out | q8_0 | 48 | 41.8 | 58.0 | +0.77 ms | 8.3% |
| attn_gate | q4_K | 48 | 24.8 | 38.0 | +0.64 ms | 6.8% |
| attn_q | q4_0 | 16 | 44.1 | 67.7 | +0.38 ms | 4.1% |
| output head | q8_0 | 1 | 1523.4 | 1842.2 | +0.32 ms | 3.4% |
| attn_output, attn_k, attn_v | q4_0 | 16 each | | | +0.40 ms | 4.3% |

**Q4_0 owns 73% of the marginal; the three FFN tensors alone own 65%.**

### 0.5 What this means for the three candidate attacks

The brief asked which of (a) new kernel, (b) MMQ tile specialisation, (c) batched-MMVQ to pursue.
The sweep answers it.

**MMQ's fixed penalty is mostly slower weight streaming, not serialized arithmetic.** The J=8 vs
J=16 delta separates the two: ffn_up costs 92.0 us at J=8 and 103.2 us at J=16, so eight extra
columns of arithmetic cost 11.2 us, i.e. **~1.4 us per column**. Subtracting that from J=8 leaves
~81 us of stream-and-overhead against MMVQ's 61 us for the same 50.1 MB — **618 GB/s vs 825 GB/s**.
So of the 31 us step, roughly 11 us is arithmetic for eight columns (of which only five are wanted)
and roughly 20 us is the MMQ load path being slower per byte than MMVQ's.

The occupancy numbers are consistent with a memory-level-parallelism shortfall. `stream_k` is true
for every one of these configs, and `launch_mul_mat_q` sizes the stream-k grid at **`nsm` = 82
blocks** whenever tile efficiency is under 90% — which it is for every qwen shape (ffn_up: 136 tiles
over 82 SMs = 82%). So these launches run **one block per SM, 256 of 1536 threads, 17% occupancy**.
Headroom exists: the x tile is `I * sram_stride * 4` = 128 * 76 * 4 = 38.9 KB plus ~2 KB for y and
ids = 40.0 KB against GA102's 101376-byte opt-in limit (2 blocks/SM), and `ptxas` reports
`mul_mat_q<Q4_0, J=8, fallback=0>` at **90 registers, no spills** with 256 threads, which also
allows 2 blocks/SM. Neither limit is the binding one today — the *grid size* is.

That re-verifies §9.3 item 1 and corrects it: the v4 note's "33% occupancy, smem-capped" is the
smem *ceiling*, but the kernel actually achieves half of that because the grid is capped at one
block per SM.

**A constraint that rules out most of the obvious tuning:** `I` is pinned to `nthreads/2`. The
warp-to-row map is `i0 = (threadIdx.y / ntx) * (ntx * tile_C::I)` (`mmq.cuh:484`) with
`rows_per_warp = 16` at J=8, so warp `nwarps-1` addresses row `16*(nwarps-1)` and needs
`I >= nthreads/2`. Stock `256/128` satisfies it exactly. This was confirmed the expensive way: arm C
below set `I=64, nthreads=256` — legal under every `static_assert` in the CASE macro — and faults
with an illegal memory access. So you cannot raise threads per block, and lowering I to cut smem
also halves the threads. **The only lever on occupancy is the grid size.**

So **(b) is what to try**, in two forms: patch 0002's I=64 recipe at J=8 (arm B), and lifting the
stream-k grid above one block per SM (arms F/G).

On the dead-end question the brief asked to settle explicitly:

- **"MMQ I=32 tiles, occupancy=2 hints: neutral-to-worse, I=64 is the in-family optimum"** — this
  dead end does not block arm B, it *predicts* it. It was measured on muse at J=16 and it concluded
  I=64 was optimal. Qwen's J=8 path is at I=128 and has never been tried at I=64.
- **"MMVQ launch-geometry / one-warp-per-token"** — distinct from multi-column reuse, as the brief
  suspected. That dead end was block/warp mapping at nb=1. But (c) is now *deprioritised for a
  different reason*: MMVQ's per-column slope in the n>=4 region is 9-13 us against MMQ's 0.3 us, so
  even a perfect x-reuse rewrite of MMVQ is chasing MMQ's existing marginal, while MMQ's problem is
  a fixed 30 us step that x-reuse does nothing about. If arm B fails, (c) becomes interesting again,
  and the specific finding to build on is that `vec_dot_q_cuda` is called inside the `j` loop with a
  j-independent x index (`mmvq.cu:619-624`), so x is re-addressed and re-unpacked per column.
- **Stream-k splitting** — not re-opened. `ggml_cuda_mmq_get_stream_k` is already true for every one
  of these configs, so qwen's verify matmuls run stream-k today.

### 0.6 A hypothesis about the env cap — **disproved e2e in §1.4, do not act on this section**

Under prod's `GGML_CUDA_MMVQ_NE11_MAX=3`, nb=5 routes to MMQ at 29.75 ms. Staying on MMVQ at n=5
costs **27.35 ms** — 2.4 ms per forward *cheaper*, op-level. The op-level crossover is at n≈6, not
the recorded n≈3.4.

This contradicts the wave-4 e2e result that set the cap ("n4 without the env is SLOWER than n2"),
but that arm moved draft-n and the cap together, and wave-5 §4.3 already found the recorded
justification unsound for muse. It is one cheap e2e arm to settle and is bundled into the M1 block.
Treat the op-level number as a hypothesis, not a result: e2e wins ties.

---

## M1 — four arms, gate FAILED, and one real but sub-threshold gain

Gate from the brief: marginal cost of four verify columns <= 6 ms at model level. Op-level that is
**<= 4.3 ms** (scaling by the 9.31/13.0 factor measured in §0.4), i.e. a per-forward n=5 total of
**<= 24.8 ms** against today's 29.75 ms. The floor if the extra columns were free is 20.5 ms.

### 1.1 The arms

All change `mmq-config-ampere.cuh`'s eight J=8 `CASE` lines for Q4_0/Q4_K/Q6_K/Q8_0, plus for F/G a
vendored env-gated change to `launch_mul_mat_q` (`GGML_CUDA_MMQ_SMALLN=1`) that raises the stream-k
grid from `nsm` to `nsm * min(8, smpbo/nbytes_shared)` when `J <= 8`. Patches in `$SCRATCH/w6-patches/`.

| arm | config at J=8 | grid | blocks/SM | threads/SM |
|---|---|---|---|---|
| stock | `256, 1, 128, 8` | 82 | 1 | 256 (17%) |
| B | `128, 1, 64, 8` | 82 | 1 | 128 (8%) |
| C | `256, 1, 64, 8` | 82 | — | **faults**, see §0.5 |
| F | `256, 1, 128, 8` | 164 | 2 | 512 (33%) |
| G | `128, 1, 64, 8` | 328 | 4 | 512 (33%) |

`$SCRATCH/w6_block1.sh`, `w6_block2.sh`; logs `$SCRATCH/kraw/w6/tbo_Q*.log`; analysis `w6_cmp.py`.

### 1.2 Op-level results

Per-forward matmul total (ms), same summation as §0.4:

| n | MMVQ | MMQ stock | MMQ B | MMQ F1 | MMQ F0 | MMQ G1 |
|---|---|---|---|---|---|---|
| 1 | 20.45 | 20.55 | 20.46 | 20.48 | 20.54 | 20.55 |
| **5** | 27.35 | **29.75** | 30.12 | 29.28 | 29.68 | **27.76** |
| 8 | 39.56 | 30.01 | 30.45 | 29.47 | 30.06 | 28.02 |

Marginal cost of columns 2..5, against the gate of <= 4.3 ms op-level:

| arm | marginal | model-level equivalent | gate |
|---|---|---|---|
| MMQ stock | +9.31 ms | ~13.0 ms | — |
| MMQ arm B | +9.67 ms | ~13.5 ms | worse than stock |
| MMQ arm F | +8.83 ms | ~12.3 ms | fail |
| **MMQ arm G** | **+7.31 ms** | **~10.2 ms** | **fail** |

`F0` (the same binary with `GGML_CUDA_MMQ_SMALLN=0`) lands at 29.68 vs stock's 29.75, so the env gate
is a clean no-op and the A/B is within-build.

**Arm B is worse than stock**, and its per-type split explains why: it *helps* the byte-light types
(ssm_out Q8_0 0.82x, attn_gate Q4_K 0.94x, attn_qkv Q6_K 0.93x) and *hurts* the Q4_0 FFN tensors that
own 65% of the marginal (1.07-1.08x). Halving `nthreads` to keep the I=nthreads/2 ratio dropped
threads/SM to 128, which is the wrong direction for a bandwidth-starved launch. This is the cleanest
evidence that the limiter is memory-level parallelism.

**Arm G is the best of the family at 27.76 ms**, a 6.7% cut of the matmul total, and it needs both
changes: the grid multiplier alone (F) gives 1.6%, and the I=64 config alone (B) is negative. It is
also not uniformly good — it regresses ffn_down (1.085x, the long-K shape) and attn_k/attn_v (1.19x,
the m=1024 shapes where 328 blocks is far more than the 8-16 tiles available), and wins on ffn_up /
ffn_gate (0.884x), ssm_out (0.824x), attn_gate (0.859x), attn_q (0.885x). A shape gate would recover
maybe another 0.5 ms.

**Correctness**: `test-backend-ops test -o MUL_MAT` on arm G with `GGML_CUDA_MMQ_SMALLN=1` passes
**1267/1267**, including the added qwen cases. Arm B was never correctness-checked (it is negative
anyway). Arm C faults, as recorded above.

### 1.3 e2e — arm G is +4.4%, not the +13.5% the raw t/s shows

`$SCRATCH/w6_block3.sh`, `w6_block4.sh`, prod flags on `v9-tree` (control) and `w6-G`, mid1k,
draft-n 4, `LLAMA_SPEC_PROF=1`. The content-independent number is the `target_decode_hist` n=5 mean,
because a kernel numerics change alters the generated text and therefore the acceptance rate:

| arm | verify n=5 | vs control | raw t/s | draft_n | acceptance | mean draft len |
|---|---|---|---|---|---|---|
| cap 3 (prod code, control) | **38.31 ms** | — | 70.60 / 70.45 | 390 | .567 | 3.26 |
| cap 5 | **44.90 ms** | **+17.2%** | 64.35 / 64.23 | 370 | .608 | 3.59 |
| **arm G, cap 3** | **36.34 ms** | **-5.1%** | 80.07 / 79.96 | 357 | .641 | 3.59 |

The control reproduces wave-5's 38.3 ms verify exactly, which validates the harness.

**Arm G's raw +13.5% t/s is mostly acceptance luck.** Verify time fell 1.97 ms (-5.1%) while mean
draft length rose 3.26 -> 3.59 (+10%) purely because the changed numerics produced different text —
exactly the trap recorded in `project_muse_glimmer_decode_perf`.

At constant acceptance the gain is **+3.8% to +4.4%**, by two derivations that agree to 0.3 ms:

- **measured** cycle time from the harness: control 4518 ms / 97.5 blocks = 46.34 ms, arm G
  3984 ms / 89.25 blocks = **44.64 ms**, i.e. -3.7% cycle = **+3.8% decode**.
- **modelled** from verify plus wave-5's mid1k non-verify remainder of ~8.0 ms: 44.37 ms, **+4.4%**.

Quote the range, and quote it as **mid1k only**. Deep context is untested here and would be lower:
wave-5 measured verify at 54k as 41.4 ms with a 9.8 ms remainder, so the same 1.97 ms saving is a
smaller share of a longer cycle (~3.9% of cycle rather than 4.3%). Both arms are a single run pair
(two requests each, reproducible to 0.15%), on one payload.

**The op-level harness transfers 1:1 for the MMQ path**: it predicted -1.99 ms/forward, e2e measured
-1.97 ms of verify. That is a useful, reusable result — this sweep can be trusted for future MMQ work
without a model load.

Marginal cost after arm G: verify n=5 36.34 ms against the unchanged n=1 floor of 25.3 ms =
**11.0 ms**, against the gate's 6 ms. **M1 gate failed.**

### 1.4 The env cap: prod's 3 is correct, and now for a measured reason

The op-level hypothesis from §0.6 is **wrong e2e and the error is large**: MMVQ at nb=5 costs
44.90 ms of verify against MMQ's 38.31, i.e. **+6.6 ms**, where the op sweep predicted -2.4 ms. So
`GGML_CUDA_MMVQ_NE11_MAX=3` stays, and wave-4's decision is vindicated on a clean
content-independent measurement rather than the coupled arm that wave-5 §4.3 criticised.

**Methodological finding worth carrying forward: `test-backend-ops perf` transfers for the MMQ path
and does not transfer for the MMVQ multi-column path.** The likely reason is in the source: MMVQ's
`has_fusion` path carries `GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1")`
(`mmvq.cu:826`), so in-model MMVQ at nb>=2 loses the fused ffn_up+ffn_gate launch that nb=1 gets and
that the unfused op harness never modelled on either side. MMVQ's curve is also non-monotonic around
n=4-5 for ffn_down (74.7 / 100.2 / 81.3 us at n=3/4/5), so n=5 is not a stable operating point.

---

## Verdict: NO SHIP

- **M0 passed** and is the durable result: the 13.0 ms is in the quantized weight matmuls, it is a
  **fixed ~31 us per-call step** when the batch leaves 1 (about 20 us of slower streaming, 618 vs
  825 GB/s, plus ~11 us of arithmetic for eight columns of which five are wanted), and the mechanism
  is a stream-k grid capped at one block per SM with `I` pinned to `nthreads/2`.
- **M1 failed.** The best arm cuts the marginal cost from 13.0 to 11.0 ms model-level and buys
  **+4.4% decode**, against a **>= +10% ship bar**. Three of the four arms are neutral-to-negative.
- **M2 partially executed, gated on a ship decision that M1's bar denies.** Arm G exists as an
  env-gated vendored patch that passes 1267/1267 MUL_MAT correctness cases and has a measured e2e
  number. It is preserved and revertible. A future wave that wants it needs three things this wave
  did not do: KLD vs prod Q4_K_L, gemma and muse smoke tests (the grid change touches **every**
  J<=8 MMQ launch, and both of those models route through it untested — this alone justifies
  not shipping), and a shape gate for the ffn_down and m<=1024 regressions.

- **Unexploited config lever for the next wave, measured here but not tested:** stock MMQ costs
  29.50 ms per forward at nb=2 and 30.01 ms at nb=8 — a verify batch of eight costs **1.7% more than
  a batch of two**, because the J=8 tile computes eight columns either way. Prod runs draft-n 4
  (nb=5). So on the matmul side, additional speculative tokens up to nb=8 are nearly free, and the
  binding constraint is acceptance and drafter cost, not verify. Testing `--spec-draft-n-max 6` and
  `7` (nb=7, 8) against the existing harness is cheap and is a config change rather than a kernel
  one. Note that nb=9 crosses to the J=16 tile and costs 9% more, so 8 is the ceiling.
- **Do not pursue (c) batched-MMVQ.** MMVQ's per-column slope at n>=4 is 9-13 us against MMQ's
  1.4 us, so perfect x-reuse in MMVQ would still be chasing a marginal cost MMQ already beats, and
  it does nothing about MMQ's fixed step. The specific reuse opportunity does exist and is recorded
  for completeness: `vec_dot_q_cuda` is called inside the `j` loop with a j-independent x index
  (`mmvq.cu:619-624`).
- **What would actually close the gap is (a), a real kernel:** the remaining 20 us per call is the
  MMQ global->shared load path running at 618 GB/s where the direct-to-register MMVQ path gets 825.
  Fixing that means software pipelining / `cp.async` double-buffering of the x tile so the load phase
  is not fenced by `__syncthreads` against the MMA phase, with only 2 resident blocks to hide it.
  That is a multi-week rewrite, it is upstream's territory, and this wave does not recommend starting
  it on the strength of a 13 ms prize of which config tuning already took 2 ms.

## Reproduction

```bash
S=/tmp/claude-1000/-data-docker-services/d63db54f-086a-496c-b46a-0536dd8d88b7/scratchpad
$S/w6_acquire.sh                      # /running gate, waits for idle, unloads, takes gpu.lock
$S/w6_block0.sh                       # M0: MMVQ + MMQ op sweep at qwen shapes, n=1..9
python3 $S/w6_parse.py                # reconciliation against the 13.0 ms
$S/w6_block1.sh B                     # arm B
$S/w6_block2.sh                       # arms F1/F0/G1
python3 $S/w6_cmp.py B F1 F0 G1
$S/w6_block3.sh                       # e2e cap 3 vs cap 5
$S/w6_block4.sh                       # e2e arm G
$S/a2b_release.sh
```

Trees: `$SCRATCH/kernel/w6-tree` (v9 + test cases), `w6-B`, `w6-C` (faults), `w6-F`, `w6-G`.
Patches: `$SCRATCH/w6-patches/` — `testcov-mulmat-qwen38-verify.patch` (worth keeping regardless;
it is the op-level harness that transfers 1:1 for MMQ), `0007-mmq-smalln-streamk-grid.patch`,
`armG-combined-i64-nthr128-plus-grid.patch`.
