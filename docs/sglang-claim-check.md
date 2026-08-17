# Does SGLang's "beyond 100 tok/s" claim survive contact with our RTX 3090?

**Verdict: (b) — true, single-request, and well-documented, but only on Blackwell-class
hardware with a weight format the RTX 3090 cannot execute.**

The number is real and it is *not* batch-aggregate: SGLang's own validated benchmarks
report 152.9 tok/s/user at concurrency 1 with 8192 input tokens. Every one of those
measurements is on an **RTX 5090 (Blackwell, sm_120, 32 GB, ~1792 GB/s)** using
**NVFP4**, which needs FP4 tensor cores the 3090 does not have. On our card there is
no validated path, and the honest expected value even if the unvalidated paths worked
is **parity with what we already run**, not a win.

The blocker is SGLang's Ampere software support, **not** our memory bandwidth. Saying
"physics forbids 100 t/s on a 3090" would be wrong, and section 5 explains why.

Literature check only — nothing was installed or run.

**The finding most likely to change what you do:** SGLang's own profiler says Gated
DeltaNet decode is **~2% of step time** and already fused, while the quantized weight
GEMM is **~61%** (≈77% on a single card, where there is no NCCL tax). See "Where the
two weeks should go."

---

## 1. Does SGLang support Qwen3.8-27B at all, and its MTP head?

**Yes to both, day-0, and the MTP support is genuine — not an EAGLE-only substitute.**

SGLang ships an official cookbook page
([docs.sglang.io](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B))
describing exactly our architecture: 48 Gated DeltaNet linear-attention layers to 16
full-attention layers, GDN at 48 value heads / 16 QK heads at head_dim 128, Gated
Attention as GQA 24/4 at head_dim 256, 262,144 native context. It states that "the
serving-relevant architecture is identical to Qwen3.6-27B," which lets us treat
Qwen3.6-27B field reports as evidence about Qwen3.8-27B.

On the MTP head, the cookbook is unambiguous — the "EAGLE vs native MTP" distinction
in the question is a false dichotomy here, because `EAGLE` is just the flag name and
the in-checkpoint MTP head is the drafter:

> `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
> --speculative-num-draft-tokens 4` uses the in-checkpoint MTP head. (This recipe was
> originally documented with `NEXTN`, an alias of `EAGLE` — same algorithm.)

A third option, **DSpark**, uses a separately trained draft checkpoint
(`RadixArk/Qwen3.8-27B-DSpark`) and produces SGLang's headline number.

SGLang also appears to have solved the GDN-state-plus-speculation problem that bit
vLLM. An independent 4×4090 tuning study
([xltzsoft/qwen3.8-27b-4x4090-sglang-tuning](https://github.com/xltzsoft/qwen3.8-27b-4x4090-sglang-tuning))
verified in SGLang 0.5.17 source that the post-verify Mamba state commit uses a fused
gather-scatter at fixed O(1) cost, with no O(sequence length) replay. They attribute
the widely repeated "EAGLE/MTP collapses at long context on hybrid linear-attention
models" belief to a benchmarking artifact — SGLang's first post-prefill
`gen throughput` log line divides by wall time that includes the entire prefill —
plus two fixable OOMs.

**Caveat, and it matters:** this is a **single unreviewed community source**, and it
contradicts club-3090's FAQ, which states EAGLE is "blocked on Qwen3-Next by DeltaNet
hybrid attention's lack of KV rollback support in vLLM/SGLang"
([club-3090 FAQ](https://github.com/noonghunna/club-3090/blob/master/docs/FAQ.md)).
I weight the 4×4090 study higher because it is newer, is specifically on Qwen3.8-27B,
and cites source-level verification with a profiler — but the "MTP works at depth in
SGLang" pillar rests on it alone and should be treated as probable, not established.

---

## 2. Where does "beyond 100 tok/s" come from?

I did not find the originating Reddit/chat comment and stopped looking, because the
upstream sources it must be echoing are better provenance than the comment. There are
two, and **every number in both is Blackwell consumer or datacenter silicon.**

**Source A — SGLang's launch announcement:**

> Day-0 support is live in SGLang: 206.1 tok/s decode on a single RTX 5090, with our
> NVFP4 plus DSpark — 38.28 tok/s decode on DGX Spark

([@sgl_project](https://x.com/sgl_project/status/2088281320422322413); read via
search-engine snippet — x.com and the xcancel mirror were both behind anti-bot walls,
so treat the exact wording as high-confidence but **not directly verified**.)

**Source B — the cookbook's validated table**, directly verifiable and more complete.
Discussing `--mamba-ssm-dtype`, it reports per-user decode for fp32 vs bf16 GDN state:

> with speculative decoding fp32 sometimes wins (NVFP4 + EAGLE: 152.9 vs 144.5
> tok/s/user) and sometimes loses (FP8 + EAGLE: 106.3 vs 116.1)

Measurement conditions are stated plainly, and they are *favorable* to the claim:

> The RTX 5090 and RTX PRO 6000 cells above — including every Speculative Decoding /
> Serving Strategy / SSM dtype combination — were validated at **ISL 8192 / OSL 1024,
> concurrency 1.**

**Against the question's checklist:** GPU = RTX 5090 (other cells: RTX PRO 6000, H200,
DGX Spark), never a 3090. Model = Qwen3.8-27B, correct size. Quantization = NVFP4
(W4A4) for 152.9 and 206.1; FP8 for 106.3. Batch = **concurrency 1, genuinely
single-request** — the docs say "tok/s/user." Depth = **8192 input tokens**, so it is
not a shallow-only number either.

**Do not dismiss this as batch-aggregate throughput.** It isn't, and that dismissal
would be wrong and easy to catch. The claim is honest on its own terms. What makes it
inapplicable is exclusively the hardware and weight-format column.

---

## 3. Any credible report of SGLang >100 t/s single-request on a 27B W4 model on a 3090?

**No. The one serious attempt is documented as a failure.**

The closest comparable effort is a detailed single-3090 build-out for Qwen3.6-27B —
same 24 GB Ampere card, same GDN hybrid architecture, same MTP head — reaching
**85 TPS sustained / 106 TPS peak at 125K context with vision, 21.3/24 GB, at 230 W**
([Medium write-up](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914)).
That is on **vLLM**. SGLang appears in the same article's "What didn't work" list:

> **SGLang + Lorbus + EAGLE spec-dec.** SGLang's Marlin `gptq_marlin_repack.cuh` has
> the same class of pad-sub-tile-n bug we fixed in vLLM (our PR #40361). Rejects
> `size_n=96` on Qwen3-Next's DeltaNet sub-block projections. Blocked pending upstream
> fix.

So the best-documented attempt to put SGLang on exactly our hardware and architecture
could not get 4-bit weights to load.

Supporting evidence that this is a standing upstream gap rather than one bad afternoon:

- **[Issue #24589](https://github.com/sgl-project/sglang/issues/24589)** —
  Qwen3.6-27B AWQ-INT4 on RTX 3090s (compute capability 8.6 confirmed in the
  environment dump), SGLang 0.5.11. Fails at precisely the layers in question: "loads
  up to the point of initializing the `Qwen3_5GatedDeltaNet` layers … then fails with
  `NotImplementedError: No compressed-tensors compatible scheme was found.`" Filed
  2026-05-07, **bot-closed for inactivity 2026-08-09 with no fix** (verified via the
  comments API; the only human reply is an unrelated Gemma-4 "same problem").
- **[Issue #19406](https://github.com/sgl-project/sglang/issues/19406)** — the Marlin
  `size_n not divisible by tile_n_size = 64` repack failure on Qwen3.5-27B AWQ/INT8.
  GitHub records `state_reason: completed`, but the closing comment is the inactivity
  bot, so read it as lapsed rather than resolved.
- **[PR #20370](https://github.com/sgl-project/sglang/pull/20370)**, the community fix,
  **closed 2026-08-10 without merging** (API: `merged: false, merged_at: null`). Its
  root cause is instructive: a name-mapping bug caused `in_proj_a`/`in_proj_b` (width
  32) to be quantized when the quant config meant to ignore them, so they hit Marlin.

**One honest qualification:** PR #20370 addresses a `size_n = 32` failure, while the
Medium article reports `size_n = 96`. Those are different widths and possibly
different call sites, so I am not claiming they are one bug. What I *can* state is
that **I found no merged fix for either width** — #20370 was the only candidate and it
closed unmerged. The section 4 argument does not depend on this point.

The cookbook corroborates the gap by omission: its hardware sections cover **H200
(SM90), RTX PRO 6000 Blackwell and RTX 5090 (SM120), and DGX Spark (SM121)**. There is
**no Ampere / sm_86 recipe or validated cell anywhere on the page.**

---

## 4. Practical viability on our rig, even accepting a slower result

**Checkpoint fit is the binding constraint.** The cookbook offers three checkpoints
and states their sizes:

| Checkpoint | Format | Size | Fits 24 GB? |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` | BF16 | ~55 GB | No |
| `Qwen/Qwen3.8-27B-FP8` | FP8 blockwise | **~28.5 GB** | No — and Ampere has no FP8 tensor cores |
| `RadixArk/Qwen3.8-27B-NVFP4` | NVFP4 W4A4 + FP8 projections | **~16.5 GB** | Only size-wise; see below |

The FP8 figure is the cookbook's own ("FP8 weights ~28.5GB … NVFP4 weights ~16.5GB
(recommended for RTX 5090-class GPUs)"). It does not fit before a single KV token.

So NVFP4 is the only candidate. I inspected its actual `config.json` on Hugging Face
rather than assuming, and the result is more nuanced than a hard refusal:

- `quant_method: **modelopt**` (not compressed-tensors)
- `group_0`: **W8A8 FP8** across **208 targets** — the `linear_attn.in_proj_qkv`,
  `in_proj_z`, `out_proj` and `self_attn.{q,k,v,o}_proj` projections
- `group_1`: **W4A4 FP4, group_size 16**, 193 targets
- `ignore: ['mtp*', 'mtp.layers.0*']` — **the MTP head is left unquantized**, which is
  good for draft quality (the same trick that made single-card MTP work on vLLM) but
  costs VRAM

This matters because the two NVFP4 code paths in SGLang are gated differently on
current `main`:

- `compressed_tensors_w4a4_nvfp4.py` → `get_min_capability() = **100**` (Blackwell
  only). **Not the path this checkpoint takes.**
- `ModelOptFp4Config.get_min_capability() = **80**` — Ampere included. **This is the
  path it takes**, and a dense W4A16 Marlin fallback genuinely exists on main
  (`prepare_nvfp4_layer_for_marlin` / `apply_fp4_marlin_linear`, selected by
  `get_fp4_gemm_runner_backend().is_marlin()`, exposed as
  `--fp4-gemm-runner-backend`).

**So the accurate statement is not "it cannot load" — it is "every route is either
unvalidated or known-broken, and SGLang has twice declined to support this one."**
Four independent signals:

1. **SGLang greys out this exact combination on a $30k datacenter GPU.** Of H200 the
   docs say: "the card has no FP4 tensor cores, so the NVFP4 checkpoint's MLP would
   fall back to the Marlin W4A16 weight-only path **and its cell is greyed out.**" A
   3090 hits the same fallback with strictly less silicon.
2. **The dedicated non-Blackwell fallback was merged and reverted within 17 hours.**
   [PR #19652](https://github.com/sgl-project/sglang/pull/19652), "NVFP4 Marlin
   fallback for non-Blackwell GPUs (SM75+)", explicitly names our card in its
   motivation ("forces users on A100/A40/H100/**RTX 3090** to fall back to less
   accurate quantization"). Merged 2026-04-03 02:48 UTC; reverted by
   [PR #22047](https://github.com/sgl-project/sglang/pull/22047) at 20:12 UTC the same
   day. The [reland, PR #22513](https://github.com/sgl-project/sglang/pull/22513),
   **closed unmerged 2026-06-06**. Broader compressed-tensors NVFP4 Marlin support is
   still an **open, unmerged PR (#34966)** today.
3. **The FP8 half has no native Ampere path.** All 208 attention and GDN projections
   are W8A8 FP8, and Ampere has no FP8 tensor cores, so they need their own Marlin
   W8A16 fallback. For reference, `ModelOptFp8Config.get_min_capability()` returns
   **89** (Ada) — above our 86 — though the mixed-precision config reports 80 via the
   FP4 class, so what actually happens on boot is untested and unpredictable from
   source alone.
4. **The narrow GDN projections are exactly what trips Marlin repack.** The widths
   that failed in #19406/#20370 (32) and in the Medium report (96) are GDN sub-block
   projections against Marlin's 64-tile requirement, and no fix has merged.

**Remaining sub-questions, answered only where evidence exists:**

- *W4A16/AWQ/GPTQ on sm_86:* Marlin is an sm_80+ kernel, so Ampere is not
  categorically excluded. The blocker is model-specific (GDN projection widths), not
  architectural.
- *Fast paths on Ampere:* several are gated off. Linear-attention backends `cutedsl`
  and `flashinfer` are "CUDA/SM100-only (gated behind `is_cuda()` /
  `is_sm100_supported()`)" ([issue #31594](https://github.com/sgl-project/sglang/issues/31594)),
  leaving `triton` as the only linear-attn backend on a 3090.
- *Max context on 24 GB:* **unverifiable — no source measures it**, because nobody has
  gotten this model onto an Ampere card under SGLang. For scale, the GDN state pool
  alone costs 153.9 MB/slot at fp32 or 78.4 MB at bf16, and the default `extra_buffer`
  strategy consumes 5 slots per in-flight request.
- *Cold start:* **unverifiable for our configuration.** The docs warn CUDA graph
  capture "can take up to several minutes."
- *GDN state + MTP together:* works, per the 4×4090 study (with the single-source
  caveat from section 1).

---

## 5. Honest comparison against our numbers, including depth

Our baselines: llama.cpp with our patches at 75.7 / 79.6 / 68.2 / **62.9 t/s** across
shallow / mid-1k / 7k / **53.4k**, 131k context resident, no cold start; vLLM at
84–91 t/s greedy shallow, no published depth curve.

| Config | Hardware | Decode | Notes |
|---|---|---|---|
| SGLang NVFP4 + DSpark | 1× RTX 5090 | 206.1 t/s | depth not stated |
| SGLang NVFP4 + EAGLE/MTP | 1× RTX 5090 | 152.9 t/s | ISL 8192, concurrency 1 |
| SGLang FP8 + EAGLE/MTP | 1× RTX 5090 | 106.3–116.1 t/s | ISL 8192, concurrency 1 |
| SGLang FP8 + EAGLE/MTP | **4× RTX 4090, TP4** | ~142 short, **107–148 @ 50k** | **four cards**, accept length 2.9–3.3 |
| SGLang FP8, no speculation | **4× RTX 4090, TP4** | 74.85 t/s | **four cards**, flat to 50k |
| vLLM AutoRound INT4 + MTP n=3 (Qwen3.6-27B) | **1× RTX 3090** | 85 sustained / 106 peak | 125K ctx, AL 3.4–3.8 |
| **our llama.cpp** | **1× RTX 3090** | **75.7 / 79.6 / 68.2 / 62.9** | **measured to 53.4k** |
| **our vLLM** | **1× RTX 3090** | **84–91** | shallow only |

Note carefully that the two 4090 rows are **TP4 across four cards**, roughly 4× our
aggregate bandwidth. They are not a single-card depth curve.

**First: 100 t/s on a 3090 is not physically forbidden.** Our no-speculation ceiling
estimate is ~55–60 t/s, and the 4×4090 study measured **accept length 2.9–3.3 on this
exact model**. Even at an effective multiplier well below the raw accept length, the
product lands in the 90–115 range, and the single-3090 vLLM result (85 sustained /
106 peak) empirically confirms the neighborhood is reachable on Ampere. The blocker is
SGLang's sm_86 support, not our bandwidth.

**Second: normalizing SGLang's best supported number predicts parity, not a win.**
Treat this as a rough **upper-bound sketch**, not an estimate — it assumes NVFP4 W4A4
and Marlin W4A16 move comparable bytes and sustain comparable accept length, which is
generous, since W4A4 also halves activation traffic and the 5090 executes FP4
natively. With that caveat: the 5090's ~1792 GB/s against the 3090's ~936 GB/s is a
1.9× gap, so scaling 152.9 down by bandwidth alone gives roughly **80 t/s** — *below*
our existing vLLM 84–91 and level with llama.cpp's 75.7. The 4×4090 datapoint points
the same way independently: four Ada cards with working EAGLE/MTP reach ~142 t/s
short-context, and one Ampere card at a quarter of that bandwidth does not arrive at
100+ by another route.

**Third: on depth, SGLang would genuinely be attractive if it ran.** 107–148 t/s at
50k is a flat-to-improving curve (albeit on four cards), and our 62.9 t/s at 53.4k is
our weakest number. Depth is the one axis where SGLang's approach beats ours. It is
simply unavailable on this card.

---

## Verdict and where the two weeks should go

**Classification: (b).** True, single-request, well-documented — exclusively on
Blackwell-class hardware with a weight format the 3090 cannot execute. On our rig the
honest expected value is roughly break-even with what we already run, gated behind one
reverted feature, one unmerged reland, one unmerged bug fix, and two lapsed bug
reports. **Do not spend two weeks chasing it.**

**More useful than the verdict:** the 4×4090 study profiled SGLang decode on this
exact model and published the step breakdown, which bears directly on where CUDA
effort pays off:

- **Gated DeltaNet decode: 0.29 ms of a 14.1 ms step — about 2%**, already a single
  fused kernel (`fused_recurrent_gated_delta_rule_packed_decode`, fusing L2norm +
  softplus gating + delta rule). Their explicit conclusion: **don't rewrite it.**
- **W8A8 FP8 GEMM: 61%**, already near the memory-bandwidth limit.
- **NCCL all-reduce: 21%** — a multi-GPU tax that does not exist on our single card,
  which means the GEMM share on our rig is effectively **~77% of every decode step**.

If the proposed rewrite targets **GDN kernels**, this is direct measured evidence
against it: there is at most 2% of the step to win. If it targets the **quantized
weight GEMM path**, that is where roughly three-quarters of every decode step goes.
That is consistent with our own shipped wins in
`k8s/workloads/apps/llama/PERFORMANCE.md`, which came from attention and MMVQ/GEMM
kernels rather than from the recurrent path.

---

## Sources

- SGLang cookbook, Qwen3.8-27B — https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- SGLang launch post, 206.1 tok/s on RTX 5090 — https://x.com/sgl_project/status/2088281320422322413 *(via search snippet; direct fetch blocked by anti-bot)*
- 4×RTX 4090 SGLang tuning study, Qwen3.8-27B-FP8 (single unreviewed source) — https://github.com/xltzsoft/qwen3.8-27b-4x4090-sglang-tuning
- Single-RTX-3090 Qwen3.6-27B stack, 85/106 TPS on vLLM; SGLang listed as failed — https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914
- NVFP4 checkpoint quant config (modelopt, W8A8 + W4A4, MTP unquantized) — https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4/raw/main/config.json
- SGLang #24589, AWQ-INT4 GatedDeltaNet load crash on RTX 3090 — https://github.com/sgl-project/sglang/issues/24589
- SGLang #19406, Marlin repack `size_n` alignment, Qwen3.5-27B AWQ/INT8 — https://github.com/sgl-project/sglang/issues/19406
- SGLang PR #20370, candidate fix — closed unmerged 2026-08-10 — https://github.com/sgl-project/sglang/pull/20370
- SGLang PR #19652, NVFP4 Marlin fallback for non-Blackwell (SM75+), names RTX 3090 — https://github.com/sgl-project/sglang/pull/19652
- SGLang PR #22047, the revert, 17 hours later — https://github.com/sgl-project/sglang/pull/22047
- SGLang PR #22513, reland — closed unmerged 2026-06-06 — https://github.com/sgl-project/sglang/pull/22513
- SGLang #31594, linear-attn backends `cutedsl`/`flashinfer` gated to SM100 — https://github.com/sgl-project/sglang/issues/31594
- club-3090 FAQ, Ampere/Ada/Blackwell dtype support matrix — https://github.com/noonghunna/club-3090/blob/master/docs/FAQ.md
- vLLM recipes entry, Qwen3.8-27B — https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml
