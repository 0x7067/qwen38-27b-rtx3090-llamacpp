# Qwen3.8-27B on one RTX 3090: llama.cpp patches and a truncated MTP draft vocabulary

This repo makes Qwen3.8-27B (hybrid Gated-DeltaNet, 27B) decode faster on one
RTX 3090 (24 GB, Ampere cc 8.6) with llama.cpp. The full 131k context stays
resident: main model, MTP drafter, and vision projector together use about
22.9 GiB. Speculative decoding runs on the model's native MTP head.

Results on the target workload (54k-token-deep agentic decode, temp 0):

- The patch stack takes decode from **49.8 to 69.3 tok/s (+39%)**. Mid-depth
  (1k) goes from **63 to 84 tok/s (+33%)**.
- The truncated draft vocabulary adds another **+5–6% at mid to 54k depth**
  and frees **0.5 GiB of VRAM**. Output stays byte-identical, verified by
  output sha256 in every A/B pair.

The approach is inspired by
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090),
which uses vLLM. This is the llama.cpp counterpart. It has no multi-minute
cold start, it hot-swaps models through llama-swap, and decode holds up at
54k depth.

## Results

Conditions: temp 0, warm, `/completion` timings, `--spec-draft-n-max 5`,
`GGML_CUDA_MMVQ_NE11_MAX=3`. "Full drafter" and "d48k drafter" run the same
image (patches 0001–0007); only the drafter file differs.

| Payload (prompt depth) | Full drafter | d48k drafter | Delta | Acceptance (full → d48k) |
|---|---|---|---|---|
| short (98 tok) | 75.3 tok/s | 75.7 | +0.5% | .527 → .488 |
| mid (870 tok, code) | 75.0 | **79.6** | **+6.1%** | .532 → .532 (identical) |
| prose (6.8k tok) | 64.7 | **68.2** | **+5.4%** | .435 → .431 |
| agentic (53.4k tok) | 59.4 | **62.9** | **+6.0%** | .439 → .439 (identical) |
| VRAM at 131k ctx | 23,438 MiB | **22,892 MiB** | −546 MiB | |

Output content is byte-identical between the two drafters for every payload,
compared by sha256. This is expected from the code: llama.cpp's verify step
samples from the target distribution and keeps a draft token only when it
matches exactly. A smaller draft vocabulary can therefore change speed, but it
cannot change output. See `docs/truncated-draft-vocab-design.md`, section 4.

Patches 0001–0006 produced the earlier 49.8 → 69.3 tok/s campaign result.
`docs/PERFORMANCE.md` is the full write-up.

## Main quant: now UD-Q4_K_XL

Prod's main-model quant moved from bartowski Q4_K_L to **unsloth
UD-Q4_K_XL**, chosen for quality: a KLD-to-Q6K sweep found UD-Q4_K_XL at
ratio **0.787** vs Q4_K_L's own 1.0 (lower is closer to the Q6_K reference),
at a ~3.4% pooled-step-time cost and a small VRAM *saving*. A faster
candidate (bartowski IQ4_XS, −7% pooled step time) was measured and rejected
for prod because its KLD ratio (1.415) is a real quality regression, not a
free lunch. Full sweep, a rejected 3.7bpw extreme, and the corrected
weight-bytes cost model: `docs/quant-selection.md`.

## The patches

Apply the patches onto llama.cpp commit `4df29be4f`, in order, with
`git apply`. Do not use `git am`: patches 0002 and 0006 are plain diffs
without format-patch headers. The `Dockerfile` does all of this for you.

| Patch | What it does |
|---|---|
| 0001 | DFlash draft-side greedy fast path, plus `LLAMA_SPEC_PROF` instrumentation |
| 0002 | Ampere MMQ small-batch (J=16) tile config: 128×64 tiles for Q4_K/Q5_K |
| 0003 | GQA-batched FlashAttention vector kernel for quantized KV, batch ≤2, cc <8.9. Reads each KV block once per GQA group instead of once per Q head, with no F16 scratch |
| 0004 | `GGML_CUDA_MMVQ_NE11_MAX` env var. Caps the MMVQ→MMQ crossover per model. The measured crossover is N≈3.4 on this GPU and model; upstream's flat 8 overpays 24–38% on speculative verify batches |
| 0005 | MMA FlashAttention reads q4_0 K/V inline (cp.async → smem → dequant in place), so it no longer dequantizes the full cache to F16 every step. Batches 2–16, D=256, GQA >4 |
| 0006 | DFlash greedy fast path requires host logits. Fixes silent acceptance=0 with GPU draft sampling |
| 0007 | Truncated draft vocabulary for MTP drafters, using the EAGLE3 `d2t` idiom. A drafter GGUF can carry a reduced LM head plus a `d2t` I64 tensor that maps head rows to target token ids. Logits scatter back to full width with −inf elsewhere. The patch is a no-op for GGUFs without `d2t` |
| 0008 | MMQ small-batch composite, env-gated `GGML_CUDA_MMQ_SMALLN` (1=stream-k grid multiplier, 2=+y-tile double buffer, 3=+m=1024 shape gate). Verify at n=5: −3.0 ms/forward → +7% mid / +5.4–6.4% deep decode. Caveat: the config-table half is compile-time and unconditional — env-off is not stock; see `docs/journal-2026-08-17.md` |

## Truncated draft vocabulary (patch 0007 and tools)

The Qwen3.8 MTP drafter has about 3.0B parameters. 2.5B of them sit in two
248,320-row embedding matrices. The input embedding is a row gather, which is
cheap. The output head is a full 0.7 GB matrix-vector product on every draft
step: **74% of the draft pass**. Truncating the head to the 49,152 tokens that
cover ~98.5% of real traffic cuts per-step draft traffic by ~62%. That becomes
the +5–6% end-to-end decode gain in the table above.

The pipeline has four steps:

1. `tools/build_draft_vocab.py` ranks token ids by frequency over a corpus
   shaped like your traffic. Ours was English technical text and code, with
   ~10% Brazilian Portuguese. The script force-includes all control and
   special tokens and the byte alphabet, then emits keep-set JSONs
   (32k/40k/48k).
2. `tools/eval_coverage.py` measures held-out and out-of-distribution
   coverage per slice. Pick the smallest set whose OOD coverage you can
   accept. We shipped 48k: 98.5% held-out, 96.5% OOD.
3. `tools/truncate_drafter.py` performs the GGUF surgery. It copies kept
   Q4_0 head rows byte-for-byte (a 5120-wide Q4_0 row is exactly 2880 B),
   keeps `token_embd` at full width (inputs are target-vocab ids), appends
   the `d2t` I64 tensor, and validates range, uniqueness, and sort order.
   `tools/validate_drafter.py` re-checks everything with 26 assertions
   against the source file.
4. Run the patched build with `--model-draft <truncated>.gguf`. To roll
   back, point `--model-draft` at the original file. The patched binary is a
   no-op without `d2t`.

## Reproduce on a fresh machine with one RTX 3090

Requirements: an NVIDIA driver compatible with CUDA 12.8 (R570 or later),
Docker with nvidia-container-toolkit, `jq`, and Python 3.10+. The GGUF surgery
uses only the standard library. Payload and vocabulary building also need
`pip install tokenizers`.

```bash
# 1. Build the image. No GPU is needed for this step.
#    It clones llama.cpp @4df29be4f, applies patches/, and builds for sm_86.
docker build -t llama:cuda-swap-v11 .

# 2. Download the models (~20.5 GB total, links below) into a $MODELS dir:
#    Qwen3.8-27B-Q4_K_L.gguf, mmproj-Qwen3.8-27B-Q8_0.gguf, mtp-Qwen3.8-27B-Q4_0.gguf.
#    Also download tokenizer.json from any Qwen3.8-27B HF repo (a few MB).

# 3. Rebuild the truncated drafter byte-for-byte, then verify it:
python3 tools/truncate_drafter.py $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf \
        data/draft_vocab_48k.json $MODELS/mtp-Qwen3.8-27B-Q4_0-d48k.gguf
python3 tools/validate_drafter.py $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf \
        $MODELS/mtp-Qwen3.8-27B-Q4_0-d48k.gguf data/draft_vocab_48k.json

# 4. Build the bench payloads. Any large code+docs tree works as the corpus;
#    a llama.cpp checkout is a good choice.
QWEN_TOKENIZER_JSON=/path/tokenizer.json CORPUS_DIR=/path/llama.cpp \
  python3 tools/make_payloads.py tools/

# 5. Run the A/B: four depths, VRAM, acceptance, and a sha256 output check.
MODELS=$MODELS tools/run_validation.sh A mtp-Qwen3.8-27B-Q4_0.gguf
MODELS=$MODELS tools/run_validation.sh B mtp-Qwen3.8-27B-Q4_0-d48k.gguf
```

Expected result: arm B beats arm A by ~5–6% at mid and deep depth, and the
output hashes match per payload. Absolute tok/s might differ from the table,
because acceptance depends on the corpus. The portable results are the A/B
delta and the hash equality.

Models:
[bartowski Q4_K_L main](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
(embeds an unused MTP block; Q8_0 embed/output), plus
[ggml-org's](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
`mtp-*-Q4_0` drafter and Q8_0 mmproj.

To serve, embed `config/llama-swap-qwen38.yaml` in a llama-swap config (the
binary ships in the image), or run the `llama-server` command from
`tools/run_validation.sh` directly. One coupling matters: draft-n 5 wins only
**with** the MMVQ cap env var. Without it, n4 is slower than n2 and nothing
fails loudly. The docs have the measurements.

## Stacked n-gram drafting (config-only, the biggest single win)

Agentic sessions spend much of their decode re-emitting files the model was
shown or just wrote. `--spec-type draft-mtp,ngram-mod` adds llama.cpp's
n-gram drafter on top of MTP: it keeps a host-RAM table of the session's
tokens and, when the last 32 tokens exactly match a stored chain
(`--spec-ngram-mod-n-match 32`; the default 24 fires on incidental quotes),
drafts up to 64 tokens of verbatim copy in one round. It preempts the MTP
drafter only on those rounds; everything else is unchanged. Same lossless
verify contract as MTP.

Measured (temp 0, v11): 8-turn cumulative file-editing session **108.8 →
260.8 tok/s** overall (warm turns 237–288); repeated queries +~50%; prose-7k
+7%; worst case −1.7% on a deliberately pathological self-repeating 54k
prompt. Zero VRAM. `tools/session_bench.py` is the workload — one-shot
benchmarks cannot see this technique, which is why early reports dismissed it.

The 200+ numbers are possible because the bandwidth ceiling applies per
verify pass, not per token: one ~30 ms pass can carry 64 draft tokens, and a
copy round drafts them for free.

## MMQ small-batch composite (wave 8, shipped as patch 0008)

The largest remaining lever is the quantized-weight GEMM (MMQ) on the
speculative verify path. The measurements behind this
(`docs/mmq-small-batch-analysis.md`):

- At batch ≥2, MMQ costs a fixed ~31 µs per call, almost independent of how
  many columns you need. ~20 µs of that is the x-tile global→shared load
  phase streaming at 618 GB/s where the direct-to-register path reaches 825.
  The load is fenced by `__syncthreads` against the compute phase, so the
  kernel waits instead of overlapping.
- A tile-and-grid reconfiguration (`experiments/armG-*.patch`) recovered part
  of this and measured +4.4% end-to-end: below the ship bar, preserved for
  reference.
- An independent SGLang profile of the same model confirms the target:
  quantized-weight GEMM is ~77% of a single-card decode step, while the
  Gated-DeltaNet path is ~2% and already fused
  (`docs/sglang-claim-check.md` — the same document checks and rejects the
  "SGLang goes beyond 100 tok/s" claim for this GPU: the numbers are real but
  Blackwell/NVFP4-only, and SGLang cannot currently load 4-bit weights for
  this architecture on Ampere).

Resolution 2026-08-17 (full story in `docs/journal-2026-08-17.md`): the
pipelining premise was **falsified by an ncu stall breakdown** — the stock
kernel sits at 56% of DRAM peak because of SM residency, not load scheduling,
and `cp.async` is structurally impossible for the x tile (nibbles unpack on
the way into shared memory). Full register pipelining recovered only ~3 µs of
a ~31 µs step: NO-GO. What survived and **shipped as `patches/0008`** is a
composite of the arm-G grid change, a type-agnostic y-tile double buffer, and
an m=1024 shape gate (one env var, `GGML_CUDA_MMQ_SMALLN`): verify at n=5
drops 38.23 → 35.24 ms at mid-1k and 41.73 → 38.63 ms at 54k — **+7% mid and
+5.4–6.4% deep decode at constant acceptance**, KLD 0.00617, correctness
1267/1267 at every level. Blast radius is structurally clean: only a backend
that both caps MMVQ and drafts ≤7 reaches the changed path. Two reviewer
notes: env-off in this binary is *not* stock (compile-time table change), and
KLD is nonzero because both changes alter float accumulation order.

## Measured dead ends

These were measured so you do not have to repeat them:

- Embedded-MTP mode (dropping `--model-draft`) routes draft logits through
  the target's Q8_0 head. Measured: −1.6% speed for −0.9 GiB VRAM. Keep the
  separate drafter.
- One draft step costs 2.25 ms marginal (measured at n=3→5), against a
  ~1.2 ms byte floor. Fixed overhead is ~45% of a draft step, which caps what
  head-shrinking can buy.
- q8_0 KV does not fit at 131k. DeltaNet leaves only 16 attention layers, so
  q4_0 KV costs ~18 KB/token. f16 KV forces a worse kernel path.
- Chain speculation (PR #27173) loses 6–12% on a single GPU.
- GDN chunked prefill (PR #26001) is lossy (KLD 0.0075).
- `docs/PERFORMANCE.md` has the full kernel-level analysis, the dead-end
  list, and the per-wave benches.

## What is reproducible from this repo alone

These parts are fully standalone and need no access to the original host:

- The image. The `Dockerfile` clones llama.cpp at the pinned ref from GitHub
  and applies `patches/`. Nothing is local.
- The truncated drafter. `data/draft_vocab_48k.json` is the exact keep-set
  that produced the shipped GGUF, and the surgery tools use only the Python
  standard library. The 32k/40k variants and the ranked frequency table
  (`data/token_freq.tsv.gz`) let you slice other set sizes without redoing
  the corpus pass. `data/coverage.json` and `data/ood_coverage.json` are the
  acceptance-risk evidence behind the 48k choice.

These parts work anywhere but default to the original machine. Override the
environment variables or edit the paths before reuse:

- `tools/build_draft_vocab.py` re-derives the keep-set from a corpus. Corpus
  roots are local paths by nature: point them at your own traffic-shaped
  text, and set the tokenizer with `QWEN_TOKENIZER_JSON`. You only need this
  script to build a different vocabulary.
- `tools/run_validation.sh` takes the models dir from `MODELS=`.
  `tools/make_payloads.py` takes `QWEN_TOKENIZER_JSON` and `CORPUS_DIR`.
  Avoid chat transcripts as corpus: at temp 0 they can make the model emit
  EOS at position 0.
- `config/llama-swap-qwen38.yaml` assumes llama-swap and this image.

## Layout

```
Dockerfile                  build llama.cpp @4df29be4f + patches (CUDA sm_86) + llama-swap
patches/0001..0007          the vendored patch stack (apply with git apply, in order)
tools/                      draft-vocab pipeline, GGUF surgery + validation, bench harnesses
                            (one-shot A/B, cumulative session, 12-turn episode metric)
data/                       keep-sets (32k/40k/48k), ranked token frequencies, coverage evidence
config/                     llama-swap model block (flags + env couplings, commented)
experiments/                measured but unshipped patches (see experiments/README.md)
docs/PERFORMANCE.md         full campaign write-up (waves, kernels, rejects, methodology)
docs/mmq-small-batch-analysis.md       MMQ verify-path analysis behind the wave-8 work
docs/journal-2026-08-17.md  day journal: what shipped, what was rejected, and why
docs/sglang-claim-check.md  the "SGLang >100 tok/s" claim, checked against this GPU
docs/truncated-draft-vocab-design.md   design doc for patch 0007 (data flow, correctness proof)
docs/thermals-and-oc.md     fan-curve dead zone, NVML offset convention, OC ladder, thermal ceiling
docs/dflash2-findings.md    DFlash2 (llama.cpp PR #27342) vs our MTP drafter: env-cap effect, caveats
docs/quant-selection.md     main-quant sweep: why UD-Q4_K_XL, the rejected 3.7bpw extreme, cost model
```

## License and credits

The `patches/` modify [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT) and follow its license. Everything else here is MIT.

Prior art and sources this work builds on:

- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) —
  the truncated-draft-vocabulary idea (their `build_draft_vocab.py`, for vLLM);
  patch 0007 and `tools/` are the llama.cpp counterpart.
- [EAGLE](https://github.com/SafeAILab/EAGLE) (Li et al.) — the `d2t`
  draft-to-target token-map mechanism, which patch 0007 reuses via llama.cpp's
  in-tree EAGLE3 implementation (`src/models/eagle3.cpp`).
- [llama-swap](https://github.com/mostlygeek/llama-swap) — the model-swapping
  proxy the `config/` block targets; its per-model `env:` lists are what make
  the env-gated kernel caps (patches 0004/0008) deployable per-backend.
- Model files: [bartowski](https://huggingface.co/bartowski) (main GGUF),
  [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (MTP drafter and
  vision projector GGUFs), and [Qwen](https://huggingface.co/Qwen) for
  Qwen3.8-27B itself.
- The `ngram-mod` stacking defaults in `config/` were informed by a community
  ablation posted to r/LocalLLaMA (u/lukaLLM); tuned values are our own
  measurements.
- [Anbeeld/BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) — a
  llama.cpp fork carrying its own low-bit KV-cache/quant work; evaluating a
  hybrid of our patch stack against it (see `docs/quant-selection.md`)
  informed how we think about weight-byte-vs-compute tradeoffs at very low
  bit widths.
- [KVarN](https://arxiv.org/abs/2606.03458) — the exact-tail KV-cache
  quantization approach BeeLlama implements; its "always retain an intrinsic
  exact-precision tail" design is the reference point our own quant-cost
  reasoning was checked against.
- [HoltYoung/vram-thermal-guard](https://github.com/HoltYoung/vram-thermal-guard)
  and ThomasBaruzier's `gputemps` approach — the thermal-watchdog and
  BAR-register GDDR6X-junction-reading approaches that made the fan-curve
  dead-zone finding in `docs/thermals-and-oc.md` possible to measure and fix.
- [club-3090](https://github.com/noonghunna/club-3090) — a cross-rig
  benchmark harness for this hardware class; its pinned bench prompts and
  report format shaped how we validated numbers in `docs/dflash2-findings.md`
  and `docs/quant-selection.md` against other rigs.
