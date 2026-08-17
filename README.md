# Qwen3.8-27B on a single RTX 3090 — llama.cpp kernel patches + truncated MTP draft vocabulary

Lossless speedups for **Qwen3.8-27B** (hybrid Gated-DeltaNet, 27B) decode on one
**RTX 3090 (24 GB, Ampere cc 8.6)** with llama.cpp, full **131k context resident**
(main model + MTP drafter + vision projector ≈ 22.9 GiB), speculative decoding via
the native MTP head.

Campaign result on the target workload (54k-token-deep agentic decode, temp-0):
**49.8 → 69.3 tok/s (+39%)**; mid-depth (1k) **63 → 84 tok/s (+33%)**.
On top of that, the truncated-draft-vocabulary work in this repo adds
**+5–6% at mid/7k depth** and **−0.5 GiB VRAM** with byte-identical output
(A/B verified by output sha256).

Inspired by [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
(vLLM). This is the llama.cpp counterpart: no multi-minute cold start, hot-swappable
via llama-swap, and decode holds up at 54k depth.

## Results

Measured temp-0, warm, `/completion` timings, `--spec-draft-n-max 5`,
`GGML_CUDA_MMVQ_NE11_MAX=3`. "v10" = patches 0001–0006; "v11 + d48k" adds patch 0007
and the 49,152-row truncated drafter.

| Payload (prompt depth) | v11 + full drafter | v11 + d48k drafter | delta | acceptance (full → d48k) |
|---|---|---|---|---|
| short (98 tok) | 75.3 tok/s | 75.7 | +0.5% | .527 → .488 |
| mid (870 tok, code) | 75.0 | **79.6** | **+6.1%** | .532 → .532 (identical) |
| prose (6.8k tok) | 64.7 | **68.2** | **+5.4%** | .435 → .431 |
| agentic (53.4k tok) | 59.4 | **62.9** | **+6.0%** | .439 → .439 (identical) |
| VRAM at 131k ctx | 23,438 MiB | **22,892 MiB** | −546 MiB | |

Output content is **byte-identical** between the full and truncated drafter for every
payload (sha256-compared): llama.cpp's verify step samples from the *target*
distribution and only keeps a draft token when it matches exactly, so the drafter's
support can shrink without affecting outputs — only acceptance (i.e. speed) is at
stake. See `docs/truncated-draft-vocab-design.md` §4.

Earlier waves (patches 0001–0006, full write-up in `docs/PERFORMANCE.md`):
54k agentic decode 49.8 → 69.3 tok/s via three custom CUDA kernels + one scheduling
cap + draft-depth tuning.

## What's in the patches

Applied onto llama.cpp `4df29be4f` (see `Dockerfile`), in order, with `git apply`
(not `git am` — 0002/0006 are plain diffs):

| Patch | What it does |
|---|---|
| 0001 | DFlash draft-side greedy fast path + `LLAMA_SPEC_PROF` instrumentation |
| 0002 | Ampere MMQ small-batch (J=16) tile config — 128×64 tiles for Q4_K/Q5_K |
| 0003 | GQA-batched FlashAttention **vector** kernel for quantized KV, batch ≤2, cc<8.9: reads each KV block once per GQA group instead of once per Q head; no F16 scratch |
| 0004 | `GGML_CUDA_MMVQ_NE11_MAX` env: caps the MMVQ→MMQ crossover per model (measured N≈3.4 on 3090/27B; upstream flat 8 overpays 24–38% on speculative verify batches) |
| 0005 | MMA FlashAttention reads q4_0 K/V **inline** (cp.async → smem → dequant in place; no full-cache F16 dequant per step), batches 2–16, D=256/GQA>4 |
| 0006 | DFlash greedy fast path requires host logits (fixes silent acceptance=0 with GPU draft sampling) |
| 0007 | **Truncated draft vocabulary for MTP drafters** via the EAGLE3 `d2t` idiom: a drafter GGUF may carry a reduced LM head + `d2t` I64 tensor mapping head rows to target token ids; logits are scattered back to full width with −inf elsewhere. No-op for GGUFs without `d2t` |

## Truncated draft vocabulary (patch 0007 + tools)

The Qwen3.8 MTP drafter is ~3.0B params, but 2.5B of that is two 248,320-row
embedding matrices. The input embedding is a row gather (cheap); the **output head is
a full 0.7 GB mat-vec on every draft step — 74% of the draft pass**. Truncating it to
the 49,152 tokens that cover ~98.5% of real traffic cuts per-step draft traffic by
~62% and turns into the +5–6% end-to-end decode gain above.

Pipeline:

1. `tools/build_draft_vocab.py` — rank token ids by frequency over a corpus shaped
   like your traffic (ours: English technical + code, ~10% pt-BR), force-include all
   control/special tokens and the byte alphabet, emit keep-set JSONs (32k/40k/48k).
2. `tools/eval_coverage.py` — held-out + out-of-distribution coverage per slice.
   Pick the smallest set whose OOD coverage you can live with (we shipped 48k:
   98.5% held-out, 96.5% OOD).
3. `tools/truncate_drafter.py` — lossless GGUF surgery: copies kept Q4_0 head rows
   byte-for-byte (a 5120-wide Q4_0 row is exactly 2880 B), keeps `token_embd` at full
   width (inputs are target-vocab ids), appends the `d2t` I64 tensor, validates
   range/uniqueness/sortedness. `tools/validate_drafter.py` re-checks everything
   (26 assertions) against the source file.
4. Run with patch 0007 and `--model-draft <truncated>.gguf`. Rollback = point
   `--model-draft` back at the original file; the patched binary is a no-op without
   `d2t`.

## Reproduce

```bash
docker build -t llama:cuda-swap-v11 .          # llama.cpp @4df29be4f + patches, sm_86
# bench harness (A/B two drafters, 4 depths, sha256 output-exactness check):
tools/run_validation.sh A mtp-Qwen3.8-27B-Q4_0.gguf
tools/run_validation.sh B mtp-Qwen3.8-27B-Q4_0-d48k.gguf
```

Models: [bartowski Q4_K_L main](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
(embeds an unused MTP block; Q8_0 embed/output), plus
[ggml-org's](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) `mtp-*-Q4_0` drafter
and Q8_0 mmproj. Serving config example (llama-swap): `config/llama-swap-qwen38.yaml`.
Couplings that matter (measured, see docs): draft-n 5 is only a win **with** the
MMVQ cap env; without it, n4 is *slower* than n2 and nothing fails loudly.

## Things measured so you don't have to

- Embedded-MTP mode (dropping `--model-draft`) routes draft logits through the
  **target's Q8_0 head**: −1.6% speed for −0.9 GiB VRAM. Keep the separate drafter.
- Marginal cost of one draft step at n=3→5: **2.25 ms**, vs ~1.2 ms byte floor —
  fixed overhead is ~45% of a draft step, which caps what head-shrinking can buy.
- q8_0 KV doesn't fit at 131k (DeltaNet leaves only 16 attention layers, ~18 KB/tok
  at q4_0); f16 KV forces a worse kernel path. Chain speculation (PR #27173) loses
  6–12% single-GPU. GDN chunked prefill (PR #26001) is lossy (KLD 0.0075).
- Full kernel-level analysis, dead ends, and per-wave benches: `docs/PERFORMANCE.md`.

## What is reproducible from this repo alone

**Fully standalone** (no access to the original host needed):

- The image: `Dockerfile` clones llama.cpp at the pinned ref from GitHub and applies
  `patches/` — nothing local.
- The truncated drafter: `data/draft_vocab_48k.json` is the exact keep-set that produced
  the shipped GGUF. `tools/truncate_drafter.py` + `tools/validate_drafter.py` +
  `tools/ggufio.py` are pure Python stdlib. Rebuild byte-for-byte with:
  `tools/truncate_drafter.py mtp-Qwen3.8-27B-Q4_0.gguf data/draft_vocab_48k.json out.gguf`
  (drafter from the HF link above). 32k/40k variants and the full ranked frequency table
  (`data/token_freq.tsv.gz`) are included for re-slicing other set sizes without redoing
  the corpus pass. `data/coverage.json` / `data/ood_coverage.json` are the acceptance-risk
  evidence behind the 48k choice.

**Host-shaped** (works anywhere, but the defaults describe the original machine — override
or edit before reuse):

- `tools/build_draft_vocab.py` re-derives the keep-set from a corpus; corpus roots are
  local paths by nature (point them at *your* traffic-shaped text; tokenizer via
  `QWEN_TOKENIZER_JSON`). You only need this to build a *different* vocabulary.
- `tools/run_validation.sh` (models dir via `MODELS=`) and `tools/make_payloads.py`
  (bench prompts built from local files — any large code/docs tree works; avoid chat
  transcripts, they can make temp-0 generation emit EOS at position 0).
- `config/llama-swap-qwen38.yaml` assumes llama-swap and this image.

## Layout

```
Dockerfile                  build llama.cpp @4df29be4f + patches (CUDA sm_86) + llama-swap
patches/0001..0007          the vendored patch stack (apply with git apply, in order)
tools/                      draft-vocab pipeline, GGUF surgery + validation, bench harness
config/                     llama-swap model block (flags + env couplings, commented)
docs/PERFORMANCE.md         full campaign write-up (waves, kernels, rejects, methodology)
data/                       keep-sets (32k/40k/48k), ranked token frequencies, coverage evidence
docs/truncated-draft-vocab-design.md   design doc for patch 0007 (data flow, correctness proof)
```

## License / credits

The `patches/` modify [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and
follow its license. Draft-vocabulary idea adapted from syv-ai's vLLM work; `d2t`
mechanism reuses llama.cpp's own EAGLE3 idiom. Everything else here: MIT.
