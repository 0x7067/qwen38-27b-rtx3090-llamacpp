# DFlash2 vs our MTP drafter: findings from llama.cpp PR #27342

[`ggml-org/llama.cpp#27342`](https://github.com/ggml-org/llama.cpp/pull/27342)
("support DFlash2") adds a second speculative-decoding drafter architecture.
This doc is our evaluation of it against this repo's shipped MTP + d48k-vocab
setup, on the same Qwen3.8-27B target, on one RTX 3090.

## Headline: env caps flip the verdict entirely

DFlash2's block-8 verify submits a 9-token batch. `ggml_cuda_should_use_mmvq`
hardcodes `MMVQ_MAX_BATCH_SIZE = 8` for every card/quant
(`ggml/src/ggml-cuda/mmvq.cuh`), so that batch size lands on the slower MMVQ
path on **any stock build** — this is the same crossover mechanism patch 0004
in this repo targets for our own MTP drafter, discovered independently here
on a completely different drafter architecture.

**On the unmodified PR binary** (greedy, ctx 131072), DFlash2 loses to our
MTP setup almost everywhere:

| payload | stock MTP n5 | stock DFlash2 blk8 |
|---|---:|---:|
| mid1k code | 74.3 tok/s | 71.0 tok/s |
| deep 55k code | 67.6 tok/s | 51.4 tok/s |

**With the two env-gated kernel-dispatch caps this repo ships**
(`GGML_CUDA_MMVQ_NE11_MAX=3`, `GGML_CUDA_MMQ_SMALLN=3`) applied to the same
binary and drafter — routing DFlash2's small-batch verify onto tuned MMQ
tiles instead of the flat-8 MMVQ cutoff — the result flips:

| payload | prod MTP d48k n5 | DFlash2 blk8 + caps | delta |
|---|---:|---:|---:|
| short (~20 tok) | 115.9 | **143.5** | **+24%** |
| mid1k code | 94.4 | **113.3** | **+20%** |
| code 7k | 126.6 | **139.8** | **+10%** |
| prose 7k | 108.6 | **116.7** | **+7%** |
| deep 55k code | **101.9** | 87.6 | **−14%** |

Isolating the two caps alone (same vendored binary, same drafter, only the
env vars toggled) shows they are the *entire* effect, not the drafter
architecture: mid1k goes 71.0 (caps off) → 113.3 (caps on), **+60%**;
caps-off vendored is statistically identical to the stock PR binary.

## Three caveats that shrink or reverse the headline

1. **Real (non-greedy) sampling shrinks the win a lot.** An 8-seed paired
   test under our actual production sampling (`temp 1.0, top_k 20, top_p
   0.95`): DFlash2 wins 5 of 8 seeds, **+5.3% mean**, with per-seed deltas
   ranging from **−7% to +25%**. The greedy +20% is a real ceiling, not the
   expected case — DFlash2's mean acceptance drops harder under sampling
   (0.640 → 0.543) than our MTP setup's does.

2. **It loses at the depth that dominates agentic coding.** −14% at 55k
   depth on Q4_K_L. Repeating on a smaller main quant (IQ4_XS) shows
   +59% at mid1k but still **−8% at 55k** — the crossover is a property of
   DFlash2's wider verify batch costing more as context depth grows, not a
   quant artifact. A smaller main model just moves where the crossover
   lands, it doesn't remove it.

3. **Stacking n-gram lookup (`ngram-mod`) on top widens the deep-depth loss,
   it does not close it.** With `--spec-type draft-*,ngram-mod` on both arms
   (our production default), the deep-payload gap widens from −14%
   (ngram off) to **−17.1%** (ngram on, cold/one-shot measurement) — our
   repetitive deep-coding payload lets ngram-mod supply long verbatim chains
   that our MTP block absorbs more cheaply than DFlash2's wider one
   (acceptance 0.810 for MTP+ngram vs 0.468 for DFlash2+ngram on the same
   payload). Caveat on the caveat: only the cold/one-shot measurement here is
   trustworthy — repeated warm reps against an identical prompt mostly
   measure the ngram lookup table itself (150–330 tok/s in both arms
   regardless of drafter), not the drafter, so the defensible range straddles
   the ngram-off −14% figure rather than pinning exactly to −17.1%.

## VRAM

DFlash2's Q4_K_M drafter costs ~580 MiB more than our MTP d48k drafter at the
same context length — pushes our 153,600-token context to 23,916 MiB against
our 23,400 MiB target ceiling, tighter than we want given the thermal margin
discussion in `docs/thermals-and-oc.md`.

## The other real DFlash2/#27342 findings, filed upstream

Two separate issues surfaced while evaluating this PR, filed against
`ggml-org/llama.cpp` directly (that project's policy does not accept
agent-authored submissions, so these were filed manually by the repo owner,
not by an agent — listed here for completeness, not as something this repo
did):

- **Draft-block VRAM is not validated at load time.** An infeasible
  `--spec-draft-n-max` for the available VRAM loads cleanly and passes health
  checks, then the *first* completion request aborts the whole server with a
  bare `CUDA error: out of memory` / `ggml_abort` (client sees only
  `Empty reply from server`). This is not a claim that abort-on-OOM itself is
  wrong — that's llama.cpp/ggml baseline behavior, reproduced identically on
  a pristine base tree with no DFlash2 code involved at all. The ask is a
  load-time feasibility check so an infeasible flag combination fails at
  startup with a sizing message, not on the first real request.
- **DFlash2 acceptance collapses on `cache_prompt` reuse.** Sending the same
  ~7k-token prompt three times with `cache_prompt` enabled: acceptance drops
  from 0.539 (cold) to 0.278 (warm, cached-prefix reuse) and stays there — a
  DFlash1 control on the same binary/payload shows the opposite, normal
  warm-up direction (0.484 → 0.513 → 0.513). Deterministic across two
  campaigns. This matches a hole already described in
  `z-lab/llama.cpp-fork#1` ("the same hole appears when the target reuses
  cached prompt prefixes the draft cache never saw") — this is the
  text-only instance of that same mechanism, not yet reproduced on our
  primary Qwen3.8-27B target at any depth.
