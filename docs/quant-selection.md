# Main-quant selection: from Q4_K_L, through an IQ4_XS speed candidate, to UD-Q4_K_XL

Prod's main quant history on this rig: **bartowski Q4_K_L → (evaluated, not
shipped: bartowski IQ4_XS) → unsloth UD-Q4_K_XL (current)**. This doc is the
evidence trail for that move, plus the quants that were tested and rejected
along the way. All KLD numbers are measured against a Q6_K reference of the
same model, not against ground truth — read them as relative rankings across
quants, not absolute quality scores.

## Why not stay on IQ4_XS (the fast candidate)

A first wave measured bartowski IQ4_XS ("B") against Q4_K_L ("A"): **+7.99% pooled
t/s (short payload; +7.95%/+9.39% mid1k/deep48k), −2.4 GiB VRAM**, but KLD-to-Q6K
ratio **1.415** vs A's own 1.0 — same-top-p 94.84% vs A's 95.66%. IQ4_XS is
faster and smaller, but it's a real quality regression, not a free lunch.

A follow-up sweep tested four more candidates against the same bar (KLD ratio
≤1.15 vs A = quality-neutral-or-better; pooled ms/step ≥5% faster = worth the
change):

| Arm | Source | Size | KLD ratio vs A | Same-top-p | Pooled Δms/step vs A | VRAM Δ vs A |
|---|---|---|---:|---:|---:|---:|
| A (Q4_K_L, prod at the time) | bartowski | 18.72 GiB | 1.000 | 95.66% | — | — |
| B (IQ4_XS) | bartowski | 15.57 GiB | 1.415 | 94.84% | **−7.07% (faster)** | −2,356 MiB |
| Q5S (Q5_K_S) | mradermacher, imatrix | 18.97 GiB | **0.553** | 96.38% | +6.73% (slower) | +406 MiB |
| **UDXL (UD-Q4_K_XL)** | unsloth | 17.92 GiB | **0.787** | 96.09% | +3.42% (slower) | −190 MiB |
| INL (IQ4_NL) | unsloth | 16.34 GiB | 1.316 | 94.90% | −0.90% (~flat) | −1,662 MiB |
| MRXS (IQ4_XS, mradermacher) | mradermacher, imatrix | 15.31 GiB | 1.472 | 94.81% | −3.97% (faster) | −2,648 MiB |

None of the four new candidates clears the speed bar (only B does, at −7.07%).
Two clear the quality bar: Q5S (best quality, worst cost — 6.7% slower, VRAM
headroom drops to 574 MiB on a 24 GiB card) and UDXL (clean quality win over
both A and B — ratio 0.787 vs A's own reference point — at a modest 3.4%
speed cost and a small VRAM saving).

**MRXS vs B is the control that matters**: two different IQ4_XS builds from
different imatrix calibrations (bartowski vs mradermacher) land at essentially
the same KLD ratio (1.415 vs 1.472). The quality gap is structural to the
IQ4_XS quant type at this model size, not a calibration artifact of one
uploader.

## Why UD-Q4_K_XL shipped

We picked quality over the last few percent of speed: UDXL is a clear KLD win
over the previous prod quant, costs only ~3.4% pooled step time, and actually
saves a little VRAM (−190 MiB) rather than costing any. IQ4_XS remains the
right call if raw speed and VRAM are the only axes that matter — nothing in
either sweep beats its tradeoff on those two axes alone — but for a model this
is served to real users behind, the quality margin was worth more than 3%.

## The rejected extreme: Ridge 3.7bpw

A 3.69 bpw GDN-aware imatrix quant (`empero-ai/Qwen3.8-27B-Ridge-GGUF`,
SHA256-verified against the repo's own checksums, embeds its own MTP head as
`blk.64`) was evaluated mid-sweep and **decisively rejected**:

| metric | A (Q4_K_L) | B (IQ4_XS) | Ridge 3.7bpw |
|---|---:|---:|---:|
| Mean KLD vs Q6_K | 0.0136 | 0.0192 | **0.1156** |
| Ratio vs A | 1.0 | 1.415 | **8.507** |
| Same-top-p | 95.66% | 94.84% | **88.38%** |
| Pooled `ms/verify-step` vs A | — | −7.07% | **+3.50%** |

Ridge is 37% smaller than A by disk size **and 3.5% slower per verify step**
— the weight-traffic model that correctly ordered every other quant in this
campaign fails completely here: its mixed 3.7bpw kernels cost more to
dequantize than the smaller footprint saves in bandwidth. Its embedded MTP
head is also markedly worse as a drafter than our external d48k drafter
(+21.98% per verify step at the same accepted length when the embedded head
is used instead). Lesson: below 4 bits, "fewer bytes" stopped predicting
"faster" on this stack — don't extrapolate the weight-bytes model past 4-bit
K-quants without measuring.

## Corrected weight-bytes cost model

An earlier note in this campaign's working docs claimed weight-byte traffic
was **77% of decode step cost**. A dedicated decomposition (comparing measured
`ms/verify-step` deltas against each quant's byte-size delta, with a
dequant-cost control) found the real figure is **closer to 40%** — the 77%
model correctly orders candidates by speed but overstates the *magnitude* of
any byte-size win by roughly 2x. Two independent arms (an IQ-quant and a
same-family K-quant control) converged on the same ~40% figure, so this isn't
an artifact of one quant type's dequant cost. Use ~40% as the working model
for predicting the effect of a future quant change; don't reuse the 77%
figure.
