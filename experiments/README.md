# Experiments (not shipped)

These patches are measured but did not meet the ship bar. They are preserved
for the next attempt, not for production use.

- `armG-combined-i64-nthr128-plus-grid.patch` — MMQ small-batch tile and grid
  reconfiguration, env-gated with `GGML_CUDA_MMQ_SMALLN=1`. It cuts the
  verify-path marginal cost from 13.0 to 11.0 ms per cycle and measured +4.4%
  end-to-end decode, below the +10% bar. It passes 1267/1267 MUL_MAT
  correctness cases. Before any ship: run KLD against production, smoke-test
  the other backends (the grid change touches every J≤8 MMQ launch), and gate
  the ffn_down and m≤1024 regressions. `docs/mmq-small-batch-analysis.md` is
  the full analysis.
- `testcov-mulmat-qwen38-verify.patch` — adds the Qwen3.8 speculative-verify
  GEMM shapes to `test-backend-ops`. Upstream has no coverage for them. Apply
  this before benching any MMQ change.
