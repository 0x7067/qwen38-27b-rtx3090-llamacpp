#!/bin/bash
# Wave 7 validation: v11 image, original vs d48k truncated drafter.
# Usage: run_validation.sh <arm-name> <drafter-file> [extra docker env args...]
set -eu
IMG=llama:cuda-swap-v11
MODELS=/data/buttercup_6tb/k3s/llama-models
BENCH="$(cd "$(dirname "$0")" && pwd)"
ARM="$1"; DRAFTER="$2"; shift 2

docker rm -f w7bench >/dev/null 2>&1 || true
docker run -d --name w7bench --runtime nvidia --gpus all \
  -v "$MODELS":/models -v "$BENCH":/bench \
  -p 127.0.0.1:5899:5899 \
  -e GGML_CUDA_MMVQ_NE11_MAX=3 "$@" \
  "$IMG" llama-server \
  -m /models/Qwen3.8-27B-Q4_K_L.gguf \
  --mmproj /models/mmproj-Qwen3.8-27B-Q8_0.gguf \
  --host 0.0.0.0 --port 5899 \
  --ctx-size 131072 --predict 131072 \
  --threads 8 --threads-batch 8 --parallel 1 \
  --batch-size 512 --ubatch-size 512 \
  --no-mmap -fa on --jinja \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --repeat-penalty 1.0 \
  --model-draft "/models/$DRAFTER" \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  -ngl 99 -ngld 99 >/dev/null

for i in $(seq 1 120); do
  curl -sf -o /dev/null localhost:5899/health && break
  sleep 3
  if [ "$i" = 120 ]; then echo "LOAD TIMEOUT"; docker logs w7bench | tail -20; docker rm -f w7bench; exit 1; fi
done

echo "=== ARM $ARM drafter=$DRAFTER ==="
docker logs w7bench 2>&1 | grep -iE "d2t|draft_vocab|creating MTP" || echo "(no d2t log line)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

for P in short mid1k prose7k agentic54k; do
  # warmup
  curl -s -o /dev/null localhost:5899/completion -d @"$BENCH/$P.json" -H "Content-Type: application/json"
  for r in 1 2; do
    OUT="$BENCH/out_${ARM}_${P}_${r}.json"
    curl -s localhost:5899/completion -d @"$BENCH/$P.json" -H "Content-Type: application/json" > "$OUT"
    jq -r .content "$OUT" > "$BENCH/content_${ARM}_${P}_${r}.txt"
    SHA=$(sha256sum "$BENCH/content_${ARM}_${P}_${r}.txt" | cut -c1-12)
    jq -r --arg arm "$ARM" --arg p "$P" --arg r "$r" --arg sha "$SHA" \
      '[$arm,$p,$r, (.timings.predicted_per_second|tostring), (.timings.predicted_ms|tostring), (.timings.draft_n|tostring), (.timings.draft_n_accepted|tostring), $sha] | join(" ")' "$OUT"
  done
done

# acceptance-per-position histogram from server log
docker logs w7bench 2>&1 | grep -E "draft acceptance|pos [0-9]" | tail -12 || true
docker rm -f w7bench >/dev/null
echo "=== ARM $ARM done ==="
