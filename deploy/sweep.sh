#!/bin/bash
# sweep.sh <MODEL_DIR> <OUT_PREFIX>
# Runs rvllm-bench at B=1,8,32,64,128,256,512 with 72GB arena + 31B shapes.
set -u
MODEL="${1:?model dir}"
OUT="${2:?out prefix}"

mkdir -p /workspace/reports
: > "${OUT}.log"
: > "${OUT}.ndjson"

echo "=== Sweep: $MODEL ==="
for B in 1 8 32 64 128 256 512; do
    echo; echo "--- B=$B ---" | tee -a "${OUT}.log"
    bash /workspace/run_bench.sh bench "$MODEL" "$B" 72 RVLLM_ITERS=20 RVLLM_WARMUP=5 2>&1 | tee -a "${OUT}.log"
    grep -E '^\{.*tok_per_sec' "${OUT}.log" | tail -1 >> "${OUT}.ndjson" || true
    sleep 3
done

echo; echo "=== Sweep summary ==="
cat "${OUT}.ndjson"
