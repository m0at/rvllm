#!/bin/bash
set -euo pipefail

export RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90
export RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so
export RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so
export RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json
export RVLLM_ITERS=512
export RVLLM_WARMUP=8
export RVLLM_REAL_PREFILL=1
export RVLLM_PREFILL_LEN=16
export RVLLM_TTFT=1
export RVLLM_ARENA_GIB=70
export RVLLM_BLOCK_SIZE=256
export RVLLM_NAN_CHECK=1

BENCH=${RVLLM_BENCH:-/workspace/runs/cfed174fb/v3/target/release/rvllm-bench}

for MODEL_NAME in qwen3-8b mistral-7b-v03; do
  echo "=== $MODEL_NAME ==="
  export RVLLM_MODEL_DIR=/workspace/models/$MODEL_NAME
  for N in 1 8 16 64 128 256 512; do
    export RVLLM_BATCH=$N
    RESULT=$($BENCH 2>&1) || true
    TOKS=$(echo "$RESULT" | grep -o '"tok_per_sec":[0-9.]*' | cut -d: -f2)
    NAN_LINES=$(echo "$RESULT" | grep -c '\[NaN\]' || true)
    echo "N=$N  toks=${TOKS:-0}  nan_layers=$NAN_LINES"
  done
done
echo "=== ALL DONE ==="
