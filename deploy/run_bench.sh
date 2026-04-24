#!/bin/bash
# run_bench.sh <MODE> <MODEL_DIR> [BATCH] [ARENA_GB] [EXTRA_VAR=VAL]...
# MODE: "bench" or "ppl"
set -u
MODE="${1:?mode: bench|ppl}"
MODEL="${2:?model dir}"
BATCH="${3:-1}"
ARENA="${4:-72}"
shift 4 || shift $#

cd /workspace/runs/2c6bbd0fc/rvllm
K=/workspace/runs/2c6bbd0fc/rvllm/kernels/sm_90

export RVLLM_MODEL_DIR="$MODEL"
export RVLLM_KERNELS_DIR="$K"
export RVLLM_CUTLASS_SO="$K/libcutlass_kernels.so"
export RVLLM_FA3_SO="$K/libfa3_kernels.so"
export RVLLM_POLICY="$K/policy.json"
export RVLLM_ARENA_GB="$ARENA"
export RVLLM_BATCH="$BATCH"
export RVLLM_ITERS="${RVLLM_ITERS:-30}"
export RVLLM_WARMUP="${RVLLM_WARMUP:-5}"
for kv in "$@"; do export "$kv"; done

BIN=./v3/target/release/rvllm-bench
[ "$MODE" = "ppl" ] && BIN=./v3/target/release/rvllm-ppl

echo "=== $MODE $MODEL B=$BATCH ARENA=${ARENA}GB ==="
timeout 600 "$BIN" 2>&1
