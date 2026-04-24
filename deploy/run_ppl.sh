#!/bin/bash
# run_ppl.sh <MODEL_DIR> [ARENA_GB]
set -u
MODEL="${1:?model dir}"
ARENA="${2:-72}"

cd /workspace/runs/2c6bbd0fc/rvllm
K=/workspace/runs/2c6bbd0fc/rvllm/kernels/sm_90

export RVLLM_MODEL_DIR="$MODEL"
export RVLLM_KERNELS_DIR="$K"
export RVLLM_CUTLASS_SO="$K/libcutlass_kernels.so"
export RVLLM_FA3_SO="$K/libfa3_kernels.so"
export RVLLM_POLICY="$K/policy.json"
export RVLLM_ARENA_GB="$ARENA"
export RVLLM_PPL_CHUNK=86
export RVLLM_PROMPT="The quick brown fox jumps over the lazy dog. In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God. All things were made through him, and without him was not any thing made that was made. In him was life, and the life was the light of men. The light shines in the darkness, and the darkness has not overcome it."

timeout 600 ./v3/target/release/rvllm-ppl 2>&1
