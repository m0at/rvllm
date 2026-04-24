#!/bin/bash
# bench_drive.sh MODEL_DIR OUT_PREFIX [extra env...]
# Runs rvllm-bench across a batch sweep + rvllm-ppl on the canonical passage.
# Emits JSON lines to $OUT_PREFIX.ndjson plus a summary to $OUT_PREFIX.summary.
set -u

MODEL_DIR="${1:?model dir required}"
OUT_PREFIX="${2:?out prefix required}"
shift 2

cd /workspace/runs/2c6bbd0fc/rvllm

KERN=/workspace/runs/2c6bbd0fc/rvllm/kernels/sm_90
BIN_BENCH=/workspace/runs/2c6bbd0fc/rvllm/v3/target/release/rvllm-bench
BIN_PPL=/workspace/runs/2c6bbd0fc/rvllm/v3/target/release/rvllm-ppl

export RVLLM_MODEL_DIR="$MODEL_DIR"
export RVLLM_KERNELS_DIR="$KERN"
export RVLLM_CUTLASS_SO="$KERN/libcutlass_kernels.so"
export RVLLM_FA3_SO="$KERN/libfa3_kernels.so"
export RVLLM_POLICY="$KERN/policy.json"
# Allow caller overrides
for kv in "$@"; do export "$kv"; done

echo "=== Model: $MODEL_DIR ==="
echo "Overrides: $*"
echo "Policy exists: $(ls -la $RVLLM_POLICY 2>&1 | head -1)"
echo ""

: > "${OUT_PREFIX}.ndjson"
: > "${OUT_PREFIX}.summary"

# --- Batch sweep ---
echo "=== Batch sweep ===" | tee -a "${OUT_PREFIX}.summary"
for B in 1 8 32 64 128 256 512; do
    echo "--- B=$B ---"
    RVLLM_BATCH=$B RVLLM_ITERS=30 RVLLM_WARMUP=5 \
        timeout 300 "$BIN_BENCH" 2>&1 | tee -a "${OUT_PREFIX}.log" | tee -a "${OUT_PREFIX}.summary"
    # Extract JSON line from stdout (bench prints a JSON summary)
    grep -E '^\{.*tok_per_sec' "${OUT_PREFIX}.log" | tail -1 >> "${OUT_PREFIX}.ndjson" || true
    sleep 2
done

# --- Perplexity (canonical passage) ---
echo "" | tee -a "${OUT_PREFIX}.summary"
echo "=== Perplexity (canonical 86-token passage) ===" | tee -a "${OUT_PREFIX}.summary"
PPL_TEXT="The quick brown fox jumps over the lazy dog. In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God. All things were made through him, and without him was not any thing made that was made. In him was life, and the life was the light of men. The light shines in the darkness, and the darkness has not overcome it."
RVLLM_PROMPT="$PPL_TEXT" RVLLM_PPL_CHUNK=128 \
    timeout 300 "$BIN_PPL" 2>&1 | tee -a "${OUT_PREFIX}.log" | tee -a "${OUT_PREFIX}.summary"

echo ""
echo "=== Done. Outputs in ${OUT_PREFIX}.{ndjson,summary,log} ==="
