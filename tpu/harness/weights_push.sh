#!/usr/bin/env bash
# Push pre-quantized int8 weights to HF for instant deploys.
# Usage: ./weights_push.sh [weights-dir]
set -euo pipefail

WEIGHTS_DIR="${1:-$HOME/weights-int8}"
HF_REPO="and-y/rvllm-kernels"

if [ ! -d "$WEIGHTS_DIR" ] || [ -z "$(ls -A $WEIGHTS_DIR/*.npy 2>/dev/null)" ]; then
    echo "no .npy files in $WEIGHTS_DIR"
    echo "run weights_export.py first"
    exit 1
fi

echo "packaging $(du -sh $WEIGHTS_DIR | cut -f1) from $WEIGHTS_DIR"
TAR="/tmp/gemma4-31B-int8-v6e4.tar"
tar -cf "$TAR" -C "$(dirname $WEIGHTS_DIR)" "$(basename $WEIGHTS_DIR)"
echo "uploading to $HF_REPO ($(du -sh $TAR | cut -f1))..."
huggingface-cli upload "$HF_REPO" "$TAR" "weights/v6e-4/gemma4-31B-int8.tar"
echo "done: weights/v6e-4/gemma4-31B-int8.tar"
rm -f "$TAR"
