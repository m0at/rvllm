#!/usr/bin/env bash
# Pull pre-quantized int8 weights from HF for instant deploys.
# Usage: ./weights_pull.sh [output-dir]
set -euo pipefail

OUTPUT_DIR="${1:-$HOME/weights-int8}"
HF_REPO="and-y/rvllm-kernels"
ARTIFACT="weights/v6e-4/gemma4-31B-int8.tar"

echo "downloading $ARTIFACT from $HF_REPO..."
huggingface-cli download "$HF_REPO" "$ARTIFACT" --local-dir /tmp/weights_dl
TAR=$(find /tmp/weights_dl -name "*.tar" | head -1)
if [ -z "$TAR" ]; then
    echo "no tarball found"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
tar -xf "$TAR" -C "$(dirname $OUTPUT_DIR)"
echo "restored to $OUTPUT_DIR ($(du -sh $OUTPUT_DIR | cut -f1))"
rm -rf /tmp/weights_dl
