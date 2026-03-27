#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building rvllm Docker image..."
echo "  CUDA 12.4 + Rust (release mode)"

# Build kernels first if nvcc available
if command -v nvcc &>/dev/null; then
    echo "Compiling CUDA kernels..."
    cd "$ROOT_DIR/kernels" && bash build.sh
fi

cd "$ROOT_DIR"
docker build -t rvllm:latest -f Dockerfile .

echo ""
echo "Build complete: rvllm:latest"
echo ""
echo "Run with:"
echo "  docker run --gpus all -p 8000:8000 -v /path/to/models:/models rvllm:latest serve --model /models/your-model"
