#!/bin/bash
set -euo pipefail
echo "Building rvllm wheel..."
maturin build --release
echo "Publishing to PyPI..."
maturin publish
