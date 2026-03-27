#!/bin/bash
# Publish rvllm crates to crates.io in dependency order
set -euo pipefail

CRATES=(
    rvllm-core
    rvllm-config
    rvllm-gpu
    rvllm-memory
    rvllm-sequence
    rvllm-tokenizer
    rvllm-telemetry
    rvllm-block-manager
    rvllm-kv-cache
    rvllm-attention
    rvllm-model-loader
    rvllm-quant
    rvllm-sampling
    rvllm-model-runner
    rvllm-scheduler
    rvllm-worker
    rvllm-executor
    rvllm-speculative
    rvllm-engine
    rvllm-api
    rvllm-server
)

for crate in "${CRATES[@]}"; do
    echo "Publishing $crate..."
    cargo publish -p "$crate" --allow-dirty
    sleep 10  # crates.io rate limit
done
