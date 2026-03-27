#!/bin/bash
# Run both rvllm and Python vLLM side by side for comparison
set -euo pipefail

MODEL=${MODEL_NAME:-"meta-llama/Llama-3.2-1B"}
echo "Starting comparison: Rust rvllm vs Python vLLM"
echo "Model: $MODEL"
echo ""

# Start both services
docker compose up -d

# Wait for both to be ready
echo "Waiting for servers to start..."
for port in 8000 8001; do
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "  Port $port: ready"
            break
        fi
        sleep 2
    done
done

echo ""
echo "Both servers ready. Run benchmarks with:"
echo "  python3 scripts/benchmark.py --rust-url http://localhost:8000 --python-url http://localhost:8001"
