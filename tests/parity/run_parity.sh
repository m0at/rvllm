#!/bin/bash
# Run full parity test suite: rvLLM vs Python vLLM
# Requires both servers running on specified URLs
set -euo pipefail

RUST_URL=${RUST_URL:-http://localhost:8000}
PYTHON_URL=${PYTHON_URL:-http://localhost:8001}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================"
echo "  rvLLM Parity Test Suite"
echo "======================================"
echo "Rust server:   $RUST_URL"
echo "Python server: $PYTHON_URL"
echo "Model:         $MODEL"
echo ""

# Check servers are reachable
echo "Checking server connectivity..."
if ! curl -sf "$RUST_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Rust server at $RUST_URL is not reachable"
    exit 1
fi
if ! curl -sf "$PYTHON_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Python server at $PYTHON_URL is not reachable"
    exit 1
fi
echo "Both servers reachable."

FAILURES=0

echo ""
echo "=== Token Parity ==="
if ! python3 "$SCRIPT_DIR/token_parity.py" \
    --rust-url "$RUST_URL" \
    --python-url "$PYTHON_URL" \
    --model "$MODEL" \
    --logprobs; then
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "=== Response Format Parity ==="
if ! python3 "$SCRIPT_DIR/response_format_parity.py" \
    --rust-url "$RUST_URL" \
    --python-url "$PYTHON_URL" \
    --model "$MODEL"; then
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "=== Sampling Parity ==="
if ! python3 "$SCRIPT_DIR/sampling_parity.py" \
    --rust-url "$RUST_URL" \
    --python-url "$PYTHON_URL" \
    --model "$MODEL" \
    --runs "${SAMPLING_RUNS:-100}"; then
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "======================================"
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL PARITY SUITES PASSED"
    exit 0
else
    echo "$FAILURES PARITY SUITE(S) FAILED"
    exit 1
fi
