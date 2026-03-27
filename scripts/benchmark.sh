#!/bin/bash
# One-command local benchmark: Rust (criterion) vs Python (numpy/torch)
# No GPU required -- benchmarks CPU sampling kernels only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "================================================================"
echo "  rvllm Local Benchmark: Rust vs Python sampling kernels"
echo "================================================================"
echo ""

# --- Rust benchmarks (criterion) ---
echo "[1/3] Building and running Rust benchmarks (criterion)..."
echo "      This takes ~2 minutes on first run."
echo ""
cargo bench --package rvllm-bench --bench sampling_bench 2>&1 | grep -E "(Benchmarking|time:)" || true
echo ""
echo "Rust benchmarks complete."
echo ""

# --- Python benchmarks ---
echo "[2/3] Running Python benchmarks (numpy + torch)..."
echo ""

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.8+ to run comparison."
    exit 1
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "ERROR: numpy not installed. Run: pip3 install numpy"
    exit 1
fi

python3 benches/bench_python.py
echo ""
echo "Python benchmarks complete."
echo ""

# --- Comparison ---
echo "[3/3] Generating comparison report..."
echo ""
python3 benches/compare.py

echo ""
echo "Done. Raw results:"
echo "  Rust:   target/criterion/*/new/estimates.json"
echo "  Python: benches/python_results.json"
