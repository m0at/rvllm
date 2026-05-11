#!/bin/bash
# rvLLM Development Environment Setup
#
# IMPORTANT: Review this script before running it. Shell scripts can execute
# arbitrary commands. Read through each section to understand what will be
# installed on your system.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# What this installs:
#   - Rust toolchain (via rustup)
#   - Python 3 + pip (if not present)
#   - maturin (for building Python bindings)
#   - CUDA toolkit info (checks, does not install)
#   - vast.ai CLI (optional, for A100 benchmarking)
#
# On macOS, uses Homebrew. On Linux, uses apt.

set -euo pipefail

echo "=== rvLLM Development Environment Setup ==="
echo ""
echo "This script will install development dependencies."
echo "Review the source before running: cat install.sh"
echo ""

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Platform: $OS $ARCH"
echo ""

# --- Rust ---
if command -v rustc &>/dev/null; then
    echo "[OK] Rust $(rustc --version | awk '{print $2}')"
else
    echo "[INSTALL] Rust toolchain via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "[OK] Rust $(rustc --version | awk '{print $2}')"
fi

# --- Python ---
if command -v python3 &>/dev/null; then
    echo "[OK] Python $(python3 --version | awk '{print $2}')"
else
    echo "[INSTALL] Python 3..."
    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            brew install python3
        else
            echo "  Install Homebrew first: https://brew.sh"
            exit 1
        fi
    elif [ "$OS" = "Linux" ]; then
        sudo apt-get update && sudo apt-get install -y python3 python3-pip
    fi
    echo "[OK] Python $(python3 --version | awk '{print $2}')"
fi

# --- pip ---
if command -v pip3 &>/dev/null; then
    echo "[OK] pip3"
else
    echo "[INSTALL] pip3..."
    python3 -m ensurepip --upgrade 2>/dev/null || sudo apt-get install -y python3-pip 2>/dev/null || true
fi

# --- maturin (for Python bindings) ---
if command -v maturin &>/dev/null; then
    echo "[OK] maturin $(maturin --version | awk '{print $2}')"
else
    echo "[INSTALL] maturin (Python binding builder)..."
    pip3 install maturin
    echo "[OK] maturin installed"
fi

# --- CUDA (check only) ---
echo ""
if command -v nvcc &>/dev/null; then
    echo "[OK] CUDA $(nvcc --version 2>&1 | grep release | awk '{print $6}')"
    if command -v nvidia-smi &>/dev/null; then
        echo "     GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
else
    echo "[INFO] CUDA not found. GPU inference requires NVIDIA CUDA toolkit."
    echo "       Install from: https://developer.nvidia.com/cuda-downloads"
    echo "       rvLLM works without CUDA using the mock-gpu backend (for development)."
fi

# --- vast.ai CLI (optional) ---
echo ""
if command -v vastai &>/dev/null; then
    echo "[OK] vast.ai CLI"
else
    echo "[OPTIONAL] vast.ai CLI (for A100 benchmarking):"
    echo "  pip3 install vastai"
fi

# --- Test dependencies ---
echo ""
echo "[INSTALL] Python test dependencies..."
pip3 install -q requests numpy scipy aiohttp pytest 2>/dev/null || true

# --- Verify build ---
echo ""
echo "=== Verifying rvLLM build ==="
# The active workspace lives in v3/. The serve binary is `rvllm-server`
# (from the rvllm-serve crate). Earlier versions had a single `rvllm`
# crate with a `serve` subcommand; that's gone.
if [ -f "v3/Cargo.toml" ]; then
    echo "Building (no CUDA — see --features cuda for the GPU path)..."
    (cd v3 && cargo build --release --bin rvllm-server) 2>&1 | tail -3
    echo ""
    echo "[OK] rvllm-server binary: $(ls -lh v3/target/release/rvllm-server 2>/dev/null | awk '{print $5, $9}')"
else
    echo "Run this script from the rvllm repo root (expects v3/Cargo.toml)."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  cd v3 && cargo build --release --bin rvllm-server                                # Build (no CUDA)"
echo "  cd v3 && cargo build --release --features cuda --bin rvllm-server                # Build (CUDA)"
echo "  cd v3 && cargo build --release --features cuda,gb10 --bin rvllm-server           # Build (GB10/sm_121)"
echo "  ./v3/target/release/rvllm-server --model-dir /path/to/hf/model --bind 0.0.0.0:8010"
echo "  cd v3 && cargo test --workspace --exclude rvllm-runtime --exclude rvllm-bench --exclude rvllm-serve"
echo "  make bench-compare                              # Benchmark"
