#!/usr/bin/env bash
set -euo pipefail

# Gemma 4 on RTX 3090 Ti (Ampere SM86) -- local build
#
# Usage:
#   ./scripts/gemma4_ampere_deploy.sh [kernels_dir]
#
# Builds SM86 paged attention .so and CUTLASS stub .so. FP8/F16 GEMMs
# go through cuBLAS/cuBLASLt per the project's "cuBLAS first, CUTLASS
# later" rule; the stub only satisfies dlsym at bring-up.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="${REPO_ROOT}/kernels"

# Default output dir: <repo-parent>/kernels/sm_86 (matches existing layout).
DEFAULT_KERNELS_DIR="$(dirname "${REPO_ROOT}")/kernels/sm_86"
KERNELS_DIR="${1:-${DEFAULT_KERNELS_DIR}}"

echo "=== Gemma 4 Ampere (SM86) Build ==="
echo "  Repo root:   ${REPO_ROOT}"
echo "  Source dir:  ${SRC_DIR}"
echo "  Kernels out: ${KERNELS_DIR}"
mkdir -p "${KERNELS_DIR}"

# --- Build SM86 paged attention .so ---
echo "  Building SM86 paged attention .so..."
cd "${SRC_DIR}"
nvcc -shared -o "${KERNELS_DIR}/libfa_sm86_kernels.so" \
    paged_attention_sm86.cu \
    -arch=sm_86 -O3 --use_fast_math -Xcompiler -fPIC \
    -I/usr/local/cuda/include
echo "  SM86 attention .so built: ${KERNELS_DIR}/libfa_sm86_kernels.so"

# --- Build CUTLASS stub .so ---
echo "  Building CUTLASS stub .so..."
gcc -shared -fPIC -o "${KERNELS_DIR}/libcutlass_stub_sm86.so" \
    "${SRC_DIR}/cutlass_stub_sm86.c"
echo "  CUTLASS stub .so built:   ${KERNELS_DIR}/libcutlass_stub_sm86.so"

echo ""
echo "=== Ampere build complete ==="
ls -la "${KERNELS_DIR}/libfa_sm86_kernels.so" "${KERNELS_DIR}/libcutlass_stub_sm86.so"
