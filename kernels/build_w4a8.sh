#!/usr/bin/env bash
# Compile rvllm_w4a8 GEMM .so on H100 box.
# Expects CUTLASS 4.x at /root/cutlass (clone from NVIDIA/cutlass main).
set -euo pipefail

CUTLASS_DIR="${CUTLASS_DIR:-/root/cutlass}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/sm_90"
mkdir -p "${OUT_DIR}"

NVCC_FLAGS=(
    -std=c++17
    -arch=sm_90a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler -fPIC
    -shared
    -O3
    -DNDEBUG
    -I"${CUTLASS_DIR}/include"
    -I"${CUTLASS_DIR}/tools/util/include"
    -I"${CUTLASS_DIR}/examples/55_hopper_mixed_dtype_gemm"
    -lineinfo
)

echo "=== Building rvllm_w4a8 GEMM .so ==="
nvcc "${NVCC_FLAGS[@]}" \
    -o "${OUT_DIR}/libw4a8_gemm.so" \
    "${SCRIPT_DIR}/cutlass_w4a8_wrapper.cu"

echo "=== Done ==="
ls -la "${OUT_DIR}/libw4a8_gemm.so"
nm -D --defined-only "${OUT_DIR}/libw4a8_gemm.so" | grep rvllm_w4a8
