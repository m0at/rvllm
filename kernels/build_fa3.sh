#!/bin/bash
# Build FA3 SM90 decode kernel as a shared library on H100.
# Requires: CUDA 12.x, CUTLASS headers at /root/cutlass, FA3 source at /root/flash-attention
#
# Parallel build: each .cu compiles to .o independently, then links once.
# Cuts wall time from ~5min to ~1-2min on a 96-core box.
set -euo pipefail

FA3_DIR="${FA3_DIR:-/root/flash-attention/hopper}"
CUTLASS_DIR="${CUTLASS_DIR:-/root/cutlass}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/sm_90"
OBJ_DIR="${OUT_DIR}/.fa3_obj"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "${OUT_DIR}" "${OBJ_DIR}"

NVCC_FLAGS=(
    -std=c++17
    -arch=sm_90a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler -fPIC
    -O3
    -DNDEBUG
    -I"${CUTLASS_DIR}/include"
    -I"${FA3_DIR}"
    -I"${SCRIPT_DIR}"
    -lineinfo
)

echo "=== Building FA3 SM90 (head_dim=128,256) — ${JOBS} parallel jobs ==="

SRCS=(
    "${SCRIPT_DIR}/fa3_sm90_wrapper.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_fp16_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_fp16_paged_split_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_e4m3_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_e4m3_paged_split_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_fp16_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_fp16_paged_split_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_e4m3_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim256_e4m3_paged_split_sm90.cu"
    "${FA3_DIR}/flash_fwd_combine.cu"
    "${FA3_DIR}/flash_prepare_scheduler.cu"
)

# Phase 1: compile each .cu to .o in parallel
echo "Compiling ${#SRCS[@]} translation units (${JOBS} parallel)..."
OBJS=()
PIDS=()
T0=$(date +%s)

for src in "${SRCS[@]}"; do
    base=$(basename "${src}" .cu)
    obj="${OBJ_DIR}/${base}.o"
    OBJS+=("${obj}")
    nvcc "${NVCC_FLAGS[@]}" -c "${src}" -o "${obj}" 2>&1 &
    PIDS+=($!)
    # Throttle to JOBS
    while [ "$(jobs -r | wc -l)" -ge "${JOBS}" ]; do
        wait -n 2>/dev/null || true
    done
done

# Wait for all
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
if [ "$FAIL" -ne 0 ]; then
    echo "ERROR: one or more nvcc compilations failed"
    exit 1
fi

T1=$(date +%s)
echo "Compilation: $((T1 - T0))s"

# Phase 2: link .o files into shared library
echo "Linking..."
nvcc -shared -arch=sm_90a -o "${OUT_DIR}/libfa3_kernels.so" "${OBJS[@]}" 2>&1
T2=$(date +%s)
echo "Link: $((T2 - T1))s"

SZ=$(stat -c%s "${OUT_DIR}/libfa3_kernels.so" 2>/dev/null || stat -f%z "${OUT_DIR}/libfa3_kernels.so")
echo ""
echo "=== FA3 build complete ($((T2 - T0))s total) ==="
echo "  Size: ${SZ} bytes"
echo "  Path: ${OUT_DIR}/libfa3_kernels.so"

if [ "$SZ" -lt 1000000 ]; then
    echo "WARNING: .so seems too small (<1MB), may not contain GPU code"
fi

echo ""
echo "Exported symbols:"
nm -D "${OUT_DIR}/libfa3_kernels.so" | grep -E 'fa3_sm90' || echo "  (none found - check build)"

rm -rf "${OBJ_DIR}"
