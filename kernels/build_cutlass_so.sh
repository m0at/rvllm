#!/bin/bash
# Compile CUTLASS kernels into a shared library (.so) for FFI from Rust.
# CUTLASS kernels are self-launching -- they compute their own grid dims internally.
# You can't launch them as raw PTX via cuLaunchKernel. They need their C++ wrapper
# function called from the host, so we compile to .so and dlopen from Rust.
#
# Requires: CUTLASS headers at $CUTLASS_DIR (default /root/cutlass)
# Usage: ./kernels/build_cutlass_so.sh [arch] [cutlass_dir]

set -euo pipefail

ARCH=${1:-sm_90}
CUTLASS_DIR=${2:-/root/cutlass}

if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
    echo "CUTLASS not found at $CUTLASS_DIR, cloning..."
    git clone --depth 1 https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
mkdir -p "$ARCH"
OBJ_DIR="$ARCH/obj"
mkdir -p "$OBJ_DIR"

NVCC=${NVCC:-nvcc}
# sm_90+ supports arch-specific features suffix (for TMA, WGMMA, FP8 tensor cores)
# sm_80 and sm_86 do not have these features and nvcc rejects the 'a' suffix
case "$ARCH" in
    sm_90|sm_100|sm_120|sm_122) ARCH_FLAG="${ARCH}a" ;;
    *) ARCH_FLAG="$ARCH" ;;
esac

NVCC_FLAGS="-std=c++17 -arch=${ARCH_FLAG} --expt-relaxed-constexpr -O3 --use_fast_math \
    -I${CUTLASS_DIR}/include \
    -I${CUTLASS_DIR}/tools/util/include \
    -I${CUTLASS_DIR}/examples/45_dual_gemm \
    --compiler-options -fPIC"

GATE_DEFINES=""
for var in \
    RVLLM_CUTLASS_GATE_TILE_M \
    RVLLM_CUTLASS_GATE_TILE_N \
    RVLLM_CUTLASS_GATE_TILE_K \
    RVLLM_CUTLASS_GATE_CLUSTER_M \
    RVLLM_CUTLASS_GATE_CLUSTER_N \
    RVLLM_CUTLASS_GATE_CLUSTER_K \
    RVLLM_CUTLASS_GATE_SCHEDULE; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        GATE_DEFINES="$GATE_DEFINES -D${var}=${val}"
    fi
done

GATE_SOURCE_GEN=${RVLLM_CUTLASS_GATE_SOURCE_GEN:-0}

generate_gate_source() {
    local src="$1"
    local out="$2"
    cp "$src" "$out"
    perl -0pi -e 's@\n#ifndef RVLLM_CUTLASS_GATE_TILE_M.*?#define RVLLM_CUTE_INT\(x\) RVLLM_CUTE_INT_\(x\)\n@@s' "$out"
    perl -0pi -e "s@using GateTileShape = Shape<.*?>;@using GateTileShape = Shape<_${RVLLM_CUTLASS_GATE_TILE_M}, _${RVLLM_CUTLASS_GATE_TILE_N}, _${RVLLM_CUTLASS_GATE_TILE_K}>;@s" "$out"
    perl -0pi -e "s@using GateClusterShape = Shape<.*?>;@using GateClusterShape = Shape<_${RVLLM_CUTLASS_GATE_CLUSTER_M}, _${RVLLM_CUTLASS_GATE_CLUSTER_N}, _${RVLLM_CUTLASS_GATE_CLUSTER_K}>;@s" "$out"
    if [ "${RVLLM_CUTLASS_GATE_SCHEDULE:-0}" = "1" ]; then
        perl -0pi -e 's@#if RVLLM_CUTLASS_GATE_SCHEDULE == 1\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;\n#else\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;\n#endif@using GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;@s' "$out"
    else
        perl -0pi -e 's@#if RVLLM_CUTLASS_GATE_SCHEDULE == 1\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;\n#else\nusing GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;\n#endif@using GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;\nusing GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;@s' "$out"
    fi
}

echo "Building CUTLASS shared library for $ARCH..."
if [ -n "$GATE_DEFINES" ]; then
    echo "  cutlass_gateup_silu.cu defines:$GATE_DEFINES"
fi

# Compile each .cu to an object file separately to avoid template conflicts
OK=0
FAIL=0
OBJS=""
GATE_ONLY=${RVLLM_CUTLASS_GATE_ONLY:-0}

for f in cutlass_qkv_bias.cu cutlass_oproj_residual.cu cutlass_gateup_silu.cu cutlass_gemm.cu cutlass_fp8_gemm.cu cutlass_hgemm_autotune.cu cutlass_oproj_residual_autotune.cu cutlass_gateup_silu_autotune.cu cutlass_fp8_gemm_autotune.cu cutlass_fp8_gemm_residual.cu cutlass_fp8_gemm_channelscale.cu; do
    [ -f "$f" ] || continue
    stem=${f%.cu}
    obj="$OBJ_DIR/${stem}.o"
    EXTRA_FLAGS=""
    SOURCE_FILE="$f"
    if [ "$f" = "cutlass_gateup_silu.cu" ]; then
        EXTRA_FLAGS="$GATE_DEFINES"
        if [ "$GATE_SOURCE_GEN" = "1" ]; then
            SOURCE_FILE="$ARCH/generated_${stem}.cu"
            generate_gate_source "$f" "$SOURCE_FILE"
            EXTRA_FLAGS=""
        fi
    fi
    if [ "$GATE_ONLY" = "1" ] && [ "$f" != "cutlass_gateup_silu.cu" ] && [ -f "$obj" ]; then
        echo "  $f -> ${stem}.o ... cached"
        OBJS="$OBJS $obj"
        continue
    fi
    echo -n "  $f -> ${stem}.o ... "
    if $NVCC -c $NVCC_FLAGS $EXTRA_FLAGS -o "$obj" "$SOURCE_FILE" 2>/tmp/nvcc_so_${stem}.log; then
        echo "ok"
        OBJS="$OBJS $obj"
        OK=$((OK + 1))
    else
        echo "FAILED"
        FAIL=$((FAIL + 1))
        tail -5 /tmp/nvcc_so_${stem}.log 2>/dev/null
    fi
done

if [ -z "$OBJS" ]; then
    echo "No objects compiled, cannot link."
    exit 1
fi

# Link into shared library
echo -n "  Linking libcutlass_kernels.so ... "
if $NVCC -shared -o "$ARCH/libcutlass_kernels.so" $OBJS -lcudart 2>/tmp/nvcc_so_link.log; then
    echo "ok"
else
    echo "FAILED"
    tail -5 /tmp/nvcc_so_link.log 2>/dev/null
    exit 1
fi

echo ""
echo "CUTLASS shared library: $DIR/$ARCH/libcutlass_kernels.so"
echo "Compiled $OK kernel files ($FAIL failed)"
