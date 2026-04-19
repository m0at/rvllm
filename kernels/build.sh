#!/bin/bash
# Compile CUDA kernels to PTX for runtime loading via cuModuleLoadData.
#
# Usage: ./build.sh [arch]
#   ./build.sh              # compile for all supported architectures
#   ./build.sh sm_80        # compile for A100 only
#   CUDA_ARCH=sm_90 ./build.sh  # compile for H100 only
#
# Environment variables:
#   NVCC       - path to nvcc (default: nvcc)
#   CUDA_ARCH  - target architecture (overrides default multi-arch build)

set -e

NVCC=${NVCC:-nvcc}
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Supported compute capabilities
# sm_70  = V100
# sm_75  = T4, RTX 2080
# sm_80  = A100, A30
# sm_86  = RTX 3090, A40
# sm_89  = RTX 4090, L40S
# sm_90  = H100, H200
# sm_100 = B100, B200
# sm_120 = RTX 5090, RTX 6000 Blackwell
# sm_121 = GB10 (Project DIGITS / DGX Spark — Grace+Blackwell consumer)
# sm_122 = RTX 5080, RTX 5070
ALL_ARCHS="sm_70 sm_75 sm_80 sm_86 sm_89 sm_90"

# Check nvcc version for sm_100 support (CUDA 12.8+)
NVCC_VERSION=$($NVCC --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "0.0")
NVCC_MAJOR=$(echo "$NVCC_VERSION" | cut -d. -f1)
NVCC_MINOR=$(echo "$NVCC_VERSION" | cut -d. -f2)
if [ "$NVCC_MAJOR" -ge 13 ] || ([ "$NVCC_MAJOR" -eq 12 ] && [ "$NVCC_MINOR" -ge 8 ]); then
    ALL_ARCHS="$ALL_ARCHS sm_100"
fi

# Check nvcc version for sm_120/sm_121/sm_122 support (CUDA 13.0+)
if [ "$NVCC_MAJOR" -ge 13 ]; then
    ALL_ARCHS="$ALL_ARCHS sm_120 sm_121 sm_122"
fi

# Use specific arch if provided
if [ -n "$1" ]; then
    ARCHS="$1"
elif [ -n "$CUDA_ARCH" ]; then
    ARCHS="$CUDA_ARCH"
else
    ARCHS="$ALL_ARCHS"
fi

# CUTLASS include paths (if available)
CUTLASS_DIR=${CUTLASS_DIR:-/root/cutlass}
CUTLASS_FLAGS=""
if [ -d "$CUTLASS_DIR/include" ]; then
    CUTLASS_FLAGS="-I$CUTLASS_DIR/include -I$CUTLASS_DIR/tools/util/include --expt-relaxed-constexpr"
    echo "CUTLASS: $CUTLASS_DIR"
else
    echo "CUTLASS: not found (CUTLASS kernels will be skipped)"
fi

echo "NVCC: $($NVCC --version 2>/dev/null | tail -1)"
echo "Target architectures: $ARCHS"
echo ""

compile_kernel() {
    local cu="$1" arch="$2" ptx="$3"
    local base=$(basename "$cu" .cu)
    local extra_flags=""
    # CUTLASS kernels need extra flags + C++17
    case "$base" in cutlass_*)
        if [ -z "$CUTLASS_FLAGS" ]; then
            echo "  SKIP: $base.cu (CUTLASS not found)"
            return 0
        fi
        extra_flags="$CUTLASS_FLAGS -std=c++17"
        ;;
    esac
    $NVCC -ptx -arch="$arch" -O3 $extra_flags -o "$ptx" "$cu" 2>/dev/null
}

REVISION=$(git -C "$DIR" rev-parse --short HEAD 2>/dev/null || echo "dev")

for arch in $ARCHS; do
    echo "=== Compiling for $arch ==="
    OUTDIR="$DIR/$arch"
    mkdir -p "$OUTDIR"

    for cu in "$DIR"/*.cu; do
        [ -f "$cu" ] || continue
        base=$(basename "$cu" .cu)
        ptx="$OUTDIR/${base}.ptx"
        echo "  $base.cu -> $arch/$base.ptx"
        compile_kernel "$cu" "$arch" "$ptx" || {
            echo "  WARNING: $base.cu failed for $arch (may need newer CUDA toolkit)"
        }
    done

    "$DIR/gen_manifest.sh" "$OUTDIR" "$REVISION" || {
        echo "  WARNING: manifest generation failed for $arch"
    }
done

# Also compile a default set (sm_80) in the root for backward compat
echo ""
echo "=== Default PTX (sm_80) ==="
for cu in "$DIR"/*.cu; do
    [ -f "$cu" ] || continue
    base=$(basename "$cu" .cu)
    ptx="$DIR/${base}.ptx"
    echo "  $base.cu -> $base.ptx"
    compile_kernel "$cu" "sm_80" "$ptx" || true
done

echo ""
echo "Done. PTX files in: $DIR/<arch>/*.ptx"
echo "Default (sm_80) PTX in: $DIR/*.ptx"
