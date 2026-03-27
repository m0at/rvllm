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
ALL_ARCHS="sm_70 sm_75 sm_80 sm_86 sm_89 sm_90"

# Check nvcc version for sm_100 support (CUDA 12.8+)
NVCC_VERSION=$($NVCC --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "0.0")
NVCC_MAJOR=$(echo "$NVCC_VERSION" | cut -d. -f1)
NVCC_MINOR=$(echo "$NVCC_VERSION" | cut -d. -f2)
if [ "$NVCC_MAJOR" -ge 13 ] || ([ "$NVCC_MAJOR" -eq 12 ] && [ "$NVCC_MINOR" -ge 8 ]); then
    ALL_ARCHS="$ALL_ARCHS sm_100"
fi

# Use specific arch if provided
if [ -n "$1" ]; then
    ARCHS="$1"
elif [ -n "$CUDA_ARCH" ]; then
    ARCHS="$CUDA_ARCH"
else
    ARCHS="$ALL_ARCHS"
fi

echo "NVCC: $($NVCC --version 2>/dev/null | tail -1)"
echo "Target architectures: $ARCHS"
echo ""

for arch in $ARCHS; do
    echo "=== Compiling for $arch ==="
    OUTDIR="$DIR/$arch"
    mkdir -p "$OUTDIR"

    for cu in "$DIR"/*.cu; do
        [ -f "$cu" ] || continue
        base=$(basename "$cu" .cu)
        ptx="$OUTDIR/${base}.ptx"
        echo "  $base.cu -> $arch/$base.ptx"
        $NVCC -ptx -arch="$arch" -O3 -o "$ptx" "$cu" 2>/dev/null || {
            echo "  WARNING: $base.cu failed for $arch (may need newer CUDA toolkit)"
        }
    done
done

# Also compile a default set (sm_80) in the root for backward compat
echo ""
echo "=== Default PTX (sm_80) ==="
for cu in "$DIR"/*.cu; do
    [ -f "$cu" ] || continue
    base=$(basename "$cu" .cu)
    ptx="$DIR/${base}.ptx"
    echo "  $base.cu -> $base.ptx"
    $NVCC -ptx -arch=sm_80 -O3 -o "$ptx" "$cu" 2>/dev/null || true
done

echo ""
echo "Done. PTX files in: $DIR/<arch>/*.ptx"
echo "Default (sm_80) PTX in: $DIR/*.ptx"
