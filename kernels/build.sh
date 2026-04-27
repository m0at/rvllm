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
    # FP8 / NVFP4 tensor-core MMA kernels need the arch-specific
    # feature set (`sm_121a` / `sm_120a`) because `.kind::f8f6f4`,
    # `mma.kind::*f8*`, and the NVFP4 hardware dequant instructions
    # (`cvt.rn.*.e2m1x2`) live in the CUDA family-specific PTX feature
    # set. Plain `sm_121` rejects them at ptxas time even though nvcc
    # -ptx emits the instruction successfully.
    if grep -q 'kind::f8f6f4\|mma.sync.*e4m3\|mma.sync.*e2m1\|fp8_mma_frag_pack\|mma_m16n8k32\|cvt.rn.f16x2.e2m1x2\|cvt.rn.bf16x2.e2m1x2\|cvt.rn.e2m1x2' "$cu" 2>/dev/null; then
        case "$arch" in
            sm_120) arch="sm_120a" ;;
            sm_121) arch="sm_121a" ;;
            sm_122) arch="sm_122a" ;;
        esac
    fi
    # Cycle 52 step 11d: stop swallowing nvcc errors. The cycle-21
    # NVFP4 split-decode tmp_out f16->f32 fix updated the main kernel
    # but not the BC16 variant; the resulting type mismatch was
    # silently dropped here for ~30 cycles, leaving a stale PTX in
    # the manifest with the wrong ABI. Print stderr so the next such
    # bug surfaces immediately.
    $NVCC -ptx -arch="$arch" -O3 $extra_flags -o "$ptx" "$cu"
}

# Revision pinned into the generated manifest.json. Precedence:
#   1. $REVISION env var (tarball / CI shallow-clone builds where
#      git isn't available or history is detached)
#   2. git short HEAD
#   3. literal "dev" (local hack build)
REVISION="${REVISION:-$(git -C "$DIR" rev-parse --short HEAD 2>/dev/null || echo "dev")}"

# Secondary kernel tree: `v3/kernels/`. Upstream shipped a second
# source tree with Gemma-4-specific fused kernels (RoPE partial, norm
# + add-residual, qk-rmsnorm, etc.) separately from this top-level
# tree. Runtime loaders reference both sets under the same manifest,
# so the per-arch OUTDIR must contain PTX for both. We co-locate the
# output in OUTDIR (no name collisions verified at branch authoring
# time; a collision would overwrite silently — audit if one appears).
#
# `$DIR/../v3/kernels` is optional — if the upstream tree ever moves
# or the v3 prefix changes, this falls through without breaking the
# top-level build.
V3_KERNELS_DIR="$DIR/../v3/kernels"

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

    if [ -d "$V3_KERNELS_DIR" ]; then
        for cu in "$V3_KERNELS_DIR"/*.cu; do
            [ -f "$cu" ] || continue
            base=$(basename "$cu" .cu)
            ptx="$OUTDIR/${base}.ptx"
            echo "  (v3) $base.cu -> $arch/$base.ptx"
            compile_kernel "$cu" "$arch" "$ptx" || {
                echo "  WARNING: v3/$base.cu failed for $arch (may need newer CUDA toolkit)"
            }
        done
    fi

    "$DIR/gen_manifest.sh" "$OUTDIR" "$REVISION" || {
        echo "  WARNING: manifest generation failed for $arch"
    }
done

# Also compile a default set (sm_80) in the root for backward compat
# with old bench scripts that look at `kernels/*.ptx`. Only run when
# the caller didn't specifically ask for a single arch — otherwise
# the explicit arch wins and we don't leave stale top-level artifacts.
if [ -z "$1" ] && [ -z "$CUDA_ARCH" ]; then
echo ""
echo "=== Default PTX (sm_80) ==="
for cu in "$DIR"/*.cu; do
    [ -f "$cu" ] || continue
    base=$(basename "$cu" .cu)
    ptx="$DIR/${base}.ptx"
    echo "  $base.cu -> $base.ptx"
    compile_kernel "$cu" "sm_80" "$ptx" || true
done
fi   # end: default sm_80 loop (skipped when specific arch requested)

echo ""
echo "Done. PTX files in: $DIR/<arch>/*.ptx"
echo "Default (sm_80) PTX in: $DIR/*.ptx"
