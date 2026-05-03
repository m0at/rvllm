#!/bin/bash
# Compile sm_120 CUTLASS FP8 blockwise GEMM into a shared library for
# FFI from Rust on Blackwell-Geforce / DGX Spark (sm_121 = family-compat
# with sm_120).
#
# Produces: kernels/sm_120/libcutlass_sm120.so
#
# This is a SEPARATE .so from `build_cutlass_so.sh` (SM90) — the
# SM120 build uses different CUTLASS schedules + arch targets, and
# we keep it split so the SM90 prod path stays untouched.
#
# Requires: CUTLASS at $CUTLASS_DIR (defaults to the vendored
# submodule at <repo>/cutlass — initialise with
# `git submodule update --init cutlass`).
#
# Usage:
#   ./kernels/build_cutlass_sm120_so.sh           # auto-detect via nvidia-smi
#   ./kernels/build_cutlass_sm120_so.sh sm_121a   # override arch
#   CUTLASS_DIR=/path/to/cutlass ./kernels/build_cutlass_sm120_so.sh

set -euo pipefail

# Codex30-1: auto-detect compute capability so the default landing
# directory matches what the runtime loader looks at. Earlier this
# defaulted hard to sm_120a→kernels/sm_120/, which is correct on
# RTX 5090 / RTX 6000 Blackwell but WRONG on DGX Spark / GB10
# (sm_121a, loader searches kernels/sm_121/ only after Codex27-1).
# Anyone running the helper without an explicit arg on a sm_121
# host got CutlassBackend::Absent at runtime even after a "successful"
# build. Auto-detect first, fall back to sm_120a if nvidia-smi can't
# tell us (cross-build / CI without GPU).
detect_arch() {
    local cc
    if cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '); then
        case "$cc" in
            12.0) echo "sm_120a"; return 0 ;;
            12.1) echo "sm_121a"; return 0 ;;
            12.2) echo "sm_122a"; return 0 ;;
        esac
    fi
    echo "sm_120a"  # safe fallback — matches the older default
}
ARCH=${1:-$(detect_arch)}
OUT_SUBDIR=${ARCH%a}   # sm_120a -> sm_120 for the per-arch kernel dir
echo "  target arch: $ARCH (output: $OUT_SUBDIR/)"

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
CUTLASS_DIR=${CUTLASS_DIR:-$REPO/cutlass}

if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
    echo "CUTLASS not found at $CUTLASS_DIR"
    echo "  (submodule init:  git submodule update --init cutlass)"
    exit 1
fi

cd "$DIR"
mkdir -p "$OUT_SUBDIR"
OBJ_DIR="$OUT_SUBDIR/obj"
mkdir -p "$OBJ_DIR"

NVCC=${NVCC:-nvcc}
NVCC_FLAGS="-std=c++17 -arch=${ARCH} --expt-relaxed-constexpr -O3 \
    --use_fast_math \
    -I${CUTLASS_DIR}/include \
    -I${CUTLASS_DIR}/tools/util/include \
    --compiler-options -fPIC \
    ${EXTRA_NVCC_FLAGS:-}"

echo "Building CUTLASS sm_120 shared library ($ARCH)..."

OK=0
FAIL=0
OBJS=""

# Sources that build the Blackwell-Geforce FP8 GEMM .so. Start with
# one — add more (autotune variants, nvfp4, etc.) here as they land.
SOURCES=(
    cutlass_fp8_gemm_blockscale_sm120.cu
)

for f in "${SOURCES[@]}"; do
    [ -f "$f" ] || { echo "  missing source $f — skipping"; continue; }
    stem=${f%.cu}
    obj="$OBJ_DIR/${stem}.o"
    echo -n "  $f -> ${stem}.o ... "
    if $NVCC -c $NVCC_FLAGS -o "$obj" "$f" 2>/tmp/nvcc_sm120_${stem}.log; then
        echo "ok"
        OBJS="$OBJS $obj"
        OK=$((OK + 1))
    else
        echo "FAILED  (see /tmp/nvcc_sm120_${stem}.log)"
        tail -10 /tmp/nvcc_sm120_${stem}.log 2>/dev/null
        FAIL=$((FAIL + 1))
    fi
done

if [ -z "$OBJS" ]; then
    echo "no objects compiled, cannot link"
    exit 1
fi

SO_PATH="$OUT_SUBDIR/libcutlass_sm120.so"
echo -n "  linking $SO_PATH ... "
if $NVCC -shared -o "$SO_PATH" $OBJS -lcudart 2>/tmp/nvcc_sm120_link.log; then
    echo "ok"
else
    echo "FAILED"
    tail -5 /tmp/nvcc_sm120_link.log 2>/dev/null
    exit 1
fi

echo ""
echo "CUTLASS sm_120 library: $DIR/$SO_PATH"
echo "Compiled $OK sources ($FAIL failed)"
