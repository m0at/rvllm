#!/bin/bash
# Pre-deploy build validation on macOS (no GPU required).
#
# Runs inside Docker with CUDA toolkit to:
#   1. Compile all .cu kernels to PTX (catches syntax/type errors)
#   2. cargo check --features cuda (catches Rust integration errors)
#   3. cargo test --workspace (unit tests, mock-gpu)
#   4. Optionally: cargo build --release --features cuda (full binary)
#
# Usage:
#   ./scripts/validate-local.sh              # quick: kernels + check + test
#   ./scripts/validate-local.sh --full       # also builds release binary
#   ./scripts/validate-local.sh --kernels    # kernels only (fastest)
#   ./scripts/validate-local.sh --arch sm_90 # specific arch (default: sm_80,sm_90)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="rvllm-validate:latest"
MODE="quick"
ARCH="sm_80,sm_90"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)    MODE="full"; shift ;;
        --kernels) MODE="kernels"; shift ;;
        --arch)    ARCH="$2"; shift 2 ;;
        --rebuild) REBUILD=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--full|--kernels] [--arch sm_XX] [--rebuild]"
            echo ""
            echo "Modes:"
            echo "  (default)   Compile kernels + cargo check + cargo test"
            echo "  --full      Also build release binary"
            echo "  --kernels   Only compile .cu -> PTX (fastest)"
            echo ""
            echo "Options:"
            echo "  --arch      Target arch(s), comma-separated (default: sm_80,sm_90)"
            echo "  --rebuild   Force rebuild of validation Docker image"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build validation image if needed
if [[ -n "$REBUILD" ]] || ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "=== Building validation image (cached after first run) ==="
    docker build -t "$IMAGE" -f "$ROOT/Dockerfile.validate" "$ROOT" \
        --build-arg BUILDKIT_INLINE_CACHE=1
    echo ""
fi

# Construct validation script to run inside container
VALIDATE_SCRIPT=$(cat <<'INNEREOF'
#!/bin/bash
set -e

ARCH="${CUDA_ARCH:-sm_80,sm_90}"
MODE="${VALIDATE_MODE:-quick}"
PASS=0
FAIL=0
SKIP=0

report() {
    local status="$1" name="$2"
    if [ "$status" = "PASS" ]; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    elif [ "$status" = "FAIL" ]; then
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
    else
        echo "  [SKIP] $name"
        SKIP=$((SKIP + 1))
    fi
}

echo "==========================================="
echo "  rvLLM Local Build Validation"
echo "==========================================="
echo "  Mode:   $MODE"
echo "  Arch:   $ARCH"
echo "  NVCC:   $(nvcc --version 2>/dev/null | tail -1)"
echo "  Rust:   $(rustc --version)"
echo "  Cargo:  $(cargo --version)"
echo "==========================================="
echo ""

# --- Step 1: Compile CUDA kernels to PTX ---
echo "=== Step 1: CUDA Kernel Compilation ==="
cd /src/kernels

CUTLASS_FLAGS=""
if [ -d "$CUTLASS_DIR/include" ]; then
    CUTLASS_FLAGS="-I$CUTLASS_DIR/include -I$CUTLASS_DIR/tools/util/include --expt-relaxed-constexpr"
    echo "  CUTLASS: $CUTLASS_DIR"
fi

IFS=',' read -ra ARCH_LIST <<< "$ARCH"
KERNEL_FAIL=0
KERNEL_PASS=0
KERNEL_SKIP=0

for arch in "${ARCH_LIST[@]}"; do
    echo ""
    echo "--- $arch ---"
    mkdir -p "/tmp/ptx-validate/$arch"
    for cu in *.cu; do
        [ -f "$cu" ] || continue
        base="${cu%.cu}"
        extra_flags=""

        # CUTLASS kernels need extra flags
        case "$base" in cutlass_*)
            if [ -z "$CUTLASS_FLAGS" ]; then
                echo "  SKIP $base.cu (no CUTLASS)"
                KERNEL_SKIP=$((KERNEL_SKIP + 1))
                continue
            fi
            extra_flags="$CUTLASS_FLAGS -std=c++17"
            ;;
        esac

        ptx="/tmp/ptx-validate/$arch/${base}.ptx"
        if nvcc -ptx -arch="$arch" -O3 $extra_flags -o "$ptx" "$cu" 2>/tmp/nvcc_err.log; then
            echo "  PASS $base.cu"
            KERNEL_PASS=$((KERNEL_PASS + 1))
        else
            echo "  FAIL $base.cu"
            cat /tmp/nvcc_err.log | head -20
            KERNEL_FAIL=$((KERNEL_FAIL + 1))
        fi
    done
done

echo ""
echo "  Kernels: $KERNEL_PASS passed, $KERNEL_FAIL failed, $KERNEL_SKIP skipped"

if [ "$KERNEL_FAIL" -gt 0 ]; then
    report "FAIL" "CUDA kernel compilation"
else
    report "PASS" "CUDA kernel compilation"
fi

# --- Step 1b: CUTLASS shared library compilation test ---
echo ""
echo "=== Step 1b: CUTLASS Shared Library ==="
CUTLASS_SO_OK=0
CUTLASS_SO_FAIL=0
SO_ARCH="${ARCH_LIST[0]}"

cd /src/kernels
mkdir -p "/tmp/cutlass-so-validate"

for f in cutlass_qkv_bias.cu cutlass_oproj_residual.cu cutlass_gateup_silu.cu cutlass_gemm.cu; do
    [ -f "$f" ] || continue
    stem="${f%.cu}"
    obj="/tmp/cutlass-so-validate/${stem}.o"
    if nvcc -c -std=c++17 -arch="${SO_ARCH}" --expt-relaxed-constexpr -O3 --use_fast_math \
        -I$CUTLASS_DIR/include -I$CUTLASS_DIR/tools/util/include \
        --compiler-options -fPIC -o "$obj" "$f" 2>/tmp/nvcc_so_err.log; then
        echo "  PASS $f -> ${stem}.o"
        CUTLASS_SO_OK=$((CUTLASS_SO_OK + 1))
    else
        echo "  FAIL $f"
        cat /tmp/nvcc_so_err.log | head -10
        CUTLASS_SO_FAIL=$((CUTLASS_SO_FAIL + 1))
    fi
done

# Link test
OBJS=$(ls /tmp/cutlass-so-validate/*.o 2>/dev/null)
if [ -n "$OBJS" ]; then
    if nvcc -shared -o /tmp/cutlass-so-validate/libcutlass_kernels.so $OBJS -lcudart 2>/tmp/nvcc_link_err.log; then
        echo "  PASS libcutlass_kernels.so linked"
    else
        echo "  FAIL link"
        cat /tmp/nvcc_link_err.log | head -5
        CUTLASS_SO_FAIL=$((CUTLASS_SO_FAIL + 1))
    fi
fi

if [ "$CUTLASS_SO_FAIL" -gt 0 ]; then
    report "FAIL" "CUTLASS shared library"
else
    report "PASS" "CUTLASS shared library ($CUTLASS_SO_OK objects)"
fi

if [ "$MODE" = "kernels" ]; then
    echo ""
    echo "==========================================="
    echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
    echo "==========================================="
    [ "$FAIL" -eq 0 ] && exit 0 || exit 1
fi

# --- Step 2: cargo check with CUDA features ---
echo ""
echo "=== Step 2: Cargo Check (CUDA features) ==="
# Active workspace is /src/v3 (mounted from repo root v3/). The
# old `cd /src` + `cargo check --workspace --features rvllm/cuda`
# missed Cargo.toml entirely AND referenced an obsolete `rvllm`
# crate. Forward the CUDA path through the rvllm-runtime feature.
cd /src/v3
if CUDA_ARCH="${ARCH_LIST[0]}" cargo check --workspace --features rvllm-runtime/cuda 2>&1 | tail -5; then
    report "PASS" "cargo check --features cuda"
else
    report "FAIL" "cargo check --features cuda"
fi

# --- Step 3: cargo test (no-CUDA crates) ---
echo ""
echo "=== Step 3: Cargo Test (no-CUDA crates) ==="
# CUDA-feature crates can't run in CI without a GPU; they're
# excluded here.
if cargo test --workspace --exclude rvllm-runtime --exclude rvllm-bench --exclude rvllm-serve 2>&1 | tail -10; then
    report "PASS" "cargo test --workspace (no-cuda)"
else
    report "FAIL" "cargo test --workspace (no-cuda)"
fi

# --- Step 4 (--full only): Full release build ---
if [ "$MODE" = "full" ]; then
    echo ""
    echo "=== Step 4: Release Build ==="
    if CUDA_ARCH="${ARCH_LIST[0]}" cargo build --release --features cuda --bin rvllm-server 2>&1 | tail -5; then
        report "PASS" "cargo build --release --features cuda --bin rvllm-server"
    else
        report "FAIL" "cargo build --release --features cuda --bin rvllm-server"
    fi
fi

echo ""
echo "==========================================="
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "==========================================="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
INNEREOF
)

echo "=== Running validation (mode: $MODE, arch: $ARCH) ==="
echo ""

# Run validation in container with source mounted read-only.
# Use a named volume for cargo/target cache so incremental builds work.
docker run --rm \
    -v "$ROOT:/src:ro" \
    -v rvllm-validate-target:/tmp/rvllm-target \
    -v rvllm-validate-cargo:/root/.cargo/registry \
    -e CUDA_ARCH="$ARCH" \
    -e VALIDATE_MODE="$MODE" \
    "$IMAGE" \
    bash -c "$VALIDATE_SCRIPT"

EXIT=$?
if [ $EXIT -eq 0 ]; then
    echo ""
    echo "Validation passed -- safe to deploy."
else
    echo ""
    echo "Validation FAILED -- fix errors before deploying."
fi
exit $EXIT
