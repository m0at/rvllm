#!/usr/bin/env bash
# rvllm-bench wrapper for GB10 / DGX Spark (sm_121).
#
# Points the binary at:
#   * model:        ~/.vllm/models/gemma-4-31b-it-fp8-block
#   * kernels dir:  <repo>/kernels (resolves sm_121/ subdir + manifest)
#   * CUTLASS SM120 .so: <repo>/kernels/sm_121/libcutlass_sm120.so
#                   (opted in via RVLLM_FP8_GEMM_CUTLASS_SM120=1)
#   * policy.json:  <repo>/kernels/.probe-minimal-policy.json
#                   (empty policy — SM121 does not use the SM90 .so
#                    variant table; the CUTLASS .so is still required
#                    as a file by the bench env but `load_for(Sm121)`
#                    returns before it's opened)
#   * FA3 .so:      <repo>/kernels/sm_121/libcutlass_sm120.so
#                   (placeholder — gb10 feature routes attention to
#                    Fa2Ptx so the FA3 symbol table is never read)
#
# Usage:
#   ./scripts/bench_sm121.sh [batch] [iters]
#     batch   default 32   (decode batch size)
#     iters   default 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V3_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V3_DIR/.." && pwd)"

BATCH=${1:-32}
ITERS=${2:-50}
WARMUP=${WARMUP:-10}

BIN="$V3_DIR/target/release/rvllm-bench"
if [ ! -x "$BIN" ]; then
    echo "binary not found — build with:"
    echo "  cargo build --release -p rvllm-bench --features gb10"
    exit 1
fi

KERNELS="$REPO_ROOT/kernels"
SM120_SO="$KERNELS/sm_121/libcutlass_sm120.so"
# Codex29-4: runtime supports CutlassBackend::Absent (PTX fallback);
# script used to abort hard when the .so was missing, blocking
# baseline / regression benches on hosts that haven't built CUTLASS.
# Now warn + fall back to the PTX path so the bench still runs.
HAVE_SM120_SO=1
if [ ! -f "$SM120_SO" ]; then
    echo "WARN: $SM120_SO missing — falling back to PTX (CutlassBackend::Absent)."
    echo "      For the CUTLASS-blockwise path: $REPO_ROOT/kernels/build_cutlass_sm120_so.sh sm_121a"
    HAVE_SM120_SO=0
fi

POLICY="$KERNELS/.probe-minimal-policy.json"
MODEL="${RVLLM_MODEL_DIR:-$HOME/.vllm/models/gemma-4-31b-it-fp8-block}"

export RVLLM_MODEL_DIR="$MODEL"
export RVLLM_KERNELS_DIR="$KERNELS"
# When the .so is missing we still need *some* path-shaped value for
# the env vars the bench reads; point them at /dev/null so any
# accidental open() fails loud rather than silently loading whatever
# happens to live at $SM120_SO.
SO_OR_DEVNULL="${SM120_SO}"
[ "$HAVE_SM120_SO" = "0" ] && SO_OR_DEVNULL="/dev/null"
export RVLLM_CUTLASS_SO="$SO_OR_DEVNULL"      # unused on sm_121 path but env var required
export RVLLM_FA3_SO="$SO_OR_DEVNULL"          # placeholder — Fa2Ptx takes over on gb10
export RVLLM_POLICY="$POLICY"
export RVLLM_BATCH="$BATCH"
export RVLLM_ITERS="$ITERS"
export RVLLM_WARMUP="$WARMUP"
if [ "$HAVE_SM120_SO" = "1" ]; then
    export RVLLM_CUTLASS_SM120_SO="$SM120_SO"
    export RVLLM_FP8_GEMM_CUTLASS_SM120=1    # opt-in the CUTLASS blockwise path
fi
export RVLLM_ARENA_GB="${RVLLM_ARENA_GB:-40}"
# Codex27-2: this only opts OUT of the F16 KV path (which on sm_121
# would hit `Fa2Ptx::paged_decode → FeatureNotAvailable`). The
# resulting default is **NVFP4**, not FP8 — that's the production
# attention path on sm_121 today. To explicitly bench FP8 KV instead,
# also set `RVLLM_FP8_KV=1` (forces fp8 on every layer regardless of
# kv_dtype default).
export RVLLM_F16_KV="${RVLLM_F16_KV:-0}"
# Default-NVFP4 path is what we want for production-shape numbers;
# don't quietly change attention dtype unless the caller asks.

echo "== rvllm-bench (sm_121 / CUTLASS blockwise) =="
echo "  batch=$BATCH iters=$ITERS warmup=$WARMUP"
echo "  model=$MODEL"
echo "  sm120_so=$SM120_SO"
echo

exec "$BIN"
