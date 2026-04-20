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
if [ ! -f "$SM120_SO" ]; then
    echo "missing: $SM120_SO"
    echo "build with: $REPO_ROOT/kernels/build_cutlass_sm120_so.sh sm_121a"
    exit 1
fi

POLICY="$KERNELS/.probe-minimal-policy.json"
MODEL="${RVLLM_MODEL_DIR:-$HOME/.vllm/models/gemma-4-31b-it-fp8-block}"

export RVLLM_MODEL_DIR="$MODEL"
export RVLLM_KERNELS_DIR="$KERNELS"
export RVLLM_CUTLASS_SO="$SM120_SO"      # unused on sm_121 path but env var required
export RVLLM_FA3_SO="$SM120_SO"          # placeholder — Fa2Ptx takes over on gb10
export RVLLM_POLICY="$POLICY"
export RVLLM_BATCH="$BATCH"
export RVLLM_ITERS="$ITERS"
export RVLLM_WARMUP="$WARMUP"
export RVLLM_CUTLASS_SM120_SO="$SM120_SO"
export RVLLM_FP8_GEMM_CUTLASS_SM120=1    # opt-in the CUTLASS blockwise path
export RVLLM_ARENA_GB="${RVLLM_ARENA_GB:-40}"
# Force FP8 KV so attention goes through `PagedDecodeFp8Launcher`,
# which Fa2Ptx implements via `flash_attention_2_decode_fp8kv_kernel`.
# The F16 `PagedDecodeLauncher` is not wired for Fa2Ptx and would
# abort with `FeatureNotAvailable`.
export RVLLM_F16_KV="${RVLLM_F16_KV:-0}"

echo "== rvllm-bench (sm_121 / CUTLASS blockwise) =="
echo "  batch=$BATCH iters=$ITERS warmup=$WARMUP"
echo "  model=$MODEL"
echo "  sm120_so=$SM120_SO"
echo

exec "$BIN"
