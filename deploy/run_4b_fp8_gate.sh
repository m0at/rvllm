#!/usr/bin/env bash
# H100-only quality gate for the Gemma 4B FP8 path.
set -euo pipefail
export PATH="/root/.cargo/bin:$HOME/.cargo/bin:$PATH"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "run_4b_fp8_gate.sh must run on the CUDA/H100 host" >&2
  exit 2
fi

ROOT="${RVLLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${RVLLM_MODEL_DIR:-/root/models/gemma-4-E4B-it-FP8}"
PROMPT="${RVLLM_PROMPT:-explain angular momentum}"
PPL_TEXT="${RVLLM_PPL_TEXT_INLINE:-Angular momentum is the rotational analogue of linear momentum. It depends on how much mass is rotating, how far that mass is from the axis, and how fast it turns. In a closed system, total angular momentum is conserved, so a spinning object speeds up when its mass moves closer to the rotation axis.}"
PPL_MAX="${RVLLM_PPL_MAX:-70}"

if [ -f "$ROOT/v3/Cargo.toml" ]; then
  WORKSPACE="$ROOT/v3"
elif [ -f "$ROOT/Cargo.toml" ] && [ -d "$ROOT/crates/rvllm-bench" ]; then
  WORKSPACE="$ROOT"
else
  echo "could not find rvLLM Cargo workspace under $ROOT" >&2
  exit 2
fi
SWARM_DIR="$WORKSPACE/crates/rvllm-swarm-egui"
TARGET_DIR="$WORKSPACE/target/release"

if [ -z "${RVLLM_KERNELS_DIR:-}" ]; then
  if [ -d "$ROOT/kernels/sm_90" ]; then
    export RVLLM_KERNELS_DIR="$ROOT/kernels/sm_90"
  elif [ -d /root/rvllm/kernels/sm_90 ]; then
    export RVLLM_KERNELS_DIR=/root/rvllm/kernels/sm_90
  else
    echo "set RVLLM_KERNELS_DIR to the sm_90 kernel directory" >&2
    exit 2
  fi
fi

export RVLLM_MODEL_DIR="$MODEL"
export RVLLM_CUTLASS_SO="${RVLLM_CUTLASS_SO:-$RVLLM_KERNELS_DIR/libcutlass_kernels.so}"
export RVLLM_FA3_SO="${RVLLM_FA3_SO:-$RVLLM_KERNELS_DIR/libfa3_kernels.so}"
export RVLLM_FA_FALLBACK_SO="${RVLLM_FA_FALLBACK_SO:-$RVLLM_KERNELS_DIR/libfa_sm89_kernels.so}"
export RVLLM_POLICY="${RVLLM_POLICY:-$RVLLM_KERNELS_DIR/policy.json}"
export RVLLM_ARENA_GB="${RVLLM_ARENA_GB:-48}"
export RVLLM_F16_KV="${RVLLM_F16_KV:-1}"

cd "$ROOT"

echo "== build rvllm-eval/rvllm-ppl/swarm-cli =="
cargo build --release --manifest-path "$WORKSPACE/Cargo.toml" -p rvllm-bench --features cuda --bin rvllm-eval --bin rvllm-ppl --bin rvllm-bench
(
  cd "$SWARM_DIR"
  cargo build --release --features cuda --bin swarm-cli
)

LOG_DIR="${RVLLM_GATE_LOG_DIR:-/tmp/rvllm-4b-fp8-gate}"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "== eval: $PROMPT =="
RVLLM_PROMPT="$PROMPT" RVLLM_MAX_TOKENS="${RVLLM_MAX_TOKENS:-96}" \
  timeout "${RVLLM_EVAL_TIMEOUT:-240}" "$TARGET_DIR/rvllm-eval" \
  2>&1 | tee "$LOG_DIR/eval.log"

if grep -Eai "(or ){20,}|(and ){20,}|�|nan|inf" "$LOG_DIR/eval.log" >/dev/null; then
  echo "eval gate failed: repetition/junk marker detected" >&2
  exit 1
fi

echo "== ppl spot =="
RVLLM_PROMPT="$PPL_TEXT" RVLLM_PPL_CHUNK="${RVLLM_PPL_CHUNK:-96}" RVLLM_PPL_CHUNKS="${RVLLM_PPL_CHUNKS:-1}" \
  timeout "${RVLLM_PPL_TIMEOUT:-240}" "$TARGET_DIR/rvllm-ppl" \
  2>&1 | tee "$LOG_DIR/ppl.log"

PPL="$(sed -n 's/.*"perplexity":\([0-9.]*\).*/\1/p' "$LOG_DIR/ppl.log" | tail -1)"
if [ -z "$PPL" ]; then
  echo "ppl gate failed: could not parse perplexity" >&2
  exit 1
fi
awk -v ppl="$PPL" -v max="$PPL_MAX" 'BEGIN { exit !(ppl <= max) }' || {
  echo "ppl gate failed: $PPL > $PPL_MAX" >&2
  exit 1
}

echo "== FP8 lm-head bench smoke =="
RVLLM_BATCH="${RVLLM_BENCH_BATCH:-1}" RVLLM_ITERS="${RVLLM_BENCH_ITERS:-2}" RVLLM_WARMUP="${RVLLM_BENCH_WARMUP:-1}" \
  timeout "${RVLLM_BENCH_TIMEOUT:-240}" "$TARGET_DIR/rvllm-bench" \
  2>&1 | tee "$LOG_DIR/bench.log"

if ! grep -F '"tok_per_sec":' "$LOG_DIR/bench.log" >/dev/null; then
  echo "bench gate failed: missing tok_per_sec JSON" >&2
  exit 1
fi

echo "== B=30 broadcast smoke =="
rm -rf "$LOG_DIR/swarm-root"
(
  cd "$SWARM_DIR"
  SWARM_REPO_ROOT="$LOG_DIR/swarm-root" \
  RVLLM_MAX_NEW_TOKENS="${RVLLM_SWARM_MAX_NEW_TOKENS:-64}" \
  RVLLM_SWARM_MODE=operator-30 \
  RVLLM_DECODE_BATCH_TARGET=30 \
  timeout "${RVLLM_SWARM_TIMEOUT:-240}" target/release/swarm-cli \
    --mode operator-30 --decode-batch 30 --broadcast "$PROMPT" \
    2>&1 | tee "$LOG_DIR/swarm.log"
)

if ! grep -F '"done":30,"failed":0' "$LOG_DIR/swarm.log" >/dev/null; then
  echo "swarm gate failed: expected 30 done, 0 failed" >&2
  exit 1
fi

echo "PASS 4B FP8 gate: ppl=$PPL logs=$LOG_DIR"
