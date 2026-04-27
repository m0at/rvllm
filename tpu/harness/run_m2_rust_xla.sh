#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

PROJECT="${PROJECT:-finance-484520}"
ZONE="${ZONE:-europe-west4-a}"
TPU_NAME="${TPU_NAME:-rvllm-m2}"
REMOTE_MODEL_DIR="${REMOTE_MODEL_DIR:-/workspace/models/m2/MiniMax-M2.7-NVFP4}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/workspace/models/runs}"
LOCAL_STAGE="${LOCAL_STAGE:-$ROOT/target/rvllm-xla-min}"
REMOTE_DECODE_LAYER_BODY="${REMOTE_DECODE_LAYER_BODY:-}"
REMOTE_DECODE_LAYER_BODY_FORMAT="${REMOTE_DECODE_LAYER_BODY_FORMAT:-lowered}"
MAX_WEIGHT_ARENA_BYTES="${MAX_WEIGHT_ARENA_BYTES:-0}"
REMOTE_M2_MOE_INT8="${RVLLM_M2_MOE_INT8:-}"
REMOTE_M2_PARITY="${RVLLM_M2_PARITY:-}"
REMOTE_M2_BODY_PROBE="${RVLLM_M2_BODY_PROBE:-}"
REMOTE_PPL_TEXT_LOCAL="${REMOTE_PPL_TEXT_LOCAL:-$ROOT/tpu/harness/m2_ppl_corpus.txt}"
REMOTE_PPL_TEXT="${REMOTE_PPL_TEXT:-$REMOTE_RUN_ROOT/m2_ppl_corpus.txt}"

case "$REMOTE_MODEL_DIR" in
  /tmp/*|/dev/shm/*)
    echo "refusing transient model dir: $REMOTE_MODEL_DIR" >&2
    exit 2
    ;;
esac

rm -rf "$LOCAL_STAGE"
mkdir -p "$LOCAL_STAGE/v3/crates"
cp "$ROOT/v3/Cargo.lock" "$ROOT/v3/rust-toolchain.toml" "$LOCAL_STAGE/v3/"
cp -R \
  "$ROOT/v3/crates/rvllm-core" \
  "$ROOT/v3/crates/rvllm-mem" \
  "$ROOT/v3/crates/rvllm-kernels" \
  "$ROOT/v3/crates/rvllm-fused" \
  "$ROOT/v3/crates/rvllm-loader" \
  "$ROOT/v3/crates/rvllm-xla" \
  "$LOCAL_STAGE/v3/crates/"

cat > "$LOCAL_STAGE/v3/Cargo.toml" <<'TOML'
[workspace]
resolver = "2"
members = [
    "crates/rvllm-core",
    "crates/rvllm-mem",
    "crates/rvllm-kernels",
    "crates/rvllm-fused",
    "crates/rvllm-loader",
    "crates/rvllm-xla",
]

[workspace.package]
version = "0.3.0"
edition = "2021"
rust-version = "1.80"
license = "Apache-2.0"
authors = ["m0at <47344131+m0at@users.noreply.github.com>"]
repository = "https://github.com/m0at/rvllm"

[workspace.dependencies]
rvllm-core    = { path = "crates/rvllm-core" }
rvllm-mem     = { path = "crates/rvllm-mem" }
rvllm-kernels = { path = "crates/rvllm-kernels" }
rvllm-fused   = { path = "crates/rvllm-fused" }
rvllm-loader  = { path = "crates/rvllm-loader" }
rvllm-xla     = { path = "crates/rvllm-xla" }

half = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
debug = 1
strip = "none"

[profile.dev]
opt-level = 1
TOML

tar -czf "$LOCAL_STAGE/rvllm-xla-min.tgz" -C "$LOCAL_STAGE" v3

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --ssh-flag='-o ConnectTimeout=10' \
  --command "set -euo pipefail; mkdir -p '$REMOTE_RUN_ROOT/rvllm-xla-min' '$REMOTE_RUN_ROOT/m2_bench_results/rust_xla'; rm -rf '$REMOTE_RUN_ROOT/rvllm-xla-min/v3'"

gcloud compute tpus tpu-vm scp "$LOCAL_STAGE/rvllm-xla-min.tgz" \
  "$TPU_NAME:$REMOTE_RUN_ROOT/rvllm-xla-min/rvllm-xla-min.tgz" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --scp-flag='-o ConnectTimeout=10'

if [[ -f "$REMOTE_PPL_TEXT_LOCAL" ]]; then
  gcloud compute tpus tpu-vm scp "$REMOTE_PPL_TEXT_LOCAL" \
    "$TPU_NAME:$REMOTE_PPL_TEXT" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --scp-flag='-o ConnectTimeout=10'
fi

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --ssh-flag='-o ConnectTimeout=10' \
  --command "set -euo pipefail
if ! command -v cargo >/dev/null 2>&1; then
  sudo apt-get update -y >'$REMOTE_RUN_ROOT/rvllm-xla-min/apt_update.log' 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential pkg-config curl ca-certificates >'$REMOTE_RUN_ROOT/rvllm-xla-min/apt_install.log' 2>&1
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal >'$REMOTE_RUN_ROOT/rvllm-xla-min/rustup.log' 2>&1
fi
. \$HOME/.cargo/env
LIBTPU_SO=\$(find \"\$HOME/.local/lib\" /usr/local/lib /usr/lib -name libtpu.so 2>/dev/null | head -1 || true)
if [[ -n \"\$LIBTPU_SO\" ]]; then
  export LD_LIBRARY_PATH=\"\$(dirname \"\$LIBTPU_SO\"):\${LD_LIBRARY_PATH:-}\"
else
  echo \"missing libtpu.so; install libtpu or expose it via LD_LIBRARY_PATH\" >&2
  exit 3
fi
export RVLLM_M2_MOE_INT8='$REMOTE_M2_MOE_INT8'
export RVLLM_M2_PARITY='$REMOTE_M2_PARITY'
export RVLLM_M2_BODY_PROBE='$REMOTE_M2_BODY_PROBE'
export RAYON_NUM_THREADS=\"\${RAYON_NUM_THREADS:-180}\"
cd '$REMOTE_RUN_ROOT/rvllm-xla-min'
tar -xzf rvllm-xla-min.tgz
cd v3
cargo build --release -p rvllm-xla --features tpu --bins
RUN='$REMOTE_RUN_ROOT/m2_bench_results/rust_xla/run_'\$(date -u +%Y%m%d_%H%M%S)
mkdir -p \"\$RUN/artifacts_int8\"
echo RUN_DIR=\"\$RUN\"
target/release/m2_rust_decode_bench \
  --model-dir '$REMOTE_MODEL_DIR' \
  --artifact-dir \"\$RUN/artifacts_int8\" \
  --out \"\$RUN/decode_plan_int8_b1_8_16_32.json\" \
  --emit-decode-artifacts \
  --batches 1,8,16,32 \
  --ctx 2048 \
  --kv-cache int8 \
  --weight-format int8 \
  --moe-impl auto \
  --prompt 'explain angular momentum' \
  --gen-tokens 64 2>&1 | tee \"\$RUN/decode_plan_int8.log\"
if [[ -n '$REMOTE_DECODE_LAYER_BODY' ]]; then
  target/release/m2_rust_decode_bench \
    --model-dir '$REMOTE_MODEL_DIR' \
    --artifact-dir \"\$RUN/real_execute_b8\" \
    --out \"\$RUN/real_execute_b8.json\" \
    --decode-layer-body '$REMOTE_DECODE_LAYER_BODY' \
    --decode-layer-body-format '$REMOTE_DECODE_LAYER_BODY_FORMAT' \
    --emit-decode-artifacts \
    --batch 8 \
    --ctx 2048 \
    --kv-cache int8 \
    --weight-format int8 \
    --prompt 'explain angular momentum' \
    --gen-tokens 64 2>&1 | tee \"\$RUN/real_execute_b8_emit.log\"
  PPL_FLAG=()
  if [[ -f '$REMOTE_PPL_TEXT' ]]; then
    PPL_FLAG=(--ppl-text '$REMOTE_PPL_TEXT')
  fi
  target/release/m2_rust_decode_bench \
    --model-dir '$REMOTE_MODEL_DIR' \
    --artifact-dir \"\$RUN/real_execute_b8\" \
    --out \"\$RUN/real_execute_b8.json\" \
    --use-existing-artifacts \
    --execute-decode \
    --batch 8 \
    --ctx 2048 \
    --iters 8 \
    --warmup 2 \
    --kv-cache int8 \
    --weight-format int8 \
    --max-weight-arena-bytes '$MAX_WEIGHT_ARENA_BYTES' \
    --prompt 'explain angular momentum' \
    \"\${PPL_FLAG[@]}\" \
    --gen-tokens 64 2>&1 | tee \"\$RUN/real_execute_b8.log\"
else
  echo SKIP_REAL_EXECUTE=no_REMOTE_DECODE_LAYER_BODY
fi
echo DONE_RUN=\"\$RUN\""
