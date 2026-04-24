#!/usr/bin/env bash
# Deploy MiniMax-M2.7-NVFP4 on TPU v6e-8.
# One-shot: create VM, upload SHA-pinned tarball, build zig, download model,
# smoke-run the JAX inference harness.
#
# Usage:
#   ./deploy_m2_tpu.sh [--dry-run]
#
# Environment overrides:
#   PROJECT      (default: `gcloud config get-value project`)
#   ZONE         (default: us-east5-b)
#   TPU_NAME     (default: rvllm-m2)
#   ACCELERATOR  (default: v6e-8)
#   VERSION      (default: v2-alpha-tpuv6e)
#   DISK         (default: 300)
#   MODEL_REPO   (default: lukealonso/MiniMax-M2.7-NVFP4)
#   HF_TOKEN     (optional; exported on remote for huggingface-cli)
set -euo pipefail

DRY_RUN=0
SKIP_CREATE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-create) SKIP_CREATE=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

run() {
  echo ">> $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$@"
  fi
}

run_heredoc() {
  # run_heredoc <description> <command...> < stdin
  local desc="$1"; shift
  echo ">> $desc"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  else
    # Drain stdin so caller heredoc is consumed.
    cat >/dev/null
  fi
}

need() {
  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: required tool not found: $bin" >&2
    echo "Install it and retry." >&2
    exit 1
  fi
}

echo ">> prereq check: gcloud, hf (huggingface CLI), git"
need gcloud
need git
# huggingface_hub 1.5+ renamed `huggingface-cli` to `hf`. Accept either.
if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  echo "ERROR: required tool not found: hf (or legacy huggingface-cli)" >&2
  echo "Install with: python3 -m pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi
echo "   using HF CLI: $HF_CLI"

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT:-}" ]]; then
  echo "ERROR: PROJECT env var is empty and 'gcloud config get-value project' returned nothing." >&2
  exit 1
fi
ZONE="${ZONE:-us-east5-b}"
TPU_NAME="${TPU_NAME:-rvllm-m2}"
ACCELERATOR="${ACCELERATOR:-v6e-8}"
VERSION="${VERSION:-v2-alpha-tpuv6e}"
DISK="${DISK:-300}"
MODEL_REPO="${MODEL_REPO:-lukealonso/MiniMax-M2.7-NVFP4}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHA="$(cd "$ROOT" && git rev-parse HEAD)"
SHORT_SHA="$(cd "$ROOT" && git rev-parse --short HEAD)"
RUN_DIR="/workspace/runs/${SHA}"
REMOTE_TAR="/tmp/rvllm-${SHORT_SHA}.tar.gz"
LOCAL_TAR="/tmp/rvllm-${SHORT_SHA}.tar.gz"

echo ">> config:"
echo "   PROJECT     = ${PROJECT}"
echo "   ZONE        = ${ZONE}"
echo "   TPU_NAME    = ${TPU_NAME}"
echo "   ACCELERATOR = ${ACCELERATOR}"
echo "   VERSION     = ${VERSION}"
echo "   DISK        = ${DISK} GB"
echo "   MODEL_REPO  = ${MODEL_REPO}"
echo "   SHA         = ${SHA}"
echo "   RUN_DIR     = ${RUN_DIR}"
echo "   DRY_RUN     = ${DRY_RUN}"

# Refuse to deploy from a dirty tree (per user's global deploy rules).
if ! (cd "$ROOT" && git diff --quiet && git diff --cached --quiet); then
  echo "WARNING: local git tree is dirty; the SHA in the tarball will not reflect uncommitted changes."
  echo "         Commit first, or export ALLOW_DIRTY=1 to continue anyway."
  if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
    exit 1
  fi
fi

echo ">> (1) check v6e quota in ${ZONE}"
QUOTA_CHECK_CMD="gcloud compute tpus tpu-vm accelerator-types list --zone='${ZONE}' --project='${PROJECT}' --filter='name~v6e' --format='value(name)' 2>/dev/null | head -n 5 || true"
if [[ "$DRY_RUN" -eq 0 ]]; then
  QUOTA_OUT="$(eval "$QUOTA_CHECK_CMD")"
  if [[ -z "$QUOTA_OUT" ]]; then
    echo "WARNING: no v6e accelerator types visible in ${ZONE} for project ${PROJECT}."
    echo "         You may be missing TPU-v6e quota. Continuing anyway."
  else
    echo "   v6e accelerators available:"
    echo "$QUOTA_OUT" | sed 's/^/     /'
  fi
else
  echo "   [dry-run] $QUOTA_CHECK_CMD"
fi

echo ">> (2) create TPU VM ${TPU_NAME} (${ACCELERATOR}) in ${ZONE}"
CREATE_CMD="gcloud compute tpus tpu-vm create '${TPU_NAME}' \
  --zone='${ZONE}' --project='${PROJECT}' \
  --accelerator-type='${ACCELERATOR}' \
  --version='${VERSION}' \
  --data-disk source=projects/${PROJECT}/zones/${ZONE}/disks/${TPU_NAME}-data,mode=read-write 2>/dev/null \
  || gcloud compute tpus tpu-vm create '${TPU_NAME}' \
       --zone='${ZONE}' --project='${PROJECT}' \
       --accelerator-type='${ACCELERATOR}' \
       --version='${VERSION}'"
# Use simpler create (boot-disk size via --disk-size if supported; else rely on default and resize).
SPOT_FLAG=""
if [[ "${SPOT:-0}" == "1" ]]; then
  SPOT_FLAG=" --spot"
  echo "   (spot/preemptible requested — may be reclaimed by Google at any time)"
fi
CREATE_CMD="gcloud compute tpus tpu-vm create '${TPU_NAME}' \
  --zone='${ZONE}' --project='${PROJECT}' \
  --accelerator-type='${ACCELERATOR}' \
  --version='${VERSION}'${SPOT_FLAG}"
if [[ "$DRY_RUN" -eq 0 ]]; then
  if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
    echo "   TPU VM ${TPU_NAME} already exists; skipping create."
  else
    run "$CREATE_CMD"
  fi
else
  echo "   [dry-run] $CREATE_CMD"
fi

echo ">> (3) wait for READY"
WAIT_CMD="until gcloud compute tpus tpu-vm describe '${TPU_NAME}' --zone='${ZONE}' --project='${PROJECT}' --format='value(state)' 2>/dev/null | grep -q READY; do sleep 10; echo '   ...still waiting'; done"
if [[ "$DRY_RUN" -eq 0 ]]; then
  eval "$WAIT_CMD"
  echo "   TPU VM is READY."
else
  echo "   [dry-run] $WAIT_CMD"
fi

SSH_BASE="gcloud compute tpus tpu-vm ssh '${TPU_NAME}' --zone='${ZONE}' --project='${PROJECT}'"
SCP_BASE="gcloud compute tpus tpu-vm scp --zone='${ZONE}' --project='${PROJECT}'"

echo ">> (4) create run dir ${RUN_DIR} on remote"
MKDIR_CMD="${SSH_BASE} --command=\"sudo mkdir -p '${RUN_DIR}' && sudo chown -R \\\$USER:\\\$USER /workspace && rm -rf '${RUN_DIR}'/* && mkdir -p '${RUN_DIR}'\""
run "$MKDIR_CMD"

echo ">> (5a) build local tarball from git HEAD"
BUILD_TAR_CMD="(cd '${ROOT}' && git archive --format=tar.gz --prefix='rvllm-${SHORT_SHA}/' HEAD -o '${LOCAL_TAR}')"
run "$BUILD_TAR_CMD"

echo ">> (5b) upload tarball to ${TPU_NAME}:${REMOTE_TAR}"
UPLOAD_CMD="${SCP_BASE} '${LOCAL_TAR}' '${TPU_NAME}':'${REMOTE_TAR}'"
run "$UPLOAD_CMD"

echo ">> (5c) unpack tarball into ${RUN_DIR} and write REVISION"
UNPACK_CMD="${SSH_BASE} --command=\"set -euo pipefail; cd '${RUN_DIR}' && tar xzf '${REMOTE_TAR}' --strip-components=1 && echo '${SHA}' > '${RUN_DIR}/REVISION' && ls -la '${RUN_DIR}' | head\""
run "$UNPACK_CMD"

echo ">> (6) install deps on remote (jax[tpu], safetensors, tokenizers, ml_dtypes, zig)"
INSTALL_SCRIPT=$(cat <<'EOS'
set -euo pipefail
echo "host cores: $(nproc)"
python3 --version
pip3 install --quiet --upgrade pip
pip3 install --quiet --upgrade \
  "jax[tpu]" \
  safetensors \
  huggingface_hub \
  tokenizers \
  ml_dtypes \
  numpy
python3 -c "import jax; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devs', len(jax.devices()))"

# Zig toolchain (pinned; matches agent 5's expectations).
ZIG_VER=0.15.1
ZIG_DIR="$HOME/zig-x86_64-linux-${ZIG_VER}"
if [[ ! -x "$ZIG_DIR/zig" ]]; then
  echo "installing zig ${ZIG_VER}"
  curl -fsSL "https://ziglang.org/download/${ZIG_VER}/zig-x86_64-linux-${ZIG_VER}.tar.xz" \
    | tar xJ -C "$HOME/"
fi
export PATH="$ZIG_DIR:$PATH"
zig version
EOS
)
INSTALL_SCRIPT="${INSTALL_SCRIPT//\$/\\\$}"
INSTALL_CMD="${SSH_BASE} --command=\"${INSTALL_SCRIPT//\"/\\\"}\""
run "$INSTALL_CMD"

echo ">> (7) build zig project in ${RUN_DIR}/zig"
BUILD_ZIG_SCRIPT=$(cat <<EOS
set -euo pipefail
export PATH="\$HOME/zig-x86_64-linux-0.15.1:\$PATH"
cd '${RUN_DIR}/zig'
zig build -Doptimize=ReleaseFast
zig build test
ls -la zig-out || true
EOS
)
BUILD_ZIG_SCRIPT="${BUILD_ZIG_SCRIPT//\$/\\\$}"
BUILD_ZIG_CMD="${SSH_BASE} --command=\"${BUILD_ZIG_SCRIPT//\"/\\\"}\""
run "$BUILD_ZIG_CMD"

echo ">> (8) download model ${MODEL_REPO} to /workspace/models/m2-nvfp4"
HF_TOKEN_EXPORT=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_EXPORT="export HF_TOKEN='${HF_TOKEN}'; export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}';"
fi
HF_SCRIPT=$(cat <<EOS
set -euo pipefail
${HF_TOKEN_EXPORT}
mkdir -p /workspace/models
pip3 install --quiet --upgrade 'huggingface_hub[cli]'
# Prefer new hf CLI (huggingface_hub 1.5+), fall back to legacy name.
if command -v hf >/dev/null 2>&1; then
  hf download '${MODEL_REPO}' --local-dir /workspace/models/m2-nvfp4 --max-workers 32
else
  huggingface-cli download '${MODEL_REPO}' --local-dir /workspace/models/m2-nvfp4 --max-workers 32
fi
du -sh /workspace/models/m2-nvfp4
EOS
)
HF_SCRIPT="${HF_SCRIPT//\$/\\\$}"
HF_CMD="${SSH_BASE} --command=\"${HF_SCRIPT//\"/\\\"}\""
run "$HF_CMD"

echo ">> (9) smoke run: m2_tpu_infer --max-tokens 16"
SMOKE_SCRIPT=$(cat <<EOS
set -euo pipefail
export PATH="\$HOME/zig-x86_64-linux-0.15.1:\$PATH"
export RVLLM_ZIG_LIB='${RUN_DIR}/zig/zig-out/lib/librvllm_zig.so'
cd '${RUN_DIR}'
cat REVISION
python3 tpu/harness/m2_tpu_infer.py \
  --model-dir /workspace/models/m2-nvfp4 \
  --max-tokens 16 \
  --prompt 'Hello'
EOS
)
SMOKE_SCRIPT="${SMOKE_SCRIPT//\$/\\\$}"
SMOKE_CMD="${SSH_BASE} --command=\"${SMOKE_SCRIPT//\"/\\\"}\""
run "$SMOKE_CMD"

echo ">> done. Run dir on ${TPU_NAME}: ${RUN_DIR}"
echo ">> to teardown: gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE} --project=${PROJECT}"
