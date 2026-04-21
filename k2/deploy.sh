#!/bin/bash
set -euo pipefail

# Deploy Kimi K2.6 assets for the KTransformers + SGLang path.
# Usage: bash k2/deploy.sh <ssh_addr> <ssh_port>

SSH_ADDR="${1:-ssh4.vast.ai}"
SSH_PORT="${2:-38164}"
SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_ADDR"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.6}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Kimi-K2.6}"

GGUF_REPO="${GGUF_REPO:-ubergarm/Kimi-K2.6-GGUF}"
GGUF_SUBDIR="${GGUF_SUBDIR:-Q4_X}"
GGUF_DIR="${GGUF_DIR:-/workspace/models/Kimi-K2.6-GGUF-Q4_X}"

MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-1200}"
MIN_AVAILABLE_RAM_GB="${MIN_AVAILABLE_RAM_GB:-900}"

MODEL_LOG="${MODEL_LOG:-/workspace/k2_model_download.log}"
GGUF_LOG="${GGUF_LOG:-/workspace/k2_gguf_download.log}"

echo "=== Phase 1: Preflight ==="
$SSH "bash -lc '
set -euo pipefail

workspace_path=/workspace
if ! test -d \"\$workspace_path\"; then
  echo \"ERROR: /workspace is missing on remote host\"
  exit 1
fi

free_disk_gb=\$(df --output=avail -BG \"\$workspace_path\" | tail -1 | tr -dc \"0-9\")
avail_ram_gb=\$(free -g | awk '\''/Mem:/ {print \$7}'\'')

echo \"workspace_free_disk_gb=\$free_disk_gb\"
echo \"available_ram_gb=\$avail_ram_gb\"

if [ \"\$free_disk_gb\" -lt \"$MIN_FREE_DISK_GB\" ]; then
  echo \"ERROR: need at least $MIN_FREE_DISK_GB GB free on /workspace for K2.6 HF + GGUF assets\"
  exit 1
fi

if [ \"\$avail_ram_gb\" -lt \"$MIN_AVAILABLE_RAM_GB\" ]; then
  echo \"ERROR: need at least $MIN_AVAILABLE_RAM_GB GB available RAM for the intended K2.6 offload setup\"
  exit 1
fi
'"

echo "=== Phase 2: Install dependencies ==="
$SSH "pip install -q safetensors transformers 'huggingface_hub[cli]' tokenizers sentencepiece"

echo "=== Phase 3: Upload helper code ==="
$SCP k2/infer.py k2/model.py k2/loader.py root@$SSH_ADDR:/workspace/

echo "=== Phase 4: Start model downloads (background) ==="
$SSH "bash -lc '
set -euo pipefail

mkdir -p \"$MODEL_DIR\" \"$GGUF_DIR\"

nohup huggingface-cli download \"$MODEL_REPO\" \
  --local-dir \"$MODEL_DIR\" \
  --local-dir-use-symlinks False \
  > \"$MODEL_LOG\" 2>&1 &
echo \$! > /workspace/k2_model_download.pid

nohup huggingface-cli download \"$GGUF_REPO\" \
  --include \"$GGUF_SUBDIR/*\" \
  --local-dir \"$GGUF_DIR\" \
  --local-dir-use-symlinks False \
  > \"$GGUF_LOG\" 2>&1 &
echo \$! > /workspace/k2_gguf_download.pid

echo model_pid=\$(cat /workspace/k2_model_download.pid)
echo gguf_pid=\$(cat /workspace/k2_gguf_download.pid)
echo model_log=\"$MODEL_LOG\"
echo gguf_log=\"$GGUF_LOG\"
'"

echo "=== Done. SSH in to monitor: ==="
echo "ssh -p $SSH_PORT root@$SSH_ADDR"
echo "tail -f $MODEL_LOG"
echo "tail -f $GGUF_LOG"
