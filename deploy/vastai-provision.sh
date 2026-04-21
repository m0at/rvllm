#!/bin/bash
set -euo pipefail

# Provision a GPU instance on vast.ai for rvllm benchmarks.
# Requires: vastai CLI (pip install vastai)
#
# Usage:
#   ./vastai-provision.sh              # provision rvllm instance
#   ./vastai-provision.sh --dry-run    # search only

DISK_GB=${DISK_GB:-100}
GPU_TYPE=${GPU_TYPE:-"H100_SXM"}
GPU_RAM_MIN=${GPU_RAM_MIN:-75}
CPU_RAM_MIN=${CPU_RAM_MIN:-0}

RVLLM_IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"

MODE="${1:---rvllm}"

SEARCH_FILTER="gpu_name=$GPU_TYPE gpu_ram>=${GPU_RAM_MIN} num_gpus=1 inet_down>200 disk_space>=${DISK_GB}"
if [[ "$CPU_RAM_MIN" != "0" ]]; then
    SEARCH_FILTER="$SEARCH_FILTER cpu_ram>=${CPU_RAM_MIN}"
fi

if [[ "$MODE" == "--dry-run" ]]; then
    echo "Searching for $GPU_TYPE instances (>=${GPU_RAM_MIN}GB VRAM, >=${DISK_GB}GB disk, >=${CPU_RAM_MIN}GB host RAM)..."
    vastai search offers \
        "$SEARCH_FILTER" \
        --order "dph_total" \
        --limit 5
    exit 0
fi

echo ""
echo "=== Provisioning rvllm instance ==="
echo "Image: $RVLLM_IMAGE"

vastai search offers \
    "$SEARCH_FILTER" \
    --order "dph_total" \
    --limit 5

echo ""
read -p "Enter offer ID: " OFFER_ID

ONSTART="apt-get update && apt-get install -y curl git build-essential pkg-config libssl-dev && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"

INSTANCE_ID=$(vastai create instance $OFFER_ID \
    --image "$RVLLM_IMAGE" \
    --disk $DISK_GB \
    --ssh \
    --env "RUST_LOG=info" \
    --onstart-cmd "$ONSTART" \
    --raw | jq -r '.new_contract')

echo "Instance created: $INSTANCE_ID"
echo "Waiting for instance to start..."

for i in $(seq 1 60); do
    STATUS=$(vastai show instance $INSTANCE_ID --raw | jq -r '.actual_status')
    if [[ "$STATUS" == "running" ]]; then
        echo "Instance running!"
        break
    fi
    echo "  Status: $STATUS (attempt $i/60)"
    sleep 10
done

SSH_INFO=$(vastai ssh-url $INSTANCE_ID)
echo "rvllm ready: $SSH_INFO"
echo "$INSTANCE_ID" > "deploy/.instance_id_rvllm"

echo ""
echo "=== Next steps ==="
echo "  1. Deploy rvllm:  ./deploy/vastai-deploy.sh"
echo "  2. Run benchmark: ./deploy/vastai-benchmark.sh"
