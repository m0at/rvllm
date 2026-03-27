#!/bin/bash
set -euo pipefail

# Provision A100 80GB on vast.ai for rvllm benchmarks
# Requires: vastai CLI (pip install vastai)
# Usage: ./vastai-provision.sh [--dry-run]

DISK_GB=${DISK_GB:-100}
GPU_TYPE=${GPU_TYPE:-"A100_SXM4"}  # A100 80GB
GPU_RAM_MIN=${GPU_RAM_MIN:-75}     # At least 75GB VRAM (filters to 80GB)
IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"

echo "Searching for $GPU_TYPE instances (>=${GPU_RAM_MIN}GB VRAM)..."

# Search for suitable instances
vastai search offers \
    "gpu_name=$GPU_TYPE gpu_ram>=${GPU_RAM_MIN} num_gpus=1 inet_down>200 disk_space>=${DISK_GB}" \
    --order "dph_total" \
    --limit 5

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Dry run, not provisioning."
    exit 0
fi

echo ""
read -p "Enter offer ID to provision: " OFFER_ID

# Create instance
INSTANCE_ID=$(vastai create instance $OFFER_ID \
    --image "$IMAGE" \
    --disk $DISK_GB \
    --ssh \
    --env "RUST_LOG=info" \
    --onstart-cmd "apt-get update && apt-get install -y curl git build-essential pkg-config libssl-dev && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" \
    --raw | jq -r '.new_contract')

echo "Instance created: $INSTANCE_ID"
echo "Waiting for instance to start..."

# Wait for SSH to be available
for i in $(seq 1 60); do
    STATUS=$(vastai show instance $INSTANCE_ID --raw | jq -r '.actual_status')
    if [[ "$STATUS" == "running" ]]; then
        echo "Instance running!"
        break
    fi
    echo "  Status: $STATUS (attempt $i/60)"
    sleep 10
done

# Get SSH info
SSH_INFO=$(vastai ssh-url $INSTANCE_ID)
echo ""
echo "Instance ready!"
echo "SSH: $SSH_INFO"
echo ""
echo "Next steps:"
echo "  1. ./deploy/vastai-deploy.sh $INSTANCE_ID"
echo "  2. ./deploy/vastai-benchmark.sh $INSTANCE_ID"
echo ""
echo "$INSTANCE_ID" > deploy/.instance_id
