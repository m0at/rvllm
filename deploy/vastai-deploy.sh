#!/bin/bash
set -euo pipefail

INSTANCE_ID=${1:-$(cat deploy/.instance_id 2>/dev/null)}
if [[ -z "$INSTANCE_ID" ]]; then
    echo "Usage: $0 <instance_id>"
    exit 1
fi

SSH_CMD="vastai ssh $INSTANCE_ID"
SCP_CMD="vastai scp $INSTANCE_ID"

echo "Deploying rvllm to instance $INSTANCE_ID..."

# Upload source code (excluding target/)
echo "Uploading source..."
tar czf /tmp/vllm-rs.tar.gz \
    --exclude='target' --exclude='.git' --exclude='.claude' \
    -C "$(dirname "$(dirname "$0")")" vllm-rs/

$SCP_CMD /tmp/vllm-rs.tar.gz /root/vllm-rs.tar.gz

# Build on the instance
echo "Building on GPU instance..."
$SSH_CMD << 'REMOTE_SCRIPT'
set -euo pipefail
export PATH="/root/.cargo/bin:$PATH"

# Extract
cd /root
tar xzf vllm-rs.tar.gz
cd vllm-rs

# Ensure Rust is installed
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="/root/.cargo/bin:$PATH"
fi

# Build with CUDA
echo "Building rvllm with CUDA support..."
cargo build --release --features cuda -p rvllm-server 2>&1 | tail -10

# Compile CUDA kernels
echo "Compiling CUDA kernels..."
cd kernels && bash build.sh && cd ..

# Install Python vLLM for comparison
echo "Installing Python vLLM..."
pip3 install vllm torch --quiet

# Download model
echo "Downloading model (Llama-3.2-1B)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.2-1B')"

echo "Deployment complete!"
echo "Rust binary: /root/vllm-rs/target/release/rvllm"
REMOTE_SCRIPT

echo "Deploy complete!"
