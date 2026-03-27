#!/bin/bash
set -euo pipefail

INSTANCE_ID=${1:-$(cat deploy/.instance_id 2>/dev/null)}
MODEL=${MODEL:-"meta-llama/Llama-3.2-1B"}
NUM_PROMPTS=${NUM_PROMPTS:-200}
CONCURRENT=${CONCURRENT:-16}

echo "Running A100 benchmark comparison on instance $INSTANCE_ID"
echo "Model: $MODEL"
echo "Prompts: $NUM_PROMPTS, Concurrency: $CONCURRENT"

SSH_CMD="vastai ssh $INSTANCE_ID"

$SSH_CMD << REMOTE_SCRIPT
set -euo pipefail
export PATH="/root/.cargo/bin:$PATH"

cd /root

# Helper: capture memory metrics for a running server
capture_memory() {
    local label=\$1
    local pid=\$2
    local output=\$3

    echo "Capturing memory metrics for \$label (PID \$pid)..."

    # CPU RSS (in MB)
    RSS_KB=\$(cat /proc/\$pid/status 2>/dev/null | grep VmRSS | awk '{print \$2}')
    RSS_MB=\$(echo "scale=1; \${RSS_KB:-0} / 1024" | bc)

    # GPU VRAM via nvidia-smi (sum all GPUs for this PID)
    GPU_MB=\$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | grep "^\$pid" | awk -F',' '{sum+=\$2} END {printf "%.1f", sum}')
    GPU_MB=\${GPU_MB:-0}
    GPU_GB=\$(echo "scale=1; \$GPU_MB / 1024" | bc)

    echo "  CPU RSS: \${RSS_MB} MB"
    echo "  GPU VRAM: \${GPU_GB} GB (\${GPU_MB} MB)"

    # Append to JSON
    python3 -c "
import json
with open('\$output') as f:
    data = json.load(f)
data['cpu_rss_mb'] = float('\$RSS_MB')
data['gpu_vram_mb'] = float('\$GPU_MB')
data['gpu_vram_gb'] = float('\$GPU_GB')
with open('\$output', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# ========== Benchmark Python vLLM ==========
echo ""
echo "=========================================="
echo "BENCHMARK: Python vLLM"
echo "=========================================="

# Record startup time
PYTHON_START=\$(date +%s%N)

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --gpu-memory-utilization 0.90 \
    --port 8001 &
PYTHON_PID=\$!

# Wait for server
for i in \$(seq 1 120); do
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then break; fi
    sleep 2
done
PYTHON_READY=\$(date +%s%N)
PYTHON_STARTUP_MS=\$(( (PYTHON_READY - PYTHON_START) / 1000000 ))
echo "Python vLLM server ready (startup: \${PYTHON_STARTUP_MS}ms)"

# Capture memory after model load, before benchmark
sleep 2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

# Run benchmark
python3 /root/vllm-rs/deploy/benchmark_client.py \
    --url http://localhost:8001 \
    --num-prompts $NUM_PROMPTS \
    --concurrent $CONCURRENT \
    --output /root/results_python.json

# Capture memory metrics while server is still loaded
capture_memory "Python vLLM" \$PYTHON_PID /root/results_python.json

# Add startup time
python3 -c "
import json
with open('/root/results_python.json') as f:
    data = json.load(f)
data['startup_ms'] = $PYTHON_STARTUP_MS if '$PYTHON_STARTUP_MS'.isdigit() else \$PYTHON_STARTUP_MS
with open('/root/results_python.json', 'w') as f:
    json.dump(data, f, indent=2)
"

kill \$PYTHON_PID 2>/dev/null || true
wait \$PYTHON_PID 2>/dev/null || true
sleep 5

# ========== Benchmark Rust rvllm ==========
echo ""
echo "=========================================="
echo "BENCHMARK: Rust rvllm"
echo "=========================================="

# Record startup time
RUST_START=\$(date +%s%N)

/root/vllm-rs/target/release/rvllm serve \
    --model $MODEL \
    --gpu-memory-utilization 0.90 \
    --port 8000 &
RUST_PID=\$!

for i in \$(seq 1 120); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then break; fi
    sleep 2
done
RUST_READY=\$(date +%s%N)
RUST_STARTUP_MS=\$(( (RUST_READY - RUST_START) / 1000000 ))
echo "Rust rvllm server ready (startup: \${RUST_STARTUP_MS}ms)"

sleep 2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

python3 /root/vllm-rs/deploy/benchmark_client.py \
    --url http://localhost:8000 \
    --num-prompts $NUM_PROMPTS \
    --concurrent $CONCURRENT \
    --output /root/results_rust.json

# Capture memory metrics while server is still loaded
capture_memory "Rust rvllm" \$RUST_PID /root/results_rust.json

# Add startup time
python3 -c "
import json
with open('/root/results_rust.json') as f:
    data = json.load(f)
data['startup_ms'] = $RUST_STARTUP_MS if '$RUST_STARTUP_MS'.isdigit() else \$RUST_STARTUP_MS
with open('/root/results_rust.json', 'w') as f:
    json.dump(data, f, indent=2)
"

kill \$RUST_PID 2>/dev/null || true
wait \$RUST_PID 2>/dev/null || true

# ========== Generate Comparison Report ==========
echo ""
python3 /root/vllm-rs/deploy/compare_results.py \
    --rust /root/results_rust.json \
    --python /root/results_python.json

# Print memory/startup summary
echo ""
echo "=========================================="
echo "RESOURCE USAGE COMPARISON"
echo "=========================================="
python3 -c "
import json

with open('/root/results_rust.json') as f:
    rust = json.load(f)
with open('/root/results_python.json') as f:
    py = json.load(f)

print(f\"{'Metric':<25s} {'Rust':>12s} {'Python':>12s} {'Ratio':>12s}\")
print(f\"{'-'*25} {'-'*12} {'-'*12} {'-'*12}\")

# Startup
rs = rust.get('startup_ms', 0)
ps = py.get('startup_ms', 0)
ratio = ps / rs if rs > 0 else 0
print(f\"{'Startup':<25s} {rs:>10.0f}ms {ps:>10.0f}ms {ratio:>10.1f}x\")

# CPU RSS
rc = rust.get('cpu_rss_mb', 0)
pc = py.get('cpu_rss_mb', 0)
ratio = pc / rc if rc > 0 else 0
print(f\"{'CPU RSS':<25s} {rc:>10.1f}MB {pc:>10.1f}MB {ratio:>10.1f}x\")

# GPU VRAM
rg = rust.get('gpu_vram_gb', 0)
pg = py.get('gpu_vram_gb', 0)
ratio = pg / rg if rg > 0 else 0
print(f\"{'GPU VRAM':<25s} {rg:>10.1f}GB {pg:>10.1f}GB {ratio:>10.1f}x\")
"

REMOTE_SCRIPT

# Download results
echo "Downloading results..."
vastai scp $INSTANCE_ID /root/results_python.json deploy/results_python.json
vastai scp $INSTANCE_ID /root/results_rust.json deploy/results_rust.json

echo "Results saved to deploy/"
