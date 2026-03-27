#!/bin/bash
# Comprehensive rvLLM vs Python vLLM benchmark
# Runs on A100 (local or via SSH). Measures startup, memory, and throughput
# across multiple concurrency levels and output lengths.
#
# Usage:
#   LOCAL:  bash scripts/final_benchmark.sh
#   SSH:    SSH_HOST=user@gpu-box bash scripts/final_benchmark.sh
#
# Outputs:
#   results/final_benchmark.json   -- machine-readable results
#   results/final_benchmark.txt    -- human-readable comparison table

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
RVLLM_BIN="${RVLLM_BIN:-$ROOT_DIR/target/release/rvLLM}"
RVLLM_PORT=8000
PYTHON_PORT=8001
NUM_PROMPTS="${NUM_PROMPTS:-100}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 4 8 16 32}"
MAX_TOKENS_LIST="${MAX_TOKENS_LIST:-32 128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"
HEALTH_TIMEOUT=180  # seconds to wait for server health

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_JSON="$RESULTS_DIR/final_benchmark.json"
RESULT_TXT="$RESULTS_DIR/final_benchmark.txt"

# --- Helper functions ---

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_health() {
    local port=$1
    local label=$2
    local elapsed=0
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "ERROR: $label did not become healthy within ${HEALTH_TIMEOUT}s"
    return 1
}

get_pid_rss_mb() {
    local pid=$1
    if [ -f "/proc/$pid/status" ]; then
        local rss_kb
        rss_kb=$(grep VmRSS "/proc/$pid/status" 2>/dev/null | awk '{print $2}')
        echo "scale=1; ${rss_kb:-0} / 1024" | bc
    else
        # macOS fallback
        local rss_pages
        rss_pages=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        echo "scale=1; ${rss_pages:-0} / 1024" | bc
    fi
}

get_pid_gpu_mb() {
    local pid=$1
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | grep "^${pid}" | awk -F',' '{sum+=$2} END {printf "%.0f", sum+0}'
}

get_gpu_total_mb() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' '
}

get_gpu_name() {
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
}

kill_server() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    sleep 3
}

run_load_test() {
    local url=$1
    local concurrency=$2
    local max_tokens=$3
    local output=$4

    python3 "$ROOT_DIR/deploy/benchmark_client.py" \
        --url "$url" \
        --num-prompts "$NUM_PROMPTS" \
        --concurrent "$concurrency" \
        --max-tokens "$max_tokens" \
        --output "$output"
}

# --- System info ---

log "=========================================="
log "  rvLLM vs Python vLLM Final Benchmark"
log "=========================================="
log ""
log "Model:        $MODEL"
log "Prompts:      $NUM_PROMPTS per test"
log "Concurrency:  $CONCURRENCY_LEVELS"
log "Max tokens:   $MAX_TOKENS_LIST"
log "Timestamp:    $TIMESTAMP"

GPU_NAME=$(get_gpu_name 2>/dev/null || echo "unknown")
GPU_TOTAL_MB=$(get_gpu_total_mb 2>/dev/null || echo "0")
log "GPU:          $GPU_NAME (${GPU_TOTAL_MB} MB)"
log ""

# Initialize result JSON
python3 -c "
import json, sys
data = {
    'timestamp': '$TIMESTAMP',
    'model': '$MODEL',
    'gpu': '$GPU_NAME',
    'gpu_total_mb': int('${GPU_TOTAL_MB}' or 0),
    'num_prompts': $NUM_PROMPTS,
    'warmup_requests': $WARMUP_REQUESTS,
    'concurrency_levels': [int(x) for x in '$CONCURRENCY_LEVELS'.split()],
    'max_tokens_list': [int(x) for x in '$MAX_TOKENS_LIST'.split()],
    'rust': {},
    'python': {}
}
with open('$RESULT_JSON', 'w') as f:
    json.dump(data, f, indent=2)
"

# =============================================
#  BENCHMARK: Rust rvLLM
# =============================================

log "=========================================="
log "  Starting Rust rvLLM"
log "=========================================="

RUST_START_NS=$(date +%s%N)

"$RVLLM_BIN" serve \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$RVLLM_PORT" &
RUST_PID=$!

if ! wait_for_health "$RVLLM_PORT" "rvLLM"; then
    kill_server "$RUST_PID"
    exit 1
fi

RUST_READY_NS=$(date +%s%N)
RUST_STARTUP_MS=$(( (RUST_READY_NS - RUST_START_NS) / 1000000 ))
log "rvLLM ready in ${RUST_STARTUP_MS}ms (PID $RUST_PID)"

# Let model settle, then capture memory
sleep 3
RUST_RSS_MB=$(get_pid_rss_mb "$RUST_PID")
RUST_GPU_MB=$(get_pid_gpu_mb "$RUST_PID")
log "  CPU RSS: ${RUST_RSS_MB} MB"
log "  GPU VRAM: ${RUST_GPU_MB} MB"

# Warmup
log "  Warming up ($WARMUP_REQUESTS requests)..."
python3 "$ROOT_DIR/deploy/benchmark_client.py" \
    --url "http://localhost:$RVLLM_PORT" \
    --num-prompts "$WARMUP_REQUESTS" \
    --concurrent 1 \
    --max-tokens 16 \
    --output /dev/null 2>/dev/null || true

# Run load tests across all concurrency x max_tokens combinations
log ""
log "Running rvLLM load tests..."
for max_tokens in $MAX_TOKENS_LIST; do
    for conc in $CONCURRENCY_LEVELS; do
        label="c${conc}_t${max_tokens}"
        outfile="$RESULTS_DIR/rust_${label}.json"
        log "  concurrency=$conc max_tokens=$max_tokens"
        run_load_test "http://localhost:$RVLLM_PORT" "$conc" "$max_tokens" "$outfile"
    done
done

# Capture post-benchmark memory (may differ after KV cache allocation)
RUST_RSS_MB_POST=$(get_pid_rss_mb "$RUST_PID")
RUST_GPU_MB_POST=$(get_pid_gpu_mb "$RUST_PID")

kill_server "$RUST_PID"
log "rvLLM stopped."
log ""

# Store rust metadata
python3 -c "
import json, glob, os

with open('$RESULT_JSON') as f:
    data = json.load(f)

data['rust']['startup_ms'] = $RUST_STARTUP_MS
data['rust']['cpu_rss_mb'] = float('$RUST_RSS_MB' or 0)
data['rust']['gpu_vram_mb'] = float('${RUST_GPU_MB:-0}' or 0)
data['rust']['cpu_rss_mb_post'] = float('$RUST_RSS_MB_POST' or 0)
data['rust']['gpu_vram_mb_post'] = float('${RUST_GPU_MB_POST:-0}' or 0)

# Collect load test results
data['rust']['load_tests'] = {}
for max_tokens in '$MAX_TOKENS_LIST'.split():
    for conc in '$CONCURRENCY_LEVELS'.split():
        label = f'c{conc}_t{max_tokens}'
        path = f'$RESULTS_DIR/rust_{label}.json'
        if os.path.exists(path):
            with open(path) as f:
                data['rust']['load_tests'][label] = json.load(f)

with open('$RESULT_JSON', 'w') as f:
    json.dump(data, f, indent=2)
"

# =============================================
#  BENCHMARK: Python vLLM
# =============================================

log "=========================================="
log "  Starting Python vLLM"
log "=========================================="

PYTHON_START_NS=$(date +%s%N)

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PYTHON_PORT" &
PYTHON_PID=$!

if ! wait_for_health "$PYTHON_PORT" "Python vLLM"; then
    kill_server "$PYTHON_PID"
    exit 1
fi

PYTHON_READY_NS=$(date +%s%N)
PYTHON_STARTUP_MS=$(( (PYTHON_READY_NS - PYTHON_START_NS) / 1000000 ))
log "Python vLLM ready in ${PYTHON_STARTUP_MS}ms (PID $PYTHON_PID)"

sleep 3
PYTHON_RSS_MB=$(get_pid_rss_mb "$PYTHON_PID")
PYTHON_GPU_MB=$(get_pid_gpu_mb "$PYTHON_PID")
log "  CPU RSS: ${PYTHON_RSS_MB} MB"
log "  GPU VRAM: ${PYTHON_GPU_MB} MB"

# Warmup
log "  Warming up ($WARMUP_REQUESTS requests)..."
python3 "$ROOT_DIR/deploy/benchmark_client.py" \
    --url "http://localhost:$PYTHON_PORT" \
    --num-prompts "$WARMUP_REQUESTS" \
    --concurrent 1 \
    --max-tokens 16 \
    --output /dev/null 2>/dev/null || true

# Run load tests
log ""
log "Running Python vLLM load tests..."
for max_tokens in $MAX_TOKENS_LIST; do
    for conc in $CONCURRENCY_LEVELS; do
        label="c${conc}_t${max_tokens}"
        outfile="$RESULTS_DIR/python_${label}.json"
        log "  concurrency=$conc max_tokens=$max_tokens"
        run_load_test "http://localhost:$PYTHON_PORT" "$conc" "$max_tokens" "$outfile"
    done
done

PYTHON_RSS_MB_POST=$(get_pid_rss_mb "$PYTHON_PID")
PYTHON_GPU_MB_POST=$(get_pid_gpu_mb "$PYTHON_PID")

kill_server "$PYTHON_PID"
log "Python vLLM stopped."
log ""

# Store python metadata
python3 -c "
import json, os

with open('$RESULT_JSON') as f:
    data = json.load(f)

data['python']['startup_ms'] = $PYTHON_STARTUP_MS
data['python']['cpu_rss_mb'] = float('$PYTHON_RSS_MB' or 0)
data['python']['gpu_vram_mb'] = float('${PYTHON_GPU_MB:-0}' or 0)
data['python']['cpu_rss_mb_post'] = float('$PYTHON_RSS_MB_POST' or 0)
data['python']['gpu_vram_mb_post'] = float('${PYTHON_GPU_MB_POST:-0}' or 0)

data['python']['load_tests'] = {}
for max_tokens in '$MAX_TOKENS_LIST'.split():
    for conc in '$CONCURRENCY_LEVELS'.split():
        label = f'c{conc}_t{max_tokens}'
        path = f'$RESULTS_DIR/python_{label}.json'
        if os.path.exists(path):
            with open(path) as f:
                data['python']['load_tests'][label] = json.load(f)

with open('$RESULT_JSON', 'w') as f:
    json.dump(data, f, indent=2)
"

# =============================================
#  Generate comparison report
# =============================================

log "=========================================="
log "  Generating comparison report"
log "=========================================="

python3 "$ROOT_DIR/scripts/generate_readme_table.py" \
    --input "$RESULT_JSON" \
    --output-txt "$RESULT_TXT"

cat "$RESULT_TXT"

log ""
log "Results saved:"
log "  JSON: $RESULT_JSON"
log "  Text: $RESULT_TXT"
log ""
log "Generate README table:"
log "  python3 scripts/generate_readme_table.py --input $RESULT_JSON"
