#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
N_VALUES="${N_VALUES:-1,32,64,96,128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
PROFILE_NS="${PROFILE_NS:-1,32,64,96,128}"
PROFILE_OUTPUT_LEN="${PROFILE_OUTPUT_LEN:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-0}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-5}"
RVLLM_FEATURES="${RVLLM_FEATURES:-cuda,cublaslt}"
BUILD_RVLLM="${BUILD_RVLLM:-1}"
RUN_RVLLM="${RUN_RVLLM:-1}"
RUN_VLLM="${RUN_VLLM:-1}"
RUN_NSYS="${RUN_NSYS:-1}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/profile_compare/$(date +%Y%m%d_%H%M%S)}"
NSYS_BIN="${NSYS_BIN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --n) N_VALUES="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --profile-ns) PROFILE_NS="$2"; shift 2 ;;
        --profile-output-len) PROFILE_OUTPUT_LEN="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --warmup-concurrency) WARMUP_CONCURRENCY="$2"; shift 2 ;;
        --warmup-max-tokens) WARMUP_MAX_TOKENS="$2"; shift 2 ;;
        --skip-build-rvllm) BUILD_RVLLM=0; shift ;;
        --skip-rvllm) RUN_RVLLM=0; shift ;;
        --skip-vllm) RUN_VLLM=0; shift ;;
        --skip-nsys) RUN_NSYS=0; shift ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$NSYS_BIN" ]]; then
    if command -v nsys >/dev/null 2>&1; then
        NSYS_BIN="$(command -v nsys)"
    elif [[ -x /opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nsys ]]; then
        NSYS_BIN=/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nsys
    fi
fi

if [[ "$RUN_NSYS" == "1" ]] && [[ -z "$NSYS_BIN" ]]; then
    echo "nsys not found; rerun with NSYS_BIN=/path/to/nsys or --skip-nsys" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"/{benchmarks,profiles,rendered}

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

run_rvllm_bench() {
    local n="$1"
    local out_json="$OUT_DIR/benchmarks/rvllm_n${n}.json"
    log "rvllm benchmark n=${n}"
    ./target/release/rvllm benchmark \
        --model "$MODEL" \
        --n "$n" \
        --output-len "$OUTPUT_LEN" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --json > "$out_json"
}

run_vllm_bench() {
    local n="$1"
    local out_json="$OUT_DIR/benchmarks/vllm_n${n}.json"
    log "vllm benchmark n=${n}"
    python3 deploy/vllm_direct_bench.py \
        --model "$MODEL" \
        --max-tokens "$OUTPUT_LEN" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --concurrency "$n" \
        --temperature "$TEMPERATURE" \
        --warmup-concurrency "$n" \
        --warmup-max-tokens "$WARMUP_MAX_TOKENS" \
        --ignore-eos \
        --output "$out_json" >/dev/null
}

profile_rvllm() {
    local n="$1"
    local base="$OUT_DIR/profiles/rvllm_n${n}"
    log "rvllm nsys n=${n}"
    "$NSYS_BIN" profile --stats=true --force-overwrite=true -o "$base" \
        ./target/release/rvllm benchmark \
        --model "$MODEL" \
        --n "$n" \
        --output-len "$PROFILE_OUTPUT_LEN" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --json > "${base}.stdout" 2> "${base}.stderr" || true
    "$NSYS_BIN" stats --force-export=true -r cuda_gpu_kern_sum "${base}.nsys-rep" \
        > "${base}.kern.txt" 2>/dev/null || true
    "$NSYS_BIN" stats --force-export=true -r cuda_gpu_mem_time_sum "${base}.nsys-rep" \
        > "${base}.mem.txt" 2>/dev/null || true
    "$NSYS_BIN" stats --force-export=true -r cuda_gpu_mem_size_sum "${base}.nsys-rep" \
        > "${base}.mem_size.txt" 2>/dev/null || true
}

profile_vllm() {
    local n="$1"
    local base="$OUT_DIR/profiles/vllm_n${n}"
    local out_json="$OUT_DIR/benchmarks/vllm_profile_n${n}.json"
    log "vllm nsys n=${n}"
    "$NSYS_BIN" profile --stats=true --force-overwrite=true -o "$base" \
        python3 deploy/vllm_direct_bench.py \
            --model "$MODEL" \
            --max-tokens "$PROFILE_OUTPUT_LEN" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --concurrency "$n" \
            --temperature "$TEMPERATURE" \
            --warmup-concurrency "$n" \
            --warmup-max-tokens "$WARMUP_MAX_TOKENS" \
            --ignore-eos \
            --output "$out_json" > "${base}.stdout" 2> "${base}.stderr" || true
    "$NSYS_BIN" stats --force-export=true -r cuda_gpu_kern_sum "${base}.nsys-rep" \
        > "${base}.kern.txt" 2>/dev/null || true
    "$NSYS_BIN" stats --force-export=true -r cuda_gpu_mem_time_sum "${base}.nsys-rep" \
        > "${base}.mem.txt" 2>/dev/null || true
    "$NSYS_BIN" stats --force-export=true -r cuda_gpu_mem_size_sum "${base}.nsys-rep" \
        > "${base}.mem_size.txt" 2>/dev/null || true
}

if [[ "$BUILD_RVLLM" == "1" ]]; then
    log "building rvllm"
    cargo build --release --features "$RVLLM_FEATURES" -p rvllm
fi

if [[ "$RUN_RVLLM" == "1" ]]; then
    IFS=',' read -r -a ns <<< "$N_VALUES"
    for n in "${ns[@]}"; do
        run_rvllm_bench "$n"
    done
fi

if [[ "$RUN_VLLM" == "1" ]]; then
    IFS=',' read -r -a ns <<< "$N_VALUES"
    for n in "${ns[@]}"; do
        run_vllm_bench "$n"
    done
fi

if [[ "$RUN_NSYS" == "1" ]]; then
    IFS=',' read -r -a ns <<< "$PROFILE_NS"
    for n in "${ns[@]}"; do
        if [[ "$RUN_RVLLM" == "1" ]]; then
            profile_rvllm "$n"
        fi
        if [[ "$RUN_VLLM" == "1" ]]; then
            profile_vllm "$n"
        fi
    done
fi

python3 - "$OUT_DIR/manifest.json" "$MODEL" "$N_VALUES" "$OUTPUT_LEN" "$PROFILE_NS" "$PROFILE_OUTPUT_LEN" "$GPU_MEM_UTIL" "$TEMPERATURE" "$MAX_MODEL_LEN" "$WARMUP_CONCURRENCY" "$WARMUP_MAX_TOKENS" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
out_dir = manifest_path.parent

def engine_paths(engine, n_values, profile_ns):
    bench = {}
    for raw in n_values.split(","):
        n = raw.strip()
        if n:
            bench[n] = f"benchmarks/{engine}_n{n}.json"
    profiles = {}
    for raw in profile_ns.split(","):
        n = raw.strip()
        if n:
            base = f"profiles/{engine}_n{n}"
            profiles[n] = {
                "stdout": f"{base}.stdout",
                "stderr": f"{base}.stderr",
                "kern_txt": f"{base}.kern.txt",
                "mem_txt": f"{base}.mem.txt",
                "mem_size_txt": f"{base}.mem_size.txt",
                "rep": f"{base}.nsys-rep",
            }
    return {"benchmark": bench, "profiles": profiles}

manifest = {
    "model": sys.argv[2],
    "n_values": [int(x) for x in sys.argv[3].split(",") if x.strip()],
    "output_len": int(sys.argv[4]),
    "profile_n_values": [int(x) for x in sys.argv[5].split(",") if x.strip()],
    "profile_output_len": int(sys.argv[6]),
    "gpu_memory_utilization": float(sys.argv[7]),
    "temperature": float(sys.argv[8]),
    "max_model_len": int(sys.argv[9]),
    "warmup_concurrency": int(sys.argv[10]),
    "warmup_max_tokens": int(sys.argv[11]),
    "engines": {
        "rvllm": engine_paths("rvllm", sys.argv[3], sys.argv[5]),
        "vllm": engine_paths("vllm", sys.argv[3], sys.argv[5]),
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(manifest_path)
PY

python3 scripts/render_profile_compare.py \
    --manifest "$OUT_DIR/manifest.json" \
    --out-dir "$OUT_DIR/rendered"

log "artifacts saved to $OUT_DIR"
