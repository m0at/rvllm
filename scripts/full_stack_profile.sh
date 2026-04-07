#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
N_VALUES="${N_VALUES:-1,32,64,96,128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
PROFILE_OUTPUT_LEN="${PROFILE_OUTPUT_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-5}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/full_stack_profile/$(date +%Y%m%d_%H%M%S)}"
NCU_SET="${NCU_SET:-full}"
NCU_BIN="${NCU_BIN:-}"
PY_SPY_BIN="${PY_SPY_BIN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --n) N_VALUES="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM_UTIL="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --ncu-set) NCU_SET="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$NCU_BIN" ]] && command -v ncu >/dev/null 2>&1; then
    NCU_BIN="$(command -v ncu)"
fi

if [[ -z "$PY_SPY_BIN" ]] && command -v py-spy >/dev/null 2>&1; then
    PY_SPY_BIN="$(command -v py-spy)"
fi

if [[ -z "$NCU_BIN" ]]; then
    echo "ncu not found" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"/{benchmarks,profiles,rendered,ncu}

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

run_ncu_rvllm() {
    local n="$1"
    local base="$OUT_DIR/ncu/rvllm_n${n}"
    log "rvllm ncu n=${n}"
    "$NCU_BIN" --set "$NCU_SET" --target-processes all --force-overwrite --export "$base" \
        ./target/release/rvllm benchmark \
            --model "$MODEL" \
            --n "$n" \
            --output-len "$PROFILE_OUTPUT_LEN" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --json > "${base}.stdout" 2> "${base}.stderr" || true
}

run_ncu_vllm() {
    local n="$1"
    local base="$OUT_DIR/ncu/vllm_n${n}"
    log "vllm ncu n=${n}"
    "$NCU_BIN" --set "$NCU_SET" --target-processes all --force-overwrite --export "$base" \
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
            --output "$OUT_DIR/benchmarks/vllm_ncu_n${n}.json" \
            > "${base}.stdout" 2> "${base}.stderr" || true
}

run_vllm_flamegraph() {
    if [[ -z "$PY_SPY_BIN" ]]; then
        return
    fi
    log "vllm flamegraph n=64"
    "$PY_SPY_BIN" record -o "$OUT_DIR/rendered/vllm_n64_cpu.svg" -- \
        python3 deploy/vllm_direct_bench.py \
            --model "$MODEL" \
            --max-tokens "$PROFILE_OUTPUT_LEN" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --concurrency 64 \
            --temperature "$TEMPERATURE" \
            --warmup-concurrency 64 \
            --warmup-max-tokens "$WARMUP_MAX_TOKENS" \
            --ignore-eos \
            --output "$OUT_DIR/benchmarks/vllm_flame_n64.json" \
            > "$OUT_DIR/rendered/vllm_n64_cpu.stdout" 2> "$OUT_DIR/rendered/vllm_n64_cpu.stderr" || true
}

log "profile_compare"
bash scripts/profile_compare.sh \
    --model "$MODEL" \
    --n "$N_VALUES" \
    --output-len "$OUTPUT_LEN" \
    --profile-ns "$N_VALUES" \
    --profile-output-len "$PROFILE_OUTPUT_LEN" \
    --gpu-mem "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --warmup-max-tokens "$WARMUP_MAX_TOKENS" \
    --out-dir "$OUT_DIR"

IFS=',' read -r -a ns <<< "$N_VALUES"
for n in "${ns[@]}"; do
    run_ncu_rvllm "$n"
    run_ncu_vllm "$n"
done

run_vllm_flamegraph

log "artifacts saved to $OUT_DIR"
