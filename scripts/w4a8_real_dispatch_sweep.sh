#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
V3_DIR="${RVLLM_V3_DIR:-${ROOT_DIR}/v3}"
TARGET_DIR="${RVLLM_TARGET_DIR:-${V3_DIR}/target/release}"

BENCH_BIN="${RVLLM_BENCH_BIN:-${TARGET_DIR}/rvllm-bench}"
PPL_BIN="${RVLLM_PPL_BIN:-${TARGET_DIR}/rvllm-ppl}"

RUN_ID="${W4A8_SWEEP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${W4A8_SWEEP_LOG_DIR:-${ROOT_DIR}/logs/w4a8-real-dispatch/${RUN_ID}}"
SUMMARY_JSONL="${LOG_DIR}/summary.jsonl"
SUMMARY_TSV="${LOG_DIR}/summary.tsv"
CONTEXT_LOG="${LOG_DIR}/context.txt"

TESTS="${W4A8_SWEEP_TESTS:-ppl_modules,down_bench}"
PPL_MODULES="${W4A8_SWEEP_PPL_MODULES:-qkv,o,gate_up,down}"
PPL_LAYERS="${W4A8_SWEEP_PPL_LAYERS:-1}"
DOWN_LAYERS="${W4A8_SWEEP_DOWN_LAYERS:-1,2,4}"
DOWN_BATCHES="${W4A8_SWEEP_DOWN_BATCHES:-1,128}"

TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
KILL_AFTER="${W4A8_SWEEP_KILL_AFTER:-15s}"
BUILD_TIMEOUT="${W4A8_SWEEP_BUILD_TIMEOUT:-1800}"
PPL_TIMEOUT="${W4A8_SWEEP_PPL_TIMEOUT:-900}"
BENCH_TIMEOUT="${W4A8_SWEEP_BENCH_TIMEOUT:-900}"

ARENA_GB="${W4A8_SWEEP_ARENA_GB:-${RVLLM_ARENA_GB:-74}}"
PPL_CHUNK="${W4A8_SWEEP_PPL_CHUNK:-32}"
PPL_CHUNKS="${W4A8_SWEEP_PPL_CHUNKS:-1}"
PPL_PROMPT="${W4A8_SWEEP_PROMPT:-Angular momentum is conserved when the net external torque on a system is zero. This short text keeps W4A8 real-dispatch PPL probes bounded.}"
BENCH_ITERS="${W4A8_SWEEP_BENCH_ITERS:-4}"
BENCH_WARMUP="${W4A8_SWEEP_BENCH_WARMUP:-1}"
RUST_LOG_VALUE="${RUST_LOG:-info}"

FAILURES=0
W4A8_SO=""
COMMON_ENV_ARGS=()
UNSET_EXPERIMENT_ARGS=()

die() {
    echo "w4a8-real-dispatch: $*" >&2
    exit 1
}

has_test() {
    local name="$1"
    [[ ",${TESTS}," == *",${name},"* ]]
}

json_escape() {
    local s="${1:-}"
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '%s' "$s"
}

tsv_escape() {
    local s="${1:-}"
    s=${s//$'\t'/ }
    s=${s//$'\n'/ }
    s=${s//$'\r'/ }
    printf '%s' "$s"
}

safe_name() {
    local s="${1:-x}"
    s=${s//[^[:alnum:]_.-]/_}
    [[ -n "$s" ]] || s="x"
    printf '%s' "$s"
}

format_cmd() {
    local arg q out=""
    for arg in "$@"; do
        printf -v q '%q' "$arg"
        out+="${q} "
    done
    printf '%s' "${out% }"
}

csv_to_array() {
    local raw="$1"
    local -n out_ref="$2"
    local item
    out_ref=()
    IFS=',' read -r -a out_ref <<< "$raw"
    for item in "${!out_ref[@]}"; do
        out_ref[$item]="$(trim "${out_ref[$item]}")"
    done
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "missing command: ${cmd}"
}

require_file() {
    local label="$1"
    local path="$2"
    [[ -f "$path" ]] || die "${label} not found: ${path}"
}

require_dir() {
    local label="$1"
    local path="$2"
    [[ -d "$path" ]] || die "${label} not found: ${path}"
}

default_w4a8_so() {
    if [[ -n "${RVLLM_W4A8_SO:-}" ]]; then
        printf '%s' "$RVLLM_W4A8_SO"
    elif [[ -f "${RVLLM_KERNELS_DIR:-}/sm_90/libw4a8_gemm.so" ]]; then
        printf '%s' "${RVLLM_KERNELS_DIR}/sm_90/libw4a8_gemm.so"
    elif [[ -f "${RVLLM_KERNELS_DIR:-}/libw4a8_gemm.so" ]]; then
        printf '%s' "${RVLLM_KERNELS_DIR}/libw4a8_gemm.so"
    else
        printf ''
    fi
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" 2>/dev/null | sed 's/[[:space:]].*//'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" 2>/dev/null | sed 's/[[:space:]].*//'
    else
        printf ''
    fi
}

gpu_memory_used_mib() {
    local out line used sum=0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    out="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -n "$out" ]] || return 0
    while IFS= read -r line; do
        used="${line//[^0-9]/}"
        [[ -n "$used" ]] || continue
        sum=$((sum + used))
    done <<< "$out"
    printf '%s' "$sum"
}

json_number_or_null() {
    local value="${1:-}"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        printf '%s' "$value"
    else
        printf 'null'
    fi
}

json_float_or_null() {
    local value="${1:-}"
    if [[ "$value" =~ ^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        printf '%s' "$value"
    else
        printf 'null'
    fi
}

extract_json_metric() {
    local key="$1"
    local log_file="$2"
    grep -Eo "\"${key}\":[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?" "$log_file" 2>/dev/null \
        | tail -n 1 \
        | sed 's/.*://' || true
}

extract_ppl_metric() {
    local log_file="$1"
    local value
    value="$(extract_json_metric perplexity "$log_file")"
    if [[ -z "$value" ]]; then
        value="$(sed -nE 's/.*perplexity = ([-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?).*/\1/p' "$log_file" | tail -n 1)"
    fi
    printf '%s' "$value"
}

extract_tok_s_metric() {
    local log_file="$1"
    local value
    value="$(extract_json_metric tok_per_sec "$log_file")"
    if [[ -z "$value" ]]; then
        value="$(sed -nE 's/.*-> ([-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?) tok\/s.*/\1/p' "$log_file" | tail -n 1)"
    fi
    if [[ -z "$value" ]]; then
        value="$(sed -nE 's/.*\(([-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?) tok\/s,.*/\1/p' "$log_file" | tail -n 1)"
    fi
    printf '%s' "$value"
}

extract_ms_step_metric() {
    local log_file="$1"
    local value
    value="$(extract_json_metric ms_per_step "$log_file")"
    if [[ -z "$value" ]]; then
        value="$(sed -nE 's/.*tok\/s \(([-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?) ms\/step\).*/\1/p' "$log_file" | tail -n 1)"
    fi
    printf '%s' "$value"
}

log_path() {
    local lane="$1"
    printf '%s/%s.log' "$LOG_DIR" "$(safe_name "$lane")"
}

append_summary() {
    local lane="$1"
    local kind="$2"
    local module="$3"
    local layers="$4"
    local batch="$5"
    local status="$6"
    local exit_code="$7"
    local elapsed_s="$8"
    local mem_before="$9"
    local mem_after="${10}"
    local log_file="${11}"
    local message="${12}"
    local ppl="${13:-}"
    local tok_s="${14:-}"
    local ms_step="${15:-}"

    printf '{"run_id":"%s","kind":"%s","module":"%s","layers":%s,"batch":%s,"status":"%s","exit_code":%s,"elapsed_s":%s,"mem_before_mib":%s,"mem_after_mib":%s,"ppl":%s,"tok_s":%s,"ms_step":%s,"log":"%s","message":"%s"}\n' \
        "$(json_escape "$RUN_ID")" \
        "$(json_escape "$kind")" \
        "$(json_escape "$module")" \
        "$(json_number_or_null "$layers")" \
        "$(json_number_or_null "$batch")" \
        "$(json_escape "$status")" \
        "$exit_code" \
        "$elapsed_s" \
        "$(json_number_or_null "$mem_before")" \
        "$(json_number_or_null "$mem_after")" \
        "$(json_float_or_null "$ppl")" \
        "$(json_float_or_null "$tok_s")" \
        "$(json_float_or_null "$ms_step")" \
        "$(json_escape "$log_file")" \
        "$(json_escape "$message")" >> "$SUMMARY_JSONL"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(tsv_escape "$RUN_ID")" \
        "$(tsv_escape "$kind")" \
        "$(tsv_escape "$module")" \
        "$layers" \
        "$batch" \
        "$(tsv_escape "$status")" \
        "$exit_code" \
        "$elapsed_s" \
        "$(tsv_escape "$mem_before")" \
        "$(tsv_escape "$mem_after")" \
        "$(tsv_escape "$ppl")" \
        "$(tsv_escape "$tok_s")" \
        "$(tsv_escape "$ms_step")" \
        "$(tsv_escape "$log_file")" \
        "$(tsv_escape "$message")" >> "$SUMMARY_TSV"

    case "$status" in
        fail|timeout)
            FAILURES=$((FAILURES + 1))
            ;;
    esac
}

run_logged() {
    local lane="$1"
    local kind="$2"
    local module="$3"
    local layers="$4"
    local batch="$5"
    local timeout_s="$6"
    local log_file="$7"
    shift 7

    local start elapsed exit_code status message cmd_text mem_before mem_after ppl tok_s ms_step
    cmd_text="$(format_cmd "$@")"

    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'kind=%s\n' "$kind"
        printf 'module=%s\n' "$module"
        printf 'layers=%s\n' "$layers"
        printf 'batch=%s\n' "$batch"
        printf 'timeout_s=%s\n' "$timeout_s"
        printf 'command=%s\n\n' "$cmd_text"
    } > "$log_file"

    echo "[w4a8] ${lane} timeout=${timeout_s}s"
    mem_before="$(gpu_memory_used_mib || true)"
    start=$SECONDS
    set +e
    "$TIMEOUT_BIN" --kill-after="$KILL_AFTER" "${timeout_s}s" "$@" >> "$log_file" 2>&1
    exit_code=$?
    set -e
    elapsed=$((SECONDS - start))
    mem_after="$(gpu_memory_used_mib || true)"

    status="fail"
    message="exit ${exit_code}"
    if [[ "$exit_code" -eq 0 ]]; then
        status="pass"
        message="ok"
    elif [[ "$exit_code" -eq 124 || "$exit_code" -eq 137 ]]; then
        status="timeout"
        message="timed out"
    fi

    ppl="$(extract_ppl_metric "$log_file")"
    tok_s="$(extract_tok_s_metric "$log_file")"
    ms_step="$(extract_ms_step_metric "$log_file")"
    if [[ -n "$ppl" || -n "$tok_s" || -n "$ms_step" ]]; then
        message="${message} ppl=${ppl:-na} tok_s=${tok_s:-na} ms_step=${ms_step:-na}"
    fi

    append_summary "$lane" "$kind" "$module" "$layers" "$batch" "$status" "$exit_code" "$elapsed" "$mem_before" "$mem_after" "$log_file" "$message" "$ppl" "$tok_s" "$ms_step"
}

validate_int_list() {
    local label="$1"
    local raw="$2"
    local values=() value
    csv_to_array "$raw" values
    [[ "${#values[@]}" -gt 0 ]] || die "${label} is empty"
    for value in "${values[@]}"; do
        [[ "$value" =~ ^[0-9]+$ ]] || die "${label} contains non-integer value: ${value}"
        [[ "$value" -gt 0 ]] || die "${label} values must be > 0"
    done
}

validate_module_list() {
    local raw="$1"
    local modules=() module
    csv_to_array "$raw" modules
    [[ "${#modules[@]}" -gt 0 ]] || die "W4A8_SWEEP_PPL_MODULES is empty"
    for module in "${modules[@]}"; do
        case "$module" in
            qkv|o|o_proj|gate_up|mlp_in|down|down_proj|mlp_out|all)
                ;;
            *)
                die "unsupported W4A8 module: ${module} (expected qkv,o,gate_up,down,all)"
                ;;
        esac
    done
}

validate_test_list() {
    local tests=() test
    csv_to_array "$TESTS" tests
    [[ "${#tests[@]}" -gt 0 ]] || die "W4A8_SWEEP_TESTS is empty"
    for test in "${tests[@]}"; do
        case "$test" in
            ppl_modules|down_bench)
                ;;
            *)
                die "unsupported W4A8_SWEEP_TESTS entry: ${test} (expected ppl_modules or down_bench)"
                ;;
        esac
    done
}

init_logs() {
    mkdir -p "$LOG_DIR"
    : > "$SUMMARY_JSONL"
    printf 'run_id\tkind\tmodule\tlayers\tbatch\tstatus\texit_code\telapsed_s\tmem_before_mib\tmem_after_mib\tppl\ttok_s\tms_step\tlog\tmessage\n' > "$SUMMARY_TSV"
}

write_context() {
    local manifest=""
    if [[ -f "${RVLLM_KERNELS_DIR:-}/sm_90/manifest.json" ]]; then
        manifest="${RVLLM_KERNELS_DIR}/sm_90/manifest.json"
    elif [[ -f "${RVLLM_KERNELS_DIR:-}/manifest.json" ]]; then
        manifest="${RVLLM_KERNELS_DIR}/manifest.json"
    fi

    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'repo=%s\n' "$ROOT_DIR"
        printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'tests=%s\n' "$TESTS"
        printf 'ppl_modules=%s\n' "$PPL_MODULES"
        printf 'ppl_layers=%s\n' "$PPL_LAYERS"
        printf 'down_layers=%s\n' "$DOWN_LAYERS"
        printf 'down_batches=%s\n' "$DOWN_BATCHES"
        printf 'arena_gb=%s\n' "$ARENA_GB"
        printf 'ppl_chunk=%s\n' "$PPL_CHUNK"
        printf 'ppl_chunks=%s\n' "$PPL_CHUNKS"
        printf 'bench_iters=%s\n' "$BENCH_ITERS"
        printf 'bench_warmup=%s\n' "$BENCH_WARMUP"
        printf 'w4a8_so=%s\n' "$W4A8_SO"
        if command -v git >/dev/null 2>&1; then
            printf 'git_branch=%s\n' "$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
            printf 'git_sha=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
        fi
        if [[ -n "$manifest" ]]; then
            printf 'kernel_manifest=%s\n' "$manifest"
            printf 'kernel_manifest_sha256=%s\n' "$(sha256_file "$manifest")"
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi -L || true
            nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap --format=csv,noheader || true
        fi
    } > "$CONTEXT_LOG"
}

maybe_build() {
    [[ "${W4A8_SWEEP_BUILD:-0}" == "1" ]] || return 0
    require_cmd cargo

    run_logged build_bench build bench 0 0 "$BUILD_TIMEOUT" "$(log_path build_bench)" \
        cargo build --release --features cuda --manifest-path "${V3_DIR}/Cargo.toml" -p rvllm-bench
}

validate_inputs() {
    require_cmd "$TIMEOUT_BIN"
    require_dir RVLLM_MODEL_DIR "${RVLLM_MODEL_DIR:-}"
    require_dir RVLLM_KERNELS_DIR "${RVLLM_KERNELS_DIR:-}"
    require_file RVLLM_CUTLASS_SO "${RVLLM_CUTLASS_SO:-}"
    require_file RVLLM_FA3_SO "${RVLLM_FA3_SO:-}"
    require_file RVLLM_POLICY "${RVLLM_POLICY:-}"
    if has_test ppl_modules; then
        require_file rvllm-ppl "$PPL_BIN"
        [[ -x "$PPL_BIN" ]] || die "rvllm-ppl is not executable: ${PPL_BIN}"
    fi
    if has_test down_bench; then
        require_file rvllm-bench "$BENCH_BIN"
        [[ -x "$BENCH_BIN" ]] || die "rvllm-bench is not executable: ${BENCH_BIN}"
    fi

    W4A8_SO="$(default_w4a8_so)"
    [[ -n "$W4A8_SO" ]] || die "set RVLLM_W4A8_SO or provide ${RVLLM_KERNELS_DIR}/sm_90/libw4a8_gemm.so"
    require_file RVLLM_W4A8_SO "$W4A8_SO"

    if has_test ppl_modules; then
        validate_module_list "$PPL_MODULES"
        validate_int_list W4A8_SWEEP_PPL_LAYERS "$PPL_LAYERS"
        [[ "$PPL_CHUNK" =~ ^[0-9]+$ && "$PPL_CHUNK" -gt 1 ]] || die "W4A8_SWEEP_PPL_CHUNK must be > 1"
        [[ "$PPL_CHUNKS" =~ ^[0-9]+$ && "$PPL_CHUNKS" -gt 0 ]] || die "W4A8_SWEEP_PPL_CHUNKS must be > 0"
    fi
    if has_test down_bench; then
        validate_int_list W4A8_SWEEP_DOWN_LAYERS "$DOWN_LAYERS"
        validate_int_list W4A8_SWEEP_DOWN_BATCHES "$DOWN_BATCHES"
        [[ "$BENCH_ITERS" =~ ^[0-9]+$ && "$BENCH_ITERS" -gt 0 ]] || die "W4A8_SWEEP_BENCH_ITERS must be > 0"
        [[ "$BENCH_WARMUP" =~ ^[0-9]+$ ]] || die "W4A8_SWEEP_BENCH_WARMUP must be an integer"
    fi
}

configure_common_env() {
    COMMON_ENV_ARGS=(
        "RVLLM_MODEL_DIR=${RVLLM_MODEL_DIR}"
        "RVLLM_KERNELS_DIR=${RVLLM_KERNELS_DIR}"
        "RVLLM_CUTLASS_SO=${RVLLM_CUTLASS_SO}"
        "RVLLM_FA3_SO=${RVLLM_FA3_SO}"
        "RVLLM_POLICY=${RVLLM_POLICY}"
        "RVLLM_W4A8_SO=${W4A8_SO}"
        "RVLLM_ARENA_GB=${ARENA_GB}"
        "RVLLM_EXPERIMENT_WEIGHT=w4a8-awq"
        "RVLLM_EXPERIMENT_ARCH=force-hopper"
        "RVLLM_W4A8=1"
        "RUST_LOG=${RUST_LOG_VALUE}"
    )
    [[ -n "${CUDA_ARCH:-}" ]] && COMMON_ENV_ARGS+=("CUDA_ARCH=${CUDA_ARCH}")

    UNSET_EXPERIMENT_ARGS=(
        -u RVLLM_EXPERIMENT
        -u RVLLM_EXPERIMENT_KV
        -u RVLLM_EXPERIMENT_ATTENTION
        -u RVLLM_EXPERIMENT_VALIDATION
        -u RVLLM_ROTORQUANT
        -u RVLLM_ROTORQUANT_BITS
        -u RVLLM_ROTORQUANT_CHUNK_DIM
    )
}

run_ppl_modules() {
    local modules=() layers_list=() module layers lane
    csv_to_array "$PPL_MODULES" modules
    csv_to_array "$PPL_LAYERS" layers_list

    for layers in "${layers_list[@]}"; do
        for module in "${modules[@]}"; do
            [[ -n "$module" ]] || continue
            lane="ppl_${module}_l${layers}"
            run_logged "$lane" ppl "$module" "$layers" 1 "$PPL_TIMEOUT" "$(log_path "$lane")" \
                env "${UNSET_EXPERIMENT_ARGS[@]}" "${COMMON_ENV_ARGS[@]}" \
                "RVLLM_EXPERIMENT_VALIDATION=ppl" \
                "RVLLM_MAX_LAYERS=${layers}" \
                "RVLLM_F16_LAYERS=${layers}" \
                "RVLLM_W4A8_ENCODE_LAYERS=${layers}" \
                "RVLLM_W4A8_MODULES=${module}" \
                "RVLLM_PPL_CHUNK=${PPL_CHUNK}" \
                "RVLLM_PPL_CHUNKS=${PPL_CHUNKS}" \
                "RVLLM_PROMPT=${PPL_PROMPT}" \
                "$PPL_BIN"
        done
    done
}

run_down_bench() {
    local layers_list=() batches=() layers batch lane
    csv_to_array "$DOWN_LAYERS" layers_list
    csv_to_array "$DOWN_BATCHES" batches

    for layers in "${layers_list[@]}"; do
        for batch in "${batches[@]}"; do
            lane="bench_down_l${layers}_b${batch}"
            run_logged "$lane" bench down "$layers" "$batch" "$BENCH_TIMEOUT" "$(log_path "$lane")" \
                env "${UNSET_EXPERIMENT_ARGS[@]}" "${COMMON_ENV_ARGS[@]}" \
                "RVLLM_EXPERIMENT_VALIDATION=throughput" \
                "RVLLM_MAX_LAYERS=${layers}" \
                "RVLLM_F16_LAYERS=${layers}" \
                "RVLLM_W4A8_ENCODE_LAYERS=${layers}" \
                "RVLLM_W4A8_MODULES=down" \
                "RVLLM_BATCH=${batch}" \
                "RVLLM_ITERS=${BENCH_ITERS}" \
                "RVLLM_WARMUP=${BENCH_WARMUP}" \
                "$BENCH_BIN"
        done
    done
}

print_done() {
    echo
    echo "[w4a8] logs: ${LOG_DIR}"
    echo "[w4a8] summary jsonl: ${SUMMARY_JSONL}"
    echo "[w4a8] summary tsv: ${SUMMARY_TSV}"
    echo
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
}

main() {
    init_logs
    validate_test_list
    maybe_build
    [[ "$FAILURES" -eq 0 ]] || die "build failed; see ${SUMMARY_TSV}"
    validate_inputs
    configure_common_env
    write_context

    echo "[w4a8] run_id=${RUN_ID}"
    echo "[w4a8] logs=${LOG_DIR}"
    echo "[w4a8] tests=${TESTS}"

    if has_test ppl_modules; then
        run_ppl_modules
    fi
    if has_test down_bench; then
        run_down_bench
    fi

    print_done
    [[ "$FAILURES" -eq 0 ]]
}

main "$@"
