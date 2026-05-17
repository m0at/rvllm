#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
V3_DIR="${RVLLM_V3_DIR:-${ROOT_DIR}/v3}"
TARGET_DIR="${RVLLM_TARGET_DIR:-${V3_DIR}/target/release}"

BENCH_BIN="${RVLLM_BENCH_BIN:-${TARGET_DIR}/rvllm-bench}"
PPL_BIN="${RVLLM_PPL_BIN:-${TARGET_DIR}/rvllm-ppl}"
SERVER_BIN="${RVLLM_SERVER_BIN:-${TARGET_DIR}/rvllm-server}"
W4A8_SMOKE_BIN="${RVLLM_W4A8_SMOKE_BIN:-${ROOT_DIR}/kernels/w4a8_smoke}"

RUN_ID="${H100_MATRIX_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${H100_MATRIX_LOG_DIR:-${ROOT_DIR}/logs/h100-experiment-matrix/${RUN_ID}}"
SUMMARY_JSONL="${LOG_DIR}/summary.jsonl"
SUMMARY_TSV="${LOG_DIR}/summary.tsv"
CONTEXT_LOG="${LOG_DIR}/context.txt"

TESTS="${H100_MATRIX_TESTS:-w4a8_smoke,server,bench,ppl}"
if [[ -n "${H100_MATRIX_EXPERIMENTS:-}" ]]; then
    EXPERIMENTS="${H100_MATRIX_EXPERIMENTS}"
elif [[ -n "${RVLLM_EXPERIMENT:-}" \
    || -n "${RVLLM_EXPERIMENT_WEIGHT:-}" \
    || -n "${RVLLM_EXPERIMENT_KV:-}" \
    || -n "${RVLLM_EXPERIMENT_ATTENTION:-}" \
    || -n "${RVLLM_EXPERIMENT_ARCH:-}" \
    || -n "${RVLLM_EXPERIMENT_VALIDATION:-}" \
    || -n "${RVLLM_W4A8:-}" \
    || -n "${RVLLM_ROTORQUANT:-}" ]]; then
    EXPERIMENTS="current"
else
    EXPERIMENTS="baseline"
fi

TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
KILL_AFTER="${H100_MATRIX_KILL_AFTER:-15s}"
BUILD_TIMEOUT="${H100_MATRIX_BUILD_TIMEOUT:-1800}"
W4A8_SMOKE_TIMEOUT="${H100_MATRIX_W4A8_SMOKE_TIMEOUT:-90}"
W4A8_MEMCHECK_TIMEOUT="${H100_MATRIX_W4A8_MEMCHECK_TIMEOUT:-300}"
SERVER_READY_TIMEOUT="${H100_MATRIX_SERVER_READY_TIMEOUT:-420}"
HTTP_TIMEOUT="${H100_MATRIX_HTTP_TIMEOUT:-60}"
CHAT_TIMEOUT="${H100_MATRIX_CHAT_TIMEOUT:-240}"
BENCH_B1_TIMEOUT="${H100_MATRIX_B1_TIMEOUT:-900}"
BENCH_B128_TIMEOUT="${H100_MATRIX_B128_TIMEOUT:-900}"
PPL_TIMEOUT="${H100_MATRIX_PPL_TIMEOUT:-900}"
COMPUTE_SANITIZER_BIN="${COMPUTE_SANITIZER_BIN:-compute-sanitizer}"

BIND_HOST="${H100_MATRIX_BIND_HOST:-127.0.0.1}"
CURL_HOST="${H100_MATRIX_CURL_HOST:-127.0.0.1}"
PORT="${H100_MATRIX_PORT:-18080}"
BASE_URL="http://${CURL_HOST}:${PORT}"
SERVED_MODEL_NAME="${H100_MATRIX_SERVED_MODEL_NAME:-${RVLLM_SERVED_MODEL_NAME:-gemma4-31b-solidsf}}"
RUST_LOG_VALUE="${RUST_LOG:-info}"

ARENA_GB="${H100_MATRIX_ARENA_GB:-${RVLLM_ARENA_GB:-74}}"
B1_ITERS="${H100_MATRIX_B1_ITERS:-8}"
B1_WARMUP="${H100_MATRIX_B1_WARMUP:-2}"
B128_ITERS="${H100_MATRIX_B128_ITERS:-5}"
B128_WARMUP="${H100_MATRIX_B128_WARMUP:-1}"
PPL_CHUNK="${H100_MATRIX_PPL_CHUNK:-32}"
PPL_CHUNKS="${H100_MATRIX_PPL_CHUNKS:-1}"
PPL_PROMPT="${H100_MATRIX_PPL_PROMPT:-Angular momentum is conserved when the net external torque on a system is zero. A spinning figure skater who pulls in their arms reduces rotational inertia, so their angular velocity rises while total angular momentum stays nearly constant. This small text is only for a bounded perplexity smoke chunk.}"
CHAT_PROMPT="${H100_MATRIX_CHAT_PROMPT:-Explain angular momentum conservation using a spinning figure skater pulling in their arms. Keep it to one sentence.}"
CHAT_MAX_TOKENS="${H100_MATRIX_CHAT_MAX_TOKENS:-48}"

SERVER_MAX_MODEL_LEN="${H100_MATRIX_MAX_MODEL_LEN:-8192}"
SERVER_MAX_NUM_SEQS="${H100_MATRIX_MAX_NUM_SEQS:-1}"
SERVER_MAX_NUM_BATCHED_TOKENS="${H100_MATRIX_MAX_NUM_BATCHED_TOKENS:-2048}"
SERVER_MAX_PREFILL_CHUNK="${H100_MATRIX_MAX_PREFILL_CHUNK:-128}"

SERVER_PID=""
FAILURES=0
VARIANT_UNSET_ARGS=()
VARIANT_SET_ARGS=()
VARIANT_LABEL_ARGS=()
COMMON_ENV_ARGS=()
GIT_BRANCH=""
GIT_SHA=""
KERNEL_MANIFEST_PATH=""
KERNEL_MANIFEST_SHA256=""

die() {
    echo "h100-matrix: $*" >&2
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

join_env_label() {
    local out="" arg
    for arg in "$@"; do
        [[ -n "$arg" ]] || continue
        if [[ -n "$out" ]]; then
            out+=",${arg}"
        else
            out="$arg"
        fi
    done
    [[ -n "$out" ]] || out="none"
    printf '%s' "$out"
}

lane_label() {
    join_env_label "${VARIANT_LABEL_ARGS[@]}" "$@"
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

capture_metadata() {
    GIT_BRANCH="${H100_MATRIX_GIT_BRANCH:-}"
    GIT_SHA="${H100_MATRIX_GIT_SHA:-}"
    if command -v git >/dev/null 2>&1; then
        [[ -n "$GIT_BRANCH" ]] || GIT_BRANCH="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
        [[ -n "$GIT_SHA" ]] || GIT_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
    fi

    KERNEL_MANIFEST_PATH="${H100_MATRIX_KERNEL_MANIFEST:-}"
    if [[ -z "$KERNEL_MANIFEST_PATH" && -n "${RVLLM_KERNELS_DIR:-}" ]]; then
        if [[ -f "${RVLLM_KERNELS_DIR}/manifest.json" ]]; then
            KERNEL_MANIFEST_PATH="${RVLLM_KERNELS_DIR}/manifest.json"
        elif [[ -f "${RVLLM_KERNELS_DIR}/sm_90/manifest.json" ]]; then
            KERNEL_MANIFEST_PATH="${RVLLM_KERNELS_DIR}/sm_90/manifest.json"
        fi
    fi
    if [[ -n "$KERNEL_MANIFEST_PATH" && -f "$KERNEL_MANIFEST_PATH" ]]; then
        KERNEL_MANIFEST_SHA256="$(sha256_file "$KERNEL_MANIFEST_PATH")"
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

append_summary() {
    local variant="$1"
    local test="$2"
    local status="$3"
    local exit_code="$4"
    local elapsed_s="$5"
    local log_file="$6"
    local message="$7"
    local lane_env="${8:-}"
    local mem_before_mib="${9:-}"
    local mem_after_mib="${10:-}"

    printf '{"run_id":"%s","git_branch":"%s","git_sha":"%s","kernel_manifest":"%s","kernel_manifest_sha256":"%s","variant":"%s","test":"%s","lane_env":"%s","status":"%s","exit_code":%s,"elapsed_s":%s,"mem_before_mib":%s,"mem_after_mib":%s,"log":"%s","message":"%s"}\n' \
        "$(json_escape "$RUN_ID")" \
        "$(json_escape "$GIT_BRANCH")" \
        "$(json_escape "$GIT_SHA")" \
        "$(json_escape "$KERNEL_MANIFEST_PATH")" \
        "$(json_escape "$KERNEL_MANIFEST_SHA256")" \
        "$(json_escape "$variant")" \
        "$(json_escape "$test")" \
        "$(json_escape "$lane_env")" \
        "$(json_escape "$status")" \
        "$exit_code" \
        "$elapsed_s" \
        "$(json_number_or_null "$mem_before_mib")" \
        "$(json_number_or_null "$mem_after_mib")" \
        "$(json_escape "$log_file")" \
        "$(json_escape "$message")" >> "$SUMMARY_JSONL"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(tsv_escape "$RUN_ID")" \
        "$(tsv_escape "$GIT_BRANCH")" \
        "$(tsv_escape "$GIT_SHA")" \
        "$(tsv_escape "$KERNEL_MANIFEST_PATH")" \
        "$(tsv_escape "$KERNEL_MANIFEST_SHA256")" \
        "$(tsv_escape "$variant")" \
        "$(tsv_escape "$test")" \
        "$(tsv_escape "$lane_env")" \
        "$(tsv_escape "$status")" \
        "$exit_code" \
        "$elapsed_s" \
        "$(tsv_escape "$mem_before_mib")" \
        "$(tsv_escape "$mem_after_mib")" \
        "$(tsv_escape "$log_file")" \
        "$(tsv_escape "$message")" >> "$SUMMARY_TSV"

    case "$status" in
        fail|timeout)
            FAILURES=$((FAILURES + 1))
            ;;
    esac
}

log_path() {
    local variant="$1"
    local test="$2"
    printf '%s/%s.%s.log' "$LOG_DIR" "$(safe_name "$variant")" "$(safe_name "$test")"
}

run_command() {
    local variant="$1"
    local test="$2"
    local timeout_s="$3"
    local log_file="$4"
    local lane_env="$5"
    shift 5

    local start elapsed exit_code status message cmd_text mem_before_mib mem_after_mib
    cmd_text="$(format_cmd "$@")"
    mem_before_mib="$(gpu_memory_used_mib)"

    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'git_branch=%s\n' "$GIT_BRANCH"
        printf 'git_sha=%s\n' "$GIT_SHA"
        printf 'kernel_manifest=%s\n' "$KERNEL_MANIFEST_PATH"
        printf 'kernel_manifest_sha256=%s\n' "$KERNEL_MANIFEST_SHA256"
        printf 'variant=%s\n' "$variant"
        printf 'test=%s\n' "$test"
        printf 'lane_env=%s\n' "$lane_env"
        printf 'timeout_s=%s\n' "$timeout_s"
        printf 'memory_before_mib=%s\n' "$mem_before_mib"
        printf 'command=%s\n\n' "$cmd_text"
    } > "$log_file"

    echo "[matrix] ${variant}/${test} timeout=${timeout_s}s"
    start=$SECONDS
    set +e
    "$TIMEOUT_BIN" --kill-after="$KILL_AFTER" "${timeout_s}s" "$@" >> "$log_file" 2>&1
    exit_code=$?
    set -e
    elapsed=$((SECONDS - start))
    mem_after_mib="$(gpu_memory_used_mib)"

    status="fail"
    message="exit ${exit_code}"
    if [[ "$exit_code" -eq 0 ]]; then
        status="pass"
        message="ok"
    elif [[ "$exit_code" -eq 124 || "$exit_code" -eq 137 ]]; then
        status="timeout"
        message="timed out"
    fi

    {
        printf '\nexit_code=%s\n' "$exit_code"
        printf 'status=%s\n' "$status"
        printf 'elapsed_s=%s\n' "$elapsed"
        printf 'memory_after_mib=%s\n' "$mem_after_mib"
    } >> "$log_file"

    append_summary "$variant" "$test" "$status" "$exit_code" "$elapsed" "$log_file" "$message" "$lane_env" "$mem_before_mib" "$mem_after_mib"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup() {
    local code=$?
    stop_server
    exit "$code"
}

trap cleanup EXIT
trap 'echo "[matrix] interrupted" >&2; exit 130' INT
trap 'echo "[matrix] terminated" >&2; exit 143' TERM

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

init_logs() {
    mkdir -p "$LOG_DIR"
    : > "$SUMMARY_JSONL"
    printf 'run_id\tgit_branch\tgit_sha\tkernel_manifest\tkernel_manifest_sha256\tvariant\ttest\tlane_env\tstatus\texit_code\telapsed_s\tmem_before_mib\tmem_after_mib\tlog\tmessage\n' > "$SUMMARY_TSV"
}

write_context() {
    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'repo=%s\n' "$ROOT_DIR"
        printf 'git_branch=%s\n' "$GIT_BRANCH"
        printf 'git_sha=%s\n' "$GIT_SHA"
        printf 'kernel_manifest=%s\n' "$KERNEL_MANIFEST_PATH"
        printf 'kernel_manifest_sha256=%s\n' "$KERNEL_MANIFEST_SHA256"
        printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'tests=%s\n' "$TESTS"
        printf 'experiments=%s\n' "$EXPERIMENTS"
        printf 'base_url=%s\n' "$BASE_URL"
        printf 'arena_gb=%s\n' "$ARENA_GB"
        printf 'memory_used_mib=%s\n' "$(gpu_memory_used_mib)"
        if command -v git >/dev/null 2>&1; then
            git -C "$ROOT_DIR" status --short 2>/dev/null || true
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi -L || true
            nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true
        fi
    } > "$CONTEXT_LOG"
}

maybe_build() {
    [[ "${H100_MATRIX_BUILD:-0}" == "1" ]] || return 0
    require_cmd "$TIMEOUT_BIN"
    require_cmd cargo

    run_command setup build_rvllm_bench "$BUILD_TIMEOUT" "$(log_path setup build_rvllm_bench)" "setup" \
        cargo build --release --features cuda --manifest-path "${V3_DIR}/Cargo.toml" -p rvllm-bench

    run_command setup build_rvllm_serve "$BUILD_TIMEOUT" "$(log_path setup build_rvllm_serve)" "setup" \
        cargo build --release --features cuda,cublaslt --manifest-path "${V3_DIR}/Cargo.toml" -p rvllm-serve
}

validate_inputs() {
    require_cmd "$TIMEOUT_BIN"
    require_cmd curl

    if has_test server || has_test bench || has_test ppl; then
        require_dir RVLLM_MODEL_DIR "${RVLLM_MODEL_DIR:-}"
        require_dir RVLLM_KERNELS_DIR "${RVLLM_KERNELS_DIR:-}"
        if [[ "${H100_MATRIX_REQUIRE_SM90_PATHS:-1}" == "1" ]]; then
            require_file RVLLM_CUTLASS_SO "${RVLLM_CUTLASS_SO:-}"
            require_file RVLLM_FA3_SO "${RVLLM_FA3_SO:-}"
            require_file RVLLM_POLICY "${RVLLM_POLICY:-}"
        fi
    fi

    if has_test server; then
        require_file rvllm-server "$SERVER_BIN"
        [[ -x "$SERVER_BIN" ]] || die "rvllm-server is not executable: ${SERVER_BIN}"
    fi
    if has_test bench; then
        require_file rvllm-bench "$BENCH_BIN"
        [[ -x "$BENCH_BIN" ]] || die "rvllm-bench is not executable: ${BENCH_BIN}"
    fi
    if has_test ppl; then
        require_file rvllm-ppl "$PPL_BIN"
        [[ -x "$PPL_BIN" ]] || die "rvllm-ppl is not executable: ${PPL_BIN}"
    fi
}

default_w4a8_so() {
    if [[ -n "${RVLLM_W4A8_SO:-}" ]]; then
        printf '%s' "$RVLLM_W4A8_SO"
    elif [[ -f "${RVLLM_KERNELS_DIR:-}/sm_90/libw4a8_gemm.so" ]]; then
        printf '%s' "${RVLLM_KERNELS_DIR}/sm_90/libw4a8_gemm.so"
    elif [[ -f "${RVLLM_KERNELS_DIR:-}/libw4a8_gemm.so" ]]; then
        printf '%s' "${RVLLM_KERNELS_DIR}/libw4a8_gemm.so"
    else
        printf '%s' ""
    fi
}

configure_common_env() {
    COMMON_ENV_ARGS=(
        "RVLLM_MODEL_DIR=${RVLLM_MODEL_DIR:-}"
        "RVLLM_KERNELS_DIR=${RVLLM_KERNELS_DIR:-}"
        "RVLLM_ARENA_GB=${ARENA_GB}"
        "RVLLM_SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
        "RUST_LOG=${RUST_LOG_VALUE}"
    )

    [[ -n "${RVLLM_CUTLASS_SO:-}" ]] && COMMON_ENV_ARGS+=("RVLLM_CUTLASS_SO=${RVLLM_CUTLASS_SO}")
    [[ -n "${RVLLM_FA3_SO:-}" ]] && COMMON_ENV_ARGS+=("RVLLM_FA3_SO=${RVLLM_FA3_SO}")
    [[ -n "${RVLLM_POLICY:-}" ]] && COMMON_ENV_ARGS+=("RVLLM_POLICY=${RVLLM_POLICY}")
    [[ -n "${CUDA_ARCH:-}" ]] && COMMON_ENV_ARGS+=("CUDA_ARCH=${CUDA_ARCH}")
    return 0
}

collect_current_variant_labels() {
    local name value
    VARIANT_LABEL_ARGS=()
    for name in \
        RVLLM_EXPERIMENT \
        RVLLM_EXPERIMENT_WEIGHT \
        RVLLM_EXPERIMENT_KV \
        RVLLM_EXPERIMENT_ATTENTION \
        RVLLM_EXPERIMENT_ARCH \
        RVLLM_W4A8 \
        RVLLM_W4A8_SO \
        RVLLM_ROTORQUANT \
        RVLLM_ROTORQUANT_BITS \
        RVLLM_ROTORQUANT_CHUNK_DIM; do
        value="${!name-}"
        [[ -n "$value" ]] || continue
        VARIANT_LABEL_ARGS+=("${name}=${value}")
    done
    [[ "${#VARIANT_LABEL_ARGS[@]}" -gt 0 ]] || VARIANT_LABEL_ARGS=("RVLLM_EXPERIMENT=current")
}

configure_variant() {
    local variant="$1"
    local w4a8_so

    VARIANT_UNSET_ARGS=()
    VARIANT_SET_ARGS=()
    VARIANT_LABEL_ARGS=()

    case "$variant" in
        baseline)
            VARIANT_UNSET_ARGS=(-u RVLLM_EXPERIMENT -u RVLLM_EXPERIMENT_WEIGHT -u RVLLM_EXPERIMENT_KV -u RVLLM_EXPERIMENT_ATTENTION -u RVLLM_EXPERIMENT_ARCH -u RVLLM_EXPERIMENT_VALIDATION -u RVLLM_W4A8 -u RVLLM_W4A8_SO -u RVLLM_ROTORQUANT -u RVLLM_ROTORQUANT_BITS -u RVLLM_ROTORQUANT_CHUNK_DIM)
            VARIANT_LABEL_ARGS=("RVLLM_EXPERIMENT=baseline")
            ;;
        current)
            collect_current_variant_labels
            ;;
        w4a8)
            w4a8_so="$(default_w4a8_so)"
            [[ -n "$w4a8_so" ]] || die "w4a8 variant needs RVLLM_W4A8_SO or ${RVLLM_KERNELS_DIR:-}/sm_90/libw4a8_gemm.so"
            require_file RVLLM_W4A8_SO "$w4a8_so"
            VARIANT_UNSET_ARGS=(-u RVLLM_EXPERIMENT -u RVLLM_EXPERIMENT_KV -u RVLLM_EXPERIMENT_ATTENTION -u RVLLM_EXPERIMENT_ARCH -u RVLLM_EXPERIMENT_VALIDATION -u RVLLM_ROTORQUANT -u RVLLM_ROTORQUANT_BITS -u RVLLM_ROTORQUANT_CHUNK_DIM)
            VARIANT_SET_ARGS=("RVLLM_EXPERIMENT_WEIGHT=w4a8-awq" "RVLLM_W4A8=1" "RVLLM_W4A8_SO=${w4a8_so}")
            VARIANT_LABEL_ARGS=("${VARIANT_SET_ARGS[@]}")
            ;;
        rotor_cl3|rotor|rotorquant)
            VARIANT_UNSET_ARGS=(-u RVLLM_EXPERIMENT -u RVLLM_EXPERIMENT_WEIGHT -u RVLLM_EXPERIMENT_ATTENTION -u RVLLM_EXPERIMENT_ARCH -u RVLLM_EXPERIMENT_VALIDATION -u RVLLM_W4A8 -u RVLLM_W4A8_SO)
            VARIANT_SET_ARGS=("RVLLM_EXPERIMENT_KV=rotorquant" "RVLLM_ROTORQUANT=rotor_cl3")
            VARIANT_LABEL_ARGS=("${VARIANT_SET_ARGS[@]}")
            ;;
        planar2)
            VARIANT_UNSET_ARGS=(-u RVLLM_EXPERIMENT -u RVLLM_EXPERIMENT_WEIGHT -u RVLLM_EXPERIMENT_ATTENTION -u RVLLM_EXPERIMENT_ARCH -u RVLLM_EXPERIMENT_VALIDATION -u RVLLM_W4A8 -u RVLLM_W4A8_SO)
            VARIANT_SET_ARGS=("RVLLM_EXPERIMENT_KV=rotorquant" "RVLLM_ROTORQUANT=planar2")
            VARIANT_LABEL_ARGS=("${VARIANT_SET_ARGS[@]}")
            ;;
        iso4)
            VARIANT_UNSET_ARGS=(-u RVLLM_EXPERIMENT -u RVLLM_EXPERIMENT_WEIGHT -u RVLLM_EXPERIMENT_ATTENTION -u RVLLM_EXPERIMENT_ARCH -u RVLLM_EXPERIMENT_VALIDATION -u RVLLM_W4A8 -u RVLLM_W4A8_SO)
            VARIANT_SET_ARGS=("RVLLM_EXPERIMENT_KV=rotorquant" "RVLLM_ROTORQUANT=iso4")
            VARIANT_LABEL_ARGS=("${VARIANT_SET_ARGS[@]}")
            ;;
        exp:*)
            VARIANT_UNSET_ARGS=(-u RVLLM_EXPERIMENT_WEIGHT -u RVLLM_EXPERIMENT_KV -u RVLLM_EXPERIMENT_ATTENTION -u RVLLM_EXPERIMENT_ARCH -u RVLLM_EXPERIMENT_VALIDATION -u RVLLM_W4A8 -u RVLLM_W4A8_SO -u RVLLM_ROTORQUANT -u RVLLM_ROTORQUANT_BITS -u RVLLM_ROTORQUANT_CHUNK_DIM)
            VARIANT_SET_ARGS=("RVLLM_EXPERIMENT=${variant#exp:}")
            VARIANT_LABEL_ARGS=("${VARIANT_SET_ARGS[@]}")
            ;;
        *)
            die "unknown H100_MATRIX_EXPERIMENTS variant: ${variant}"
            ;;
    esac
}

run_w4a8_smoke() {
    local log_file lane_env
    log_file="$(log_path setup w4a8_smoke)"
    lane_env="RVLLM_EXPERIMENT_VALIDATION=smoke"
    if [[ ! -x "$W4A8_SMOKE_BIN" ]]; then
        append_summary setup w4a8_smoke skip 0 0 "$log_file" "missing optional executable: ${W4A8_SMOKE_BIN}" "$lane_env"
        return 0
    fi
    run_command setup w4a8_smoke "$W4A8_SMOKE_TIMEOUT" "$log_file" "$lane_env" "$W4A8_SMOKE_BIN"
}

run_w4a8_memcheck() {
    local log_file lane_env
    log_file="$(log_path setup w4a8_memcheck)"
    lane_env="RVLLM_EXPERIMENT_VALIDATION=smoke,RVLLM_MEMCHECK=w4a8"
    if [[ ! -x "$W4A8_SMOKE_BIN" ]]; then
        append_summary setup w4a8_memcheck skip 0 0 "$log_file" "missing optional executable: ${W4A8_SMOKE_BIN}" "$lane_env"
        return 0
    fi
    if ! command -v "$COMPUTE_SANITIZER_BIN" >/dev/null 2>&1; then
        append_summary setup w4a8_memcheck skip 0 0 "$log_file" "missing optional command: ${COMPUTE_SANITIZER_BIN}" "$lane_env"
        return 0
    fi
    run_command setup w4a8_memcheck "$W4A8_MEMCHECK_TIMEOUT" "$log_file" "$lane_env" \
        "$COMPUTE_SANITIZER_BIN" --tool memcheck --error-exitcode 99 "$W4A8_SMOKE_BIN"
}

wait_for_server() {
    local log_file="$1"
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))

    while (( SECONDS < deadline )); do
        if curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            printf '\nserver exited before readiness\n' >> "$log_file"
            wait "$SERVER_PID" >> "$log_file" 2>&1 || true
            return 1
        fi
        sleep 1
    done

    printf '\nserver did not become ready in %ss\n' "$SERVER_READY_TIMEOUT" >> "$log_file"
    return 1
}

run_server_suite() {
    local variant="$1"
    local log_file start elapsed lane_env mem_before_mib mem_after_mib
    log_file="$(log_path "$variant" server_ready)"
    lane_env="$(lane_label "RVLLM_EXPERIMENT_VALIDATION=chat")"

    stop_server
    mem_before_mib="$(gpu_memory_used_mib)"
    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'git_branch=%s\n' "$GIT_BRANCH"
        printf 'git_sha=%s\n' "$GIT_SHA"
        printf 'kernel_manifest=%s\n' "$KERNEL_MANIFEST_PATH"
        printf 'kernel_manifest_sha256=%s\n' "$KERNEL_MANIFEST_SHA256"
        printf 'variant=%s\n' "$variant"
        printf 'test=server_ready\n'
        printf 'lane_env=%s\n' "$lane_env"
        printf 'base_url=%s\n' "$BASE_URL"
        printf 'timeout_s=%s\n\n' "$SERVER_READY_TIMEOUT"
        printf 'memory_before_mib=%s\n' "$mem_before_mib"
    } > "$log_file"

    echo "[matrix] ${variant}/server_ready timeout=${SERVER_READY_TIMEOUT}s"
    start=$SECONDS
    env "${VARIANT_UNSET_ARGS[@]}" "${COMMON_ENV_ARGS[@]}" "${VARIANT_SET_ARGS[@]}" \
        "RVLLM_EXPERIMENT_VALIDATION=chat" \
        "$SERVER_BIN" \
        --host "$BIND_HOST" \
        --port "$PORT" \
        --max-model-len "$SERVER_MAX_MODEL_LEN" \
        --max-num-seqs "$SERVER_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$SERVER_MAX_NUM_BATCHED_TOKENS" \
        --max-prefill-chunk "$SERVER_MAX_PREFILL_CHUNK" >> "$log_file" 2>&1 &
    SERVER_PID=$!

    if wait_for_server "$log_file"; then
        elapsed=$((SECONDS - start))
        mem_after_mib="$(gpu_memory_used_mib)"
        {
            printf '\nstatus=pass\n'
            printf 'elapsed_s=%s\n' "$elapsed"
            printf 'memory_after_mib=%s\n' "$mem_after_mib"
        } >> "$log_file"
        append_summary "$variant" server_ready pass 0 "$elapsed" "$log_file" "ready" "$lane_env" "$mem_before_mib" "$mem_after_mib"
    else
        elapsed=$((SECONDS - start))
        mem_after_mib="$(gpu_memory_used_mib)"
        {
            printf '\nstatus=fail\n'
            printf 'elapsed_s=%s\n' "$elapsed"
            printf 'memory_after_mib=%s\n' "$mem_after_mib"
        } >> "$log_file"
        append_summary "$variant" server_ready fail 1 "$elapsed" "$log_file" "not ready" "$lane_env" "$mem_before_mib" "$mem_after_mib"
        append_summary "$variant" server_health skip 0 0 "$(log_path "$variant" server_health)" "server not ready" "$lane_env"
        append_summary "$variant" server_models skip 0 0 "$(log_path "$variant" server_models)" "server not ready" "$lane_env"
        append_summary "$variant" server_chat_angular_momentum skip 0 0 "$(log_path "$variant" server_chat_angular_momentum)" "server not ready" "$lane_env"
        stop_server
        return 0
    fi

    run_command "$variant" server_health "$HTTP_TIMEOUT" "$(log_path "$variant" server_health)" "$lane_env" \
        curl -fsS --max-time "$HTTP_TIMEOUT" "${BASE_URL}/health"

    run_command "$variant" server_models "$HTTP_TIMEOUT" "$(log_path "$variant" server_models)" "$lane_env" \
        curl -fsS --max-time "$HTTP_TIMEOUT" "${BASE_URL}/v1/models"

    local model_json prompt_json payload
    model_json="$(json_escape "$SERVED_MODEL_NAME")"
    prompt_json="$(json_escape "$CHAT_PROMPT")"
    payload="{\"model\":\"${model_json}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt_json}\"}],\"max_tokens\":${CHAT_MAX_TOKENS},\"temperature\":0}"
    run_command "$variant" server_chat_angular_momentum "$CHAT_TIMEOUT" "$(log_path "$variant" server_chat_angular_momentum)" "$lane_env" \
        curl -fsS --max-time "$CHAT_TIMEOUT" "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload"

    stop_server
}

run_bench_suite() {
    local variant="$1"
    local lane_env_b1 lane_env_b128
    lane_env_b1="$(lane_label "RVLLM_EXPERIMENT_VALIDATION=throughput" "RVLLM_BATCH=1")"
    lane_env_b128="$(lane_label "RVLLM_EXPERIMENT_VALIDATION=throughput" "RVLLM_BATCH=128")"

    run_command "$variant" rvllm_bench_b1 "$BENCH_B1_TIMEOUT" "$(log_path "$variant" rvllm_bench_b1)" "$lane_env_b1" \
        env "${VARIANT_UNSET_ARGS[@]}" "${COMMON_ENV_ARGS[@]}" "${VARIANT_SET_ARGS[@]}" \
        "RVLLM_EXPERIMENT_VALIDATION=throughput" \
        "RVLLM_BATCH=1" "RVLLM_ITERS=${B1_ITERS}" "RVLLM_WARMUP=${B1_WARMUP}" \
        "$BENCH_BIN"

    run_command "$variant" rvllm_bench_b128 "$BENCH_B128_TIMEOUT" "$(log_path "$variant" rvllm_bench_b128)" "$lane_env_b128" \
        env "${VARIANT_UNSET_ARGS[@]}" "${COMMON_ENV_ARGS[@]}" "${VARIANT_SET_ARGS[@]}" \
        "RVLLM_EXPERIMENT_VALIDATION=throughput" \
        "RVLLM_BATCH=128" "RVLLM_ITERS=${B128_ITERS}" "RVLLM_WARMUP=${B128_WARMUP}" \
        "$BENCH_BIN"
}

run_ppl_suite() {
    local variant="$1"
    local lane_env
    lane_env="$(lane_label "RVLLM_EXPERIMENT_VALIDATION=ppl" "RVLLM_PPL_CHUNK=${PPL_CHUNK}" "RVLLM_PPL_CHUNKS=${PPL_CHUNKS}")"

    run_command "$variant" rvllm_ppl_small_chunk "$PPL_TIMEOUT" "$(log_path "$variant" rvllm_ppl_small_chunk)" "$lane_env" \
        env "${VARIANT_UNSET_ARGS[@]}" "${COMMON_ENV_ARGS[@]}" "${VARIANT_SET_ARGS[@]}" \
        "RVLLM_EXPERIMENT_VALIDATION=ppl" \
        "RVLLM_PPL_CHUNK=${PPL_CHUNK}" "RVLLM_PPL_CHUNKS=${PPL_CHUNKS}" "RVLLM_PROMPT=${PPL_PROMPT}" \
        "$PPL_BIN"
}

print_done() {
    echo
    echo "[matrix] logs: ${LOG_DIR}"
    echo "[matrix] summary jsonl: ${SUMMARY_JSONL}"
    echo "[matrix] summary tsv: ${SUMMARY_TSV}"
    echo
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
}

main() {
    init_logs
    capture_metadata
    write_context
    maybe_build
    validate_inputs
    configure_common_env

    echo "[matrix] run_id=${RUN_ID}"
    echo "[matrix] logs=${LOG_DIR}"
    echo "[matrix] tests=${TESTS}"
    echo "[matrix] experiments=${EXPERIMENTS}"

    if has_test w4a8_smoke; then
        run_w4a8_smoke
    fi
    if has_test w4a8_memcheck; then
        run_w4a8_memcheck
    fi

    local variants_raw variant
    IFS=',' read -r -a variants_raw <<< "$EXPERIMENTS"
    for variant in "${variants_raw[@]}"; do
        [[ -n "$variant" ]] || continue
        configure_variant "$variant"
        if has_test server; then
            run_server_suite "$variant"
        fi
        if has_test bench; then
            run_bench_suite "$variant"
        fi
        if has_test ppl; then
            run_ppl_suite "$variant"
        fi
    done

    print_done
    [[ "$FAILURES" -eq 0 ]]
}

main "$@"
