#!/bin/bash
# Sweep N profile variants through clean_smoke.sh, write a single
# structured Markdown report to /tmp/sweep_<TS>.md with EVERY prompt
# and its full response (not just FAILs) plus per-prompt decode tok/s.
#
# Variant definition: each entry in VARIANTS_ORDER is a label that
# maps via VARIANT_ENV[label] to a space-separated list of
# `KEY=VALUE` overrides applied to the active rvllm-serve profile
# before launching the smoke. Source-level changes (FA2_THREADS) use
# the pseudo-key __SOURCE_FA2_THREADS__ which triggers a kernel-source
# edit + PTX rebuild + cargo rebuild.
#
# Variant A_prod = baseline (no overrides). Each variant returns the
# profile to baseline before applying its own overrides, so the order
# of variants doesn't matter.

set -u
OUTDIR="${1:-/tmp}"
TS=$(date +%Y%m%d-%H%M%S)
REPORT="$OUTDIR/sweep_${TS}.md"
PROFILE=/home/r00t/.rvllm/active-profile.env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

declare -A VARIANT_ENV
declare -a VARIANTS_ORDER
# Cycle-55 sweep, post-leak rework — every variant explicitly declares
# its FULL state. `disable_all_flags` zeroes every NVFP4 toggle; each
# variant re-enables only what it tests. No reliance on cross-variant
# cleanup, no "production baseline" implicit state. A_prod runs LAST
# as a regression control after B/C/D/E.
#
# Order rationale: B/C/D/E first while A_prod is fresh in mind, then
# A_prod at the end — if A passes after the variants, baseline is
# proven not to have drifted across the variant series.
VARIANTS_ORDER=( B_pure_mse C_no_hadamard D_split_gqa E_hybrid_global_fp8 A_prod )

# Production NVFP4 baseline (used by A_prod and added to B/D/E):
#   HADAMARD=1 + PER_TOKEN_Q_SCALE=1 + HADAMARD_V=1
#   K_SCALE_POLICY=amax6 + V_SCALE_POLICY=amax6
PROD_BASE="RVLLM_NVFP4_HADAMARD=1 RVLLM_PER_TOKEN_Q_SCALE=1 RVLLM_NVFP4_HADAMARD_V=1 RVLLM_NVFP4_K_SCALE_POLICY=amax6 RVLLM_NVFP4_V_SCALE_POLICY=amax6"

# B: V-policy flipped to mse (cycle-37c historical winner; cycle-53 cliff).
VARIANT_ENV[B_pure_mse]="RVLLM_NVFP4_HADAMARD=1 RVLLM_PER_TOKEN_Q_SCALE=1 RVLLM_NVFP4_HADAMARD_V=1 RVLLM_NVFP4_K_SCALE_POLICY=amax6 RVLLM_NVFP4_V_SCALE_POLICY=mse"

# C: Hadamard rotation off, per-token-Q off. K/V policies still amax6
# so the test isolates *just* the rotation.
VARIANT_ENV[C_no_hadamard]="RVLLM_NVFP4_HADAMARD=0 RVLLM_PER_TOKEN_Q_SCALE=0 RVLLM_NVFP4_HADAMARD_V=0 RVLLM_NVFP4_K_SCALE_POLICY=amax6 RVLLM_NVFP4_V_SCALE_POLICY=amax6"

# D: env-gated GQA-shared split-decode kernel + production baseline.
# Codex flagged this kernel repeatedly as default-off-pending-validation;
# this sweep IS the validation pass.
VARIANT_ENV[D_split_gqa]="${PROD_BASE} RVLLM_NVFP4_SPLIT_GQA=1"

# E: hybrid path — global layers on FP8 KV, sliding stays NVFP4, all
# else baseline. Cycle-55 said "null/regressive" pre-Codex43-54;
# re-check under the new fixes.
VARIANT_ENV[E_hybrid_global_fp8]="${PROD_BASE} RVLLM_NVFP4_HYBRID_GLOBAL_FP8=1"

# A_prod (control, last): pure production baseline.
VARIANT_ENV[A_prod]="${PROD_BASE}"

# Every NVFP4 toggle that any variant might touch. disable_all_flags
# explicitly writes the OFF/zero value for every key here so no leak
# can survive across variants. apply_env_overrides then UPDATES
# (sed-delete + append) only the keys the variant declares.
ALL_NVFP4_KEYS=(
    RVLLM_NVFP4_HADAMARD
    RVLLM_NVFP4_HADAMARD_V
    RVLLM_PER_TOKEN_Q_SCALE
    RVLLM_NVFP4_K_SCALE_POLICY
    RVLLM_NVFP4_V_SCALE_POLICY
    RVLLM_NVFP4_SPLIT_GQA
    RVLLM_NVFP4_DECODE_GQA
    RVLLM_FP8_DECODE_GQA
    RVLLM_NVFP4_HYBRID_GLOBAL_FP8
    RVLLM_NVFP4_HYBRID_SLIDING_FP8
)

apply_source_fa2_threads() {
    local val="$1"
    sed -i "s/^#define FA2_THREADS [0-9]\+$/#define FA2_THREADS ${val}/" \
        "${REPO_ROOT}/kernels/flash_attention_split_decode_nvfp4kv.cu" \
        "${REPO_ROOT}/kernels/flash_attention_split_decode_nvfp4kv_bf16out.cu"
    sed -i "s/const FA2_THREADS: i32 = [0-9]\+;/const FA2_THREADS: i32 = ${val};/" \
        "${REPO_ROOT}/v3/crates/rvllm-attention/src/decode.rs"
    echo "[sweep] kernel rebuild for FA2_THREADS=${val}..."
    bash "${REPO_ROOT}/kernels/build.sh" sm_121 >/dev/null 2>&1
    ( cd "${REPO_ROOT}/v3" && cargo build --release --bin rvllm-server --features cuda,gb10 ) >/dev/null 2>&1
    echo "[sweep] kernel/server rebuild done"
}

apply_env_overrides() {
    local overrides="$1"
    [[ -z "$overrides" ]] && return
    for kv in $overrides; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        if [[ "$key" == __SOURCE_FA2_THREADS__ ]]; then
            apply_source_fa2_threads "$val"
            continue
        fi
        sed -i "/^${key}=/d" "$PROFILE"
        echo "${key}=${val}" >> "$PROFILE"
    done
}

disable_all_flags() {
    # Write an explicit zero/empty value for every NVFP4 toggle.
    # `apply_env_overrides` then re-sets only the keys the variant
    # declares; any flag the variant doesn't mention stays explicitly
    # disabled. Earlier `revert_to_baseline` only deleted keys, which
    # made the post-revert state depend on runtime defaults — at least
    # one of those defaults turned out to be "carry over what was set
    # last time" (HADAMARD=0 leaked from C into D). Explicit-write
    # here can't leak.
    for key in "${ALL_NVFP4_KEYS[@]}"; do
        sed -i "/^${key}=/d" "$PROFILE"
        # Policy keys want an empty string (kernels read empty as
        # "fall back to RVLLM_NVFP4_SCALE_POLICY"); numeric toggles
        # want an explicit 0.
        case "$key" in
            *POLICY*) echo "${key}=" >> "$PROFILE" ;;
            *)        echo "${key}=0" >> "$PROFILE" ;;
        esac
    done
    # Reset kernel-source FA2_THREADS to the production default.
    apply_source_fa2_threads 128
}

profile_snapshot() {
    grep -E '^RVLLM_NVFP4_(K|V)_SCALE_POLICY|^RVLLM_NVFP4_PARTITION_SIZE|^RVLLM_NVFP4_DECODE_GQA|^RVLLM_NVFP4_HADAMARD|^RVLLM_NVFP4_HADAMARD_V|^RVLLM_PER_TOKEN_Q_SCALE|^RVLLM_REPETITION_PENALTY=|^RVLLM_TOOL_CALL_OPEN_BIAS|^RVLLM_PREFILL_CHUNK_SIZE|^RVLLM_RESIDUAL_BF16|^RVLLM_DECODE_GRAPH' "$PROFILE" \
        2>/dev/null | sort | tr '\n' ' '
    echo "FA2_THREADS=$(grep -m1 '^#define FA2_THREADS' "${REPO_ROOT}/kernels/flash_attention_split_decode_nvfp4kv.cu" | awk '{print $3}')"
}

extract_blocks() {
    local prefix="$1"
    local logfile="$2"
    awk -v prefix="$prefix" '
        /^===/ {
            if (prefix_match && body != "") {
                gsub(/\n+$/, "", body)
                gsub(/\|/, "\\|", body)
                gsub(/\n/, " ", body)
                printf "| %d | %s | %s | %s |\n", n, verdict, prompt, body
                body = ""
            }
            prefix_match = 0
            if ($0 ~ "^===" prefix " \\[") {
                line = $0
                sub("^===" prefix " \\[", "", line)
                verdict = line
                sub(/\].*$/, "", verdict)
                prompt = line
                sub(/^[^]]*\] /, "", prompt)
                sub(/===$/, "", prompt)
                gsub(/\|/, "\\|", prompt)
                n++
                prefix_match = 1
                next
            }
        }
        prefix_match { body = body $0 "\n" }
        END {
            if (prefix_match && body != "") {
                gsub(/\n+$/, "", body)
                gsub(/\|/, "\\|", body)
                gsub(/\n/, " ", body)
                printf "| %d | %s | %s | %s |\n", n, verdict, prompt, body
            }
        }
    ' "$logfile"
}

direct_tokens_per_sec() {
    local since="$1"
    sudo journalctl -u rvllm-serve --since "$since" --no-pager 2>/dev/null \
        | grep -E "tok/s" | tail -8
}

{
    echo "# rvllm-serve variant sweep"
    echo
    echo "Started: $(date -Iseconds)"
    echo "Output: $REPORT"
    echo
    echo "## Rubric (strict)"
    echo
    echo "REGRESSION = repetition-guard abort, char/word/multi-token cycles, multi-line block repeat ≥3×, ≥4 consecutive Hangul/CJK in DE/EN context, mid-codepoint mojibake."
    echo
    echo "NOT REGRESSION = any coherent text in any language, including \"I don't know X\" / \"X not found in your data\" answers (after the default-scenario wipe Vinz/Lola don't exist; \"unbekannt\" answers are correct)."
    echo
} > "$REPORT"

for variant in "${VARIANTS_ORDER[@]}"; do
    overrides="${VARIANT_ENV[$variant]:-}"
    label="$variant"
    SMOKE_LOG="/tmp/sweep_smoke_${label}.log"
    echo "[sweep] === ${label} ==="
    echo "[sweep] overrides: ${overrides:-(none)}"

    disable_all_flags
    apply_env_overrides "$overrides"
    # Belt-and-suspenders: also stop the service here so the next
    # clean_smoke.sh start is guaranteed to read the freshly-rewritten
    # profile. Earlier we hit a leak where systemctl restart was
    # picking up env from a previous start despite the profile being
    # rewritten; a hard stop between variants forces a fresh exec.
    sudo systemctl stop rvllm-serve >/dev/null 2>&1 || true
    sleep 2

    sweep_t0=$(date +%s)
    SINCE_STAMP=$(date '+%Y-%m-%d %H:%M:%S')
    # Disable mid-run cold-restart in the harness — sweep semantics
    # are: ONE wipe + restart per variant, then all 30 ZC + 5 direct
    # prompts in one continuous session, then advance to the next
    # variant. The mid-run-restart-every-4 was a workaround for the
    # prefix-cache corruption bug; if it surfaces in this layout it
    # shows up as a real cycle-FAIL in the late prompts and we
    # capture it as a real result.
    PROMPTS_BEFORE_RESTART=0 \
        bash "${SCRIPT_DIR}/clean_smoke.sh" "$label" > "$SMOKE_LOG" 2>&1
    sweep_dur=$(( $(date +%s) - sweep_t0 ))

    harness_actual=$(ls -t /tmp/rvllm_smoke_${label}_*.log 2>/dev/null | head -1)

    zsumm=$(grep -m1 '^SUMMARY' "$harness_actual" 2>/dev/null || echo "no SUMMARY")
    dsumm=$(grep -m1 '^DIRECT_SUMMARY' "$harness_actual" 2>/dev/null || echo "no DIRECT_SUMMARY")

    {
        echo "## Variant ${label}"
        echo
        echo "**Wall:** ${sweep_dur}s   **Start:** ${SINCE_STAMP}"
        echo
        echo "**Profile snapshot:** $(profile_snapshot)"
        echo
        echo "**ZeroClaw:** ${zsumm}"
        echo "**Direct:** ${dsumm}"
        echo
        echo "### ZeroClaw stage (30 prompts via webhook)"
        echo
        echo "| # | Verdict | Prompt | Response |"
        echo "|---|---|---|---|"
        extract_blocks PROMPT "$harness_actual"
        echo
        echo "### Direct stage (5 prompts via /v1/chat/completions)"
        echo
        echo "| # | Verdict | Prompt | Response |"
        echo "|---|---|---|---|"
        extract_blocks DIRECT "$harness_actual"
        echo
        echo "### Decode tok/s (last generates during this variant)"
        echo '```'
        direct_tokens_per_sec "$SINCE_STAMP"
        echo '```'
        echo
        echo "Harness log: \`$harness_actual\`"
        echo "Sweep stdout: \`$SMOKE_LOG\`"
        echo
    } >> "$REPORT"

    echo "[sweep] ${label}: ${zsumm} / ${dsumm} (wall ${sweep_dur}s)"
done

disable_all_flags

{
    echo "## Summary table"
    echo
    echo "| Variant | ZeroClaw | Direct | Wall |"
    echo "|---|---|---|---|"
    for variant in "${VARIANTS_ORDER[@]}"; do
        harness_actual=$(ls -t /tmp/rvllm_smoke_${variant}_*.log 2>/dev/null | head -1)
        zsumm=$(grep -m1 '^SUMMARY' "$harness_actual" 2>/dev/null | sed 's/^SUMMARY [^:]*: //')
        dsumm=$(grep -m1 '^DIRECT_SUMMARY' "$harness_actual" 2>/dev/null | sed 's/^DIRECT_SUMMARY [^:]*: //')
        echo "| ${variant} | ${zsumm:-?} | ${dsumm:-?} | — |"
    done
    echo
    echo "Done: $(date -Iseconds)"
} >> "$REPORT"

echo "[sweep] DONE — report: $REPORT"
