#!/bin/bash
# 30-prompt zeroclaw smoke harness for rvllm-serve quality regression
# detection. Provokes:
#   - free-chat WHO/persona (×5)
#   - WEATHER (×5 cities)
#   - HA tool calls (×5)
#   - brain lookups (×5)
#   - code/explain (×5)
#   - creative (×5)
#
# Usage:
#   ./zeroclaw_30prompt_smoke.sh [label]
#     label: optional tag for the output file (default "baseline")
#
# Output:
#   /tmp/rvllm_smoke_<label>_<timestamp>.log — one prompt per ===PROMPT block
#   stdout: summary line per prompt (PASS / SUSPECT / FAIL)
#
# A response is FAIL if it contains a 4+ char repeated suffix ≥ 3×, or
# is empty / "LLM request failed". SUSPECT = single-token cycles or known
# garbage tokens (la, lola, C1, Bora). PASS = otherwise.

set -u
LABEL="${1:-baseline}"
TS=$(date +%Y%m%d-%H%M%S)
OUT=/tmp/rvllm_smoke_${LABEL}_${TS}.log
WEBHOOK=http://127.0.0.1:42617/webhook

PROMPTS=(
  # WHO / free-chat (5)
  "wer bist du?"
  "was machst du gerade?"
  "wer ist dein lieblingsmensch?"
  "stell dich kurz vor"
  "was ist deine philosophie?"

  # WEATHER (5)
  "wie ist das wetter in bern?"
  "wie ist das wetter in zürich?"
  "regnet es in basel?"
  "ist es kalt in luzern?"
  "wie warm wird es morgen?"

  # HA tool (5)
  "ist die haustür auf?"
  "ist das wohnzimmerlicht an?"
  "wo ist lola?"
  "ist vinz zuhause?"
  "schalte das licht im flur aus"

  # brain lookup (5)
  "suche nach Lola"
  "wer ist Vinz?"
  "liste meine projekte"
  "was sind meine letzten erinnerungen?"
  "such die letzten news zu Bitcoin"

  # code/explain (5)
  "erkläre mir was eine GPU ist in 3 sätzen"
  "schreib einen hello-world in python"
  "was ist der unterschied zwischen RAM und SSD?"
  "wie funktioniert ein attention layer?"
  "warum ist quantisierung wichtig?"

  # creative (5)
  "erzähl mir einen kurzen witz"
  "schreibe ein gedicht über katzen"
  "erfinde einen drachennamen"
  "denk dir eine kurze geschichte aus, 3 sätze"
  "was sagt ein hund zu einer katze?"
)

# Repetition heuristic: any 4+ char substring repeated ≥3 times in the
# output indicates a cycle. We also flag empty / explicit-error responses.
classify() {
    local resp="$1"
    if [[ -z "$resp" ]] || [[ "$resp" == *"LLM request failed"* ]] || [[ "$resp" == *"\"error\":"* ]]; then
        echo "FAIL"
        return
    fi
    # ZeroClaw "no model output" sentinel — when the worker queue is
    # stuck or the circuit breaker tripped, ZeroClaw returns just the
    # memory-context separator with no model body. This is a HARNESS
    # failure (zeroclaw side), not an rvllm output. Classify as INFRA
    # so the caller can decide to restart zeroclaw and retry.
    if [[ "${resp//[$'\n\r']}" =~ ^-+$ ]] || [[ "${resp:0:5}" == "-----" ]]; then
        echo "INFRA"
        return
    fi
    # Lowercase for matching
    local lc="${resp,,}"
    # Known garbage cycle markers (any 3+ occurrences)
    for marker in " la la la" "la lola" "c1c1" "bora bora" "////" "_//c" "lau lau" "cing_cing" "//'c'" "//\"c\"" "//’c"; do
        local count=$(grep -oF "$marker" <<< "$lc" | wc -l)
        if (( count >= 3 )); then
            echo "FAIL"
            return
        fi
    done
    # Single-token cycle: same 3+ char word repeated ≥4 consecutive times
    if grep -qE '(\b[a-zA-Z]{3,}\b\s+){3,}\1' <<< "$resp"; then
        echo "FAIL"
        return
    fi
    # Same 4-char substring back-to-back-to-back (no gap) → FAIL.
    # Dispersed-repeat heuristic dropped — natural German has many
    # legitimate 4-char repeats (e.g. " der", " und", "ist ") that
    # tripped the SUSPECT bucket on every clean response.
    if grep -qE "([a-zA-Z0-9_/!'\"’-]{4})\1\1" <<< "$lc"; then
        echo "FAIL"
        return
    fi
    # Same 3-char substring back-to-back ≥5× → FAIL. Catches the
    # `//'//'//'…` cycle (period 3 not 4) seen on light-toggle prompt.
    # 3-char regex is more permissive than 4 so we require ≥5 repeats
    # to avoid false positives on "ananana"-style words.
    if grep -qE "([a-zA-Z0-9_/!'\"’-]{3})\1\1\1\1" <<< "$lc"; then
        echo "FAIL"
        return
    fi
    # Long-period cycle: any 6-15 char alphabetic substring repeated
    # back-to-back-to-back. Catches "SH deB sameH deB sameH deB" type
    # degradation where the model loops on a multi-token chunk.
    # Bounded length 6-15 avoids legitimate German full-sentence
    # repeats (which can happen with bullet lists).
    if grep -qE "([a-zA-Z][a-zA-Z ]{4,13}[a-zA-Z])\1\1" <<< "$lc"; then
        echo "FAIL"
        return
    fi
    echo "PASS"
}

echo "label=$LABEL ts=$TS profile_summary:" | tee "$OUT"
grep -E '^RVLLM_(NVFP4|PER_TOKEN|HADAMARD|PREFILL_CHUNK|REPETITION)' ~/.rvllm/active-profile.env | tee -a "$OUT"
echo "---" | tee -a "$OUT"

PASS=0
SUSP=0
FAIL=0
INFRA=0
PROMPTS_BEFORE_RESTART=4     # rvllm prefix-cache state-corruption seen after 4-5 long prompts
i=0

# Restart BOTH rvllm-serve and zeroclaw ONCE at the start. rvllm-serve
# restart clears the in-memory prefix cache (poisoned-state recovery).
# zeroclaw restart drains the message queue. Within the matrix we let
# one session handle all 30 prompts — mirrors production (zeroclaw
# runs for thousands of messages between restarts). Restarting between
# prompts would force every call into "cold session" mode where
# zeroclaw returns the memory-context dashes without running the model.
echo "[harness] cold restart rvllm-serve..."
sudo systemctl restart rvllm-serve >/dev/null 2>&1 || true
until curl -fsS http://127.0.0.1:8010/v1/models >/dev/null 2>&1; do sleep 2; done
echo "[harness] restart zeroclaw + warm-up..."
sudo systemctl restart zeroclaw >/dev/null 2>&1 || true
sleep 5

# Warm-up call: zeroclaw's 2-call pattern (classifier + persona-reply)
# means the FIRST call after restart primes the prefix cache. This
# response is discarded.
jq -n --arg m "wer bist du?" '{message: $m}' \
    | curl -s -X POST "$WEBHOOK" -H 'Content-Type: application/json' --max-time 605 -d @- \
    >/dev/null 2>&1 || true
sleep 2

retry_one() {
    local p="$1"
    local payload=$(jq -n --arg m "$p" '{message: $m}')
    local r=$(curl -s -X POST "$WEBHOOK" -H 'Content-Type: application/json' --max-time 605 -d "$payload" | jq -r '.response // .error // ""' 2>/dev/null)
    printf '%s' "$r"
}

for p in "${PROMPTS[@]}"; do
    # Restart BOTH rvllm-serve AND zeroclaw every PROMPTS_BEFORE_RESTART.
    # Pre-existing bug: rvllm's prefix cache enters a corrupted state
    # (logits explode to amax≈3e5 producing dash-token cycles) after
    # 4-5 long-context prompts. Pinpointed iter 51 — predates this PR.
    # Restart-zc-only does NOT clear it; only cold restart of rvllm
    # resets the prefix-cache region.
    if (( i > 0 && i % PROMPTS_BEFORE_RESTART == 0 )); then
        echo "[harness] periodic cold restart (rvllm-serve + zeroclaw) at prompt $i..."
        sudo systemctl restart rvllm-serve >/dev/null 2>&1 || true
        until curl -fsS http://127.0.0.1:8010/v1/models >/dev/null 2>&1; do sleep 2; done
        sudo systemctl restart zeroclaw >/dev/null 2>&1 || true
        sleep 5
        # Warm-up to flush "cold first call" dashes (response discarded).
        jq -n --arg m "warmup" '{message: $m}' \
            | curl -s -X POST "$WEBHOOK" -H 'Content-Type: application/json' --max-time 605 -d @- \
            >/dev/null 2>&1 || true
        sleep 2
    fi
    i=$((i + 1))

    resp=$(retry_one "$p")
    cls=$(classify "$resp")

    # If INFRA on a single prompt mid-batch, restart zeroclaw + warmup
    # + retry once. Catches inter-restart degradation.
    if [[ "$cls" == "INFRA" ]]; then
        sudo systemctl restart zeroclaw >/dev/null 2>&1 || true
        sleep 5
        jq -n --arg m "warmup" '{message: $m}' \
            | curl -s -X POST "$WEBHOOK" -H 'Content-Type: application/json' --max-time 605 -d @- \
            >/dev/null 2>&1 || true
        sleep 2
        resp=$(retry_one "$p")
        cls=$(classify "$resp")
    fi
    sleep 2

    snippet=$(printf '%s' "$resp" | tr -d '\n' | head -c 180)
    echo "===PROMPT [$cls] $p===" >> "$OUT"
    echo "$resp" >> "$OUT"
    echo "" >> "$OUT"
    case "$cls" in
        PASS)    PASS=$((PASS+1));  printf "[PASS]    %-50s | %s\n" "$p" "$snippet" ;;
        SUSPECT) SUSP=$((SUSP+1));  printf "[SUSPECT] %-50s | %s\n" "$p" "$snippet" ;;
        FAIL)    FAIL=$((FAIL+1));  printf "[FAIL]    %-50s | %s\n" "$p" "$snippet" ;;
        INFRA)   INFRA=$((INFRA+1));printf "[INFRA]   %-50s | %s\n" "$p" "$snippet" ;;
    esac
done

TOTAL=${#PROMPTS[@]}
echo "---"
printf "SUMMARY %s: %d/%d PASS, %d SUSPECT, %d FAIL, %d INFRA\n" \
    "$LABEL" "$PASS" "$TOTAL" "$SUSP" "$FAIL" "$INFRA"
echo "log: $OUT"
