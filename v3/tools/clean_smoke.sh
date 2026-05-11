#!/bin/bash
# Automated end-to-end clean-smoke runner for rvllm-serve quality
# regression testing.
#
# Usage:
#   ./clean_smoke.sh [label]
#     label: optional tag for the output file (default "clean")
#
# Steps (top-to-bottom, no intervention):
#   1) Stop zeroclaw, then rvllm-serve. Drain GPU.
#   2) HARD-WIPE every record in the `default` brain scenario. Cycle 60
#      confirmed: chat-history alone wasn't enough; diary entries and
#      notes also captured prior model garbage and got retrieved as
#      [Memory context], reinforcing poison patterns. We wipe ALL
#      scenario-tagged tables (memory/entity/task/bookmark/project/
#      tool/event/location/credential/snippet/routine/media/linked/
#      reminder) so the test starts from a guaranteed-empty retrieval
#      state. The scenario record itself is preserved.
#   3) Start rvllm-serve. Wait for /v1/models → 200.
#   4) Start zeroclaw. Wait 5 s for HTTP listener.
#   5) Discard one warm-up call (zeroclaw 2-call classifier+reply
#      pattern primes prefix cache; the first call is unrepresentative).
#   6) Re-wipe (warmup may have stored a chat-history turn).
#   7) Invoke v3/tools/zeroclaw_30prompt_smoke.sh — by default cold-
#      restarts both services every 4 prompts and re-wipes between
#      each so within-smoke poison can't accumulate. Caller can set
#      PROMPTS_BEFORE_RESTART=0 to disable mid-run restart and run
#      all 30 prompts as one continuous session (sweep_variants.sh
#      uses this for single-session-per-variant semantics).
#
# Requirements:
#   - sudo NOPASSWD on systemctl stop|start|restart for rvllm-serve
#     and zeroclaw (already true on this host).
#   - brain CLI (/usr/bin/brain) installed and DB up.

set -u
LABEL="${1:-clean}"
TS=$(date +%Y%m%d-%H%M%S)
OUT=/tmp/rvllm_smoke_${LABEL}_${TS}.log

echo "[clean-smoke] $TS  label=$LABEL"

# 1) Stop services in reverse dependency order.
echo "[clean-smoke] stopping zeroclaw + rvllm-serve..."
sudo systemctl stop zeroclaw    >/dev/null 2>&1 || true
sudo systemctl stop rvllm-serve >/dev/null 2>&1 || true
sleep 3

# 2) Wipe default scenario (all tables).
wipe_default_scenario() {
    bash "$(dirname "$0")/wipe_default_scenario.sh" \
        | sed 's/^/[clean-smoke] /'
}

# Vanilla zeroclaw stores conversation history + sessions in local
# SQLite files inside the workspace dir. The brain-scenario wipe
# above does NOT touch those — they are vanilla-zeroclaw state, not
# brain state. Delete them while services are stopped so the next
# variant starts with a fresh memory store. Safe because sqlite
# files have no readers when zeroclaw is stopped.
wipe_zeroclaw_local_state() {
    local ws=/home/r00t/workspace/data/zeroclaw/workspace
    local count=0
    for f in "$ws/memory/brain.db" "$ws/memory/brain.db-shm" "$ws/memory/brain.db-wal" \
             "$ws/sessions/sessions.db" "$ws/sessions/sessions.db-shm" "$ws/sessions/sessions.db-wal"; do
        if [[ -f "$f" ]]; then
            rm -f "$f" && count=$((count+1))
        fi
    done
    echo "[clean-smoke] wiped $count zeroclaw sqlite files (memory/brain.db + sessions/sessions.db)"
}

wipe_default_scenario
wipe_zeroclaw_local_state

# 3) Start rvllm-serve, wait for models endpoint.
echo "[clean-smoke] starting rvllm-serve..."
sudo systemctl start rvllm-serve >/dev/null 2>&1
deadline=$(( $(date +%s) + 180 ))
until curl -fsS http://127.0.0.1:8010/v1/models >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
        echo "[clean-smoke] FATAL: rvllm-serve did not boot within 180 s"
        exit 1
    fi
    sleep 2
done
echo "[clean-smoke] rvllm-serve ready"

# 4) Start zeroclaw.
echo "[clean-smoke] starting zeroclaw..."
sudo systemctl start zeroclaw >/dev/null 2>&1
sleep 5

# 5) Warmup discard.
echo "[clean-smoke] warmup call (discarded)..."
jq -n --arg m "wer bist du?" '{message: $m}' \
    | curl -s -X POST http://127.0.0.1:42617/webhook \
        -H 'Content-Type: application/json' --max-time 605 -d @- \
        >/dev/null 2>&1 || true
sleep 2

# Re-wipe brain after warmup (zeroclaw may have stored the warmup turn).
# Note: zeroclaw's sqlite memory is NOT wiped here — it stays so the
# 30+5 prompts inside the smoke run see realistic multi-turn context
# accumulation. Sqlite is only wiped between variants (in stop phase).
wipe_default_scenario

# 6) Run the 30-prompt smoke. The smoke itself cold-restarts both
# services every 4 prompts; we wrap it so we re-purge between every
# restart by setting an env var the smoke checks (or, simpler, we
# re-purge AFTER the smoke completes and rely on the cold-restart
# loop to surface any single-prompt regressions).
echo "[clean-smoke] starting 30-prompt smoke..."
export PURGE_CHAT_HISTORY_CMD="$(realpath "$0")::purge"
bash "$(dirname "$0")/zeroclaw_30prompt_smoke.sh" "$LABEL" 2>&1 | tee -a "$OUT"

# Final wipe so the next clean run starts truly clean.
wipe_default_scenario
# zeroclaw sqlite files stay until the NEXT variant's stop+wipe phase
# triggers wipe_zeroclaw_local_state.
echo "[clean-smoke] DONE  log=$OUT"
