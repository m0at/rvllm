#!/usr/bin/env bash
# K/V scale-policy matrix tester for NVFP4 KV cache.
#
# Per cell: rewrite ~/.rvllm/active-profile.env with the cell's env,
# restart rvllm-serve + zeroclaw to clear prefix-cache + message queue,
# fire "wer bist du?" (warm-up + classifier-then-real KV state), then
# fire "wie ist das wetter in bern?" (the failure trigger).  Print both
# responses with the cell label so we can read the matrix top-to-bottom.
#
# The profile file is preserved except for the K/V/HYBRID gates the
# script overwrites.  Restored to the user's chosen baseline at the end.
set -uo pipefail

PROFILE=/home/r00t/.rvllm/active-profile.env
LOG=/tmp/kv_policy_matrix.$(date +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

WHO_PROMPT='wer bist du?'
WEATHER_PROMPT='wie ist das wetter in bern?'

# Stash the operator's profile so we can put it back at the end.
ORIGINAL=$(mktemp)
cp "$PROFILE" "$ORIGINAL"

cleanup() {
  echo
  echo "=== restoring original profile ==="
  cp "$ORIGINAL" "$PROFILE"
  rm -f "$ORIGINAL"
  sudo systemctl restart rvllm-serve zeroclaw
  echo "(services restarted on original profile; matrix log: $LOG)"
}
trap cleanup EXIT

write_cell_env() {
  # Rewrite the K/V/HYBRID lines in-place; leave everything else alone.
  local k_pol=$1 v_pol=$2 hybrid=$3
  sed -i \
    -e '/^RVLLM_NVFP4_K_SCALE_POLICY=/d' \
    -e '/^RVLLM_NVFP4_V_SCALE_POLICY=/d' \
    -e '/^RVLLM_NVFP4_HYBRID_GLOBAL_FP8=/d' \
    "$PROFILE"
  cat >>"$PROFILE" <<EOF
RVLLM_NVFP4_HYBRID_GLOBAL_FP8=$hybrid
RVLLM_NVFP4_K_SCALE_POLICY=$k_pol
RVLLM_NVFP4_V_SCALE_POLICY=$v_pol
EOF
}

wait_rvllm_ready() {
  # Block until /v1/models returns 200 (CUDA worker fully loaded).
  until curl -fsS -m 3 http://127.0.0.1:8010/v1/models -o /dev/null 2>/dev/null; do
    sleep 5
  done
}

wait_zeroclaw_ready() {
  until curl -fsS -m 3 http://127.0.0.1:42617/health -o /dev/null 2>/dev/null; do
    sleep 2
  done
}

fire_webhook() {
  # Fire a single webhook prompt; print the model's response text only.
  local prompt=$1
  local payload
  payload=$(jq -nc --arg m "$prompt" '{message:$m}')
  curl -s -X POST http://127.0.0.1:42617/webhook \
    -H 'Content-Type: application/json' \
    --max-time 600 \
    -d "$payload" \
    | jq -r '.response // .error // "<no response field>"'
}

run_cell() {
  local label=$1 k_pol=$2 v_pol=$3 hybrid=$4
  echo
  echo "============================================================"
  echo "CELL: $label  (K=$k_pol, V=$v_pol, HYBRID_GLOBAL_FP8=$hybrid)"
  echo "============================================================"
  write_cell_env "$k_pol" "$v_pol" "$hybrid"

  echo "[$(date +%H:%M:%S)] restarting rvllm-serve..."
  sudo systemctl restart rvllm-serve
  wait_rvllm_ready
  echo "[$(date +%H:%M:%S)] rvllm-serve ready"

  echo "[$(date +%H:%M:%S)] restarting zeroclaw..."
  sudo systemctl restart zeroclaw
  wait_zeroclaw_ready
  echo "[$(date +%H:%M:%S)] zeroclaw ready"

  # Verify env actually loaded (catches inline-comment poisoning etc).
  echo "--- effective env ---"
  cat /proc/"$(pgrep -f /home/r00t/.rvllm/bin/rvllm-server | head -1)"/environ \
    | tr '\0' '\n' \
    | grep -iE 'K_SCALE_POLICY|V_SCALE_POLICY|HYBRID|HADAMARD|PER_TOKEN|^RVLLM_Q_SCALE='
  echo

  # Mark journal cursor so we can scrape rvllm-serve's per-call logs for
  # this cell only. ZeroClaw fires TWO rvllm calls per webhook turn — a
  # 3633-token classifier (stream=false, expected output "REPLY"|"NO_REPLY")
  # and a ~16k-token real reply (stream=true). The webhook's own
  # `.response` field shows only the real reply text, so a "classifier
  # poisoned the KV but real reply masked it" failure is invisible from
  # the webhook alone — read the journal too.
  local journal_since
  journal_since=$(date '+%Y-%m-%d %H:%M:%S')

  echo "[$(date +%H:%M:%S)] firing WHO: $WHO_PROMPT"
  local who_resp
  who_resp=$(fire_webhook "$WHO_PROMPT")
  echo "WHO_RESPONSE: $who_resp"

  echo "[$(date +%H:%M:%S)] firing WEATHER: $WEATHER_PROMPT"
  local weather_resp
  weather_resp=$(fire_webhook "$WEATHER_PROMPT")
  echo "WEATHER_RESPONSE: $weather_resp"

  # Pull every rvllm-serve [generate] / prompt_tokens / Reply line emitted
  # since the journal cursor — gives a per-call audit trail (token counts,
  # decode latency, classifier outputs that aren't in the webhook reply).
  echo "--- rvllm-serve calls this cell ---"
  journalctl -u rvllm-serve --since "$journal_since" --no-pager 2>&1 \
    | grep -E 'prompt_tokens=|tokens decoded in|prefix-cache' \
    | sed 's/^.*rvllm-server\[[0-9]*\]: //' \
    | tail -30 || true
  echo

  # Stash a one-line summary into a tmp file the trailer reads back.
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$k_pol" "$v_pol" "$hybrid" \
    "${who_resp:0:60}" "${weather_resp:0:80}" \
    >> /tmp/kv_policy_matrix.summary
}

# Reset the summary file for this run.
: > /tmp/kv_policy_matrix.summary

echo "matrix log: $LOG"
echo "started:    $(date)"
echo

# === The matrix ===
# Column convention: label, K_SCALE, V_SCALE, HYBRID_GLOBAL_FP8.
# All cells: HADAMARD=1, PER_TOKEN_Q_SCALE=1, Q_SCALE=2.0 (set in
# the original profile, untouched here).
run_cell "1-baseline-mse-mse"            mse   mse   0
run_cell "2-K-amax6-V-mse"               amax6 mse   0
run_cell "3-K-mse-V-amax6"               mse   amax6 0
run_cell "4-K-amax6-V-amax6"             amax6 amax6 0
run_cell "5-hybrid-global-fp8-mse-mse"   mse   mse   1

echo
echo "============================================================"
echo "MATRIX SUMMARY"
echo "============================================================"
printf '%-32s  %-6s  %-6s  %-6s  %-30s  %s\n' label K V HYBR who weather
while IFS=$'\t' read -r label k v hybr who weather; do
  printf '%-32s  %-6s  %-6s  %-6s  %-30s  %s\n' "$label" "$k" "$v" "$hybr" "$who" "$weather"
done < /tmp/kv_policy_matrix.summary
echo
echo "full log: $LOG"
