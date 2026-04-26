#!/usr/bin/env bash
# Tool-call failure probe — pure-NVFP4 profile, weather query, multi-config.
#
# Tests: K=amax6 V=mse [HADAMARD on/off] × [hybrid 0/1] = 4 cells.
# Each cell fires WEATHER 3x via webhook to establish reproducibility.
# Webhook channel doesn't trigger zeroclaw chat-history storage, so brain
# stays clean across runs.
set -uo pipefail

PROFILE=/home/r00t/.rvllm/active-profile.env
LOG=/tmp/weather_probe.$(date +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

ORIGINAL=$(mktemp)
cp "$PROFILE" "$ORIGINAL"

cleanup() {
  echo
  echo "=== restoring original profile ==="
  cp "$ORIGINAL" "$PROFILE"
  rm -f "$ORIGINAL"
  sudo systemctl restart rvllm-serve
  echo "(rvllm-serve restarted on original profile; log: $LOG)"
}
trap cleanup EXIT

write_cell_env() {
  local k_pol=$1 v_pol=$2 hybrid=$3 hadamard=$4
  sed -i \
    -e '/^RVLLM_NVFP4_K_SCALE_POLICY=/d' \
    -e '/^RVLLM_NVFP4_V_SCALE_POLICY=/d' \
    -e '/^RVLLM_NVFP4_HYBRID_GLOBAL_FP8=/d' \
    -e '/^RVLLM_NVFP4_HADAMARD=/d' \
    "$PROFILE"
  cat >>"$PROFILE" <<EOF
RVLLM_NVFP4_HYBRID_GLOBAL_FP8=$hybrid
RVLLM_NVFP4_K_SCALE_POLICY=$k_pol
RVLLM_NVFP4_V_SCALE_POLICY=$v_pol
RVLLM_NVFP4_HADAMARD=$hadamard
EOF
}

wait_ready() {
  until curl -fsS -m 3 http://127.0.0.1:8010/v1/models -o /dev/null 2>/dev/null; do
    sleep 5
  done
  until curl -fsS -m 3 http://127.0.0.1:42617/health -o /dev/null 2>/dev/null; do
    sleep 2
  done
}

fire() {
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
  local label=$1 k_pol=$2 v_pol=$3 hybrid=$4 hadamard=$5
  echo
  echo "============================================================"
  echo "CELL: $label  (K=$k_pol V=$v_pol hybrid=$hybrid hadamard=$hadamard)"
  echo "============================================================"
  write_cell_env "$k_pol" "$v_pol" "$hybrid" "$hadamard"

  echo "[$(date +%H:%M:%S)] restarting rvllm-serve..."
  sudo systemctl restart rvllm-serve
  wait_ready
  echo "[$(date +%H:%M:%S)] ready"
  echo "--- effective env ---"
  cat /proc/"$(pgrep -f /home/r00t/.rvllm/bin/rvllm-server | head -1)"/environ \
    | tr '\0' '\n' \
    | grep -iE 'K_SCALE_POLICY|V_SCALE_POLICY|HYBRID|HADAMARD|PER_TOKEN'
  echo

  for i in 1 2 3; do
    echo "[$(date +%H:%M:%S)] WEATHER fire #$i:"
    local resp
    resp=$(fire 'wie ist das wetter in bern?')
    # Score garbage: la-la-la or Hangul/Japanese chars
    local has_garbage
    has_garbage=$(printf '%s' "$resp" | python3 -c "
import sys
t = sys.stdin.read()
hangul = sum(1 for c in t if 0xAC00 <= ord(c) <= 0xD7AF)
cjk = sum(1 for c in t if 0x4E00 <= ord(c) <= 0x9FFF)
hira = sum(1 for c in t if 0x3040 <= ord(c) <= 0x309F)
kata = sum(1 for c in t if 0x30A0 <= ord(c) <= 0x30FF)
print('YES' if (t.count('la la la') > 1 or hangul + cjk + hira + kata > 5) else 'NO')
")
    echo "  garbage=$has_garbage  resp=${resp:0:200}"
    printf '%s\t%d\t%s\t%s\n' "$label" "$i" "$has_garbage" "${resp:0:120}" \
      >> /tmp/weather_probe.summary
  done
}

: > /tmp/weather_probe.summary

echo "weather probe log: $LOG"
echo "started:           $(date)"

# Cells: vary HADAMARD and HYBRID; K=amax6 V=mse stays (matrix winner).
run_cell "1-hadamard-on-hybrid-off"  amax6 mse 0 1
run_cell "2-hadamard-off-hybrid-off" amax6 mse 0 0
run_cell "3-hadamard-on-hybrid-on"   amax6 mse 1 1

echo
echo "============================================================"
echo "WEATHER PROBE SUMMARY"
echo "============================================================"
printf '%-32s  %-3s  %-7s  %s\n' label fire garbage resp
while IFS=$'\t' read -r label fire garbage resp; do
  printf '%-32s  %-3s  %-7s  %s\n' "$label" "$fire" "$garbage" "$resp"
done < /tmp/weather_probe.summary
echo "log: $LOG"
