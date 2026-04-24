#!/bin/bash
# Wait for vast.ai instance to reach 'running' status.
# Usage: ./wait_instance.sh <instance_id> [max_attempts] [sleep_sec]
set -u
INSTANCE_ID="${1:?instance id required}"
MAX_ATTEMPTS="${2:-20}"
SLEEP_SEC="${3:-15}"

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  RAW=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)
  STATUS=$(printf '%s' "$RAW" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get("actual_status","?"), "|", d.get("status_msg","")[:80] if d.get("status_msg") else "", "|", "ssh:", d.get("ssh_host","?"), d.get("ssh_port","?"))
except Exception as e:
    print("parse_error:", e)' 2>/dev/null)
  echo "[$i/$MAX_ATTEMPTS] $STATUS"
  if printf '%s' "$STATUS" | grep -q '^running'; then
    echo 'READY'
    exit 0
  fi
  sleep "$SLEEP_SEC"
done
echo 'TIMEOUT'
exit 1
