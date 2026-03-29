#!/bin/bash
set -euo pipefail

RVLLM_URL="${RVLLM_URL:-http://localhost:8000}"
if [[ -z "${RVLLM_MODEL:-}" ]]; then
  RVLLM_MODEL="$(
    curl -fsS "${RVLLM_URL}/v1/models" | python3 -c '
import json
import sys

data = json.load(sys.stdin).get("data", [])
if not data:
    raise SystemExit("No models returned by /v1/models")
print(data[0]["id"])
'
  )"
fi

echo "Running beam-search smoke tests against ${RVLLM_URL} with model ${RVLLM_MODEL}"
python3 - <<'PY'
import importlib.util
import sys

missing = [name for name in ("pytest", "requests") if importlib.util.find_spec(name) is None]
if missing:
    sys.stderr.write(
        "Missing Python dependencies for beam smoke test: "
        + ", ".join(missing)
        + "\nInstall them in your active environment and retry.\n"
    )
    sys.exit(1)
PY
RVLLM_URL="${RVLLM_URL}" RVLLM_MODEL="${RVLLM_MODEL}" \
  python3 -m pytest tests/api_compat/test_openai_client.py -k beam_search -v --tb=short
