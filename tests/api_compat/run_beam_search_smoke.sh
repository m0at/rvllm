#!/bin/bash
set -euo pipefail

echo "Running beam-search smoke tests against ${RVLLM_URL:-http://localhost:8000}"
pip install -q requests pytest
RVLLM_URL="${RVLLM_URL:-http://localhost:8000}" \
  python3 -m pytest tests/api_compat/test_openai_client.py -k beam_search -v --tb=short
