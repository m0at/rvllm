#!/bin/bash
# Run API compatibility tests against both servers
set -euo pipefail
echo "Testing rvllm API compatibility..."
echo "Server: ${RVLLM_URL:-http://localhost:8000}"
pip install -q requests pytest
python3 -m pytest tests/api_compat/ -v --tb=short
