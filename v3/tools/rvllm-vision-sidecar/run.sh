#!/usr/bin/env bash
# Start rvllm-vision-sidecar for a single model (qwen3_vl or gemma4_mm).
# Usage:
#   ./run.sh                    # default qwen3_vl
#   ./run.sh gemma4_mm
#   VISION_GPU=1 ./run.sh       # use GPU 1 (off-device from rvllm-serve)
#   PYTHON=/path/to/python ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-qwen3_vl}"
PORT="${PORT:-8765}"
PYTHON="${PYTHON:-/home/r00t/.venv/bin/python3}"
export RVLLM_VISION_MODEL="$MODEL"
# Optional GPU pin (default: same GPU as rvllm-serve, contention OK for tests).
if [[ -n "${VISION_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$VISION_GPU"
fi
echo "[run.sh] model=$MODEL port=$PORT python=$PYTHON" >&2
exec "$PYTHON" -m uvicorn main:app --host 127.0.0.1 --port "$PORT" --workers 1
