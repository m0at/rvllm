#!/bin/bash
set -euo pipefail

# Launch Kimi K2.6 on the vast A40 box via KTransformers + SGLang.
#
# Example:
#   bash k2/kt_a40_launch.sh
#   KT_NUM_GPU_EXPERTS=8 MEM_FRACTION_STATIC=0.92 bash k2/kt_a40_launch.sh

SSH_ADDR="${SSH_ADDR:-ssh4.vast.ai}"
SSH_PORT="${SSH_PORT:-38164}"
REMOTE_PORT="${REMOTE_PORT:-31245}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Kimi-K2.6}"
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-/workspace/models/Kimi-K2.6-GGUF-Q4_X/Q4_X}"
LOG_PATH="${LOG_PATH:-/workspace/k2.6_kt_sglang.log}"
PID_PATH="${PID_PATH:-/workspace/k2.6_kt_sglang.pid}"

KT_CPUINFER="${KT_CPUINFER:-128}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-2}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-0}"
KT_METHOD="${KT_METHOD:-LLAMAFILE}"
KT_GPU_PREFILL_TOKEN_THRESHOLD="${KT_GPU_PREFILL_TOKEN_THRESHOLD:-256}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K2.6}"

SSH="ssh -o StrictHostKeyChecking=no -p ${SSH_PORT} root@${SSH_ADDR}"

${SSH} "bash -lc '
set -euo pipefail

if [ -f \"${PID_PATH}\" ]; then
  old_pid=\$(cat \"${PID_PATH}\" || true)
  if [ -n \"\${old_pid}\" ] && ps -p \"\${old_pid}\" >/dev/null 2>&1; then
    kill \"\${old_pid}\" || true
    sleep 2
  fi
fi

nohup python3 -m sglang.launch_server \
  --host 0.0.0.0 \
  --port ${REMOTE_PORT} \
  --model ${MODEL_DIR} \
  --kt-weight-path ${KT_WEIGHT_PATH} \
  --kt-cpuinfer ${KT_CPUINFER} \
  --kt-threadpool-count ${KT_THREADPOOL_COUNT} \
  --kt-num-gpu-experts ${KT_NUM_GPU_EXPERTS} \
  --kt-method ${KT_METHOD} \
  --kt-gpu-prefill-token-threshold ${KT_GPU_PREFILL_TOKEN_THRESHOLD} \
  --trust-remote-code \
  --mem-fraction-static ${MEM_FRACTION_STATIC} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion \
  --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
  --max-running-requests ${MAX_RUNNING_REQUESTS} \
  --max-total-tokens ${MAX_TOTAL_TOKENS} \
  --attention-backend ${ATTENTION_BACKEND} \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  > ${LOG_PATH} 2>&1 &

echo \$! > ${PID_PATH}
echo PID=\$(cat ${PID_PATH})
echo LOG=${LOG_PATH}
'"
