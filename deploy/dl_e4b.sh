#!/bin/bash
# Download E4B FP8 checkpoints (priority: leon-se Dynamic). Try multiple sources.
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /root/models

echo '=== Try leon-se/gemma-4-E4B-it-FP8-Dynamic ==='
python3 - << 'PY' 2>&1 | tee /tmp/dl_e4b_leon.log
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(
        'leon-se/gemma-4-E4B-it-FP8-Dynamic',
        local_dir='/root/models/gemma-4-E4B-it-FP8',
        max_workers=8,
    )
    print('OK', p)
except Exception as e:
    print('FAIL', type(e).__name__, str(e)[:300])
PY

if [ -f /root/models/gemma-4-E4B-it-FP8/config.json ]; then
    echo '=== E4B FP8 ready ==='
    du -sh /root/models/gemma-4-E4B-it-FP8
else
    echo '=== Fallback: bf16 google/gemma-4-E4B-it ==='
    python3 - << 'PY' 2>&1 | tee /tmp/dl_e4b_bf16.log
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(
        'google/gemma-4-E4B-it',
        local_dir='/root/models/gemma-4-E4B-it-BF16',
        max_workers=8,
    )
    print('OK', p)
except Exception as e:
    print('FAIL', type(e).__name__, str(e)[:300])
PY
fi
