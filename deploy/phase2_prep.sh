#!/bin/bash
# Prep step for Phase 2 on the H100 box:
# - clone CUTLASS + flash-attention
# - install huggingface_hub
# - start downloading 31B FP8-Dynamic model in background (it's ~33GB)
set -euo pipefail

echo '=== Cloning CUTLASS ==='
if [ ! -d /root/cutlass/include/cutlass ]; then
    git clone --depth 1 https://github.com/NVIDIA/cutlass /root/cutlass 2>&1 | tail -5
else
    echo 'CUTLASS already present'
fi
ls -d /root/cutlass/include/cutlass && echo 'CUTLASS OK'

echo ''
echo '=== Cloning flash-attention ==='
if [ ! -d /root/flash-attention/hopper ]; then
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention /root/flash-attention 2>&1 | tail -5
else
    echo 'flash-attention already present'
fi
ls /root/flash-attention/hopper/flash_fwd_combine.cu && echo 'FA OK'

echo ''
echo '=== pip deps ==='
pip install -q huggingface_hub hf_transfer 2>&1 | tail -3

echo ''
echo '=== Launching 31B FP8-Dynamic download in background (~33GB) ==='
mkdir -p /root/models
export HF_HUB_ENABLE_HF_TRANSFER=1
nohup python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    'RedHatAI/gemma-4-31B-it-FP8-Dynamic',
    local_dir='/root/models/gemma-4-31B-it-FP8',
    max_workers=8,
)
print('DONE', p)
" > /tmp/download_31b.log 2>&1 &
echo "download PID: $!"

echo ''
echo '=== E4B FP8 downloads in background (~5GB) ==='
nohup python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    'RedHatAI/gemma-4-E4B-it-FP8-Dynamic',
    local_dir='/root/models/gemma-4-E4B-it-FP8',
    max_workers=8,
)
print('DONE', p)
" > /tmp/download_e4b.log 2>&1 &
echo "download PID: $!"

echo ''
echo '=== Kicked off. Check /tmp/download_*.log for progress. ==='
