#!/bin/bash
set -euo pipefail

# bench_v2.sh -- Deploy dirty tree to vast.ai H100, build, run v1 vs v2 benchmark.
#
# Usage:
#   bash deploy/bench_v2.sh <instance_id> <tarball_path>

INSTANCE_ID=${1:?usage: bench_v2.sh <instance_id> <tarball>}
TARBALL=${2:?usage: bench_v2.sh <instance_id> <tarball>}
MODEL="Qwen/Qwen2.5-7B"
SHA=$(basename "$TARBALL" | sed 's/rvllm-//;s/-dirty.tar.gz//;s/.tar.gz//')
RUN_DIR="/workspace/runs/${SHA}"

if [[ ! -f "$TARBALL" ]]; then
    echo "Tarball not found: $TARBALL"
    exit 1
fi

# Resolve SSH
SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
# ssh-url returns ssh://root@host:port -- parse carefully
SSH_HOSTPORT=$(echo "$SSH_URL" | sed 's|ssh://||')
SSH_USER_HOST=$(echo "$SSH_HOSTPORT" | cut -d: -f1)  # root@1.2.3.4
SSH_PORT=$(echo "$SSH_HOSTPORT" | cut -d: -f2)        # 40041
SSH_BARE_HOST=$(echo "$SSH_USER_HOST" | cut -d@ -f2)  # 1.2.3.4
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT $SSH_USER_HOST"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "Instance: $INSTANCE_ID"
echo "SSH: $SSH_USER_HOST:$SSH_PORT"
echo "SHA: $SHA"
echo "Run dir: $RUN_DIR"
echo ""

# Wait for SSH
echo "Waiting for SSH..."
for i in $(seq 1 60); do
    if $SSH 'echo ok' 2>/dev/null; then break; fi
    sleep 5
done

# Wait for Rust (onstart installs it)
echo "Waiting for Rust toolchain..."
for i in $(seq 1 60); do
    if $SSH 'source $HOME/.cargo/env && cargo --version' 2>/dev/null; then break; fi
    sleep 10
done

# Upload tarball
echo "Uploading tarball..."
$SCP "$TARBALL" "root@${SSH_BARE_HOST}:/tmp/$(basename $TARBALL)"

# Remote: unpack, build, bench
$SSH 'bash -l' << REMOTE
set -euo pipefail
source \$HOME/.cargo/env 2>/dev/null || true

echo "=== Environment ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
CC=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
ARCH="sm_\$(echo \$CC | tr -d '.')"
echo "Arch: \$ARCH"
rustc --version
nvcc --version 2>&1 | tail -1

# Kill rogue GPU processes
ROGUE=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
if [[ -n "\$ROGUE" ]]; then
    echo "Killing rogue GPU procs: \$ROGUE"
    for pid in \$ROGUE; do kill -9 "\$pid" 2>/dev/null || true; done
    sleep 2
fi

# Fresh unpack
echo ""
echo "=== Unpack to ${RUN_DIR} ==="
rm -rf ${RUN_DIR}
mkdir -p ${RUN_DIR}
tar xzf /tmp/$(basename $TARBALL) -C ${RUN_DIR} --strip-components=1
cd ${RUN_DIR}
echo "SHA marker: ${SHA}"
echo "${SHA}" > REVISION

# Compile kernels
echo ""
echo "=== Compile kernels (\$ARCH) ==="
mkdir -p kernels/\$ARCH
FAIL=0
OK=0
for cu in kernels/*.cu; do
    [[ -f "\$cu" ]] || continue
    stem=\$(basename "\$cu" .cu)
    if [[ "\$stem" == "persistent_layer_decode" ]]; then
        if nvcc -cubin -arch=\$ARCH -O3 --use_fast_math -rdc=true \
            -o "kernels/\$ARCH/\${stem}.cubin" "\$cu" 2>/dev/null; then
            OK=\$((OK+1))
        else
            FAIL=\$((FAIL+1))
            echo "  FAIL: \$stem"
        fi
    elif [[ "\$stem" == cutlass_* ]]; then
        : # skip cutlass kernels
    else
        if nvcc -ptx -arch=\$ARCH -O3 --use_fast_math \
            -o "kernels/\$ARCH/\${stem}.ptx" "\$cu" 2>/dev/null; then
            OK=\$((OK+1))
        else
            FAIL=\$((FAIL+1))
            echo "  FAIL: \$stem"
        fi
    fi
done
echo "Kernels: \$OK compiled, \$FAIL failed"
export RVLLM_PTX_DIR="${RUN_DIR}/kernels/\$ARCH"

# Build CUTLASS .so if possible
echo ""
echo "=== CUTLASS shared library ==="
CUTLASS_DIR="/workspace/cutlass"
if [ ! -d "\$CUTLASS_DIR/include/cutlass" ]; then
    echo "Cloning CUTLASS..."
    git clone --depth 1 https://github.com/NVIDIA/cutlass "\$CUTLASS_DIR"
fi
if [[ -f "kernels/build_cutlass_so.sh" ]]; then
    if bash kernels/build_cutlass_so.sh "\$ARCH" "\$CUTLASS_DIR" 2>&1 | tail -3; then
        echo "CUTLASS .so built"
    else
        echo "CUTLASS .so failed (will use cuBLAS only)"
    fi
fi

# Build rvllm (v1 server binary)
echo ""
echo "=== Build rvllm (release) ==="
BUILD_T0=\$(date +%s)
CUDA_ARCH=\$ARCH cargo build --release --features cuda,cublaslt -p rvllm 2>&1 | tail -5
BUILD_T1=\$(date +%s)
echo "Build time: \$((BUILD_T1 - BUILD_T0))s"
ls -la target/release/rvllm

# Build v2 bench binary
echo ""
echo "=== Build v2 bench binary ==="
CUDA_ARCH=\$ARCH cargo build --release --features cuda-graphs -p rvllm-v2 --bin rvllm-v2-bench 2>&1 | tail -5
ls -la target/release/rvllm-v2-bench

# Download model
echo ""
echo "=== Download model ==="
pip3 install -q huggingface_hub 2>/dev/null || true
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}')
print('Model ready')
" 2>&1 | tail -3

echo ""
echo "============================================"
echo "  v1 BENCHMARK (direct engine, no HTTP)"
echo "============================================"
echo ""

target/release/rvllm benchmark \
    --model ${MODEL} \
    --n 1,32,64,96,128 \
    --output-len 128 \
    --gpu-memory-utilization 0.90 \
    --json > /tmp/bench_v1.json 2>&1 || true

cat /tmp/bench_v1.json
echo ""

echo ""
echo "============================================"
echo "  v2 BENCHMARK (direct engine, no HTTP)"
echo "============================================"
echo ""

target/release/rvllm-v2-bench \
    --model ${MODEL} \
    --n 1,32,64,96,128 \
    --output-len 128 \
    --gpu-memory-utilization 0.90 \
    --json > /tmp/bench_v2.json 2>&1 || true

cat /tmp/bench_v2.json
echo ""

echo ""
echo "============================================"
echo "  COMPARISON"
echo "============================================"
python3 -c "
import json

with open('/tmp/bench_v1.json') as f:
    v1 = json.load(f)
with open('/tmp/bench_v2.json') as f:
    v2 = json.load(f)

v1_by_n = {r['n']: r for r in v1['results']}
v2_by_n = {r['n']: r for r in v2['results']}

print(f'Model: ${MODEL}')
print(f'Output tokens: 128')
print()
print(f'{\"N\":>5} {\"v1 tok/s\":>12} {\"v2 tok/s\":>12} {\"v2/v1\":>8}')
print('-' * 42)
for n in sorted(set(list(v1_by_n.keys()) + list(v2_by_n.keys()))):
    t1 = v1_by_n.get(n, {}).get('tok_per_sec', 0)
    t2 = v2_by_n.get(n, {}).get('tok_per_sec', 0)
    ratio = t2/t1 if t1 > 0 else 0
    print(f'{n:>5} {t1:>12.1f} {t2:>12.1f} {ratio:>7.2f}x')
" 2>&1 || echo "(comparison script failed)"

echo ""
echo "=== Done ==="
REMOTE
