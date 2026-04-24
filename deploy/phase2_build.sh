#!/bin/bash
# Build all kernel artifacts on the H100 box for sm_90.
# Runs in /workspace/runs/<SHA>/rvllm.
set -euo pipefail

cd /workspace/runs/2c6bbd0fc/rvllm
export PATH="/root/.cargo/bin:$PATH"

echo '=== 1. PTX build (kernels/build.sh) ==='
T0=$(date +%s)
bash kernels/build.sh 2>&1 | tail -20
T1=$(date +%s)
echo "build.sh: $((T1-T0))s"
ls kernels/sm_90/ | wc -l
echo ''

echo '=== 2. CUTLASS .so build (kernels/build_cutlass_so.sh sm_90) ==='
bash kernels/build_cutlass_so.sh sm_90 /root/cutlass 2>&1 | tail -10
T2=$(date +%s)
echo "build_cutlass_so: $((T2-T1))s"
ls -la kernels/sm_90/libcutlass_kernels.so 2>&1
echo ''

echo '=== 3. FA3 .so build (kernels/build_fa3.sh) ==='
bash kernels/build_fa3.sh 2>&1 | tail -15
T3=$(date +%s)
echo "build_fa3: $((T3-T2))s"
ls -la kernels/sm_90/libfa3_kernels.so 2>&1
echo ''

echo '=== Kernel artifacts ==='
ls -la kernels/sm_90/ | tail -30
