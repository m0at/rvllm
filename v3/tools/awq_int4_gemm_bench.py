#!/usr/bin/env python3
# Cycle 51 step 10d.2: throughput bench for the WMMA AWQ GEMM kernel
# on canonical Gemma 4 31B prefill shapes.
#
# Reports ms/launch, achieved GFLOPS (2 * M * N * K / time), and GB/s
# bandwidth utilization (total bytes touched / time). FLOPS is the
# primary metric for compute-bound prefill GEMM; GB/s is the
# secondary cross-check against the 273 GB/s LPDDR5x ceiling.
#
# Usage:
#   ~/.venv/bin/python3 v3/tools/awq_int4_gemm_bench.py [iters]

import pathlib
import subprocess
import sys
import time

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
PTX = pathlib.Path("/tmp/awq_int4_gemm_sm120_wmma.ptx")
SRC = REPO / "kernels/awq_int4_gemm_sm120_wmma.cu"
ARCH = "sm_121a"
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
GB10_BW_GBS = 273.0

if not PTX.exists() or PTX.stat().st_mtime < SRC.stat().st_mtime:
    print(f"building {PTX} for {ARCH}...")
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", f"-arch={ARCH}", "-ptx",
         str(SRC), "-o", str(PTX)],
        check=True,
    )


def CK(r, what):
    err, *rest = r if isinstance(r, tuple) else (r,)
    if err != drv.CUresult.CUDA_SUCCESS:
        sys.exit(f"{what}: {err}")
    return rest[0] if len(rest) == 1 else (tuple(rest) if rest else None)


CK(drv.cuInit(0), "init")
dev = CK(drv.cuDeviceGet(0), "dev")
ctx = CK(drv.cuDevicePrimaryCtxRetain(dev), "ctx")
CK(drv.cuCtxSetCurrent(ctx), "set")
mod = CK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "mod")
fn = CK(drv.cuModuleGetFunction(mod, b"awq_int4_gemm_sm120_wmma_kernel"), "fn")

# Canonical Gemma 4 31B prefill shapes. M sweeps batch sizes typical
# for prefill; N/K reflect the four linear projections per layer.
SHAPES = [
    # (label, M, N, K)
    ("q_proj M=128",     128, 8192, 5376),
    ("q_proj M=512",     512, 8192, 5376),
    ("q_proj M=2048",   2048, 8192, 5376),
    ("o_proj M=128",     128, 5376, 8192),
    ("gate_up M=128",    128, 43008, 5376),  # 2*intermediate
    ("down M=128",       128, 5376, 21504),
]
GROUP = 128

print(f"AWQ INT4 GEMM bench (WMMA kernel), group_size={GROUP}, iters={ITERS}")
print(f"  hardware ceiling: {GB10_BW_GBS} GB/s LPDDR5x")
for label, M, N, K in SHAPES:
    act_b = M * K * 2
    w_b   = N * (K // 2)              # INT4 packed
    s_b   = N * (K // GROUP) * 2      # bf16
    z_b   = (((N + 7) // 8) * 8) // 8 * (K // GROUP) * 4
    out_b = M * N * 2
    bytes_total = act_b + w_b + s_b + z_b + out_b
    flops = 2.0 * M * N * K           # 2 ops per MAC

    d_act = CK(drv.cuMemAlloc(act_b), "act")
    d_w   = CK(drv.cuMemAlloc(w_b), "w")
    d_s   = CK(drv.cuMemAlloc(s_b), "s")
    d_z   = CK(drv.cuMemAlloc(z_b), "z")
    d_o   = CK(drv.cuMemAlloc(out_b), "o")
    for h, n in ((d_act, act_b), (d_w, w_b), (d_s, s_b), (d_z, z_b)):
        CK(drv.cuMemsetD8(h, 0x11, n), "fill")

    # Kernel ABI: (D, A, B_packed, B_scale, B_zero, M, N, K, group_size)
    args = [np.array([int(p)], dtype=np.uint64)
            for p in (d_o, d_act, d_w, d_s, d_z)]
    args += [np.array([M], dtype=np.int32),
             np.array([N], dtype=np.int32),
             np.array([K], dtype=np.int32),
             np.array([GROUP], dtype=np.int32),
             np.array([N], dtype=np.int32)]  # ld_d = N
    arg_ptrs = np.array([a.ctypes.data for a in args], dtype=np.uint64)

    gx = (N + 15) // 16
    gy = (M + 15) // 16
    # warm up
    for _ in range(3):
        CK(drv.cuLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, 0,
                              arg_ptrs.ctypes.data, 0), "")
    CK(drv.cuCtxSynchronize(), "")

    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        CK(drv.cuLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, 0,
                              arg_ptrs.ctypes.data, 0), "")
    CK(drv.cuCtxSynchronize(), "")
    t1 = time.perf_counter_ns()

    avg_ms = (t1 - t0) / ITERS / 1e6
    gflops = flops / (avg_ms * 1e6)
    bw_gbs = bytes_total / (avg_ms * 1e6)
    pct_bw = 100.0 * bw_gbs / GB10_BW_GBS
    print(f"  {label:>16}  M={M:>5} N={N:>5} K={K:>5}  "
          f"{avg_ms:>7.2f} ms  {gflops:>7.1f} GFLOPS  "
          f"{bw_gbs:>5.1f} GB/s ({pct_bw:>4.1f}%)")

    for h in (d_act, d_w, d_s, d_z, d_o):
        CK(drv.cuMemFree(h), "")
