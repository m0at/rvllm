#!/usr/bin/env python3
# Cycle 39 step 2 perf baseline: bandwidth utilization of
# `awq_int4_gemv_f16_kernel` on canonical Gemma 4 31B GEMV shapes.
#
# Each GEMV reads:
#   weight_packed: N * K / 2 bytes
#   scales:        N * (K / g) * 2 bytes
#   zeros:         N * (K / (2*g)) bytes
#   activation:    K * 2 bytes
#   output:        N * 2 bytes
#
# Compares achieved GB/s against the GB10 LPDDR5x ceiling (273 GB/s).
#
# Usage:
#   ~/.venv/bin/python3 v3/tools/awq_int4_gemv_bench.py [iters]

import pathlib
import subprocess
import sys
import time

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
PTX = pathlib.Path("/tmp/awq_int4_gemv_f16.ptx")
SRC = REPO / "kernels/awq_int4_gemv_f16.cu"
ARCH = "sm_121a"
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
GB10_BW_GBS = 273.0

if not PTX.exists() or PTX.stat().st_mtime < SRC.stat().st_mtime:
    print(f"building {PTX} for {ARCH}...")
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", f"-arch={ARCH}", "-ptx", str(SRC), "-o", str(PTX)],
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
fn = CK(drv.cuModuleGetFunction(mod, b"awq_int4_gemv_f16_kernel"), "fn")

# Gemma 4 31B canonical projection shapes per layer.
# Hidden = 5376, intermediate = 21504. Sliding layers: 16 Q heads, 8 KV heads,
# head_dim = 256. Global layers: 4 KV heads at head_dim = 512.
SHAPES = [
    ("QKV_sliding",   4608, 5376),  # (16 + 2*8) * 256 = 4608
    ("O_sliding",     5376, 4096),  # 5376 × (16 * 256)
    ("gate_up",      43008, 5376),  # 2 * intermediate
    ("down",          5376, 21504),
]
GROUP = 128

print(f"AWQ INT4 GEMV bench, group_size={GROUP}, iters={ITERS}, ceiling={GB10_BW_GBS} GB/s")
total_us = 0.0
for name, N, K in SHAPES:
    K_pad = (K // GROUP) * GROUP
    if K_pad != K:
        K = K_pad
    act_b = K * 2
    w_b = N * (K // 2)
    s_b = N * (K // GROUP) * 2
    z_b = max(N * (K // (2 * GROUP)), N)  # ≥1 byte/row
    out_b = N * 2

    d_act = CK(drv.cuMemAlloc(act_b), "act")
    d_w = CK(drv.cuMemAlloc(w_b), "w")
    d_s = CK(drv.cuMemAlloc(s_b), "s")
    d_z = CK(drv.cuMemAlloc(z_b), "z")
    d_o = CK(drv.cuMemAlloc(out_b), "o")
    for h, n in ((d_act, act_b), (d_w, w_b), (d_s, s_b), (d_z, z_b)):
        CK(drv.cuMemsetD8(h, 0x33, n), "fill")

    args = [np.array([int(p)], dtype=np.uint64) for p in (d_act, d_w, d_s, d_z, d_o)]
    args += [np.array([N], dtype=np.int32),
             np.array([K], dtype=np.int32),
             np.array([GROUP], dtype=np.int32)]
    arg_ptrs = np.array([a.ctypes.data for a in args], dtype=np.uint64)

    for _ in range(5):
        CK(drv.cuLaunchKernel(fn, N, 1, 1, 32, 1, 1, 0, 0, arg_ptrs.ctypes.data, 0), "")
    CK(drv.cuCtxSynchronize(), "")

    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        CK(drv.cuLaunchKernel(fn, N, 1, 1, 32, 1, 1, 0, 0, arg_ptrs.ctypes.data, 0), "")
    CK(drv.cuCtxSynchronize(), "")
    t1 = time.perf_counter_ns()

    avg_us = (t1 - t0) / ITERS / 1000.0
    total_us += avg_us
    bw_bytes = w_b + s_b + z_b + act_b + out_b
    bw_gbs = bw_bytes / (avg_us * 1e3)
    pct = 100.0 * bw_gbs / GB10_BW_GBS
    print(f"  {name:>12}  N={N:>6} K={K:>6}  {avg_us:>7.1f} µs   "
          f"{bw_gbs:>6.1f} GB/s  ({pct:>4.1f}%)")

    for h in (d_act, d_w, d_s, d_z, d_o):
        CK(drv.cuMemFree(h), "")

print()
per_layer_us = total_us  # one iteration of all four projections
gemma4_layers = 60
decode_step_us = per_layer_us * gemma4_layers
projected_tok_s = 1e6 / decode_step_us
print(f"per-layer GEMV total: {per_layer_us:7.1f} µs")
print(f"60-layer GEMV total:  {decode_step_us / 1e3:7.1f} ms")
print(f"projected GEMV-only ceiling: {projected_tok_s:5.1f} tok/s")
print(f"(real decode adds attention + RoPE + RMSNorm — actual will be lower)")
