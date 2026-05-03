#!/usr/bin/env python3
# Cycle 40 step 3: numerical check for awq_int4_gemv_f16_kernel
# against the actual compressed-tensors AWQ layout from Gemma 4.
#
# Layout (verified against ebircak/gemma-4-31B-it-4bit-W4A16-AWQ):
#   weight_packed:    int32 [N, K/8]   8 INT4 per int32 along K
#   weight_scale:     bf16  [N, K/g]
#   weight_zero_point:int32 [N/8, K/g] 8 INT4 per int32 along N
#
# Usage:
#   ~/.venv/bin/python3 v3/tools/awq_int4_gemv_check.py [sm_xxx]

import pathlib
import subprocess
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121a"
PTX = pathlib.Path("/tmp/awq_int4_gemv_f16.ptx")
SRC = REPO / "kernels/awq_int4_gemv_f16.cu"

if not PTX.exists() or PTX.stat().st_mtime < SRC.stat().st_mtime:
    print(f"building PTX for {ARCH}...")
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", f"-arch={ARCH}", "-ptx",
         str(SRC), "-o", str(PTX)],
        check=True,
    )


def CK(res, what):
    err, *rest = res if isinstance(res, tuple) else (res,)
    if err != drv.CUresult.CUDA_SUCCESS:
        _, name = drv.cuGetErrorName(err)
        sys.exit(f"{what}: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else (tuple(rest) if rest else None)


CK(drv.cuInit(0), "cuInit")
dev = CK(drv.cuDeviceGet(0), "cuDeviceGet")
ctx = CK(drv.cuDevicePrimaryCtxRetain(dev), "ctx retain")
CK(drv.cuCtxSetCurrent(ctx), "ctx set")

mod = CK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load PTX")
fn = CK(drv.cuModuleGetFunction(mod, b"awq_int4_gemv_f16_kernel"), "get fn")


def pack_int4_into_i32_along_axis(vals: np.ndarray, axis: int) -> np.ndarray:
    """Pack 8 INT4 values into one int32 along the chosen axis.
    Lane order: lane i occupies bits (4*i)..(4*i+3) — matches compressed-
    tensors convention.
    Input: any-shape uint8 in [0..15], with vals.shape[axis] % 8 == 0.
    Output: int32 with shape[axis] // 8 along the packing axis.
    """
    vals = np.moveaxis(vals, axis, -1)
    assert vals.shape[-1] % 8 == 0, f"axis len must be /8, got {vals.shape}"
    n = vals.shape[-1] // 8
    packed = np.zeros(vals.shape[:-1] + (n,), dtype=np.int32)
    for lane in range(8):
        packed |= (vals[..., lane::8].astype(np.int32) & 0xF) << (4 * lane)
    return np.moveaxis(packed, -1, axis)


def run_one(N: int, K: int, group_size: int, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    # Random INT4 weights in [0..15]
    w_int4 = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    w_packed = pack_int4_into_i32_along_axis(w_int4, axis=1)  # [N, K/8] i32

    # Scales: BF16 per (n, g)
    g_per_row = K // group_size
    scales_f32 = rng.normal(0.0, 0.02, size=(N, g_per_row)).astype(np.float32)
    # Cast to bf16 by truncating fp32 mantissa to 7 bits, store as
    # raw uint16 in same buffer.
    scales_bits = (scales_f32.view(np.uint32) >> 16).astype(np.uint16)

    # Zero points: INT4 in [0..15], packed 8-per-int32 along N.
    # Pad N to multiple of 8 for packing if needed.
    N_pad = ((N + 7) // 8) * 8
    z = rng.integers(0, 16, size=(N_pad, g_per_row), dtype=np.uint8)
    z[N:, :] = 0  # zero out padding so it doesn't affect anything visible
    z_packed = pack_int4_into_i32_along_axis(z, axis=0)  # [N_pad/8, g_per_row]

    act = rng.normal(0.0, 1.0, size=(K,)).astype(np.float16)

    # Ground truth in float64
    expected = np.zeros(N, dtype=np.float64)
    for n in range(N):
        row = np.zeros(K, dtype=np.float64)
        for g in range(g_per_row):
            ks, ke = g * group_size, (g + 1) * group_size
            zv = float(z[n, g])
            sv = float(scales_f32[n, g])
            row[ks:ke] = (w_int4[n, ks:ke].astype(np.float64) - zv) * sv
        expected[n] = float(row @ act.astype(np.float64))

    # Allocate device buffers
    d_act = CK(drv.cuMemAlloc(act.nbytes), "alloc act")
    d_w   = CK(drv.cuMemAlloc(w_packed.nbytes), "alloc w")
    d_s   = CK(drv.cuMemAlloc(scales_bits.nbytes), "alloc s")
    d_z   = CK(drv.cuMemAlloc(z_packed.nbytes), "alloc z")
    d_out = CK(drv.cuMemAlloc(N * 2), "alloc out")
    CK(drv.cuMemcpyHtoD(d_act, act.tobytes(), act.nbytes), "H2D act")
    CK(drv.cuMemcpyHtoD(d_w, w_packed.tobytes(), w_packed.nbytes), "H2D w")
    CK(drv.cuMemcpyHtoD(d_s, scales_bits.tobytes(), scales_bits.nbytes), "H2D s")
    CK(drv.cuMemcpyHtoD(d_z, z_packed.tobytes(), z_packed.nbytes), "H2D z")
    CK(drv.cuMemsetD8(d_out, 0, N * 2), "memset out")

    args = [np.array([int(p)], dtype=np.uint64) for p in (d_act, d_w, d_s, d_z, d_out)]
    args += [np.array([N], dtype=np.int32),
             np.array([K], dtype=np.int32),
             np.array([group_size], dtype=np.int32)]
    arg_ptrs = np.array([a.ctypes.data for a in args], dtype=np.uint64)

    CK(drv.cuLaunchKernel(fn, N, 1, 1, 32, 1, 1, 0, 0, arg_ptrs.ctypes.data, 0),
       "launch")
    CK(drv.cuCtxSynchronize(), "sync")

    out = np.empty(N, dtype=np.float16)
    CK(drv.cuMemcpyDtoH(out.ctypes.data, d_out, N * 2), "D2H")
    for h in (d_act, d_w, d_s, d_z, d_out):
        CK(drv.cuMemFree(h), "free")

    out_f32 = out.astype(np.float32)
    abs_err = np.abs(out_f32 - expected.astype(np.float32))
    s_max = float(np.max(np.abs(scales_f32)))
    tol = 1.5 * K * s_max * 0.5  # generous fp16 accumulation noise floor
    rel = abs_err / (np.abs(expected) + 1e-6)
    bad = int((abs_err > tol).sum())
    print(f"  N={N:>5} K={K:>5} g={group_size:>3} seed={seed:>3}  "
          f"max_abs={abs_err.max():.4f}  mean_rel={rel.mean():.4f}  bad={bad}/{N}")
    return bad == 0


print(f"AWQ INT4 GEMV check (compressed-tensors I32 packing) on {ARCH}")
shapes = [
    (8,    128, 32, 1),     # N must be /8 for the zero-point packing
    (16,  1024, 128, 7),
    (32,  2048, 128, 42),
    (128, 5376, 128, 99),   # Gemma 4 hidden_size = 5376
    (8192, 5376, 128, 13),  # Gemma 4 q_proj canonical shape
]
ok = all(run_one(*s) for s in shapes)
print()
print("all pass" if ok else "FAIL")
sys.exit(0 if ok else 1)
