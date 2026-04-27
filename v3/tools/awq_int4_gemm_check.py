#!/usr/bin/env python3
# Cycle 51 step 10d.1b: numerical check for awq_int4_gemm_sm120_wmma_kernel
# against the same compressed-tensors AWQ layout the GEMV kernel uses.
# Adapted from awq_int4_gemv_check.py (cycle 40 step 3) — same packing,
# same dequant, just M>1 activation matrix.
#
# Layout (verified against ebircak/gemma-4-31B-it-4bit-W4A16-AWQ):
#   weight_packed:    int32 [N, K/8]   8 INT4 per int32 along K
#   weight_scale:     bf16  [N, K/g]
#   weight_zero_point: int32 [N/8, K/g] 8 INT4 per int32 along N
#
# Usage:
#   ~/.venv/bin/python3 v3/tools/awq_int4_gemm_check.py [sm_xxx]

import pathlib
import subprocess
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121a"
PTX = pathlib.Path("/tmp/awq_int4_gemm_sm120_wmma.ptx")
SRC = REPO / "kernels/awq_int4_gemm_sm120_wmma.cu"

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
fn = CK(drv.cuModuleGetFunction(mod, b"awq_int4_gemm_sm120_wmma_kernel"), "get fn")


def pack_int4_into_i32_along_axis(vals: np.ndarray, axis: int) -> np.ndarray:
    """Pack 8 INT4 values into one int32 along the chosen axis.
    Lane order: lane i occupies bits (4*i)..(4*i+3) — matches compressed-
    tensors convention.
    """
    vals = np.moveaxis(vals, axis, -1)
    assert vals.shape[-1] % 8 == 0, f"axis len must be /8, got {vals.shape}"
    n = vals.shape[-1] // 8
    packed = np.zeros(vals.shape[:-1] + (n,), dtype=np.int32)
    for lane in range(8):
        packed |= (vals[..., lane::8].astype(np.int32) & 0xF) << (4 * lane)
    return np.moveaxis(packed, -1, axis)


def run_one(M: int, N: int, K: int, group_size: int, seed: int, ld_d: int = None) -> bool:
    if ld_d is None:
        ld_d = N
    rng = np.random.default_rng(seed)
    # Random INT4 weights in [0..15]
    w_int4 = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    w_packed = pack_int4_into_i32_along_axis(w_int4, axis=1)  # [N, K/8] i32

    g_per_row = K // group_size
    scales_f32 = rng.normal(0.0, 0.02, size=(N, g_per_row)).astype(np.float32)
    scales_bits = (scales_f32.view(np.uint32) >> 16).astype(np.uint16)

    N_pad = ((N + 7) // 8) * 8
    z = rng.integers(0, 16, size=(N_pad, g_per_row), dtype=np.uint8)
    z[N:, :] = 0
    z_packed = pack_int4_into_i32_along_axis(z, axis=0)  # [N_pad/8, g_per_row]

    # M-row activation matrix
    act = rng.normal(0.0, 1.0, size=(M, K)).astype(np.float16)

    # Ground truth in float64
    # Dequantize weights once.
    expected = np.zeros((M, N), dtype=np.float64)
    w_dq = np.zeros((N, K), dtype=np.float64)
    for n in range(N):
        for g in range(g_per_row):
            ks, ke = g * group_size, (g + 1) * group_size
            zv = float(z[n, g])
            sv = float(scales_f32[n, g])
            w_dq[n, ks:ke] = (w_int4[n, ks:ke].astype(np.float64) - zv) * sv
    expected = act.astype(np.float64) @ w_dq.T  # [M, N]

    # Device buffers
    d_act = CK(drv.cuMemAlloc(act.nbytes), "alloc act")
    d_w   = CK(drv.cuMemAlloc(w_packed.nbytes), "alloc w")
    d_s   = CK(drv.cuMemAlloc(scales_bits.nbytes), "alloc s")
    d_z   = CK(drv.cuMemAlloc(z_packed.nbytes), "alloc z")
    # Allocate D as [M, ld_d] so column offsets within rows don't OOB.
    d_out = CK(drv.cuMemAlloc(M * ld_d * 2), "alloc out")
    CK(drv.cuMemcpyHtoD(d_act, act.tobytes(), act.nbytes), "H2D act")
    CK(drv.cuMemcpyHtoD(d_w, w_packed.tobytes(), w_packed.nbytes), "H2D w")
    CK(drv.cuMemcpyHtoD(d_s, scales_bits.tobytes(), scales_bits.nbytes), "H2D s")
    CK(drv.cuMemcpyHtoD(d_z, z_packed.tobytes(), z_packed.nbytes), "H2D z")
    CK(drv.cuMemsetD8(d_out, 0, M * ld_d * 2), "memset out")

    # Kernel ABI: (D, A, B_packed, B_scale, B_zero, M, N, K, group_size, ld_d)
    args = [np.array([int(p)], dtype=np.uint64)
            for p in (d_out, d_act, d_w, d_s, d_z)]
    args += [np.array([M], dtype=np.int32),
             np.array([N], dtype=np.int32),
             np.array([K], dtype=np.int32),
             np.array([group_size], dtype=np.int32),
             np.array([ld_d], dtype=np.int32)]
    arg_ptrs = np.array([a.ctypes.data for a in args], dtype=np.uint64)

    # Production v2: gridDim = (ceil(N/32), ceil(M/32), 1), blockDim = 128.
    gx = (N + 31) // 32
    gy = (M + 31) // 32
    CK(drv.cuLaunchKernel(fn, gx, gy, 1, 128, 1, 1, 0, 0,
                          arg_ptrs.ctypes.data, 0), "launch")
    CK(drv.cuCtxSynchronize(), "sync")

    # Pull the full [M, ld_d] buffer; we'll slice [:, :N] for comparison.
    out_full = np.empty((M, ld_d), dtype=np.float16)
    CK(drv.cuMemcpyDtoH(out_full.ctypes.data, d_out, M * ld_d * 2), "D2H")
    out = out_full[:, :N]
    for h in (d_act, d_w, d_s, d_z, d_out):
        CK(drv.cuMemFree(h), "free")

    out_f32 = out.astype(np.float32)
    abs_err = np.abs(out_f32 - expected.astype(np.float32))
    s_max = float(np.max(np.abs(scales_f32)))
    tol = 1.5 * K * s_max * 0.5  # fp16 accumulation noise floor
    rel = abs_err / (np.abs(expected) + 1e-6)
    bad = int((abs_err > tol).sum())
    total = M * N
    print(f"  M={M:>4} N={N:>5} K={K:>5} g={group_size:>3} seed={seed:>3}  "
          f"max_abs={abs_err.max():.4f}  mean_rel={rel.mean():.4f}  "
          f"bad={bad}/{total}")
    return bad == 0


print(f"AWQ INT4 GEMM check (compressed-tensors I32 packing) on {ARCH}")
shapes = [
    # (M, N, K, group_size, seed, ld_d?)
    (8,    128,  128, 32, 1),     # tiny smoke (ld_d=N)
    (16,   128,  256, 64, 7),     # multi-block in M and N (ld_d=N)
    (128,  256, 1024, 128, 42),   # mid-size (ld_d=N)
    (128, 8192, 5376, 128, 99),   # Gemma 4 q_proj prefill small batch (ld_d=N)
    (2048, 8192, 5376, 128, 13),  # Gemma 4 q_proj prefill max batch (ld_d=N)
    # ld_d > N — composes Q into wider QKV scratch (qkv_rows=10240
    # for Gemma 4 sliding: 8192 + 2*1024).
    (128, 8192, 5376, 128, 21, 10240),
]
ok = all(run_one(*s) for s in shapes)
print()
print("all pass" if ok else "FAIL")
sys.exit(0 if ok else 1)
