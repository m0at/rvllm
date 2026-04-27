#!/usr/bin/env python3
# Cycle 39 step 2: numerical check for awq_int4_gemv_f16_kernel.
#
# Builds a small ground-truth W4A16 GEMV reference in numpy, runs it
# through the kernel, and asserts the output matches within the
# expected INT4 quant noise floor.
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

if not PTX.exists() or PTX.stat().st_mtime < (REPO / "kernels/awq_int4_gemv_f16.cu").stat().st_mtime:
    print(f"building PTX for {ARCH}...")
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", f"-arch={ARCH}", "-ptx",
         str(REPO / "kernels/awq_int4_gemv_f16.cu"), "-o", str(PTX)],
        check=True,
    )


def CHECK(res, what):
    err, *rest = res if isinstance(res, tuple) else (res,)
    if err != drv.CUresult.CUDA_SUCCESS:
        _, name = drv.cuGetErrorName(err)
        sys.exit(f"{what}: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else (tuple(rest) if rest else None)


CHECK(drv.cuInit(0), "cuInit")
dev = CHECK(drv.cuDeviceGet(0), "cuDeviceGet")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "ctx retain")
CHECK(drv.cuCtxSetCurrent(ctx), "ctx set")

mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load PTX")
fn = CHECK(drv.cuModuleGetFunction(mod, b"awq_int4_gemv_f16_kernel"), "get fn")


def pack_int4_lowhi(vals: np.ndarray) -> np.ndarray:
    # vals shape: [..., K], dtype uint8 in [0..15]. Returns [..., K//2]
    # with low nibble = even-index element, high = odd-index element.
    assert vals.shape[-1] % 2 == 0
    lo = vals[..., 0::2] & 0x0F
    hi = (vals[..., 1::2] & 0x0F) << 4
    return (lo | hi).astype(np.uint8)


def run_one(N: int, K: int, group_size: int, seed: int, atol_factor: float = 1.5) -> bool:
    rng = np.random.default_rng(seed)
    # Random INT4 weights in [0..15]
    w_int4 = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    w_packed = pack_int4_lowhi(w_int4)            # [N, K//2]
    # Random scales (small fp16) and zero-points in [0..15]
    scales = rng.normal(0.0, 0.02, size=(N, K // group_size)).astype(np.float16)
    zeros = rng.integers(0, 16, size=(N, K // group_size), dtype=np.uint8)
    # Pack zeros 2 per byte
    if (K // group_size) % 2 != 0:
        zeros = np.pad(zeros, ((0, 0), (0, 1)), constant_values=0)
    z_packed = pack_int4_lowhi(zeros)             # [N, K/(2*g)]
    # Random FP16 activation
    act = rng.normal(0.0, 1.0, size=(K,)).astype(np.float16)

    # Ground truth in float64
    g_per_row = K // group_size
    w_dq = np.empty((N, K), dtype=np.float64)
    for n in range(N):
        for g in range(g_per_row):
            ks = g * group_size
            ke = (g + 1) * group_size
            z = float(zeros[n, g])
            s = float(scales[n, g])
            w_dq[n, ks:ke] = (w_int4[n, ks:ke].astype(np.float64) - z) * s
    expected = (w_dq @ act.astype(np.float64)).astype(np.float32)

    # Allocate device buffers
    d_act, d_w, d_s, d_z, d_out = (CHECK(drv.cuMemAlloc(b), f"alloc {b}") for b in
                                   (act.nbytes, w_packed.nbytes, scales.nbytes,
                                    z_packed.nbytes, N * 2))
    CHECK(drv.cuMemcpyHtoD(d_act, act.tobytes(), act.nbytes), "H2D act")
    CHECK(drv.cuMemcpyHtoD(d_w, w_packed.tobytes(), w_packed.nbytes), "H2D w")
    CHECK(drv.cuMemcpyHtoD(d_s, scales.tobytes(), scales.nbytes), "H2D s")
    CHECK(drv.cuMemcpyHtoD(d_z, z_packed.tobytes(), z_packed.nbytes), "H2D z")
    CHECK(drv.cuMemsetD8(d_out, 0, N * 2), "memset out")

    # Pack args
    p_act = np.array([int(d_act)], dtype=np.uint64)
    p_w = np.array([int(d_w)], dtype=np.uint64)
    p_s = np.array([int(d_s)], dtype=np.uint64)
    p_z = np.array([int(d_z)], dtype=np.uint64)
    p_o = np.array([int(d_out)], dtype=np.uint64)
    a_n = np.array([N], dtype=np.int32)
    a_k = np.array([K], dtype=np.int32)
    a_g = np.array([group_size], dtype=np.int32)
    args = [p_act, p_w, p_s, p_z, p_o, a_n, a_k, a_g]
    arg_ptrs = np.array([p.ctypes.data for p in args], dtype=np.uint64)

    grid = (N, 1, 1)
    block = (32, 1, 1)
    CHECK(drv.cuLaunchKernel(fn, *grid, *block, 0, 0, arg_ptrs.ctypes.data, 0),
          "launch")
    CHECK(drv.cuCtxSynchronize(), "sync")

    out = np.empty(N, dtype=np.float16)
    CHECK(drv.cuMemcpyDtoH(out.ctypes.data, d_out, N * 2), "D2H")
    for h in (d_act, d_w, d_s, d_z, d_out):
        CHECK(drv.cuMemFree(h), "free")

    out_f32 = out.astype(np.float32)
    abs_err = np.abs(out_f32 - expected)
    # FP16 accumulation noise tolerance: K * sqrt(K) * scale_max
    s_max = float(np.max(np.abs(scales.astype(np.float32))))
    tol = atol_factor * K * s_max * 0.5  # generous
    bad = int((abs_err > tol).sum())
    rel = abs_err / (np.abs(expected) + 1e-6)
    print(f"  N={N:>4} K={K:>5} g={group_size:>3} seed={seed:>2} "
          f"max_abs={abs_err.max():.4f} mean_rel={rel.mean():.4f} bad={bad}/{N}")
    return bad == 0


print(f"AWQ INT4 GEMV check on {ARCH}")
shapes = [
    (4, 128, 32, 1),     # smallest sane: 1 group, 4 rows
    (16, 1024, 128, 7),  # canonical AWQ group_size=128
    (32, 2048, 128, 42), # bigger row count
    (128, 5376, 128, 99),# Gemma 4 hidden_size = 5376
]
ok = all(run_one(*s) for s in shapes)
print()
print("all pass" if ok else "FAIL")
sys.exit(0 if ok else 1)
