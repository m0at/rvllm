#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fp8_mma_smem_probe_check.py [sm_xxx]
#
# Exercises the smem → fragment packers in
# `kernels/fp8_mma_frag_pack.cuh` via `fp8_mma_smem_probe_kernel`.
# Host provides A as flat row-major [16][32] FP8 bytes and B as
# col-major [8][32] FP8 bytes. Kernel does: load-to-smem → pack →
# MMA → unpack-to-smem → write out. Pass criterion: output matches
# fp64 `A @ B^T` reference within FP8-quant noise (scale_rel ≤ 5e-2,
# same bar as the F1 probe).

import sys, pathlib
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "fp8_mma_smem_probe.ptx"
if not PTX.exists():
    sys.exit(f"missing PTX: {PTX}")


def CHECK(res, what):
    if isinstance(res, tuple):
        err, *rest = res
    else:
        err, rest = res, ()
    if err != drv.CUresult.CUDA_SUCCESS:
        _, name = drv.cuGetErrorName(err)
        sys.exit(f"{what} failed: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else tuple(rest) if rest else None


CHECK(drv.cuInit(0), "cuInit")
dev = CHECK(drv.cuDeviceGet(0), "cuDeviceGet")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
CHECK(drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
print(f"PTX: {PTX.name}")

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(mod, b"fp8_mma_smem_probe_kernel"),
           "cuModuleGetFunction")


FP8_MAX = 448.0


def f32_to_e4m3(v):
    if v == 0.0:
        return 0
    sign = 1 if v < 0.0 else 0
    a = abs(v)
    if a > FP8_MAX:
        a = FP8_MAX
    e = 0
    m = a
    while m >= 2.0:
        m /= 2.0
        e += 1
    while m < 1.0 and e > -6:
        m *= 2.0
        e -= 1
    exp_bits = e + 7
    if exp_bits < 0:
        return (sign << 7)
    if exp_bits > 15:
        exp_bits = 15
    mant_bits = int(round((m - 1.0) * 8))
    if mant_bits == 8:
        mant_bits = 0
        exp_bits += 1
    if exp_bits > 15:
        exp_bits, mant_bits = 15, 7
    return (sign << 7) | ((exp_bits & 0xF) << 3) | (mant_bits & 0x7)


def e4m3_to_f32(b):
    if b == 0 or b == 0x80:
        return 0.0
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


rng = np.random.default_rng(31)
A_f32 = rng.normal(0, 0.5, (16, 32)).astype(np.float32)
B_f32 = rng.normal(0, 0.5, (8, 32)).astype(np.float32)

A_b = np.zeros((16, 32), dtype=np.uint8)
for m in range(16):
    for k in range(32):
        A_b[m, k] = f32_to_e4m3(float(A_f32[m, k]))
B_b = np.zeros((8, 32), dtype=np.uint8)
for n in range(8):
    for k in range(32):
        B_b[n, k] = f32_to_e4m3(float(B_f32[n, k]))

A_rt = np.vectorize(e4m3_to_f32, otypes=[np.float64])(A_b)
B_rt = np.vectorize(e4m3_to_f32, otypes=[np.float64])(B_b)
D_ref = (A_rt @ B_rt.T).astype(np.float32)


def alloc(n):
    return CHECK(drv.cuMemAlloc(n), "cuMemAlloc")


def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")


d_a = alloc(A_b.nbytes); h2d(d_a, A_b)
d_b = alloc(B_b.nbytes); h2d(d_b, B_b)
d_d = alloc(16 * 8 * 4)

# smem budget: A (512) + B (256) + D (16*8*4 = 512) + cushion
smem = 512 + 256 + 512 + 64
if smem >= 48 * 1024:
    CHECK(drv.cuFuncSetAttribute(
        fn,
        drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem,
    ), "cuFuncSetAttribute")

params = [
    np.array([int(d_a)], dtype=np.uint64),
    np.array([int(d_b)], dtype=np.uint64),
    np.array([int(d_d)], dtype=np.uint64),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

CHECK(drv.cuLaunchKernel(
    fn,
    1, 1, 1,
    32, 1, 1,
    smem, 0,
    param_ptrs.ctypes.data, 0,
), "cuLaunchKernel")
CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

D = np.empty((16, 8), dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(D.ctypes.data, d_d, D.nbytes), "DtoH")

abs_err = np.abs(D - D_ref)
ref_mean_abs = float(np.abs(D_ref).mean())
scale_rel = abs_err / max(ref_mean_abs, 1e-30)

print(f"ref:    range [{D_ref.min():+.3e}, {D_ref.max():+.3e}], "
      f"|ref| mean {ref_mean_abs:.3e}")
print(f"kernel: range [{D.min():+.3e}, {D.max():+.3e}]")
print(f"abs_err:  max {abs_err.max():.3e}  mean {abs_err.mean():.3e}")
print(f"scale_rel: max {scale_rel.max():.3e}  mean {scale_rel.mean():.3e}")

THRESHOLD = 5e-2
if scale_rel.max() > THRESHOLD:
    print(f"\nFAIL: scale_rel.max {scale_rel.max():.3e} > {THRESHOLD:.0e}")
    print("Δ layout:")
    print((D - D_ref).round(4))
    sys.exit(1)
print(f"\nOK: scale_rel.max {scale_rel.max():.3e} <= {THRESHOLD:.0e}")
