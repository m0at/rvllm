#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fp8_mma_probe_check.py [sm_xxx]
#
# Validates `fp8_e4m3_mma_probe_kernel`: the standalone one-warp FP8
# E4M3 tensor-core MMA that will become the inner engine of the
# unified prefill kernel's Q·Kᵀ + P·V matmuls. Pass criteria:
#
#   * PTX loads + launches on the target arch (assembly + link check).
#   * D fragment read back from the kernel matches an fp64 reference
#     A @ B^T with scale_rel (abs_err / mean|ref|) <= 5e-2 — i.e.
#     within FP8-quant noise.
#
# Fragment layout under test (per `kernels/fp8_e4m3_mma_probe.cu`,
# which is the canonical spec we'll reuse in the full FA2 rewrite):
#
#   A (m16, k32, row-major):   lane i  rows {i/4, i/4+8}
#                              a[0..3] as 4 × u32 spanning k=(i%4)*8..+8
#   B (n8,  k32, col-major):   lane i  col (i/4)
#                              b[0..1] as 2 × u32 spanning k=(i%4)*8..+8
#   D (m16, n8,  f32):         lane i  rows {i/4, i/4+8}
#                              d[0..3] at cols (i%4)*2 + [0, 1]

import sys, pathlib, struct
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "fp8_e4m3_mma_probe.ptx"
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
cc_major = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc")
cc_minor = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc")
print(f"device: cc {cc_major}.{cc_minor}, PTX: {PTX.name}")

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(mod, b"fp8_e4m3_mma_probe_kernel"),
           "cuModuleGetFunction")

# --- FP8 E4M3 round-trip identical to the kernel's `fp8kv_decode_byte` --
def e4m3_to_f32(b):
    if b == 0 or b == 0x80:
        return 0.0
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


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


# --- Test inputs: small-magnitude random values that round cleanly ---
rng = np.random.default_rng(13)
A_f32 = (rng.normal(0, 0.25, (16, 32))).astype(np.float32)
B_f32 = (rng.normal(0, 0.25, (8, 32))).astype(np.float32)

# Quantise to FP8 bytes
A_b = np.zeros((16, 32), dtype=np.uint8)
B_b = np.zeros((8, 32), dtype=np.uint8)
for m in range(16):
    for k in range(32):
        A_b[m, k] = f32_to_e4m3(float(A_f32[m, k]))
for n in range(8):
    for k in range(32):
        B_b[n, k] = f32_to_e4m3(float(B_f32[n, k]))

# Round-trip the bytes so the reference uses what the kernel will see
A_rt = np.vectorize(e4m3_to_f32, otypes=[np.float64])(A_b)
B_rt = np.vectorize(e4m3_to_f32, otypes=[np.float64])(B_b)

# Reference: D = A @ B^T in fp64
D_ref = A_rt @ B_rt.T  # shape (16, 8)

# --- Pack A into per-lane fragments ---------------------------------------
# Lane i ∈ [0, 31]:
#   rows     = {i/4, i/4+8}
#   k_group  = i % 4
#   a[0]: row (i/4),    k = k_group*8 + [0..3]
#   a[1]: row (i/4+8),  k = k_group*8 + [0..3]
#   a[2]: row (i/4),    k = k_group*8 + [4..7]
#   a[3]: row (i/4+8),  k = k_group*8 + [4..7]
a_frag = np.zeros((32, 4), dtype=np.uint32)
for lane in range(32):
    r_lo = lane // 4
    r_hi = r_lo + 8
    k_base = (lane % 4) * 8
    def pack4(row, k0):
        u = 0
        for j in range(4):
            u |= int(A_b[row, k0 + j]) << (j * 8)
        return u
    a_frag[lane, 0] = pack4(r_lo, k_base + 0)
    a_frag[lane, 1] = pack4(r_hi, k_base + 0)
    a_frag[lane, 2] = pack4(r_lo, k_base + 4)
    a_frag[lane, 3] = pack4(r_hi, k_base + 4)

# --- Pack B into per-lane fragments ---------------------------------------
# Lane i: col = i/4, k_group = i%4
#   b[0]: k = k_group*8 + [0..3]
#   b[1]: k = k_group*8 + [4..7]
b_frag = np.zeros((32, 2), dtype=np.uint32)
for lane in range(32):
    n = lane // 4
    k_base = (lane % 4) * 8
    def pack4b(col, k0):
        u = 0
        for j in range(4):
            u |= int(B_b[col, k0 + j]) << (j * 8)
        return u
    b_frag[lane, 0] = pack4b(n, k_base + 0)
    b_frag[lane, 1] = pack4b(n, k_base + 4)

# --- GPU launch -----------------------------------------------------------


def alloc(n):
    return CHECK(drv.cuMemAlloc(n), "cuMemAlloc")


def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")


d_a = alloc(a_frag.nbytes); h2d(d_a, a_frag)
d_b = alloc(b_frag.nbytes); h2d(d_b, b_frag)
d_d = alloc(32 * 4 * 4)  # 32 lanes × 4 f32

params = [
    np.array([int(d_a)], dtype=np.uint64),
    np.array([int(d_b)], dtype=np.uint64),
    np.array([int(d_d)], dtype=np.uint64),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

CHECK(drv.cuLaunchKernel(
    fn,
    1, 1, 1,
    32, 1, 1,   # one warp
    0, 0,
    param_ptrs.ctypes.data, 0,
), "cuLaunchKernel")
CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

d_frag = np.empty((32, 4), dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(d_frag.ctypes.data, d_d, d_frag.nbytes), "DtoH")

# --- Unpack D fragment into [16, 8] f32 -----------------------------------
# Lane i: rows {i/4, i/4+8}, cols (i%4)*2 + [0, 1]
#   d[0]: row (i/4),    col (i%4)*2 + 0
#   d[1]: row (i/4),    col (i%4)*2 + 1
#   d[2]: row (i/4+8),  col (i%4)*2 + 0
#   d[3]: row (i/4+8),  col (i%4)*2 + 1
D = np.full((16, 8), np.nan, dtype=np.float32)
for lane in range(32):
    r_lo = lane // 4
    r_hi = r_lo + 8
    c = (lane % 4) * 2
    D[r_lo, c + 0] = d_frag[lane, 0]
    D[r_lo, c + 1] = d_frag[lane, 1]
    D[r_hi, c + 0] = d_frag[lane, 2]
    D[r_hi, c + 1] = d_frag[lane, 3]

if np.any(np.isnan(D)):
    sys.exit(f"FAIL: unpack left NaN holes — lane-layout hypothesis wrong.\n{D}")

# --- Compare --------------------------------------------------------------
D_ref_f32 = D_ref.astype(np.float32)
abs_err = np.abs(D - D_ref_f32)
ref_mean_abs = float(np.abs(D_ref_f32).mean())
scale_rel = abs_err / max(ref_mean_abs, 1e-30)

print(f"A, B: random N(0, 0.25), shape A={A_f32.shape} B={B_f32.shape}")
print(f"ref (fp64 A@B.T):  range [{D_ref_f32.min():+.3e}, {D_ref_f32.max():+.3e}], "
      f"|ref| mean {ref_mean_abs:.3e}")
print(f"kernel D:          range [{D.min():+.3e}, {D.max():+.3e}]")
print(f"abs_err:  max {abs_err.max():.3e}  mean {abs_err.mean():.3e}")
print(f"scale_rel: max {scale_rel.max():.3e}  mean {scale_rel.mean():.3e}")

THRESHOLD = 5e-2
if scale_rel.max() > THRESHOLD:
    print(f"\nFAIL: scale_rel.max {scale_rel.max():.3e} > {THRESHOLD:.0e}")
    print("mma output:")
    print(D)
    print("ref:")
    print(D_ref_f32)
    sys.exit(1)
print(f"\nOK: scale_rel.max {scale_rel.max():.3e} <= {THRESHOLD:.0e}")
