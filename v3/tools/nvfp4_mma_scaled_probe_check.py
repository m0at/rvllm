#!/usr/bin/env python3
# Phase-B MVP: verify the NVFP4 native-MMA path with realistic per-16-
# element E4M3 microscales (not the hardcoded 1.0 scales used by
# `nvfp4_mma_probe_check.py`).
#
# Setup:
#   A: 16 × 64 random fp16 → quantize to NVFP4 (packed bytes + per-16
#      E4M3 microscales). Shape packed = 16 × 32 bytes. Shape scales =
#      16 × 4 E4M3 bytes.
#   B:  8 × 64 same pattern.
#
# Run the `nvfp4_mma_scaled_probe_kernel` which loads bytes + scales in
# the FA2-integration contract and runs a single
#   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
#       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
# call.
#
# Compare to an fp64 reference computed from the DEQUANTIZED A/B (the
# same values the MMA hardware sees). Tolerance: 5e-3 × peak(|ref|),
# same gate as `fa2_nvfp4_decode_check.py`.

import pathlib, sys
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path("/home/r00t/workspace/upstream/rvllm-serve")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX  = REPO / "kernels" / ARCH / "nvfp4_mma_probe.ptx"

def CHECK(res, what):
    if isinstance(res, tuple):
        err, *rest = res
    else:
        err, rest = res, ()
    if err != drv.CUresult.CUDA_SUCCESS:
        sys.exit(f"{what} failed: {err}")
    return rest[0] if len(rest) == 1 else tuple(rest) if rest else None

CHECK(drv.cuInit(0), "cuInit")
dev = CHECK(drv.cuDeviceGet(0), "dev")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "ctx")
CHECK(drv.cuCtxSetCurrent(ctx), "set ctx")
mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load")
fn  = CHECK(drv.cuModuleGetFunction(mod, b"nvfp4_mma_scaled_probe_kernel"),
            "fn")


# ----- NVFP4 quant helpers -----

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)

def nearest_fp4_nibble(x: float) -> int:
    """fp32 → 4-bit e2m1 nibble (sign bit + 3 magnitude bits)."""
    sign = 0 if x >= 0 else 1
    mag = abs(x)
    # find nearest magnitude in FP4_TABLE
    idx = int(np.argmin(np.abs(FP4_TABLE - mag)))
    return (sign << 3) | idx

def e4m3_encode(x: float) -> int:
    """fp32 → 8-bit E4M3 scale byte (value 1.0 = 0x38, value 0 = 0x00)."""
    if x == 0.0:
        return 0x00
    sign = 0 if x >= 0 else 1
    mag = abs(x)
    mag = min(mag, 448.0)    # E4M3 max
    # exp bias=7, mantissa 3 bits
    # mag = 2^(E-7) * (1 + M/8), 0 < E <= 15
    import math
    e = int(math.floor(math.log2(max(mag, 1e-45))))
    e = max(-6, min(8, e))       # clamp
    frac = mag / (2.0 ** e)      # in [1, 2)
    m_full = round((frac - 1.0) * 8.0)
    if m_full == 8:
        m_full = 0
        e += 1
    m_full = max(0, min(7, m_full))
    exp_field = e + 7
    if exp_field <= 0:
        # subnormal
        m_sub = round(mag * (2.0 ** 9))
        m_sub = max(0, min(7, m_sub))
        return (sign << 7) | m_sub
    exp_field = min(15, exp_field)
    return (sign << 7) | (exp_field << 3) | m_full

def e4m3_decode(byte: int) -> float:
    """E4M3 byte → fp32. Matches the canonical OCP FP8 E4M3 spec."""
    s = (byte >> 7) & 1
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    sign = -1.0 if s else 1.0
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return 0.0  # NaN slot in E4M3 — treat as zero for this probe
    return sign * (2.0 ** (e - 7)) * (1.0 + m / 8.0)

def quantize_nvfp4_row(row_f32: np.ndarray):
    """Given [K] fp32, return (packed_bytes [K/2], scales_e4m3 [K/16])."""
    K = row_f32.shape[0]
    assert K % 16 == 0
    n_blocks = K // 16
    scales = np.zeros(n_blocks, dtype=np.uint8)
    nibbles = np.zeros(K, dtype=np.uint8)
    for b in range(n_blocks):
        block = row_f32[b * 16:(b + 1) * 16]
        amax = float(np.max(np.abs(block)))
        if amax == 0.0:
            scales[b] = 0
            continue
        # Choose scale so that amax/scale maps to FP4_TABLE[-1] = 6.0.
        raw_scale = amax / 6.0
        scales[b] = e4m3_encode(raw_scale)
        s_dec = e4m3_decode(int(scales[b]))
        if s_dec == 0:
            continue
        for i in range(16):
            v = block[i] / s_dec
            nibbles[b * 16 + i] = nearest_fp4_nibble(float(v))
    # Pack two nibbles per byte: low = even k, high = odd k.
    packed = np.zeros(K // 2, dtype=np.uint8)
    for j in range(K // 2):
        lo = nibbles[2 * j]
        hi = nibbles[2 * j + 1]
        packed[j] = lo | (hi << 4)
    return packed, scales

def dequantize_nvfp4_row(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    K2 = packed.shape[0]
    n_blocks = scales.shape[0]
    assert K2 * 2 == n_blocks * 16
    out = np.zeros(K2 * 2, dtype=np.float64)
    for b in range(n_blocks):
        s = e4m3_decode(int(scales[b]))
        for i in range(16):
            k = b * 16 + i
            byte = packed[k // 2]
            nib = byte & 0xF if (k % 2 == 0) else (byte >> 4) & 0xF
            sign = -1.0 if (nib >> 3) & 1 else 1.0
            mag = FP4_TABLE[nib & 0x7]
            out[k] = sign * mag * s
    return out

# ----- Build realistic probe input -----

rng = np.random.default_rng(42)
A_f32 = rng.standard_normal((16, 64)).astype(np.float64) * 1.5
B_f32 = rng.standard_normal((8,  64)).astype(np.float64) * 1.5

A_packed  = np.zeros((16, 32), dtype=np.uint8)
A_scales  = np.zeros((16,  4), dtype=np.uint8)
B_packed  = np.zeros(( 8, 32), dtype=np.uint8)
B_scales  = np.zeros(( 8,  4), dtype=np.uint8)
for r in range(16):
    p, s = quantize_nvfp4_row(A_f32[r])
    A_packed[r] = p; A_scales[r] = s
for r in range(8):
    p, s = quantize_nvfp4_row(B_f32[r])
    B_packed[r] = p; B_scales[r] = s

# Fp64 reference from the DEQUANTIZED bytes — this is what the MMA sees.
A_dq = np.zeros((16, 64), dtype=np.float64)
B_dq = np.zeros(( 8, 64), dtype=np.float64)
for r in range(16):
    A_dq[r] = dequantize_nvfp4_row(A_packed[r], A_scales[r])
for r in range(8):
    B_dq[r] = dequantize_nvfp4_row(B_packed[r], B_scales[r])

D_ref = A_dq @ B_dq.T  # 16 × 8

# ----- Run kernel -----

A_bytes = A_packed.tobytes()
B_bytes = B_packed.tobytes()
A_sc    = A_scales.tobytes()
B_sc    = B_scales.tobytes()

da = CHECK(drv.cuMemAlloc(len(A_bytes)), "da")
db = CHECK(drv.cuMemAlloc(len(B_bytes)), "db")
dsa = CHECK(drv.cuMemAlloc(len(A_sc)),   "dsa")
dsb = CHECK(drv.cuMemAlloc(len(B_sc)),   "dsb")
dd = CHECK(drv.cuMemAlloc(16*8*4),       "dd")
CHECK(drv.cuMemcpyHtoD(da,  A_bytes, len(A_bytes)), "ah")
CHECK(drv.cuMemcpyHtoD(db,  B_bytes, len(B_bytes)), "bh")
CHECK(drv.cuMemcpyHtoD(dsa, A_sc,    len(A_sc)),    "sah")
CHECK(drv.cuMemcpyHtoD(dsb, B_sc,    len(B_sc)),    "sbh")
CHECK(drv.cuMemsetD8(dd, 0, 16*8*4), "dz")

# Dynamic smem = A (512) + B (256) + D (512) = 1280 bytes.
smem = 16*32 + 8*32 + 16*8*4
params = [np.array([int(x)], dtype=np.uint64) for x in (da, db, dsa, dsb, dd)]
pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
CHECK(drv.cuLaunchKernel(fn, 1,1,1, 32,1,1, smem, 0, pp.ctypes.data, 0), "launch")
CHECK(drv.cuCtxSynchronize(), "sync")

D_dev = np.empty(16*8, dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(D_dev.ctypes.data, dd, 16*8*4), "dh")
D_dev = D_dev.reshape(16, 8).astype(np.float64)

err   = np.abs(D_dev - D_ref)
peak  = np.abs(D_ref).max()
tol   = 5e-3 * max(peak, 1e-6)
bad   = int((err > tol).sum())

print(f"D_ref[0, :4] = {D_ref[0, :4]}")
print(f"D_dev[0, :4] = {D_dev[0, :4]}")
print(f"max abs err  = {err.max():.3e}   (tol = {tol:.3e})")
print(f"peak |ref|   = {peak:.3e}")
print(f"mismatches   = {bad} / {D_dev.size}")
print()
print("PASS — FA2-integration contract validated" if bad == 0
      else "FAIL — scale-loading or MMA contract misaligned")
sys.exit(0 if bad == 0 else 1)
