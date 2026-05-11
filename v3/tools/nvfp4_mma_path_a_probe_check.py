#!/usr/bin/env python3
# Path A validation: mixed FP8 E4M3 (Q) × NVFP4 E2M1 (K) via two unscaled
# `f8f6f4.m16n8k32.e4m3.e2m1.f32` MMAs with per-K=16 scales applied
# post-MMA. Compares against an fp64 reference computed from the
# DEQUANTIZED operand values (the same values the MMA hardware sees).
#
# Gate: 5e-3 × peak(|ref|), matching the existing NVFP4 decode/prefill
# harnesses. Passing here validates every moving part needed for FA2
# integration:
#   * fp8 packer (unchanged from fp8 family, reused verbatim)
#   * NEW e2m1 `<<2`-shift byte-unpacker for the K operand
#   * per-K=16 scale application against the 2×MMA masked pattern
#   * f64 reference using the dequanted operand values

import pathlib, sys, math, numpy as np
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
        _, name = drv.cuGetErrorName(err)
        sys.exit(f"{what}: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else tuple(rest) if rest else None

CHECK(drv.cuInit(0), "init"); dev = CHECK(drv.cuDeviceGet(0), "dev")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "ctx"); CHECK(drv.cuCtxSetCurrent(ctx), "set")
mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load")
fn  = CHECK(drv.cuModuleGetFunction(mod, b"nvfp4_mma_path_a_probe_kernel"), "fn")


# --- decode helpers ---

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)

def e2m1_decode_nibble(nib):
    s = -1.0 if (nib >> 3) & 1 else 1.0
    return s * FP4_TABLE[nib & 7]

def e2m1_nearest_nibble(v):
    s = 0 if v >= 0 else 1
    m = abs(v)
    idx = int(np.argmin(np.abs(FP4_TABLE - m)))
    return (s << 3) | idx

def e4m3_decode(byte):
    s = (byte >> 7) & 1
    e = (byte >> 3) & 0xF
    m = byte & 7
    sgn = -1.0 if s else 1.0
    if e == 0:
        return sgn * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return 0.0  # NaN slot — not used
    return sgn * (2.0 ** (e - 7)) * (1.0 + m / 8.0)

def e4m3_encode(x):
    s = 0 if x >= 0 else 1
    m = abs(x)
    if m == 0:
        return 0
    e = int(math.floor(math.log2(max(m, 1e-45))))
    e = max(-6, min(8, e))
    f = m / (2.0 ** e)
    mm = round((f - 1) * 8)
    if mm == 8:
        mm = 0; e += 1
    mm = max(0, min(7, mm))
    return (s << 7) | ((e + 7) << 3) | mm


# --- build test inputs ---

rng = np.random.default_rng(4242)

# Q: fp8 E4M3 values. Sample random fp32 then quantize to E4M3.
Q_f32 = rng.standard_normal((16, 32)).astype(np.float64) * 1.2
Q_bytes = np.zeros((16, 32), dtype=np.uint8)
Q_dq    = np.zeros((16, 32), dtype=np.float64)
for r in range(16):
    for k in range(32):
        b = e4m3_encode(Q_f32[r, k])
        Q_bytes[r, k] = b
        Q_dq[r, k]    = e4m3_decode(int(b))

# K: NVFP4. Per-16 E4M3 microscale + 4-bit E2M1 nibbles.
K_f32 = rng.standard_normal((8, 32)).astype(np.float64) * 1.5
K_bytes_nvfp4 = np.zeros((8, 16), dtype=np.uint8)  # 2 e2m1 per byte
K_scales_e4m3 = np.zeros((8, 2), dtype=np.uint8)   # 2 scales per row (K[0..15], K[16..31])
K_dq = np.zeros((8, 32), dtype=np.float64)

for r in range(8):
    for blk in range(2):
        block = K_f32[r, blk*16:(blk+1)*16]
        amax = float(np.max(np.abs(block)))
        if amax == 0:
            K_scales_e4m3[r, blk] = 0
            continue
        raw_s = amax / 6.0
        K_scales_e4m3[r, blk] = e4m3_encode(raw_s)
        s_dec = e4m3_decode(int(K_scales_e4m3[r, blk]))
        if s_dec == 0:
            continue
        # Quantize each element.
        nibs = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            n = e2m1_nearest_nibble(block[i] / s_dec)
            nibs[i] = n
            K_dq[r, blk*16 + i] = e2m1_decode_nibble(int(n)) * s_dec
        # Pack 2 e2m1 per byte (low nibble = even K, high nibble = odd K
        # within the sub-block).
        for j in range(8):
            K_bytes_nvfp4[r, blk*8 + j] = nibs[2*j] | (nibs[2*j+1] << 4)

# fp64 reference from the DEQUANTIZED operand values — what the MMA sees.
D_ref = Q_dq @ K_dq.T  # [16, 8]

# --- run kernel ---

da = CHECK(drv.cuMemAlloc(16*32), "a")
db = CHECK(drv.cuMemAlloc(8*16), "b")
dsb = CHECK(drv.cuMemAlloc(8*2), "sb")
dd = CHECK(drv.cuMemAlloc(16*8*4), "d")
CHECK(drv.cuMemcpyHtoD(da, Q_bytes.tobytes(), 16*32), "ah")
CHECK(drv.cuMemcpyHtoD(db, K_bytes_nvfp4.tobytes(), 8*16), "bh")
CHECK(drv.cuMemcpyHtoD(dsb, K_scales_e4m3.tobytes(), 8*2), "sbh")
CHECK(drv.cuMemsetD8(dd, 0, 16*8*4), "dz")

# Smem = A (512) + B (128) + D (512) = 1152 bytes.
smem = 16*32 + 8*16 + 16*8*4
params = [np.array([int(x)], dtype=np.uint64) for x in (da, db, dsb, dd)]
pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
CHECK(drv.cuLaunchKernel(fn, 1,1,1, 32,1,1, smem, 0, pp.ctypes.data, 0), "launch")
CHECK(drv.cuCtxSynchronize(), "sync")

D_dev = np.empty(16*8, dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(D_dev.ctypes.data, dd, 16*8*4), "dh")
D_dev = D_dev.reshape(16, 8).astype(np.float64)

err  = np.abs(D_dev - D_ref)
peak = np.abs(D_ref).max()
tol  = 5e-3 * max(peak, 1e-6)
bad  = int((err > tol).sum())

print(f"D_ref[0, :4] = {D_ref[0, :4]}")
print(f"D_dev[0, :4] = {D_dev[0, :4]}")
print(f"max abs err  = {err.max():.3e}")
print(f"peak |ref|   = {peak:.3e}")
print(f"tolerance    = {tol:.3e}")
print(f"mismatches   = {bad} / {D_dev.size}")
print()
print("PASS — Path A contract validated end-to-end" if bad == 0
      else "FAIL — see delta above")
sys.exit(0 if bad == 0 else 1)
