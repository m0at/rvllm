#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fp8_precision_check.py [sm_xxx]
#
# Validates kernels/<sm_xxx>/fp8_gemv.ptx against a torch reference.
# Loads the PTX via cuda-python's driver bindings, allocates managed
# memory, uploads random FP8 weights + scales + f32 input, runs
# `fp8_gemv_blockwise_wpr_lut_kernel`, DtoH's the output, computes the
# same GEMV in fp64 on CPU (fully dequantised reference), and reports
# max / mean relative + absolute error.
#
# Why this test exists:
#   * GB10 has no reference FP8 GEMM in cuBLAS we can compare against
#     directly — we have to build the reference ourselves.
#   * The scale-axis bug documented in v3/SPEC (rows vs cols confusion)
#     would manifest here as rel-err proportional to N (row-striping
#     pattern in the output).
#   * On a correct kernel, the error floor is f32 accumulation noise:
#     ~sqrt(K) * eps * mean(|val|). For K=512 and E4M3 scale ~0.1 that
#     lands around 1e-3 rel-err worst case, 1e-4 mean.
#     Quantisation noise is NOT in play: the reference dequantises with
#     the *same* FP8 byte → float mapping as the kernel.
#
# Pass criteria (TWO must both hold):
#   * max rel-err <= 5e-3  (f32 accumulation headroom)
#   * no row-correlation in errors (scale-axis-bug detector): if scales
#     are indexed on the WRONG axis (cols vs rows), errors would cluster
#     in 128-row bands. A random-looking per-row error distribution is
#     the signal that scale indexing is correct.

import os, sys, struct, math, pathlib
import numpy as np
from cuda.bindings import driver as drv

# Silence numpy division-by-zero warnings in rel-err computation
np.seterr(divide="ignore", invalid="ignore")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent  # <repo>/
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "fp8_gemv.ptx"
if not PTX.exists():
    sys.exit(f"missing PTX: {PTX}  (build with: kernels/build.sh {ARCH})")

# -------- CUDA init ---------------------------------------------------------

def CHECK(res, what):
    if isinstance(res, tuple):
        err, *rest = res
    else:
        err, rest = res, ()
    if err != drv.CUresult.CUDA_SUCCESS:
        name_err, name = drv.cuGetErrorName(err)
        sys.exit(f"{what} failed: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else tuple(rest) if rest else None

CHECK(drv.cuInit(0), "cuInit")
dev = CHECK(drv.cuDeviceGet(0), "cuDeviceGet")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
CHECK(drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

cc_major = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev),
    "cc major")
cc_minor = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev),
    "cc minor")
print(f"device: cc {cc_major}.{cc_minor}, PTX: {PTX.name} ({ARCH})")

# -------- PTX load ----------------------------------------------------------

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(mod, b"fp8_gemv_blockwise_wpr_lut_kernel"),
           "cuModuleGetFunction")

# -------- Test case ---------------------------------------------------------
# Blockwise scales: one scalar per (128-row block, 128-col block).
# Kernel expectation: scale layout is [N/128, K/128], row-major.

M, N, K = 1, 256, 512          # one token, 256 output rows, 512 input cols
assert N % 128 == 0 and K % 128 == 0 and N % 8 == 0
BN, BK = N // 128, K // 128    # scale block counts
rng = np.random.default_rng(42)

# Random FP8 E4M3 bytes. Skip the NaN encoding (0x7f and 0xff in E4M3 are
# +/- NaN — mapping those through the LUT produces NaN which poisons the
# rel-err stats).
weight_bytes = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
weight_bytes[weight_bytes == 0x7f] = 0x7e
weight_bytes[weight_bytes == 0xff] = 0xfe

# Scales ~ N(0, 0.1) — realistic order of magnitude for a blockwise
# post-training quant. Non-zero to avoid trivial reference.
scales  = rng.normal(loc=0.0, scale=0.1, size=(BN, BK)).astype(np.float32)
input_  = rng.normal(size=(M, K)).astype(np.float32)

# -------- Reference: decode E4M3 -> f32 the same way the LUT does -----------
# Match the `fp8e4m3_to_float` host-side closed-form from dequant_fp8.cu:
#   sign | exp(4) | mant(3), bias=7, denormal support, finite clamp.
def e4m3_to_f32(b):
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        v = (m / 8.0) * (2.0 ** -6)                    # denormal, 2^(1-7)
    elif e == 0xF and m == 0x7:
        v = 0.0                                        # treat NaN as 0 (filtered above anyway)
    else:
        v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return -v if s else v

lut = np.array([e4m3_to_f32(b) for b in range(256)], dtype=np.float32)

# Reference: dequant each weight, then GEMV in fp64 for accuracy.
dequant = lut[weight_bytes].astype(np.float64)                      # [N, K]
# Expand scales to full [N, K] shape
expanded = np.repeat(np.repeat(scales.astype(np.float64), 128, axis=0), 128, axis=1)
dequant *= expanded                                                 # [N, K]

ref = (input_.astype(np.float64) @ dequant.T).astype(np.float32)    # [M, N]

# -------- GPU launch --------------------------------------------------------

def alloc(bytes_):
    return CHECK(drv.cuMemAlloc(bytes_), "cuMemAlloc")

d_output = alloc(M * N * 4)
d_weight = alloc(N * K)
d_scale  = alloc(BN * BK * 4)
d_input  = alloc(M * K * 4)

def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")

h2d(d_weight, weight_bytes)
h2d(d_scale, scales)
h2d(d_input, input_)

# Launch: grid=(ceil(N/8), M), block=(256,1,1), num_col_blocks = K/128
grid_x = (N + 7) // 8
block_x = 256
num_col_blocks = BK

params = [
    np.array([int(d_output)], dtype=np.uint64),
    np.array([int(d_weight)], dtype=np.uint64),
    np.array([int(d_scale)],  dtype=np.uint64),
    np.array([int(d_input)],  dtype=np.uint64),
    np.array([M], dtype=np.int32),
    np.array([N], dtype=np.int32),
    np.array([K], dtype=np.int32),
    np.array([num_col_blocks], dtype=np.int32),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

CHECK(drv.cuLaunchKernel(
    fn,
    grid_x, M, 1,
    block_x, 1, 1,
    0,                       # shared mem
    0,                       # default stream
    param_ptrs.ctypes.data,  # kernel_params
    0,                       # extra
), "cuLaunchKernel")
CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

# -------- DtoH + compare ----------------------------------------------------

out = np.empty((M, N), dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(out.ctypes.data, d_output, out.nbytes), "DtoH")

abs_err = np.abs(out - ref)
# Pointwise rel-err blows up on near-zero cells due to cancellation —
# that's a property of the input, not the kernel. The stable signal is
# abs_err normalised by the *typical* output magnitude.
ref_mean_abs = float(np.abs(ref).mean())
scale_rel = abs_err / max(ref_mean_abs, 1e-30)

# Per-cell rel-err is still useful as a diagnostic, just not as a gate.
denom = np.maximum(np.abs(ref), ref_mean_abs * 1e-3)  # floor at 0.1% of mean
rel_err = abs_err / denom

print(f"shapes: M={M} N={N} K={K}  scales=[{BN},{BK}]")
print(f"ref    range: [{ref.min():+.4e}, {ref.max():+.4e}], |ref| mean {ref_mean_abs:.4e}")
print(f"kernel range: [{out.min():+.4e}, {out.max():+.4e}]")
print(f"abs_err:        max {abs_err.max():.4e}  mean {abs_err.mean():.4e}")
print(f"scale_rel:      max {scale_rel.max():.4e}  mean {scale_rel.mean():.4e}   (abs_err / mean|ref|)")
print(f"rel_err floored:max {rel_err.max():.4e}  mean {rel_err.mean():.4e}   (floor 0.1%*mean|ref|)")

# Per-row rel-err + scale-axis-bug detector.
# If the kernel indexes `scale[col_block, row_block]` (wrong axis), the
# per-128-row-band mean error becomes bimodal instead of uniform.
row_rel = rel_err.reshape(-1, N)[0]  # M=1
worst_rows = np.argsort(row_rel)[-5:][::-1]
print("worst 5 rows: " + ", ".join(f"{r}: {row_rel[r]:.3e}" for r in worst_rows))

band_means = np.array([row_rel[b * 128 : (b + 1) * 128].mean() for b in range(BN)])
band_max = band_means.max()
band_min = band_means.min()
band_ratio = band_max / max(band_min, 1e-30)
print(f"band mean rel_err (per 128-row block): {band_means}")
print(f"band max/min ratio: {band_ratio:.2f}  (axis-bug signal if >> 1)")

# Gate on the cancellation-robust metric (scale_rel) + the axis-bug band
# ratio. f32 accumulation floor for K=512 is ~sqrt(K)*eps ≈ 2.7e-6, so
# scale_rel up to ~1e-4 is normal f32 noise; we gate at 1e-3.
THRESHOLD_SCALE_REL = 1e-3
THRESHOLD_BAND_RATIO = 5.0

fail = []
if scale_rel.max() > THRESHOLD_SCALE_REL:
    fail.append(
        f"max scale_rel {scale_rel.max():.4e} > {THRESHOLD_SCALE_REL:.0e}"
    )
if band_ratio > THRESHOLD_BAND_RATIO:
    fail.append(
        f"band max/min ratio {band_ratio:.2f} > {THRESHOLD_BAND_RATIO} "
        "(possible scale-axis bug)"
    )
if fail:
    print("\nFAIL: " + "; ".join(fail))
    sys.exit(1)
print(f"\nOK: scale_rel.max {scale_rel.max():.4e} <= {THRESHOLD_SCALE_REL:.0e}, "
      f"band ratio {band_ratio:.2f} <= {THRESHOLD_BAND_RATIO}")
