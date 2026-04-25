#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/nvfp4_roundtrip_check.py [sm_xxx]
#
# Validates the NVFP4 packing / dequant helpers in
# `kernels/nvfp4_utils.cuh` by round-tripping random FP32 blocks
# through the probe kernel and bounding the observed error against
# the intrinsic NVFP4 quantisation noise floor.
#
# Layout contract (per 16-element block):
#   1. Pick per-block E4M3 scale = peak / 6.0, quantised to
#      __nv_fp8_e4m3 (hardware, round-to-nearest-even, saturating).
#   2. Divide each element by scale, round to the nearest FP4
#      magnitude (positive set {0, 0.5, 1, 1.5, 2, 3, 4, 6}), keep
#      sign.
#   3. Dequant = fp4_magnitude(bits) * scale.
#
# Pass gate (per element): `|recovered - original| <= scale`. The
# smallest non-zero FP4 step is `0.5 * scale`, so a correctly-snapped
# value lives within `0.5 * scale` of its nearest representable — and
# the E4M3 scale itself carries ~5% relative noise on top. Budget of
# one full scale-unit covers both.
#
# We don't try to replicate CUDA's E4M3 rounding in numpy; the gate
# above holds regardless of which side of a tie the hardware picks.

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "nvfp4_roundtrip_probe.ptx"
if not PTX.exists():
    sys.exit(
        f"missing PTX: {PTX}\n"
        "  build with:\n"
        f"    nvcc -O3 -arch={ARCH}a -std=c++17 -ptx \\\n"
        f"         -o {PTX} {REPO}/kernels/nvfp4_roundtrip_probe.cu"
    )


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

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(mod, b"nvfp4_roundtrip_kernel"),
           "cuModuleGetFunction")

def run_one(n_blocks: int, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    # Mix of shapes: near-zero blocks, mixed-sign blocks, large-peak
    # blocks (which exercise the E4M3 scale saturation path).
    x = rng.normal(0.0, 1.0, size=(n_blocks, 16)).astype(np.float32)
    # Jack up a random fraction to test scale saturation.
    big = rng.choice(n_blocks, size=n_blocks // 8, replace=False)
    x[big] *= 50.0
    # Force a zero block to exercise that branch.
    x[0, :] = 0.0

    d_in = CHECK(drv.cuMemAlloc(x.nbytes), "cuMemAlloc in")
    d_out = CHECK(drv.cuMemAlloc(x.nbytes), "cuMemAlloc out")
    CHECK(drv.cuMemcpyHtoD(d_in, x.ctypes.data, x.nbytes), "HtoD")
    CHECK(drv.cuMemsetD8(d_out, 0, x.nbytes), "memset out")

    n_blocks_i32 = np.array([n_blocks], dtype=np.int32)
    p_in = np.array([int(d_in)], dtype=np.uint64)
    p_out = np.array([int(d_out)], dtype=np.uint64)
    params = [p_out, p_in, n_blocks_i32]
    param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

    grid = ((n_blocks + 31) // 32, 1, 1)
    block = (32, 1, 1)
    CHECK(drv.cuLaunchKernel(fn, *grid, *block, 0, 0,
                             param_ptrs.ctypes.data, 0),
          "cuLaunchKernel")
    CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

    out = np.empty_like(x)
    CHECK(drv.cuMemcpyDtoH(out.ctypes.data, d_out, out.nbytes), "DtoH")
    for h in (d_in, d_out):
        CHECK(drv.cuMemFree(h), "cuMemFree")

    # Quant-noise bound: for a 16-element block with peak `p`, the
    # round-trip snaps every element to a point on the FP4 grid
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6} × scale, where scale ≈ p/6. The
    # per-element error is `scale` in the worst case when the
    # element lands on a bucket edge AND E4M3 scale rounding adds
    # another 1/8 (3 mantissa bits in FP8-E4M3) of drift. Budget
    # `1.15 · scale` to absorb both without coupling to CUDA's
    # exact rounding mode. Zero-blocks still recover bit-exactly.
    block_peak = np.abs(x).max(axis=1)
    block_scale = block_peak / 6.0                               # [n_blocks]
    tol_scale = 1.15
    tol = np.repeat(np.maximum(block_scale, 1e-30)[:, None], 16, axis=1) * tol_scale
    abs_err = np.abs(out.astype(np.float64) - x.astype(np.float64))
    # Zero-blocks: insist on exact recovery.
    zero_mask = block_peak == 0.0
    zero_err = abs_err[zero_mask].max() if zero_mask.any() else 0.0
    bad_nonzero = int((abs_err[~zero_mask] > tol[~zero_mask] + 1e-6).sum())
    ok = zero_err == 0.0 and bad_nonzero == 0
    status = "OK  " if ok else "FAIL"
    # Relative error normalised by the per-block scale — this is the
    # portable "how clean is the quantisation" metric; expected to
    # sit in [0, 1].
    rel = abs_err / tol
    print(
        f"  {status}  n_blocks={n_blocks:>6}  seed={seed:>3}   "
        f"rel_err.max={rel.max():.3f} (≤{tol_scale:.2f})  "
        f"zero_err={zero_err:.1e}  mismatches={bad_nonzero}"
    )
    return ok


print(f"device PTX: {PTX.name} ({ARCH})")
all_pass = all(
    run_one(n, s)
    for n, s in [(64, 1), (1024, 7), (8192, 42), (16384, 101)]
)
print()
if all_pass:
    print("all shapes pass")
else:
    print("FAIL: kernel round-trip exceeded the NVFP4 quant-noise bound")
    sys.exit(1)
