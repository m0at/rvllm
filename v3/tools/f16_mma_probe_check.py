#!/usr/bin/env python3
# Numerical-correctness probe for the SM80-era f16 tensor-core MMA
# (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`).
#
# Phase 1 of task aa01001nvf4f16mma — validates the per-lane packer
# layout in `kernels/f16_mma_frag_pack.cuh` against an fp64 reference.
# Pass gate is a precondition for the FA2 NVFP4 kernel rewrite that
# replaces dequant-to-fp32 + scalar FMA with dequant-to-f16 + f16 MMA.

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "f16_mma_probe.ptx"
if not PTX.exists():
    sys.exit(
        f"missing PTX: {PTX}\n"
        f"  build with: nvcc -O3 -arch={ARCH}a -std=c++17 -ptx \\\n"
        f"              -I kernels -o {PTX} {REPO}/kernels/f16_mma_probe.cu"
    )


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

mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load ptx")
fn  = CHECK(drv.cuModuleGetFunction(mod, b"f16_mma_probe_kernel"), "get fn")


def run_case(label, a_f64, b_f64):
    """A: [16, 16] f64; B: [8, 16] f64. Returns (pass, device_output, ref)."""
    M, K = a_f64.shape
    N, Kb = b_f64.shape
    assert (M, K) == (16, 16) and (N, Kb) == (8, 16)

    a_f16 = a_f64.astype(np.float16)
    b_f16 = b_f64.astype(np.float16)

    d_a = CHECK(drv.cuMemAlloc(a_f16.nbytes), "alloc a")
    d_b = CHECK(drv.cuMemAlloc(b_f16.nbytes), "alloc b")
    d_d = CHECK(drv.cuMemAlloc(M * N * 4), "alloc d")
    try:
        CHECK(drv.cuMemcpyHtoD(d_a, a_f16.ctypes.data, a_f16.nbytes), "H2D a")
        CHECK(drv.cuMemcpyHtoD(d_b, b_f16.ctypes.data, b_f16.nbytes), "H2D b")
        CHECK(drv.cuMemsetD8(d_d, 0, M * N * 4), "zero d")

        A_BYTES = 16 * 16 * 2
        B_BYTES =  8 * 16 * 2
        D_BYTES = 16 *  8 * 4
        smem = A_BYTES + B_BYTES + D_BYTES

        # Keep per-argument wrapper arrays alive through the launch.
        pa = np.array([int(d_a)], dtype=np.uint64)
        pb = np.array([int(d_b)], dtype=np.uint64)
        pd = np.array([int(d_d)], dtype=np.uint64)
        pp = np.array([pa.ctypes.data, pb.ctypes.data, pd.ctypes.data],
                      dtype=np.uint64)

        CHECK(drv.cuLaunchKernel(
            fn, 1, 1, 1, 32, 1, 1, smem, 0, pp.ctypes.data, 0), "launch")
        CHECK(drv.cuCtxSynchronize(), "sync")

        d_host = np.empty(M * N, dtype=np.float32)
        CHECK(drv.cuMemcpyDtoH(d_host.ctypes.data, d_d, d_host.nbytes), "D2H d")
        d_dev = d_host.reshape(M, N)
    finally:
        for d in (d_a, d_b, d_d):
            CHECK(drv.cuMemFree(d), "free")

    # fp64 reference: run the matmul on the f16-rounded values so we
    # only measure MMA + packer correctness, not host-side fp16 quant
    # noise. D = A_f16 @ B_f16.T, accumulated in fp64.
    a_ref = a_f16.astype(np.float64)
    b_ref = b_f16.astype(np.float64)
    d_ref = a_ref @ b_ref.T

    # Tolerance: 4 × fp32 ULP × per-output magnitude bound × K, plus
    # a fixed floor for near-zero outputs. The MMA accumulates in f32
    # so per-element error should sit near machine epsilon of the
    # magnitude.
    abs_prod = np.abs(a_ref) @ np.abs(b_ref).T
    tol = 4.0 * np.finfo(np.float32).eps * abs_prod * a_ref.shape[1] + 1e-3

    max_err = float(np.max(np.abs(d_dev.astype(np.float64) - d_ref)))
    ok = bool(np.all(np.abs(d_dev.astype(np.float64) - d_ref) <= tol))

    print(f"[{label}] max |d_dev - d_ref| = {max_err:.4e}, "
          f"per-element bound max = {float(tol.max()):.4e}  "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"  d_ref[0, :] = {d_ref[0]}")
        print(f"  d_dev[0, :] = {d_dev[0].astype(np.float64)}")
    return ok


print(f"PTX: {PTX.name} ({ARCH})")
rng = np.random.default_rng(2026)

results = []

# (1) All-ones — D should equal K = 16 everywhere.
a = np.ones((16, 16), dtype=np.float64)
b = np.ones((8, 16), dtype=np.float64)
results.append(run_case("all-ones", a, b))

# (2) Identity-ish: A[i, i] = 1, B[j, j] = 1 for i, j < 8 — D should
# be an 8×8 identity embedded into [16, 8], zeros elsewhere.
a = np.zeros((16, 16), dtype=np.float64)
for i in range(16):
    a[i, i] = 1.0
b = np.zeros((8, 16), dtype=np.float64)
for j in range(8):
    b[j, j] = 1.0
results.append(run_case("identity", a, b))

# (3) Random small-magnitude — exercises every (row, col) dot product.
a = rng.standard_normal((16, 16)) * 0.5
b = rng.standard_normal(( 8, 16)) * 0.5
results.append(run_case("random-0.5", a, b))

# (4) Random wider-range — exercises fp16 rounding + sign mix.
a = rng.standard_normal((16, 16)) * 2.0
b = rng.standard_normal(( 8, 16)) * 2.0
results.append(run_case("random-2.0", a, b))

print()
if not all(results):
    print("FAIL: one or more cases diverged from the fp64 reference")
    sys.exit(1)
print("f16 m16n8k16 MMA packer + PTX validated against fp64 on hw.")
