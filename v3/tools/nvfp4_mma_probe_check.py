#!/usr/bin/env python3
# Smoke-test for the Blackwell native E2M1 tensor-core MMA
# (`mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32`).
#
# Loads trivial operand fragments per-lane and launches the probe
# kernel. The goal here is NOT to validate numerical correctness
# against a reference — the per-lane operand layout for this PTX is
# fixed by the spec in complex ways (each warp-lane holds a strided
# slice of the m×k=16×32 A and n×k=8×32 B matrices). Writing a
# matching host-side packer is the next step when we integrate this
# into the full FA2 kernel.
#
# For now the probe just verifies:
#   1. The PTX assembles on sm_121a.
#   2. The kernel launches without a CUDA error.
#   3. The accumulator outputs are finite fp32 (not NaN / all-zero,
#      unless every lane got zero operands — which is the case here
#      because we fill A and B with zeros → expected D == 0).
#
# A proper correctness gate requires the per-lane operand packer
# that mirrors the PTX thread-mapping. That lives in the FA2
# integration follow-up; this probe stays minimal on purpose.

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "nvfp4_mma_probe.ptx"
if not PTX.exists():
    sys.exit(
        f"missing PTX: {PTX}\n"
        f"  build with: nvcc -O3 -arch={ARCH}a -std=c++17 -ptx \\\n"
        f"              -o {PTX} {REPO}/kernels/nvfp4_mma_probe.cu"
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
fn = CHECK(drv.cuModuleGetFunction(mod, b"nvfp4_mma_probe_kernel"), "get fn")

# 32 lanes × 4 u32 A + 32 × 2 u32 B + 32 × 4 fp32 D.
a_host = np.zeros(32 * 4, dtype=np.uint32)
b_host = np.zeros(32 * 2, dtype=np.uint32)

d_a = CHECK(drv.cuMemAlloc(a_host.nbytes), "alloc a")
d_b = CHECK(drv.cuMemAlloc(b_host.nbytes), "alloc b")
d_d = CHECK(drv.cuMemAlloc(32 * 4 * 4), "alloc d")

CHECK(drv.cuMemcpyHtoD(d_a, a_host.ctypes.data, a_host.nbytes), "H2D a")
CHECK(drv.cuMemcpyHtoD(d_b, b_host.ctypes.data, b_host.nbytes), "H2D b")
CHECK(drv.cuMemsetD8(d_d, 0, 32 * 4 * 4), "zero d")

params = [
    np.array([int(d_a)], dtype=np.uint64),
    np.array([int(d_b)], dtype=np.uint64),
    np.array([int(d_d)], dtype=np.uint64),
]
pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)

CHECK(drv.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, pp.ctypes.data, 0), "launch")
CHECK(drv.cuCtxSynchronize(), "sync")

d_host = np.empty(32 * 4, dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(d_host.ctypes.data, d_d, d_host.nbytes), "D2H d")

finite_ok = np.isfinite(d_host).all()
zero_ok = np.all(d_host == 0.0)   # zero × zero → zero

print(f"PTX: {PTX.name} ({ARCH})")
print(f"D finite: {finite_ok}")
print(f"D all-zero (expected for zero operands): {zero_ok}")
print(f"D sample (first 8): {d_host[:8]}")

if not (finite_ok and zero_ok):
    print("FAIL: MMA produced unexpected output")
    sys.exit(1)
print("\nPTX path proven: native e2m1 MMA assembles, launches, and runs on hw.")
