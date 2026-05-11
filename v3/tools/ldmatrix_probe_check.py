#!/usr/bin/env python3
# Usage: ~/.venv/bin/python3 v3/tools/ldmatrix_probe_check.py [sm_xxx]
#
# F10 probe — runs `kernels/<sm>/ldmatrix_probe.ptx` and compares the
# per-lane output of the manual `pack_a_frag_row_major_m16k32` packer
# (used by the unified prefill today) against a single
# `ldmatrix.sync.aligned.m8n8.x4.shared.b16` instruction. Prints
# both fragments per lane and flags matches / mismatches, so we can
# pick a smem-stride / address-pattern that makes ldmatrix a
# drop-in replacement.

import sys, pathlib
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "ldmatrix_probe.ptx"
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
mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(mod, b"ldmatrix_a_probe_kernel"), "getfn")

import ctypes
manual_bytes = 32 * 4 * 4
ldmat_bytes = 32 * 4 * 4
d_man = CHECK(drv.cuMemAlloc(manual_bytes), "alloc man")
d_ldm = CHECK(drv.cuMemAlloc(ldmat_bytes), "alloc ldm")
CHECK(drv.cuMemsetD8(d_man, 0, manual_bytes), "memset1")
CHECK(drv.cuMemsetD8(d_ldm, 0, ldmat_bytes), "memset2")

import numpy as np
p1 = np.array([int(d_man)], dtype=np.uint64)
p2 = np.array([int(d_ldm)], dtype=np.uint64)
pp = np.array([p1.ctypes.data, p2.ctypes.data], dtype=np.uint64)
CHECK(drv.cuLaunchKernel(fn, 1, 1, 1, 128, 1, 1, 0, 0, pp.ctypes.data, 0),
      "launch")
CHECK(drv.cuCtxSynchronize(), "sync")

man = np.empty((32, 4), dtype=np.uint32)
ldm = np.empty((32, 4), dtype=np.uint32)
CHECK(drv.cuMemcpyDtoH(man.ctypes.data, d_man, man.nbytes), "d2h man")
CHECK(drv.cuMemcpyDtoH(ldm.ctypes.data, d_ldm, ldm.nbytes), "d2h ldm")

matches = (man == ldm)
print(f"Per-lane 4-u32 manual pack vs ldmatrix.m8n8.x4.b16:")
print(f"  lanes matching position-for-position: {matches.all(axis=1).sum()}/32")
print(f"  total u32 matching: {matches.sum()}/{man.size}")

# Show first 8 lanes side-by-side
print(f"\n{'lane':>4}  {'manual a[0..3]':<40}  {'ldmat a[0..3]':<40}")
for lane in range(8):
    m = ' '.join(f"{x:08x}" for x in man[lane])
    l = ' '.join(f"{x:08x}" for x in ldm[lane])
    print(f"  {lane:>2}  {m:<40}  {l:<40}")

# Decode expected bytes per our spec:
#   a[0]: row (lane/4),    k = (lane%4)*8 + [0..3]
#   a[1]: row (lane/4+8),  k = (lane%4)*8 + [0..3]
#   a[2]: row (lane/4),    k = (lane%4)*8 + [4..7]
#   a[3]: row (lane/4+8),  k = (lane%4)*8 + [4..7]
# With source s[row, col] = (row & 0xF) << 4 | (col & 0xF):
expected = np.zeros((32, 4), dtype=np.uint32)
for lane in range(32):
    r_lo = lane // 4
    r_hi = r_lo + 8
    k0 = (lane % 4) * 8
    for idx, (r, k_off) in enumerate([(r_lo, 0), (r_hi, 0), (r_lo, 4), (r_hi, 4)]):
        v = 0
        for b in range(4):
            byte = ((r & 0xF) << 4) | ((k0 + k_off + b) & 0xF)
            v |= byte << (b * 8)
        expected[lane, idx] = v
print(f"\nmanual == expected MMA layout: {(man == expected).all()}")
