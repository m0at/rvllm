#!/usr/bin/env python3
# Numerical-correctness probe for the Blackwell native E2M1 tensor-core
# MMA (`mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32`).
#
# **WIP** — the Test 2 packed MMA check currently FAILS. The PTX
# assembles and runs (Test 1 smoke passes), but the byte-level packer
# inherited from the FP8 MMA header does not produce correct e2m1
# MMA output. See the header comment in `kernels/nvfp4_mma_frag_pack.cuh`
# for the layout-isolation work still needed. This script is checked
# in as the testbed for that follow-up — running it shows the current
# failure signature (all-ones A × all-ones B returns 2.0 everywhere
# instead of K=64 or K=32).
#
# The PTX operand layout is fixed by the spec — lane `i` holds a
# strided slice of the m×k=16×64 A and n×k=8×64 B tiles (the K
# dimension is 64 NVFP4 elements per MMA, not 32 — `m16n8k32` in the
# mnemonic refers to the FP8-equivalent shape; each byte of the same
# register lane now packs two 4-bit values). We validate the two
# kernel entry points in `kernels/nvfp4_mma_probe.cu`:
#
#   (1) `nvfp4_mma_probe_kernel` — zero-input smoke test. Assertion:
#       output finite and all-zero for zero operands.
#
#   (2) `nvfp4_mma_packed_probe_kernel` — new. Drives the kernel-side
#       packer (`nvfp4_mma_frag_pack.cuh`) with a 16×64 A tile and
#       8×64 B tile of random NVFP4 values, compares the 16×8 f32
#       output tile to an fp64 reference that replays the dequant
#       + matmul on host. Gate: max abs error <= 2 * sum of per-
#       element magnitudes * fp32 ULP of the max accumulator — the
#       MMA is exact up to rounding since e2m1 values land on a
#       small discrete grid that fp32 reproduces without loss.

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


# --- E2M1 <-> float helpers ---------------------------------------------------
# Matches `kernels/nvfp4_utils.cuh` fp4_decode / fp4_encode exactly.
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def e2m1_decode(nibble: np.ndarray) -> np.ndarray:
    """nibble: uint8 array with e2m1 in low 4 bits each."""
    bits = nibble & 0xF
    mag = E2M1_MAG[bits & 0x7]
    return np.where(bits & 0x8, -mag, mag)


def e2m1_encode(values: np.ndarray) -> np.ndarray:
    """values: float array. returns uint8 nibble (low 4 bits)."""
    sign = (values < 0).astype(np.uint8) * 0x8
    mag = np.abs(values)
    e = np.zeros_like(mag, dtype=np.uint8)
    # Same threshold ladder as the kernel's fp4_encode.
    e = np.where(mag < 0.25, 0, e)
    e = np.where((mag >= 0.25) & (mag < 0.75), 1, e)
    e = np.where((mag >= 0.75) & (mag < 1.25), 2, e)
    e = np.where((mag >= 1.25) & (mag < 1.75), 3, e)
    e = np.where((mag >= 1.75) & (mag < 2.50), 4, e)
    e = np.where((mag >= 2.50) & (mag < 3.50), 5, e)
    e = np.where((mag >= 3.50) & (mag < 5.00), 6, e)
    e = np.where(mag >= 5.00, 7, e)
    return sign | e


def pack_e2m1_bytes(values: np.ndarray) -> np.ndarray:
    """values: [..., N] float; returns [..., N/2] uint8 with
    low nibble = even index, high nibble = odd index."""
    assert values.shape[-1] % 2 == 0
    nib = e2m1_encode(values)
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_e2m1_bytes(packed: np.ndarray) -> np.ndarray:
    """inverse of pack_e2m1_bytes."""
    lo_nib = packed & 0x0F
    hi_nib = (packed >> 4) & 0x0F
    lo = e2m1_decode(lo_nib)
    hi = e2m1_decode(hi_nib)
    out_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    out = np.empty(out_shape, dtype=np.float64)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


# --- CUDA boilerplate ---------------------------------------------------------

CHECK(drv.cuInit(0), "cuInit")
dev = CHECK(drv.cuDeviceGet(0), "cuDeviceGet")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
CHECK(drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load ptx")
fn_zero   = CHECK(drv.cuModuleGetFunction(mod, b"nvfp4_mma_probe_kernel"),
                  "get probe_kernel")
fn_packed = CHECK(drv.cuModuleGetFunction(mod, b"nvfp4_mma_packed_probe_kernel"),
                  "get packed_probe_kernel")


# --- Test 1: zero-input smoke (original probe behavior). ----------------------

a_host = np.zeros(32 * 4, dtype=np.uint32)
b_host = np.zeros(32 * 2, dtype=np.uint32)

d_a = CHECK(drv.cuMemAlloc(a_host.nbytes), "alloc a")
d_b = CHECK(drv.cuMemAlloc(b_host.nbytes), "alloc b")
d_d = CHECK(drv.cuMemAlloc(32 * 4 * 4), "alloc d")

CHECK(drv.cuMemcpyHtoD(d_a, a_host.ctypes.data, a_host.nbytes), "H2D a")
CHECK(drv.cuMemcpyHtoD(d_b, b_host.ctypes.data, b_host.nbytes), "H2D b")
CHECK(drv.cuMemsetD8(d_d, 0, 32 * 4 * 4), "zero d")

params = [np.array([int(p)], dtype=np.uint64) for p in (d_a, d_b, d_d)]
pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
CHECK(drv.cuLaunchKernel(fn_zero, 1, 1, 1, 32, 1, 1, 0, 0, pp.ctypes.data, 0),
      "launch zero")
CHECK(drv.cuCtxSynchronize(), "sync zero")

d_host = np.empty(32 * 4, dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(d_host.ctypes.data, d_d, d_host.nbytes), "D2H d")

zero_ok = bool(np.all(d_host == 0.0))
finite_ok = bool(np.isfinite(d_host).all())

for d in (d_a, d_b, d_d):
    CHECK(drv.cuMemFree(d), "free")

print(f"PTX: {PTX.name} ({ARCH})")
print(f"[smoke] D finite: {finite_ok}, D all-zero for zero-input: {zero_ok}")
if not (zero_ok and finite_ok):
    sys.exit("FAIL: smoke probe")


# --- Test 2: packed-input matmul against fp64 reference. ----------------------
# m=16, n=8, k=64 (NVFP4 element count). Random values chosen from
# the representable NVFP4 grid so quant noise is zero and any mismatch
# is a kernel-side bug.

rng = np.random.default_rng(2026)

def rand_on_grid(shape, rng):
    """Sample from the 15 non-zero representable e2m1 values +0.
    Equivalent to picking a uniform-random e2m1 bit pattern."""
    bits = rng.integers(0, 16, size=shape, dtype=np.uint8)
    return e2m1_decode(bits), bits

M, N, K = 16, 8, 64
a_f64, _ = rand_on_grid((M, K), rng)
b_f64, _ = rand_on_grid((N, K), rng)

a_bytes = pack_e2m1_bytes(a_f64)   # [16, 32]
b_bytes = pack_e2m1_bytes(b_f64)   # [ 8, 32]

# Sanity: round-trip through encode/decode is exact for on-grid
# inputs (we picked rand_on_grid precisely so this holds).
a_rt = unpack_e2m1_bytes(a_bytes)
b_rt = unpack_e2m1_bytes(b_bytes)
assert np.array_equal(a_rt, a_f64), "host-side A round-trip lost precision"
assert np.array_equal(b_rt, b_f64), "host-side B round-trip lost precision"

# fp64 reference: D = A @ B^T   (m=16 × n=8)
d_ref = (a_f64 @ b_f64.T).astype(np.float64)

# Launch the packed probe.
d_a = CHECK(drv.cuMemAlloc(a_bytes.nbytes), "alloc a bytes")
d_b = CHECK(drv.cuMemAlloc(b_bytes.nbytes), "alloc b bytes")
d_d = CHECK(drv.cuMemAlloc(M * N * 4), "alloc d tile")

CHECK(drv.cuMemcpyHtoD(d_a, a_bytes.ctypes.data, a_bytes.nbytes), "H2D a bytes")
CHECK(drv.cuMemcpyHtoD(d_b, b_bytes.ctypes.data, b_bytes.nbytes), "H2D b bytes")
CHECK(drv.cuMemsetD8(d_d, 0, M * N * 4), "zero d tile")

A_BYTES = M * 32
B_BYTES = N * 32
D_BYTES = M * N * 4
smem_bytes = A_BYTES + B_BYTES + D_BYTES

params = [np.array([int(p)], dtype=np.uint64) for p in (d_a, d_b, d_d)]
pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
CHECK(drv.cuLaunchKernel(fn_packed, 1, 1, 1, 32, 1, 1, smem_bytes, 0,
                         pp.ctypes.data, 0),
      "launch packed probe")
CHECK(drv.cuCtxSynchronize(), "sync packed probe")

d_tile = np.empty(M * N, dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(d_tile.ctypes.data, d_d, d_tile.nbytes), "D2H d tile")
d_dev = d_tile.reshape(M, N)

# Cleanup.
for d in (d_a, d_b, d_d):
    CHECK(drv.cuMemFree(d), "free packed")

# Bound: the MMA is exact up to fp32 rounding since all partial
# products land in small integer multiples of 0.25, 0.5, 0.75, ...
# Total magnitude bound: sum(|a_ik| * |b_jk|) over k per (i, j).
abs_prod = np.abs(a_f64) @ np.abs(b_f64).T
# Two orders of fp32 ULP (2 * 2^-23) per reduction step, times K
# terms, is a generous upper bound.
tol = 2.0 * np.finfo(np.float32).eps * abs_prod * K + 1e-6

max_err = float(np.max(np.abs(d_dev.astype(np.float64) - d_ref)))
rel_bound_hit = bool(np.all(np.abs(d_dev.astype(np.float64) - d_ref) <= tol))

print(f"[packed] d_ref[0, :4] = {d_ref[0, :4]}")
print(f"[packed] d_dev[0, :4] = {d_dev[0, :4].astype(np.float64)}")
print(f"[packed] max |d_dev - d_ref| = {max_err:.4e}, "
      f"per-element bound max = {float(tol.max()):.4e}")
print(f"[packed] within bound: {rel_bound_hit}")

if not rel_bound_hit:
    print("FAIL: packed MMA output diverged from fp64 reference")
    sys.exit(1)
print("\nPTX path + per-lane packer validated against fp64 on hw.")
