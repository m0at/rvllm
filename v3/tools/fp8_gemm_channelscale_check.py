#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fp8_gemm_channelscale_check.py [sm_xxx] \
#                                                       [--shape=M,N,K]
#
# Validates `fp8_gemm_channelscale_sm121_kernel` against a pure-numpy
# fp64 reference. This is the native mma.sync replacement for the
# cuBLASLt scalar-fallback path on the prefill/M>1 side of Gemma 4 on
# GB10 — see `v3/GB10_SPEC.md` § "Remaining follow-ups".
#
# Layout contract:
#   A         : fp8 E4M3 [M, K]            activations
#   B         : fp8 E4M3 [N, K]            weights (pre-transposed)
#   a_scale   : f32      [M]               per-token
#   b_scale   : f32      [ceil(N/128), ceil(K/128)]   block-scale
#   C (out)   : f16      [M, N]            = (A*B^T dequant) * scales
#
# Pass gate: `scale_rel.max <= 5e-3` on every shape. FP8 is a 3-bit-
# mantissa format; error is dominated by quantisation noise + f32 mma
# accumulation + f16 store rounding — looser than the `fp8_gemv` 1e-3
# gate (that kernel doesn't round to f16).

import sys
import pathlib
import numpy as np
from cuda.bindings import driver as drv

np.seterr(divide="ignore", invalid="ignore")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = "sm_121"
shape_overrides = []
for arg in sys.argv[1:]:
    if arg.startswith("--shape="):
        shape_overrides.append(tuple(int(x) for x in arg[8:].split(",")))
    else:
        ARCH = arg

PTX = REPO / "kernels" / ARCH / "fp8_gemm_channelscale_sm121.ptx"
if not PTX.exists():
    sys.exit(f"missing PTX: {PTX}  (build with: kernels/build.sh {ARCH})")


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

cc_major = CHECK(
    drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
    ),
    "cc major",
)
cc_minor = CHECK(
    drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
    ),
    "cc minor",
)
print(f"device: cc {cc_major}.{cc_minor}   PTX: {PTX.name} ({ARCH})")

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(
    drv.cuModuleGetFunction(mod, b"fp8_gemm_channelscale_sm121_kernel"),
    "cuModuleGetFunction",
)


def e4m3_to_f32(b):
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        v = (m / 8.0) * (2.0 ** -6)
    elif e == 0xF and m == 0x7:
        v = 0.0
    else:
        v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return -v if s else v


LUT = np.array([e4m3_to_f32(b) for b in range(256)], dtype=np.float32)


def run_one(M: int, N: int, K: int, seed: int = 7) -> bool:
    if N % 128 != 0 or K % 128 != 0:
        print(f"  SKIP {M}x{N}x{K}: N/K must be 128-aligned")
        return True
    BN, BK = N // 128, K // 128
    rng = np.random.default_rng(seed)

    a_bytes = rng.integers(0, 256, size=(M, K), dtype=np.uint8)
    b_bytes = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
    for arr in (a_bytes, b_bytes):
        arr[arr == 0x7F] = 0x7E
        arr[arr == 0xFF] = 0xFE

    a_scale = rng.normal(0.0, 0.1, size=(M,)).astype(np.float32)
    b_scale = rng.normal(0.0, 0.1, size=(BN, BK)).astype(np.float32)

    a_f64 = LUT[a_bytes].astype(np.float64)
    b_f64 = LUT[b_bytes].astype(np.float64)
    b_f64 *= np.repeat(np.repeat(b_scale.astype(np.float64), 128, 0), 128, 1)
    a_f64 *= a_scale.astype(np.float64)[:, None]
    ref = (a_f64 @ b_f64.T).astype(np.float32)

    d_c  = CHECK(drv.cuMemAlloc(M * N * 2), "cuMemAlloc c")
    d_a  = CHECK(drv.cuMemAlloc(M * K), "cuMemAlloc a")
    d_b  = CHECK(drv.cuMemAlloc(N * K), "cuMemAlloc b")
    d_as = CHECK(drv.cuMemAlloc(M * 4), "cuMemAlloc as")
    d_bs = CHECK(drv.cuMemAlloc(BN * BK * 4), "cuMemAlloc bs")

    def h2d(dst, arr):
        arr = np.ascontiguousarray(arr)
        CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")

    h2d(d_a, a_bytes)
    h2d(d_b, b_bytes)
    h2d(d_as, a_scale)
    h2d(d_bs, b_scale)

    M_TILE = 16
    N_TILE = 128
    grid_x = N // N_TILE
    grid_y = (M + M_TILE - 1) // M_TILE

    params = [
        np.array([int(d_c)],  dtype=np.uint64),
        np.array([int(d_a)],  dtype=np.uint64),
        np.array([int(d_b)],  dtype=np.uint64),
        np.array([int(d_as)], dtype=np.uint64),
        np.array([int(d_bs)], dtype=np.uint64),
        np.array([M], dtype=np.int32),
        np.array([N], dtype=np.int32),
        np.array([K], dtype=np.int32),
        np.array([BK], dtype=np.int32),
    ]
    param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

    CHECK(drv.cuMemsetD8(d_c, 0, M * N * 2), "cuMemsetD8")
    CHECK(
        drv.cuLaunchKernel(
            fn,
            grid_x, grid_y, 1,
            128, 1, 1,
            0, 0,
            param_ptrs.ctypes.data, 0,
        ),
        "cuLaunchKernel",
    )
    CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

    out_f16 = np.empty((M, N), dtype=np.float16)
    CHECK(drv.cuMemcpyDtoH(out_f16.ctypes.data, d_c, out_f16.nbytes), "DtoH")
    out = out_f16.astype(np.float32)

    for h in (d_c, d_a, d_b, d_as, d_bs):
        CHECK(drv.cuMemFree(h), "cuMemFree")

    abs_err = np.abs(out - ref)
    ref_mean_abs = float(np.abs(ref).mean())
    scale_rel = float(abs_err.max() / max(ref_mean_abs, 1e-30))
    mean_rel = float(abs_err.mean() / max(ref_mean_abs, 1e-30))

    GATE = 5e-3
    status = "OK  " if scale_rel <= GATE else "FAIL"
    print(
        f"  {status}  M={M:>4} N={N:>5} K={K:>5}   "
        f"scale_rel.max {scale_rel:.3e}   mean {mean_rel:.3e}   "
        f"(grid {grid_x}×{grid_y})"
    )
    return scale_rel <= GATE


SHAPES = shape_overrides or [
    (8,   128, 128),
    (8,   256, 256),
    (16,  256, 256),
    (8,   512, 512),
    (8,  1152,1024),  # Gemma 4-ish (q_proj ≈ 4608 / 4 warps-of-N, hidden-ish K)
    (16, 2304,2048),
]

print(f"\nrunning {len(SHAPES)} shapes:")
all_pass = all(run_one(*s) for s in SHAPES)

print()
if all_pass:
    print("all shapes pass")
else:
    print("FAIL: some shapes exceeded the precision gate")
    sys.exit(1)
