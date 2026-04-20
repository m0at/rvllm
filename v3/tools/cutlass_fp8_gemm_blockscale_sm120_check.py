#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/cutlass_fp8_gemm_blockscale_sm120_check.py
#
# Validates the CUTLASS SM120 blockwise FP8 GEMM + the SFA broadcast /
# SFB transpose prep kernels in `libcutlass_sm120.so` against an fp64
# numpy reference. Also measures wall-clock tok/s on each shape so we
# can eyeball the CUTLASS path vs. the hand-rolled `mma.sync` kernel.
#
# Gate: scale_rel.max ≤ 5e-3 on every shape.
# Layout (must match the .so):
#   A              : fp8 E4M3 [M, K]      row-major
#   B              : fp8 E4M3 [N, K]      row-major (read as ColumnMajor)
#   a_scale        : f32      [M]         per-token
#   b_chscale      : f32      [N/128, K/128]  row-major
#   output         : f16      [M, N]      row-major
# The .so helpers:
#   cutlass_fp8_gemm_blockscale_sm120_sfa_bytes(M, K)  -> SFA scratch bytes
#   cutlass_fp8_gemm_blockscale_sm120_sfb_bytes(N, K)  -> SFB scratch bytes
#   cutlass_fp8_gemm_blockscale_sm120_prep_sfa(a_scale, sfa, M, K, stream)
#   cutlass_fp8_gemm_blockscale_sm120_prep_sfb(b_chscale, sfb, N, K, stream)
#   cutlass_fp8_gemm_blockscale_sm120_workspace(M, N, K) -> CUTLASS workspace bytes
#   cutlass_fp8_gemm_blockscale_sm120(output, a, b, sfa, sfb, M, N, K,
#                                    ws, ws_bytes, stream)

import ctypes
import pathlib
import sys
import time

import numpy as np
from cuda.bindings import driver as drv

np.seterr(divide="ignore", invalid="ignore")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SO_CANDIDATES = [
    REPO / "kernels" / "sm_121" / "libcutlass_sm120.so",
    REPO / "kernels" / "sm_120" / "libcutlass_sm120.so",
]
SO = next((p for p in SO_CANDIDATES if p.exists()), None)
if SO is None:
    sys.exit(f"missing libcutlass_sm120.so in {SO_CANDIDATES}")
if not SO.exists():
    sys.exit(f"missing: {SO}  (build with kernels/build_cutlass_sm120_so.sh)")


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
print(f"device cc {cc_major}.{cc_minor}   .so: {SO}")

lib = ctypes.CDLL(str(SO))

lib.cutlass_fp8_gemm_blockscale_sm120_sfa_bytes.restype = ctypes.c_size_t
lib.cutlass_fp8_gemm_blockscale_sm120_sfa_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
lib.cutlass_fp8_gemm_blockscale_sm120_sfb_bytes.restype = ctypes.c_size_t
lib.cutlass_fp8_gemm_blockscale_sm120_sfb_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
lib.cutlass_fp8_gemm_blockscale_sm120_workspace.restype = ctypes.c_size_t
lib.cutlass_fp8_gemm_blockscale_sm120_workspace.argtypes = [ctypes.c_int] * 3

lib.cutlass_fp8_gemm_blockscale_sm120_prep_sfa.restype = ctypes.c_int
lib.cutlass_fp8_gemm_blockscale_sm120_prep_sfa.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
lib.cutlass_fp8_gemm_blockscale_sm120_prep_sfb.restype = ctypes.c_int
lib.cutlass_fp8_gemm_blockscale_sm120_prep_sfb.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
lib.cutlass_fp8_gemm_blockscale_sm120.restype = ctypes.c_int
lib.cutlass_fp8_gemm_blockscale_sm120.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
]


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


def alloc(n):
    return CHECK(drv.cuMemAlloc(n), f"cuMemAlloc {n}")


def free(p):
    CHECK(drv.cuMemFree(p), "cuMemFree")


def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")


def run_one(M, N, K, iters=50, seed=7):
    if M % 128 != 0 and M > 128:
        # SFA broadcast path only validated for M ≤ 128 here; max-reduce
        # covers M > 128 but we don't have a tight ref model for it.
        print(f"  SKIP {M}x{N}x{K}: keep M ≤ 128 for the ref check")
        return True
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

    # Keep scales small enough that the f16 output doesn't saturate at
    # large K — real Gemma 4 scales are ~1/K^0.5 order to keep the
    # dequant magnitude bounded.
    s = 1.0 / np.sqrt(max(K, 1))
    a_scale = np.abs(rng.normal(s, s * 0.1, size=(M,))).astype(np.float32)
    b_chscale = rng.normal(0.0, s, size=(BN, BK)).astype(np.float32)

    # Reference: SFA broadcast = max over M-tile, replicated across K.
    MB = (M + 127) // 128
    sfa_ref = np.zeros((MB, BK), dtype=np.float32)
    for mt in range(MB):
        lo = mt * 128
        hi = min(lo + 128, M)
        sfa_ref[mt, :] = a_scale[lo:hi].max()

    # fp64 reference: use the *tile-level* a_scale (max-reduced) to match
    # what the CUTLASS path actually computes (per-128-row granularity).
    a_f64 = LUT[a_bytes].astype(np.float64)
    b_f64 = LUT[b_bytes].astype(np.float64)
    b_f64 *= np.repeat(np.repeat(b_chscale.astype(np.float64), 128, 0), 128, 1)
    # Broadcast sfa_ref[m_tile, k_block] over the 128×128 atom.
    sfa_full = np.repeat(np.repeat(sfa_ref, 128, 0), 128, 1)[:M, :K]
    a_f64 *= sfa_full
    ref = (a_f64 @ b_f64.T).astype(np.float32)

    # Device memory.
    d_c = alloc(M * N * 2)
    d_a = alloc(M * K)
    d_b = alloc(N * K)
    d_as = alloc(M * 4)
    d_bs = alloc(BN * BK * 4)
    sfa_bytes = lib.cutlass_fp8_gemm_blockscale_sm120_sfa_bytes(M, K)
    sfb_bytes = lib.cutlass_fp8_gemm_blockscale_sm120_sfb_bytes(N, K)
    ws_bytes = lib.cutlass_fp8_gemm_blockscale_sm120_workspace(M, N, K)
    d_sfa = alloc(max(sfa_bytes, 16))
    d_sfb = alloc(max(sfb_bytes, 16))
    d_ws = alloc(max(ws_bytes, 16))

    h2d(d_a, a_bytes)
    h2d(d_b, b_bytes)
    h2d(d_as, a_scale)
    h2d(d_bs, b_chscale)

    # Prep SFA / SFB.
    rc = lib.cutlass_fp8_gemm_blockscale_sm120_prep_sfa(
        int(d_as), int(d_sfa), M, K, None
    )
    if rc != 0:
        sys.exit(f"prep_sfa rc={rc}")
    rc = lib.cutlass_fp8_gemm_blockscale_sm120_prep_sfb(
        int(d_bs), int(d_sfb), N, K, None
    )
    if rc != 0:
        sys.exit(f"prep_sfb rc={rc}")
    CHECK(drv.cuCtxSynchronize(), "sync after prep")

    # Warmup.
    for _ in range(3):
        rc = lib.cutlass_fp8_gemm_blockscale_sm120(
            int(d_c), int(d_a), int(d_b), int(d_sfa), int(d_sfb),
            M, N, K, int(d_ws), ws_bytes, None,
        )
        if rc != 0:
            sys.exit(f"gemm rc={rc} ({M}x{N}x{K})")
    CHECK(drv.cuCtxSynchronize(), "sync after warmup")

    # Timing.
    t0 = time.perf_counter()
    for _ in range(iters):
        lib.cutlass_fp8_gemm_blockscale_sm120(
            int(d_c), int(d_a), int(d_b), int(d_sfa), int(d_sfb),
            M, N, K, int(d_ws), ws_bytes, None,
        )
    CHECK(drv.cuCtxSynchronize(), "sync after bench")
    elapsed = time.perf_counter() - t0
    ms_per_gemm = 1000.0 * elapsed / iters
    tflops = 2.0 * M * N * K * iters / elapsed / 1e12

    # Correctness.
    out_f16 = np.empty((M, N), dtype=np.float16)
    CHECK(drv.cuMemcpyDtoH(out_f16.ctypes.data, d_c, out_f16.nbytes), "DtoH")
    out = out_f16.astype(np.float32)

    for h in (d_c, d_a, d_b, d_as, d_bs, d_sfa, d_sfb, d_ws):
        free(h)

    abs_err = np.abs(out - ref)
    ref_mean_abs = float(np.abs(ref).mean())
    scale_rel = float(abs_err.max() / max(ref_mean_abs, 1e-30))
    mean_rel = float(abs_err.mean() / max(ref_mean_abs, 1e-30))

    GATE = 5e-3
    status = "OK  " if scale_rel <= GATE else "FAIL"
    print(
        f"  {status}  M={M:>4} N={N:>5} K={K:>5}   "
        f"scale_rel.max {scale_rel:.3e}   mean {mean_rel:.3e}   "
        f"{ms_per_gemm:.3f}ms/gemm  {tflops:.2f} TFLOPS"
    )
    return scale_rel <= GATE


SHAPES = [
    (128,  128,  128),
    (128,  256,  256),
    (128, 1152, 1024),
    (128, 2304, 2048),
    (128, 4608, 5376),   # Gemma 4 QKV prefill, batch=128 tokens
    (256, 4608, 5376),
]

print(f"\nrunning {len(SHAPES)} shapes:")
all_pass = all(run_one(*s) for s in SHAPES)

print()
if all_pass:
    print("all shapes pass")
else:
    print("FAIL: some shapes exceeded the precision gate")
    sys.exit(1)
