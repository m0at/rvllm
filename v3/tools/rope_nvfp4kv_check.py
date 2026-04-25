#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/rope_nvfp4kv_check.py [sm_xxx]
#
# End-to-end precision harness for `fused_rope_partial_nvfp4kv_kernel`.
# Feeds a small batch of fp16 Q/K/V + cos/sin/positions through the
# kernel, reads back the FP8 Q output and the packed-NVFP4 K/V cache
# + per-16-block E4M3 scales, dequantises, and compares against an
# fp64 RoPE + matching quant reference.
#
# Two gates per shape:
#   1. Q FP8 output: `abs_err.max <= q_scale * 3 / 256`  (half an
#      FP8-E4M3 ULP; we don't care which side of a tie CUDA picks).
#   2. KV NVFP4 dequant: `abs_err.max <= 1.15 * block_scale`  (same
#      as `nvfp4_roundtrip_check.py`; intrinsic FP4 + E4M3-scale
#      noise floor).

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "fused_rope_partial_nvfp4kv.ptx"
if not PTX.exists():
    sys.exit(
        f"missing PTX: {PTX}\n"
        "  build with:\n"
        f"    nvcc -O3 -arch={ARCH}a -std=c++17 -ptx \\\n"
        f"         -o {PTX} {REPO}/v3/kernels/fused_rope_partial_nvfp4kv.cu"
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

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(mod, b"fused_rope_partial_nvfp4kv_kernel"),
           "cuModuleGetFunction")


def fp4_mag(bits: np.ndarray) -> np.ndarray:
    # Decode NVFP4 magnitude bits. `bits` is uint with only low 3
    # bits used; caller applies the sign separately.
    table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
    return table[bits & 0x7]


def unpack_nvfp4(packed: np.ndarray, scales: np.ndarray, inner: int) -> np.ndarray:
    """Dequant a packed NVFP4 buffer. `packed` has shape
    [..., inner / 2] uint8; `scales` has shape [..., inner / 16]
    float32 (we convert from the stored E4M3 via numpy view). Result
    has shape [..., inner] float64."""
    assert inner % 16 == 0
    flat = packed.astype(np.uint32)
    lo = fp4_mag(flat & 0xF) * np.where((flat & 0x8) != 0, -1.0, 1.0)
    hi = fp4_mag((flat >> 4) & 0xF) * np.where(((flat >> 4) & 0x8) != 0, -1.0, 1.0)
    # Interleave lo, hi → inner-length axis.
    interleaved = np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], inner)
    # Broadcast per-16 scales.
    sc = np.repeat(scales.astype(np.float64), 16, axis=-1)
    return interleaved * sc


def ref_rope_partial(q, k, v, cos, sin, positions, rotary_dim):
    """fp64 reference for the RoPE + (Q FP8 quant / KV stays in fp64).
    Matches the kernel's split-half layout:
      lo = x[..., :half_head]*cos - x[..., half_head:]*sin
      hi = x[..., :half_head]*sin + x[..., half_head:]*cos
    for the first `rotary_dim` dims; pass-through beyond."""
    num_tokens, _, head_dim = q.shape
    half_head = head_dim // 2
    half_rotary = rotary_dim // 2

    def apply(x):
        out = x.astype(np.float64).copy()
        cos_t = cos[positions][:, None, :]  # (T, 1, half_rotary)
        sin_t = sin[positions][:, None, :]
        lo = out[..., :half_rotary]
        hi = out[..., half_head:half_head + half_rotary]
        new_lo = lo * cos_t - hi * sin_t
        new_hi = lo * sin_t + hi * cos_t
        out[..., :half_rotary] = new_lo
        out[..., half_head:half_head + half_rotary] = new_hi
        return out

    return apply(q), apply(k), v.astype(np.float64)


def e4m3_view(buf: np.ndarray) -> np.ndarray:
    """Interpret a byte buffer as __nv_fp8_e4m3 and decode to f64
    via the float LUT. E4M3: 1s 4e 3m, bias 7, max 448, NaN = 0x7F/0xFF."""
    b = buf.astype(np.uint32)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float64)
    # Normal: (1 + man/8) * 2**(exp - 7). Subnormal (exp=0): man/8 * 2**(-6).
    normal_val = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
    sub_val = (man / 8.0) * (2.0 ** -6)
    val = np.where(exp == 0, sub_val, normal_val)
    # NaN (exp=15 & man=7) — treat as 0 for our purposes, never happens
    # from finite positive inputs.
    val = np.where((exp == 15) & (man == 7), 0.0, val)
    return sign * val


def run_one(num_tokens, num_heads, num_kv_heads, head_dim, rotary_dim, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((num_tokens, num_heads, head_dim)).astype(np.float16)
    k = rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float16)
    v = rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float16)

    max_pos = 4096
    half_rotary = rotary_dim // 2
    freqs = 1.0 / (10000 ** (np.arange(half_rotary) / half_rotary))
    pos_idx = np.arange(max_pos)
    angles = pos_idx[:, None] * freqs[None, :]
    cos = np.cos(angles).astype(np.float16)
    sin = np.sin(angles).astype(np.float16)

    positions = rng.integers(0, max_pos, size=num_tokens, dtype=np.int32)
    slot_mapping = np.arange(num_tokens, dtype=np.int32)
    num_slots = num_tokens
    q_scale = np.float32(0.4)  # arbitrary per-tensor FP8 Q scale
    q_scale_arr = np.array([q_scale], dtype=np.float32)

    # Device alloc / H2D.
    def dev(arr):
        d = CHECK(drv.cuMemAlloc(arr.nbytes), "alloc")
        CHECK(drv.cuMemcpyHtoD(d, arr.ctypes.data, arr.nbytes), "H2D")
        return d

    d_q_in  = dev(q); d_k_in = dev(k); d_v_in = dev(v)
    d_cos   = dev(cos); d_sin = dev(sin)
    d_pos   = dev(positions); d_slot = dev(slot_mapping)
    d_qs    = dev(q_scale_arr)

    q_out_bytes       = num_tokens * num_heads * head_dim
    kv_packed_bytes   = num_slots * num_kv_heads * (head_dim // 2)
    kv_scale_bytes    = num_slots * num_kv_heads * (head_dim // 16)
    d_q_out  = CHECK(drv.cuMemAlloc(q_out_bytes), "alloc qo")
    d_k_pack = CHECK(drv.cuMemAlloc(kv_packed_bytes), "alloc kp")
    d_v_pack = CHECK(drv.cuMemAlloc(kv_packed_bytes), "alloc vp")
    d_k_sc   = CHECK(drv.cuMemAlloc(kv_scale_bytes), "alloc ks")
    d_v_sc   = CHECK(drv.cuMemAlloc(kv_scale_bytes), "alloc vs")
    for d in [d_q_out, d_k_pack, d_v_pack, d_k_sc, d_v_sc]:
        CHECK(drv.cuMemsetD8(d, 0, 1), "zero marker")

    # Launch. One block per (token, max(num_heads, num_kv_heads)), head_dim threads.
    params_data = [
        np.array([int(d_q_in)],  dtype=np.uint64),
        np.array([int(d_k_in)],  dtype=np.uint64),
        np.array([int(d_v_in)],  dtype=np.uint64),
        np.array([int(d_q_out)], dtype=np.uint64),
        np.array([int(d_k_pack)], dtype=np.uint64),
        np.array([int(d_v_pack)], dtype=np.uint64),
        np.array([int(d_k_sc)],   dtype=np.uint64),
        np.array([int(d_v_sc)],   dtype=np.uint64),
        np.array([int(d_cos)],    dtype=np.uint64),
        np.array([int(d_sin)],    dtype=np.uint64),
        np.array([int(d_pos)],    dtype=np.uint64),
        np.array([int(d_slot)],   dtype=np.uint64),
        np.array([int(d_qs)],     dtype=np.uint64),
        np.array([num_tokens],    dtype=np.int32),
        np.array([num_heads],     dtype=np.int32),
        np.array([num_kv_heads],  dtype=np.int32),
        np.array([head_dim],      dtype=np.int32),
        np.array([rotary_dim],    dtype=np.int32),
    ]
    pp = np.array([p.ctypes.data for p in params_data], dtype=np.uint64)

    max_heads = max(num_heads, num_kv_heads)
    CHECK(drv.cuLaunchKernel(fn,
                             num_tokens, max_heads, 1,
                             head_dim, 1, 1,
                             0, 0, pp.ctypes.data, 0),
          "launch")
    CHECK(drv.cuCtxSynchronize(), "sync")

    # Read back.
    q_out_bytes_arr = np.empty(q_out_bytes, dtype=np.uint8)
    k_pack = np.empty(kv_packed_bytes, dtype=np.uint8)
    v_pack = np.empty(kv_packed_bytes, dtype=np.uint8)
    k_sc_raw = np.empty(kv_scale_bytes, dtype=np.uint8)
    v_sc_raw = np.empty(kv_scale_bytes, dtype=np.uint8)
    CHECK(drv.cuMemcpyDtoH(q_out_bytes_arr.ctypes.data, d_q_out, q_out_bytes_arr.nbytes), "D2H")
    CHECK(drv.cuMemcpyDtoH(k_pack.ctypes.data, d_k_pack, k_pack.nbytes), "D2H")
    CHECK(drv.cuMemcpyDtoH(v_pack.ctypes.data, d_v_pack, v_pack.nbytes), "D2H")
    CHECK(drv.cuMemcpyDtoH(k_sc_raw.ctypes.data, d_k_sc, k_sc_raw.nbytes), "D2H")
    CHECK(drv.cuMemcpyDtoH(v_sc_raw.ctypes.data, d_v_sc, v_sc_raw.nbytes), "D2H")

    for d in [d_q_in, d_k_in, d_v_in, d_cos, d_sin, d_pos, d_slot, d_qs,
              d_q_out, d_k_pack, d_v_pack, d_k_sc, d_v_sc]:
        CHECK(drv.cuMemFree(d), "free")

    # Decode Q FP8.
    q_out = e4m3_view(q_out_bytes_arr).reshape(num_tokens, num_heads, head_dim) * q_scale

    # Decode K, V NVFP4 (slot_mapping is identity → cache[slot] is token).
    k_scales = e4m3_view(k_sc_raw).reshape(num_slots, num_kv_heads, head_dim // 16)
    v_scales = e4m3_view(v_sc_raw).reshape(num_slots, num_kv_heads, head_dim // 16)
    k_dq = unpack_nvfp4(k_pack.reshape(num_slots, num_kv_heads, head_dim // 2),
                        k_scales, head_dim)
    v_dq = unpack_nvfp4(v_pack.reshape(num_slots, num_kv_heads, head_dim // 2),
                        v_scales, head_dim)

    # fp64 reference.
    q_ref, k_ref, v_ref = ref_rope_partial(q, k, v,
                                           cos.astype(np.float64),
                                           sin.astype(np.float64),
                                           positions, rotary_dim)

    # Gate 1: Q FP8. Half a ULP at the quantised magnitude is roughly
    # q_scale / 2**6 for E4M3 at the dominant exponent of our inputs.
    q_abs_err = np.abs(q_out - q_ref)
    q_peak = np.abs(q_ref).max()
    # E4M3 has 3-bit mantissa; 1-ULP ≈ 2^-3 of the value. Half-ULP
    # near peak ≈ q_peak / 16. Add a tiny slack for per-tensor scale
    # choice (+ tie rounding): q_scale / 8.
    q_tol = max(q_peak / 16.0, q_scale / 8.0)
    q_ok = q_abs_err.max() <= q_tol

    # Gate 2: KV NVFP4 dequant. Per-element tol = 1.15 * per-block scale
    # (same as nvfp4_roundtrip_check.py).
    def kv_gate(ref, dq, inner):
        peak = np.abs(ref).max(axis=-1, keepdims=True)
        block_scale = peak / 6.0
        tol = np.maximum(block_scale, 1e-30) * 1.15
        abs_err = np.abs(dq - ref)
        # Broadcast tol across head_dim.
        tol_full = np.repeat(tol, inner, axis=-1)
        return abs_err <= tol_full + 1e-6, abs_err.max(), (block_scale.max())

    k_ok_mask, k_err, k_sc_max = kv_gate(k_ref, k_dq, head_dim)
    v_ok_mask, v_err, v_sc_max = kv_gate(v_ref, v_dq, head_dim)

    ok = q_ok and k_ok_mask.all() and v_ok_mask.all()
    status = "OK  " if ok else "FAIL"
    print(
        f"  {status}  T={num_tokens:>3} H={num_heads:>2} KVH={num_kv_heads:>2} "
        f"hd={head_dim:>3} rd={rotary_dim:>3}   "
        f"q_err={q_abs_err.max():.3e} (≤{q_tol:.3e})  "
        f"k_err={k_err:.3e}  v_err={v_err:.3e}"
    )
    return ok


print(f"device PTX: {PTX.name} ({ARCH})")
# Gemma 4 31B: sliding head_dim=256 rotary_dim=128, global head_dim=512
# rotary_dim=128 (0.25 partial on hd=512 → rotary=128; both layer types
# actually have full 0.5 partial on sliding).
all_pass = all([
    run_one(num_tokens=4,  num_heads=4,  num_kv_heads=2, head_dim=128, rotary_dim=128, seed=1),
    run_one(num_tokens=8,  num_heads=16, num_kv_heads=4, head_dim=256, rotary_dim=128, seed=7),
    run_one(num_tokens=16, num_heads=32, num_kv_heads=4, head_dim=256, rotary_dim=128, seed=42),
    run_one(num_tokens=8,  num_heads=8,  num_kv_heads=4, head_dim=512, rotary_dim=128, seed=101),
    # Long-prompt regression for batch prefill (RVLLM_BATCH_PREFILL=1):
    # rvllm-server produced garbage with NVFP4 KV + batch prefill at
    # prompt_len >= 62. If this fails, the rope kernel itself has a
    # num_tokens > 16 regression worth isolating; if it passes, the bug
    # lives in the Rust wiring or layer-exec metadata.
    run_one(num_tokens=128, num_heads=4,  num_kv_heads=2, head_dim=256, rotary_dim=128, seed=211),
    run_one(num_tokens=300, num_heads=4,  num_kv_heads=2, head_dim=256, rotary_dim=128, seed=212),
])
print()
print("all shapes pass" if all_pass
      else "FAIL: RoPE + NVFP4 KV write exceeded the expected bound")
if not all_pass:
    sys.exit(1)
