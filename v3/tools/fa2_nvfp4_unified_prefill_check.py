#!/usr/bin/env python3
# Numerical-correctness harness for
# `flash_attention_2_prefill_nvfp4kv_unified_kernel` (Phase 2b of task
# aa01001nvf4f16mma — commit 976792a).
#
# The existing fa2_nvfp4_prefill_check.py validates the per-qi variant
# via the RoPE kernel for cache setup; that pipeline doesn't transfer
# cleanly to the unified kernel because the argument list + grid
# geometry differ. This harness builds the NVFP4 KV cache + FP8 Q
# directly via numpy quantisation, launches the unified kernel with
# a single sequence, and compares the f16 output against an fp64
# reference that replays the attention on the dequanted tensors.
#
# Gate: abs_err ≤ 5e-3 · peak(|ref|), matching the per-qi harness.

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "flash_attention_unified_prefill_nvfp4kv.ptx"
if not PTX.exists():
    sys.exit(
        f"missing PTX: {PTX}\n"
        f"  build: nvcc -O3 -arch={ARCH}a -std=c++17 -ptx -I kernels \\\n"
        f"         -o {PTX} {REPO}/kernels/flash_attention_unified_prefill_nvfp4kv.cu"
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
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "ctx")
CHECK(drv.cuCtxSetCurrent(ctx), "set ctx")
mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load ptx")
fn  = CHECK(drv.cuModuleGetFunction(
    mod, b"flash_attention_2_prefill_nvfp4kv_unified_kernel"), "get fn")

SMEM_OPT_IN = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


# --- NVFP4 quant helpers (match kernels/nvfp4_utils.cuh) ---------------------
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def e2m1_decode(nibbles: np.ndarray) -> np.ndarray:
    bits = nibbles & 0xF
    mag = E2M1_MAG[bits & 0x7]
    return np.where(bits & 0x8, -mag, mag)


def nvfp4_quantise_row(row_f32: np.ndarray, block: int = 16):
    """Per-16-element microscale quantisation. Returns (packed_bytes, scales).
    scales: E4M3 bytes, packed: nibble pairs per byte, low=even high=odd.
    Mirrors the kernel's encode path in `pack16_fp32_to_nvfp4` +
    `block_scale_e4m3`."""
    assert row_f32.shape[-1] % block == 0
    D = row_f32.shape[-1]
    blocks = row_f32.reshape(*row_f32.shape[:-1], D // block, block)
    peak = np.abs(blocks).max(axis=-1)
    # Scale chosen so max-magnitude maps near 6.0 (NVFP4 max).
    scale_raw = np.where(peak > 0, peak / 6.0, 1.0).astype(np.float32)
    scale_e4m3 = encode_e4m3(scale_raw).astype(np.uint8)
    scale_f64 = e4m3_to_f64(scale_e4m3)
    inv = np.where(scale_f64 > 0, 1.0 / scale_f64, 0.0)
    scaled = blocks.astype(np.float64) * inv[..., None]
    nibbles = encode_e2m1(scaled)
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8).reshape(*row_f32.shape[:-1], D // 2)
    return packed, scale_e4m3


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    sign = (values < 0).astype(np.uint8) * 0x8
    mag = np.abs(values)
    e = np.zeros_like(mag, dtype=np.uint8)
    for lo, hi, v in [(0, 0.25, 0), (0.25, 0.75, 1), (0.75, 1.25, 2),
                      (1.25, 1.75, 3), (1.75, 2.5, 4), (2.5, 3.5, 5),
                      (3.5, 5.0, 6), (5.0, 1e12, 7)]:
        e = np.where((mag >= lo) & (mag < hi), v, e)
    return sign | e


def encode_e4m3(vals: np.ndarray) -> np.ndarray:
    """Naive f32 → E4M3 via CUDA semantics; enough for scale slot."""
    v = np.abs(vals.astype(np.float32))
    sign = (vals < 0).astype(np.uint32) << 7
    out = np.zeros_like(v, dtype=np.uint8)
    nz = v > 0
    if nz.any():
        logs = np.floor(np.log2(v[nz]))
        # E4M3 exp bias = 7, subnormal boundary at e = 0.
        exp = np.clip(logs + 7, 0, 15).astype(np.uint32)
        frac = v[nz] / (2.0 ** (exp - 7)) - 1.0
        man = np.clip(np.round(frac * 8.0), 0, 7).astype(np.uint32)
        # Handle rounding overflow.
        over = man >= 8
        man = np.where(over, 0, man)
        exp = np.where(over, exp + 1, exp)
        exp = np.clip(exp, 0, 15)
        man = np.where(exp == 15, np.minimum(man, 6), man)  # clamp to 448
        b = (exp << 3) | man
        result = np.zeros_like(v, dtype=np.uint32)
        result[nz] = b
        out = (sign | result).astype(np.uint8)
    else:
        out = sign.astype(np.uint8)
    return out


def e4m3_to_f64(buf: np.ndarray) -> np.ndarray:
    b = buf.astype(np.uint32)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float64)
    val = np.where(exp == 0,
                   (man / 8.0) * (2.0 ** -6),
                   (1.0 + man / 8.0) * (2.0 ** (exp - 7)))
    val = np.where((exp == 15) & (man == 7), 0.0, val)
    return sign * val


def fp8_e4m3_quantise(x: np.ndarray, scale: float) -> np.ndarray:
    """f32 → FP8 E4M3 bytes after dividing by scale."""
    v = x.astype(np.float32) / scale
    return encode_e4m3(v)


def fp8_e4m3_decode(buf: np.ndarray) -> np.ndarray:
    return e4m3_to_f64(buf)


# --- Test shape + dispatch ---------------------------------------------------


def run_unified(num_heads, num_kv_heads, head_dim, q_len, block_size,
                tile_size, seed=7):
    """Single-sequence self-prefill (query_len = context_len). Matches
    the production case where run_generate dispatches through this
    kernel on the Prefill phase."""
    rng = np.random.default_rng(seed)
    num_seqs    = 1
    ctx_len     = q_len
    num_queries_per_kv = num_heads // num_kv_heads
    block_q     = 16 // num_queries_per_kv
    blocks_per_seq = (ctx_len + block_size - 1) // block_size

    # Raw f32 Q, K, V.
    q_f32 = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32) * 0.5
    k_f32 = rng.standard_normal((ctx_len, num_kv_heads, head_dim)).astype(np.float32) * 0.5
    v_f32 = rng.standard_normal((ctx_len, num_kv_heads, head_dim)).astype(np.float32) * 0.5

    # Quantise Q (FP8 per-tensor).
    q_scale = 0.5
    q_fp8 = fp8_e4m3_quantise(q_f32, q_scale).reshape(q_len * num_heads * head_dim)
    q_f64 = fp8_e4m3_decode(q_fp8.reshape(q_len, num_heads, head_dim)) * q_scale

    # Quantise KV (NVFP4 per-16-elem-microscale).
    k_packed, k_scales = nvfp4_quantise_row(k_f32.astype(np.float64))
    v_packed, v_scales = nvfp4_quantise_row(v_f32.astype(np.float64))

    # Build paged layout: [num_blocks, block_size, num_kv_heads, head_dim/{2,16}].
    num_blocks = blocks_per_seq * num_seqs
    k_cache_packed = np.zeros((num_blocks, block_size, num_kv_heads, head_dim // 2),
                              dtype=np.uint8)
    v_cache_packed = np.zeros_like(k_cache_packed)
    k_cache_scale  = np.zeros((num_blocks, block_size, num_kv_heads, head_dim // 16),
                              dtype=np.uint8)
    v_cache_scale  = np.zeros_like(k_cache_scale)
    for t in range(ctx_len):
        blk = t // block_size
        off = t % block_size
        k_cache_packed[blk, off] = k_packed[t]
        v_cache_packed[blk, off] = v_packed[t]
        k_cache_scale [blk, off] = k_scales[t]
        v_cache_scale [blk, off] = v_scales[t]

    block_table = np.arange(num_blocks, dtype=np.int32).reshape(num_seqs, blocks_per_seq)
    cu_seqlens_q = np.array([0, q_len], dtype=np.int32)
    context_lens = np.array([ctx_len], dtype=np.int32)

    # --- Device alloc + copy ---
    def alloc(n): return CHECK(drv.cuMemAlloc(n), f"alloc {n}")

    def h2d(d, a):
        CHECK(drv.cuMemcpyHtoD(d, a.ctypes.data, a.nbytes), "h2d")

    d_out = alloc(q_len * num_heads * head_dim * 2)
    d_q   = alloc(q_fp8.nbytes); h2d(d_q, q_fp8)
    d_kp  = alloc(k_cache_packed.nbytes); h2d(d_kp, k_cache_packed)
    d_vp  = alloc(v_cache_packed.nbytes); h2d(d_vp, v_cache_packed)
    d_ks  = alloc(k_cache_scale.nbytes);  h2d(d_ks, k_cache_scale)
    d_vs  = alloc(v_cache_scale.nbytes);  h2d(d_vs, v_cache_scale)
    d_bt  = alloc(block_table.nbytes); h2d(d_bt, block_table)
    d_cu  = alloc(cu_seqlens_q.nbytes); h2d(d_cu, cu_seqlens_q)
    d_ctx = alloc(context_lens.nbytes); h2d(d_ctx, context_lens)
    q_descale = np.array([q_scale], dtype=np.float32)
    d_qd  = alloc(4); h2d(d_qd, q_descale)
    CHECK(drv.cuMemsetD8(d_out, 0, q_len * num_heads * head_dim * 2), "zero out")

    attn_scale = float(1.0 / np.sqrt(head_dim))

    # --- Smem budget (matches kernel's inline layout) ---
    BLOCK_M = 16
    MMA_K   = 16
    def sz_half(n): return n * 2
    def sz_float(n): return n * 4
    # Must match the kernel's extern __shared__ allocation order.
    smem = (
        sz_half(BLOCK_M * head_dim) +           # s_q_f16
        sz_float(BLOCK_M) +                     # s_q_scale
        sz_half(tile_size * head_dim) +         # s_k_f16
        sz_half(tile_size * head_dim) +         # s_v_f16
        sz_half(MMA_K * head_dim) +             # s_v_f16_T
        sz_float(BLOCK_M * max(tile_size, MMA_K)) +  # s_s (sized for tile_size)
        sz_float(BLOCK_M) +                     # s_m
        sz_float(BLOCK_M) +                     # s_l
        sz_float(BLOCK_M) +                     # s_alpha
        sz_half(BLOCK_M * MMA_K) +              # s_p_f16
        sz_float(BLOCK_M) +                     # s_p_scale
        sz_float(BLOCK_M * head_dim) +          # s_acc
        256                                     # alignment cushion
    )
    if smem >= 48 * 1024:
        CHECK(drv.cuFuncSetAttribute(fn, SMEM_OPT_IN, smem), "smem opt-in")

    # total_num_q_blocks = sum(ceil(q_len/block_q)) + num_seqs (see kernel).
    total_q_blocks = (q_len + block_q - 1) // block_q + num_seqs

    # q_scale_cache = nullptr → kernel falls back to *q_descale per row.
    params = [
        np.array([int(d_out)], dtype=np.uint64),
        np.array([int(d_q)],   dtype=np.uint64),
        np.array([int(d_kp)],  dtype=np.uint64),
        np.array([int(d_vp)],  dtype=np.uint64),
        np.array([int(d_ks)],  dtype=np.uint64),
        np.array([int(d_vs)],  dtype=np.uint64),
        np.array([0],          dtype=np.uint64),   # q_scale_cache = nullptr
        np.array([int(d_bt)],  dtype=np.uint64),
        np.array([int(d_cu)],  dtype=np.uint64),
        np.array([int(d_ctx)], dtype=np.uint64),
        np.array([int(d_qd)],  dtype=np.uint64),
        np.array([attn_scale], dtype=np.float32),
        np.array([num_heads],  dtype=np.int32),
        np.array([num_kv_heads], dtype=np.int32),
        np.array([head_dim],   dtype=np.int32),
        np.array([block_size], dtype=np.int32),
        np.array([blocks_per_seq], dtype=np.int32),
        np.array([tile_size],  dtype=np.int32),
        np.array([num_queries_per_kv], dtype=np.int32),
        np.array([block_q],    dtype=np.int32),
        np.array([num_seqs],   dtype=np.int32),
        np.array([-1],         dtype=np.int32),  # window_size_left
    ]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(
        fn, total_q_blocks, num_kv_heads, 1, 128, 1, 1, smem, 0,
        pp.ctypes.data, 0), "launch")
    CHECK(drv.cuCtxSynchronize(), "sync")

    out = np.empty(q_len * num_heads * head_dim, dtype=np.float16)
    CHECK(drv.cuMemcpyDtoH(out.ctypes.data, d_out, out.nbytes), "d2h")
    out = out.reshape(q_len, num_heads, head_dim)

    # --- fp64 reference (on dequanted tensors, causal mask) ---
    k_dq = e2m1_bytes_to_f64(k_cache_packed, k_cache_scale)
    v_dq = e2m1_bytes_to_f64(v_cache_packed, v_cache_scale)
    k_history = np.stack([k_dq[block_table[0, t // block_size], t % block_size]
                          for t in range(ctx_len)])
    v_history = np.stack([v_dq[block_table[0, t // block_size], t % block_size]
                          for t in range(ctx_len)])
    out_ref = np.empty((q_len, num_heads, head_dim), dtype=np.float64)
    for qi in range(q_len):
        q_abs = ctx_len - q_len + qi
        for h in range(num_heads):
            kvh = h // num_queries_per_kv
            scores = (k_history[:, kvh] @ q_f64[qi, h]) * attn_scale
            scores[q_abs + 1:] = -np.inf
            scores -= scores.max()
            probs = np.exp(scores); probs /= probs.sum()
            out_ref[qi, h] = (probs[:, None] * v_history[:, kvh]).sum(axis=0)

    abs_err = np.abs(out.astype(np.float64) - out_ref)
    peak = max(float(np.abs(out_ref).max()), 1e-6)
    tol = 5e-3 * peak
    bad = int((abs_err > tol).sum())
    ok = bad == 0
    tag = "OK  " if ok else "FAIL"
    print(f"  {tag}  H={num_heads:>2} KVH={num_kv_heads:>2} hd={head_dim:>3} "
          f"q_len={q_len:>4} bs={block_size:>2} ts={tile_size:>2}  "
          f"abs_err.max={abs_err.max():.3e} (≤{tol:.3e})  "
          f"mismatches={bad}/{out.size}")

    for d in (d_out, d_q, d_kp, d_vp, d_ks, d_vs, d_bt, d_cu, d_ctx, d_qd):
        CHECK(drv.cuMemFree(d), "free")
    return ok


def e2m1_bytes_to_f64(packed, scales):
    """packed: [..., D/2] u8, scales: [..., D/16] u8 (E4M3). → [..., D] f64."""
    lo_nib = packed & 0x0F
    hi_nib = (packed >> 4) & 0x0F
    lo = e2m1_decode(lo_nib)
    hi = e2m1_decode(hi_nib)
    out_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    vals = np.empty(out_shape, dtype=np.float64)
    vals[..., 0::2] = lo
    vals[..., 1::2] = hi
    scales_f64 = e4m3_to_f64(scales)
    sc_rep = np.repeat(scales_f64, 16, axis=-1)
    return vals * sc_rep


# --- Test drivers ----------------------------------------------------------

print(f"device PTX: {PTX.name} ({ARCH})")
cases = [
    # Small — sanity, tile_size=16 (global-layer-like)
    dict(num_heads=4, num_kv_heads=2, head_dim=128, q_len=16,
         block_size=16, tile_size=16, seed=1),
    # Medium — tile_size=32 (sliding-layer-like)
    dict(num_heads=8, num_kv_heads=2, head_dim=256, q_len=32,
         block_size=16, tile_size=32, seed=7),
    # head_dim=512 path (global-layer-like on Gemma 4)
    dict(num_heads=8, num_kv_heads=2, head_dim=512, q_len=32,
         block_size=16, tile_size=16, seed=42),
    # Longer prompt crossing tile boundaries
    dict(num_heads=4, num_kv_heads=2, head_dim=256, q_len=128,
         block_size=16, tile_size=32, seed=211),
]
all_pass = all(run_unified(**c) for c in cases)
print()
if not all_pass:
    print("FAIL")
    sys.exit(1)
print("unified NVFP4 prefill (f16 MMA) validated against fp64 on hw.")
