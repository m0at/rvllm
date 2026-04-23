#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fa2_unified_prefill_check.py \
#       [sm_xxx] [prompt_len] [head_dim]
#
# Validates `flash_attention_2_prefill_fp8kv_unified_kernel`
# (`kernels/<sm_xxx>/flash_attention_unified_prefill.ptx`) against an
# fp64 NumPy reference. Single-sequence, single-layer self-attention
# over a FP8-E4M3 KV cache with per-slot K/V scales and a per-(tok,
# head) Q scale — the exact shape we'll call from the Rust
# `gemma4_layer_exec::Gemma4Phase::Prefill` path.
#
# Pass criterion:
#   max |out_kernel - out_ref| / mean(|out_ref|) <= 5e-2
# (looser than decode because we cascade FP8 dequant on both Q and K/V
# with per-slot scale; on Gemma 4 shapes the residual hits ≈1-3%.)

import sys, pathlib
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "flash_attention_unified_prefill.ptx"
if not PTX.exists():
    sys.exit(f"missing PTX: {PTX}  (build with: kernels/build.sh {ARCH})")


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

cc_major = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc")
cc_minor = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc")
print(f"device: cc {cc_major}.{cc_minor}, PTX: {PTX.name} ({ARCH})")

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(
    mod, b"flash_attention_2_prefill_fp8kv_unified_kernel"),
    "cuModuleGetFunction")

# -------- FP8 E4M3 quantisation reference ------------------------------------
# Match the kernel's `fp8kv_decode_byte` exactly: native
# `__nv_cvt_fp8_to_halfraw(E4M3)` on sm_121 rounds to the nearest
# representable half-float; the branchless pre-Blackwell decode is a
# byte-identical emulation.
#
# We emulate by computing the true IEEE E4M3 value (pos subnormals +
# normals; inf/nan not used in our data) — sign + 4-bit exp + 3-bit
# mantissa, bias 7.
FP8_MAX = 448.0
FP8_MIN = -448.0


def quantize_to_fp8(x, per_vec_scale=None):
    """Quantise a tensor to FP8 E4M3 bytes. Returns (bytes, scale).

    `per_vec_scale` (f32) can pin the scale so producer + consumer
    agree. When None, we pick scale = max(|x|)/FP8_MAX per vector.
    """
    x = np.asarray(x, dtype=np.float32)
    if per_vec_scale is None:
        amax = np.max(np.abs(x))
        if amax < 1e-12:
            scale = 1.0
        else:
            scale = amax / FP8_MAX
    else:
        scale = float(per_vec_scale)
    x_scaled = np.clip(x / scale, FP8_MIN, FP8_MAX)
    # Quantise to E4M3 via NumPy float64 → E4M3 bytes.
    flat = x_scaled.astype(np.float64).ravel()
    bytes_out = np.empty(flat.size, dtype=np.uint8)
    for i, v in enumerate(flat):
        bytes_out[i] = f32_to_e4m3_byte(float(v))
    return bytes_out.reshape(x.shape), np.float32(scale)


def f32_to_e4m3_byte(v):
    if v == 0.0:
        return 0
    sign = 1 if v < 0.0 else 0
    a = abs(v)
    # normalise to [1, 2) × 2^e, bias 7
    # clamp
    if a > FP8_MAX:
        a = FP8_MAX
    # find exponent
    e = 0
    m = a
    while m >= 2.0:
        m /= 2.0
        e += 1
    while m < 1.0 and e > -6:
        m *= 2.0
        e -= 1
    exp_bits = e + 7
    if exp_bits < 0:
        # subnormal — quantise toward zero
        return (sign << 7)
    if exp_bits > 15:
        exp_bits = 15  # inf pattern avoided by FP8_MAX clamp
    # mantissa bits: 3 bits of m-1 in [0, 1)
    m_frac = m - 1.0
    mant_bits = int(round(m_frac * 8))
    if mant_bits == 8:
        mant_bits = 0
        exp_bits += 1
    if exp_bits > 15:
        exp_bits, mant_bits = 15, 7  # max representable
    return (sign << 7) | ((exp_bits & 0xF) << 3) | (mant_bits & 0x7)


def e4m3_byte_to_f32(b):
    if b == 0 or b == 0x80:
        return 0.0
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    # IEEE-like: (1 + m/8) × 2^(exp-7); bias 7, subnormals map to 0
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def dequantize(bytes_, scale):
    """Reverse of quantize_to_fp8 — what the kernel sees."""
    f = np.vectorize(e4m3_byte_to_f32, otypes=[np.float32])(bytes_)
    return f * np.float32(scale)


# -------- Test shape ---------------------------------------------------------
# Gemma 4 sliding layer: 32 Q heads, 16 KV heads, head_dim 256.
prompt_len = int(sys.argv[2]) if len(sys.argv) > 2 else 128
head_dim   = int(sys.argv[3]) if len(sys.argv) > 3 else 256
num_heads     = 32
num_kv_heads  = 16
block_size    = 32
tile_size     = 32 if head_dim <= 256 else 16
sliding_window = 0  # 0 = disabled; test full causal first
scale         = 1.0 / np.sqrt(head_dim)
num_queries_per_kv = num_heads // num_kv_heads
BLOCK_M       = 16
block_q       = BLOCK_M // num_queries_per_kv

max_blocks_per_seq = (prompt_len + block_size - 1) // block_size
num_blocks = max_blocks_per_seq
print(f"shape: prompt_len={prompt_len}, head_dim={head_dim}, "
      f"heads={num_heads}/{num_kv_heads}, tile={tile_size}, "
      f"blocks={num_blocks}×{block_size}, sliding={sliding_window}")

rng = np.random.default_rng(7)
# Raw f32 Q, K, V — these get quantised to FP8 with their own scales.
Q_f32 = rng.normal(0, 1, (prompt_len, num_heads, head_dim)).astype(np.float32)
K_f32 = rng.normal(0, 1, (num_blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
V_f32 = rng.normal(0, 1, (num_blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)

# Pick a common Q scale per-(token, head): amax / FP8_MAX.
q_scale = np.zeros((prompt_len, num_heads), dtype=np.float32)
Q_bytes = np.zeros((prompt_len, num_heads, head_dim), dtype=np.uint8)
for t in range(prompt_len):
    for h in range(num_heads):
        amax = max(float(np.max(np.abs(Q_f32[t, h]))), 1e-12)
        s = amax / FP8_MAX
        q_scale[t, h] = s
        Q_bytes[t, h] = [f32_to_e4m3_byte(float(v / s)) for v in Q_f32[t, h]]

# Per-(slot, head) K/V scales: [num_blocks*block_size, num_kv_heads]
total_slots = num_blocks * block_size
k_scale = np.zeros((total_slots, num_kv_heads), dtype=np.float32)
v_scale = np.zeros((total_slots, num_kv_heads), dtype=np.float32)
K_bytes = np.zeros_like(K_f32, dtype=np.uint8)
V_bytes = np.zeros_like(V_f32, dtype=np.uint8)
for b in range(num_blocks):
    for o in range(block_size):
        slot = b * block_size + o
        for kh in range(num_kv_heads):
            ka = max(float(np.max(np.abs(K_f32[b, o, kh]))), 1e-12)
            va = max(float(np.max(np.abs(V_f32[b, o, kh]))), 1e-12)
            sk = ka / FP8_MAX
            sv = va / FP8_MAX
            k_scale[slot, kh] = sk
            v_scale[slot, kh] = sv
            K_bytes[b, o, kh] = [f32_to_e4m3_byte(float(x / sk)) for x in K_f32[b, o, kh]]
            V_bytes[b, o, kh] = [f32_to_e4m3_byte(float(x / sv)) for x in V_f32[b, o, kh]]

# Fp8 → f32 round-trip so the reference uses what the kernel actually sees.
Q_rt = np.zeros_like(Q_f32)
K_rt = np.zeros_like(K_f32)
V_rt = np.zeros_like(V_f32)
for t in range(prompt_len):
    for h in range(num_heads):
        Q_rt[t, h] = dequantize(Q_bytes[t, h], q_scale[t, h])
for b in range(num_blocks):
    for o in range(block_size):
        slot = b * block_size + o
        for kh in range(num_kv_heads):
            K_rt[b, o, kh] = dequantize(K_bytes[b, o, kh], k_scale[slot, kh])
            V_rt[b, o, kh] = dequantize(V_bytes[b, o, kh], v_scale[slot, kh])

block_tables = np.arange(num_blocks, dtype=np.int32).reshape(1, max_blocks_per_seq)
context_lens = np.array([prompt_len], dtype=np.int32)
cu_seqlens_q = np.array([0, prompt_len], dtype=np.int32)

# -------- Reference (fp64, causal + optional sliding) ------------------------
out_ref = np.zeros((prompt_len, num_heads, head_dim), dtype=np.float64)
for t in range(prompt_len):
    for h in range(num_heads):
        kh = h // (num_heads // num_kv_heads)
        k_rows = np.zeros((prompt_len, head_dim), dtype=np.float64)
        v_rows = np.zeros((prompt_len, head_dim), dtype=np.float64)
        for pos in range(prompt_len):
            b = block_tables[0, pos // block_size]
            o = pos % block_size
            k_rows[pos] = K_rt[b, o, kh]
            v_rows[pos] = V_rt[b, o, kh]
        q = Q_rt[t, h].astype(np.float64)
        scores = (k_rows @ q) * scale
        # Causal: key pos <= query pos
        mask = np.arange(prompt_len) <= t
        if sliding_window > 0:
            mask &= (t - np.arange(prompt_len)) < sliding_window
        scores = np.where(mask, scores, -np.inf)
        m = scores.max()
        p = np.exp(scores - m)
        p_sum = p.sum()
        if p_sum == 0:
            out_ref[t, h] = 0
            continue
        p /= p_sum
        out_ref[t, h] = p @ v_rows
out_ref_f16 = out_ref.astype(np.float16)

# -------- GPU launch ---------------------------------------------------------


def alloc(n):
    return CHECK(drv.cuMemAlloc(n), "cuMemAlloc")


def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")


d_out  = alloc(prompt_len * num_heads * head_dim * 2)  # f16
d_q    = alloc(Q_bytes.nbytes);  h2d(d_q, Q_bytes)
d_k    = alloc(K_bytes.nbytes);  h2d(d_k, K_bytes)
d_v    = alloc(V_bytes.nbytes);  h2d(d_v, V_bytes)
d_ks   = alloc(k_scale.nbytes);  h2d(d_ks, k_scale)
d_vs   = alloc(v_scale.nbytes);  h2d(d_vs, v_scale)
d_qs   = alloc(q_scale.nbytes);  h2d(d_qs, q_scale)
d_bt   = alloc(block_tables.nbytes);  h2d(d_bt, block_tables)
d_cu   = alloc(cu_seqlens_q.nbytes);  h2d(d_cu, cu_seqlens_q)
d_cl   = alloc(context_lens.nbytes);  h2d(d_cl, context_lens)

# Dummy scalar fallbacks (unused when *_cache != nullptr)
d_kfb = alloc(4); h2d(d_kfb, np.array([1.0], dtype=np.float32))
d_vfb = alloc(4); h2d(d_vfb, np.array([1.0], dtype=np.float32))
d_qd  = alloc(4); h2d(d_qd, np.array([1.0], dtype=np.float32))

FA2_THREADS = 128
MMA_K = 32
smem_bytes = (
    BLOCK_M * head_dim * 1            # s_q_fp8 (Phase F3: FP8, not f32)
    + BLOCK_M * 4                     # s_q_scale
    + head_dim * tile_size * 1        # s_k_fp8
    + tile_size * head_dim * 1        # s_v_fp8
    + MMA_K * head_dim * 1            # s_v_fp8_T (F4+F7: MMA_K-padded)
    + tile_size * 4                   # s_k_scale
    + tile_size * 4                   # s_v_scale
    + BLOCK_M * MMA_K * 4             # s_s (F7: MMA_K, not tile_size)
    + BLOCK_M * 4 * 3                 # s_m + s_l + s_alpha
    + BLOCK_M * MMA_K * 1             # s_p_fp8 (F4+F7)
    + BLOCK_M * 4                     # s_p_scale (F4)
    + BLOCK_M * head_dim * 4          # s_acc
    + (FA2_THREADS // 32) * 4         # tail reduce
    + 128                             # safety margin
)
print(f"smem request: {smem_bytes} bytes")
if smem_bytes >= 48 * 1024:
    CHECK(drv.cuFuncSetAttribute(
        fn,
        drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_bytes,
    ), "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE)")

# Grid: total_num_q_blocks = prompt_len / block_q + num_seqs (upper bound).
total_num_q_blocks = prompt_len // block_q + 1
window_size_left = -1 if sliding_window <= 0 else sliding_window
num_seqs = 1
# Phase F3: `use_mma=1` routes Q·Kᵀ through the sm_121a FP8 tensor
# cores. Pass via CLI (e.g. `sm_121 256 256 1`) so the same harness
# validates both paths against the same NumPy reference.
use_mma = int(sys.argv[4]) if len(sys.argv) > 4 else 0

params = [
    np.array([int(d_out)], dtype=np.uint64),
    np.array([int(d_q)],   dtype=np.uint64),
    np.array([int(d_k)],   dtype=np.uint64),
    np.array([int(d_v)],   dtype=np.uint64),
    np.array([int(d_ks)],  dtype=np.uint64),
    np.array([int(d_vs)],  dtype=np.uint64),
    np.array([int(d_qs)],  dtype=np.uint64),
    np.array([int(d_kfb)], dtype=np.uint64),
    np.array([int(d_vfb)], dtype=np.uint64),
    np.array([int(d_bt)],  dtype=np.uint64),
    np.array([int(d_cu)],  dtype=np.uint64),
    np.array([int(d_cl)],  dtype=np.uint64),
    np.array([int(d_qd)],  dtype=np.uint64),
    np.array([scale],                dtype=np.float32),
    np.array([num_heads],            dtype=np.int32),
    np.array([num_kv_heads],         dtype=np.int32),
    np.array([head_dim],             dtype=np.int32),
    np.array([block_size],           dtype=np.int32),
    np.array([max_blocks_per_seq],   dtype=np.int32),
    np.array([tile_size],            dtype=np.int32),
    np.array([num_queries_per_kv],   dtype=np.int32),
    np.array([block_q],              dtype=np.int32),
    np.array([num_seqs],             dtype=np.int32),
    np.array([window_size_left],     dtype=np.int32),
    np.array([use_mma],              dtype=np.int32),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

print(f"grid: ({total_num_q_blocks}, {num_kv_heads}, 1), "
      f"block: ({FA2_THREADS}, 1, 1), smem={smem_bytes}")
CHECK(drv.cuLaunchKernel(
    fn,
    total_num_q_blocks, num_kv_heads, 1,
    FA2_THREADS, 1, 1,
    smem_bytes, 0,
    param_ptrs.ctypes.data, 0,
), "cuLaunchKernel")
CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

out = np.empty((prompt_len, num_heads, head_dim), dtype=np.float16)
CHECK(drv.cuMemcpyDtoH(out.ctypes.data, d_out, out.nbytes), "DtoH")

# -------- Compare ------------------------------------------------------------
out_f32 = out.astype(np.float32)
ref_f32 = out_ref_f16.astype(np.float32)
abs_err = np.abs(out_f32 - ref_f32)
ref_mean_abs = float(np.abs(ref_f32).mean())
scale_rel = abs_err / max(ref_mean_abs, 1e-30)

print(f"ref    range: [{ref_f32.min():+.4e}, {ref_f32.max():+.4e}], "
      f"|ref| mean {ref_mean_abs:.4e}")
print(f"kernel range: [{out_f32.min():+.4e}, {out_f32.max():+.4e}]")
print(f"abs_err:   max {abs_err.max():.4e}  mean {abs_err.mean():.4e}")
print(f"scale_rel: max {scale_rel.max():.4e}  mean {scale_rel.mean():.4e}")

# Worst token / head
worst = np.argsort(abs_err.reshape(prompt_len * num_heads, -1).max(axis=1))[-3:][::-1]
print("worst (tok, head): " +
      ", ".join(f"({w // num_heads},{w % num_heads}): "
                f"{abs_err.reshape(prompt_len*num_heads,-1)[w].max():.3e}"
                for w in worst))

THRESHOLD = 5e-2
if scale_rel.max() > THRESHOLD:
    print(f"\nFAIL: max scale_rel {scale_rel.max():.4e} > {THRESHOLD:.0e}")
    sys.exit(1)
print(f"\nOK: scale_rel.max {scale_rel.max():.4e} <= {THRESHOLD:.0e}")
