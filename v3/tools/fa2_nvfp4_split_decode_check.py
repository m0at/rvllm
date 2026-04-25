#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fa2_nvfp4_split_decode_check.py [sm_xxx]
#
# Split-KV (paged_attention_v2-style) protocol harness.
#
# Decodes a single query token against a paged NVFP4 KV cache using the
# vLLM two-phase algorithm:
#
#   Phase 1 — split: partition the KV sequence into P tiles of
#     PARTITION_SIZE each. For each partition:
#       - Run standard softmax(Q · K^T · scale) * V restricted to that
#         partition's KV slice.
#       - Store `max_logit[p]` (pre-softmax row-max of the partition),
#         `exp_sum[p]`  (sum_t exp(score[p, t] - max_logit[p])),
#         `tmp_out[p]`  (partition's attention output AFTER dividing by
#                        its own exp_sum — this is what vLLM does so
#                        that the reduce kernel's reweighting math
#                        works).
#
#   Phase 2 — reduce: combine across partitions using the standard
#     online-softmax recombine:
#       global_max = max_p(max_logit[p])
#       w[p]       = exp_sum[p] * exp(max_logit[p] - global_max)
#       inv        = 1 / (sum_p(w[p]) + 1e-6)
#       out[d]     = sum_p(tmp_out[p, d] * w[p] * inv)
#
# This harness:
#   1. Builds a real NVFP4 paged cache via the already-validated
#      `fused_rope_partial_nvfp4kv_kernel` (same setup as
#      `fa2_nvfp4_decode_check.py`).
#   2. Dequants the cache back through numpy (fp64).
#   3. Computes the reference full-softmax decode against the exact
#      K/V the kernel would see.
#   4. Computes the numpy split-decode + numpy reduce against the SAME
#      K/V using the two-phase protocol above.
#   5. Asserts that (3) and (4) match to fp64 precision (they MUST —
#      they're the same math, just recomputed in two passes).
#
# No CUDA kernels beyond the existing RoPE setup are touched. This
# validates our understanding of the vLLM split-reduce semantic
# contract. Once the CUDA split + reduce kernels land, a sibling
# harness (or an extension of this one) will swap step (4)'s numpy
# path for the PTX path and the tolerance widens to fp16 (5e-3 *
# peak, same gate as the existing decode harness).
#
# Gate for step (5): abs error ≤ 1e-9 * peak(|ref|). This is a pure
# numerical-protocol check — anything wider means our phase-2
# reconstruction is wrong.

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
ROPE_PTX  = REPO / "kernels" / ARCH / "fused_rope_partial_nvfp4kv.ptx"
SPLIT_PTX = REPO / "kernels" / ARCH / "flash_attention_split_decode_nvfp4kv.ptx"

if not ROPE_PTX.exists():
    sys.exit(f"missing RoPE PTX: {ROPE_PTX}\n  build: kernels/build.sh {ARCH}")


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

rope_mod = CHECK(drv.cuModuleLoadData(ROPE_PTX.read_bytes() + b"\0"), "load rope")
fn_rope = CHECK(drv.cuModuleGetFunction(rope_mod, b"fused_rope_partial_nvfp4kv_kernel"),
                "get rope fn")

fn_split_bc32 = fn_split_bc16 = fn_reduce = None
if SPLIT_PTX.exists():
    split_mod = CHECK(drv.cuModuleLoadData(SPLIT_PTX.read_bytes() + b"\0"),
                     "load split")
    fn_split_bc32 = CHECK(drv.cuModuleGetFunction(
        split_mod, b"flash_attention_2_decode_nvfp4kv_split_kernel"),
        "get split bc32")
    fn_split_bc16 = CHECK(drv.cuModuleGetFunction(
        split_mod, b"flash_attention_2_decode_nvfp4kv_split_bc16_kernel"),
        "get split bc16")
    fn_reduce = CHECK(drv.cuModuleGetFunction(
        split_mod, b"paged_attention_reduce_f16_kernel"),
        "get reduce")

SMEM_OPT_IN = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


# --- NVFP4 / FP8 decode helpers (copied verbatim from fa2_nvfp4_decode_check.py) ---

def fp4_decode_table():
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def e4m3_decode_bytes(buf: np.ndarray) -> np.ndarray:
    b = buf.astype(np.uint32)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float64)
    normal = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
    sub    = (man / 8.0) * (2.0 ** -6)
    val = np.where(exp == 0, sub, normal)
    val = np.where((exp == 15) & (man == 7), 0.0, val)
    return sign * val


def nvfp4_dequant(packed: np.ndarray, scales: np.ndarray, inner: int) -> np.ndarray:
    table = fp4_decode_table()
    flat = packed.astype(np.uint32)
    lo_mag = table[flat & 0x7]
    lo_sgn = np.where((flat & 0x8) != 0, -1.0, 1.0)
    hi_mag = table[(flat >> 4) & 0x7]
    hi_sgn = np.where(((flat >> 4) & 0x8) != 0, -1.0, 1.0)
    interleaved = np.stack([lo_mag * lo_sgn, hi_mag * hi_sgn], axis=-1)
    interleaved = interleaved.reshape(*packed.shape[:-1], inner)
    sc = np.repeat(scales, 16, axis=-1)
    return interleaved * sc


def fp8_e4m3_decode(buf: np.ndarray) -> np.ndarray:
    return e4m3_decode_bytes(buf)


# --- Phase-1 (split) + Phase-2 (reduce) numpy reference ---


def split_phase1_np(q_ref, k_hist, v_hist, scale, partition_size,
                    window_start):
    """Returns (tmp_out, max_logits, exp_sums).

    Shapes:
      q_ref    [S, H, D]
      k_hist   [T, S, KH, D]
      v_hist   [T, S, KH, D]
      tmp_out     [S, H, P, D]  (per-partition output, DIVIDED by its exp_sum)
      max_logits  [S, H, P]
      exp_sums    [S, H, P]
    """
    S, H, D = q_ref.shape
    T = k_hist.shape[0]
    KH = k_hist.shape[2]
    gqa = H // KH
    P = (T + partition_size - 1) // partition_size

    tmp_out = np.zeros((S, H, P, D), dtype=np.float64)
    max_logits = np.full((S, H, P), -np.inf, dtype=np.float64)
    exp_sums = np.zeros((S, H, P), dtype=np.float64)

    for s in range(S):
        for h in range(H):
            kvh = h // gqa
            q_row = q_ref[s, h]
            for p in range(P):
                t0 = p * partition_size
                t1 = min(t0 + partition_size, T)
                if t0 >= T:
                    # Empty partition — kernel would early-return. Leave
                    # sentinels so the reduce weights this with zero.
                    continue
                k_part = k_hist[t0:t1, s, kvh]  # [tp, D]
                v_part = v_hist[t0:t1, s, kvh]
                scores = (k_part @ q_row) * scale
                # Sliding-window mask — positions before window_start
                # contribute nothing.
                if window_start > 0:
                    abs_pos = np.arange(t0, t1)
                    scores = np.where(abs_pos < window_start, -np.inf, scores)
                # Per-partition softmax.
                mx = scores.max()
                if not np.isfinite(mx):
                    # Whole partition masked out — leave sentinels (-inf/0).
                    max_logits[s, h, p] = -np.inf
                    exp_sums[s, h, p] = 0.0
                    continue
                expv = np.exp(scores - mx)
                sumv = expv.sum()
                probs = expv / sumv
                # Per-partition output — already normalized by partition's
                # own sum. Phase-2 will reweight with the global normalizer.
                tmp_out[s, h, p] = (probs[:, None] * v_part).sum(axis=0)
                max_logits[s, h, p] = mx
                exp_sums[s, h, p] = sumv
    return tmp_out, max_logits, exp_sums


def reduce_phase2_np(tmp_out, max_logits, exp_sums, context_lens,
                     partition_size):
    """Combine partition outputs back into final per-head attention.

    out[s,h,d] = sum_p(tmp_out[s,h,p,d] * w[p] * inv_sum_w)
      where w[p] = exp_sums[p] * exp(max_logits[p] - global_max).
    Partitions beyond ceil(ctx/P) are ignored (handled by masking w=0
    since max_logits stays at -inf there).
    """
    S, H, P, D = tmp_out.shape
    out = np.zeros((S, H, D), dtype=np.float64)
    for s in range(S):
        seq_len = context_lens[s]
        num_p = (seq_len + partition_size - 1) // partition_size
        for h in range(H):
            mls = max_logits[s, h, :num_p]
            esm = exp_sums[s, h, :num_p]
            valid = np.isfinite(mls)
            if not valid.any():
                continue
            gmax = mls[valid].max()
            w = np.where(valid, esm * np.exp(mls - gmax), 0.0)
            # NOTE: CUDA reduce kernel uses `1 / (sum + 1e-6)` for
            # numerical stability. We drop the ε here since the direct
            # reference does the same — any mismatch should surface as
            # pure fp64 recombine roundoff. The CUDA-facing harness
            # re-adds the ε when comparing the split + reduce PTX
            # kernels against this numpy path.
            inv = 1.0 / w.sum()
            for p in range(num_p):
                out[s, h] += tmp_out[s, h, p] * w[p] * inv
    return out


# ----- Run one shape ------------------------------------------------


def run_one(num_seqs, num_heads, num_kv_heads, head_dim, rotary_dim,
            context_len, block_size, partition_size, seed,
            window_size_left=-1):
    rng = np.random.default_rng(seed)
    assert context_len % block_size == 0 or context_len < block_size, \
        "keep context_len a multiple of block_size for the probe"
    assert partition_size % block_size == 0, \
        "partition_size must be a multiple of block_size"

    # History K/V (fp16) — will land in NVFP4 paged cache via RoPE kernel.
    history_k = rng.standard_normal((context_len, num_seqs, num_kv_heads, head_dim)).astype(np.float16)
    history_v = rng.standard_normal((context_len, num_seqs, num_kv_heads, head_dim)).astype(np.float16)

    max_pos = context_len + 16
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10000 ** (np.arange(half_rot) / half_rot))
    cos = np.cos(np.arange(max_pos)[:, None] * freqs[None, :]).astype(np.float16)
    sin = np.sin(np.arange(max_pos)[:, None] * freqs[None, :]).astype(np.float16)

    num_blocks = (context_len + block_size - 1) // block_size * num_seqs
    packed_bytes_total = num_blocks * block_size * num_kv_heads * (head_dim // 2)
    scale_bytes_total  = num_blocks * block_size * num_kv_heads * (head_dim // 16)

    d_key_pack = CHECK(drv.cuMemAlloc(packed_bytes_total), "alloc kp")
    d_val_pack = CHECK(drv.cuMemAlloc(packed_bytes_total), "alloc vp")
    d_key_sc   = CHECK(drv.cuMemAlloc(scale_bytes_total),  "alloc ks")
    d_val_sc   = CHECK(drv.cuMemAlloc(scale_bytes_total),  "alloc vs")
    for d, n in [(d_key_pack, packed_bytes_total), (d_val_pack, packed_bytes_total),
                 (d_key_sc, scale_bytes_total), (d_val_sc, scale_bytes_total)]:
        CHECK(drv.cuMemsetD8(d, 0, n), "zero cache")

    q_scale_val = np.float32(0.5)
    d_qs = CHECK(drv.cuMemAlloc(4), "alloc qs")
    CHECK(drv.cuMemcpyHtoD(d_qs, q_scale_val.tobytes(), 4), "qs H2D")

    d_cos = CHECK(drv.cuMemAlloc(cos.nbytes), "alloc cos")
    d_sin = CHECK(drv.cuMemAlloc(sin.nbytes), "alloc sin")
    CHECK(drv.cuMemcpyHtoD(d_cos, cos.ctypes.data, cos.nbytes), "cos H2D")
    CHECK(drv.cuMemcpyHtoD(d_sin, sin.ctypes.data, sin.nbytes), "sin H2D")

    blocks_per_seq = (context_len + block_size - 1) // block_size
    block_table = np.arange(num_seqs * blocks_per_seq, dtype=np.int32).reshape(
        num_seqs, blocks_per_seq
    )
    d_bt = CHECK(drv.cuMemAlloc(block_table.nbytes), "alloc bt")
    CHECK(drv.cuMemcpyHtoD(d_bt, block_table.ctypes.data, block_table.nbytes), "bt H2D")

    dummy_q = np.zeros((num_seqs, num_heads, head_dim), dtype=np.float16)
    d_dummy_q = CHECK(drv.cuMemAlloc(dummy_q.nbytes), "alloc dq")
    CHECK(drv.cuMemcpyHtoD(d_dummy_q, dummy_q.ctypes.data, dummy_q.nbytes), "dq H2D")
    d_dummy_q_out = CHECK(drv.cuMemAlloc(dummy_q.nbytes), "alloc dqo")

    for step in range(context_len):
        k_step = np.ascontiguousarray(history_k[step]).astype(np.float16)
        v_step = np.ascontiguousarray(history_v[step]).astype(np.float16)
        d_k_step = CHECK(drv.cuMemAlloc(k_step.nbytes), "alloc k_step")
        d_v_step = CHECK(drv.cuMemAlloc(v_step.nbytes), "alloc v_step")
        CHECK(drv.cuMemcpyHtoD(d_k_step, k_step.ctypes.data, k_step.nbytes), "k_step H2D")
        CHECK(drv.cuMemcpyHtoD(d_v_step, v_step.ctypes.data, v_step.nbytes), "v_step H2D")

        pos_step = np.full(num_seqs, step, dtype=np.int32)
        slot_step = np.array([
            block_table[s, step // block_size] * block_size + (step % block_size)
            for s in range(num_seqs)
        ], dtype=np.int32)
        d_pos  = CHECK(drv.cuMemAlloc(pos_step.nbytes),  "alloc pos")
        d_slot = CHECK(drv.cuMemAlloc(slot_step.nbytes), "alloc slot")
        CHECK(drv.cuMemcpyHtoD(d_pos,  pos_step.ctypes.data,  pos_step.nbytes),  "pos H2D")
        CHECK(drv.cuMemcpyHtoD(d_slot, slot_step.ctypes.data, slot_step.nbytes), "slot H2D")

        params = [
            np.array([int(d_dummy_q)],     dtype=np.uint64),
            np.array([int(d_k_step)],      dtype=np.uint64),
            np.array([int(d_v_step)],      dtype=np.uint64),
            np.array([int(d_dummy_q_out)], dtype=np.uint64),
            np.array([int(d_key_pack)],    dtype=np.uint64),
            np.array([int(d_val_pack)],    dtype=np.uint64),
            np.array([int(d_key_sc)],      dtype=np.uint64),
            np.array([int(d_val_sc)],      dtype=np.uint64),
            np.array([int(d_cos)],         dtype=np.uint64),
            np.array([int(d_sin)],         dtype=np.uint64),
            np.array([int(d_pos)],         dtype=np.uint64),
            np.array([int(d_slot)],        dtype=np.uint64),
            np.array([int(d_qs)],          dtype=np.uint64),
            np.array([num_seqs],           dtype=np.int32),
            np.array([num_heads],          dtype=np.int32),
            np.array([num_kv_heads],       dtype=np.int32),
            np.array([head_dim],           dtype=np.int32),
            np.array([rotary_dim],         dtype=np.int32),
        ]
        pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
        CHECK(drv.cuLaunchKernel(fn_rope,
                                 num_seqs, max(num_heads, num_kv_heads), 1,
                                 head_dim, 1, 1,
                                 0, 0, pp.ctypes.data, 0), "rope launch")
        CHECK(drv.cuCtxSynchronize(), "rope sync")
        for d in (d_k_step, d_v_step, d_pos, d_slot):
            CHECK(drv.cuMemFree(d), "free step")

    # Decode Q — RoPE + FP8-quantise to match kernel's contract.
    q_decode = rng.standard_normal((num_seqs, num_heads, head_dim)).astype(np.float16)
    zero_k = np.zeros((num_seqs, num_kv_heads, head_dim), dtype=np.float16)
    zero_v = np.zeros_like(zero_k)
    d_q_in  = CHECK(drv.cuMemAlloc(q_decode.nbytes), "alloc qin")
    d_k_z   = CHECK(drv.cuMemAlloc(zero_k.nbytes),   "alloc kz")
    d_v_z   = CHECK(drv.cuMemAlloc(zero_v.nbytes),   "alloc vz")
    CHECK(drv.cuMemcpyHtoD(d_q_in, q_decode.ctypes.data, q_decode.nbytes), "qin H2D")
    CHECK(drv.cuMemcpyHtoD(d_k_z,  zero_k.ctypes.data,   zero_k.nbytes),   "kz H2D")
    CHECK(drv.cuMemcpyHtoD(d_v_z,  zero_v.ctypes.data,   zero_v.nbytes),   "vz H2D")
    d_q_fp8 = CHECK(drv.cuMemAlloc(num_seqs * num_heads * head_dim), "alloc qfp8")

    pos_d  = np.full(num_seqs, context_len, dtype=np.int32)
    slot_d = np.full(num_seqs, -1, dtype=np.int32)
    d_pos_d  = CHECK(drv.cuMemAlloc(pos_d.nbytes),  "alloc posd")
    d_slot_d = CHECK(drv.cuMemAlloc(slot_d.nbytes), "alloc slotd")
    CHECK(drv.cuMemcpyHtoD(d_pos_d,  pos_d.ctypes.data,  pos_d.nbytes),  "posd H2D")
    CHECK(drv.cuMemcpyHtoD(d_slot_d, slot_d.ctypes.data, slot_d.nbytes), "slotd H2D")

    params = [
        np.array([int(d_q_in)],    dtype=np.uint64),
        np.array([int(d_k_z)],     dtype=np.uint64),
        np.array([int(d_v_z)],     dtype=np.uint64),
        np.array([int(d_q_fp8)],   dtype=np.uint64),
        np.array([int(d_key_pack)],dtype=np.uint64),
        np.array([int(d_val_pack)],dtype=np.uint64),
        np.array([int(d_key_sc)],  dtype=np.uint64),
        np.array([int(d_val_sc)],  dtype=np.uint64),
        np.array([int(d_cos)],     dtype=np.uint64),
        np.array([int(d_sin)],     dtype=np.uint64),
        np.array([int(d_pos_d)],   dtype=np.uint64),
        np.array([int(d_slot_d)],  dtype=np.uint64),
        np.array([int(d_qs)],      dtype=np.uint64),
        np.array([num_seqs],       dtype=np.int32),
        np.array([num_heads],      dtype=np.int32),
        np.array([num_kv_heads],   dtype=np.int32),
        np.array([head_dim],       dtype=np.int32),
        np.array([rotary_dim],     dtype=np.int32),
    ]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(fn_rope,
                             num_seqs, max(num_heads, num_kv_heads), 1,
                             head_dim, 1, 1,
                             0, 0, pp.ctypes.data, 0), "rope decode Q")
    CHECK(drv.cuCtxSynchronize(), "sync qfp8")

    # Read the NVFP4 cache back.
    kp = np.empty(packed_bytes_total, dtype=np.uint8)
    vp = np.empty(packed_bytes_total, dtype=np.uint8)
    ks = np.empty(scale_bytes_total,  dtype=np.uint8)
    vs = np.empty(scale_bytes_total,  dtype=np.uint8)
    CHECK(drv.cuMemcpyDtoH(kp.ctypes.data, d_key_pack, packed_bytes_total), "kp D2H")
    CHECK(drv.cuMemcpyDtoH(vp.ctypes.data, d_val_pack, packed_bytes_total), "vp D2H")
    CHECK(drv.cuMemcpyDtoH(ks.ctypes.data, d_key_sc,   scale_bytes_total),  "ks D2H")
    CHECK(drv.cuMemcpyDtoH(vs.ctypes.data, d_val_sc,   scale_bytes_total),  "vs D2H")

    kp_r = kp.reshape(num_blocks, block_size, num_kv_heads, head_dim // 2)
    vp_r = vp.reshape(num_blocks, block_size, num_kv_heads, head_dim // 2)
    ks_r = e4m3_decode_bytes(ks).reshape(num_blocks, block_size, num_kv_heads, head_dim // 16)
    vs_r = e4m3_decode_bytes(vs).reshape(num_blocks, block_size, num_kv_heads, head_dim // 16)
    k_dq_full = nvfp4_dequant(kp_r, ks_r, head_dim)
    v_dq_full = nvfp4_dequant(vp_r, vs_r, head_dim)

    k_hist = np.empty((context_len, num_seqs, num_kv_heads, head_dim), dtype=np.float64)
    v_hist = np.empty_like(k_hist)
    for s in range(num_seqs):
        for t in range(context_len):
            blk = block_table[s, t // block_size]
            off = t % block_size
            k_hist[t, s] = k_dq_full[blk, off]
            v_hist[t, s] = v_dq_full[blk, off]

    q_fp8 = np.empty(num_seqs * num_heads * head_dim, dtype=np.uint8)
    CHECK(drv.cuMemcpyDtoH(q_fp8.ctypes.data, d_q_fp8, q_fp8.nbytes), "qfp8 D2H")
    q_ref = fp8_e4m3_decode(q_fp8).reshape(num_seqs, num_heads, head_dim) * float(q_scale_val)

    scale = 1.0 / np.sqrt(head_dim)
    decode_q_abs = context_len - 1
    window_start = (0 if window_size_left < 0
                    else max(0, decode_q_abs - window_size_left))

    # --- Reference: full-softmax decode (single pass). ---
    gqa = num_heads // num_kv_heads
    out_ref = np.empty_like(q_ref)
    for s in range(num_seqs):
        for h in range(num_heads):
            kvh = h // gqa
            q_row = q_ref[s, h]
            k_hist_h = k_hist[:, s, kvh]
            v_hist_h = v_hist[:, s, kvh]
            scores = (k_hist_h @ q_row) * scale
            if window_start > 0:
                scores[:window_start] = -np.inf
            scores -= scores.max()
            probs = np.exp(scores); probs /= probs.sum()
            out_ref[s, h] = (probs[:, None] * v_hist_h).sum(axis=0)

    # --- Split-reduce path (numpy protocol). ---
    tmp_out, max_logits, exp_sums = split_phase1_np(
        q_ref, k_hist, v_hist, scale, partition_size, window_start)
    ctx_lens = np.full(num_seqs, context_len, dtype=np.int64)
    out_split = reduce_phase2_np(tmp_out, max_logits, exp_sums, ctx_lens,
                                 partition_size)

    abs_err = np.abs(out_split - out_ref)
    peak = np.abs(out_ref).max()
    # Pure fp64 recombine roundoff — two exponentials per term plus
    # ~P additions. 1e-12 * peak is the loosest we should ever need.
    tol = 1e-12 * max(peak, 1e-6)
    bad = int((abs_err > tol).sum())
    ok_np = bad == 0
    num_parts = (context_len + partition_size - 1) // partition_size
    wtag = f" ws={window_size_left}" if window_size_left >= 0 else ""

    # --- CUDA split + reduce vs numpy split+reduce (fp16 gate). ---
    ok_cuda = True
    cuda_err = None
    cuda_peak = None
    cuda_tol = None
    cuda_bad = 0
    if fn_split_bc32 is not None and fn_reduce is not None:
        max_num_partitions = num_parts  # launcher-level; matches harness
        # Scratch: tmp_out[S,H,P,D] f16, max_logits/exp_sums [S,H,P] f32.
        tmp_out_bytes = num_seqs * num_heads * max_num_partitions * head_dim * 2
        meta_bytes    = num_seqs * num_heads * max_num_partitions * 4
        d_tmp_out = CHECK(drv.cuMemAlloc(tmp_out_bytes), "alloc tmp_out")
        d_max_log = CHECK(drv.cuMemAlloc(meta_bytes),    "alloc max_log")
        d_exp_sum = CHECK(drv.cuMemAlloc(meta_bytes),    "alloc exp_sum")
        CHECK(drv.cuMemsetD8(d_tmp_out, 0, tmp_out_bytes), "zero tmp_out")
        CHECK(drv.cuMemsetD8(d_max_log, 0, meta_bytes),    "zero max_log")
        CHECK(drv.cuMemsetD8(d_exp_sum, 0, meta_bytes),    "zero exp_sum")

        # Context lens + attn scale.
        ctx_lens_i32 = np.full(num_seqs, context_len, dtype=np.int32)
        d_ctx = CHECK(drv.cuMemAlloc(ctx_lens_i32.nbytes), "alloc ctx_i32")
        CHECK(drv.cuMemcpyHtoD(d_ctx, ctx_lens_i32.ctypes.data,
                               ctx_lens_i32.nbytes), "ctx_i32 H2D")
        attn_scale = np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32)

        # Final output for reduce phase.
        out_bytes = num_seqs * num_heads * head_dim * 2
        d_out = CHECK(drv.cuMemAlloc(out_bytes), "alloc out_cuda")
        CHECK(drv.cuMemsetD8(d_out, 0, out_bytes), "zero out_cuda")

        use_bc16 = head_dim > 256
        fa2_bc = 16 if use_bc16 else 32
        fn_split = fn_split_bc16 if use_bc16 else fn_split_bc32
        # Smem for split kernel: 2 * BC * D (f16 K + V) + BC*4 (scores) +
        # 128/32 * 4 (reduce).
        split_smem = 2 * fa2_bc * head_dim * 2 + fa2_bc * 4 + (128 // 32) * 4
        if split_smem >= 48 * 1024:
            CHECK(drv.cuFuncSetAttribute(fn_split, SMEM_OPT_IN, split_smem),
                  "opt-in split smem")

        split_params = [
            np.array([int(d_tmp_out)],    dtype=np.uint64),
            np.array([int(d_max_log)],    dtype=np.uint64),
            np.array([int(d_exp_sum)],    dtype=np.uint64),
            np.array([int(d_q_fp8)],      dtype=np.uint64),
            np.array([int(d_key_pack)],   dtype=np.uint64),
            np.array([int(d_val_pack)],   dtype=np.uint64),
            np.array([int(d_key_sc)],     dtype=np.uint64),
            np.array([int(d_val_sc)],     dtype=np.uint64),
            np.array([int(d_bt)],         dtype=np.uint64),
            np.array([int(d_ctx)],        dtype=np.uint64),
            np.array([int(d_qs)],         dtype=np.uint64),
            attn_scale,
            np.array([num_heads],         dtype=np.int32),
            np.array([num_kv_heads],      dtype=np.int32),
            np.array([head_dim],          dtype=np.int32),
            np.array([block_size],        dtype=np.int32),
            np.array([blocks_per_seq],    dtype=np.int32),
            np.array([window_size_left],  dtype=np.int32),
            np.array([partition_size],    dtype=np.int32),
            np.array([max_num_partitions], dtype=np.int32),
        ]
        pp = np.array([p.ctypes.data for p in split_params], dtype=np.uint64)
        CHECK(drv.cuLaunchKernel(fn_split,
                                 num_seqs, num_heads, max_num_partitions,
                                 128, 1, 1,
                                 split_smem, 0, pp.ctypes.data, 0),
              "split launch")

        # Reduce: smem = 2 * max_num_partitions * 4 + (128/32)*4.
        reduce_smem = 2 * max_num_partitions * 4 + (128 // 32) * 4
        if reduce_smem >= 48 * 1024:
            CHECK(drv.cuFuncSetAttribute(fn_reduce, SMEM_OPT_IN, reduce_smem),
                  "opt-in reduce smem")
        reduce_params = [
            np.array([int(d_out)],     dtype=np.uint64),
            np.array([int(d_tmp_out)], dtype=np.uint64),
            np.array([int(d_max_log)], dtype=np.uint64),
            np.array([int(d_exp_sum)], dtype=np.uint64),
            np.array([int(d_ctx)],     dtype=np.uint64),
            np.array([num_heads],      dtype=np.int32),
            np.array([head_dim],       dtype=np.int32),
            np.array([max_num_partitions], dtype=np.int32),
            np.array([partition_size], dtype=np.int32),
        ]
        pr = np.array([p.ctypes.data for p in reduce_params], dtype=np.uint64)
        CHECK(drv.cuLaunchKernel(fn_reduce,
                                 num_seqs, num_heads, 1,
                                 128, 1, 1,
                                 reduce_smem, 0, pr.ctypes.data, 0),
              "reduce launch")
        CHECK(drv.cuCtxSynchronize(), "cuda sync")

        out_cuda_f16 = np.empty((num_seqs, num_heads, head_dim),
                                dtype=np.float16)
        CHECK(drv.cuMemcpyDtoH(out_cuda_f16.ctypes.data, d_out, out_bytes),
              "out_cuda D2H")
        out_cuda = out_cuda_f16.astype(np.float64)

        # Compare CUDA to the numpy split+reduce path (same math; tolerance
        # = fp16 quant noise 5e-3 * peak, matches the existing decode
        # harness).
        cuda_err = np.abs(out_cuda - out_split)
        cuda_peak = np.abs(out_split).max()
        cuda_tol = 5e-3 * max(cuda_peak, 1e-6)
        cuda_bad = int((cuda_err > cuda_tol).sum())
        ok_cuda = cuda_bad == 0

        for d in [d_tmp_out, d_max_log, d_exp_sum, d_ctx, d_out]:
            CHECK(drv.cuMemFree(d), "free cuda")

    ok = ok_np and ok_cuda
    status = "OK  " if ok else "FAIL"
    cuda_tag = ""
    if fn_split_bc32 is not None:
        cuda_tag = (f"  cuda.abs={cuda_err.max():.3e} (≤{cuda_tol:.3e}) "
                    f"cuda.mm={cuda_bad}")
    print(
        f"  {status}  S={num_seqs:>2} H={num_heads:>2} KVH={num_kv_heads:>2} "
        f"hd={head_dim:>3} ctx={context_len:>4} P={partition_size:>3} "
        f"nP={num_parts:>2}{wtag}   "
        f"np.abs={abs_err.max():.3e} (≤{tol:.3e}) np.mm={bad}"
        f"{cuda_tag}"
    )

    for d in [d_key_pack, d_val_pack, d_key_sc, d_val_sc, d_qs, d_cos, d_sin,
              d_bt, d_dummy_q, d_dummy_q_out, d_q_in, d_k_z, d_v_z, d_q_fp8,
              d_pos_d, d_slot_d]:
        CHECK(drv.cuMemFree(d), "free")
    return ok


print(f"device PTX: {ROPE_PTX.name} ({ARCH})")
print("protocol check: numpy split+reduce vs single-pass reference")
print()

all_pass = all([
    # Sanity: P == ctx (single-partition fallback path).
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=32, block_size=16, partition_size=32, seed=301),
    # Two partitions.
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=64, block_size=16, partition_size=32, seed=302),
    # Partition ratio matches vLLM default (P=512 with a 2k context).
    # Keep ctx small here so numpy ref completes quickly; the partition
    # protocol is seq-len-agnostic.
    run_one(num_seqs=2, num_heads=8, num_kv_heads=2,
            head_dim=256, rotary_dim=128,
            context_len=256, block_size=16, partition_size=64, seed=303),
    # Uneven last partition.
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=80, block_size=16, partition_size=32, seed=304),
    # head_dim=512 (Gemma 4 global-attn shape).
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=512, rotary_dim=128,
            context_len=128, block_size=16, partition_size=32, seed=305),
    # GQA ratio 4 (edge of Gemma family).
    run_one(num_seqs=1, num_heads=8, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=128, block_size=16, partition_size=32, seed=306),
    # Sliding window — window_start inside some partition, earlier
    # partitions get fully masked (sentinel path).
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=128, block_size=16, partition_size=32, seed=307,
            window_size_left=47),
    # Sliding window smaller than one partition — tests the edge mask.
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=256, rotary_dim=128,
            context_len=256, block_size=16, partition_size=64, seed=308,
            window_size_left=31),
    # Many partitions (stresses phase-2 numerical combine).
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=512, block_size=16, partition_size=32, seed=309),
])
print()
print("protocol OK — numpy split+reduce matches single-pass reference"
      if all_pass
      else "FAIL: split-reduce protocol disagrees with direct reference")
if not all_pass:
    sys.exit(1)
