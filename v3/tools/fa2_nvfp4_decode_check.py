#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fa2_nvfp4_decode_check.py [sm_xxx]
#
# End-to-end precision harness for
# `flash_attention_2_decode_nvfp4kv_kernel` (+ _bc16). We build the
# cache by running real FP16 K/V through the already-validated
# `fused_rope_partial_nvfp4kv_kernel`, then decode-query through the
# NVFP4 FA2 kernel, and compare the f16 output to an fp64 reference
# attention that consumes the same quantised+dequantised K/V
# tensors. That way the test only measures kernel fidelity vs the
# fp64 idealised softmax — NOT vs. raw (pre-quant) FP16, which would
# conflate the FA2 path with the intrinsic NVFP4 quant noise already
# covered by rope_nvfp4kv_check.py.
#
# Gate: per-element abs error on the f16 output ≤ 5e-3 * peak(|ref|).
# That's the same looseness used by `fp8_gemm_channelscale_check.py`
# for a similar pattern (small-batch decode, f16 output), and it
# covers one f16 mantissa ULP plus the softmax-on-quantised-K
# sensitivity.

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
DECODE_PTX = REPO / "kernels" / ARCH / "flash_attention_nvfp4kv.ptx"
ROPE_PTX   = REPO / "kernels" / ARCH / "fused_rope_partial_nvfp4kv.ptx"

for p, name in [(DECODE_PTX, "decode kernel"), (ROPE_PTX, "RoPE kernel")]:
    if not p.exists():
        sys.exit(
            f"missing {name} PTX: {p}\n"
            f"  build with kernels/build.sh {ARCH} (or nvcc -arch={ARCH}a)"
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

rope_mod   = CHECK(drv.cuModuleLoadData(ROPE_PTX.read_bytes() + b"\0"),   "load rope")
decode_mod = CHECK(drv.cuModuleLoadData(DECODE_PTX.read_bytes() + b"\0"), "load decode")
fn_rope = CHECK(drv.cuModuleGetFunction(rope_mod, b"fused_rope_partial_nvfp4kv_kernel"),
                "get rope fn")
fn_decode_bc32 = CHECK(drv.cuModuleGetFunction(
    decode_mod, b"flash_attention_2_decode_nvfp4kv_kernel"), "get decode bc32")
fn_decode_bc16 = CHECK(drv.cuModuleGetFunction(
    decode_mod, b"flash_attention_2_decode_nvfp4kv_bc16_kernel"), "get decode bc16")

# Opt the decode kernels into the sm_121 99 KB dynamic-smem ceiling
# on-demand (matches the Rust launcher's cuFuncSetAttribute call).
SMEM_OPT_IN = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


def fp4_decode_table():
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def e4m3_decode_bytes(buf: np.ndarray) -> np.ndarray:
    # Same routine as rope_nvfp4kv_check.py.
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


def fp8_e4m3_quantise(x: np.ndarray) -> np.ndarray:
    """Quantise fp32 → fp8 E4M3 → back to fp32. Used to mirror Q's
    per-tensor FP8 quantise-dequant inside the reference."""
    sign = np.sign(x)
    mag = np.abs(x).astype(np.float64)
    mag = np.minimum(mag, 448.0)
    mag_safe = np.maximum(mag, 1e-45)
    exp = np.floor(np.log2(mag_safe))
    exp = np.clip(exp, -6, 8)
    frac = mag / (2.0 ** exp)
    q = np.clip(np.round((frac - 1.0) * 8.0), 0, 7)
    rec = (1.0 + q / 8.0) * (2.0 ** exp)
    sub = mag < 2.0 ** -6
    if sub.any():
        sub_q = np.clip(np.round(mag[sub] * (2.0 ** 9)), 0, 7)
        rec[sub] = sub_q * (2.0 ** -9)
    rec[mag == 0.0] = 0.0
    return sign * rec


# ----- Run one shape ------------------------------------------------

def run_one(
    num_seqs, num_heads, num_kv_heads, head_dim, rotary_dim,
    context_len, block_size, seed, window_size_left=-1
):
    rng = np.random.default_rng(seed)
    assert context_len % block_size == 0 or context_len < block_size, \
        "keep context_len a multiple of block_size for the probe"

    # --- Host-side setup of Q, K, V history, positions, slots. ---
    # Each seq gets a fresh KV history of length `context_len`. Q is a
    # single row per seq (decode step).
    history_q = rng.standard_normal((context_len, num_seqs, num_heads, head_dim)).astype(np.float16)  # noqa: F841 (not used; decode uses only the current Q)
    history_k = rng.standard_normal((context_len, num_seqs, num_kv_heads, head_dim)).astype(np.float16)
    history_v = rng.standard_normal((context_len, num_seqs, num_kv_heads, head_dim)).astype(np.float16)

    max_pos = context_len + 16
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10000 ** (np.arange(half_rot) / half_rot))
    cos = np.cos(np.arange(max_pos)[:, None] * freqs[None, :]).astype(np.float16)
    sin = np.sin(np.arange(max_pos)[:, None] * freqs[None, :]).astype(np.float16)

    # --- Step 1: Allocate cache + populate it by looping the RoPE
    # write kernel once per history-step. This mirrors what the engine
    # does during prefill — appending tokens to the paged cache one
    # by one. ---
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

    # Per-tensor Q FP8 scale.
    q_scale_val = np.float32(0.5)
    d_qs = CHECK(drv.cuMemAlloc(4), "alloc qs")
    CHECK(drv.cuMemcpyHtoD(d_qs, q_scale_val.tobytes(), 4), "qs H2D")

    # cos / sin tables, positions, slots on device.
    d_cos = CHECK(drv.cuMemAlloc(cos.nbytes), "alloc cos")
    d_sin = CHECK(drv.cuMemAlloc(sin.nbytes), "alloc sin")
    CHECK(drv.cuMemcpyHtoD(d_cos, cos.ctypes.data, cos.nbytes), "cos H2D")
    CHECK(drv.cuMemcpyHtoD(d_sin, sin.ctypes.data, sin.nbytes), "sin H2D")

    # Block table: contiguous per-seq layout.
    blocks_per_seq = (context_len + block_size - 1) // block_size
    block_table = np.arange(num_seqs * blocks_per_seq, dtype=np.int32).reshape(
        num_seqs, blocks_per_seq
    )
    d_bt = CHECK(drv.cuMemAlloc(block_table.nbytes), "alloc bt")
    CHECK(drv.cuMemcpyHtoD(d_bt, block_table.ctypes.data, block_table.nbytes), "bt H2D")

    # Dummy Q input for the RoPE kernel during history population (we
    # don't use the Q output during history; only K/V land in the cache).
    dummy_q = np.zeros((num_seqs, num_heads, head_dim), dtype=np.float16)
    d_dummy_q = CHECK(drv.cuMemAlloc(dummy_q.nbytes), "alloc dq")
    CHECK(drv.cuMemcpyHtoD(d_dummy_q, dummy_q.ctypes.data, dummy_q.nbytes), "dq H2D")
    d_dummy_q_out = CHECK(drv.cuMemAlloc(dummy_q.nbytes), "alloc dqo")

    # Populate the cache one history step at a time.
    for step in range(context_len):
        k_step = np.ascontiguousarray(history_k[step]).astype(np.float16)
        v_step = np.ascontiguousarray(history_v[step]).astype(np.float16)
        d_k_step = CHECK(drv.cuMemAlloc(k_step.nbytes), "alloc k_step")
        d_v_step = CHECK(drv.cuMemAlloc(v_step.nbytes), "alloc v_step")
        CHECK(drv.cuMemcpyHtoD(d_k_step, k_step.ctypes.data, k_step.nbytes), "k_step H2D")
        CHECK(drv.cuMemcpyHtoD(d_v_step, v_step.ctypes.data, v_step.nbytes), "v_step H2D")

        pos_step = np.full(num_seqs, step, dtype=np.int32)
        # Slots: seq s, history step t → logical KV position t, which
        # lives in block `block_table[s, t/bs]`, offset `t % bs`. The
        # kernel uses `slot_mapping[seq] = phys_block*bs + off`.
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

    # --- Step 2: Build the Q row for the decode step + feed it. ---
    q_decode = rng.standard_normal((num_seqs, num_heads, head_dim)).astype(np.float16)
    # Emulate what the engine does: RoPE the Q at step=context_len,
    # quantise to FP8 per-tensor with scale q_scale_val. Run a small
    # RoPE launch just to produce the FP8-quantised Q consistently.
    zero_k = np.zeros((num_seqs, num_kv_heads, head_dim), dtype=np.float16)
    zero_v = np.zeros_like(zero_k)
    d_q_in  = CHECK(drv.cuMemAlloc(q_decode.nbytes), "alloc qin")
    d_k_z   = CHECK(drv.cuMemAlloc(zero_k.nbytes),   "alloc kz")
    d_v_z   = CHECK(drv.cuMemAlloc(zero_v.nbytes),   "alloc vz")
    CHECK(drv.cuMemcpyHtoD(d_q_in, q_decode.ctypes.data, q_decode.nbytes), "qin H2D")
    CHECK(drv.cuMemcpyHtoD(d_k_z,  zero_k.ctypes.data,   zero_k.nbytes),   "kz H2D")
    CHECK(drv.cuMemcpyHtoD(d_v_z,  zero_v.ctypes.data,   zero_v.nbytes),   "vz H2D")
    d_q_fp8 = CHECK(drv.cuMemAlloc(num_seqs * num_heads * head_dim), "alloc qfp8")

    pos_d = np.full(num_seqs, context_len, dtype=np.int32)
    slot_d = np.full(num_seqs, -1, dtype=np.int32)  # slot<0 → no KV write
    d_pos_d  = CHECK(drv.cuMemAlloc(pos_d.nbytes),  "alloc posd")
    d_slot_d = CHECK(drv.cuMemAlloc(slot_d.nbytes), "alloc slotd")
    CHECK(drv.cuMemcpyHtoD(d_pos_d, pos_d.ctypes.data, pos_d.nbytes), "posd H2D")
    CHECK(drv.cuMemcpyHtoD(d_slot_d, slot_d.ctypes.data, slot_d.nbytes), "slotd H2D")

    params = [
        np.array([int(d_q_in)],   dtype=np.uint64),
        np.array([int(d_k_z)],    dtype=np.uint64),
        np.array([int(d_v_z)],    dtype=np.uint64),
        np.array([int(d_q_fp8)],  dtype=np.uint64),
        np.array([int(d_key_pack)],    dtype=np.uint64),
        np.array([int(d_val_pack)],    dtype=np.uint64),
        np.array([int(d_key_sc)],      dtype=np.uint64),
        np.array([int(d_val_sc)],      dtype=np.uint64),
        np.array([int(d_cos)],         dtype=np.uint64),
        np.array([int(d_sin)],         dtype=np.uint64),
        np.array([int(d_pos_d)],       dtype=np.uint64),
        np.array([int(d_slot_d)],      dtype=np.uint64),
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
                             0, 0, pp.ctypes.data, 0), "rope decode Q")
    CHECK(drv.cuCtxSynchronize(), "sync qfp8")

    # --- Step 3: Run the decode kernel. ---
    ctx_lens = np.full(num_seqs, context_len, dtype=np.int32)
    d_ctx = CHECK(drv.cuMemAlloc(ctx_lens.nbytes), "alloc ctx")
    CHECK(drv.cuMemcpyHtoD(d_ctx, ctx_lens.ctypes.data, ctx_lens.nbytes), "ctx H2D")

    output_bytes = num_seqs * num_heads * head_dim * 2
    d_out = CHECK(drv.cuMemAlloc(output_bytes), "alloc out")
    CHECK(drv.cuMemsetD8(d_out, 0, output_bytes), "zero out")

    attn_scale = np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32)
    use_bc16 = head_dim > 256
    fa2_bc = 16 if use_bc16 else 32
    smem_bytes = 4 * (2 * fa2_bc * head_dim + fa2_bc + 128 // 32)
    fn_decode = fn_decode_bc16 if use_bc16 else fn_decode_bc32
    if smem_bytes >= 48 * 1024:
        CHECK(drv.cuFuncSetAttribute(fn_decode, SMEM_OPT_IN, smem_bytes),
              "opt-in smem")

    params = [
        np.array([int(d_out)],        dtype=np.uint64),
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
    ]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(fn_decode,
                             num_seqs, num_heads, 1,
                             128, 1, 1,
                             smem_bytes, 0, pp.ctypes.data, 0), "decode launch")
    CHECK(drv.cuCtxSynchronize(), "decode sync")

    out_f16 = np.empty((num_seqs, num_heads, head_dim), dtype=np.float16)
    CHECK(drv.cuMemcpyDtoH(out_f16.ctypes.data, d_out, output_bytes), "out D2H")

    # --- Step 4: fp64 reference. ---
    # Read packed cache + scales back, dequant to get the K/V history
    # the kernel actually saw. Build reference attention that also
    # quantises Q through FP8 per-tensor (matches kernel).
    kp = np.empty(packed_bytes_total, dtype=np.uint8)
    vp = np.empty(packed_bytes_total, dtype=np.uint8)
    ks = np.empty(scale_bytes_total,  dtype=np.uint8)
    vs = np.empty(scale_bytes_total,  dtype=np.uint8)
    CHECK(drv.cuMemcpyDtoH(kp.ctypes.data, d_key_pack, packed_bytes_total), "kp D2H")
    CHECK(drv.cuMemcpyDtoH(vp.ctypes.data, d_val_pack, packed_bytes_total), "vp D2H")
    CHECK(drv.cuMemcpyDtoH(ks.ctypes.data, d_key_sc,   scale_bytes_total),  "ks D2H")
    CHECK(drv.cuMemcpyDtoH(vs.ctypes.data, d_val_sc,   scale_bytes_total),  "vs D2H")

    # Reshape to [num_blocks, block_size, num_kv_heads, head_dim/2 (or /16)]
    kp_r = kp.reshape(num_blocks, block_size, num_kv_heads, head_dim // 2)
    vp_r = vp.reshape(num_blocks, block_size, num_kv_heads, head_dim // 2)
    ks_r = e4m3_decode_bytes(ks).reshape(num_blocks, block_size, num_kv_heads, head_dim // 16)
    vs_r = e4m3_decode_bytes(vs).reshape(num_blocks, block_size, num_kv_heads, head_dim // 16)

    k_dq_full = nvfp4_dequant(kp_r, ks_r, head_dim)   # [NB, bs, KH, D]
    v_dq_full = nvfp4_dequant(vp_r, vs_r, head_dim)

    # Pull the history back into [T, S, KH, D] via block_table.
    k_hist = np.empty((context_len, num_seqs, num_kv_heads, head_dim), dtype=np.float64)
    v_hist = np.empty_like(k_hist)
    for s in range(num_seqs):
        for t in range(context_len):
            blk = block_table[s, t // block_size]
            off = t % block_size
            k_hist[t, s] = k_dq_full[blk, off]
            v_hist[t, s] = v_dq_full[blk, off]

    # Q reference: dequant the FP8 Q the kernel saw (same as its load).
    q_fp8 = np.empty(num_seqs * num_heads * head_dim, dtype=np.uint8)
    CHECK(drv.cuMemcpyDtoH(q_fp8.ctypes.data, d_q_fp8, q_fp8.nbytes), "qfp8 D2H")
    q_ref = fp8_e4m3_decode(q_fp8).reshape(num_seqs, num_heads, head_dim) * float(q_scale_val)

    # Reference attention: scores = softmax(Q · K^T * scale), out = scores · V.
    # K/V indexed by [T, S, KH, D]; need to broadcast KH to H via GQA.
    gqa = num_heads // num_kv_heads

    out_ref = np.empty_like(q_ref)
    # Sliding-window reference: mirror the kernel's `window_start`
    # computation. `window_size_left < 0` disables the mask.
    decode_q_abs = context_len - 1
    ref_window_start = (0 if window_size_left < 0
                         else max(0, decode_q_abs - window_size_left))
    for s in range(num_seqs):
        for h in range(num_heads):
            kvh = h // gqa
            q_row = q_ref[s, h]                              # [D]
            k_hist_h = k_hist[:, s, kvh]                     # [T, D]
            v_hist_h = v_hist[:, s, kvh]                     # [T, D]
            scores = (k_hist_h @ q_row) * float(attn_scale[0])  # [T]
            if ref_window_start > 0:
                scores[:ref_window_start] = -np.inf
            scores -= scores.max()
            probs = np.exp(scores); probs /= probs.sum()
            out_ref[s, h] = (probs[:, None] * v_hist_h).sum(axis=0)

    out_got = out_f16.astype(np.float64)
    abs_err = np.abs(out_got - out_ref)
    peak = np.abs(out_ref).max()
    tol = 5e-3 * max(peak, 1e-6)
    bad = int((abs_err > tol).sum())
    ok = bad == 0
    status = "OK  " if ok else "FAIL"
    wtag = f" ws={window_size_left}" if window_size_left >= 0 else ""
    print(
        f"  {status}  S={num_seqs:>2} H={num_heads:>2} KVH={num_kv_heads:>2} "
        f"hd={head_dim:>3} rd={rotary_dim:>3} ctx={context_len:>3} bs={block_size:>2} "
        f"bc={fa2_bc:>2}{wtag}   "
        f"abs_err.max={abs_err.max():.3e} (≤{tol:.3e})  "
        f"mismatches={bad}/{out_got.size}"
    )

    # Cleanup.
    for d in [d_key_pack, d_val_pack, d_key_sc, d_val_sc, d_qs, d_cos, d_sin,
              d_bt, d_dummy_q, d_dummy_q_out, d_q_in, d_k_z, d_v_z, d_q_fp8,
              d_pos_d, d_slot_d, d_ctx, d_out]:
        CHECK(drv.cuMemFree(d), "free")
    return ok


print(f"device PTX: {DECODE_PTX.name} + {ROPE_PTX.name} ({ARCH})")
all_pass = all([
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=32, block_size=16, seed=1),
    run_one(num_seqs=2, num_heads=8, num_kv_heads=2,
            head_dim=256, rotary_dim=128,
            context_len=64, block_size=16, seed=7),
    run_one(num_seqs=2, num_heads=8, num_kv_heads=2,
            head_dim=512, rotary_dim=128,
            context_len=64, block_size=16, seed=42),  # exercises BC=16
    # Sliding-window cases — decode kernel's tile-skip + mask path.
    # Gemma 4 sliding layers use window_size_left = 1023, so at large
    # context most tiles are below window_start and must be skipped.
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            context_len=128, block_size=16, seed=101, window_size_left=31),
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=256, rotary_dim=128,
            context_len=256, block_size=16, seed=102, window_size_left=63),
    # Edge tile: window_start straddles the middle of a BC-sized block.
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=256, rotary_dim=128,
            context_len=128, block_size=16, seed=103, window_size_left=40),
])
print()
print("all shapes pass" if all_pass
      else "FAIL: decode kernel exceeded the fp64 reference bound")
if not all_pass:
    sys.exit(1)
