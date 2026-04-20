#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fa2_nvfp4_prefill_check.py [sm_xxx]
#
# End-to-end precision harness for
# `flash_attention_2_prefill_nvfp4kv_kernel` (+ _bc16). Mirrors the
# decode harness: RoPE-writes a synthetic KV history into the paged
# cache via the already-validated `fused_rope_partial_nvfp4kv_kernel`,
# feeds a multi-token query batch through the prefill kernel, and
# compares the f16 output to fp64 reference attention that consumes
# the same quantised+dequantised K/V tensors + applies the causal
# mask.
#
# Gate: abs_err ≤ 5e-3 · peak(|ref|), same as the decode harness.

import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
DECODE_PTX = REPO / "kernels" / ARCH / "flash_attention_nvfp4kv.ptx"
ROPE_PTX   = REPO / "kernels" / ARCH / "fused_rope_partial_nvfp4kv.ptx"

for p, name in [(DECODE_PTX, "prefill/decode"), (ROPE_PTX, "RoPE")]:
    if not p.exists():
        sys.exit(f"missing {name} PTX: {p}")


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
pre_mod  = CHECK(drv.cuModuleLoadData(DECODE_PTX.read_bytes() + b"\0"), "load pre")
fn_rope = CHECK(drv.cuModuleGetFunction(rope_mod, b"fused_rope_partial_nvfp4kv_kernel"),
                "get rope fn")
fn_pre_bc32 = CHECK(drv.cuModuleGetFunction(
    pre_mod, b"flash_attention_2_prefill_nvfp4kv_kernel"), "get pre bc32")
fn_pre_bc16 = CHECK(drv.cuModuleGetFunction(
    pre_mod, b"flash_attention_2_prefill_nvfp4kv_bc16_kernel"), "get pre bc16")

SMEM_OPT_IN = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


def fp4_table():
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def e4m3_bytes_to_f64(buf: np.ndarray) -> np.ndarray:
    b = buf.astype(np.uint32)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float64)
    val = np.where(exp == 0, (man / 8.0) * (2.0 ** -6),
                             (1.0 + man / 8.0) * (2.0 ** (exp - 7)))
    val = np.where((exp == 15) & (man == 7), 0.0, val)
    return sign * val


def nvfp4_dequant(packed: np.ndarray, scales: np.ndarray, inner: int) -> np.ndarray:
    table = fp4_table()
    flat = packed.astype(np.uint32)
    lo = table[flat & 0x7] * np.where((flat & 0x8) != 0, -1.0, 1.0)
    hi = table[(flat >> 4) & 0x7] * np.where(((flat >> 4) & 0x8) != 0, -1.0, 1.0)
    inter = np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], inner)
    sc = np.repeat(scales, 16, axis=-1)
    return inter * sc


def run_one(num_seqs, num_heads, num_kv_heads, head_dim, rotary_dim,
            q_lens, block_size, seed):
    """q_lens: list[int] of length num_seqs — per-seq query length
    during this prefill. KV history length = q_len (typical prefill
    case: Q and K align on the diagonal)."""
    assert len(q_lens) == num_seqs
    rng = np.random.default_rng(seed)
    context_lens = np.array(q_lens, dtype=np.int32)
    max_ctx = int(context_lens.max())

    # Build the "history" K/V. In a real prefill, K/V at position i
    # come from the same tokens as Q[i]; we mirror that by running
    # the RoPE-write kernel over the full q_len per seq.
    history_k = rng.standard_normal((max_ctx, num_seqs, num_kv_heads, head_dim)).astype(np.float16)
    history_v = rng.standard_normal((max_ctx, num_seqs, num_kv_heads, head_dim)).astype(np.float16)
    history_q = rng.standard_normal((max_ctx, num_seqs, num_heads, head_dim)).astype(np.float16)

    half_rot = rotary_dim // 2
    freqs = 1.0 / (10000 ** (np.arange(half_rot) / half_rot))
    cos = np.cos(np.arange(max_ctx + 1)[:, None] * freqs[None, :]).astype(np.float16)
    sin = np.sin(np.arange(max_ctx + 1)[:, None] * freqs[None, :]).astype(np.float16)

    blocks_per_seq = (max_ctx + block_size - 1) // block_size
    num_blocks = blocks_per_seq * num_seqs
    block_table = np.arange(num_seqs * blocks_per_seq, dtype=np.int32).reshape(
        num_seqs, blocks_per_seq)
    cu_seqlens_q = np.concatenate([[0], np.cumsum(context_lens)]).astype(np.int32)
    total_q = int(cu_seqlens_q[-1])

    packed_total = num_blocks * block_size * num_kv_heads * (head_dim // 2)
    scale_total  = num_blocks * block_size * num_kv_heads * (head_dim // 16)

    def alloc(n):  return CHECK(drv.cuMemAlloc(n), f"alloc {n}")
    def h2d(d, a): CHECK(drv.cuMemcpyHtoD(d, a.ctypes.data, a.nbytes), "H2D")
    def d2h(a, d): CHECK(drv.cuMemcpyDtoH(a.ctypes.data, d, a.nbytes), "D2H")

    d_kp = alloc(packed_total); d_vp = alloc(packed_total)
    d_ks = alloc(scale_total);  d_vs = alloc(scale_total)
    for d, n in [(d_kp, packed_total), (d_vp, packed_total), (d_ks, scale_total), (d_vs, scale_total)]:
        CHECK(drv.cuMemsetD8(d, 0, n), "zero")

    q_scale_val = np.array([0.5], dtype=np.float32)
    d_qs = alloc(4); h2d(d_qs, q_scale_val)
    d_cos = alloc(cos.nbytes); h2d(d_cos, cos)
    d_sin = alloc(sin.nbytes); h2d(d_sin, sin)
    d_bt = alloc(block_table.nbytes); h2d(d_bt, block_table)
    d_cu = alloc(cu_seqlens_q.nbytes); h2d(d_cu, cu_seqlens_q)

    # --- Step 1: populate KV history by running RoPE kernel per step. ---
    # Build the real Q buffer inline: each seq's Q rows go at cu_seqlens_q
    # offsets; between seqs the buffer is packed tight.
    q_dense = np.zeros((total_q, num_heads, head_dim), dtype=np.float16)
    for s in range(num_seqs):
        q_dense[cu_seqlens_q[s]:cu_seqlens_q[s + 1]] = history_q[:q_lens[s], s]
    q_fp8_out = alloc(total_q * num_heads * head_dim)

    dummy_q_step = np.zeros((num_seqs, num_heads, head_dim), dtype=np.float16)
    d_dq_in  = alloc(dummy_q_step.nbytes); h2d(d_dq_in, dummy_q_step)
    d_dq_out = alloc(dummy_q_step.nbytes)

    for step in range(max_ctx):
        k_step = np.ascontiguousarray(history_k[step])
        v_step = np.ascontiguousarray(history_v[step])
        d_kstep = alloc(k_step.nbytes); h2d(d_kstep, k_step)
        d_vstep = alloc(v_step.nbytes); h2d(d_vstep, v_step)
        pos_step = np.full(num_seqs, step, dtype=np.int32)
        slot_step = np.array([
            block_table[s, step // block_size] * block_size + (step % block_size)
            if step < context_lens[s] else -1
            for s in range(num_seqs)
        ], dtype=np.int32)
        d_ps = alloc(pos_step.nbytes); h2d(d_ps, pos_step)
        d_ss = alloc(slot_step.nbytes); h2d(d_ss, slot_step)

        params = [
            np.array([int(d_dq_in)], dtype=np.uint64),
            np.array([int(d_kstep)], dtype=np.uint64),
            np.array([int(d_vstep)], dtype=np.uint64),
            np.array([int(d_dq_out)], dtype=np.uint64),
            np.array([int(d_kp)], dtype=np.uint64),
            np.array([int(d_vp)], dtype=np.uint64),
            np.array([int(d_ks)], dtype=np.uint64),
            np.array([int(d_vs)], dtype=np.uint64),
            np.array([int(d_cos)], dtype=np.uint64),
            np.array([int(d_sin)], dtype=np.uint64),
            np.array([int(d_ps)], dtype=np.uint64),
            np.array([int(d_ss)], dtype=np.uint64),
            np.array([int(d_qs)], dtype=np.uint64),
            np.array([num_seqs], dtype=np.int32),
            np.array([num_heads], dtype=np.int32),
            np.array([num_kv_heads], dtype=np.int32),
            np.array([head_dim], dtype=np.int32),
            np.array([rotary_dim], dtype=np.int32),
        ]
        pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
        CHECK(drv.cuLaunchKernel(fn_rope, num_seqs, max(num_heads, num_kv_heads), 1,
                                 head_dim, 1, 1, 0, 0, pp.ctypes.data, 0), "rope")
        CHECK(drv.cuCtxSynchronize(), "rope sync")
        for d in (d_kstep, d_vstep, d_ps, d_ss):
            CHECK(drv.cuMemFree(d), "free step")

    # --- Step 2: per-token Q → FP8 via one RoPE pass. Q is the same
    # per-token RoPE'd + FP8-quantised as during bring-up; run it
    # per-token by calling the RoPE kernel with num_seqs=total_q and
    # slot_mapping = -1 everywhere. Each Q row gets its matching
    # position from `positions_q`. Saves writing a dedicated quantiser. ---
    positions_q = np.zeros(total_q, dtype=np.int32)
    for s in range(num_seqs):
        for ti in range(q_lens[s]):
            positions_q[cu_seqlens_q[s] + ti] = ti
    slot_q = np.full(total_q, -1, dtype=np.int32)

    d_q_in  = alloc(q_dense.nbytes); h2d(d_q_in, q_dense)
    zero_k  = np.zeros((total_q, num_kv_heads, head_dim), dtype=np.float16)
    zero_v  = np.zeros_like(zero_k)
    d_zk = alloc(zero_k.nbytes); h2d(d_zk, zero_k)
    d_zv = alloc(zero_v.nbytes); h2d(d_zv, zero_v)
    d_pq = alloc(positions_q.nbytes); h2d(d_pq, positions_q)
    d_sq = alloc(slot_q.nbytes); h2d(d_sq, slot_q)

    params = [
        np.array([int(d_q_in)],    dtype=np.uint64),
        np.array([int(d_zk)],      dtype=np.uint64),
        np.array([int(d_zv)],      dtype=np.uint64),
        np.array([int(q_fp8_out)], dtype=np.uint64),
        np.array([int(d_kp)],      dtype=np.uint64),
        np.array([int(d_vp)],      dtype=np.uint64),
        np.array([int(d_ks)],      dtype=np.uint64),
        np.array([int(d_vs)],      dtype=np.uint64),
        np.array([int(d_cos)],     dtype=np.uint64),
        np.array([int(d_sin)],     dtype=np.uint64),
        np.array([int(d_pq)],      dtype=np.uint64),
        np.array([int(d_sq)],      dtype=np.uint64),
        np.array([int(d_qs)],      dtype=np.uint64),
        np.array([total_q],        dtype=np.int32),
        np.array([num_heads],      dtype=np.int32),
        np.array([num_kv_heads],   dtype=np.int32),
        np.array([head_dim],       dtype=np.int32),
        np.array([rotary_dim],     dtype=np.int32),
    ]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(fn_rope, total_q, max(num_heads, num_kv_heads), 1,
                             head_dim, 1, 1, 0, 0, pp.ctypes.data, 0), "rope Q")
    CHECK(drv.cuCtxSynchronize(), "rope Q sync")

    # --- Step 3: prefill launch. ---
    d_ctx = alloc(context_lens.nbytes); h2d(d_ctx, context_lens)
    d_out = alloc(total_q * num_heads * head_dim * 2)
    CHECK(drv.cuMemsetD8(d_out, 0, total_q * num_heads * head_dim * 2), "zero out")

    attn_scale = np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32)
    use_bc16 = head_dim > 256
    fa2_bc = 16 if use_bc16 else 32
    smem = 4 * (2 * fa2_bc * head_dim + fa2_bc + 128 // 32)
    fn = fn_pre_bc16 if use_bc16 else fn_pre_bc32
    if smem >= 48 * 1024:
        CHECK(drv.cuFuncSetAttribute(fn, SMEM_OPT_IN, smem), "smem opt-in")

    params = [
        np.array([int(d_out)],     dtype=np.uint64),
        np.array([int(q_fp8_out)], dtype=np.uint64),
        np.array([int(d_kp)],      dtype=np.uint64),
        np.array([int(d_vp)],      dtype=np.uint64),
        np.array([int(d_ks)],      dtype=np.uint64),
        np.array([int(d_vs)],      dtype=np.uint64),
        np.array([int(d_bt)],      dtype=np.uint64),
        np.array([int(d_ctx)],     dtype=np.uint64),
        np.array([int(d_cu)],      dtype=np.uint64),
        np.array([int(d_qs)],      dtype=np.uint64),
        attn_scale,
        np.array([num_heads],      dtype=np.int32),
        np.array([num_kv_heads],   dtype=np.int32),
        np.array([head_dim],       dtype=np.int32),
        np.array([block_size],     dtype=np.int32),
        np.array([blocks_per_seq], dtype=np.int32),
        np.array([-1],             dtype=np.int32),
    ]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(fn, num_seqs, num_heads, 1,
                             128, 1, 1, smem, 0, pp.ctypes.data, 0), "prefill launch")
    CHECK(drv.cuCtxSynchronize(), "prefill sync")

    out_f16 = np.empty((total_q, num_heads, head_dim), dtype=np.float16)
    d2h(out_f16, d_out)

    # --- Reference: pull packed cache + FP8 Q back, build fp64 attn. ---
    kp = np.empty(packed_total, dtype=np.uint8); d2h(kp, d_kp)
    vp = np.empty(packed_total, dtype=np.uint8); d2h(vp, d_vp)
    ks = np.empty(scale_total,  dtype=np.uint8); d2h(ks, d_ks)
    vs = np.empty(scale_total,  dtype=np.uint8); d2h(vs, d_vs)
    q_fp8 = np.empty(total_q * num_heads * head_dim, dtype=np.uint8); d2h(q_fp8, q_fp8_out)

    kp_r = kp.reshape(num_blocks, block_size, num_kv_heads, head_dim // 2)
    vp_r = vp.reshape(num_blocks, block_size, num_kv_heads, head_dim // 2)
    ks_r = e4m3_bytes_to_f64(ks).reshape(num_blocks, block_size, num_kv_heads, head_dim // 16)
    vs_r = e4m3_bytes_to_f64(vs).reshape(num_blocks, block_size, num_kv_heads, head_dim // 16)
    k_dq = nvfp4_dequant(kp_r, ks_r, head_dim)
    v_dq = nvfp4_dequant(vp_r, vs_r, head_dim)

    q_ref = (e4m3_bytes_to_f64(q_fp8).reshape(total_q, num_heads, head_dim)
             * float(q_scale_val[0]))

    out_ref = np.empty_like(q_ref)
    gqa = num_heads // num_kv_heads
    for s in range(num_seqs):
        ctx = context_lens[s]
        q_rows = q_lens[s]
        # Build the per-seq K/V history via block_table.
        k_h = np.stack([k_dq[block_table[s, t // block_size], t % block_size]
                        for t in range(ctx)])  # [ctx, KH, D]
        v_h = np.stack([v_dq[block_table[s, t // block_size], t % block_size]
                        for t in range(ctx)])
        for qi in range(q_rows):
            q_abs = ctx - q_rows + qi
            for h in range(num_heads):
                kvh = h // gqa
                scores = (k_h[:, kvh] @ q_ref[cu_seqlens_q[s] + qi, h]) * float(attn_scale[0])
                # Causal mask: positions > q_abs get -inf.
                scores[q_abs + 1:] = -np.inf
                scores -= scores.max()
                probs = np.exp(scores); probs /= probs.sum()
                out_ref[cu_seqlens_q[s] + qi, h] = (probs[:, None] * v_h[:, kvh]).sum(axis=0)

    abs_err = np.abs(out_f16.astype(np.float64) - out_ref)
    peak = np.abs(out_ref).max()
    tol = 5e-3 * max(peak, 1e-6)
    bad = int((abs_err > tol).sum())
    ok = bad == 0
    status = "OK  " if ok else "FAIL"
    print(
        f"  {status}  S={num_seqs:>2} H={num_heads:>2} KVH={num_kv_heads:>2} "
        f"hd={head_dim:>3} q_lens={q_lens} bs={block_size:>2} bc={fa2_bc:>2}   "
        f"abs_err.max={abs_err.max():.3e} (≤{tol:.3e})  mismatches={bad}/{out_f16.size}"
    )

    for d in [d_kp, d_vp, d_ks, d_vs, d_qs, d_cos, d_sin, d_bt, d_cu,
              d_dq_in, d_dq_out, d_q_in, d_zk, d_zv, d_pq, d_sq,
              q_fp8_out, d_ctx, d_out]:
        CHECK(drv.cuMemFree(d), "free")
    return ok


print(f"device PTX: {DECODE_PTX.name} + {ROPE_PTX.name} ({ARCH})")
all_pass = all([
    run_one(num_seqs=1, num_heads=4, num_kv_heads=2,
            head_dim=128, rotary_dim=128,
            q_lens=[16], block_size=16, seed=1),
    run_one(num_seqs=2, num_heads=8, num_kv_heads=2,
            head_dim=256, rotary_dim=128,
            q_lens=[32, 16], block_size=16, seed=7),
    run_one(num_seqs=2, num_heads=8, num_kv_heads=2,
            head_dim=512, rotary_dim=128,
            q_lens=[16, 32], block_size=16, seed=42),  # exercises BC=16
])
print()
print("all shapes pass" if all_pass
      else "FAIL: prefill kernel exceeded the fp64 reference bound")
if not all_pass:
    sys.exit(1)
