"""MiniMax-M2.7-NVFP4 synthetic-weight throughput benchmark on TPU v6e-8.

Skips the 190K-tensor real model load. Allocates RANDOM NVFP4-packed weights
at the real architecture shapes, builds the same mesh + sharding, runs the
forward pass in a timed loop, reports ms/step and tok/s.

NVFP4 packing matches modelopt format used by the real checkpoint:
  - weight: packed uint8, shape (rows, cols/2), two FP4 values per byte
  - weight_scale: uint8 (FP8 E4M3 bits), shape (rows, cols/16), one scale per 16 weights
  - weight_scale_2: FP32 scalar, per-tensor global scale
Memory footprint per chip ~= 130 GB / 8 = ~16 GB (fits in 32 GB v6e HBM).

Attention backbone + router + embed stay bf16 (matches real checkpoint where
only MoE experts are NVFP4 quantized).

Usage:
    python3 m2_synth_bench.py [--batch 1,8,32,128] [--ctx 2048] \
                              [--iters 20] [--warmup 5] [--nl 62]
"""

import argparse
import os
import sys
import time
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Use the real shard_map MoE from m2_moe.py. This path has proper all-to-all
# dispatch across the 'expert' axis — not the naive gather-based reference.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_moe import moe_block_nvfp4

# --- Arch constants (MiniMax-M2.7-NVFP4) ---
H = 3072
NH = 48
NKV = 8
HEAD_DIM = 128
ROTARY_DIM = 64
ROPE_THETA = 5_000_000.0
MOE_INTER = 1536
NUM_EXPERTS = 256
TOP_K = 8
NL_FULL = 62
VOCAB = 200064
RMS_EPS = 1e-6
GROUP_SIZE = 16

DTYPE = jnp.bfloat16


# --- FP4 E2M1 decode LUT (16 entries) ---
FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32)
FP4_LUT = jnp.asarray(FP4_VALUES, dtype=DTYPE)


def fp8_e4m3_decode(bits_u8: jax.Array) -> jax.Array:
    """FP8 E4M3 -> f32 via IEEE-754 f32 bit layout (Agent 15 fast path)."""
    b = bits_u8.astype(jnp.uint32)
    sign = (b >> 7) & jnp.uint32(1)
    exp = (b >> 3) & jnp.uint32(0xF)
    mant = b & jnp.uint32(0x7)
    sign_bit = sign << jnp.uint32(31)
    u_norm = sign_bit | ((exp + jnp.uint32(120)) << jnp.uint32(23)) | (mant << jnp.uint32(20))
    val_norm = jax.lax.bitcast_convert_type(u_norm, jnp.float32)
    sub = mant.astype(jnp.float32) * jnp.float32((1.0 / 8.0) * (2.0 ** -6))
    sign_f = jnp.where(sign == jnp.uint32(1), jnp.float32(-1.0), jnp.float32(1.0))
    val = jnp.where(exp == jnp.uint32(0), sign_f * sub, val_norm)
    return val


def nvfp4_dequant_bf16(packed: jax.Array, scales_u8: jax.Array, global_scale_f32: float,
                     rows: int, cols: int) -> jax.Array:
    """Dequant NVFP4 weight: packed (rows, cols/2) u8 + scales (rows, cols/16) u8 -> bf16 (rows, cols)."""
    lo = (packed & 0x0F).astype(jnp.uint32)
    hi = ((packed >> 4) & 0x0F).astype(jnp.uint32)
    # Interleave lo (even cols) and hi (odd cols) -> (rows, cols)
    fp4_idx = jnp.stack([lo, hi], axis=-1).reshape(rows, cols)
    vals = FP4_LUT[fp4_idx]  # bf16
    # Decode block scales (F8 E4M3) and broadcast 16x along col axis
    block_scale_f32 = fp8_e4m3_decode(scales_u8)  # (rows, cols/16) f32
    block_scale_bf16 = (block_scale_f32 * global_scale_f32).astype(DTYPE)
    scale_expanded = jnp.repeat(block_scale_bf16, GROUP_SIZE, axis=1)  # (rows, cols) bf16
    return (vals * scale_expanded).astype(DTYPE)


def nvfp4_matmul(x_bf16: jax.Array, w_packed: jax.Array, w_scales: jax.Array,
                 w_global_scale: float, out_features: int, in_features: int) -> jax.Array:
    """Fused: dequant W on the fly, then x @ W^T. XLA should fuse."""
    w_bf16 = nvfp4_dequant_bf16(w_packed, w_scales, w_global_scale, out_features, in_features)
    return x_bf16 @ w_bf16.T


# --- Arch ops ---

def make_mesh_v6e8():
    devs = jax.devices()
    n = len(devs)
    print(f"  devices={n} kind={devs[0].device_kind if n else '?'}", file=sys.stderr)
    mesh_devs = np.array(devs[:min(n, 8)]).reshape((min(n, 8),))
    return Mesh(mesh_devs, ('expert',))


def precompute_rope(theta, rotary_dim, max_ctx):
    freqs = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    angles = np.outer(np.arange(max_ctx, dtype=np.float32), freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rms_norm(x, g):
    x32 = x.astype(jnp.float32)
    return (x * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + RMS_EPS).astype(x.dtype)) * g


def head_rms(h, g):
    h32 = h.astype(jnp.float32)
    return (h * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + RMS_EPS).astype(h.dtype)) * g


def rope_partial(x, cos, sin):
    half = ROTARY_DIM // 2
    xr, xp = x[..., :ROTARY_DIM], x[..., ROTARY_DIM:]
    x0, x1 = xr[..., :half], xr[..., half:]
    rotated = jnp.concatenate([
        x0 * cos.astype(x.dtype) - x1 * sin.astype(x.dtype),
        x0 * sin.astype(x.dtype) + x1 * cos.astype(x.dtype),
    ], axis=-1)
    return jnp.concatenate([rotated, xp], axis=-1)


def attn_layer(x, w, ln, k_cache, v_cache, pos, cos, sin):
    h = rms_norm(x, ln)
    q = (h @ w['q'].T).reshape(-1, NH, HEAD_DIM)
    k = (h @ w['k'].T).reshape(-1, NKV, HEAD_DIM)
    v = (h @ w['v'].T).reshape(-1, NKV, HEAD_DIM)
    q = head_rms(q, w['qn'])
    k = head_rms(k, w['kn'])
    q = rope_partial(q, cos[pos], sin[pos])
    k = rope_partial(k, cos[pos], sin[pos])
    k_cache = jax.lax.dynamic_update_slice(k_cache, k[:, None], (0, pos, 0, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, v[:, None], (0, pos, 0, 0))
    B_ = q.shape[0]
    q_g = q.reshape(B_, NKV, NH // NKV, HEAD_DIM)
    scale = 1.0 / jnp.sqrt(float(HEAD_DIM))
    scores = jnp.einsum('bhgd,bkhd->bhgk', q_g, k_cache.astype(q.dtype)) * scale
    mask = (jnp.arange(k_cache.shape[1]) <= pos)
    scores = jnp.where(mask[None, None, None, :], scores, -1e30)
    p = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    ctx = jnp.einsum('bhgk,bkhd->bhgd', p, v_cache.astype(q.dtype))
    ctx = ctx.reshape(B_, NH * HEAD_DIM)
    out = ctx @ w['o'].T
    return out, k_cache, v_cache


def layer_step(x, lw, k_cache, v_cache, pos, cos, sin, mesh):
    att_out, k_cache, v_cache = attn_layer(x, lw['attn'], lw['ln1'],
                                           k_cache, v_cache, pos, cos, sin)
    x = x + att_out
    h = rms_norm(x, lw['ln2'])
    # Real perf path: shard_map dispatch along 'expert' axis (all-to-all).
    moe_out = moe_block_nvfp4(h, lw['rg'], lw['rb'], lw['w1'], lw['w2'], lw['w3'], mesh)
    return x + moe_out, k_cache, v_cache


def forward_step(x, all_w, k_cache, v_cache, pos, cos, sin, final_norm, lm_head, mesh):
    NL = len(all_w)
    for i in range(NL):
        lw = all_w[i]
        x, k_i, v_i = layer_step(x, lw, k_cache[i], v_cache[i], pos, cos, sin, mesh)
        k_cache = k_cache.at[i].set(k_i)
        v_cache = v_cache.at[i].set(v_i)
    h = rms_norm(x, final_norm)
    logits = h @ lm_head.T
    tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return tok, k_cache, v_cache


def random_weights(B, max_ctx, mesh, nl):
    """Random NVFP4-packed weights. Experts sharded along 'expert' axis.
    Memory ~= 130 GB / 8 chips = ~16 GB/chip for NL=62."""
    expert_spec = NamedSharding(mesh, P('expert', None, None))
    rep_spec = NamedSharding(mesh, P())
    rng = np.random.default_rng(42)

    def rand_bf16(shape, spec):
        arr = (rng.standard_normal(size=shape) * 0.02).astype(np.float32)
        return jax.device_put(jnp.asarray(arr, dtype=DTYPE), spec)

    def rand_nvfp4(E, rows, cols, spec):
        """Random NVFP4 weight for E experts. Returns (packed, scales, scale2).
        scale2 is shape (E,) — per-expert global scale, placed on 'expert' axis so
        m2_moe.py's shard_map can see it aligned with the expert dim.
        """
        packed = rng.integers(0, 256, size=(E, rows, cols // 2), dtype=np.uint8)
        scales = rng.integers(0, 256, size=(E, rows, cols // GROUP_SIZE), dtype=np.uint8)
        scale2 = (rng.standard_normal(size=(E,)) * 0.01 + 0.01).astype(np.float32)
        packed_d = jax.device_put(jnp.asarray(packed), spec)
        scales_d = jax.device_put(jnp.asarray(scales), spec)
        # scale2 is (E,) — shard along expert axis
        scale2_spec = NamedSharding(spec.mesh, P('expert')) if 'expert' in spec.mesh.axis_names else rep_spec
        scale2_d = jax.device_put(jnp.asarray(scale2), scale2_spec)
        return (packed_d, scales_d, scale2_d)

    # Backbone (bf16, replicated)
    embed = rand_bf16((VOCAB, H), rep_spec)
    final_norm = rand_bf16((H,), rep_spec)
    lm_head = rand_bf16((VOCAB, H), rep_spec)

    # Per-layer
    all_w = []
    for i in range(nl):
        lw = {
            'ln1': rand_bf16((H,), rep_spec),
            'ln2': rand_bf16((H,), rep_spec),
            'attn': {
                'q': rand_bf16((NH * HEAD_DIM, H), rep_spec),
                'k': rand_bf16((NKV * HEAD_DIM, H), rep_spec),
                'v': rand_bf16((NKV * HEAD_DIM, H), rep_spec),
                'o': rand_bf16((H, NH * HEAD_DIM), rep_spec),
                'qn': rand_bf16((HEAD_DIM,), rep_spec),
                'kn': rand_bf16((HEAD_DIM,), rep_spec),
            },
            'rg': rand_bf16((NUM_EXPERTS, H), rep_spec),
            'rb': rand_bf16((NUM_EXPERTS,), rep_spec),
            # NVFP4 expert weights, sharded along expert axis
            'w1': rand_nvfp4(NUM_EXPERTS, MOE_INTER, H, expert_spec),
            'w3': rand_nvfp4(NUM_EXPERTS, MOE_INTER, H, expert_spec),
            'w2': rand_nvfp4(NUM_EXPERTS, H, MOE_INTER, expert_spec),
        }
        all_w.append(lw)
        print(f"  layer {i+1}/{nl}", file=sys.stderr, end='\r')
    print(file=sys.stderr)

    # KV cache (bf16, replicated — small at B=1,ctx=2048)
    k_cache = jax.device_put(
        jnp.zeros((nl, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), rep_spec)
    v_cache = jax.device_put(
        jnp.zeros((nl, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), rep_spec)

    cos, sin = precompute_rope(ROPE_THETA, ROTARY_DIM, max_ctx)
    cos = jax.device_put(jnp.array(cos), rep_spec)
    sin = jax.device_put(jnp.array(sin), rep_spec)

    return embed, final_norm, lm_head, all_w, k_cache, v_cache, cos, sin


def bench_batch(mesh, B, max_ctx, iters, warmup, nl):
    embed, final_norm, lm_head, all_w, k_cache, v_cache, cos, sin = random_weights(
        B, max_ctx, mesh, nl)

    # mesh is static, passed as closed-over. Wrap forward_step to avoid passing mesh as traced arg.
    def _forward(x, all_w, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        return forward_step(x, all_w, k_cache, v_cache, pos, cos, sin, final_norm, lm_head, mesh)
    forward_jit = jax.jit(_forward)

    tok = jnp.zeros((B,), dtype=jnp.int32)
    x = embed[tok]

    print(f"  warmup ({warmup} iters)...", file=sys.stderr)
    for it in range(warmup):
        tok, k_cache, v_cache = forward_jit(x, all_w, k_cache, v_cache,
                                            jnp.int32(it), cos, sin,
                                            final_norm, lm_head)
        x = embed[tok]
    jax.block_until_ready(tok)

    print(f"  measure ({iters} iters)...", file=sys.stderr)
    times = []
    for it in range(iters):
        t0 = time.perf_counter()
        tok, k_cache, v_cache = forward_jit(x, all_w, k_cache, v_cache,
                                            jnp.int32(warmup + it), cos, sin,
                                            final_norm, lm_head)
        jax.block_until_ready(tok)
        times.append((time.perf_counter() - t0) * 1000)
        x = embed[tok]
    times = np.array(times)
    return {
        'batch': B, 'ctx': max_ctx, 'nl': nl,
        'iters': iters,
        'ms_min': float(times.min()),
        'ms_mean': float(times.mean()),
        'ms_max': float(times.max()),
        'ms_p50': float(np.median(times)),
        'tok_per_s': float(1000.0 * B / times.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', default='1,8,32', help='comma-separated batch sizes')
    p.add_argument('--ctx', type=int, default=2048)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--nl', type=int, default=NL_FULL,
                   help='number of layers to bench (default 62=full)')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    batches = [int(b) for b in args.batch.split(',')]
    mesh = make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)
    print(f"layers: {args.nl} / {NL_FULL}", file=sys.stderr)

    results = []
    for B in batches:
        print(f"\n=== batch B={B} ctx={args.ctx} NL={args.nl} ===", file=sys.stderr)
        try:
            r = bench_batch(mesh, B, args.ctx, args.iters, args.warmup, args.nl)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            results.append({'batch': B, 'error': f'{type(e).__name__}: {e}'})
            continue
        per_layer_ms = r['ms_mean'] / args.nl
        full_model_ms = per_layer_ms * NL_FULL
        full_tok_s = 1000.0 * B / full_model_ms
        print(f"  B={B:4d} NL={args.nl} ms/step={r['ms_mean']:.2f} (min={r['ms_min']:.2f})  tok/s@NL={args.nl}={r['tok_per_s']:.1f}")
        print(f"         per-layer={per_layer_ms:.3f}ms  extrapolated full 62L: ms/step={full_model_ms:.2f}  tok/s={full_tok_s:.1f}")
        r['per_layer_ms'] = per_layer_ms
        r['extrap_full_ms'] = full_model_ms
        r['extrap_full_tok_per_s'] = full_tok_s
        results.append(r)

    out_path = args.out or f"/tmp/m2_synth_bench_{int(time.time())}.json"
    with open(out_path, 'w') as f:
        json.dump({
            'arch': 'MiniMax-M2.7-NVFP4 synth',
            'slice': 'v6e-8',
            'n_layers_full': NL_FULL, 'n_layers_run': args.nl,
            'n_experts': NUM_EXPERTS, 'top_k': TOP_K, 'moe_inter': MOE_INTER,
            'results': results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
