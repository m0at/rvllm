"""MiniMax-M2.7-NVFP4 synthetic-weight throughput benchmark on TPU v6e-8.

Uses jax.lax.scan over layers (NOT Python unroll) so XLA streams one layer's
weights through HBM at a time. Peak argument memory stays bounded by one
layer's sharded weight ~= 340 MB per chip instead of 21+ GB for 62 unrolled
layers.

Architecture (MiniMax-M2.7-NVFP4):
- 62 layers, hidden=3072, 48 Q / 8 KV heads, head_dim=128, rotary_dim=64
- 256 experts top-8, MOE_INTER=1536, sigmoid+bias routing
- NVFP4 experts: packed uint8 + FP8 E4M3 scales + FP32 global scale
- bf16 attention + router + embed

Expert weights sharded 8-way across `('expert',)` mesh axis.

Usage:
    python3 m2_synth_bench.py --batch 1 --ctx 2048 --iters 10 --warmup 3 --nl 62
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_moe import moe_block_nvfp4
from m2_moe_dense import moe_block_dense_b1

# MoE dispatch strategy selector. Set via env var M2_MOE=dense|shardmap.
# "dense" = compute all 32 local experts per shard, mask+psum (Agent 12).
#          Better for small B. No all_to_all. Wastes compute on unselected experts.
# "shardmap" = sort + all_to_all dispatch (default). Better at large B.
_MOE_PATH = os.environ.get('M2_MOE', 'shardmap').lower()

# --- Arch constants ---
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
    """Per-head RMSNorm. h: (..., NH, HEAD_DIM). g may be (HEAD_DIM,) shared-per-head
    or (NH*HEAD_DIM,) per-head — reshape to match h's trailing two dims."""
    h32 = h.astype(jnp.float32)
    normed = h * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + RMS_EPS).astype(h.dtype)
    # If g is flat (NH*HEAD_DIM,), reshape to (NH, HEAD_DIM) so broadcast matches.
    if g.ndim == 1 and g.shape[0] == h.shape[-2] * h.shape[-1]:
        g = g.reshape(h.shape[-2], h.shape[-1])
    return normed * g


def rope_partial(x, cos, sin):
    half = ROTARY_DIM // 2
    xr, xp = x[..., :ROTARY_DIM], x[..., ROTARY_DIM:]
    x0, x1 = xr[..., :half], xr[..., half:]
    rotated = jnp.concatenate([
        x0 * cos.astype(x.dtype) - x1 * sin.astype(x.dtype),
        x0 * sin.astype(x.dtype) + x1 * cos.astype(x.dtype),
    ], axis=-1)
    return jnp.concatenate([rotated, xp], axis=-1)


def attn_layer(x, wq, wk, wv, wo, qn, kn, ln, k_cache, v_cache, pos, cos, sin):
    h = rms_norm(x, ln)
    q = (h @ wq.T).reshape(-1, NH, HEAD_DIM)
    k = (h @ wk.T).reshape(-1, NKV, HEAD_DIM)
    v = (h @ wv.T).reshape(-1, NKV, HEAD_DIM)
    q = head_rms(q, qn)
    k = head_rms(k, kn)
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
    out = ctx @ wo.T
    return out, k_cache, v_cache


# --- Scan over layers. One layer's weights slice out of the stacked pytree
# per scan step, so XLA streams them through HBM instead of holding all 62. ---

def forward_step(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head, mesh):
    """stacked: pytree with leading axis = NL for every leaf."""

    def layer_body(carry, layer_w):
        x, k_cache, v_cache, i = carry
        # Slice the per-layer k_cache/v_cache (axis 0 is NL)
        k_i = jax.lax.dynamic_index_in_dim(k_cache, i, axis=0, keepdims=False)
        v_i = jax.lax.dynamic_index_in_dim(v_cache, i, axis=0, keepdims=False)

        att_out, k_i_new, v_i_new = attn_layer(
            x,
            layer_w['attn_q'], layer_w['attn_k'], layer_w['attn_v'], layer_w['attn_o'],
            layer_w['attn_qn'], layer_w['attn_kn'], layer_w['ln1'],
            k_i, v_i, pos, cos, sin)
        x = x + att_out
        h = rms_norm(x, layer_w['ln2'])

        if _MOE_PATH == 'dense':
            moe_out = moe_block_dense_b1(
                h, layer_w['rg'], layer_w['rb'],
                (layer_w['w1_p'], layer_w['w1_s'], layer_w['w1_s2']),
                (layer_w['w2_p'], layer_w['w2_s'], layer_w['w2_s2']),
                (layer_w['w3_p'], layer_w['w3_s'], layer_w['w3_s2']),
                mesh,
            )
        else:
            moe_out = moe_block_nvfp4(
                h, layer_w['rg'], layer_w['rb'],
                (layer_w['w1_p'], layer_w['w1_s'], layer_w['w1_s2']),
                (layer_w['w2_p'], layer_w['w2_s'], layer_w['w2_s2']),
                (layer_w['w3_p'], layer_w['w3_s'], layer_w['w3_s2']),
                mesh,
            )
        x = x + moe_out

        k_cache = jax.lax.dynamic_update_index_in_dim(k_cache, k_i_new, i, axis=0)
        v_cache = jax.lax.dynamic_update_index_in_dim(v_cache, v_i_new, i, axis=0)
        return (x, k_cache, v_cache, i + 1), None

    (x, k_cache, v_cache, _), _ = jax.lax.scan(
        layer_body, (x, k_cache, v_cache, jnp.int32(0)), stacked)

    h = rms_norm(x, final_norm)
    logits = h @ lm_head.T
    tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return tok, k_cache, v_cache


def random_weights(B, max_ctx, mesh, nl):
    """Stack random NVFP4-packed weights across NL layers for lax.scan consumption.
    Expert-axis sharded per weight. Uses make_array_from_callback so each chip only
    materializes its LOCAL shard (not the full tensor), avoiding the one-chip
    staging OOM that device_put(np.ndarray, sharding) causes on big tensors."""

    expert_spec = NamedSharding(mesh, P(None, 'expert', None, None))
    scale2_spec = NamedSharding(mesh, P(None, 'expert'))
    rep_spec = NamedSharding(mesh, P())

    def _make(shape, sharding, fill_fn):
        """fill_fn(local_shape) -> np.ndarray of that shape."""
        def cb(idx):
            local_shape = []
            for i, s in enumerate(idx):
                if isinstance(s, slice):
                    start = s.start if s.start is not None else 0
                    stop = s.stop if s.stop is not None else shape[i]
                    local_shape.append(stop - start)
                else:
                    local_shape.append(1)
            return fill_fn(tuple(local_shape))
        return jax.make_array_from_callback(shape, sharding, cb)

    def rand_bf16_stack(shape, spec):
        def fill(local):
            return (np.random.default_rng().standard_normal(size=local) * 0.02).astype(
                np.float32).astype('bfloat16') if False else \
                (np.random.default_rng().standard_normal(size=local) * 0.02).astype(np.float32).view(np.uint32)
        # Actually just build bf16 via ml_dtypes
        import ml_dtypes
        def fill_bf(local):
            v = (np.random.default_rng().standard_normal(size=local) * 0.02).astype(np.float32)
            return v.astype(ml_dtypes.bfloat16)
        return _make(shape, spec, fill_bf)

    def rand_u8_stack(shape, spec):
        def fill(local):
            return np.random.default_rng().integers(0, 256, size=local, dtype=np.uint8)
        return _make(shape, spec, fill)

    def rand_f32_stack(shape, spec):
        def fill(local):
            return (np.random.default_rng().standard_normal(size=local) * 0.01 + 0.01).astype(np.float32)
        return _make(shape, spec, fill)

    # Backbone (bf16, replicated across all chips)
    embed = rand_bf16_stack((VOCAB, H), rep_spec)
    final_norm = rand_bf16_stack((H,), rep_spec)
    lm_head = rand_bf16_stack((VOCAB, H), rep_spec)

    # Per-layer stacked weights. Shape leading axis = nl.
    print(f"  stacking {nl} layers of weights...", file=sys.stderr)
    stacked = {
        'ln1':      rand_bf16_stack((nl, H), rep_spec),
        'ln2':      rand_bf16_stack((nl, H), rep_spec),
        'attn_q':   rand_bf16_stack((nl, NH * HEAD_DIM, H), rep_spec),
        'attn_k':   rand_bf16_stack((nl, NKV * HEAD_DIM, H), rep_spec),
        'attn_v':   rand_bf16_stack((nl, NKV * HEAD_DIM, H), rep_spec),
        'attn_o':   rand_bf16_stack((nl, H, NH * HEAD_DIM), rep_spec),
        'attn_qn':  rand_bf16_stack((nl, HEAD_DIM,), rep_spec),
        'attn_kn':  rand_bf16_stack((nl, HEAD_DIM,), rep_spec),
        'rg':       rand_bf16_stack((nl, NUM_EXPERTS, H), rep_spec),
        'rb':       rand_bf16_stack((nl, NUM_EXPERTS,), rep_spec),
        'w1_p':     rand_u8_stack((nl, NUM_EXPERTS, MOE_INTER, H // 2), expert_spec),
        'w1_s':     rand_u8_stack((nl, NUM_EXPERTS, MOE_INTER, H // GROUP_SIZE), expert_spec),
        'w1_s2':    rand_f32_stack((nl, NUM_EXPERTS), scale2_spec),
        'w3_p':     rand_u8_stack((nl, NUM_EXPERTS, MOE_INTER, H // 2), expert_spec),
        'w3_s':     rand_u8_stack((nl, NUM_EXPERTS, MOE_INTER, H // GROUP_SIZE), expert_spec),
        'w3_s2':    rand_f32_stack((nl, NUM_EXPERTS), scale2_spec),
        'w2_p':     rand_u8_stack((nl, NUM_EXPERTS, H, MOE_INTER // 2), expert_spec),
        'w2_s':     rand_u8_stack((nl, NUM_EXPERTS, H, MOE_INTER // GROUP_SIZE), expert_spec),
        'w2_s2':    rand_f32_stack((nl, NUM_EXPERTS), scale2_spec),
    }

    # KV caches (bf16, replicated; small at B=1 ctx=2048)
    k_cache = jax.device_put(
        jnp.zeros((nl, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), rep_spec)
    v_cache = jax.device_put(
        jnp.zeros((nl, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), rep_spec)

    cos, sin = precompute_rope(ROPE_THETA, ROTARY_DIM, max_ctx)
    cos = jax.device_put(jnp.array(cos), rep_spec)
    sin = jax.device_put(jnp.array(sin), rep_spec)

    return embed, final_norm, lm_head, stacked, k_cache, v_cache, cos, sin


def bench_batch(mesh, B, max_ctx, iters, warmup, nl):
    embed, final_norm, lm_head, stacked, k_cache, v_cache, cos, sin = random_weights(
        B, max_ctx, mesh, nl)

    def _forward(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        return forward_step(x, stacked, k_cache, v_cache, pos, cos, sin,
                          final_norm, lm_head, mesh)
    forward_jit = jax.jit(_forward)

    tok = jnp.zeros((B,), dtype=jnp.int32)
    x = embed[tok]

    print(f"  warmup ({warmup} iters)...", file=sys.stderr)
    for it in range(warmup):
        tok, k_cache, v_cache = forward_jit(x, stacked, k_cache, v_cache,
                                            jnp.int32(it), cos, sin,
                                            final_norm, lm_head)
        x = embed[tok]
    jax.block_until_ready(tok)

    print(f"  measure ({iters} iters)...", file=sys.stderr)
    times = []
    for it in range(iters):
        t0 = time.perf_counter()
        tok, k_cache, v_cache = forward_jit(x, stacked, k_cache, v_cache,
                                            jnp.int32(warmup + it), cos, sin,
                                            final_norm, lm_head)
        jax.block_until_ready(tok)
        times.append((time.perf_counter() - t0) * 1000)
        x = embed[tok]
    times = np.array(times)
    return {
        'batch': B, 'ctx': max_ctx, 'nl': nl, 'iters': iters,
        'ms_min': float(times.min()),
        'ms_mean': float(times.mean()),
        'ms_max': float(times.max()),
        'ms_p50': float(np.median(times)),
        'tok_per_s': float(1000.0 * B / times.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', default='1', help='comma-separated batch sizes')
    p.add_argument('--ctx', type=int, default=2048)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--nl', type=int, default=NL_FULL)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    batches = [int(b) for b in args.batch.split(',')]
    mesh = make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)
    print(f"layers: {args.nl} / {NL_FULL} (scan-based)", file=sys.stderr)

    results = []
    for B in batches:
        print(f"\n=== batch B={B} ctx={args.ctx} NL={args.nl} ===", file=sys.stderr)
        try:
            r = bench_batch(mesh, B, args.ctx, args.iters, args.warmup, args.nl)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            results.append({'batch': B, 'error': f'{type(e).__name__}: {e}'})
            continue
        print(f"  B={B:4d} NL={args.nl} ms/step={r['ms_mean']:.2f} (min={r['ms_min']:.2f})  tok/s={r['tok_per_s']:.1f}")
        results.append(r)

    out_path = args.out or f"/tmp/m2_synth_bench_scan_{int(time.time())}.json"
    with open(out_path, 'w') as f:
        json.dump({
            'arch': 'MiniMax-M2.7-NVFP4 synth (scan)',
            'slice': 'v6e-8',
            'n_layers_full': NL_FULL, 'n_layers_run': args.nl,
            'n_experts': NUM_EXPERTS, 'top_k': TOP_K, 'moe_inter': MOE_INTER,
            'results': results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
