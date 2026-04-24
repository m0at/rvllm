"""MiniMax-M2.7-NVFP4 synthetic-weight throughput benchmark on TPU v6e-8.

Skips the 190K-tensor real model load. Allocates RANDOM BF16 weights at the
real architecture shapes, builds the same mesh + sharding, runs
`forward_step_m2` in a timed loop, reports ms/step and tok/s at B=1,8,32,128.

This measures hardware throughput ceiling for the MiniMax-M2 architecture
(GQA attention + 256-expert MoE top-8, 62 layers). Model quality (PPL, token
identity) is a separate concern that requires the real checkpoint — see
`m2_tpu_infer.py` for that path.

Usage:
    python3 m2_synth_bench.py [--batch 1,8,32,128] [--ctx 2048] \
                              [--iters 30] [--warmup 5]
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
NL = 62
VOCAB = 200064
RMS_EPS = 1e-6

DTYPE = jnp.bfloat16


def make_mesh_v6e8():
    devs = jax.devices()
    n = len(devs)
    print(f"  devices={n} kind={devs[0].device_kind if n else '?'}", file=sys.stderr)
    if n >= 8:
        mesh_devs = np.array(devs[:8]).reshape((8,))
        return Mesh(mesh_devs, ('expert',))
    mesh_devs = np.array(devs[:n]).reshape((n,))
    return Mesh(mesh_devs, ('expert',))


def precompute_rope(theta, rotary_dim, max_ctx):
    half = rotary_dim // 2
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
    # x: (B, H), returns (x_out, k_new, v_new)
    h = rms_norm(x, ln)
    q = (h @ w['q'].T).reshape(-1, NH, HEAD_DIM)
    k = (h @ w['k'].T).reshape(-1, NKV, HEAD_DIM)
    v = (h @ w['v'].T).reshape(-1, NKV, HEAD_DIM)
    q = head_rms(q, w['qn'])
    k = head_rms(k, w['kn'])
    q = rope_partial(q, cos[pos], sin[pos])
    k = rope_partial(k, cos[pos], sin[pos])
    # Cache write
    k_cache = jax.lax.dynamic_update_slice(k_cache, k[:, None], (0, pos, 0, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, v[:, None], (0, pos, 0, 0))
    # GQA attention
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


def moe_layer(x, router_w, router_bias, w1_stack, w2_stack, w3_stack):
    # x: (B, H), router_w: (E, H), biases (E,), experts (E, MOE_INTER, H) etc.
    B_ = x.shape[0]
    logits = jax.nn.sigmoid((x.astype(jnp.float32) @ router_w.T.astype(jnp.float32)))  # (B, E)
    biased = logits + router_bias.astype(jnp.float32)
    _, topk_idx = jax.lax.top_k(biased, TOP_K)  # (B, K)
    # gather unbiased scores
    gathered = jnp.take_along_axis(logits, topk_idx, axis=1)  # (B, K)
    weights = (gathered / (gathered.sum(axis=-1, keepdims=True) + 1e-9)).astype(x.dtype)  # (B, K)
    # For each token, for each of TOP_K, gather expert weights and compute
    # Reference (gather) path — slow but correct. Shard_map path is in m2_moe.py.
    w1_sel = w1_stack[topk_idx]  # (B, K, MOE_INTER, H)
    w3_sel = w3_stack[topk_idx]
    w2_sel = w2_stack[topk_idx]  # (B, K, H, MOE_INTER)
    gate = jnp.einsum('bki,bkmi->bkm', x, w1_sel)
    up = jnp.einsum('bki,bkmi->bkm', x, w3_sel)
    h = jax.nn.silu(gate) * up  # (B, K, MOE_INTER)
    out_per_k = jnp.einsum('bkm,bkhm->bkh', h, w2_sel)  # (B, K, H)
    return jnp.einsum('bk,bkh->bh', weights, out_per_k)


def layer_step(x, layer_w, k_cache, v_cache, pos, cos, sin, post_ln):
    att_out, k_cache, v_cache = attn_layer(x, layer_w['attn'], layer_w['ln1'],
                                           k_cache, v_cache, pos, cos, sin)
    x = x + att_out
    h = rms_norm(x, post_ln)
    moe_out = moe_layer(h, layer_w['rg'], layer_w['rb'],
                        layer_w['w1'], layer_w['w2'], layer_w['w3'])
    return x + moe_out, k_cache, v_cache


def forward_step(x, all_w, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
    # x: (B, H), k_cache/v_cache: (NL, B, max_ctx, NKV, HEAD_DIM)
    for i in range(NL):
        lw = all_w[i]
        x, k_i, v_i = layer_step(x, lw, k_cache[i], v_cache[i], pos, cos, sin, lw['ln2'])
        k_cache = k_cache.at[i].set(k_i)
        v_cache = v_cache.at[i].set(v_i)
    h = rms_norm(x, final_norm)
    logits = h @ lm_head.T
    tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return tok, k_cache, v_cache


def random_weights(key, B, max_ctx, mesh):
    """Allocate random bf16 weights at real shapes. Experts sharded across 'expert' axis."""
    expert_spec = NamedSharding(mesh, P('expert', None, None))
    rep_spec = NamedSharding(mesh, P())
    kv_spec = NamedSharding(mesh, P())

    def rand(shape, spec):
        return jax.device_put(
            jax.random.normal(key, shape, dtype=DTYPE) * 0.02, spec)

    # Backbone
    embed = rand((VOCAB, H), rep_spec)
    final_norm = rand((H,), rep_spec)
    lm_head = rand((VOCAB, H), rep_spec)

    # Per-layer
    all_w = []
    for i in range(NL):
        key, sub = jax.random.split(key)
        lw = {
            'ln1': rand((H,), rep_spec),
            'ln2': rand((H,), rep_spec),
            'attn': {
                'q': rand((NH * HEAD_DIM, H), rep_spec),
                'k': rand((NKV * HEAD_DIM, H), rep_spec),
                'v': rand((NKV * HEAD_DIM, H), rep_spec),
                'o': rand((H, NH * HEAD_DIM), rep_spec),
                'qn': rand((HEAD_DIM,), rep_spec),
                'kn': rand((HEAD_DIM,), rep_spec),
            },
            'rg': rand((NUM_EXPERTS, H), rep_spec),
            'rb': rand((NUM_EXPERTS,), rep_spec),
            'w1': rand((NUM_EXPERTS, MOE_INTER, H), expert_spec),
            'w2': rand((NUM_EXPERTS, H, MOE_INTER), expert_spec),
            'w3': rand((NUM_EXPERTS, MOE_INTER, H), expert_spec),
        }
        all_w.append(lw)
        print(f"  layer {i+1}/{NL}", file=sys.stderr, end='\r')
    print(file=sys.stderr)

    # KV cache
    k_cache = jax.device_put(
        jnp.zeros((NL, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), kv_spec)
    v_cache = jax.device_put(
        jnp.zeros((NL, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), kv_spec)

    # RoPE tables
    cos, sin = precompute_rope(ROPE_THETA, ROTARY_DIM, max_ctx)
    cos = jax.device_put(jnp.array(cos), rep_spec)
    sin = jax.device_put(jnp.array(sin), rep_spec)

    return embed, final_norm, lm_head, all_w, k_cache, v_cache, cos, sin


def bench_batch(mesh, B, max_ctx, iters, warmup):
    key = jax.random.PRNGKey(0)
    embed, final_norm, lm_head, all_w, k_cache, v_cache, cos, sin = random_weights(
        key, B, max_ctx, mesh)

    forward_jit = jax.jit(forward_step)

    # Dummy input token at pos=0
    tok = jnp.zeros((B,), dtype=jnp.int32)
    x = embed[tok]

    # Warmup
    print(f"  warmup ({warmup} iters)...", file=sys.stderr)
    for it in range(warmup):
        tok, k_cache, v_cache = forward_jit(x, all_w, k_cache, v_cache,
                                            jnp.int32(it), cos, sin,
                                            final_norm, lm_head)
        x = embed[tok]
    jax.block_until_ready(tok)

    # Timed
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
        'batch': B,
        'ctx': max_ctx,
        'iters': iters,
        'ms_min': float(times.min()),
        'ms_mean': float(times.mean()),
        'ms_max': float(times.max()),
        'ms_p50': float(np.median(times)),
        'tok_per_s': float(1000.0 * B / times.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', default='1,8,32,128',
                   help='comma-separated batch sizes')
    p.add_argument('--ctx', type=int, default=2048)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    batches = [int(b) for b in args.batch.split(',')]
    mesh = make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)

    results = []
    for B in batches:
        print(f"\n=== batch B={B} ctx={args.ctx} ===", file=sys.stderr)
        r = bench_batch(mesh, B, args.ctx, args.iters, args.warmup)
        print(f"  B={B:4d} ms/step={r['ms_mean']:.2f} (min={r['ms_min']:.2f})  tok/s={r['tok_per_s']:.1f}")
        results.append(r)

    out_path = args.out or f"/tmp/m2_synth_bench_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'arch': 'MiniMax-M2.7 synth-weights',
            'slice': 'v6e-8',
            'n_layers': NL,
            'n_experts': NUM_EXPERTS,
            'top_k': TOP_K,
            'moe_inter': MOE_INTER,
            'results': results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
