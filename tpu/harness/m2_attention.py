"""MiniMax-M2 attention primitives for JAX (GQA, partial RoPE, per-layer QK-norm).

Standalone module per M2_SWARM_SPEC.md agent 8. No imports from other agent files.
"""

import jax
import jax.numpy as jnp
import numpy as np


NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROTARY_DIM = 64
PARTIAL_ROTARY_FACTOR = 0.5
ROPE_THETA = 5_000_000.0
H = 3072
RMS_EPS = 1e-6
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 6


def precompute_rope_m2(theta, rotary_dim, max_ctx):
    """Compute RoPE cos/sin tables. Returns (cos, sin) of shape (max_ctx, rotary_dim//2)."""
    freqs = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    angles = np.outer(np.arange(max_ctx, dtype=np.float32), freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rope_partial_apply(x, cos, sin, rotary_dim):
    """Apply RoPE to first rotary_dim dims; leave remaining head_dim - rotary_dim pass-through.

    x shape: (..., head_dim). cos/sin broadcast-compatible with (..., rotary_dim//2).
    """
    half = rotary_dim // 2
    xr = x[..., :rotary_dim]
    xp = x[..., rotary_dim:]
    x0 = xr[..., :half]
    x1 = xr[..., half:]
    c = cos.astype(x.dtype)
    s = sin.astype(x.dtype)
    rotated = jnp.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
    return jnp.concatenate([rotated, xp], axis=-1)


def qk_rmsnorm(x, g, eps=RMS_EPS):
    """Per-head RMSNorm. x shape (..., head_dim); g shape (head_dim,)."""
    x32 = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps).astype(x.dtype)
    return (x * inv) * g


def gqa_attention_decode(q, k_cache, v_cache, pos, cache_len,
                         num_q_heads=NUM_Q_HEADS,
                         num_kv_heads=NUM_KV_HEADS,
                         head_dim=HEAD_DIM):
    """Softmax attention decode, GQA 48→8, full (non-sliding) causal context.

    q:          (B, num_q_heads, head_dim)
    k_cache:    (B, max_ctx, num_kv_heads, head_dim)
    v_cache:    (B, max_ctx, num_kv_heads, head_dim)
    pos:        (B,) current position (unused in mask — cache_len governs)
    cache_len:  (B,) valid K/V length in cache
    Returns:    (B, num_q_heads, head_dim)
    """
    del pos
    gqa = num_q_heads // num_kv_heads
    B, max_ctx, _, _ = k_cache.shape
    # Reshape Q into (B, num_kv_heads, gqa, head_dim) to group Q heads per KV head.
    q_g = q.reshape(B, num_kv_heads, gqa, head_dim)

    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
    # scores: (B, num_kv_heads, gqa, max_ctx)
    scores = jnp.einsum('bghd,btgd->bght',
                        q_g.astype(jnp.float32),
                        k_cache.astype(jnp.float32)) * scale

    t = jnp.arange(max_ctx)
    valid = t[None, :] < cache_len[:, None]  # (B, max_ctx)
    scores = jnp.where(valid[:, None, None, :], scores, jnp.float32(-1e30))

    probs = jax.nn.softmax(scores, axis=-1)
    # out: (B, num_kv_heads, gqa, head_dim)
    out = jnp.einsum('bght,btgd->bghd',
                     probs.astype(v_cache.dtype),
                     v_cache)
    return out.reshape(B, num_q_heads, head_dim).astype(q.dtype)


def _rms_norm(x, g, eps=RMS_EPS):
    x32 = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps).astype(x.dtype)
    return (x * inv) * g


def attention_layer(x, weights, ln_w, k_cache, v_cache, pos, cos, sin, cache_len):
    """Single M2 attention block decode step (one token per batch element).

    x:         (B, H) bf16
    weights:   dict with keys qw, kw, vw, ow, qn, kn (all bf16)
      qw: (NUM_Q_HEADS*HEAD_DIM, H)
      kw: (NUM_KV_HEADS*HEAD_DIM, H)
      vw: (NUM_KV_HEADS*HEAD_DIM, H)
      ow: (H, NUM_Q_HEADS*HEAD_DIM)
      qn: (HEAD_DIM,)  per-layer Q RMSNorm scale
      kn: (HEAD_DIM,)  per-layer K RMSNorm scale
    ln_w:      (H,) input layernorm weight
    k_cache:   (B, max_ctx, NUM_KV_HEADS, HEAD_DIM)
    v_cache:   (B, max_ctx, NUM_KV_HEADS, HEAD_DIM)
    pos:       scalar int32, current decode position
    cos, sin:  (max_ctx, ROTARY_DIM//2)
    cache_len: (B,) valid cache len including the new token being written this step

    Returns (x_out, k_cache_new, v_cache_new).
    """
    B = x.shape[0]
    h = _rms_norm(x, ln_w)

    # Projections
    q = h @ weights['qw'].T  # (B, NUM_Q_HEADS*HEAD_DIM)
    k = h @ weights['kw'].T  # (B, NUM_KV_HEADS*HEAD_DIM)
    v = h @ weights['vw'].T  # (B, NUM_KV_HEADS*HEAD_DIM)

    q = q.reshape(B, NUM_Q_HEADS, HEAD_DIM)
    k = k.reshape(B, NUM_KV_HEADS, HEAD_DIM)
    v = v.reshape(B, NUM_KV_HEADS, HEAD_DIM)

    # Per-layer QK-norm
    q = qk_rmsnorm(q, weights['qn'])
    k = qk_rmsnorm(k, weights['kn'])

    # Partial RoPE on first ROTARY_DIM dims only
    c = cos[pos][None, None, :]
    s = sin[pos][None, None, :]
    q = rope_partial_apply(q, c, s, ROTARY_DIM)
    k = rope_partial_apply(k, c, s, ROTARY_DIM)

    # Write K/V to cache at `pos`
    # k_cache: (B, max_ctx, NUM_KV_HEADS, HEAD_DIM); k: (B, NUM_KV_HEADS, HEAD_DIM)
    k_cache_new = jax.lax.dynamic_update_slice(
        k_cache, k[:, None, :, :].astype(k_cache.dtype), (0, pos, 0, 0))
    v_cache_new = jax.lax.dynamic_update_slice(
        v_cache, v[:, None, :, :].astype(v_cache.dtype), (0, pos, 0, 0))

    attn = gqa_attention_decode(
        q, k_cache_new, v_cache_new, pos=pos, cache_len=cache_len,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM)

    # Output projection
    attn_flat = attn.reshape(B, NUM_Q_HEADS * HEAD_DIM)
    x_out = attn_flat @ weights['ow'].T  # (B, H)
    return x_out.astype(x.dtype), k_cache_new, v_cache_new


if __name__ == '__main__':
    jax.config.update('jax_platforms', 'cpu')

    rng = np.random.default_rng(0)
    B = 2
    max_ctx = 32
    pos = 5

    def randb(*shape):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * 0.02,
                           dtype=jnp.bfloat16)

    weights = {
        'qw': randb(NUM_Q_HEADS * HEAD_DIM, H),
        'kw': randb(NUM_KV_HEADS * HEAD_DIM, H),
        'vw': randb(NUM_KV_HEADS * HEAD_DIM, H),
        'ow': randb(H, NUM_Q_HEADS * HEAD_DIM),
        'qn': jnp.ones((HEAD_DIM,), dtype=jnp.bfloat16),
        'kn': jnp.ones((HEAD_DIM,), dtype=jnp.bfloat16),
    }
    ln_w = jnp.ones((H,), dtype=jnp.bfloat16)

    x = randb(B, H)
    k_cache = jnp.zeros((B, max_ctx, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
    v_cache = jnp.zeros((B, max_ctx, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.bfloat16)

    cos_np, sin_np = precompute_rope_m2(ROPE_THETA, ROTARY_DIM, max_ctx)
    cos = jnp.asarray(cos_np)
    sin = jnp.asarray(sin_np)
    cache_len = jnp.asarray([pos + 1, pos + 1], dtype=jnp.int32)

    x_out, k_new, v_new = attention_layer(
        x, weights, ln_w, k_cache, v_cache,
        jnp.int32(pos), cos, sin, cache_len)

    assert x_out.shape == (B, H), f"bad x_out shape {x_out.shape}"
    assert k_new.shape == (B, max_ctx, NUM_KV_HEADS, HEAD_DIM)
    assert v_new.shape == (B, max_ctx, NUM_KV_HEADS, HEAD_DIM)
    assert jnp.all(jnp.isfinite(x_out.astype(jnp.float32))), "non-finite x_out"
    assert jnp.all(jnp.isfinite(k_new.astype(jnp.float32))), "non-finite k_cache"
    assert jnp.all(jnp.isfinite(v_new.astype(jnp.float32))), "non-finite v_cache"

    # Also exercise the bare decode helper
    q_t = jnp.asarray(rng.standard_normal((B, NUM_Q_HEADS, HEAD_DIM)).astype(np.float32),
                      dtype=jnp.bfloat16)
    o = gqa_attention_decode(q_t, k_new, v_new,
                             pos=jnp.int32(pos), cache_len=cache_len)
    assert o.shape == (B, NUM_Q_HEADS, HEAD_DIM)
    assert jnp.all(jnp.isfinite(o.astype(jnp.float32)))

    # Sanity: partial RoPE preserves tail dims
    head = jnp.ones((1, 1, HEAD_DIM), dtype=jnp.bfloat16)
    rotated = rope_partial_apply(head, cos[0][None, None, :], sin[0][None, None, :], ROTARY_DIM)
    tail_in = head[..., ROTARY_DIM:]
    tail_out = rotated[..., ROTARY_DIM:]
    assert jnp.array_equal(tail_in, tail_out), "partial RoPE corrupted tail"

    print("m2_attention smoke test OK")
    print(f"  x_out shape = {tuple(x_out.shape)} dtype = {x_out.dtype}")
    print(f"  k_cache shape = {tuple(k_new.shape)} dtype = {k_new.dtype}")
    print(f"  gqa_attention_decode shape = {tuple(o.shape)} dtype = {o.dtype}")
