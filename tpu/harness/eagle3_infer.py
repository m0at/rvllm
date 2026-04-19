#!/usr/bin/env python3
"""EAGLE-3 speculative decoding for Gemma 4 on TPU v6e.

Add-on to gemma4_tpu_infer.py. B=1 single-user ultra-low-latency inference.
Draft-verify cycles with a trained lightweight head for 2.5-3.5x speedup.

Usage:
    python3 eagle3_infer.py --model-dir /path/to/gemma-4-31B-it \
        --max-tokens 64 --prompt "Hello" --random-draft
"""
import argparse, json, os, struct, sys, time, functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes

# ── constants ──

H      = 5376
NH     = 32
INTER  = 21504
VOCAB  = 262144
NL     = 60
WINDOW = 1024
SOFTCAP_VAL = 30.0
EPS    = 1e-6

MAX_Q  = 16384
MAX_KV = 4096
MAX_O  = 16384
MAX_NORM_HD = 512

S_Q, S_KV, S_HD, S_KVH = 8192, 4096, 256, 16
S_GQA = NH // S_KVH
G_Q, G_KV, G_HD, G_KVH = 16384, 2048, 512, 4
G_GQA = NH // G_KVH

LAYER_IS_GLOBAL = np.array([1 if (i+1) % 6 == 0 else 0 for i in range(NL)], dtype=np.int32)

K_DRAFT = 5
FEAT_LAYERS = (2, 30, 59)
DRAFT_INTER = 10752

# ── base utilities (inlined to avoid import side effects) ──

def rms_norm(x, g):
    x32 = x.astype(jnp.float32)
    return (x * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + EPS).astype(x.dtype)) * g

def head_norm(h, g):
    h32 = h.astype(jnp.float32)
    return (h * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + EPS).astype(h.dtype)) * g

def head_norm_noscale(h):
    h32 = h.astype(jnp.float32)
    return (h * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + EPS).astype(h.dtype))

def rope(x, cos, sin, rot_dim):
    half = rot_dim // 2
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    x0, x1 = xr[..., :half], xr[..., half:]
    rotated = jnp.concatenate([
        x0 * cos.astype(x.dtype) - x1 * sin.astype(x.dtype),
        x0 * sin.astype(x.dtype) + x1 * cos.astype(x.dtype),
    ], axis=-1)
    return jnp.concatenate([rotated, xp], axis=-1)

def precompute_rope(theta, rot_dim, max_pos):
    half = rot_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    angles = np.outer(np.arange(max_pos, dtype=np.float32), freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

def int8_matmul(x, w_int8, scale):
    return (x @ w_int8.astype(jnp.bfloat16).T) * scale

def quantize_int8_perchannel(arr_bf16):
    w = arr_bf16.astype(np.float32)
    amax = np.abs(w).max(axis=-1, keepdims=True).clip(min=1e-10)
    scale = (amax / 127.0).astype(np.float32)
    w_int8 = np.round(w / scale).clip(-127, 127).astype(np.int8)
    return w_int8, scale.squeeze(-1).astype(ml_dtypes.bfloat16)

# ── single-step attention (B=1) ──

def _sliding_attn(q_flat, k_flat, v_flat, qn, kn, cos_s, sin_s, kc, vc, pos, ctx, max_ctx):
    q = head_norm(q_flat[:, :S_Q].reshape(1, NH, S_HD), qn[:S_HD])
    k = head_norm(k_flat[:, :S_KV].reshape(1, S_KVH, S_HD), kn[:S_HD])
    v = head_norm_noscale(v_flat[:, :S_KV].reshape(1, S_KVH, S_HD))
    c = cos_s[pos][None, None, :]
    s = sin_s[pos][None, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)
    kc = kc.at[pos].set(k.reshape(S_KV).astype(kc.dtype))
    vc = vc.at[pos].set(v.reshape(S_KV).astype(vc.dtype))
    k_ctx = kc[:max_ctx].reshape(max_ctx, S_KVH, S_HD)
    v_ctx = vc[:max_ctx].reshape(max_ctx, S_KVH, S_HD)
    q_g = q.reshape(1, S_KVH, S_GQA, S_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    t = jnp.arange(max_ctx)
    valid = (t < ctx) & (t >= jnp.maximum(0, pos - WINDOW + 1))
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(1, S_Q)
    return out, kc, vc

def _global_attn(q_flat, k_flat, v_flat, qn, kn, cos_g, sin_g, kc, vc, pos, ctx, max_ctx):
    k_raw = k_flat[:, :G_KV].reshape(1, G_KVH, G_HD)
    q = head_norm(q_flat[:, :G_Q].reshape(1, NH, G_HD), qn)
    k = head_norm(k_raw, kn)
    v = head_norm_noscale(k_raw)
    c = cos_g[pos][None, None, :]
    s = sin_g[pos][None, None, :]
    q = rope(q, c, s, 128)
    k = rope(k, c, s, 128)
    k_val = jnp.pad(k.reshape(G_KV), (0, MAX_KV - G_KV)).astype(kc.dtype)
    v_val = jnp.pad(v.reshape(G_KV), (0, MAX_KV - G_KV)).astype(vc.dtype)
    kc = kc.at[pos].set(k_val)
    vc = vc.at[pos].set(v_val)
    k_ctx = kc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    v_ctx = vc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    q_g = q.reshape(1, G_KVH, G_GQA, G_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    valid = jnp.arange(max_ctx) < ctx
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(1, G_Q)
    return out, kc, vc

# ── single-step scan body (B=1) ──

def one_layer(carry, xs):
    x, pos, ctx, cos_s, sin_s, cos_g, sin_g = carry
    max_ctx = xs['kc'].shape[0]
    residual = x
    h = rms_norm(x, xs['ln1'])
    q_flat = int8_matmul(h, xs['qw'], xs['qw_s'])
    k_flat = int8_matmul(h, xs['kw'], xs['kw_s'])
    v_flat = int8_matmul(h, xs['vw'], xs['vw_s'])
    ig = xs['ig']

    def do_sliding(args):
        q, k, v, qn, kn, kc, vc = args
        out, kc2, vc2 = _sliding_attn(q, k, v, qn, kn, cos_s, sin_s, kc, vc, pos, ctx, max_ctx)
        return jnp.pad(out, ((0,0),(0, MAX_Q - S_Q))), kc2, vc2

    def do_global(args):
        q, k, v, qn, kn, kc, vc = args
        out, kc2, vc2 = _global_attn(q, k, v, qn, kn, cos_g, sin_g, kc, vc, pos, ctx, max_ctx)
        return out, kc2, vc2

    attn_out, kc, vc = jax.lax.cond(
        ig, do_global, do_sliding,
        (q_flat, k_flat, v_flat, xs['qn'], xs['kn'], xs['kc'], xs['vc']))

    o_out = int8_matmul(attn_out, xs['ow'], xs['ow_s'])
    h = rms_norm(o_out, xs['ln2'])
    x = residual + h

    residual = x
    h = rms_norm(x, xs['ln3'])
    gate = int8_matmul(h, xs['gw'], xs['gw_s'])
    up = int8_matmul(h, xs['uw'], xs['uw_s'])
    h = jax.nn.gelu(gate, approximate=True) * up
    h = int8_matmul(h, xs['dw'], xs['dw_s'])
    h = rms_norm(h, xs['ln4'])
    x = (residual + h) * xs['ls']
    return (x, pos, ctx, cos_s, sin_s, cos_g, sin_g), {'kc': kc, 'vc': vc}

# ── multi-position attention for verify (T=K+1) ──

def _verify_sliding_attn(q_flat, k_flat, v_flat, qn, kn, cos_s, sin_s, kc, vc, start_pos, max_ctx):
    T = q_flat.shape[0]
    q = head_norm(q_flat[:, :S_Q].reshape(T, NH, S_HD), qn[:S_HD])
    k = head_norm(k_flat[:, :S_KV].reshape(T, S_KVH, S_HD), kn[:S_HD])
    v = head_norm_noscale(v_flat[:, :S_KV].reshape(T, S_KVH, S_HD))

    positions = start_pos + jnp.arange(T)
    c = cos_s[positions][:, None, :]
    s = sin_s[positions][:, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)

    kc = jax.lax.dynamic_update_slice(kc, k.reshape(T, S_KV).astype(kc.dtype), (start_pos, 0))
    vc = jax.lax.dynamic_update_slice(vc, v.reshape(T, S_KV).astype(vc.dtype), (start_pos, 0))

    k_ctx = kc[:max_ctx].reshape(max_ctx, S_KVH, S_HD)
    v_ctx = vc[:max_ctx].reshape(max_ctx, S_KVH, S_HD)
    q_g = q.reshape(T, S_KVH, S_GQA, S_HD)
    sc = jnp.einsum('tghd,cgd->tghc', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))

    ctx_pos = jnp.arange(max_ctx)
    valid = (ctx_pos[None, :] < (positions[:, None] + 1)) & \
            (ctx_pos[None, :] >= jnp.maximum(0, positions[:, None] + 1 - WINDOW))
    sc = jnp.where(valid[:, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('tghc,cgd->tghd', p, v_ctx).reshape(T, S_Q)
    return jnp.pad(out, ((0, 0), (0, MAX_Q - S_Q))), kc, vc

def _verify_global_attn(q_flat, k_flat, v_flat, qn, kn, cos_g, sin_g, kc, vc, start_pos, max_ctx):
    T = q_flat.shape[0]
    k_raw = k_flat[:, :G_KV].reshape(T, G_KVH, G_HD)
    q = head_norm(q_flat[:, :G_Q].reshape(T, NH, G_HD), qn)
    k = head_norm(k_raw, kn)
    v = head_norm_noscale(k_raw)

    positions = start_pos + jnp.arange(T)
    c = cos_g[positions][:, None, :]
    s = sin_g[positions][:, None, :]
    q = rope(q, c, s, 128)
    k = rope(k, c, s, 128)

    k_vals = jnp.pad(k.reshape(T, G_KV), ((0, 0), (0, MAX_KV - G_KV))).astype(kc.dtype)
    v_vals = jnp.pad(v.reshape(T, G_KV), ((0, 0), (0, MAX_KV - G_KV))).astype(vc.dtype)
    kc = jax.lax.dynamic_update_slice(kc, k_vals, (start_pos, 0))
    vc = jax.lax.dynamic_update_slice(vc, v_vals, (start_pos, 0))

    k_ctx = kc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    v_ctx = vc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    q_g = q.reshape(T, G_KVH, G_GQA, G_HD)
    sc = jnp.einsum('tghd,cgd->tghc', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))

    ctx_pos = jnp.arange(max_ctx)
    valid = ctx_pos[None, :] < (positions[:, None] + 1)
    sc = jnp.where(valid[:, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('tghc,cgd->tghd', p, v_ctx).reshape(T, G_Q)
    return out, kc, vc

# ── verify scan body (T positions through one layer) ──

def verify_one_layer(carry, xs):
    x, start_pos, ctx, cos_s, sin_s, cos_g, sin_g = carry
    max_ctx = xs['kc'].shape[0]
    residual = x
    h = rms_norm(x, xs['ln1'])
    q_flat = int8_matmul(h, xs['qw'], xs['qw_s'])
    k_flat = int8_matmul(h, xs['kw'], xs['kw_s'])
    v_flat = int8_matmul(h, xs['vw'], xs['vw_s'])
    ig = xs['ig']

    def do_sliding(args):
        q, k, v, qn, kn, kc, vc = args
        return _verify_sliding_attn(q, k, v, qn, kn, cos_s, sin_s, kc, vc, start_pos, max_ctx)

    def do_global(args):
        q, k, v, qn, kn, kc, vc = args
        return _verify_global_attn(q, k, v, qn, kn, cos_g, sin_g, kc, vc, start_pos, max_ctx)

    attn_out, kc, vc = jax.lax.cond(
        ig, do_global, do_sliding,
        (q_flat, k_flat, v_flat, xs['qn'], xs['kn'], xs['kc'], xs['vc']))

    o_out = int8_matmul(attn_out, xs['ow'], xs['ow_s'])
    h = rms_norm(o_out, xs['ln2'])
    x = residual + h

    residual = x
    h = rms_norm(x, xs['ln3'])
    gate = int8_matmul(h, xs['gw'], xs['gw_s'])
    up = int8_matmul(h, xs['uw'], xs['uw_s'])
    h = jax.nn.gelu(gate, approximate=True) * up
    h = int8_matmul(h, xs['dw'], xs['dw_s'])
    h = rms_norm(h, xs['ln4'])
    x = (residual + h) * xs['ls']
    return (x, start_pos, ctx, cos_s, sin_s, cos_g, sin_g), {'kc': kc, 'vc': vc}

# ── scan body with branchless feature capture ──

def maybe_capture_feature(li, target_li, x, prev):
    return jax.lax.cond(li == target_li, lambda _: x, lambda _: prev, operand=None)

def one_layer_feats(carry, xs):
    x, pos, ctx, cos_s, sin_s, cos_g, sin_g, fl, fm, fh = carry
    max_ctx = xs['kc'].shape[0]
    residual = x
    h = rms_norm(x, xs['ln1'])
    q_flat = int8_matmul(h, xs['qw'], xs['qw_s'])
    k_flat = int8_matmul(h, xs['kw'], xs['kw_s'])
    v_flat = int8_matmul(h, xs['vw'], xs['vw_s'])
    ig = xs['ig']

    def do_sliding(args):
        q, k, v, qn, kn, kc, vc = args
        out, kc2, vc2 = _sliding_attn(q, k, v, qn, kn, cos_s, sin_s, kc, vc, pos, ctx, max_ctx)
        return jnp.pad(out, ((0,0),(0, MAX_Q - S_Q))), kc2, vc2

    def do_global(args):
        q, k, v, qn, kn, kc, vc = args
        out, kc2, vc2 = _global_attn(q, k, v, qn, kn, cos_g, sin_g, kc, vc, pos, ctx, max_ctx)
        return out, kc2, vc2

    attn_out, kc, vc = jax.lax.cond(
        ig, do_global, do_sliding,
        (q_flat, k_flat, v_flat, xs['qn'], xs['kn'], xs['kc'], xs['vc']))

    o_out = int8_matmul(attn_out, xs['ow'], xs['ow_s'])
    h = rms_norm(o_out, xs['ln2'])
    x = residual + h

    residual = x
    h = rms_norm(x, xs['ln3'])
    gate = int8_matmul(h, xs['gw'], xs['gw_s'])
    up = int8_matmul(h, xs['uw'], xs['uw_s'])
    h = jax.nn.gelu(gate, approximate=True) * up
    h = int8_matmul(h, xs['dw'], xs['dw_s'])
    h = rms_norm(h, xs['ln4'])
    x = (residual + h) * xs['ls']

    li = xs['li']
    fl = maybe_capture_feature(li, FEAT_LAYERS[0], x, fl)
    fm = maybe_capture_feature(li, FEAT_LAYERS[1], x, fm)
    fh = maybe_capture_feature(li, FEAT_LAYERS[2], x, fh)

    return (x, pos, ctx, cos_s, sin_s, cos_g, sin_g, fl, fm, fh), {'kc': kc, 'vc': vc}

def verify_one_layer_feats(carry, xs):
    x, start_pos, ctx, cos_s, sin_s, cos_g, sin_g, fl, fm, fh = carry
    max_ctx = xs['kc'].shape[0]
    residual = x
    h = rms_norm(x, xs['ln1'])
    q_flat = int8_matmul(h, xs['qw'], xs['qw_s'])
    k_flat = int8_matmul(h, xs['kw'], xs['kw_s'])
    v_flat = int8_matmul(h, xs['vw'], xs['vw_s'])
    ig = xs['ig']

    def do_sliding(args):
        q, k, v, qn, kn, kc, vc = args
        return _verify_sliding_attn(q, k, v, qn, kn, cos_s, sin_s, kc, vc, start_pos, max_ctx)

    def do_global(args):
        q, k, v, qn, kn, kc, vc = args
        return _verify_global_attn(q, k, v, qn, kn, cos_g, sin_g, kc, vc, start_pos, max_ctx)

    attn_out, kc, vc = jax.lax.cond(
        ig, do_global, do_sliding,
        (q_flat, k_flat, v_flat, xs['qn'], xs['kn'], xs['kc'], xs['vc']))

    o_out = int8_matmul(attn_out, xs['ow'], xs['ow_s'])
    h = rms_norm(o_out, xs['ln2'])
    x = residual + h

    residual = x
    h = rms_norm(x, xs['ln3'])
    gate = int8_matmul(h, xs['gw'], xs['gw_s'])
    up = int8_matmul(h, xs['uw'], xs['uw_s'])
    h = jax.nn.gelu(gate, approximate=True) * up
    h = int8_matmul(h, xs['dw'], xs['dw_s'])
    h = rms_norm(h, xs['ln4'])
    x = (residual + h) * xs['ls']

    li = xs['li']
    fl = maybe_capture_feature(li, FEAT_LAYERS[0], x, fl)
    fm = maybe_capture_feature(li, FEAT_LAYERS[1], x, fm)
    fh = maybe_capture_feature(li, FEAT_LAYERS[2], x, fh)

    return (x, start_pos, ctx, cos_s, sin_s, cos_g, sin_g, fl, fm, fh), {'kc': kc, 'vc': vc}

# ── helpers ──

def slice_weights(ws, start, end):
    return jax.tree.map(lambda x: x[start:end], ws)

# ── target forward with feature capture (single step, B=1) ──

def forward_with_features(token_id, pos, ctx, embed, final_norm, weights, caches,
                          cos_s, sin_s, cos_g, sin_g):
    x = embed[token_id].reshape(1, H) * jnp.sqrt(jnp.float32(H))
    fl = jnp.zeros_like(x)
    fm = jnp.zeros_like(x)
    fh = jnp.zeros_like(x)
    init = (x, pos, ctx, cos_s, sin_s, cos_g, sin_g, fl, fm, fh)
    layer_xs = {**weights, 'kc': caches['kc'], 'vc': caches['vc'],
                'li': jnp.arange(NL, dtype=jnp.int32)}

    final, scan_out = jax.lax.scan(one_layer_feats, init, layer_xs)
    x = final[0]
    feat_low, feat_mid, feat_high = final[7], final[8], final[9]

    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    new_caches = {'kc': scan_out['kc'], 'vc': scan_out['vc']}
    return next_tok, new_caches, feat_low, feat_mid, feat_high

# ── target forward (single step, no features, for prefill) ──

def forward_step(token_id, pos, ctx, embed, final_norm, weights, caches,
                 cos_s, sin_s, cos_g, sin_g):
    x = embed[token_id].reshape(1, H) * jnp.sqrt(jnp.float32(H))
    init = (x, pos, ctx, cos_s, sin_s, cos_g, sin_g)
    layer_xs = {**weights, 'kc': caches['kc'], 'vc': caches['vc']}
    final, scan_out = jax.lax.scan(one_layer, init, layer_xs)
    x = rms_norm(final[0], final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    new_caches = {'kc': scan_out['kc'], 'vc': scan_out['vc']}
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), new_caches

# ── verify forward (T=K+1 positions, captures features) ──

def verify_forward(verify_tokens, start_pos, embed, final_norm, weights, caches,
                   cos_s, sin_s, cos_g, sin_g):
    T = K_DRAFT + 1
    x = embed[verify_tokens] * jnp.sqrt(jnp.float32(H))
    ctx = start_pos + T
    fl = jnp.zeros_like(x)
    fm = jnp.zeros_like(x)
    fh = jnp.zeros_like(x)
    init = (x, start_pos, ctx, cos_s, sin_s, cos_g, sin_g, fl, fm, fh)
    layer_xs = {**weights, 'kc': caches['kc'], 'vc': caches['vc'],
                'li': jnp.arange(NL, dtype=jnp.int32)}

    final, scan_out = jax.lax.scan(verify_one_layer_feats, init, layer_xs)
    x = final[0]
    feat_low, feat_mid, feat_high = final[7], final[8], final[9]

    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)

    new_caches = {'kc': scan_out['kc'], 'vc': scan_out['vc']}
    return logits, new_caches, feat_low, feat_mid, feat_high

# ── draft head ──

def fuse_features(feat_low, feat_mid, feat_high, dw):
    concat = jnp.concatenate([feat_low.reshape(1, H), feat_mid.reshape(1, H),
                               feat_high.reshape(1, H)], axis=-1)
    return concat @ dw['fc_fuse_w'].T + dw['fc_fuse_b']

def draft_step(e, g, dw, draft_kv_k, draft_kv_v, draft_pos, embed, cos_s, sin_s):
    u = jnp.concatenate([e, g], axis=-1)
    u = u @ dw['fc_in_w'].T + dw['fc_in_b']

    residual = u
    h = rms_norm(u, dw['d_ln1'])

    q = h @ dw['d_qw'].T
    k = h @ dw['d_kw'].T
    v = h @ dw['d_vw'].T

    q = head_norm(q.reshape(1, NH, S_HD), dw['d_qn'])
    k = head_norm(k.reshape(1, S_KVH, S_HD), dw['d_kn'])
    v = head_norm_noscale(v.reshape(1, S_KVH, S_HD))

    c = cos_s[draft_pos][None, None, :]
    s = sin_s[draft_pos][None, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)

    draft_kv_k = draft_kv_k.at[draft_pos].set(k.reshape(S_KV))
    draft_kv_v = draft_kv_v.at[draft_pos].set(v.reshape(S_KV))

    k_ctx = draft_kv_k.reshape(K_DRAFT, S_KVH, S_HD)
    v_ctx = draft_kv_v.reshape(K_DRAFT, S_KVH, S_HD)
    q_g = q.reshape(1, S_KVH, S_GQA, S_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    t = jnp.arange(K_DRAFT)
    valid = t < (draft_pos + 1)
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    attn_out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(1, S_Q)

    x = residual + attn_out @ dw['d_ow'].T

    residual = x
    h = rms_norm(x, dw['d_ln2'])
    gate = h @ dw['d_gw'].T
    up = h @ dw['d_uw'].T
    h = jax.nn.gelu(gate, approximate=True) * up
    x = residual + h @ dw['d_dw'].T

    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    return x, logits, next_tok, draft_kv_k, draft_kv_v

def draft_chain(last_token, g_last, dw, embed, cos_s, sin_s):
    dk = jnp.zeros((K_DRAFT, S_KV), dtype=jnp.bfloat16)
    dv = jnp.zeros((K_DRAFT, S_KV), dtype=jnp.bfloat16)

    def step_fn(carry, k):
        e, g, dk, dv = carry
        x, logits, next_tok, dk, dv = draft_step(e, g, dw, dk, dv, k, embed, cos_s, sin_s)
        e_next = embed[next_tok[0]].reshape(1, H) * jnp.sqrt(jnp.float32(H))
        return (e_next, x, dk, dv), (next_tok, logits)

    e_init = embed[last_token].reshape(1, H) * jnp.sqrt(jnp.float32(H))
    init = (e_init, g_last.reshape(1, H), dk, dv)
    _, (draft_tokens, draft_logits) = jax.lax.scan(step_fn, init, jnp.arange(K_DRAFT))
    return draft_tokens, draft_logits

# ── draft weight init / load ──

def init_random_draft(mesh, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 20)
    i = [0]
    def nk():
        k = keys[i[0] % len(keys)]
        i[0] += 1
        return k

    dw = {
        'fc_fuse_w': jax.random.normal(nk(), (H, 3 * H), dtype=jnp.bfloat16) * 0.02,
        'fc_fuse_b': jnp.zeros(H, dtype=jnp.bfloat16),
        'fc_in_w':   jax.random.normal(nk(), (H, 2 * H), dtype=jnp.bfloat16) * 0.02,
        'fc_in_b':   jnp.zeros(H, dtype=jnp.bfloat16),
        'd_ln1':     jnp.ones(H, dtype=jnp.bfloat16),
        'd_qw':      jax.random.normal(nk(), (S_Q, H), dtype=jnp.bfloat16) * 0.02,
        'd_kw':      jax.random.normal(nk(), (S_KV, H), dtype=jnp.bfloat16) * 0.02,
        'd_vw':      jax.random.normal(nk(), (S_KV, H), dtype=jnp.bfloat16) * 0.02,
        'd_ow':      jax.random.normal(nk(), (H, S_Q), dtype=jnp.bfloat16) * 0.02,
        'd_qn':      jnp.ones(S_HD, dtype=jnp.bfloat16),
        'd_kn':      jnp.ones(S_HD, dtype=jnp.bfloat16),
        'd_ln2':     jnp.ones(H, dtype=jnp.bfloat16),
        'd_gw':      jax.random.normal(nk(), (DRAFT_INTER, H), dtype=jnp.bfloat16) * 0.02,
        'd_uw':      jax.random.normal(nk(), (DRAFT_INTER, H), dtype=jnp.bfloat16) * 0.02,
        'd_dw':      jax.random.normal(nk(), (H, DRAFT_INTER), dtype=jnp.bfloat16) * 0.02,
    }
    rep = NamedSharding(mesh, P())
    return jax.tree.map(lambda x: jax.device_put(x, rep), dw)

def load_draft_weights(draft_dir, mesh):
    print(f"loading draft head from {draft_dir}", file=sys.stderr)
    from gemma4_tpu_infer import read_safetensors, to_np_bf16
    all_t = read_safetensors(os.path.join(draft_dir, 'draft_head.safetensors'))

    def get_bf16(name):
        return to_np_bf16(all_t[name])

    dw = {
        'fc_fuse_w': get_bf16('fc_fuse.weight'),
        'fc_fuse_b': get_bf16('fc_fuse.bias'),
        'fc_in_w':   get_bf16('fc_in.weight'),
        'fc_in_b':   get_bf16('fc_in.bias'),
        'd_ln1':     get_bf16('draft_layer.input_layernorm.weight'),
        'd_qw':      get_bf16('draft_layer.self_attn.q_proj.weight'),
        'd_kw':      get_bf16('draft_layer.self_attn.k_proj.weight'),
        'd_vw':      get_bf16('draft_layer.self_attn.v_proj.weight'),
        'd_ow':      get_bf16('draft_layer.self_attn.o_proj.weight'),
        'd_qn':      get_bf16('draft_layer.self_attn.q_norm.weight'),
        'd_kn':      get_bf16('draft_layer.self_attn.k_norm.weight'),
        'd_ln2':     get_bf16('draft_layer.pre_feedforward_layernorm.weight'),
        'd_gw':      get_bf16('draft_layer.mlp.gate_proj.weight'),
        'd_uw':      get_bf16('draft_layer.mlp.up_proj.weight'),
        'd_dw':      get_bf16('draft_layer.mlp.down_proj.weight'),
    }
    rep = NamedSharding(mesh, P())
    dw = {k: jax.device_put(jnp.array(v), rep) for k, v in dw.items()}
    print(f"  draft head loaded: {sum(v.size for v in dw.values())} params", file=sys.stderr)
    return dw

# ── target weight loading (reuse from base) ──

def read_safetensors(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        data = np.memmap(path, dtype=np.uint8, mode='r', offset=data_start)
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        shape = tuple(info['shape'])
        dtype_str = info['dtype']
        start, end = info['data_offsets']
        raw = np.array(data[start:end])
        if dtype_str in ('BF16', 'bf16', 'bfloat16'):
            tensors[name] = raw.view(np.uint16).reshape(shape)
        elif dtype_str in ('F16', 'f16', 'float16'):
            tensors[name] = raw.view(np.float16).reshape(shape)
        elif dtype_str in ('F32', 'f32', 'float32'):
            tensors[name] = raw.view(np.float32).reshape(shape)
    return tensors

def to_np_bf16(arr):
    if arr.dtype == np.uint16:
        return arr.view(ml_dtypes.bfloat16)
    if arr.dtype == np.float16:
        return arr.astype(np.float32).astype(ml_dtypes.bfloat16)
    if arr.dtype == np.float32:
        return arr.astype(ml_dtypes.bfloat16)
    if arr.dtype == ml_dtypes.bfloat16:
        return arr
    raise ValueError(f"unsupported dtype {arr.dtype}")

def pad_to(arr, target_shape):
    pads = [(0, t - s) for s, t in zip(arr.shape, target_shape)]
    return np.pad(arr, pads)

def make_mesh():
    devs = jax.devices()
    n = min(len(devs), 4)
    return Mesh(np.array(devs[:n]), ('tp',))

def load_tokenizer(model_dir):
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tok_path)
    return None

def load_model(model_dir, mesh, max_ctx):
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        weight_map = index['weight_map']
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ['model.safetensors']
        weight_map = None

    prefix = 'model'
    if weight_map:
        for k in weight_map:
            if k.startswith('model.language_model.'):
                prefix = 'model.language_model'
                break

    print(f"loading from {model_dir}, prefix={prefix}", file=sys.stderr)
    all_t = {}
    for sn in shard_names:
        print(f"  reading {sn}...", file=sys.stderr)
        all_t.update(read_safetensors(os.path.join(model_dir, sn)))
    print(f"  {len(all_t)} tensors", file=sys.stderr)

    def get(name): return all_t[name]
    def has(name): return name in all_t

    def put(arr, spec):
        return jax.device_put(to_np_bf16(arr), NamedSharding(mesh, spec))

    embed = put(get(f'{prefix}.embed_tokens.weight'), P(None, None))
    final_norm = put(get(f'{prefix}.norm.weight'), P(None))

    print("  stacking 60 layers (padded, int8 quantized)...", file=sys.stderr)
    matmul_keys = ['qw','kw','vw','ow','gw','uw','dw']
    bf16_keys = ['qn','kn','ln1','ln2','ln3','ln4','ls']
    stacked_i8 = {k: [] for k in matmul_keys}
    stacked_sc = {k+'_s': [] for k in matmul_keys}
    stacked_bf = {k: [] for k in bf16_keys}
    stacked_ig = []

    for i in range(NL):
        lp = f'{prefix}.layers.{i}'
        is_global = LAYER_IS_GLOBAL[i]
        qw = to_np_bf16(get(f'{lp}.self_attn.q_proj.weight'))
        kw = to_np_bf16(get(f'{lp}.self_attn.k_proj.weight'))
        ow = to_np_bf16(get(f'{lp}.self_attn.o_proj.weight'))
        if has(f'{lp}.self_attn.v_proj.weight'):
            vw = to_np_bf16(get(f'{lp}.self_attn.v_proj.weight'))
        else:
            vw = np.zeros((MAX_KV, H), dtype=ml_dtypes.bfloat16)
        gw = to_np_bf16(get(f'{lp}.mlp.gate_proj.weight'))
        uw = to_np_bf16(get(f'{lp}.mlp.up_proj.weight'))
        dw = to_np_bf16(get(f'{lp}.mlp.down_proj.weight'))

        raw = {'qw': pad_to(qw, (MAX_Q, H)), 'kw': pad_to(kw, (MAX_KV, H)),
               'vw': pad_to(vw, (MAX_KV, H)), 'ow': pad_to(ow, (H, MAX_O)),
               'gw': gw, 'uw': uw, 'dw': dw}
        for k in matmul_keys:
            w_i8, sc = quantize_int8_perchannel(raw[k])
            stacked_i8[k].append(w_i8)
            stacked_sc[k+'_s'].append(sc)

        stacked_bf['qn'].append(pad_to(to_np_bf16(get(f'{lp}.self_attn.q_norm.weight')), (MAX_NORM_HD,)))
        stacked_bf['kn'].append(pad_to(to_np_bf16(get(f'{lp}.self_attn.k_norm.weight')), (MAX_NORM_HD,)))
        stacked_bf['ln1'].append(to_np_bf16(get(f'{lp}.input_layernorm.weight')))
        stacked_bf['ln2'].append(to_np_bf16(get(f'{lp}.post_attention_layernorm.weight')))
        stacked_bf['ln3'].append(to_np_bf16(get(f'{lp}.pre_feedforward_layernorm.weight')))
        stacked_bf['ln4'].append(to_np_bf16(get(f'{lp}.post_feedforward_layernorm.weight')))
        stacked_bf['ls'].append(to_np_bf16(get(f'{lp}.layer_scalar')) if has(f'{lp}.layer_scalar')
                                else np.array([1.0], dtype=ml_dtypes.bfloat16))
        stacked_ig.append(np.array(is_global, dtype=np.int32))
        if i % 15 == 0:
            print(f"    layer {i}", file=sys.stderr)

    i8_sharding = {
        'qw': P(None, 'tp', None), 'kw': P(None, 'tp', None), 'vw': P(None, 'tp', None),
        'ow': P(None, None, 'tp'), 'gw': P(None, 'tp', None), 'uw': P(None, 'tp', None),
        'dw': P(None, None, 'tp'),
    }
    sc_sharding = {
        'qw_s': P(None, 'tp'), 'kw_s': P(None, 'tp'), 'vw_s': P(None, 'tp'),
        'ow_s': P(None, None), 'gw_s': P(None, 'tp'), 'uw_s': P(None, 'tp'),
        'dw_s': P(None, None),
    }

    def put_i8(arr, spec):
        return jax.device_put(jnp.array(arr, dtype=jnp.int8), NamedSharding(mesh, spec))

    weights = {}
    for k in matmul_keys:
        arr = np.stack(stacked_i8[k])
        weights[k] = put_i8(arr, i8_sharding[k])
        print(f"    {k}: {arr.shape} int8", file=sys.stderr)
        sc_arr = np.stack(stacked_sc[k+'_s'])
        weights[k+'_s'] = put(sc_arr, sc_sharding[k+'_s'])
    for k in bf16_keys:
        arr = np.stack(stacked_bf[k])
        weights[k] = put(arr, P(None, None))
    weights['ig'] = jax.device_put(jnp.array(np.array(stacked_ig)), NamedSharding(mesh, P(None)))

    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    caches = {
        'kc': jax.device_put(jnp.zeros((NL, max_ctx, MAX_KV), dtype=jnp.bfloat16), kv_sh),
        'vc': jax.device_put(jnp.zeros((NL, max_ctx, MAX_KV), dtype=jnp.bfloat16), kv_sh),
    }

    del all_t, stacked_i8, stacked_sc, stacked_bf
    print("  done loading", file=sys.stderr)
    return embed, final_norm, weights, caches

# ── fused on-device cycle ──

def verify_forward_fused(verify_tokens, start_pos, embed, final_norm, weights, caches,
                         cos_s, sin_s, cos_g, sin_g):
    T = K_DRAFT + 1
    x = embed[verify_tokens] * jnp.sqrt(jnp.float32(H))
    ctx = start_pos + T
    fl = jnp.zeros_like(x)
    fm = jnp.zeros_like(x)
    fh = jnp.zeros_like(x)
    init = (x, start_pos, ctx, cos_s, sin_s, cos_g, sin_g, fl, fm, fh)
    layer_xs = {**weights, 'kc': caches['kc'], 'vc': caches['vc'],
                'li': jnp.arange(NL, dtype=jnp.int32)}
    final, scan_out = jax.lax.scan(verify_one_layer_feats, init, layer_xs)
    x = final[0]
    feat_low, feat_mid, feat_high = final[7], final[8], final[9]
    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    target_argmax = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    new_caches = {'kc': scan_out['kc'], 'vc': scan_out['vc']}
    return target_argmax, new_caches, feat_low, feat_mid, feat_high

def on_device_accept(draft_flat, target_argmax):
    matches = (draft_flat == target_argmax[:K_DRAFT])
    all_match = jnp.cumprod(matches.astype(jnp.int32))
    accepted = jnp.sum(all_match).astype(jnp.int32)
    correction = target_argmax[accepted]
    return accepted, correction

def make_eagle3_fused_loop(max_tokens):
    def eagle3_loop(last_token_init, g_last_init, pos_init, caches,
                    embed, final_norm, weights, draft_weights,
                    cos_s, sin_s, cos_g, sin_g):
        # Keep a small overrun margin so each cycle can write a full K+1 block
        # without a branchy per-token loop near the decode boundary.
        generated = jnp.zeros(max_tokens + K_DRAFT + 1, dtype=jnp.int32)
        generated = generated.at[0].set(last_token_init)
        EOS = jnp.array([1, 2, 107], dtype=jnp.int32)

        init_state = (jnp.int32(1), last_token_init, g_last_init,
                      jnp.int32(pos_init), caches, generated,
                      jnp.int32(0), jnp.int32(0))

        def cond_fn(state):
            gen_idx, last_token, *_ = state
            return (gen_idx < max_tokens) & ~jnp.any(last_token == EOS)

        def body_fn(state):
            gen_idx, last_token, g_last, pos, caches, generated, cycles, acc_total = state

            draft_tokens, _ = draft_chain(last_token, g_last, draft_weights,
                                          embed, cos_s, sin_s)
            draft_flat = draft_tokens[:, 0]

            verify_input = jnp.concatenate([jnp.array([last_token]), draft_flat])

            target_argmax, caches, fl_v, fm_v, fh_v = verify_forward_fused(
                verify_input, pos, embed, final_norm, weights, caches,
                cos_s, sin_s, cos_g, sin_g)

            accepted, correction = on_device_accept(draft_flat, target_argmax)
            n_accepted = accepted + 1

            out_tokens = jnp.zeros(K_DRAFT + 1, dtype=jnp.int32)
            out_tokens = out_tokens.at[:K_DRAFT].set(draft_flat)
            out_tokens = out_tokens.at[accepted].set(correction)
            generated = jax.lax.dynamic_update_slice(generated, out_tokens, (gen_idx,))

            new_g = fuse_features(fl_v[accepted], fm_v[accepted], fh_v[accepted],
                                  draft_weights)
            new_gen_idx = jnp.minimum(gen_idx + n_accepted, max_tokens)
            new_pos = pos + n_accepted

            return (new_gen_idx, correction, new_g, new_pos, caches, generated,
                    cycles + 1, acc_total + n_accepted)

        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        gen_idx, _, _, _, _, generated, cycles, acc_total = final
        return generated, gen_idx, cycles, acc_total

    return eagle3_loop

# ── EAGLE-3 generate (fused) ──

def eagle3_generate_fused(args, mesh, embed, final_norm, weights, caches, draft_weights,
                          cos_s, sin_s, cos_g, sin_g):
    tokenizer = load_tokenizer(args.model_dir)
    if tokenizer and not args.prompt.replace(',', '').isdigit():
        prompt_ids = [2] + tokenizer.encode(args.prompt).ids
    else:
        prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    print(f"prompt: {len(prompt_ids)} tokens", file=sys.stderr)

    fwd_jit = jax.jit(forward_step)
    fwd_feat_jit = jax.jit(forward_with_features)

    print("prefilling...", file=sys.stderr, flush=True)
    t0 = time.time()
    for step in range(len(prompt_ids) - 1):
        tok = jnp.array([prompt_ids[step]], dtype=jnp.int32)
        _, caches = fwd_jit(tok, jnp.int32(step), jnp.int32(step + 1),
                            embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
    step = len(prompt_ids) - 1
    tok = jnp.array([prompt_ids[step]], dtype=jnp.int32)
    first_tok, caches, fl, fm, fh = fwd_feat_jit(
        tok, jnp.int32(step), jnp.int32(step + 1),
        embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
    first_tok.block_until_ready()
    print(f"prefill done: {time.time()-t0:.2f}s", file=sys.stderr)

    last_token = jnp.int32(first_tok[0])
    g_last = fuse_features(fl[0], fm[0], fh[0], draft_weights)
    pos = len(prompt_ids)

    loop_fn = make_eagle3_fused_loop(args.max_tokens)
    loop_jit = jax.jit(loop_fn, donate_argnums=(3,))

    print("compiling fused EAGLE-3 loop...", file=sys.stderr, flush=True)
    t0 = time.time()
    gen_buf, gen_count, cycles, acc_total = loop_jit(
        last_token, g_last, jnp.int32(pos), caches,
        embed, final_norm, weights, draft_weights,
        cos_s, sin_s, cos_g, sin_g)
    gen_buf.block_until_ready()
    total_time = time.time() - t0

    generated = list(np.array(gen_buf)[:int(gen_count)])
    n_cycles = int(cycles)
    n_acc = int(acc_total)
    tau = n_acc / max(n_cycles, 1)

    print(f"\n=== EAGLE-3 Fused Results ===", file=sys.stderr)
    print(f"generated:     {len(generated)} tokens", file=sys.stderr)
    print(f"time:          {total_time:.2f}s (incl compile)", file=sys.stderr)
    print(f"tok/s:         {len(generated)/max(total_time, 0.001):.1f}", file=sys.stderr)
    print(f"cycles:        {n_cycles}", file=sys.stderr)
    print(f"avg tau:       {tau:.2f} tokens/cycle", file=sys.stderr)
    if n_cycles > 0:
        print(f"ms/cycle:      {total_time/n_cycles*1000:.1f} (incl compile amortized)", file=sys.stderr)

    if tokenizer:
        text = tokenizer.decode(generated)
        print(f"\n--- output ---\n{text}\n--- end ---", file=sys.stderr)

    # Re-run with fresh caches to get pure execution time
    print("\nre-running (cached compile)...", file=sys.stderr, flush=True)
    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    caches2 = {
        'kc': jax.device_put(jnp.zeros((NL, args.max_ctx, MAX_KV), dtype=jnp.bfloat16), kv_sh),
        'vc': jax.device_put(jnp.zeros((NL, args.max_ctx, MAX_KV), dtype=jnp.bfloat16), kv_sh),
    }
    for step in range(len(prompt_ids) - 1):
        tok = jnp.array([prompt_ids[step]], dtype=jnp.int32)
        _, caches2 = fwd_jit(tok, jnp.int32(step), jnp.int32(step + 1),
                             embed, final_norm, weights, caches2, cos_s, sin_s, cos_g, sin_g)
    step = len(prompt_ids) - 1
    tok = jnp.array([prompt_ids[step]], dtype=jnp.int32)
    first_tok2, caches2, fl2, fm2, fh2 = fwd_feat_jit(
        tok, jnp.int32(step), jnp.int32(step + 1),
        embed, final_norm, weights, caches2, cos_s, sin_s, cos_g, sin_g)
    last_token2 = jnp.int32(first_tok2[0])
    g_last2 = fuse_features(fl2[0], fm2[0], fh2[0], draft_weights)

    t0 = time.time()
    gen_buf2, gen_count2, cycles2, acc2 = loop_jit(
        last_token2, g_last2, jnp.int32(pos), caches2,
        embed, final_norm, weights, draft_weights,
        cos_s, sin_s, cos_g, sin_g)
    gen_buf2.block_until_ready()
    pure_time = time.time() - t0
    n_gen2 = int(gen_count2)
    n_cyc2 = int(cycles2)

    print(f"pure run:      {pure_time:.3f}s", file=sys.stderr)
    print(f"tok/s (pure):  {n_gen2/max(pure_time, 0.001):.1f}", file=sys.stderr)
    if n_cyc2 > 0:
        print(f"ms/cycle:      {pure_time/n_cyc2*1000:.2f}", file=sys.stderr)
    return generated

# ── EAGLE-3 generate (unfused, for debugging) ──

def eagle3_generate(args, mesh, embed, final_norm, weights, caches, draft_weights,
                    cos_s, sin_s, cos_g, sin_g):
    tokenizer = load_tokenizer(args.model_dir)
    if tokenizer and not args.prompt.replace(',', '').isdigit():
        prompt_ids = [2] + tokenizer.encode(args.prompt).ids
    else:
        prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    print(f"prompt: {len(prompt_ids)} tokens", file=sys.stderr)

    fwd_jit = jax.jit(forward_step)
    fwd_feat_jit = jax.jit(forward_with_features)

    def single_cycle(last_token, g_last, pos, caches,
                     embed, final_norm, weights, draft_weights,
                     cos_s, sin_s, cos_g, sin_g):
        draft_tokens, _ = draft_chain(last_token, g_last, draft_weights, embed, cos_s, sin_s)
        draft_flat = draft_tokens[:, 0]
        verify_input = jnp.concatenate([jnp.array([last_token]), draft_flat])
        target_argmax, caches, fl_v, fm_v, fh_v = verify_forward_fused(
            verify_input, pos, embed, final_norm, weights, caches,
            cos_s, sin_s, cos_g, sin_g)
        accepted, correction = on_device_accept(draft_flat, target_argmax)
        n_accepted = accepted + 1
        new_g = fuse_features(fl_v[accepted], fm_v[accepted], fh_v[accepted], draft_weights)
        new_pos = pos + n_accepted
        return draft_flat, accepted, correction, n_accepted, new_g, new_pos, caches

    cycle_jit = jax.jit(single_cycle, donate_argnums=(3,))

    print("prefilling...", file=sys.stderr, flush=True)
    t0 = time.time()
    for step in range(len(prompt_ids) - 1):
        tok = jnp.array([prompt_ids[step]], dtype=jnp.int32)
        _, caches = fwd_jit(tok, jnp.int32(step), jnp.int32(step + 1),
                            embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)

    step = len(prompt_ids) - 1
    tok = jnp.array([prompt_ids[step]], dtype=jnp.int32)
    first_tok, caches, fl, fm, fh = fwd_feat_jit(
        tok, jnp.int32(step), jnp.int32(step + 1),
        embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
    first_tok.block_until_ready()
    prefill_time = time.time() - t0
    print(f"prefill done: {prefill_time:.2f}s", file=sys.stderr)

    last_token = jnp.int32(first_tok[0])
    generated = [int(last_token)]
    g_last = fuse_features(fl[0], fm[0], fh[0], draft_weights)
    pos = jnp.int32(len(prompt_ids))

    total_cycles = 0
    total_accepted = 0
    t_start = time.time()

    print("compiling EAGLE-3 cycle...", file=sys.stderr, flush=True)

    while len(generated) < args.max_tokens:
        if int(last_token) in (1, 2, 107):
            break

        draft_flat, accepted, correction, n_acc, g_last, pos, caches = cycle_jit(
            last_token, g_last, pos, caches,
            embed, final_norm, weights, draft_weights,
            cos_s, sin_s, cos_g, sin_g)

        acc_int = int(accepted)
        draft_np = np.array(draft_flat)
        for j in range(acc_int):
            generated.append(int(draft_np[j]))
        generated.append(int(correction))
        last_token = correction

        total_cycles += 1
        total_accepted += int(n_acc)

        if total_cycles == 1:
            elapsed_first = time.time() - t_start
            print(f"first cycle done ({elapsed_first:.2f}s incl compile), decoding...", file=sys.stderr)

        if int(correction) in (1, 2, 107):
            break

    elapsed = time.time() - t_start
    tau = total_accepted / max(total_cycles, 1)

    print(f"\n=== EAGLE-3 Results ===", file=sys.stderr)
    print(f"generated:     {len(generated)} tokens", file=sys.stderr)
    print(f"decode time:   {elapsed:.2f}s", file=sys.stderr)
    print(f"tok/s:         {len(generated)/max(elapsed, 0.001):.1f}", file=sys.stderr)
    print(f"cycles:        {total_cycles}", file=sys.stderr)
    print(f"avg tau:       {tau:.2f} tokens/cycle", file=sys.stderr)
    print(f"acceptance:    {(tau - 1) / K_DRAFT:.1%} per draft position", file=sys.stderr)

    if tokenizer:
        text = tokenizer.decode(generated)
        print(f"\n--- output ---\n{text}\n--- end ---", file=sys.stderr)
    else:
        print(f"tokens: {generated[:40]}", file=sys.stderr)

    return generated

# ── baseline comparison (no speculation) ──

def baseline_generate(args, mesh, embed, final_norm, weights, caches,
                      cos_s, sin_s, cos_g, sin_g):
    tokenizer = load_tokenizer(args.model_dir)
    if tokenizer and not args.prompt.replace(',', '').isdigit():
        prompt_ids = [2] + tokenizer.encode(args.prompt).ids
    else:
        prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]

    fwd_jit = jax.jit(forward_step)
    generated = []
    last_tok = None

    print("baseline (no speculation)...", file=sys.stderr)
    t_start = time.time()
    for step in range(len(prompt_ids) + args.max_tokens):
        if step < len(prompt_ids):
            token_id = prompt_ids[step]
        else:
            token_id = last_tok
        tok = jnp.array([token_id], dtype=jnp.int32)
        next_tok, caches = fwd_jit(tok, jnp.int32(step), jnp.int32(step + 1),
                                   embed, final_norm, weights, caches,
                                   cos_s, sin_s, cos_g, sin_g)
        last_tok = int(next_tok[0])
        if step >= len(prompt_ids):
            generated.append(last_tok)
            if last_tok in (1, 2, 107):
                break

    elapsed = time.time() - t_start
    decode_tokens = len(generated)
    print(f"baseline: {decode_tokens} tokens in {elapsed:.2f}s = {decode_tokens/max(elapsed,0.001):.1f} tok/s",
          file=sys.stderr)
    return generated

# ── main ──

def main():
    parser = argparse.ArgumentParser(description='EAGLE-3 speculative decode for Gemma 4 on TPU')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--draft-dir', default=None, help='Trained draft head directory')
    parser.add_argument('--random-draft', action='store_true', help='Use random draft weights (pipeline test)')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--max-ctx', type=int, default=2048)
    parser.add_argument('--prompt', default='2')
    parser.add_argument('--baseline', action='store_true', help='Run baseline comparison')
    parser.add_argument('--fused', action='store_true', help='Fused on-device decode (zero dispatch overhead)')
    args = parser.parse_args()

    if not args.draft_dir and not args.random_draft and not args.baseline:
        print("ERROR: specify --draft-dir or --random-draft", file=sys.stderr)
        sys.exit(1)

    mesh = make_mesh()
    print(f"mesh: {mesh} ({len(jax.devices())} devices)", file=sys.stderr)

    embed, final_norm, weights, caches = load_model(args.model_dir, mesh, args.max_ctx)

    cos_s, sin_s = precompute_rope(10000.0, S_HD, args.max_ctx)
    cos_g, sin_g = precompute_rope(1000000.0, 128, args.max_ctx)
    cos_s = jax.device_put(jnp.array(cos_s), NamedSharding(mesh, P(None, None)))
    sin_s = jax.device_put(jnp.array(sin_s), NamedSharding(mesh, P(None, None)))
    cos_g = jax.device_put(jnp.array(cos_g), NamedSharding(mesh, P(None, None)))
    sin_g = jax.device_put(jnp.array(sin_g), NamedSharding(mesh, P(None, None)))

    if args.baseline:
        baseline_generate(args, mesh, embed, final_norm, weights, caches,
                          cos_s, sin_s, cos_g, sin_g)
        return

    if args.draft_dir:
        draft_weights = load_draft_weights(args.draft_dir, mesh)
    else:
        print("init random draft weights (pipeline test, expect ~0% acceptance)", file=sys.stderr)
        draft_weights = init_random_draft(mesh)

    if args.fused:
        eagle3_generate_fused(args, mesh, embed, final_norm, weights, caches, draft_weights,
                              cos_s, sin_s, cos_g, sin_g)
    else:
        eagle3_generate(args, mesh, embed, final_norm, weights, caches, draft_weights,
                        cos_s, sin_s, cos_g, sin_g)

if __name__ == '__main__':
    main()
