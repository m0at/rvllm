#!/usr/bin/env python3
"""MiniMax-M2.7 Multi-Token-Prediction heads (speculative decode).

Per the M2 config:
  num_mtp_modules      = 3
  mtp_transformer_layers = 1

Each MTP module i takes:
  h_i       : (B, H)   previous hidden state
  prev_tok  : (B,)     previously predicted token id
and produces:
  logits_i  : (B, V)   logits over the i-th-next token
  h_{i+1}   : (B, H)   next hidden state for the next module

Per-module submodules:
  enorm       : RMSNorm on embedded token   (H,)
  hnorm       : RMSNorm on previous hidden  (H,)
  eh_proj     : Linear (2H -> H)             (H, 2H)
  input_ln    : RMSNorm (H,)      -- attention pre-norm
  q_proj, k_proj, v_proj, o_proj : attention projections (GQA)
  q_norm, k_norm                  : per-head RMSNorm (head_dim,)
  post_attn_ln: RMSNorm (H,)      -- FFN pre-norm
  gate_proj, up_proj, down_proj   : SwiGLU FFN
  final_ln    : RMSNorm (H,)

lm_head is shared with the main model and passed in as an argument.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

# Constants from M2 config
NUM_MTP_MODULES = 3
MTP_TRANSFORMER_LAYERS = 1

H = 3072
NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
MOE_INTER_FALLBACK = 1536  # dense FFN inside MTP transformer layer; actual value from weights
ROTARY_DIM = 64
ROPE_THETA = 5_000_000.0
RMS_EPS = 1e-6


def _rms_norm(x, g, eps=RMS_EPS):
    x32 = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x * inv.astype(x.dtype)) * g


def _head_norm(h, g, eps=RMS_EPS):
    h32 = h.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + eps)
    return (h * inv.astype(h.dtype)) * g


def _rope_partial(x, cos, sin, rot_dim):
    half = rot_dim // 2
    xr = x[..., :rot_dim]
    xp = x[..., rot_dim:]
    x0 = xr[..., :half]
    x1 = xr[..., half:]
    rotated = jnp.concatenate([
        x0 * cos.astype(x.dtype) - x1 * sin.astype(x.dtype),
        x0 * sin.astype(x.dtype) + x1 * cos.astype(x.dtype),
    ], axis=-1)
    return jnp.concatenate([rotated, xp], axis=-1)


def _precompute_rope(theta, rot_dim, pos_ids):
    half = rot_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, rot_dim, 2, dtype=jnp.float32) / rot_dim))
    angles = pos_ids.astype(jnp.float32)[:, None] * inv_freq[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def _attention_block(x, w, pos_ids):
    """Single-token / short-seq self-attention inside one MTP transformer layer.

    Here the MTP module sees only one new hidden per step, so we run attention
    WITHOUT a KV cache: the query attends to itself (plus an optional external
    cache if the integrator passes one in — kept minimal for the smoke test).

    Shapes:
      x : (B, T, H)   T = 1 during decode
      returns (B, T, H)
    """
    B, T, _ = x.shape
    h = _rms_norm(x, w['input_ln'])

    q = (h @ w['q_proj'].T).reshape(B, T, NUM_Q_HEADS, HEAD_DIM)
    k = (h @ w['k_proj'].T).reshape(B, T, NUM_KV_HEADS, HEAD_DIM)
    v = (h @ w['v_proj'].T).reshape(B, T, NUM_KV_HEADS, HEAD_DIM)

    q = _head_norm(q, w['q_norm'])
    k = _head_norm(k, w['k_norm'])

    cos, sin = _precompute_rope(ROPE_THETA, ROTARY_DIM, pos_ids)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q = _rope_partial(q, cos, sin, ROTARY_DIM)
    k = _rope_partial(k, cos, sin, ROTARY_DIM)

    gqa = NUM_Q_HEADS // NUM_KV_HEADS
    q_g = q.reshape(B, T, NUM_KV_HEADS, gqa, HEAD_DIM)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    sc = jnp.einsum('btghd,bsgd->btghs', q_g.astype(jnp.float32),
                    k.astype(jnp.float32)) * scale
    mask = jnp.arange(T)[:, None] >= jnp.arange(T)[None, :]
    sc = jnp.where(mask[None, :, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(x.dtype)
    out = jnp.einsum('btghs,bsgd->btghd', p, v).reshape(B, T, NUM_Q_HEADS * HEAD_DIM)
    out = out @ w['o_proj'].T
    return x + out


def _ffn_block(x, w):
    h = _rms_norm(x, w['post_attn_ln'])
    gate = h @ w['gate_proj'].T
    up = h @ w['up_proj'].T
    h = jax.nn.silu(gate) * up
    return x + h @ w['down_proj'].T


def _mtp_module_forward(h_prev, embed_tok, w, pos_ids):
    """Run one MTP module.

    h_prev    : (B, H)   previous hidden state
    embed_tok : (B, H)   embedded previous token
    w         : dict with per-module weights
    pos_ids   : (T,)     position ids for rope (T=1 at decode)
    returns   : (B, H)   new hidden state
    """
    e = _rms_norm(embed_tok, w['enorm'])
    h = _rms_norm(h_prev, w['hnorm'])
    cat = jnp.concatenate([e, h], axis=-1)          # (B, 2H)
    z = cat @ w['eh_proj'].T                        # (B, H)
    z = z[:, None, :]                               # (B, 1, H)
    z = _attention_block(z, w, pos_ids)
    z = _ffn_block(z, w)
    z = _rms_norm(z, w['final_ln'])
    return z[:, 0, :]


def mtp_forward(h_last, mtp_weights, embed, prev_token_id, lm_head=None):
    """Run all NUM_MTP_MODULES MTP heads in a chain.

    h_last        : (B, H)             hidden state after main model's final norm
    mtp_weights   : list of dicts      per-module weights (len == NUM_MTP_MODULES)
    embed         : (V, H)             token embedding matrix (shared with main)
    prev_token_id : (B,)               token id predicted by the main model
    lm_head       : (V, H) or None     if None, tied to embed

    Returns draft_logits : (B, NUM_MTP_MODULES, V)
    """
    assert len(mtp_weights) == NUM_MTP_MODULES, \
        f"expected {NUM_MTP_MODULES} MTP modules, got {len(mtp_weights)}"

    head = lm_head if lm_head is not None else embed
    B = h_last.shape[0]
    h = h_last
    prev_tok = prev_token_id
    out_logits = []
    pos_ids = jnp.zeros((1,), dtype=jnp.int32)

    for i in range(NUM_MTP_MODULES):
        etok = embed[prev_tok]                       # (B, H)
        h = _mtp_module_forward(h, etok, mtp_weights[i], pos_ids)
        logits = h.astype(jnp.float32) @ head.astype(jnp.float32).T   # (B, V)
        out_logits.append(logits)
        prev_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    return jnp.stack(out_logits, axis=1)             # (B, NUM_MTP_MODULES, V)


# ── weight loading ──

def _to_bf16(arr):
    if arr.dtype == np.uint16:
        return arr.view(ml_dtypes.bfloat16)
    if arr.dtype == np.float16:
        return arr.astype(np.float32).astype(ml_dtypes.bfloat16)
    if arr.dtype == np.float32:
        return arr.astype(ml_dtypes.bfloat16)
    if arr.dtype == ml_dtypes.bfloat16:
        return arr
    raise ValueError(f"unsupported dtype {arr.dtype}")


def mtp_load_weights(reader, prefix):
    """Load NUM_MTP_MODULES MTP modules from a ModeloptSafetensorsReader.

    `reader` has methods `.read_bf16(name) -> np.ndarray`, `.read_nvfp4(name)`,
    `.is_nvfp4(name) -> bool`, and `.list_tensors() -> list[str]`.
    `prefix` is e.g. 'model.mtp_modules.' — modules indexed as `{prefix}{i}.*`.

    All MTP tensors live in the bf16 ignore list, so we always use
    `reader.read_bf16(name)`. Missing tensors raise — no fallbacks.
    """
    out = []
    for i in range(NUM_MTP_MODULES):
        base = f'{prefix}{i}.'

        def g(suffix):
            name = base + suffix
            arr = reader.read_bf16(name)
            arr_bf16 = _to_bf16(arr)
            return jnp.asarray(arr_bf16, dtype=jnp.bfloat16)

        w = {
            'enorm':        g('enorm.weight'),
            'hnorm':        g('hnorm.weight'),
            'eh_proj':      g('eh_proj.weight'),
            'input_ln':     g('transformer.input_layernorm.weight'),
            'q_proj':       g('transformer.self_attn.q_proj.weight'),
            'k_proj':       g('transformer.self_attn.k_proj.weight'),
            'v_proj':       g('transformer.self_attn.v_proj.weight'),
            'o_proj':       g('transformer.self_attn.o_proj.weight'),
            'q_norm':       g('transformer.self_attn.q_norm.weight'),
            'k_norm':       g('transformer.self_attn.k_norm.weight'),
            'post_attn_ln': g('transformer.post_attention_layernorm.weight'),
            'gate_proj':    g('transformer.mlp.gate_proj.weight'),
            'up_proj':      g('transformer.mlp.up_proj.weight'),
            'down_proj':    g('transformer.mlp.down_proj.weight'),
            'final_ln':     g('final_layernorm.weight'),
        }
        out.append(w)
    return out


# ── smoke test ──

def _random_mtp_weights(key, ffn_inter):
    keys = jax.random.split(key, 32)
    ki = [0]
    def nk():
        k = keys[ki[0] % len(keys)]
        ki[0] += 1
        return k

    modules = []
    for _ in range(NUM_MTP_MODULES):
        w = {
            'enorm':        jnp.ones(H, dtype=jnp.bfloat16),
            'hnorm':        jnp.ones(H, dtype=jnp.bfloat16),
            'eh_proj':      jax.random.normal(nk(), (H, 2 * H), dtype=jnp.bfloat16) * 0.02,
            'input_ln':     jnp.ones(H, dtype=jnp.bfloat16),
            'q_proj':       jax.random.normal(nk(), (NUM_Q_HEADS * HEAD_DIM, H), dtype=jnp.bfloat16) * 0.02,
            'k_proj':       jax.random.normal(nk(), (NUM_KV_HEADS * HEAD_DIM, H), dtype=jnp.bfloat16) * 0.02,
            'v_proj':       jax.random.normal(nk(), (NUM_KV_HEADS * HEAD_DIM, H), dtype=jnp.bfloat16) * 0.02,
            'o_proj':       jax.random.normal(nk(), (H, NUM_Q_HEADS * HEAD_DIM), dtype=jnp.bfloat16) * 0.02,
            'q_norm':       jnp.ones(HEAD_DIM, dtype=jnp.bfloat16),
            'k_norm':       jnp.ones(HEAD_DIM, dtype=jnp.bfloat16),
            'post_attn_ln': jnp.ones(H, dtype=jnp.bfloat16),
            'gate_proj':    jax.random.normal(nk(), (ffn_inter, H), dtype=jnp.bfloat16) * 0.02,
            'up_proj':      jax.random.normal(nk(), (ffn_inter, H), dtype=jnp.bfloat16) * 0.02,
            'down_proj':    jax.random.normal(nk(), (H, ffn_inter), dtype=jnp.bfloat16) * 0.02,
            'final_ln':     jnp.ones(H, dtype=jnp.bfloat16),
        }
        modules.append(w)
    return modules


def _smoke():
    jax.config.update('jax_platforms', 'cpu')

    B = 2
    V = 1024            # small vocab for smoke
    ffn_inter = 256     # small FFN for smoke
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    embed = jax.random.normal(k1, (V, H), dtype=jnp.bfloat16) * 0.02
    lm_head = jax.random.normal(k2, (V, H), dtype=jnp.bfloat16) * 0.02
    h_last = jax.random.normal(k3, (B, H), dtype=jnp.bfloat16)
    prev_tok = jnp.array([7, 42], dtype=jnp.int32)
    mtp_w = _random_mtp_weights(k4, ffn_inter)

    print(f"smoke: B={B} H={H} V={V} modules={NUM_MTP_MODULES}", file=sys.stderr)

    logits = mtp_forward(h_last, mtp_w, embed, prev_tok, lm_head=lm_head)
    logits.block_until_ready()
    print(f"  logits.shape = {logits.shape}  dtype={logits.dtype}", file=sys.stderr)
    assert logits.shape == (B, NUM_MTP_MODULES, V), logits.shape
    assert jnp.all(jnp.isfinite(logits)), "non-finite logits"

    # tied-embedding path (lm_head=None -> use embed)
    logits2 = mtp_forward(h_last, mtp_w, embed, prev_tok, lm_head=None)
    logits2.block_until_ready()
    assert logits2.shape == (B, NUM_MTP_MODULES, V)

    # jit path
    jitted = jax.jit(mtp_forward)
    logits3 = jitted(h_last, mtp_w, embed, prev_tok, lm_head)
    logits3.block_until_ready()
    assert logits3.shape == (B, NUM_MTP_MODULES, V)

    print("  OK", file=sys.stderr)


if __name__ == '__main__':
    _smoke()
