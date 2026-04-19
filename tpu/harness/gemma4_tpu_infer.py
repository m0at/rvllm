#!/usr/bin/env python3
"""Gemma 4 inference on TPU v6e-4 via JAX SPMD (TP=4).

Auto-detects model dimensions from config.json. Supports:
  - Gemma 4 31B (google/gemma-4-31B-it) -- dual head dims (256/512)
  - Gemma 4 E4B (google/gemma-4-E4B-it) -- uniform head dim

Dual-path architecture:
  - max_ctx <= 32768: single-scan path (one jax.lax.scan over NL layers with
    jax.lax.cond dispatch, unified KV cache [NL, max_ctx, MAX_KV]).
  - max_ctx > 32768: split-cache path (groups of sliding + global,
    circular buffer for sliding, blockwise for global).

Usage:
    python3 gemma4_tpu_infer.py --model-dir /path/to/gemma-4-31B-it \
        --max-tokens 32 --prompt "2,3257"
"""
import argparse, json, os, struct, sys, time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes

# Universal constants (model-independent)
BLOCK_K = 8192
SOFTCAP_VAL = 30.0
EPS    = 1e-6
B      = 1  # batch size; set via --batch flag
SPLIT_THRESHOLD = 32768

# Model-specific globals (set by load_config)
H = NH = INTER = VOCAB = NL = WINDOW = 0
N_GROUPS = SLIDING_PER_GROUP = N_SLIDING = N_GLOBAL = 0
MAX_Q = MAX_KV = MAX_O = MAX_NORM_HD = 0
S_Q = S_KV = S_HD = S_KVH = S_GQA = 0
G_Q = G_KV = G_HD = G_KVH = G_GQA = 0
MAX_KVH = 0
LAYER_IS_GLOBAL = np.array([], dtype=np.int32)

# MoE globals (set by load_config when enable_moe_block is True)
ENABLE_MOE = False
NUM_EXPERTS = 0
TOP_K_EXPERTS = 0
MOE_INTER = 0


def load_config(model_dir):
    """Read config.json, handle text_config nesting, set all model globals."""
    global H, NH, INTER, VOCAB, NL, WINDOW
    global N_GROUPS, SLIDING_PER_GROUP, N_SLIDING, N_GLOBAL
    global MAX_Q, MAX_KV, MAX_O, MAX_NORM_HD
    global S_Q, S_KV, S_HD, S_KVH, S_GQA
    global G_Q, G_KV, G_HD, G_KVH, G_GQA
    global MAX_KVH, LAYER_IS_GLOBAL
    global ENABLE_MOE, NUM_EXPERTS, TOP_K_EXPERTS, MOE_INTER

    cfg_path = os.path.join(model_dir, 'config.json')
    with open(cfg_path) as f:
        raw = json.load(f)

    # Gemma4ForConditionalGeneration nests text params under text_config
    tc = raw.get('text_config', raw)

    H     = tc['hidden_size']
    NH    = tc['num_attention_heads']
    NL    = tc['num_hidden_layers']
    INTER = tc['intermediate_size']
    VOCAB = tc['vocab_size']
    WINDOW = tc.get('sliding_window', 1024)

    # Determine layer pattern: which layers are global
    # Gemma 4 uses sliding_window_pattern (int): every N-th layer is global
    # e.g. pattern=6 means layer indices where (i+1)%6==0 are global
    sw_pattern = tc.get('sliding_window_pattern', 6)
    LAYER_IS_GLOBAL = np.array(
        [1 if (i + 1) % sw_pattern == 0 else 0 for i in range(NL)],
        dtype=np.int32)
    N_GLOBAL = int(LAYER_IS_GLOBAL.sum())
    N_SLIDING = NL - N_GLOBAL
    # Groups: each group = (sw_pattern-1) sliding + 1 global
    SLIDING_PER_GROUP = sw_pattern - 1
    N_GROUPS = N_GLOBAL  # one global per group

    # Head dimensions -- detect dual-attention (different sliding vs global)
    # Fields: head_dim (sliding), query_pre_attn_scalar, per-layer attn config
    # Gemma 4 31B: head_dim=256 (sliding), but global layers use 512
    # Smaller models may have uniform head_dim
    s_hd = tc.get('head_dim', 256)
    # Global head dim: check for separate field, else same as sliding
    g_hd = tc.get('global_head_dim', None)

    # KV heads: may be int or may differ per layer type
    kv_heads = tc.get('num_key_value_heads', NH)

    if g_hd is not None:
        # Dual attention: different head dims for sliding vs global
        s_kvh = kv_heads  # sliding kv heads
        # Global kv heads: NH * g_hd must equal global q_proj size
        # and kv_heads * s_hd = sliding kv_proj size
        # For global: total kv dim = hidden_size * (g_hd / (NH * g_hd / NH))
        # Simpler: infer from the ratio
        g_kvh = tc.get('global_num_key_value_heads') or tc.get('num_global_key_value_heads') or kv_heads
    else:
        # Check if model has attn_types or similar for dual attention
        # Gemma 4 31B config has: head_dim=256, but global layers actually use
        # q_proj.weight shape [NH*global_hd, H] which differs from sliding
        # We detect this from num_key_value_heads and head_dim
        # For truly uniform models, sliding==global
        g_hd = s_hd
        s_kvh = kv_heads
        g_kvh = kv_heads

    # Allow explicit overrides for known dual-attention configs
    # Gemma 4 31B: hidden=5376, heads=32, head_dim=256 -> q_dim=8192 (sliding)
    #   but global q_proj is 16384 = 32*512, kv_proj is 2048 = 4*512
    # Detection: if config has 'global_head_dim' we already handled it.
    # Otherwise, check query_pre_attn_scalar as a hint:
    #   scalar = head_dim (sliding) for Gemma 4
    # For models where head_dim * NH != q_proj actual size on global layers,
    # the loader will discover the mismatch from weight shapes.
    # Here we just set the dims and the loader's pad_to handles it.

    S_HD = s_hd
    S_KVH = s_kvh
    S_Q = NH * S_HD
    S_KV = S_KVH * S_HD
    S_GQA = NH // S_KVH

    G_HD = g_hd
    G_KVH = g_kvh
    G_Q = NH * G_HD
    G_KV = G_KVH * G_HD
    G_GQA = NH // G_KVH

    MAX_Q = max(S_Q, G_Q)
    MAX_KV = max(S_KV, G_KV)
    MAX_O = MAX_Q
    MAX_NORM_HD = max(S_HD, G_HD)
    MAX_KVH = max(S_KVH, G_KVH)

    # MoE detection
    ENABLE_MOE = bool(tc.get('enable_moe_block', False))
    if ENABLE_MOE:
        NUM_EXPERTS = tc['num_experts']
        TOP_K_EXPERTS = tc['top_k_experts']
        MOE_INTER = tc['moe_intermediate_size']

    print(f"config: H={H} NH={NH} NL={NL} INTER={INTER} VOCAB={VOCAB} WINDOW={WINDOW}", file=sys.stderr)
    print(f"  sliding: HD={S_HD} KVH={S_KVH} Q={S_Q} KV={S_KV} GQA={S_GQA}", file=sys.stderr)
    print(f"  global:  HD={G_HD} KVH={G_KVH} Q={G_Q} KV={G_KV} GQA={G_GQA}", file=sys.stderr)
    print(f"  groups={N_GROUPS} sliding/group={SLIDING_PER_GROUP} "
          f"global_layers={N_GLOBAL} sliding_layers={N_SLIDING}", file=sys.stderr)
    print(f"  MAX_Q={MAX_Q} MAX_KV={MAX_KV} MAX_NORM_HD={MAX_NORM_HD}", file=sys.stderr)
    if ENABLE_MOE:
        print(f"  MoE: experts={NUM_EXPERTS} top_k={TOP_K_EXPERTS} moe_inter={MOE_INTER}", file=sys.stderr)

def make_mesh():
    devs = jax.devices()
    return Mesh(np.array(devs[:4]), ('tp',))

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

# -- KV quantization --

def _quant_kv(k_heads, num_kv_heads, head_dim):
    amax = jnp.max(jnp.abs(k_heads), axis=-1, keepdims=True).clip(min=1e-8)
    scale = amax / 127.0
    ki8 = jnp.round(k_heads / scale).clip(-127, 127).astype(jnp.int8)
    return ki8, scale.squeeze(-1)

def _dequant_kv(ki8, scale):
    return ki8.astype(jnp.bfloat16) * scale[:, :, None]

# -- int8 matmul --

def int8_matmul(x, w_int8, scale):
    return (x @ w_int8.astype(jnp.bfloat16).T) * scale

# -- MoE FFN --

def moe_ffn(x, router_w, router_scale, per_expert_scale, expert_gate_up, expert_down):
    """Mixture-of-Experts FFN with top-k routing.

    Args:
        x: [B, H] input (bf16)
        router_w: [NUM_EXPERTS, H] router projection (bf16)
        router_scale: scalar (bf16)
        per_expert_scale: [NUM_EXPERTS] per-expert scale (bf16)
        expert_gate_up: [NUM_EXPERTS, 2*MOE_INTER, H] fused gate+up (bf16)
        expert_down: [NUM_EXPERTS, H, MOE_INTER] down projection (bf16)
    Returns:
        [B, H] MoE output (bf16)
    """
    logits = (x @ router_w.T) * router_scale          # [B, NUM_EXPERTS]
    logits = logits * per_expert_scale                 # per-expert scaling
    topk_vals, topk_idx = jax.lax.top_k(logits, TOP_K_EXPERTS)  # [B, TOP_K]
    weights = jax.nn.softmax(topk_vals.astype(jnp.float32), axis=-1).astype(x.dtype)  # [B, TOP_K]

    # Unrolled loop over top-k experts (8 iterations, traced at compile time)
    out = jnp.zeros_like(x)  # [B, H]
    for e in range(TOP_K_EXPERTS):
        idx_e = topk_idx[:, e]                          # [B] expert indices
        gu_e = expert_gate_up[idx_e[0]]                 # [2*MOE_INTER, H] (B=1)
        d_e = expert_down[idx_e[0]]                     # [H, MOE_INTER]
        h_gu = x @ gu_e.T                               # [B, 2*MOE_INTER]
        gate_e = h_gu[:, :MOE_INTER]
        up_e = h_gu[:, MOE_INTER:]
        h_e = jax.nn.gelu(gate_e, approximate=True) * up_e  # [B, MOE_INTER]
        d_out = h_e @ d_e.T                             # [B, H]
        out = out + weights[:, e:e+1] * d_out
    return out

# -- sliding attention (WINDOW-sized circular buffer) --

def _sliding_attn(q_flat, k_flat, v_flat, qn, kn, cos_s, sin_s, kc, vc, kc_s, vc_s, pos):
    q = head_norm(q_flat[:, :S_Q].reshape(B, NH, S_HD), qn[:S_HD])
    k = head_norm(k_flat[:, :S_KV].reshape(B, S_KVH, S_HD), kn[:S_HD])
    v = head_norm_noscale(v_flat[:, :S_KV].reshape(B, S_KVH, S_HD))
    c = cos_s[pos][None, None, :]
    s = sin_s[pos][None, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)
    ki8, ks = _quant_kv(k[0], S_KVH, S_HD)
    vi8, vs = _quant_kv(v[0], S_KVH, S_HD)
    # Write at circular position
    write_pos = pos % WINDOW
    kc = kc.at[write_pos].set(ki8.reshape(S_KV))
    vc = vc.at[write_pos].set(vi8.reshape(S_KV))
    kc_s = kc_s.at[write_pos].set(ks)
    vc_s = vc_s.at[write_pos].set(vs)
    # Read full WINDOW and mask
    k_ctx = _dequant_kv(kc[:, :S_KV].reshape(WINDOW, S_KVH, S_HD),
                         kc_s[:, :S_KVH])
    v_ctx = _dequant_kv(vc[:, :S_KV].reshape(WINDOW, S_KVH, S_HD),
                         vc_s[:, :S_KVH])
    q_g = q.reshape(B, S_KVH, S_GQA, S_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    # Build validity mask for circular buffer
    # Entry at slot t holds position: we need to figure out which absolute positions are valid
    # After writing pos at write_pos, valid entries are [max(0, pos-WINDOW+1) .. pos]
    # Slot t holds absolute position: (pos - (write_pos - t) % WINDOW) but simpler to think:
    # number of valid entries = min(pos+1, WINDOW)
    # Slots are valid if they have been written to
    num_valid = jnp.minimum(pos + 1, WINDOW)
    # Slots written: write_pos, write_pos-1, ..., write_pos-num_valid+1 (mod WINDOW)
    # Equivalently, slot t is valid if ((write_pos - t) % WINDOW) < num_valid
    t = jnp.arange(WINDOW)
    dist = (write_pos - t) % WINDOW
    valid = dist < num_valid
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(B, S_Q)
    return out, kc, vc, kc_s, vc_s

# -- global attention (blockwise, full context) --

def _global_attn(q_flat, k_flat, v_flat, qn, kn, cos_g, sin_g, kc, vc, kc_s, vc_s, pos, ctx, max_ctx):
    k_raw = k_flat[:, :G_KV].reshape(B, G_KVH, G_HD)
    q = head_norm(q_flat[:, :G_Q].reshape(B, NH, G_HD), qn)
    k = head_norm(k_raw, kn)
    v = head_norm_noscale(k_raw)
    c = cos_g[pos][None, None, :]
    s = sin_g[pos][None, None, :]
    q = rope(q, c, s, 128)
    k = rope(k, c, s, 128)
    ki8, ks = _quant_kv(k[0], G_KVH, G_HD)
    vi8, vs = _quant_kv(v[0], G_KVH, G_HD)
    kc = kc.at[pos].set(ki8.reshape(G_KV))
    vc = vc.at[pos].set(vi8.reshape(G_KV))
    kc_s = kc_s.at[pos].set(ks)
    vc_s = vc_s.at[pos].set(vs)
    q_g = q.reshape(B, G_KVH, G_GQA, G_HD)
    eff_block = min(BLOCK_K, max_ctx)
    num_blocks = max_ctx // eff_block

    def block_fn(carry, block_idx):
        m_prev, l_prev, o_prev = carry
        start = block_idx * eff_block
        kb = jax.lax.dynamic_slice(kc, [start, 0], [eff_block, G_KV])
        vb = jax.lax.dynamic_slice(vc, [start, 0], [eff_block, G_KV])
        kbs = jax.lax.dynamic_slice(kc_s, [start, 0], [eff_block, G_KVH])
        vbs = jax.lax.dynamic_slice(vc_s, [start, 0], [eff_block, G_KVH])
        k_block = _dequant_kv(kb.reshape(eff_block, G_KVH, G_HD), kbs)
        v_block = _dequant_kv(vb.reshape(eff_block, G_KVH, G_HD), vbs)
        sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_block.astype(jnp.float32))
        abs_pos = start + jnp.arange(eff_block)
        valid = abs_pos < ctx
        sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
        m_new = jnp.maximum(m_prev, jnp.max(sc, axis=-1, keepdims=True))
        exp_sc = jnp.exp(sc - m_new)
        scale = jnp.exp(m_prev - m_new)
        l_new = l_prev * scale + jnp.sum(exp_sc, axis=-1, keepdims=True)
        o_new = o_prev * scale + jnp.einsum('bght,tgd->bghd', exp_sc.astype(jnp.bfloat16), v_block)
        return (m_new, l_new, o_new), None

    init = (jnp.full((B, G_KVH, G_GQA, 1), -1e30, dtype=jnp.float32),
            jnp.zeros((B, G_KVH, G_GQA, 1), dtype=jnp.float32),
            jnp.zeros((B, G_KVH, G_GQA, G_HD), dtype=jnp.float32))
    (_, l_final, o_final), _ = jax.lax.scan(block_fn, init, jnp.arange(num_blocks))
    out = (o_final / l_final).reshape(B, G_Q).astype(jnp.bfloat16)
    return out, kc, vc, kc_s, vc_s

# -- scan bodies --

def sliding_one_layer(carry, xs):
    """Scan body for a single sliding-window layer."""
    x, pos, cos_s, sin_s = carry

    residual = x
    h = rms_norm(x, xs['ln1'])
    q_flat = int8_matmul(h, xs['qw'], xs['qw_s'])
    k_flat = int8_matmul(h, xs['kw'], xs['kw_s'])
    v_flat = int8_matmul(h, xs['vw'], xs['vw_s'])

    attn_out, kc, vc, kc_s, vc_s = _sliding_attn(
        q_flat, k_flat, v_flat, xs['qn'], xs['kn'],
        cos_s, sin_s, xs['kc'], xs['vc'], xs['kc_s'], xs['vc_s'], pos)

    # Pad to MAX_Q for o_proj (which is [H, MAX_O])
    attn_out_pad = jnp.pad(attn_out, ((0, 0), (0, MAX_Q - S_Q))).astype(jnp.bfloat16)
    o_out = int8_matmul(attn_out_pad, xs['ow'], xs['ow_s'])

    h = o_out
    h = rms_norm(h, xs['ln2'])
    x = residual + h

    if ENABLE_MOE:
        residual = x
        h_dense = rms_norm(x, xs['ln3'])
        gate = int8_matmul(h_dense, xs['gw'], xs['gw_s'])
        up = int8_matmul(h_dense, xs['uw'], xs['uw_s'])
        h_dense = jax.nn.gelu(gate, approximate=True) * up
        h_dense = int8_matmul(h_dense, xs['dw'], xs['dw_s'])
        h_dense = rms_norm(h_dense, xs['ln4'])

        h_moe = rms_norm(x, xs['ln3_moe'])
        h_moe = moe_ffn(h_moe, xs['router_w'], xs['router_s'],
                         xs['router_ps'], xs['expert_gu'], xs['expert_dw'])
        h_moe = rms_norm(h_moe, xs['ln4_moe'])

        x = residual + h_dense + h_moe
        x = rms_norm(x, xs['ln4_combine'])
        x = x * xs['ls']
    else:
        residual = x
        h = rms_norm(x, xs['ln3'])
        gate = int8_matmul(h, xs['gw'], xs['gw_s'])
        up = int8_matmul(h, xs['uw'], xs['uw_s'])
        h = jax.nn.gelu(gate, approximate=True) * up
        h = int8_matmul(h, xs['dw'], xs['dw_s'])
        h = rms_norm(h, xs['ln4'])
        x = (residual + h) * xs['ls']

    return (x, pos, cos_s, sin_s), {'kc': kc, 'vc': vc, 'kc_s': kc_s, 'vc_s': vc_s}


def global_one_layer(x, pos, ctx, cos_g, sin_g, max_ctx, ws):
    """Process a single global layer (no scan, called directly)."""
    residual = x
    h = rms_norm(x, ws['ln1'])
    q_flat = int8_matmul(h, ws['qw'], ws['qw_s'])
    k_flat = int8_matmul(h, ws['kw'], ws['kw_s'])
    v_flat = int8_matmul(h, ws['vw'], ws['vw_s'])

    attn_out, kc, vc, kc_s, vc_s = _global_attn(
        q_flat, k_flat, v_flat, ws['qn'], ws['kn'],
        cos_g, sin_g, ws['kc'], ws['vc'], ws['kc_s'], ws['vc_s'], pos, ctx, max_ctx)

    o_out = int8_matmul(attn_out, ws['ow'], ws['ow_s'])

    h = o_out
    h = rms_norm(h, ws['ln2'])
    x = residual + h

    if ENABLE_MOE:
        residual = x
        h_dense = rms_norm(x, ws['ln3'])
        gate = int8_matmul(h_dense, ws['gw'], ws['gw_s'])
        up = int8_matmul(h_dense, ws['uw'], ws['uw_s'])
        h_dense = jax.nn.gelu(gate, approximate=True) * up
        h_dense = int8_matmul(h_dense, ws['dw'], ws['dw_s'])
        h_dense = rms_norm(h_dense, ws['ln4'])

        h_moe = rms_norm(x, ws['ln3_moe'])
        h_moe = moe_ffn(h_moe, ws['router_w'], ws['router_s'],
                         ws['router_ps'], ws['expert_gu'], ws['expert_dw'])
        h_moe = rms_norm(h_moe, ws['ln4_moe'])

        x = residual + h_dense + h_moe
        x = rms_norm(x, ws['ln4_combine'])
        x = x * ws['ls']
    else:
        residual = x
        h = rms_norm(x, ws['ln3'])
        gate = int8_matmul(h, ws['gw'], ws['gw_s'])
        up = int8_matmul(h, ws['uw'], ws['uw_s'])
        h = jax.nn.gelu(gate, approximate=True) * up
        h = int8_matmul(h, ws['dw'], ws['dw_s'])
        h = rms_norm(h, ws['ln4'])
        x = (residual + h) * ws['ls']

    return x, kc, vc, kc_s, vc_s

# -- forward --

def _slice_weight(w, start, count):
    """Slice leading (layer) dim of a weight array."""
    return jax.lax.dynamic_slice_in_dim(w, start, count, axis=0)

def _build_sliding_xs(sw, sc, g):
    """Build scan input dict for group g's 5 sliding layers."""
    s = g * SLIDING_PER_GROUP
    xs = {}
    for k in ('qw', 'kw', 'vw', 'ow', 'gw', 'uw', 'dw'):
        xs[k] = _slice_weight(sw[k], s, SLIDING_PER_GROUP)
        xs[k + '_s'] = _slice_weight(sw[k + '_s'], s, SLIDING_PER_GROUP)
    for k in ('qn', 'kn', 'ln1', 'ln2', 'ln3', 'ln4', 'ls'):
        xs[k] = _slice_weight(sw[k], s, SLIDING_PER_GROUP)
    if ENABLE_MOE:
        for k in ('ln3_moe', 'ln4_moe', 'ln4_combine'):
            xs[k] = _slice_weight(sw[k], s, SLIDING_PER_GROUP)
        for k in ('router_w', 'router_s', 'router_ps', 'expert_gu', 'expert_dw'):
            xs[k] = _slice_weight(sw[k], s, SLIDING_PER_GROUP)
    xs['kc'] = _slice_weight(sc['kc'], s, SLIDING_PER_GROUP)
    xs['vc'] = _slice_weight(sc['vc'], s, SLIDING_PER_GROUP)
    xs['kc_s'] = _slice_weight(sc['kc_s'], s, SLIDING_PER_GROUP)
    xs['vc_s'] = _slice_weight(sc['vc_s'], s, SLIDING_PER_GROUP)
    return xs

def _build_global_ws(gw, gc, g):
    """Build weight dict for group g's 1 global layer."""
    ws = {}
    for k in ('qw', 'kw', 'vw', 'ow', 'gw', 'uw', 'dw'):
        ws[k] = gw[k][g]
        ws[k + '_s'] = gw[k + '_s'][g]
    for k in ('qn', 'kn', 'ln1', 'ln2', 'ln3', 'ln4', 'ls'):
        ws[k] = gw[k][g]
    if ENABLE_MOE:
        for k in ('ln3_moe', 'ln4_moe', 'ln4_combine'):
            ws[k] = gw[k][g]
        for k in ('router_w', 'router_s', 'router_ps', 'expert_gu', 'expert_dw'):
            ws[k] = gw[k][g]
    ws['kc'] = gc['kc'][g]
    ws['vc'] = gc['vc'][g]
    ws['kc_s'] = gc['kc_s'][g]
    ws['vc_s'] = gc['vc_s'][g]
    return ws

def _update_sliding_caches(sc, scan_out, g):
    s = g * SLIDING_PER_GROUP
    sc['kc'] = jax.lax.dynamic_update_slice_in_dim(sc['kc'], scan_out['kc'], s, axis=0)
    sc['vc'] = jax.lax.dynamic_update_slice_in_dim(sc['vc'], scan_out['vc'], s, axis=0)
    sc['kc_s'] = jax.lax.dynamic_update_slice_in_dim(sc['kc_s'], scan_out['kc_s'], s, axis=0)
    sc['vc_s'] = jax.lax.dynamic_update_slice_in_dim(sc['vc_s'], scan_out['vc_s'], s, axis=0)
    return sc

def _update_global_caches(gc, kc, vc, kc_s, vc_s, g):
    gc['kc'] = gc['kc'].at[g].set(kc)
    gc['vc'] = gc['vc'].at[g].set(vc)
    gc['kc_s'] = gc['kc_s'].at[g].set(kc_s)
    gc['vc_s'] = gc['vc_s'].at[g].set(vc_s)
    return gc


def forward(token_id, pos, ctx, embed, final_norm,
            sliding_weights, global_weights,
            sliding_caches, global_caches,
            cos_s, sin_s, cos_g, sin_g):
    max_ctx = global_caches['kc'].shape[1]
    x = embed[token_id].reshape(B, H) * jnp.sqrt(jnp.float32(H))

    for g in range(N_GROUPS):
        # 5 sliding layers via scan
        sl_xs = _build_sliding_xs(sliding_weights, sliding_caches, g)
        carry_init = (x, pos, cos_s, sin_s)
        carry_out, scan_out = jax.lax.scan(sliding_one_layer, carry_init, sl_xs)
        x = carry_out[0]
        sliding_caches = _update_sliding_caches(sliding_caches, scan_out, g)

        # 1 global layer
        gl_ws = _build_global_ws(global_weights, global_caches, g)
        x, g_kc, g_vc, g_kc_s, g_vc_s = global_one_layer(
            x, pos, ctx, cos_g, sin_g, max_ctx, gl_ws)
        global_caches = _update_global_caches(global_caches, g_kc, g_vc, g_kc_s, g_vc_s, g)

    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), log_probs, sliding_caches, global_caches


def forward_step(token_id, pos, ctx, embed, final_norm,
                 sliding_weights, global_weights,
                 sliding_caches, global_caches,
                 cos_s, sin_s, cos_g, sin_g):
    max_ctx = global_caches['kc'].shape[1]
    x = embed[token_id].reshape(B, H) * jnp.sqrt(jnp.float32(H))

    for g in range(N_GROUPS):
        sl_xs = _build_sliding_xs(sliding_weights, sliding_caches, g)
        carry_init = (x, pos, cos_s, sin_s)
        carry_out, scan_out = jax.lax.scan(sliding_one_layer, carry_init, sl_xs)
        x = carry_out[0]
        sliding_caches = _update_sliding_caches(sliding_caches, scan_out, g)

        gl_ws = _build_global_ws(global_weights, global_caches, g)
        x, g_kc, g_vc, g_kc_s, g_vc_s = global_one_layer(
            x, pos, ctx, cos_g, sin_g, max_ctx, gl_ws)
        global_caches = _update_global_caches(global_caches, g_kc, g_vc, g_kc_s, g_vc_s, g)

    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), sliding_caches, global_caches


def make_decode_loop(num_prompt, total):
    def decode_loop(prompt_ids, embed, final_norm,
                    sliding_weights, global_weights,
                    sliding_caches, global_caches,
                    cos_s, sin_s, cos_g, sin_g):
        generated = jnp.zeros(total, dtype=jnp.int32)

        init_tok = jnp.broadcast_to(prompt_ids[0:1], (B,))
        init_state = (jnp.int32(0), init_tok, sliding_caches, global_caches, generated)

        def prompt_body(state):
            i, tok, sc, gc, gen = state
            _, sc2, gc2 = forward_step(
                tok, i, i + 1, embed, final_norm,
                sliding_weights, global_weights, sc, gc,
                cos_s, sin_s, cos_g, sin_g)
            next_id = jax.lax.dynamic_slice(prompt_ids, [i + 1], [1])
            next_tok = jnp.where(i + 1 < num_prompt, jnp.broadcast_to(next_id, (B,)), tok)
            return (i + 1, next_tok, sc2, gc2, gen)

        def prompt_cond(state):
            return state[0] < num_prompt

        state = jax.lax.while_loop(prompt_cond, prompt_body, init_state)

        i, tok, sc, gc, gen = state
        first_tok, sc, gc = forward_step(
            tok, i, i + 1, embed, final_norm,
            sliding_weights, global_weights, sc, gc,
            cos_s, sin_s, cos_g, sin_g)
        gen = gen.at[i].set(first_tok[0])
        decode_state = (i + 1, first_tok, sc, gc, gen)

        def decode_body(state):
            i, tok, sc, gc, gen = state
            next_tok, sc2, gc2 = forward_step(
                tok, i, i + 1, embed, final_norm,
                sliding_weights, global_weights, sc, gc,
                cos_s, sin_s, cos_g, sin_g)
            gen = gen.at[i].set(next_tok[0])
            return (i + 1, next_tok, sc2, gc2, gen)

        def decode_cond(state):
            return state[0] < total

        final_state = jax.lax.while_loop(decode_cond, decode_body, decode_state)
        return final_state[4]
    return decode_loop

# ============================================================
# SINGLE-SCAN (UNIFIED) PATH -- fast for max_ctx <= 32768
# ============================================================

def _sliding_attn_unified(q_flat, k_flat, v_flat, qn, kn, cos_s, sin_s, kc, vc, pos, ctx, max_ctx):
    """Sliding attention on unified [max_ctx, MAX_KV] bf16 cache."""
    q = head_norm(q_flat[:, :S_Q].reshape(B, NH, S_HD), qn[:S_HD])
    k = head_norm(k_flat[:, :S_KV].reshape(B, S_KVH, S_HD), kn[:S_HD])
    v = head_norm_noscale(v_flat[:, :S_KV].reshape(B, S_KVH, S_HD))
    c = cos_s[pos][None, None, :]
    s = sin_s[pos][None, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)
    k_val = jnp.pad(k.reshape(B, S_KV)[0], (0, MAX_KV - S_KV)).astype(kc.dtype)
    v_val = jnp.pad(v.reshape(B, S_KV)[0], (0, MAX_KV - S_KV)).astype(vc.dtype)
    kc = kc.at[pos].set(k_val)
    vc = vc.at[pos].set(v_val)
    k_ctx = kc[:max_ctx, :S_KV].reshape(max_ctx, S_KVH, S_HD)
    v_ctx = vc[:max_ctx, :S_KV].reshape(max_ctx, S_KVH, S_HD)
    q_g = q.reshape(B, S_KVH, S_GQA, S_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    t = jnp.arange(max_ctx)
    valid = (t < ctx) & (t >= jnp.maximum(0, pos - WINDOW + 1))
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(B, S_Q)
    return out, kc, vc

def _global_attn_unified(q_flat, k_flat, v_flat, qn, kn, cos_g, sin_g, kc, vc, pos, ctx, max_ctx):
    """Global attention on unified [max_ctx, MAX_KV] bf16 cache."""
    k_raw = k_flat[:, :G_KV].reshape(B, G_KVH, G_HD)
    q = head_norm(q_flat[:, :G_Q].reshape(B, NH, G_HD), qn)
    k = head_norm(k_raw, kn)
    v = head_norm_noscale(k_raw)
    c = cos_g[pos][None, None, :]
    s = sin_g[pos][None, None, :]
    q = rope(q, c, s, 128)
    k = rope(k, c, s, 128)
    k_val = jnp.pad(k.reshape(B, G_KV)[0], (0, MAX_KV - G_KV)).astype(kc.dtype)
    v_val = jnp.pad(v.reshape(B, G_KV)[0], (0, MAX_KV - G_KV)).astype(vc.dtype)
    kc = kc.at[pos].set(k_val)
    vc = vc.at[pos].set(v_val)
    k_ctx = kc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    v_ctx = vc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    q_g = q.reshape(B, G_KVH, G_GQA, G_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    valid = jnp.arange(max_ctx) < ctx
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(B, G_Q)
    return out, kc, vc


def one_layer_unified(carry, xs):
    """Single-scan body: one layer with jax.lax.cond for sliding vs global. bf16 KV cache."""
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
        out, kc2, vc2 = _sliding_attn_unified(q, k, v, qn, kn, cos_s, sin_s, kc, vc, pos, ctx, max_ctx)
        return jnp.pad(out, ((0, 0), (0, MAX_Q - S_Q))).astype(jnp.bfloat16), kc2, vc2

    def do_global(args):
        q, k, v, qn, kn, kc, vc = args
        out, kc2, vc2 = _global_attn_unified(q, k, v, qn, kn, cos_g, sin_g, kc, vc, pos, ctx, max_ctx)
        return out.astype(jnp.bfloat16), kc2, vc2

    attn_out, kc, vc = jax.lax.cond(
        ig, do_global, do_sliding,
        (q_flat, k_flat, v_flat, xs['qn'], xs['kn'], xs['kc'], xs['vc']))

    o_out = int8_matmul(attn_out, xs['ow'], xs['ow_s'])

    h = o_out
    h = rms_norm(h, xs['ln2'])
    x = residual + h

    if ENABLE_MOE:
        residual = x
        # Dense FFN
        h_dense = rms_norm(x, xs['ln3'])
        gate = int8_matmul(h_dense, xs['gw'], xs['gw_s'])
        up = int8_matmul(h_dense, xs['uw'], xs['uw_s'])
        h_dense = jax.nn.gelu(gate, approximate=True) * up
        h_dense = int8_matmul(h_dense, xs['dw'], xs['dw_s'])
        h_dense = rms_norm(h_dense, xs['ln4'])

        # MoE FFN
        h_moe = rms_norm(x, xs['ln3_moe'])
        h_moe = moe_ffn(h_moe, xs['router_w'], xs['router_s'],
                         xs['router_ps'], xs['expert_gu'], xs['expert_dw'])
        h_moe = rms_norm(h_moe, xs['ln4_moe'])

        # Combine dense + MoE + residual, then post-combine norm
        x = residual + h_dense + h_moe
        x = rms_norm(x, xs['ln4_combine'])
        x = x * xs['ls']
    else:
        residual = x
        h = rms_norm(x, xs['ln3'])
        gate = int8_matmul(h, xs['gw'], xs['gw_s'])
        up = int8_matmul(h, xs['uw'], xs['uw_s'])
        h = jax.nn.gelu(gate, approximate=True) * up
        h = int8_matmul(h, xs['dw'], xs['dw_s'])
        h = rms_norm(h, xs['ln4'])
        x = (residual + h) * xs['ls']

    return (x, pos, ctx, cos_s, sin_s, cos_g, sin_g), {'kc': kc, 'vc': vc}


def forward_unified(token_id, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    x = embed[token_id].reshape(B, H) * jnp.sqrt(jnp.float32(H))
    init = (x, pos, ctx, cos_s, sin_s, cos_g, sin_g)
    layer_xs = {**weights, 'kc': caches['kc'], 'vc': caches['vc']}
    final, scan_out = jax.lax.scan(one_layer_unified, init, layer_xs)
    x = final[0]
    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    new_caches = {'kc': scan_out['kc'], 'vc': scan_out['vc']}
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), log_probs, new_caches


def forward_step_unified(token_id, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    x = embed[token_id].reshape(B, H) * jnp.sqrt(jnp.float32(H))
    init = (x, pos, ctx, cos_s, sin_s, cos_g, sin_g)
    layer_xs = {**weights, 'kc': caches['kc'], 'vc': caches['vc']}
    final, scan_out = jax.lax.scan(one_layer_unified, init, layer_xs)
    x = final[0]
    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    new_caches = {'kc': scan_out['kc'], 'vc': scan_out['vc']}
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), new_caches


def make_decode_loop_unified(num_prompt, total):
    def decode_loop(prompt_ids, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
        generated = jnp.zeros(total, dtype=jnp.int32)

        init_tok = jnp.broadcast_to(prompt_ids[0:1], (B,))
        init_state = (jnp.int32(0), init_tok, caches, generated)

        def prompt_body(state):
            i, tok, caches, gen = state
            _, new_caches = forward_step_unified(
                tok, i, i + 1, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
            next_id = jax.lax.dynamic_slice(prompt_ids, [i + 1], [1])
            next_tok = jnp.where(i + 1 < num_prompt, jnp.broadcast_to(next_id, (B,)), tok)
            return (i + 1, next_tok, new_caches, gen)

        def prompt_cond(state):
            return state[0] < num_prompt

        state = jax.lax.while_loop(prompt_cond, prompt_body, init_state)

        i, tok, caches, gen = state
        first_tok, caches = forward_step_unified(
            tok, i, i + 1, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
        gen = gen.at[i].set(first_tok[0])
        decode_state = (i + 1, first_tok, caches, gen)

        def decode_body(state):
            i, tok, caches, gen = state
            next_tok, new_caches = forward_step_unified(
                tok, i, i + 1, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
            gen = gen.at[i].set(next_tok[0])
            return (i + 1, next_tok, new_caches, gen)

        def decode_cond(state):
            return state[0] < total

        final_state = jax.lax.while_loop(decode_cond, decode_body, decode_state)
        return final_state[3]
    return decode_loop


# -- safetensors reader --

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

# -- weight loading --

def pad_to(arr, target_shape):
    pads = [(0, t - s) for s, t in zip(arr.shape, target_shape)]
    return np.pad(arr, pads)

def quantize_int8_perchannel(arr_bf16):
    w = arr_bf16.astype(np.float32)
    amax = np.abs(w).max(axis=-1, keepdims=True).clip(min=1e-10)
    scale = (amax / 127.0).astype(np.float32)
    w_int8 = np.round(w / scale).clip(-127, 127).astype(np.int8)
    return w_int8, scale.squeeze(-1).astype(ml_dtypes.bfloat16)

def _sharded_zeros(shape, dtype, sharding):
    def _cb(idx):
        local_shape = []
        for i, s in enumerate(idx):
            if s.start is None:
                local_shape.append(shape[i])
            else:
                local_shape.append(s.stop - s.start)
        return np.zeros(local_shape, dtype=dtype)
    return jax.make_array_from_callback(shape, sharding, _cb)

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
    else:
        peek_path = os.path.join(model_dir, shard_names[0])
        peek_t = read_safetensors(peek_path)
        for k in peek_t:
            if k.startswith('model.language_model.'):
                prefix = 'model.language_model'
                break
        del peek_t

    print(f"loading from {model_dir}, prefix={prefix}", file=sys.stderr)
    all_t = {}
    for sn in shard_names:
        print(f"  reading {sn}...", file=sys.stderr)
        all_t.update(read_safetensors(os.path.join(model_dir, sn)))
    print(f"  {len(all_t)} tensors", file=sys.stderr)

    def get(name):
        return all_t[name]
    def has(name):
        return name in all_t

    def put(arr, spec):
        return jax.device_put(to_np_bf16(arr), NamedSharding(mesh, spec))

    embed = put(get(f'{prefix}.embed_tokens.weight'), P(None, None))
    final_norm = put(get(f'{prefix}.norm.weight'), P(None))

    # Load all 60 layers, split into sliding (50) and global (10)
    print("  stacking 60 layers -> 50 sliding + 10 global (int8 quantized)...", file=sys.stderr)
    matmul_keys = ['qw','kw','vw','ow','gw','uw','dw']
    bf16_keys = ['qn','kn','ln1','ln2','ln3','ln4','ls']

    # Separate stacks for sliding and global
    sl_i8 = {k: [] for k in matmul_keys}
    sl_sc = {k+'_s': [] for k in matmul_keys}
    sl_bf = {k: [] for k in bf16_keys}

    gl_i8 = {k: [] for k in matmul_keys}
    gl_sc = {k+'_s': [] for k in matmul_keys}
    gl_bf = {k: [] for k in bf16_keys}

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

        gw_l = to_np_bf16(get(f'{lp}.mlp.gate_proj.weight'))
        uw_l = to_np_bf16(get(f'{lp}.mlp.up_proj.weight'))
        dw_l = to_np_bf16(get(f'{lp}.mlp.down_proj.weight'))

        raw = {'qw': pad_to(qw, (MAX_Q, H)), 'kw': pad_to(kw, (MAX_KV, H)),
               'vw': pad_to(vw, (MAX_KV, H)), 'ow': pad_to(ow, (H, MAX_O)),
               'gw': gw_l, 'uw': uw_l, 'dw': dw_l}

        target_i8 = gl_i8 if is_global else sl_i8
        target_sc = gl_sc if is_global else sl_sc
        target_bf = gl_bf if is_global else sl_bf

        for k in matmul_keys:
            w_i8, sc = quantize_int8_perchannel(raw[k])
            target_i8[k].append(w_i8)
            target_sc[k+'_s'].append(sc)

        target_bf['qn'].append(pad_to(to_np_bf16(get(f'{lp}.self_attn.q_norm.weight')), (MAX_NORM_HD,)))
        target_bf['kn'].append(pad_to(to_np_bf16(get(f'{lp}.self_attn.k_norm.weight')), (MAX_NORM_HD,)))
        target_bf['ln1'].append(to_np_bf16(get(f'{lp}.input_layernorm.weight')))
        target_bf['ln2'].append(to_np_bf16(get(f'{lp}.post_attention_layernorm.weight')))
        target_bf['ln3'].append(to_np_bf16(get(f'{lp}.pre_feedforward_layernorm.weight')))
        target_bf['ln4'].append(to_np_bf16(get(f'{lp}.post_feedforward_layernorm.weight')))
        target_bf['ls'].append(to_np_bf16(get(f'{lp}.layer_scalar')) if has(f'{lp}.layer_scalar') else np.array([1.0], dtype=ml_dtypes.bfloat16))

        if i % 15 == 0:
            print(f"    layer {i} ({'global' if is_global else 'sliding'})", file=sys.stderr)

    # Sharding specs
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

    # Build sliding_weights [50, ...]
    print(f"  building sliding_weights ({N_SLIDING} layers)...", file=sys.stderr)
    sliding_weights = {}
    for k in matmul_keys:
        arr = np.stack(sl_i8[k])
        sliding_weights[k] = put_i8(arr, i8_sharding[k])
        print(f"    sl.{k}: {arr.shape} int8", file=sys.stderr)
        sc_arr = np.stack(sl_sc[k+'_s'])
        sliding_weights[k+'_s'] = put(sc_arr, sc_sharding[k+'_s'])
    for k in bf16_keys:
        arr = np.stack(sl_bf[k])
        sliding_weights[k] = put(arr, P(None, None))
        print(f"    sl.{k}: {arr.shape} bf16", file=sys.stderr)

    # Build global_weights [10, ...]
    print(f"  building global_weights ({N_GLOBAL} layers)...", file=sys.stderr)
    global_weights = {}
    for k in matmul_keys:
        arr = np.stack(gl_i8[k])
        global_weights[k] = put_i8(arr, i8_sharding[k])
        print(f"    gl.{k}: {arr.shape} int8", file=sys.stderr)
        sc_arr = np.stack(gl_sc[k+'_s'])
        global_weights[k+'_s'] = put(sc_arr, sc_sharding[k+'_s'])
    for k in bf16_keys:
        arr = np.stack(gl_bf[k])
        global_weights[k] = put(arr, P(None, None))
        print(f"    gl.{k}: {arr.shape} bf16", file=sys.stderr)

    # Sliding caches: [50, WINDOW, dim] -- tiny
    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    kvs_sh = NamedSharding(mesh, P(None, None, None))
    sl_mem = N_SLIDING * WINDOW * S_KV * 1 * 2  # int8, k+v
    print(f"  sliding caches: {N_SLIDING} x {WINDOW} x {S_KV} = {sl_mem/1e6:.0f}MB", file=sys.stderr)
    sliding_caches = {
        'kc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
    }

    # Global caches: [10, max_ctx, dim] -- the big one but 6x smaller than before
    gl_mem = N_GLOBAL * max_ctx * G_KV * 1 * 2
    print(f"  global caches: {N_GLOBAL} x {max_ctx} x {G_KV} = {gl_mem/1e6:.0f}MB", file=sys.stderr)
    global_caches = {
        'kc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
    }

    total_mem = sl_mem + gl_mem
    print(f"  total KV cache: {total_mem/1e6:.0f}MB (was {NL*max_ctx*MAX_KV*2/1e6:.0f}MB)", file=sys.stderr)

    del all_t, sl_i8, sl_sc, sl_bf, gl_i8, gl_sc, gl_bf
    print("  done loading", file=sys.stderr)
    return embed, final_norm, sliding_weights, global_weights, sliding_caches, global_caches


def load_model_unified(model_dir, mesh, max_ctx):
    """Load all 60 layers as [60, ...] unified stacks for single-scan path."""
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
    else:
        # Single safetensors file -- peek at keys to detect prefix
        peek_path = os.path.join(model_dir, shard_names[0])
        peek_t = read_safetensors(peek_path)
        for k in peek_t:
            if k.startswith('model.language_model.'):
                prefix = 'model.language_model'
                break
        del peek_t

    print(f"loading from {model_dir}, prefix={prefix} (unified)", file=sys.stderr)
    all_t = {}
    for sn in shard_names:
        print(f"  reading {sn}...", file=sys.stderr)
        all_t.update(read_safetensors(os.path.join(model_dir, sn)))
    print(f"  {len(all_t)} tensors", file=sys.stderr)

    def get(name):
        return all_t[name]
    def has(name):
        return name in all_t

    def put(arr, spec):
        return jax.device_put(to_np_bf16(arr), NamedSharding(mesh, spec))

    embed = put(get(f'{prefix}.embed_tokens.weight'), P(None, None))
    final_norm = put(get(f'{prefix}.norm.weight'), P(None))

    print(f"  stacking {NL} layers (padded, int8 quantized, unified)...", file=sys.stderr)
    matmul_keys = ['qw','kw','vw','ow','gw','uw','dw']
    bf16_keys = ['qn','kn','ln1','ln2','ln3','ln4','ls']
    if ENABLE_MOE:
        bf16_keys += ['ln3_moe', 'ln4_moe', 'ln4_combine']
    stacked_i8 = {k: [] for k in matmul_keys}
    stacked_sc = {k+'_s': [] for k in matmul_keys}
    stacked_bf = {k: [] for k in bf16_keys}
    stacked_ig = []
    # MoE weight stacks (bf16, NOT int8 -- sparse access)
    moe_keys = ['router_w', 'router_s', 'router_ps', 'expert_gu', 'expert_dw']
    stacked_moe = {k: [] for k in moe_keys} if ENABLE_MOE else {}

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

        gw_l = to_np_bf16(get(f'{lp}.mlp.gate_proj.weight'))
        uw_l = to_np_bf16(get(f'{lp}.mlp.up_proj.weight'))
        dw_l = to_np_bf16(get(f'{lp}.mlp.down_proj.weight'))

        raw = {'qw': pad_to(qw, (MAX_Q, H)), 'kw': pad_to(kw, (MAX_KV, H)),
               'vw': pad_to(vw, (MAX_KV, H)), 'ow': pad_to(ow, (H, MAX_O)),
               'gw': gw_l, 'uw': uw_l, 'dw': dw_l}
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
        stacked_bf['ls'].append(to_np_bf16(get(f'{lp}.layer_scalar')) if has(f'{lp}.layer_scalar') else np.array([1.0], dtype=ml_dtypes.bfloat16))
        stacked_ig.append(np.array(is_global, dtype=np.int32))

        if ENABLE_MOE:
            stacked_bf['ln3_moe'].append(to_np_bf16(get(f'{lp}.pre_feedforward_layernorm_2.weight')))
            stacked_bf['ln4_moe'].append(to_np_bf16(get(f'{lp}.post_feedforward_layernorm_1.weight')))
            stacked_bf['ln4_combine'].append(to_np_bf16(get(f'{lp}.post_feedforward_layernorm_2.weight')))
            stacked_moe['router_w'].append(to_np_bf16(get(f'{lp}.router.proj.weight')))
            # router.scale is a scalar -- read and wrap as 1-elem array for stacking
            rs = to_np_bf16(get(f'{lp}.router.scale'))
            stacked_moe['router_s'].append(rs.reshape(1) if rs.ndim == 0 else rs)
            stacked_moe['router_ps'].append(to_np_bf16(get(f'{lp}.router.per_expert_scale')))
            stacked_moe['expert_gu'].append(to_np_bf16(get(f'{lp}.experts.gate_up_proj')))
            stacked_moe['expert_dw'].append(to_np_bf16(get(f'{lp}.experts.down_proj')))

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
        print(f"    {k}: {arr.shape} bf16", file=sys.stderr)
    weights['ig'] = jax.device_put(jnp.array(np.array(stacked_ig)), NamedSharding(mesh, P(None)))

    if ENABLE_MOE:
        # MoE weights: replicated on all chips (NOT TP-sharded -- gathered sparsely)
        rep = NamedSharding(mesh, P(None, None))
        rep3 = NamedSharding(mesh, P(None, None, None))
        rep4 = NamedSharding(mesh, P(None, None, None, None))
        for k in moe_keys:
            arr = np.stack(stacked_moe[k])
            ndim = arr.ndim
            if ndim == 2:
                weights[k] = jax.device_put(jnp.array(arr, dtype=jnp.bfloat16), rep)
            elif ndim == 3:
                weights[k] = jax.device_put(jnp.array(arr, dtype=jnp.bfloat16), rep3)
            elif ndim == 4:
                weights[k] = jax.device_put(jnp.array(arr, dtype=jnp.bfloat16), rep4)
            print(f"    moe.{k}: {arr.shape} bf16", file=sys.stderr)

    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    kv_mem = NL * max_ctx * MAX_KV * 2 * 2
    print(f"  unified KV cache: {NL} x {max_ctx} x {MAX_KV} bf16 = {kv_mem/1e6:.0f}MB", file=sys.stderr)
    caches = {
        'kc': _sharded_zeros((NL, max_ctx, MAX_KV), ml_dtypes.bfloat16, kv_sh),
        'vc': _sharded_zeros((NL, max_ctx, MAX_KV), ml_dtypes.bfloat16, kv_sh),
    }

    del all_t, stacked_i8, stacked_sc, stacked_bf
    if ENABLE_MOE:
        del stacked_moe
    print("  done loading (unified)", file=sys.stderr)
    return embed, final_norm, weights, caches


# -- main --

def load_tokenizer(model_dir):
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tok_path)
    return None

def run_perplexity(args, mesh, embed, final_norm, sliding_weights, global_weights,
                   sliding_caches, global_caches, cos_s, sin_s, cos_g, sin_g):
    tokenizer = load_tokenizer(args.model_dir)
    if tokenizer is None:
        print("ERROR: no tokenizer.json found", file=sys.stderr)
        return

    if args.ppl_file:
        with open(args.ppl_file) as f:
            text = f.read()
    else:
        text = "The quick brown fox jumps over the lazy dog. In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God. All things were made through him, and without him was not any thing made that was made. In him was life, and the life was the light of men. The light shines in the darkness, and the darkness has not overcome it."

    token_ids = tokenizer.encode(text).ids
    max_tokens = min(len(token_ids), args.max_ctx - 1)
    token_ids = [2] + token_ids[:max_tokens]
    print(f"perplexity eval: {len(token_ids)} tokens", file=sys.stderr)

    fwd_jit = jax.jit(forward, donate_argnums=(7, 8))
    total_nll = 0.0
    n_tokens = 0
    t_start = time.time()

    for step in range(len(token_ids) - 1):
        tok_arr = jnp.array([token_ids[step]], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)

        next_tok, log_probs, sliding_caches, global_caches = fwd_jit(
            tok_arr, pos, ctx, embed, final_norm,
            sliding_weights, global_weights,
            sliding_caches, global_caches,
            cos_s, sin_s, cos_g, sin_g)

        target = token_ids[step + 1]
        nll = -float(log_probs[0, target])
        total_nll += nll
        n_tokens += 1

        if step % 50 == 0:
            ppl_so_far = float(np.exp(total_nll / max(n_tokens, 1)))
            elapsed = time.time() - t_start
            tps = n_tokens / max(elapsed, 0.01)
            print(f"  step {step}/{len(token_ids)-1} nll={nll:.3f} ppl={ppl_so_far:.2f} ({tps:.1f} tok/s)", file=sys.stderr)

    avg_nll = total_nll / n_tokens
    ppl = float(np.exp(avg_nll))
    elapsed = time.time() - t_start
    print(file=sys.stderr)
    print(f"=== Perplexity ===", file=sys.stderr)
    print(f"tokens:     {n_tokens}", file=sys.stderr)
    print(f"avg NLL:    {avg_nll:.4f}", file=sys.stderr)
    print(f"perplexity: {ppl:.2f}", file=sys.stderr)
    print(f"time:       {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)", file=sys.stderr)

def run_fused(args, mesh, embed, final_norm, sliding_weights, global_weights,
              sliding_caches, global_caches, cos_s, sin_s, cos_g, sin_g):
    prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    num_prompt = len(prompt_ids)
    num_decode = args.max_tokens
    total = num_prompt + num_decode
    print(f"fused decode: {num_prompt} prompt + {num_decode} decode = {total} steps", file=sys.stderr)

    prompt_arr = jnp.array(prompt_ids + [0] * (total - num_prompt), dtype=jnp.int32)

    loop_fn = make_decode_loop(num_prompt, total)
    fused_jit = jax.jit(loop_fn, donate_argnums=(5, 6))

    print("compiling fused loop...", file=sys.stderr, flush=True)
    t0 = time.time()
    gen = fused_jit(prompt_arr, embed, final_norm,
                    sliding_weights, global_weights,
                    sliding_caches, global_caches,
                    cos_s, sin_s, cos_g, sin_g)
    gen.block_until_ready()
    compile_time = time.time() - t0
    print(f"compiled + ran in {compile_time:.1f}s", file=sys.stderr)

    tokens = np.array(gen)
    decode_tokens = tokens[num_prompt:total]
    print(f"\n=== Fused Decode ===", file=sys.stderr)
    print(f"compile+run:  {compile_time:.2f}s", file=sys.stderr)
    print(f"total steps:  {total}", file=sys.stderr)
    if compile_time > 0:
        print(f"tok/s (incl compile): {total/compile_time:.1f}", file=sys.stderr)
    wall_per_tok = compile_time / total * 1000
    print(f"ms/token (incl compile): {wall_per_tok:.1f}", file=sys.stderr)
    print(f"generated: {decode_tokens[:20].tolist()}", file=sys.stderr)

    print("\nre-running (cached compile)...", file=sys.stderr, flush=True)
    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    kvs_sh = NamedSharding(mesh, P(None, None, None))
    sliding_caches2 = {
        'kc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
    }
    global_caches2 = {
        'kc': _sharded_zeros((N_GLOBAL, args.max_ctx, G_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_GLOBAL, args.max_ctx, G_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_GLOBAL, args.max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_GLOBAL, args.max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
    }
    t0 = time.time()
    gen2 = fused_jit(prompt_arr, embed, final_norm,
                     sliding_weights, global_weights,
                     sliding_caches2, global_caches2,
                     cos_s, sin_s, cos_g, sin_g)
    gen2.block_until_ready()
    pure_time = time.time() - t0
    print(f"pure run:     {pure_time:.3f}s", file=sys.stderr)
    print(f"steps/s:      {total/pure_time:.1f}", file=sys.stderr)
    print(f"tok/s:        {total*B/pure_time:.1f} (B={B})", file=sys.stderr)
    print(f"ms/step:      {pure_time/total*1000:.2f}", file=sys.stderr)

def run_generate(args, mesh, embed, final_norm, sliding_weights, global_weights,
                 sliding_caches, global_caches, cos_s, sin_s, cos_g, sin_g):
    prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    print(f"prompt: {len(prompt_ids)} tokens {prompt_ids[:10]}", file=sys.stderr)

    fwd_jit = jax.jit(forward, donate_argnums=(7, 8))
    generated = []
    last_sampled = None
    total_steps = len(prompt_ids) + args.max_tokens
    t_start = time.time()
    ttft = None

    for step in range(total_steps):
        if step < len(prompt_ids):
            token_id = prompt_ids[step]
        else:
            token_id = last_sampled
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        tok_arr = jnp.array([token_id] * B, dtype=jnp.int32)

        t0 = time.time()
        next_tok, _log_probs, sliding_caches, global_caches = fwd_jit(
            tok_arr, pos, ctx, embed, final_norm,
            sliding_weights, global_weights,
            sliding_caches, global_caches,
            cos_s, sin_s, cos_g, sin_g)
        next_tok.block_until_ready()
        dt = time.time() - t0

        sampled = int(next_tok[0])
        last_sampled = sampled

        if step < len(prompt_ids):
            print('.', end='', file=sys.stderr, flush=True)
            if step == len(prompt_ids) - 1:
                ttft = time.time() - t_start
                print(f"\nTTFT: {ttft*1000:.1f}ms ({len(prompt_ids)} prompt tokens)", file=sys.stderr)
        else:
            generated.append(sampled)
            if sampled in (1, 2, 107):
                print(f"\n[EOS tok={sampled} at step {step}]", file=sys.stderr)
                break
            print(f"[{sampled}]", end='', file=sys.stderr, flush=True)
            if step < len(prompt_ids) + 3:
                print(f" ({dt*1000:.1f}ms)", end='', file=sys.stderr, flush=True)

    total = time.time() - t_start
    print(file=sys.stderr)
    print("=== Results ===", file=sys.stderr)
    print(f"prompt tokens:    {len(prompt_ids)}", file=sys.stderr)
    print(f"generated tokens: {len(generated)}", file=sys.stderr)
    if ttft:
        print(f"TTFT:             {ttft*1000:.1f}ms", file=sys.stderr)
    if len(generated) > 1 and ttft:
        decode_time = total - ttft
        tps = len(generated) / decode_time
        print(f"decode tok/s:     {tps:.1f}", file=sys.stderr)
        print(f"ms/token:         {decode_time/len(generated)*1000:.1f}", file=sys.stderr)
    print(f"total time:       {total:.1f}s", file=sys.stderr)
    print(f"generated:        {generated[:20]}", file=sys.stderr)

def run_perplexity_unified(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    tokenizer = load_tokenizer(args.model_dir)
    if tokenizer is None:
        print("ERROR: no tokenizer.json found", file=sys.stderr)
        return

    if args.ppl_file:
        with open(args.ppl_file) as f:
            text = f.read()
    else:
        text = "The quick brown fox jumps over the lazy dog. In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God. All things were made through him, and without him was not any thing made that was made. In him was life, and the life was the light of men. The light shines in the darkness, and the darkness has not overcome it."

    token_ids = tokenizer.encode(text).ids
    max_tokens = min(len(token_ids), args.max_ctx - 1)
    token_ids = [2] + token_ids[:max_tokens]
    print(f"perplexity eval: {len(token_ids)} tokens (unified)", file=sys.stderr)

    fwd_jit = jax.jit(forward_unified, donate_argnums=(6,))
    total_nll = 0.0
    n_tokens = 0
    t_start = time.time()

    for step in range(len(token_ids) - 1):
        tok_arr = jnp.array([token_ids[step]], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)

        next_tok, log_probs, caches = fwd_jit(
            tok_arr, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)

        target = token_ids[step + 1]
        nll = -float(log_probs[0, target])
        total_nll += nll
        n_tokens += 1

        if step % 50 == 0:
            ppl_so_far = float(np.exp(total_nll / max(n_tokens, 1)))
            elapsed = time.time() - t_start
            tps = n_tokens / max(elapsed, 0.01)
            print(f"  step {step}/{len(token_ids)-1} nll={nll:.3f} ppl={ppl_so_far:.2f} ({tps:.1f} tok/s)", file=sys.stderr)

    avg_nll = total_nll / n_tokens
    ppl = float(np.exp(avg_nll))
    elapsed = time.time() - t_start
    print(file=sys.stderr)
    print(f"=== Perplexity (unified) ===", file=sys.stderr)
    print(f"tokens:     {n_tokens}", file=sys.stderr)
    print(f"avg NLL:    {avg_nll:.4f}", file=sys.stderr)
    print(f"perplexity: {ppl:.2f}", file=sys.stderr)
    print(f"time:       {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)", file=sys.stderr)

def run_fused_unified(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    num_prompt = len(prompt_ids)
    num_decode = args.max_tokens
    total = num_prompt + num_decode
    print(f"fused decode (unified): {num_prompt} prompt + {num_decode} decode = {total} steps", file=sys.stderr)

    prompt_arr = jnp.array(prompt_ids + [0] * (total - num_prompt), dtype=jnp.int32)

    loop_fn = make_decode_loop_unified(num_prompt, total)
    fused_jit = jax.jit(loop_fn, donate_argnums=(4,))

    print("compiling fused loop (unified)...", file=sys.stderr, flush=True)
    t0 = time.time()
    gen = fused_jit(prompt_arr, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
    gen.block_until_ready()
    compile_time = time.time() - t0
    print(f"compiled + ran in {compile_time:.1f}s", file=sys.stderr)

    tokens = np.array(gen)
    decode_tokens = tokens[num_prompt:total]
    print(f"\n=== Fused Decode (unified) ===", file=sys.stderr)
    print(f"compile+run:  {compile_time:.2f}s", file=sys.stderr)
    print(f"total steps:  {total}", file=sys.stderr)
    if compile_time > 0:
        print(f"tok/s (incl compile): {total/compile_time:.1f}", file=sys.stderr)
    wall_per_tok = compile_time / total * 1000
    print(f"ms/token (incl compile): {wall_per_tok:.1f}", file=sys.stderr)
    print(f"generated: {decode_tokens[:20].tolist()}", file=sys.stderr)

    print("\nre-running (cached compile)...", file=sys.stderr, flush=True)
    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    caches2 = {
        'kc': _sharded_zeros((NL, args.max_ctx, MAX_KV), ml_dtypes.bfloat16, kv_sh),
        'vc': _sharded_zeros((NL, args.max_ctx, MAX_KV), ml_dtypes.bfloat16, kv_sh),
    }
    t0 = time.time()
    gen2 = fused_jit(prompt_arr, embed, final_norm, weights, caches2, cos_s, sin_s, cos_g, sin_g)
    gen2.block_until_ready()
    pure_time = time.time() - t0
    print(f"pure run:     {pure_time:.3f}s", file=sys.stderr)
    print(f"steps/s:      {total/pure_time:.1f}", file=sys.stderr)
    print(f"tok/s:        {total*B/pure_time:.1f} (B={B})", file=sys.stderr)
    print(f"ms/step:      {pure_time/total*1000:.2f}", file=sys.stderr)

def run_generate_unified(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    print(f"prompt: {len(prompt_ids)} tokens {prompt_ids[:10]} (unified)", file=sys.stderr)

    fwd_jit = jax.jit(forward_unified, donate_argnums=(6,))
    generated = []
    last_sampled = None
    total_steps = len(prompt_ids) + args.max_tokens
    t_start = time.time()
    ttft = None

    for step in range(total_steps):
        if step < len(prompt_ids):
            token_id = prompt_ids[step]
        else:
            token_id = last_sampled
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        tok_arr = jnp.array([token_id] * B, dtype=jnp.int32)

        t0 = time.time()
        next_tok, _log_probs, caches = fwd_jit(
            tok_arr, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
        next_tok.block_until_ready()
        dt = time.time() - t0

        sampled = int(next_tok[0])
        last_sampled = sampled

        if step < len(prompt_ids):
            print('.', end='', file=sys.stderr, flush=True)
            if step == len(prompt_ids) - 1:
                ttft = time.time() - t_start
                print(f"\nTTFT: {ttft*1000:.1f}ms ({len(prompt_ids)} prompt tokens)", file=sys.stderr)
        else:
            generated.append(sampled)
            if sampled in (1, 2, 107):
                print(f"\n[EOS tok={sampled} at step {step}]", file=sys.stderr)
                break
            print(f"[{sampled}]", end='', file=sys.stderr, flush=True)
            if step < len(prompt_ids) + 3:
                print(f" ({dt*1000:.1f}ms)", end='', file=sys.stderr, flush=True)

    total = time.time() - t_start
    print(file=sys.stderr)
    print("=== Results (unified) ===", file=sys.stderr)
    print(f"prompt tokens:    {len(prompt_ids)}", file=sys.stderr)
    print(f"generated tokens: {len(generated)}", file=sys.stderr)
    if ttft:
        print(f"TTFT:             {ttft*1000:.1f}ms", file=sys.stderr)
    if len(generated) > 1 and ttft:
        decode_time = total - ttft
        tps = len(generated) / decode_time
        print(f"decode tok/s:     {tps:.1f}", file=sys.stderr)
        print(f"ms/token:         {decode_time/len(generated)*1000:.1f}", file=sys.stderr)
    print(f"total time:       {total:.1f}s", file=sys.stderr)
    print(f"generated:        {generated[:20]}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--max-ctx', type=int, default=131072)
    parser.add_argument('--prompt', default='2')
    parser.add_argument('--perplexity', action='store_true')
    parser.add_argument('--ppl-file', default=None)
    parser.add_argument('--fused', action='store_true', help='On-chip decode loop (zero host overhead)')
    parser.add_argument('--batch', type=int, default=1, help='Batch size (lockstep decode)')
    args = parser.parse_args()

    global B
    B = args.batch

    load_config(args.model_dir)

    mesh = make_mesh()
    print(f"mesh: {mesh}", file=sys.stderr)

    max_ctx = args.max_ctx

    if max_ctx <= SPLIT_THRESHOLD:
        # -- single-scan (unified) path: fast for short contexts --
        print(f"using single-scan architecture (fast path, max_ctx={max_ctx})", file=sys.stderr)
        embed, final_norm, weights, caches = load_model_unified(args.model_dir, mesh, max_ctx)

        cos_s, sin_s = precompute_rope(10000.0, S_HD, max_ctx)
        cos_g, sin_g = precompute_rope(1000000.0, 128, max_ctx)
        cos_s = jax.device_put(jnp.array(cos_s), NamedSharding(mesh, P(None, None)))
        sin_s = jax.device_put(jnp.array(sin_s), NamedSharding(mesh, P(None, None)))
        cos_g = jax.device_put(jnp.array(cos_g), NamedSharding(mesh, P(None, None)))
        sin_g = jax.device_put(jnp.array(sin_g), NamedSharding(mesh, P(None, None)))

        if args.perplexity:
            run_perplexity_unified(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
        elif args.fused:
            run_fused_unified(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
        else:
            run_generate_unified(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
    else:
        # -- split-cache path: fits 128K --
        print(f"using split-cache architecture (128K path, max_ctx={max_ctx})", file=sys.stderr)
        if max_ctx % BLOCK_K != 0:
            max_ctx = ((max_ctx + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
            print(f"  rounded max_ctx to {max_ctx} (multiple of BLOCK_K={BLOCK_K})", file=sys.stderr)
        embed, final_norm, sliding_weights, global_weights, sliding_caches, global_caches = load_model(args.model_dir, mesh, max_ctx)

        cos_s, sin_s = precompute_rope(10000.0, S_HD, max_ctx)
        cos_g, sin_g = precompute_rope(1000000.0, 128, max_ctx)
        cos_s = jax.device_put(jnp.array(cos_s), NamedSharding(mesh, P(None, None)))
        sin_s = jax.device_put(jnp.array(sin_s), NamedSharding(mesh, P(None, None)))
        cos_g = jax.device_put(jnp.array(cos_g), NamedSharding(mesh, P(None, None)))
        sin_g = jax.device_put(jnp.array(sin_g), NamedSharding(mesh, P(None, None)))

        if args.perplexity:
            run_perplexity(args, mesh, embed, final_norm,
                           sliding_weights, global_weights,
                           sliding_caches, global_caches,
                           cos_s, sin_s, cos_g, sin_g)
        elif args.fused:
            run_fused(args, mesh, embed, final_norm,
                      sliding_weights, global_weights,
                      sliding_caches, global_caches,
                      cos_s, sin_s, cos_g, sin_g)
        else:
            run_generate(args, mesh, embed, final_norm,
                         sliding_weights, global_weights,
                         sliding_caches, global_caches,
                         cos_s, sin_s, cos_g, sin_g)

if __name__ == '__main__':
    main()
