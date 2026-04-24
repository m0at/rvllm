"""Gather-top-K MoE block for MiniMax-M2.7-NVFP4 (Agents 1+2 combined).

Per-step flow at B=1:
  router_sigmoid_topk -> topk_idx shape (B, TOP_K=8)
  Each shard holds 32 experts locally (256 / 8 chips).
  For each selected expert, dynamic_slice out its NVFP4 packed + scales.
  Stack the K selected experts' weights into (K, ...) leading-batched tensors.
  Compute gate/up/down with ONE batched nvfp4_matmul per projection — XLA/MXU
  fuses the K independent matmuls into a single kernel launch instead of
  K separate per-expert calls.

Key insight: XLA doesn't skip compute on masked values, but dynamic_slice
DOES skip the unselected experts' HBM reads (Agent 2). Only the K chosen
experts' weight bytes are loaded, not all 32 local experts.

This is the Gemma4-style dispatch adapted for NVFP4 sharded-expert layout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from nvfp4_jax_ops import nvfp4_matmul

NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 1536
H = 3072


def router_sigmoid_topk(x, router_w, router_bias):
    """sigmoid + bias router, top-K selection with renormalized weights."""
    logits = x.astype(jnp.float32) @ router_w.T.astype(jnp.float32)
    biased = logits + router_bias.astype(jnp.float32)
    _, topk_idx = jax.lax.top_k(biased, TOP_K)
    gathered = jnp.take_along_axis(logits, topk_idx, axis=1)
    weights = gathered / (gathered.sum(axis=-1, keepdims=True) + 1e-9)
    return weights.astype(x.dtype), topk_idx


def _nvfp4_matmul_batched(x, wp, ws, ws2, out_feat, in_feat):
    """Batched NVFP4 matmul over leading axis K.

    x:   (K, B, in_feat) bf16
    wp:  (K, out_feat, in_feat/2) uint8
    ws:  (K, out_feat, in_feat/16) uint8
    ws2: (K,) f32
    Returns (K, B, out_feat) bf16.

    vmap over the first axis. Each call to nvfp4_matmul is over (B, in_feat)
    against a single expert's weight — XLA fuses the K calls into one MXU
    batched GEMM where K becomes an outer batch dim.
    """
    def one(w_packed, w_scales, gs, xb):
        return nvfp4_matmul(xb, w_packed, w_scales, gs, out_feat, in_feat)
    return jax.vmap(one, in_axes=(0, 0, 0, 0))(wp, ws, ws2, x)


def moe_block_gather_topk(x, router_w, router_bias, w1, w2, w3, mesh):
    """Gather top-K path: each shard gathers the 0-K experts selected by this
    token that happen to live on this shard, then computes those in a single
    batched nvfp4_matmul call per projection.

    Args:
        x: (B, H) bf16, replicated
        router_w: (NUM_EXPERTS, H) bf16, replicated
        router_bias: (NUM_EXPERTS,) bf16, replicated
        w1, w3: tuple(packed (E, MI, H/2) u8, scales (E, MI, H/16) u8, scale2 (E,) f32)
                sharded P('expert', None, None) and P('expert')
        w2: tuple(packed (E, H, MI/2) u8, scales (E, H, MI/16) u8, scale2 (E,) f32)
        mesh: Mesh with 'expert' axis.

    Returns:
        (B, H) bf16 replicated (via psum).
    """
    # Router replicated.
    topk_w, topk_idx = router_sigmoid_topk(x, router_w, router_bias)  # (B,K),(B,K)

    w1_p, w1_s, w1_s2 = w1
    w2_p, w2_s, w2_s2 = w2
    w3_p, w3_s, w3_s2 = w3

    has_expert_axis = (mesh is not None and 'expert' in mesh.axis_names)
    if not has_expert_axis:
        # Single-device: just gather and compute
        return _gather_topk_singledev(
            x, topk_w, topk_idx, w1_p, w1_s, w1_s2, w2_p, w2_s, w2_s2, w3_p, w3_s, w3_s2)

    in_specs = (
        P(),                                # x (B, H)
        P(),                                # topk_w (B, K)
        P(),                                # topk_idx (B, K)
        P('expert', None, None),            # w1_p sharded
        P('expert', None, None),            # w1_s
        P('expert'),                        # w1_s2
        P('expert', None, None),            # w2_p
        P('expert', None, None),            # w2_s
        P('expert'),                        # w2_s2
        P('expert', None, None),            # w3_p
        P('expert', None, None),            # w3_s
        P('expert'),                        # w3_s2
    )
    out_specs = P()  # replicated via psum

    def _local(x_l, topk_w_l, topk_idx_l,
               w1p_l, w1s_l, w1s2_l,
               w2p_l, w2s_l, w2s2_l,
               w3p_l, w3s_l, w3s2_l):
        """
        Local shard view:
            x_l: (B, H)
            topk_idx_l: (B, K) global expert ids
            w*p_l, w*s_l: (EPS, ...) where EPS = 256/n_shards local experts
            w*s2_l: (EPS,)
        """
        shard_id = jax.lax.axis_index('expert')
        n_shards = jax.lax.psum(1, 'expert')
        eps = NUM_EXPERTS // n_shards        # experts per shard (static)
        base = shard_id * eps

        # Compute LOCAL expert index (-1 if not on this shard) and validity mask.
        local_idx = topk_idx_l - base                    # (B, K)
        valid = (local_idx >= 0) & (local_idx < eps)     # (B, K)
        # Clamp out-of-range to 0 so `take` still works; multiply by valid later.
        local_idx_clamped = jnp.where(valid, local_idx, 0)

        B_ = x_l.shape[0]
        K = TOP_K
        flat_local_idx = local_idx_clamped.reshape(B_ * K)  # (B*K,)
        flat_valid = valid.reshape(B_ * K).astype(x_l.dtype)  # (B*K,)
        flat_weights = (topk_w_l.reshape(B_ * K).astype(x_l.dtype)
                        * flat_valid)                         # (B*K,) zero-out invalid

        # Gather the (up to B*K) expert weights from local experts.
        # Shape (B*K, MI, H/2) packed, (B*K, MI, H/16) scales, (B*K,) scale2
        w1p_g = jnp.take(w1p_l, flat_local_idx, axis=0)
        w1s_g = jnp.take(w1s_l, flat_local_idx, axis=0)
        w1s2_g = jnp.take(w1s2_l, flat_local_idx, axis=0)
        w3p_g = jnp.take(w3p_l, flat_local_idx, axis=0)
        w3s_g = jnp.take(w3s_l, flat_local_idx, axis=0)
        w3s2_g = jnp.take(w3s2_l, flat_local_idx, axis=0)
        w2p_g = jnp.take(w2p_l, flat_local_idx, axis=0)
        w2s_g = jnp.take(w2s_l, flat_local_idx, axis=0)
        w2s2_g = jnp.take(w2s2_l, flat_local_idx, axis=0)

        # Replicate x so each of the B*K slots gets the right token.
        # At B=1 this is K copies of the same x row.
        token_idx = jnp.repeat(jnp.arange(B_), K)    # (B*K,) which token
        x_exp = x_l[token_idx]                        # (B*K, H)

        # Batched expert matmuls. vmap over leading (B*K) axis.
        def _one_expert(xi, p1, s1, g1, p3, s3, g3, p2, s2, g2):
            gate = nvfp4_matmul(xi[None], p1, s1, g1, MOE_INTER, H)[0]
            up = nvfp4_matmul(xi[None], p3, s3, g3, MOE_INTER, H)[0]
            h = jax.nn.silu(gate) * up
            out = nvfp4_matmul(h[None], p2, s2, g2, H, MOE_INTER)[0]
            return out

        per_slot_out = jax.vmap(_one_expert)(
            x_exp,
            w1p_g, w1s_g, w1s2_g,
            w3p_g, w3s_g, w3s2_g,
            w2p_g, w2s_g, w2s2_g,
        )  # (B*K, H)

        # Weight and mask out invalid slots.
        weighted = per_slot_out * flat_weights[:, None]  # (B*K, H)

        # Scatter-add back to per-token positions.
        out = jnp.zeros((B_, H), dtype=x_l.dtype)
        out = out.at[token_idx].add(weighted)

        # Cross-shard reduce: each shard contributes its slots' output.
        return jax.lax.psum(out, 'expert')

    return shard_map(
        _local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
    )(x, topk_w, topk_idx,
      w1_p, w1_s, w1_s2,
      w2_p, w2_s, w2_s2,
      w3_p, w3_s, w3_s2)


def _gather_topk_singledev(x, topk_w, topk_idx,
                            w1_p, w1_s, w1_s2,
                            w2_p, w2_s, w2_s2,
                            w3_p, w3_s, w3_s2):
    """Single-device reference path."""
    B_ = x.shape[0]
    flat_idx = topk_idx.reshape(B_ * TOP_K)
    flat_w = topk_w.reshape(B_ * TOP_K).astype(x.dtype)
    token_idx = jnp.repeat(jnp.arange(B_), TOP_K)
    x_exp = x[token_idx]

    def _one(xi, idx):
        g = nvfp4_matmul(xi[None], w1_p[idx], w1_s[idx], w1_s2[idx], MOE_INTER, H)[0]
        u = nvfp4_matmul(xi[None], w3_p[idx], w3_s[idx], w3_s2[idx], MOE_INTER, H)[0]
        h = jax.nn.silu(g) * u
        return nvfp4_matmul(h[None], w2_p[idx], w2_s[idx], w2_s2[idx], H, MOE_INTER)[0]

    out_per = jax.vmap(_one)(x_exp, flat_idx)
    weighted = out_per * flat_w[:, None]
    out = jnp.zeros((B_, H), dtype=x.dtype)
    out = out.at[token_idx].add(weighted)
    return out
