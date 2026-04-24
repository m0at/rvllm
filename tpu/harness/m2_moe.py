"""MiniMax-M2 MoE layer: sigmoid + bias router, top-8 of 256 experts, SwiGLU.

Per M2_SWARM_SPEC.md agent 9 scope.

Differences vs. Gemma 4 MoE:
  - sigmoid (not softmax) activation on router logits
  - per-expert routing bias (aux-loss-free), added ONLY for top-K selection
  - no shared expert (shared_intermediate_size = 0)
  - norm_topk_prob = True: renormalize selected weights to sum=1
  - 256 experts, top-8, per-expert inter dim 1536
  - SwiGLU (silu(gate) * up) not gated-GELU
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
    """Sigmoid router with aux-loss-free bias for M2.

    Args:
        x: (B, H) input activations (bf16 / f32)
        router_w: (NUM_EXPERTS, H) router projection
        router_bias: (NUM_EXPERTS,) per-expert bias (aux-loss-free)

    Returns:
        (topk_weights, topk_indices) both (B, TOP_K).
        Weights are unbiased sigmoid scores at selected indices, renormalized
        to sum=1 along TOP_K axis (norm_topk_prob=True).
    """
    logits = (x.astype(jnp.float32) @ router_w.astype(jnp.float32).T)  # (B, E)
    scores = jax.nn.sigmoid(logits)                                    # (B, E)
    biased = scores + router_bias.astype(jnp.float32)[None, :]         # (B, E)
    _, topk_idx = jax.lax.top_k(biased, TOP_K)                         # (B, TOP_K)
    b = jnp.arange(x.shape[0])[:, None]
    topk_sel = scores[b, topk_idx]                                     # (B, TOP_K)
    denom = jnp.sum(topk_sel, axis=-1, keepdims=True).clip(min=1e-20)
    topk_w = (topk_sel / denom).astype(x.dtype)
    return topk_w, topk_idx


def expert_ffn(x, gate_up_w, down_w):
    """Single-expert SwiGLU FFN.

    Args:
        x: (N, H)
        gate_up_w: (2*MOE_INTER, H) fused [gate || up] weight
        down_w: (H, MOE_INTER)

    Returns:
        (N, H)
    """
    gu = x @ gate_up_w.T                          # (N, 2*MOE_INTER)
    inter = gate_up_w.shape[0] // 2
    gate = gu[:, :inter]
    up = gu[:, inter:]
    h = jax.nn.silu(gate) * up                    # (N, MOE_INTER)
    return h @ down_w.T                           # (N, H)


def _moe_gather_reference(x, router_w, router_bias, expert_gate_up, expert_down):
    """Gather-based reference implementation (no expert parallelism)."""
    topk_w, topk_idx = router_sigmoid_topk(x, router_w, router_bias)  # (B, K)
    B = x.shape[0]
    out = jnp.zeros_like(x)
    # Expand each token K times, run every selected expert, weighted sum.
    # Straightforward loop over TOP_K slots (each slot has B different experts).
    for k in range(topk_idx.shape[1]):
        idx_k = topk_idx[:, k]                     # (B,)
        gu = expert_gate_up[idx_k]                  # (B, 2*MOE_INTER, H)
        dn = expert_down[idx_k]                     # (B, H, MOE_INTER)
        # per-token matmul via einsum
        gu_out = jnp.einsum('bh,boh->bo', x, gu)   # (B, 2*MOE_INTER)
        inter = expert_gate_up.shape[1] // 2
        gate = gu_out[:, :inter]
        up = gu_out[:, inter:]
        h = jax.nn.silu(gate) * up                  # (B, MOE_INTER)
        d_out = jnp.einsum('bm,bhm->bh', h, dn)    # (B, H)
        out = out + topk_w[:, k:k+1].astype(out.dtype) * d_out.astype(out.dtype)
    return out


def _moe_shard_map(x, router_w, router_bias, expert_gate_up, expert_down, mesh):
    """shard_map variant with all-to-all dispatch along 'expert' axis.

    experts are sharded along axis 0: (NUM_EXPERTS, ...) -> per-shard
    (NUM_EXPERTS / n_shards, ...). Router runs replicated. We pack each
    token's K dispatches into a dense send buffer bucketed by destination
    shard, all-to-all exchange, run local experts, all-to-all back, combine.
    """
    n_shards = mesh.shape['expert']
    assert NUM_EXPERTS % n_shards == 0, 'experts must divide evenly across shards'
    experts_per_shard = NUM_EXPERTS // n_shards

    B = x.shape[0]
    # Route globally (router is replicated).
    topk_w, topk_idx = router_sigmoid_topk(x, router_w, router_bias)  # (B, K)

    in_specs = (
        P(),                       # x replicated
        P(),                       # topk_w replicated
        P(),                       # topk_idx replicated
        P('expert', None, None),   # expert_gate_up sharded on experts
        P('expert', None, None),   # expert_down sharded on experts
    )
    out_specs = P()                # output replicated

    def _local(x_l, w_l, idx_l, egu_l, edn_l):
        # Everything here is local to one 'expert' shard.
        # egu_l: (experts_per_shard, 2*MOE_INTER, H)
        # edn_l: (experts_per_shard, H, MOE_INTER)
        # Build capacity-bounded send buffer: bucket (B, K) dispatches by
        # destination shard (idx_l // experts_per_shard). Each shard sends
        # `capacity` slots to every other shard.
        flat_idx = idx_l.reshape(-1)                        # (B*K,)
        flat_w = w_l.reshape(-1)                            # (B*K,)
        dest_shard = flat_idx // experts_per_shard          # (B*K,)
        local_expert = flat_idx % experts_per_shard        # (B*K,)
        token_id = jnp.repeat(jnp.arange(B), TOP_K)         # (B*K,)

            # Static capacity per (src_shard, dst_shard) slot. Overflow
            # factor of 2 (spec default); drop overflow for v0.
        capacity = (B * TOP_K + n_shards - 1) // n_shards * 2

        # Build per-destination packed buffers of shape (n_shards, capacity, H).
        # Use scatter via sort-by-destination + positional index within bucket.
        # Stable sort on dest_shard so original order preserved within bucket.
        order = jnp.argsort(dest_shard, stable=True)
        sorted_dest = dest_shard[order]
        sorted_local = local_expert[order]
        sorted_token = token_id[order]
        sorted_w = flat_w[order]
        # Position within its destination bucket: cumulative count of same dest.
        same = (sorted_dest[:, None] == jnp.arange(n_shards)[None, :])  # (BK, S)
        pos_in_bucket = jnp.cumsum(same, axis=0) - 1                    # (BK, S)
        # Select the column matching this row's dest.
        pos = jnp.take_along_axis(pos_in_bucket, sorted_dest[:, None], 1)[:, 0]
        within = pos < capacity

        send_tokens = jnp.zeros((n_shards, capacity, x_l.shape[-1]), x_l.dtype)
        send_local = jnp.zeros((n_shards, capacity), jnp.int32)
        send_valid = jnp.zeros((n_shards, capacity), jnp.bool_)
        send_w = jnp.zeros((n_shards, capacity), flat_w.dtype)
        send_tok_id = jnp.zeros((n_shards, capacity), jnp.int32)

        payload = x_l[sorted_token]                                      # (BK, H)
        send_tokens = send_tokens.at[sorted_dest, pos].set(
            jnp.where(within[:, None], payload, 0)
        )
        send_local = send_local.at[sorted_dest, pos].set(
            jnp.where(within, sorted_local, 0)
        )
        send_valid = send_valid.at[sorted_dest, pos].set(within)
        send_w = send_w.at[sorted_dest, pos].set(
            jnp.where(within, sorted_w, 0)
        )
        send_tok_id = send_tok_id.at[sorted_dest, pos].set(
            jnp.where(within, sorted_token, 0)
        )

        # all-to-all: split along axis 0 (dest shard), concat along axis 0
        # (src shard) after exchange.
        recv_tokens = jax.lax.all_to_all(
            send_tokens, 'expert', split_axis=0, concat_axis=0, tiled=True
        )  # (n_shards * capacity, H) — each src's slice for this dst
        recv_local = jax.lax.all_to_all(
            send_local, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        recv_valid = jax.lax.all_to_all(
            send_valid, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        recv_w = jax.lax.all_to_all(
            send_w, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        recv_tok_id = jax.lax.all_to_all(
            send_tok_id, 'expert', split_axis=0, concat_axis=0, tiled=True
        )

        # Run local experts: iterate experts_per_shard, mask to tokens for this
        # local expert, apply SwiGLU, accumulate.
        local_out = jnp.zeros_like(recv_tokens)
        for e in range(experts_per_shard):
            mask = (recv_local == e) & recv_valid                        # (N,)
            x_e = recv_tokens * mask[:, None].astype(recv_tokens.dtype)
            gu = x_e @ egu_l[e].T                                        # (N, 2*MOE_INTER)
            inter_dim = egu_l.shape[1] // 2
            gate = gu[:, :inter_dim]
            up = gu[:, inter_dim:]
            h = jax.nn.silu(gate) * up
            d = h @ edn_l[e].T                                           # (N, H)
            local_out = local_out + d * mask[:, None].astype(d.dtype)

        # Apply routing weight here so that combine is a pure scatter-add.
        local_out = local_out * recv_w[:, None].astype(local_out.dtype)

        # Reshape back to (n_shards, capacity, H) and all-to-all back so each
        # src gets its own slots.
        send_back = local_out.reshape(n_shards, capacity, x_l.shape[-1])
        recv_back = jax.lax.all_to_all(
            send_back, 'expert', split_axis=0, concat_axis=0, tiled=True
        )  # (n_shards, capacity, H) on the original src
        # Also return token IDs through the same trip so we can scatter.
        send_tok_back = recv_tok_id.reshape(n_shards, capacity)
        recv_tok_back = jax.lax.all_to_all(
            send_tok_back, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        send_valid_back = recv_valid.reshape(n_shards, capacity)
        recv_valid_back = jax.lax.all_to_all(
            send_valid_back, 'expert', split_axis=0, concat_axis=0, tiled=True
        )

        flat_back = recv_back.reshape(-1, x_l.shape[-1])
        flat_tok = recv_tok_back.reshape(-1)
        flat_valid = recv_valid_back.reshape(-1)
        contrib = flat_back * flat_valid[:, None].astype(flat_back.dtype)

        out = jnp.zeros_like(x_l)
        out = out.at[flat_tok].add(contrib)
        return out

    return shard_map(
        _local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(x, topk_w, topk_idx, expert_gate_up, expert_down)


def moe_block(x_normed, router_w, router_bias, expert_gate_up, expert_down, mesh):
    """Top-8 routing + expert FFN + weighted combine.

    If mesh has an 'expert' axis, use shard_map with all-to-all dispatch;
    otherwise, use the gather reference.
    """
    has_expert_axis = (
        mesh is not None
        and hasattr(mesh, 'axis_names')
        and 'expert' in mesh.axis_names
    )
    if has_expert_axis:
        return _moe_shard_map(
            x_normed, router_w, router_bias, expert_gate_up, expert_down, mesh
        )
    return _moe_gather_reference(
        x_normed, router_w, router_bias, expert_gate_up, expert_down
    )


def _moe_gather_reference_nvfp4(
    x, router_w, router_bias,
    w1_packed, w1_scale, w1_scale2,
    w2_packed, w2_scale, w2_scale2,
    w3_packed, w3_scale, w3_scale2,
):
    """Gather-based reference for NVFP4-packed expert weights (three projections).

    Each of w1 (gate), w3 (up), w2 (down) has its own packed tensor + block
    scales + FP32 per-tensor global scale, stacked along expert axis.

    w1_packed: (NUM_EXPERTS, MOE_INTER, H/2) uint8
    w1_scale:  (NUM_EXPERTS, MOE_INTER, H/16) uint8
    w1_scale2: (NUM_EXPERTS,) float32
    w3_packed: (NUM_EXPERTS, MOE_INTER, H/2) uint8
    w3_scale:  (NUM_EXPERTS, MOE_INTER, H/16) uint8
    w3_scale2: (NUM_EXPERTS,) float32
    w2_packed: (NUM_EXPERTS, H, MOE_INTER/2) uint8
    w2_scale:  (NUM_EXPERTS, H, MOE_INTER/16) uint8
    w2_scale2: (NUM_EXPERTS,) float32
    """
    topk_w, topk_idx = router_sigmoid_topk(x, router_w, router_bias)  # (B, K)
    out = jnp.zeros_like(x)
    H_dim = x.shape[-1]
    inter = w1_packed.shape[1]
    for k in range(topk_idx.shape[1]):
        for b in range(x.shape[0]):
            e = topk_idx[b, k]
            x_b = x[b:b+1]                 # (1, H)
            gate = nvfp4_matmul(
                x_b, w1_packed[e], w1_scale[e], w1_scale2[e], inter, H_dim
            )                              # (1, MOE_INTER)
            up = nvfp4_matmul(
                x_b, w3_packed[e], w3_scale[e], w3_scale2[e], inter, H_dim
            )                              # (1, MOE_INTER)
            h = jax.nn.silu(gate) * up     # (1, MOE_INTER)
            y = nvfp4_matmul(
                h, w2_packed[e], w2_scale[e], w2_scale2[e], H_dim, inter
            )                              # (1, H)
            out = out.at[b].add(
                (topk_w[b, k].astype(y.dtype) * y[0]).astype(out.dtype)
            )
    return out


def _moe_shard_map_nvfp4(
    x, router_w, router_bias,
    w1_packed, w1_scale, w1_scale2,
    w2_packed, w2_scale, w2_scale2,
    w3_packed, w3_scale, w3_scale2,
    mesh,
):
    """shard_map MoE with on-the-fly NVFP4 dequant per expert.

    Expert weights are NVFP4-packed and sharded along the expert axis.
    Three separate projections w1 (gate), w3 (up), w2 (down), each with a
    per-tensor FP32 global scale stacked as (NUM_EXPERTS,).
    """
    n_shards = mesh.shape['expert']
    assert NUM_EXPERTS % n_shards == 0, 'experts must divide evenly across shards'
    experts_per_shard = NUM_EXPERTS // n_shards

    B = x.shape[0]
    topk_w, topk_idx = router_sigmoid_topk(x, router_w, router_bias)  # (B, K)

    in_specs = (
        P(),                       # x replicated
        P(),                       # topk_w replicated
        P(),                       # topk_idx replicated
        P('expert', None, None),   # w1_packed sharded on experts
        P('expert', None, None),   # w1_scale sharded on experts
        P('expert'),               # w1_scale2 sharded on experts
        P('expert', None, None),   # w2_packed sharded on experts
        P('expert', None, None),   # w2_scale sharded on experts
        P('expert'),               # w2_scale2 sharded on experts
        P('expert', None, None),   # w3_packed sharded on experts
        P('expert', None, None),   # w3_scale sharded on experts
        P('expert'),               # w3_scale2 sharded on experts
    )
    out_specs = P()

    def _local(x_l, w_l, idx_l,
               w1p_l, w1s_l, w1g_l,
               w2p_l, w2s_l, w2g_l,
               w3p_l, w3s_l, w3g_l):
        flat_idx = idx_l.reshape(-1)
        flat_w = w_l.reshape(-1)
        dest_shard = flat_idx // experts_per_shard
        local_expert = flat_idx % experts_per_shard
        token_id = jnp.repeat(jnp.arange(B), TOP_K)

        capacity = (B * TOP_K + n_shards - 1) // n_shards * 2

        order = jnp.argsort(dest_shard, stable=True)
        sorted_dest = dest_shard[order]
        sorted_local = local_expert[order]
        sorted_token = token_id[order]
        sorted_w = flat_w[order]
        same = (sorted_dest[:, None] == jnp.arange(n_shards)[None, :])
        pos_in_bucket = jnp.cumsum(same, axis=0) - 1
        pos = jnp.take_along_axis(pos_in_bucket, sorted_dest[:, None], 1)[:, 0]
        within = pos < capacity

        H_dim = x_l.shape[-1]
        inter_dim = w1p_l.shape[1]

        send_tokens = jnp.zeros((n_shards, capacity, H_dim), x_l.dtype)
        send_local = jnp.zeros((n_shards, capacity), jnp.int32)
        send_valid = jnp.zeros((n_shards, capacity), jnp.bool_)
        send_w = jnp.zeros((n_shards, capacity), flat_w.dtype)
        send_tok_id = jnp.zeros((n_shards, capacity), jnp.int32)

        payload = x_l[sorted_token]
        send_tokens = send_tokens.at[sorted_dest, pos].set(
            jnp.where(within[:, None], payload, 0)
        )
        send_local = send_local.at[sorted_dest, pos].set(
            jnp.where(within, sorted_local, 0)
        )
        send_valid = send_valid.at[sorted_dest, pos].set(within)
        send_w = send_w.at[sorted_dest, pos].set(
            jnp.where(within, sorted_w, 0)
        )
        send_tok_id = send_tok_id.at[sorted_dest, pos].set(
            jnp.where(within, sorted_token, 0)
        )

        recv_tokens = jax.lax.all_to_all(
            send_tokens, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        recv_local = jax.lax.all_to_all(
            send_local, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        recv_valid = jax.lax.all_to_all(
            send_valid, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        recv_w = jax.lax.all_to_all(
            send_w, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        recv_tok_id = jax.lax.all_to_all(
            send_tok_id, 'expert', split_axis=0, concat_axis=0, tiled=True
        )

        local_out = jnp.zeros_like(recv_tokens)
        for e in range(experts_per_shard):
            mask = (recv_local == e) & recv_valid
            x_e = recv_tokens * mask[:, None].astype(recv_tokens.dtype)
            gate = nvfp4_matmul(
                x_e, w1p_l[e], w1s_l[e], w1g_l[e], inter_dim, H_dim
            )  # (N, MOE_INTER)
            up = nvfp4_matmul(
                x_e, w3p_l[e], w3s_l[e], w3g_l[e], inter_dim, H_dim
            )  # (N, MOE_INTER)
            h = jax.nn.silu(gate) * up
            d = nvfp4_matmul(
                h, w2p_l[e], w2s_l[e], w2g_l[e], H_dim, inter_dim
            )  # (N, H)
            local_out = local_out + d * mask[:, None].astype(d.dtype)

        local_out = local_out * recv_w[:, None].astype(local_out.dtype)

        send_back = local_out.reshape(n_shards, capacity, H_dim)
        recv_back = jax.lax.all_to_all(
            send_back, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        send_tok_back = recv_tok_id.reshape(n_shards, capacity)
        recv_tok_back = jax.lax.all_to_all(
            send_tok_back, 'expert', split_axis=0, concat_axis=0, tiled=True
        )
        send_valid_back = recv_valid.reshape(n_shards, capacity)
        recv_valid_back = jax.lax.all_to_all(
            send_valid_back, 'expert', split_axis=0, concat_axis=0, tiled=True
        )

        flat_back = recv_back.reshape(-1, H_dim)
        flat_tok = recv_tok_back.reshape(-1)
        flat_valid = recv_valid_back.reshape(-1)
        contrib = flat_back * flat_valid[:, None].astype(flat_back.dtype)

        out = jnp.zeros_like(x_l)
        out = out.at[flat_tok].add(contrib)
        return out

    return shard_map(
        _local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(x, topk_w, topk_idx,
      w1_packed, w1_scale, w1_scale2,
      w2_packed, w2_scale, w2_scale2,
      w3_packed, w3_scale, w3_scale2)


def moe_block_nvfp4(
    x_normed,
    router_w,
    router_bias,
    w1,
    w2,
    w3,
    mesh,
):
    """Path A MoE block with NVFP4-packed expert weights (three projections).

    Same top-8 sigmoid+bias routing as moe_block. Per-expert SwiGLU FFN:
        gate = nvfp4_matmul(x, w1)
        up   = nvfp4_matmul(x, w3)
        h    = silu(gate) * up
        out  = nvfp4_matmul(h, w2)

    Each of w1, w2, w3 is a tuple of three arrays:
        (packed, scale, scale2)
    with shapes per expert, stacked along axis 0:
        w1: packed (NE, MOE_INTER, H/2)  scale (NE, MOE_INTER, H/16)  scale2 (NE,)
        w3: packed (NE, MOE_INTER, H/2)  scale (NE, MOE_INTER, H/16)  scale2 (NE,)
        w2: packed (NE, H, MOE_INTER/2)  scale (NE, H, MOE_INTER/16)  scale2 (NE,)

    Shapes:
        x_normed: (B, H)
        router_w: (NUM_EXPERTS, H) bf16
        router_bias: (NUM_EXPERTS,) bf16
    """
    w1_packed, w1_scale, w1_scale2 = w1
    w2_packed, w2_scale, w2_scale2 = w2
    w3_packed, w3_scale, w3_scale2 = w3
    has_expert_axis = (
        mesh is not None
        and hasattr(mesh, 'axis_names')
        and 'expert' in mesh.axis_names
    )
    if has_expert_axis:
        return _moe_shard_map_nvfp4(
            x_normed, router_w, router_bias,
            w1_packed, w1_scale, w1_scale2,
            w2_packed, w2_scale, w2_scale2,
            w3_packed, w3_scale, w3_scale2,
            mesh,
        )
    return _moe_gather_reference_nvfp4(
        x_normed, router_w, router_bias,
        w1_packed, w1_scale, w1_scale2,
        w2_packed, w2_scale, w2_scale2,
        w3_packed, w3_scale, w3_scale2,
    )


if __name__ == '__main__':
    import os
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import numpy as np

    rng = np.random.default_rng(0)
    # Scaled-down smoke test: 4 experts, top-2, small hidden dim.
    B_s = 4
    H_s = 32
    INTER_s = 16
    E_s = 4
    K_s = 2

    x = jnp.asarray(rng.standard_normal((B_s, H_s)), dtype=jnp.float32)
    rw = jnp.asarray(rng.standard_normal((E_s, H_s)) * 0.1, dtype=jnp.float32)
    rb = jnp.asarray(rng.standard_normal((E_s,)) * 0.01, dtype=jnp.float32)
    egu = jnp.asarray(
        rng.standard_normal((E_s, 2 * INTER_s, H_s)) * 0.05, dtype=jnp.float32
    )
    edn = jnp.asarray(
        rng.standard_normal((E_s, H_s, INTER_s)) * 0.05, dtype=jnp.float32
    )

    def gather_ref_small(x, rw, rb, egu, edn):
        logits = x @ rw.T
        scores = jax.nn.sigmoid(logits)
        biased = scores + rb[None, :]
        _, idx = jax.lax.top_k(biased, K_s)
        b = jnp.arange(x.shape[0])[:, None]
        sel = scores[b, idx]
        w = sel / sel.sum(axis=-1, keepdims=True)
        out = jnp.zeros_like(x)
        for k in range(K_s):
            i = idx[:, k]
            gu = egu[i]
            dn = edn[i]
            go = jnp.einsum('bh,boh->bo', x, gu)
            gate = go[:, :INTER_s]
            up = go[:, INTER_s:]
            h = jax.nn.silu(gate) * up
            d = jnp.einsum('bm,bhm->bh', h, dn)
            out = out + w[:, k:k+1] * d
        return out

    # Run gather variant via moe_block with mesh=None.
    # Patch module-level constants for this smoke test by calling the gather
    # reference function directly with our small tensors. moe_block uses
    # TOP_K=8 hardcoded for router_sigmoid_topk, so we only exercise the small
    # path with a local reference and the single-expert helpers.

    # Smoke test 1: expert_ffn shape.
    y = expert_ffn(x, egu[0], edn[0])
    assert y.shape == (B_s, H_s), f'expert_ffn shape {y.shape}'

    # Smoke test 2: the local gather reference with TOP_K=K_s.
    ref = gather_ref_small(x, rw, rb, egu, edn)
    assert ref.shape == (B_s, H_s), f'ref shape {ref.shape}'
    assert jnp.all(jnp.isfinite(ref)), 'ref has non-finite values'

    # Smoke test 3: router_sigmoid_topk weights sum to 1.
    # Use full E=NUM_EXPERTS sized router for this check.
    rw_full = jnp.asarray(
        rng.standard_normal((NUM_EXPERTS, H)) * 0.01, dtype=jnp.float32
    )
    rb_full = jnp.asarray(
        rng.standard_normal((NUM_EXPERTS,)) * 0.001, dtype=jnp.float32
    )
    x_full = jnp.asarray(
        rng.standard_normal((2, H)), dtype=jnp.float32
    )
    tw, ti = router_sigmoid_topk(x_full, rw_full, rb_full)
    assert tw.shape == (2, TOP_K)
    assert ti.shape == (2, TOP_K)
    s = jnp.sum(tw, axis=-1)
    assert jnp.allclose(s, jnp.ones_like(s), atol=1e-5), f'weights sum {s}'

    print('m2_moe smoke tests passed:')
    print(f'  expert_ffn out shape: {y.shape}')
    print(f'  gather reference out shape: {ref.shape}')
    print(f'  router topk sum to 1: max dev {float(jnp.max(jnp.abs(s - 1.0))):.2e}')
