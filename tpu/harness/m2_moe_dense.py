"""Dense-B1 MoE block for MiniMax-M2.7-NVFP4 (Agent 12).

At B<=8, `moe_block_nvfp4` in m2_moe.py pays heavy all_to_all + sort/scatter
overhead that's not amortized by tiny per-expert GEMM work. This variant
eliminates all_to_all entirely:

  1. Each shard has experts [base .. base+32). It computes ALL 32 local
     experts' SwiGLU on the input x in ONE batched MXU call per projection.
  2. Builds a `(B, NUM_EXPERTS)` gate matrix with routing weights at the
     selected top-K global expert ids, zero elsewhere. Each shard selects
     its slice of columns.
  3. Weighted sum of local-expert outputs (`(32, B, H) * (B, 32) -> (B, H)`).
  4. `jax.lax.psum` over the 'expert' axis collects contributions across shards.

Trade-off: wastes (NUM_EXPERTS / TOP_K = 32x) compute — but MoE is usually
HBM-bandwidth-bound, so reading each expert's weights once is the cost,
not the FLOPs. At B=1, the shard is doing `(32, 1, H) @ (32, MI, H)^T` =
one 32-wide batched GEMM, which the MXU handles as one dispatch.

Expected perf vs shard_map all_to_all at B=1: ~5-10x faster.
Expected perf at B>=32: ~same or slightly slower (all_to_all wins when
dispatch fully amortizes).
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
    """Sigmoid scoring + aux-loss-free bias for top-K selection.

    Returns (topk_weights, topk_indices). Weights sum to 1 per row
    (norm_topk_prob=True); selection uses biased scores, weights are unbiased.
    """
    logits = x.astype(jnp.float32) @ router_w.T.astype(jnp.float32)   # (B, E)
    biased = logits + router_bias.astype(jnp.float32)
    _, topk_idx = jax.lax.top_k(biased, TOP_K)                        # (B, K)
    gathered = jnp.take_along_axis(logits, topk_idx, axis=1)
    weights = gathered / (gathered.sum(axis=-1, keepdims=True) + 1e-9)
    return weights.astype(x.dtype), topk_idx


def moe_block_dense_b1(x, router_w, router_bias, w1, w2, w3, mesh):
    """Dense-B1 MoE block. Each expert-shard computes its 32 local experts' SwiGLU
    in one batched GEMM, masks to selected top-K, weighted-sums, psum across shards.

    Args:
        x:           (B, H) bf16, replicated
        router_w:    (NUM_EXPERTS, H) bf16, replicated
        router_bias: (NUM_EXPERTS,) bf16, replicated
        w1, w3:      tuple(packed (E, MI, H/2) u8, scales (E, MI, H/16) u8, scale2 (E,) f32)
                     sharded P('expert', None, None) and P('expert')
        w2:          tuple(packed (E, H, MI/2) u8, scales (E, H, MI/16) u8, scale2 (E,) f32)
        mesh:        Mesh with 'expert' axis of size n_shards

    Returns:
        (B, H) bf16, replicated via psum.
    """
    # Router runs replicated — it's cheap.
    topk_w, topk_idx = router_sigmoid_topk(x, router_w, router_bias)   # (B, K), (B, K)

    # Build full (B, E) gate matrix: topk_w at selected indices, 0 elsewhere.
    B_ = x.shape[0]
    gate_full = jnp.zeros((B_, NUM_EXPERTS), dtype=topk_w.dtype)
    # Scatter topk weights to their indices.
    batch_idx = jnp.repeat(jnp.arange(B_, dtype=jnp.int32), TOP_K)
    flat_idx = topk_idx.reshape(-1)
    gate_full = gate_full.at[batch_idx, flat_idx].add(topk_w.reshape(-1))

    w1_p, w1_s, w1_s2 = w1
    w2_p, w2_s, w2_s2 = w2
    w3_p, w3_s, w3_s2 = w3

    # shard_map: within each shard, do all 32 local experts. Replicated inputs
    # stay replicated; expert-sharded weights get the local slice.
    has_expert_axis = (mesh is not None and 'expert' in mesh.axis_names)
    if not has_expert_axis:
        # Fallback: no sharding — compute all 256 experts sequentially.
        return _moe_dense_reference(
            x, gate_full, w1_p, w1_s, w1_s2, w2_p, w2_s, w2_s2, w3_p, w3_s, w3_s2)

    in_specs = (
        P(),                                # x (B, H) replicated
        P(),                                # gate_full (B, E) replicated
        P('expert', None, None),            # w1_p
        P('expert', None, None),            # w1_s
        P('expert'),                        # w1_s2 (E,)
        P('expert', None, None),            # w2_p
        P('expert', None, None),            # w2_s
        P('expert'),                        # w2_s2
        P('expert', None, None),            # w3_p
        P('expert', None, None),            # w3_s
        P('expert'),                        # w3_s2
    )
    out_specs = P()  # (B, H) replicated (via psum inside _local)

    def _local(x_l, gate_l, w1p_l, w1s_l, w1s2_l,
               w2p_l, w2s_l, w2s2_l, w3p_l, w3s_l, w3s2_l):
        """
        Local shapes:
            x_l:    (B, H)
            gate_l: (B, NUM_EXPERTS) — this shard will look at its slice later
            w1p_l:  (experts_per_shard, MI, H/2) uint8   (E/n_shards experts)
            w1s_l:  (experts_per_shard, MI, H/16) uint8
            w1s2_l: (experts_per_shard,) f32
            (w2, w3 similarly shaped)
        """
        # Figure out this shard's global expert index range.
        shard_id = jax.lax.axis_index('expert')
        n_shards = jax.lax.psum(1, 'expert')
        eps = NUM_EXPERTS // n_shards  # experts per shard
        base = shard_id * eps
        # Local gate slice: (B, experts_per_shard)
        local_gate = jax.lax.dynamic_slice_in_dim(gate_l, base, eps, axis=1)
        # local_gate shape: (B, eps)

        # Compute gate (w1) and up (w3) for ALL local experts at once.
        # Per-expert: out = nvfp4_matmul(x, w1p_l[e], w1s_l[e], w1s2_l[e], MI, H)
        # We vmap over the expert axis.
        def one_expert(w1p_e, w1s_e, w1s2_e, w3p_e, w3s_e, w3s2_e,
                       w2p_e, w2s_e, w2s2_e):
            gate = nvfp4_matmul(x_l, w1p_e, w1s_e, w1s2_e, MOE_INTER, H)  # (B, MI)
            up = nvfp4_matmul(x_l, w3p_e, w3s_e, w3s2_e, MOE_INTER, H)    # (B, MI)
            h = jax.nn.silu(gate) * up
            out = nvfp4_matmul(h, w2p_e, w2s_e, w2s2_e, H, MOE_INTER)     # (B, H)
            return out

        # (eps, B, H)
        per_expert_out = jax.vmap(one_expert)(
            w1p_l, w1s_l, w1s2_l,
            w3p_l, w3s_l, w3s2_l,
            w2p_l, w2s_l, w2s2_l,
        )

        # Weighted sum: (eps, B, H) * (eps, B) -> (B, H)
        # local_gate is (B, eps), transpose to (eps, B).
        gate_T = local_gate.T.astype(per_expert_out.dtype)           # (eps, B)
        weighted = per_expert_out * gate_T[..., None]                 # (eps, B, H)
        shard_out = weighted.sum(axis=0)                              # (B, H)

        # psum across 'expert' axis to combine shards.
        return jax.lax.psum(shard_out, 'expert')

    return shard_map(
        _local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
    )(x, gate_full,
      w1_p, w1_s, w1_s2,
      w2_p, w2_s, w2_s2,
      w3_p, w3_s, w3_s2)


def _moe_dense_reference(x, gate_full,
                         w1_p, w1_s, w1_s2,
                         w2_p, w2_s, w2_s2,
                         w3_p, w3_s, w3_s2):
    """Single-device reference: run all NUM_EXPERTS experts and apply gate.
    Not used in production; for unit-test parity only."""
    def one(e):
        g = nvfp4_matmul(x, w1_p[e], w1_s[e], w1_s2[e], MOE_INTER, H)
        u = nvfp4_matmul(x, w3_p[e], w3_s[e], w3_s2[e], MOE_INTER, H)
        h = jax.nn.silu(g) * u
        return nvfp4_matmul(h, w2_p[e], w2_s[e], w2_s2[e], H, MOE_INTER)
    all_out = jax.vmap(one)(jnp.arange(NUM_EXPERTS))   # (E, B, H)
    # Weight: gate_full (B, E) -> (E, B) and broadcast to (E, B, H).
    gate_T = gate_full.T.astype(all_out.dtype)
    return (all_out * gate_T[..., None]).sum(axis=0)
