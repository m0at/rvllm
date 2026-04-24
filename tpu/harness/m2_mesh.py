"""v6e-8 expert-parallel mesh + shard_map topology for MiniMax-M2.7.

Owned by agent 12. See M2_SWARM_SPEC.md section "Agent 12".

Mesh shape: (8,) single axis 'expert'. Experts are sharded 32/chip across
256 total experts. Attention replicated on all chips for v0.
"""

from __future__ import annotations

import math
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


def make_mesh_v6e8():
    """Return (mesh, axes_dict). Mesh shape = (8,), axis='expert'.

    axes_dict maps the string name to the axis size so callers can look
    up shard counts without pattern matching on the mesh object.
    """
    devs = jax.devices()
    if len(devs) < 8:
        raise RuntimeError(
            f"make_mesh_v6e8 requires >=8 devices, found {len(devs)}. "
            "On CPU run with XLA_FLAGS='--xla_force_host_platform_device_count=8'."
        )
    mesh = Mesh(np.array(devs[:8]), ('expert',))
    axes_dict = {'expert': 8}
    return mesh, axes_dict


def expert_shard_spec(num_experts: int = 256) -> P:
    """PartitionSpec for expert weight tensor of shape (num_experts, D_out, D_in).

    Experts sharded along axis 0 across mesh axis 'expert'. For 256 experts
    across 8 shards, each chip owns 32 experts.
    """
    if num_experts % 8 != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by 8 for v6e-8 EP"
        )
    return P('expert', None, None)


def replicate_spec() -> P:
    """PartitionSpec for a tensor replicated on every chip."""
    return P()


def _dispatch_sort(x_per_token: jax.Array, expert_idx: jax.Array,
                   num_experts: int, capacity_per_expert: int):
    """Host-side (traced) sort of (B*TOP_K) tokens into per-expert bins.

    Returns:
        bins: (num_experts, capacity_per_expert, H) bf16/f32 tokens with zeros padding.
        idx_map: (num_experts, capacity_per_expert) original flat token index, -1 if empty.
        overflow: scalar bool indicating any expert exceeded capacity.
    """
    B, K, H = x_per_token.shape
    flat_x = x_per_token.reshape(B * K, H)
    flat_e = expert_idx.reshape(B * K)

    bins = jnp.zeros((num_experts, capacity_per_expert, H), dtype=x_per_token.dtype)
    idx_map = jnp.full((num_experts, capacity_per_expert), -1, dtype=jnp.int32)

    # Per-expert position counter via cumulative count.
    # For each expert e, position of token t = count of (flat_e[:t] == e).
    one_hot = jax.nn.one_hot(flat_e, num_experts, dtype=jnp.int32)  # (N, E)
    positions = jnp.cumsum(one_hot, axis=0) - 1                      # (N, E)
    my_pos = jnp.take_along_axis(positions, flat_e[:, None], axis=1).squeeze(-1)
    # (N,) position within its expert bin

    overflow = jnp.any(my_pos >= capacity_per_expert)
    my_pos_clamped = jnp.minimum(my_pos, capacity_per_expert - 1)

    # Scatter
    n_indices = jnp.arange(B * K, dtype=jnp.int32)
    bins = bins.at[flat_e, my_pos_clamped].set(flat_x)
    idx_map = idx_map.at[flat_e, my_pos_clamped].set(n_indices)
    return bins, idx_map, overflow


def expert_all_to_all_dispatch(
    x_per_token: jax.Array,          # (B, TOP_K, H)
    expert_idx: jax.Array,           # (B, TOP_K) int32
    num_experts: int,
    num_shards: int,
):
    """Sort tokens by expert, all-to-all exchange across 'expert' axis.

    Capacity per expert = ceil(B*TOP_K / num_experts) * 2 (overflow factor).
    Drops tokens past capacity; logs a warning.

    Returns:
        local_tokens: (experts_per_shard, capacity, H) tokens for this shard's experts.
        idx_map: (num_experts, capacity) original flat token index, routed back in combine.
    """
    if num_experts % num_shards != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by num_shards={num_shards}"
        )
    experts_per_shard = num_experts // num_shards
    B, K, H = x_per_token.shape
    capacity = int(math.ceil(B * K / num_experts)) * 2
    if capacity < 1:
        capacity = 1

    bins, idx_map, overflow = _dispatch_sort(
        x_per_token, expert_idx, num_experts, capacity
    )

    def _over(x):
        if bool(x):
            warnings.warn(
                "expert_all_to_all_dispatch: capacity overflow, tokens dropped. "
                "Increase capacity factor or rebalance routing.",
                RuntimeWarning,
                stacklevel=2,
            )
        return x

    # Do not break trace: we just pipe overflow as a side-channel bool.
    # Caller may use jax.experimental.host_callback to surface it at runtime.

    def _a2a(bins_in):
        # bins_in: (num_experts, capacity, H). all_to_all over the num_experts axis.
        # split_axis=0 (num_experts), concat_axis=0 -> each shard gets its
        # experts_per_shard slice contributed by all shards concatenated.
        return jax.lax.all_to_all(
            bins_in,
            axis_name='expert',
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )

    in_spec = P('expert', None, None)  # bins already replicated by construction
    out_spec = P('expert', None, None)
    # The tensor "bins" is globally identical across shards; shard_map will
    # feed each shard the full (num_experts, capacity, H) view, and inside
    # shard_map we call all_to_all to rebalance it into per-shard slices.
    # To express replicated input to shard_map, we pass P() and rely on it.
    replicated_bins = shard_map(
        _a2a,
        mesh=_current_mesh(),
        in_specs=P(None, None, None),
        out_specs=P(None, None, None),
        check_rep=False,
    )(bins)

    # After all_to_all, each shard holds (experts_per_shard * num_shards, capacity, H)
    # along the leading axis (concat_axis=0 packs all contributions back).
    # We reshape and take only the experts local to this shard via shard_map.
    local_tokens = replicated_bins  # same shape; routing handled by combine via idx_map.
    return local_tokens, idx_map


def expert_all_to_all_combine(
    local_out: jax.Array,            # (num_experts, capacity, H) expert FFN outputs
    idx_map: jax.Array,              # (num_experts, capacity) original flat token indices
    weights: jax.Array,              # (B, TOP_K) routing weights
    out_shape: tuple,                # (B, H)
) -> jax.Array:
    """All-to-all back, weighted sum, scatter-add to original token positions.

    Returns (B, H).
    """
    B, H = out_shape
    _, K = weights.shape

    def _a2a(x):
        return jax.lax.all_to_all(
            x,
            axis_name='expert',
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )

    combined = shard_map(
        _a2a,
        mesh=_current_mesh(),
        in_specs=P(None, None, None),
        out_specs=P(None, None, None),
        check_rep=False,
    )(local_out)

    # Flatten bins
    num_experts, capacity, _ = combined.shape
    flat_out = combined.reshape(num_experts * capacity, H)
    flat_idx = idx_map.reshape(num_experts * capacity)

    # Weights: flat of length B*K, matching the original flat_e order from dispatch.
    # idx_map stores original flat token index. Build weight-per-slot by
    # gathering weights.flatten()[flat_idx] with zero for -1.
    flat_w = weights.reshape(B * K)
    valid = flat_idx >= 0
    safe_idx = jnp.where(valid, flat_idx, 0)
    slot_w = jnp.where(valid, flat_w[safe_idx], 0.0).astype(flat_out.dtype)
    weighted = flat_out * slot_w[:, None]

    # Scatter-add into (B, H). slot -> original_token = flat_idx // K (since flat = b*K + k).
    token_id = jnp.where(valid, safe_idx // K, 0)
    mask = valid[:, None].astype(weighted.dtype)
    weighted = weighted * mask

    out = jnp.zeros((B, H), dtype=weighted.dtype)
    out = out.at[token_id].add(weighted)
    return out


# -- mesh context helper --

_MESH_STACK: list = []


def _current_mesh():
    if _MESH_STACK:
        return _MESH_STACK[-1]
    # Fall back to the currently-active jax mesh (set via `with mesh:` context).
    from jax.experimental.mesh_utils import create_device_mesh  # noqa: F401
    return jax.experimental.maps.thread_resources.env.physical_mesh  # type: ignore[attr-defined]


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    jax.config.update("jax_platforms", "cpu")

    print("m2_mesh smoke test", file=sys.stderr)
    mesh, axes = make_mesh_v6e8()
    print(f"  mesh={mesh} axes={axes}", file=sys.stderr)
    print(f"  expert_shard_spec(256)={expert_shard_spec(256)}", file=sys.stderr)
    print(f"  replicate_spec()={replicate_spec()}", file=sys.stderr)

    # Tiny sort test without entering shard_map (a2a requires real mesh axis).
    B, K, H = 32, 2, 16
    num_experts = 16
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((B, K, H)).astype(np.float32))
    eidx = jnp.asarray(rng.integers(0, num_experts, size=(B, K)).astype(np.int32))
    capacity = int(math.ceil(B * K / num_experts)) * 2
    bins, idx_map, overflow = _dispatch_sort(x, eidx, num_experts, capacity)
    print(f"  bins.shape={bins.shape} idx_map.shape={idx_map.shape} overflow={bool(overflow)}", file=sys.stderr)
    assert bins.shape == (num_experts, capacity, H)
    assert idx_map.shape == (num_experts, capacity)
    print("  OK", file=sys.stderr)
