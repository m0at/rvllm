"""MiniMax-M2.7 KV cache layout.

GQA config: 62 layers, 8 KV heads, head_dim=128, bf16.
KV bytes per token = 8 * 128 * 2 (K,V) * 2 bytes * 62 layers = 253_952.

Dense path: (num_layers, B, max_ctx, num_kv_heads, head_dim) for ctx <= 32K.
Paged path: fixed block_size=256 pages, (num_layers, num_pages, block_size,
num_kv_heads, head_dim) plus (B, max_pages) int32 block_table. Used when
ctx > 32K (up to the model's 196608 cap).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


M2_KV_LAYOUT = {
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_layers': 62,
    'dtype': 'bf16',
    'block_size': 256,
}


_DENSE_CTX_LIMIT = 32 * 1024
_MAX_POSITION = 196608


def kv_bytes_per_token() -> int:
    """Bytes of KV cache consumed by a single token across all layers.

    Arithmetic: 8 kv_heads * 128 head_dim * 2 (K and V) * 2 bytes (bf16) * 62 layers
    = 253_952 bytes/token. (The spec had 260_096 as a typo; corrected to the real product.)
    """
    return 253_952


def _kv_sharding(mesh):
    # Replicate KV across the 'expert' axis; if mesh lacks that axis, fall back
    # to fully replicated so this module stays usable on CPU/test meshes.
    axis_names = tuple(mesh.axis_names)
    if 'expert' in axis_names:
        return NamedSharding(mesh, P(None, None, None, None, None))
    return NamedSharding(mesh, P(None, None, None, None, None))


def _table_sharding(mesh):
    return NamedSharding(mesh, P(None, None))


def make_kv_caches(B: int, max_ctx: int, mesh):
    """Dense KV caches for ctx <= 32K.

    Returns {'k','v'} each shape (62, B, max_ctx, 8, 128) bf16, replicated
    across the 'expert' axis.
    """
    if max_ctx > _DENSE_CTX_LIMIT:
        raise ValueError(
            f"make_kv_caches: max_ctx={max_ctx} exceeds dense limit "
            f"{_DENSE_CTX_LIMIT}; use allocate_paged_kv instead"
        )
    if max_ctx > _MAX_POSITION:
        raise ValueError(
            f"make_kv_caches: max_ctx={max_ctx} exceeds model max_position "
            f"{_MAX_POSITION}"
        )

    nl = M2_KV_LAYOUT['num_layers']
    nkv = M2_KV_LAYOUT['num_kv_heads']
    hd = M2_KV_LAYOUT['head_dim']
    shape = (nl, B, max_ctx, nkv, hd)
    sh = _kv_sharding(mesh)

    def zeros():
        return jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), sh)

    return {'k': zeros(), 'v': zeros()}


def allocate_paged_kv(max_pages: int, block_size: int, mesh, B: int = 1):
    """Paged KV for long context (>32K).

    Returns (paged_k, paged_v, block_table) where:
      paged_k/v: (62, max_pages, block_size, 8, 128) bf16.
      block_table: (B, max_pages) int32, initialized to -1 (unmapped).
    """
    if block_size != M2_KV_LAYOUT['block_size']:
        raise ValueError(
            f"allocate_paged_kv: block_size must be {M2_KV_LAYOUT['block_size']}, "
            f"got {block_size}"
        )
    if max_pages <= 0:
        raise ValueError(f"allocate_paged_kv: max_pages must be positive, got {max_pages}")

    nl = M2_KV_LAYOUT['num_layers']
    nkv = M2_KV_LAYOUT['num_kv_heads']
    hd = M2_KV_LAYOUT['head_dim']
    shape = (nl, max_pages, block_size, nkv, hd)
    sh = _kv_sharding(mesh)
    tsh = _table_sharding(mesh)

    paged_k = jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), sh)
    paged_v = jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), sh)
    block_table = jax.device_put(
        jnp.full((B, max_pages), -1, dtype=jnp.int32), tsh
    )
    return paged_k, paged_v, block_table


def compute_max_ctx_for_hbm(available_gb: int, B: int) -> int:
    """Max context length that fits into `available_gb` GB of HBM at batch B.

    Rounds down to a multiple of block_size so either the dense or paged path
    can use the result directly. Capped at the model's 196608 max_position.
    """
    if B <= 0:
        raise ValueError(f"compute_max_ctx_for_hbm: B must be positive, got {B}")
    if available_gb <= 0:
        return 0

    bpt = kv_bytes_per_token()
    avail_bytes = available_gb * (1024 ** 3)
    raw = avail_bytes // (bpt * B)
    bs = M2_KV_LAYOUT['block_size']
    ctx = int(raw // bs) * bs
    if ctx > _MAX_POSITION:
        ctx = _MAX_POSITION
    return ctx


def _fmt_bytes(n: int) -> str:
    gib = n / (1024 ** 3)
    if gib >= 1.0:
        return f"{n} bytes ({gib:.2f} GiB)"
    mib = n / (1024 ** 2)
    return f"{n} bytes ({mib:.2f} MiB)"


if __name__ == '__main__':
    bpt = kv_bytes_per_token()
    print(f"M2_KV_LAYOUT: {M2_KV_LAYOUT}")
    print(f"kv_bytes_per_token: {bpt}")
    assert bpt == 253_952, f"expected 253952 bytes/token, got {bpt}"

    # Case A: B=1, ctx=196K (paged territory).
    B_a, ctx_a = 1, 196608
    bytes_a = B_a * ctx_a * bpt
    print(f"B={B_a} ctx={ctx_a}: KV = {_fmt_bytes(bytes_a)}")

    # Case B: B=8, ctx=32K (dense).
    B_b, ctx_b = 8, 32 * 1024
    bytes_b = B_b * ctx_b * bpt
    print(f"B={B_b} ctx={ctx_b}: KV = {_fmt_bytes(bytes_b)}")

    # HBM sizing helper sanity checks.
    for gb in (64, 96, 128):
        for B in (1, 2, 8):
            ctx = compute_max_ctx_for_hbm(gb, B)
            print(f"compute_max_ctx_for_hbm(available_gb={gb}, B={B}) = {ctx}")

    # Shape sanity check without needing a real TPU mesh: just compute shapes.
    dense_shape = (
        M2_KV_LAYOUT['num_layers'], 1, _DENSE_CTX_LIMIT,
        M2_KV_LAYOUT['num_kv_heads'], M2_KV_LAYOUT['head_dim'],
    )
    paged_shape = (
        M2_KV_LAYOUT['num_layers'], 768, M2_KV_LAYOUT['block_size'],
        M2_KV_LAYOUT['num_kv_heads'], M2_KV_LAYOUT['head_dim'],
    )
    print(f"dense cache shape (B=1, ctx=32K): {dense_shape}")
    print(f"paged cache shape (max_pages=768): {paged_shape}  "
          f"= {768 * M2_KV_LAYOUT['block_size']} tokens")
