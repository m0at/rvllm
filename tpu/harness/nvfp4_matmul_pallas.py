"""Opt-in Pallas matmul backend for MiniMax NVFP4 weights on TPU.

This preserves the known-good JAX NVFP4 dequant path, then routes the bf16
tile matmul through JAX's official TPU Pallas matmul kernel. It is useful as a
correctness-preserving experiment, but current v6e-8 measurements are slower
than the default XLA dot path for B=8.
"""

from __future__ import annotations

import os

import jax.numpy as jnp


def _env_int(name, default, multiple):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0 or value % multiple != 0:
        raise ValueError(f"{name} must be a positive multiple of {multiple}, got {value}")
    return value


def _pallas_dot_2d(x_2d, w_bf16, BM, BN):
    from jax.experimental.pallas.ops.tpu.matmul import matmul as tpu_matmul

    M = x_2d.shape[0]
    K = x_2d.shape[1]
    pad_M = (BM - M % BM) % BM
    if pad_M:
        x_2d = jnp.pad(x_2d, ((0, pad_M), (0, 0)))

    block_k = 256 if K % 256 == 0 else K
    out = tpu_matmul(
        x_2d,
        jnp.swapaxes(w_bf16, 0, 1),
        block_shape=(BM, BN),
        block_k=block_k,
        out_dtype=jnp.float32,
    )
    if pad_M:
        out = out[:M]
    return out


def nvfp4_matmul_pallas(x_bf16, w_packed, w_scales, w_scale_2,
                        out_features, in_features,
                        BM=None, BN=None, BK=None, GS=16):
    """Dequant NVFP4 weights with JAX, then run the bf16 dot with TPU Pallas."""
    from nvfp4_jax_ops import nvfp4_to_bf16_jax

    BM = _env_int("RVLLM_NVFP4_PALLAS_BM", 128, 1) if BM is None else BM
    BN = _env_int("RVLLM_NVFP4_PALLAS_BN", 256, 16) if BN is None else BN
    BK = _env_int("RVLLM_NVFP4_PALLAS_BK", 512, GS) if BK is None else BK
    N = out_features
    K = in_features
    if N % BN != 0:
        raise ValueError(f"out_features {N} must be multiple of BN {BN}")

    leading = x_bf16.shape[:-1]
    M = 1
    for d in leading:
        M *= d
    x_2d = x_bf16.reshape(M, K)

    if K <= BK or (K % BK) != 0 or (BK % GS) != 0:
        w_bf16 = nvfp4_to_bf16_jax(w_packed, w_scales, w_scale_2, (N, K))
        out = _pallas_dot_2d(x_2d, w_bf16, BM, BN)
        return out.reshape(*leading, N).astype(jnp.bfloat16)

    n_blk = K // BK
    xb = x_2d.reshape(M, n_blk, BK)
    wp = w_packed.reshape(N, n_blk, BK // 2)
    ws = w_scales.reshape(N, n_blk, BK // GS)
    acc = jnp.zeros((M, N), dtype=jnp.float32)
    for i in range(n_blk):
        w_tile = nvfp4_to_bf16_jax(wp[:, i, :], ws[:, i, :], w_scale_2, (N, BK))
        acc = acc + _pallas_dot_2d(xb[:, i, :], w_tile, BM, BN)
    return acc.reshape(*leading, N).astype(jnp.bfloat16)
