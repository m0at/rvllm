"""Pallas kernel for fused NVFP4 -> MXU matmul on TPU v6e (Agent 14).

Replaces `nvfp4_jax_ops.nvfp4_matmul` for Path A perf. Instead of the pure-JAX
tiled scan (which dequants to bf16 in an intermediate XLA buffer per K-block),
this kernel loads packed uint8 + FP8 E4M3 scales into SRAM tiles, dequants to
bf16 in registers, then pipelines bf16 x bf16 MXU dot_general with f32 accum.

Usage:
    from nvfp4_matmul_pallas import nvfp4_matmul_pallas
    y = nvfp4_matmul_pallas(x_bf16, w_packed_u8, w_scales_u8, w_scale_2_f32,
                            out_features, in_features)

Tile choice:
    BM=128, BN=256, BK=512 — matches v6e MXU 128x128 tile granularity.
    GS=16 = NVFP4 scale group size.

SRAM budget per tile pair:
    x tile:       BM*BK  bf16 = 128*512*2  = 128 KiB
    packed tile:  BK*BN/2 u8  = 512*128     = 64  KiB
    scale tile:   BK/GS*BN u8 = 32*256      = 8   KiB
    accum:        BM*BN  f32  = 128*256*4   = 128 KiB
    total         ~= 330 KiB (fits easily in v6e per-core VMEM).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# FP4 E2M1 decode LUT (bf16) — indices 0..15 -> signed FP4 values.
NVFP4_LUT = jnp.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=jnp.bfloat16,
)


def _fp8_e4m3_to_bf16_vec(bits_u8):
    """Fast FP8 E4M3 -> bf16 via IEEE-754 f32 bit layout (Agent 15 trick).

    Rebias E4M3 exp (bias 7) to f32 exp (bias 127) by adding 120, place
    3-bit mantissa into f32 mantissa bits 20..22, bitcast to f32, cast to bf16.
    """
    b = bits_u8.astype(jnp.uint32)
    sign = (b >> 7) & jnp.uint32(1)
    exp = (b >> 3) & jnp.uint32(0xF)
    mant = b & jnp.uint32(0x7)
    sign_bit = sign << jnp.uint32(31)
    u_norm = sign_bit | ((exp + jnp.uint32(120)) << jnp.uint32(23)) | (mant << jnp.uint32(20))
    val_norm = jax.lax.bitcast_convert_type(u_norm, jnp.float32)
    # Subnormal: exp==0 -> (mant/8) * 2^-6
    sub = mant.astype(jnp.float32) * jnp.float32((1.0 / 8.0) * (2.0 ** -6))
    sign_f = jnp.where(sign == jnp.uint32(1), jnp.float32(-1.0), jnp.float32(1.0))
    val = jnp.where(exp == jnp.uint32(0), sign_f * sub, val_norm)
    return val.astype(jnp.bfloat16)


def _nvfp4_dequant_tile(packed_tile, scale_tile, global_scale, rows, cols):
    """Dequant one (rows, cols/2) packed + (rows, cols/16) scale tile -> (rows, cols) bf16."""
    lo = jnp.bitwise_and(packed_tile, jnp.uint8(0x0F))
    hi = jnp.bitwise_and(jnp.right_shift(packed_tile, jnp.uint8(4)), jnp.uint8(0x0F))
    nibbles = jnp.stack([lo, hi], axis=-1).reshape(rows, cols)
    fp4_bf16 = NVFP4_LUT[nibbles.astype(jnp.int32)]
    scale_bf16 = _fp8_e4m3_to_bf16_vec(scale_tile)
    scale_bf16 = scale_bf16 * jnp.asarray(global_scale, dtype=jnp.bfloat16)
    scale_expanded = jnp.repeat(scale_bf16, 16, axis=1)
    return fp4_bf16 * scale_expanded


def _kernel_body(x_ref, wq_ref, ws_ref, gs_ref, acc_ref, *, BK, GS):
    """Per-tile body executed inside Pallas block grid.
    Dequants one BK-slice of W and accumulates x @ W^T into `acc_ref`.
    """
    x = x_ref[...]                   # (BM, BK) bf16
    wq = wq_ref[...]                 # (BN, BK/2) u8
    ws = ws_ref[...]                 # (BN, BK/GS) u8
    gs = gs_ref[...]                 # scalar f32
    # NVFP4 dequant -> (BN, BK) bf16
    BN, BK_over_2 = wq.shape
    BK_ = BK_over_2 * 2
    w_bf = _nvfp4_dequant_tile(wq, ws, gs, BN, BK_)  # (BN, BK) bf16
    # x: (BM, BK), w_bf: (BN, BK). Compute x @ w_bf.T -> (BM, BN) f32.
    part = jax.lax.dot_general(
        x, w_bf,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    # Accumulate into acc_ref.
    acc_ref[...] = acc_ref[...] + part


def nvfp4_matmul_pallas(x_bf16, w_packed, w_scales, w_scale_2,
                       out_features, in_features,
                       BM=128, BN=256, BK=512, GS=16):
    """Pallas fused NVFP4 -> MXU matmul.

    x_bf16:    (..., in_features) bf16
    w_packed:  (out_features, in_features/2) uint8
    w_scales:  (out_features, in_features/16) uint8
    w_scale_2: () f32 (scalar)
    Returns:   (..., out_features) bf16

    Tiles:
        M -> BM (parallel)
        N -> BN (parallel)
        K -> BK (reduction, serial accum into f32 acc)
    """
    # Reshape inputs to 2D for the kernel; preserve leading batch dims via reshape.
    leading = x_bf16.shape[:-1]
    M = 1
    for d in leading:
        M *= d
    x_2d = x_bf16.reshape(M, in_features)

    N = out_features
    K = in_features
    assert K % BK == 0, f"in_features {K} must be multiple of BK {BK}"
    assert N % BN == 0, f"out_features {N} must be multiple of BN {BN}"
    # M doesn't need to align to BM; we pad if needed.
    pad_M = (BM - M % BM) % BM
    if pad_M:
        x_2d = jnp.pad(x_2d, ((0, pad_M), (0, 0)))
    M_padded = M + pad_M

    gs = jnp.asarray(w_scale_2, dtype=jnp.float32)

    def kernel(x_ref, wq_ref, ws_ref, gs_ref, acc_ref):
        _kernel_body(x_ref, wq_ref, ws_ref, gs_ref, acc_ref, BK=BK, GS=GS)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M_padded, N), jnp.float32),
        grid=(M_padded // BM, N // BN, K // BK),
        in_specs=[
            pl.BlockSpec((BM, BK),      lambda i, j, k: (i, k)),        # x
            pl.BlockSpec((BN, BK // 2), lambda i, j, k: (j, k)),        # packed weight
            pl.BlockSpec((BN, BK // GS), lambda i, j, k: (j, k)),       # scales
            pl.BlockSpec(memory_space=pl.ANY),                           # global scale (scalar)
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j, k: (i, j)),
    )(x_2d, w_packed, w_scales, gs)

    if pad_M:
        out = out[:M]
    return out.reshape(*leading, N).astype(jnp.bfloat16)


if __name__ == "__main__":
    import numpy as np
    jax.config.update("jax_platforms", "cpu")

    rng = np.random.default_rng(0)
    M, K, N = 2, 128, 64
    x = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    wp = rng.integers(0, 256, size=(N, K // 2), dtype=np.uint8)
    ws = rng.integers(0, 256, size=(N, K // 16), dtype=np.uint8)
    # Clamp scales to avoid NaN for the CPU smoke
    ws = np.where((ws == 0x7F) | (ws == 0xFF), np.uint8(0x38), ws).astype(np.uint8)

    x_bf = jnp.asarray(x).astype(jnp.bfloat16)
    try:
        y = nvfp4_matmul_pallas(x_bf, jnp.asarray(wp), jnp.asarray(ws),
                              jnp.float32(1.0), N, K,
                              BM=2, BN=64, BK=128)
        print("pallas output:", y.shape, y.dtype)
    except Exception as e:
        # Pallas TPU kernels don't run on CPU — expected.
        print(f"(pallas compile/run requires TPU: {type(e).__name__}: {e})")
