#!/usr/bin/env python3
"""NVFP4 on-the-fly JAX dequant (path A).

Pure JAX ops. No custom XLA calls. Traceable under jax.jit.

Layout (per modelopt producer):
  - packed:  uint8, shape (rows, cols/2). Low nibble = even index, high = odd.
  - scales:  uint8 (FP8 E4M3 bits), shape (rows, cols/16). One scale per 16 elements.
  - FP4 E2M1 decode table (signed, 16 values):
      {+/-{0, 0.5, 1, 1.5, 2, 3, 4, 6}}
  - Effective bf16 weight = fp4_decoded * fp8_scale_decoded (both as bf16).

Exports:
  nvfp4_to_bf16_jax(packed, scales, global_scale, out_shape) -> bf16
  nvfp4_matmul(x_bf16, w_packed, w_scales, w_scale_2, out_features, in_features) -> bf16

Per modelopt NVFP4 two-level scaling: decoded = fp4_lut * fp8_block_scale * global_scale.
The `global_scale` / `w_scale_2` is a per-tensor FP32 scalar.
"""
import os

import jax
import jax.numpy as jnp


# FP4 E2M1 lookup: indices 0..15 map to signed values listed in the spec.
_FP4_LUT = jnp.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=jnp.bfloat16,
)


def _env_k_block(default):
    raw = os.environ.get("RVLLM_NVFP4_K_BLOCK", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"RVLLM_NVFP4_K_BLOCK must be an integer multiple of 16, got {raw!r}"
        ) from exc
    if value <= 0 or value % 16 != 0:
        raise ValueError(
            f"RVLLM_NVFP4_K_BLOCK must be a positive multiple of 16, got {value}"
        )
    return value


def _fp8_e4m3_to_f32(bits):
    """FP8 E4M3 -> f32 via direct IEEE-754 f32 bit layout (Agent 15).

    Rebias E4M3 exp (bias 7) to f32 exp (bias 127) by adding 120, and shift
    3-bit mantissa into f32 mantissa bits 20..22. Avoids `jnp.power`/`jnp.exp2`
    which XLA can't lower to a cheap integer shift.
    """
    b = bits.astype(jnp.uint32)
    sign = (b >> 7) & jnp.uint32(1)
    exp = (b >> 3) & jnp.uint32(0xF)
    mant = b & jnp.uint32(0x7)
    sign_bit = sign << jnp.uint32(31)
    u_norm = sign_bit | ((exp + jnp.uint32(120)) << jnp.uint32(23)) | (mant << jnp.uint32(20))
    val_norm = jax.lax.bitcast_convert_type(u_norm, jnp.float32)
    # Subnormal (exp==0): (mant/8) * 2^-6
    sub = mant.astype(jnp.float32) * jnp.float32((1.0 / 8.0) * (2.0 ** -6))
    sign_f = jnp.where(sign == jnp.uint32(1), jnp.float32(-1.0), jnp.float32(1.0))
    val = jnp.where(exp == jnp.uint32(0), sign_f * sub, val_norm)
    is_nan = (exp == jnp.uint32(0xF)) & (mant == jnp.uint32(0x7))
    return jnp.where(is_nan, jnp.float32("nan"), val)


def nvfp4_to_bf16_jax(packed, scales, global_scale, out_shape):
    """Dequant packed NVFP4 (+ FP8 E4M3 scales + FP32 global scale) -> bf16.

    packed:       uint8, shape (rows, cols/2)
    scales:       uint8, shape (rows, cols/16)
    global_scale: f32 scalar (modelopt per-tensor weight_scale_2)
    out_shape:    (rows, cols)
    """
    rows, cols = out_shape
    if cols % 16 != 0:
        raise ValueError(f"cols must be multiple of 16, got {cols}")

    packed = packed.astype(jnp.uint8)
    lo = jnp.bitwise_and(packed, jnp.uint8(0x0F))
    hi = jnp.bitwise_and(jnp.right_shift(packed, jnp.uint8(4)), jnp.uint8(0x0F))

    # Interleave lo/hi along col axis: even index = lo, odd = hi.
    stacked = jnp.stack([lo, hi], axis=-1)          # (rows, half, 2)
    nibbles = stacked.reshape(rows, cols)           # (rows, cols)

    fp4_bf16 = jnp.take(_FP4_LUT, nibbles.astype(jnp.int32), axis=0)  # (rows, cols) bf16

    # Decode block scales: uint8 FP8 E4M3 -> f32. Fold per-tensor global_scale
    # in FP32 then cast to bf16 before broadcasting across the 16-element block.
    scale_f32 = _fp8_e4m3_to_f32(scales)            # (rows, cols/16)
    gs = jnp.asarray(global_scale, dtype=jnp.float32)
    scale_bf16 = (scale_f32 * gs).astype(jnp.bfloat16)
    scale_expanded = jnp.repeat(scale_bf16, 16, axis=1)  # (rows, cols)

    return fp4_bf16 * scale_expanded


def nvfp4_matmul(x_bf16, w_packed, w_scales, w_scale_2, out_features, in_features,
                 k_block=None):
    """Fused: dequant W on the fly in K-blocks, then x @ W^T.

    Tiles along the reduction (K / in_features) axis via lax.scan so the full
    (out_features, in_features) bf16 intermediate never materializes. XLA only
    holds one K_BLOCK-sized tile live per step — peak ~3 MB for out=3072 K=512
    vs ~18 MB for the whole weight. Fix (Agent 9) for the 36.84G HBM compile OOM.

    x_bf16:    (..., in_features) bf16
    w_packed:  (out_features, in_features/2) uint8
    w_scales:  (out_features, in_features/16) uint8
    w_scale_2: f32 scalar (modelopt per-tensor global scale)
    Returns:   (..., out_features) bf16
    """
    backend = os.environ.get("RVLLM_NVFP4_BACKEND", "jax").strip().lower()
    if backend == "pallas":
        from nvfp4_matmul_pallas import nvfp4_matmul_pallas
        return nvfp4_matmul_pallas(
            x_bf16, w_packed, w_scales, w_scale_2, out_features, in_features)
    if backend not in ("", "jax"):
        raise ValueError(f"unknown RVLLM_NVFP4_BACKEND={backend!r}")

    if k_block is None:
        k_block = _env_k_block(512)

    # If in_features is too small to tile, fall back to single-shot dequant.
    if in_features <= k_block or (in_features % k_block) != 0 or (k_block % 16) != 0:
        w_bf16 = nvfp4_to_bf16_jax(
            w_packed, w_scales, w_scale_2, (out_features, in_features))
        out = jax.lax.dot_general(
            x_bf16, w_bf16,
            dimension_numbers=(((x_bf16.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=jnp.bfloat16,
        )
        return out

    n_blk = in_features // k_block
    # Reshape inputs/weights for per-block scan.
    # x_bf16: (..., in_features) -> (..., n_blk, k_block) -> scan over axis -2
    leading = x_bf16.shape[:-1]
    xb = x_bf16.reshape(*leading, n_blk, k_block)                     # (..., n_blk, k_block)
    xb = jnp.moveaxis(xb, -2, 0)                                      # (n_blk, ..., k_block)
    wp = w_packed.reshape(out_features, n_blk, k_block // 2)          # (out, n_blk, kb/2)
    wp = jnp.moveaxis(wp, 1, 0)                                       # (n_blk, out, kb/2)
    ws = w_scales.reshape(out_features, n_blk, k_block // 16)         # (out, n_blk, kb/16)
    ws = jnp.moveaxis(ws, 1, 0)                                       # (n_blk, out, kb/16)

    gs = jnp.asarray(w_scale_2, dtype=jnp.float32)

    def step(acc, tup):
        xk, wpk, wsk = tup
        # Dequant just this K-block.
        w_tile = nvfp4_to_bf16_jax(wpk, wsk, gs, (out_features, k_block))   # (out, kb) bf16
        # Contract last axis of xk with last axis of w_tile, accumulate in f32.
        part = jax.lax.dot_general(
            xk, w_tile,
            dimension_numbers=(((xk.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return acc + part, None

    acc0 = jnp.zeros((*leading, out_features), dtype=jnp.float32)
    acc, _ = jax.lax.scan(step, acc0, (xb, wp, ws))
    return acc.astype(jnp.bfloat16)


if __name__ == "__main__":
    # Run pure-CPU smoke test.
    jax.config.update("jax_platforms", "cpu")

    import numpy as np

    rng = np.random.default_rng(0)

    # Small shapes for quick validation.
    rows, cols = 4, 64
    in_features, out_features = 64, 8

    packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(rows, cols // 16), dtype=np.uint8)
    # Clamp scales so we avoid NaN in the tiny demo (NaN pattern = 0x7F/0xFF).
    scales = np.where((scales == 0x7F) | (scales == 0xFF), np.uint8(0x38), scales).astype(np.uint8)

    packed_j = jnp.asarray(packed)
    scales_j = jnp.asarray(scales)

    jit_deq = jax.jit(nvfp4_to_bf16_jax, static_argnums=(3,))
    gs = jnp.float32(1.0)
    w_bf16 = jit_deq(packed_j, scales_j, gs, (rows, cols))
    print("dequant shape:", w_bf16.shape, "dtype:", w_bf16.dtype)

    # Hand-computed reference for a single 16-element block of row 0.
    fp4_values = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    row0 = packed[0]
    block0 = []
    for b in range(8):  # 8 bytes -> 16 FP4 values -> one group
        byte = int(row0[b])
        lo_i = byte & 0x0F
        hi_i = (byte >> 4) & 0x0F
        block0.append(fp4_values[lo_i])
        block0.append(fp4_values[hi_i])

    # Reference FP8 E4M3 decode for scales[0, 0].
    def _ref_fp8_e4m3(b):
        sign = (b >> 7) & 1
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 0xF and mant == 0x7:
            return float("nan")
        s = -1.0 if sign == 1 else 1.0
        if exp == 0:
            return s * (mant / 8.0) * (2.0 ** -6)
        return s * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))

    s0 = _ref_fp8_e4m3(int(scales[0, 0]))
    ref_block = np.array([v * s0 for v in block0], dtype=np.float32)
    got_block = np.asarray(w_bf16[0, :16]).astype(np.float32)

    # bf16 quant error: ~2^-7 relative. Tolerance here is generous.
    max_abs = float(np.max(np.abs(got_block - ref_block)))
    max_ref = float(np.max(np.abs(ref_block)) + 1e-12)
    print(f"block0 max_abs_err={max_abs:.6f}  max_ref={max_ref:.6f}")
    assert max_abs <= max(1e-2, 1e-2 * max_ref), "block0 mismatch"

    # Now exercise the fused matmul.
    wp = rng.integers(0, 256, size=(out_features, in_features // 2), dtype=np.uint8)
    ws = rng.integers(0, 256, size=(out_features, in_features // 16), dtype=np.uint8)
    ws = np.where((ws == 0x7F) | (ws == 0xFF), np.uint8(0x38), ws).astype(np.uint8)
    x = rng.standard_normal((2, in_features)).astype(np.float32)
    x_bf = jnp.asarray(x).astype(jnp.bfloat16)

    jit_mm = jax.jit(nvfp4_matmul, static_argnums=(4, 5))
    y = jit_mm(x_bf, jnp.asarray(wp), jnp.asarray(ws), jnp.float32(1.0),
              out_features, in_features)
    print("matmul shape:", y.shape, "dtype:", y.dtype)
    print("ok")
