"""Pure-JAX NVFP4 decode + matmul, mirroring v3/crates/rvllm-fused/src/m2_nvfp4.rs.

The .pyc in tpu/harness/__pycache__/ was the only artifact left of an older
external module; this is a from-scratch reimplementation following the Rust
ground truth (decode_fp4_e2m1, decode_fp8_e4m3, nvfp4_weight_at,
nvfp4_matmul_ref). Used by tpu/harness/m2_full_bench.py and the JAX baseline
harness on the TPU VM at /workspace/runs/m2bench/tpu/harness/.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp


# FP4 E2M1 lookup table -- 16 entries, indexed by 4-bit nibble.
# Mirrors v3/crates/rvllm-fused/src/m2_nvfp4.rs::FP4_E2M1_LUT.
_FP4_LUT = jnp.array(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=jnp.float32,
)


def _env_k_block(default: int = 16) -> int:
    """NVFP4 group size; default 16 matches the M2 / Rust constant."""
    raw = os.environ.get("RVLLM_M2_NVFP4_K_BLOCK")
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _fp8_e4m3_to_f32(bits):
    """Decode FP8 E4M3 (1 sign, 4 exp, 3 mantissa) to f32.

    Mirrors decode_fp8_e4m3 in m2_nvfp4.rs:
      sign = +1 if msb==0 else -1
      exp  = (bits >> 3) & 0x0f
      mant = bits & 0x07
      if exp == 0xf and mant == 7: NaN
      if exp == 0:     sign * (mant/8) * 2^-6  (subnormal)
      else:            sign * (1 + mant/8) * 2^(exp - 7)
    """
    b = bits.astype(jnp.uint32)
    sign = jnp.where((b & 0x80) == 0, 1.0, -1.0).astype(jnp.float32)
    exp = ((b >> 3) & 0x0f).astype(jnp.int32)
    mant_int = (b & 0x07).astype(jnp.int32)
    mant = mant_int.astype(jnp.float32)
    # Subnormal path (exp == 0)
    subnormal = sign * (mant / 8.0) * (2.0 ** -6)
    # Normal path
    exp_f = exp.astype(jnp.float32) - 7.0
    normal = sign * (1.0 + mant / 8.0) * (jnp.float32(2.0) ** exp_f)
    is_nan = (exp == 0x0f) & (mant_int == 7)
    val = jnp.where(exp == 0, subnormal, normal)
    val = jnp.where(is_nan, jnp.float32("nan"), val)
    return val


def nvfp4_to_bf16_jax(packed, scales, global_scale, out_shape):
    """Decode an NVFP4 weight matrix to bf16.

    Args:
      packed: u8 array, last dim must be K/2 (each byte holds two FP4 nibbles).
              Layout: packed[n, k//2] = (high<<4) | low; column k uses low if
              k is even, high if k is odd. Mirrors nvfp4_weight_at byte/nibble
              indexing in m2_nvfp4.rs.
      scales: u8 (FP8 E4M3) array, last dim K/16, one scale per 16-element block.
      global_scale: scalar f32 (or 0-D array convertible to it).
      out_shape: (N, K) target shape of the decoded weight.

    Returns bf16 array of shape (N, K).
    """
    N, K = int(out_shape[0]), int(out_shape[1])
    if K % 16 != 0:
        raise ValueError(f"K={K} must be a multiple of 16 (NVFP4 group size)")
    if K % 2 != 0:
        raise ValueError(f"K={K} must be even (FP4 packing)")

    packed = jnp.asarray(packed).reshape(N, K // 2).astype(jnp.uint32)
    scales = jnp.asarray(scales).reshape(N, K // 16).astype(jnp.uint32)
    gs = jnp.asarray(global_scale, dtype=jnp.float32)

    low = (packed & 0x0f).astype(jnp.int32)
    high = ((packed >> 4) & 0x0f).astype(jnp.int32)
    # Interleave so column k uses low at even k, high at odd k
    nibbles = jnp.stack([low, high], axis=-1).reshape(N, K)
    fp4_vals = _FP4_LUT[nibbles]  # (N, K) f32

    fp8_scales = _fp8_e4m3_to_f32(scales)  # (N, K/16) f32
    fp8_scales_expanded = jnp.repeat(fp8_scales, 16, axis=1)  # (N, K)

    decoded = fp4_vals * fp8_scales_expanded * gs
    return decoded.astype(jnp.bfloat16)


def nvfp4_matmul(x_bf16, w_packed, w_scales, w_scale_2, out_features,
                 in_features, k_block=None):
    """x @ decoded_w.T using NVFP4-encoded weights.

    Args:
      x_bf16: (..., in_features) bf16/f32 input.
      w_packed: NVFP4 packed weight (u8), last dim in_features/2.
      w_scales: FP8 E4M3 scales (u8), last dim in_features/16.
      w_scale_2: scalar f32 global scale.
      out_features: N.
      in_features:  K.
      k_block: NVFP4 group; default reads RVLLM_M2_NVFP4_K_BLOCK or 16.

    Returns bf16 array of shape (..., out_features).
    """
    if k_block is None:
        k_block = _env_k_block(16)
    if k_block != 16:
        # The decode_fp8_e4m3-based path assumes 16-element groups (the M2
        # NVFP4_GROUP constant). A different block would need a generalized
        # repeat factor; fail loudly so we don't silently produce wrong values.
        raise ValueError(
            f"nvfp4_matmul: k_block={k_block} not supported; only 16 is M2-canonical"
        )
    decoded = nvfp4_to_bf16_jax(
        w_packed, w_scales, w_scale_2, (out_features, in_features)
    )
    x = jnp.asarray(x_bf16).astype(jnp.bfloat16)
    # x: (..., K), decoded: (N, K) -> output (..., N)
    return jnp.einsum("...k,nk->...n", x, decoded).astype(jnp.bfloat16)


# Tiny self-test against a known fixture so the file fails loudly if the
# math drifts. Mirrors v3/crates/rvllm-fused/src/m2_nvfp4.rs::tests::scalar_decode.
if __name__ == "__main__":
    # FP4 LUT spot-checks
    import numpy as np

    assert _FP4_LUT[0] == 0.0
    assert _FP4_LUT[1] == 0.5
    assert _FP4_LUT[7] == 6.0
    assert _FP4_LUT[15] == -6.0
    # FP8 E4M3 0x38 = sign 0, exp 0b0111=7, mant 0 -> 1.0
    assert float(_fp8_e4m3_to_f32(jnp.uint8(0x38))) == 1.0, "FP8 0x38 != 1.0"
    # FP8 E4M3 0x40 = sign 0, exp 0b1000=8, mant 0 -> 2.0
    assert float(_fp8_e4m3_to_f32(jnp.uint8(0x40))) == 2.0, "FP8 0x40 != 2.0"

    # Round-trip: shape (1, 16) packed nibble all-1s with FP8 scale 0x38=1.0,
    # global_scale=3.0 => decoded weight = 0.5 * 1.0 * 3.0 = 1.5.
    packed = np.zeros((1, 8), dtype=np.uint8)
    packed[:, :] = 0x11  # nibble 1 = 0.5 in both halves
    scales = np.full((1, 1), 0x38, dtype=np.uint8)
    decoded = nvfp4_to_bf16_jax(packed, scales, 3.0, (1, 16))
    expected = np.full((1, 16), 1.5, dtype=np.float32)
    diff = float(jnp.max(jnp.abs(decoded.astype(jnp.float32) - expected)))
    assert diff < 1e-3, f"decode mismatch: {decoded}"

    # Tiny matmul check: x = [1, 1, ..., 1] (16 ones) @ decoded.T = [1.5*16] = 24
    x = np.ones((1, 16), dtype=np.float32)
    y = nvfp4_matmul(x, packed, scales, 3.0, 1, 16)
    print("matmul self-test:", float(y[0, 0]))
    assert abs(float(y[0, 0]) - 24.0) < 0.5, f"matmul self-test failed: {y}"

    print("nvfp4_jax_ops self-tests passed")
