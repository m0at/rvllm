"""Marlin-style W4A16 NVFP4 dequant oracle vs rvllm dumps.

Tests whether rvllm's `combined = scale * alpha * (1/6)` kernel matches
the Marlin/Compressed-Tensors W4A16 NVFP4 numerical contract.

CT on-disk convention (verified for /home/r00t/mistral-3.5):
  weight_global_scale  fp32   stores 2688/amax (large number, e.g. 12416)
  weight_scale         fp8    per 16-elem block, exposed via .float()
  weight_packed        uint8  2 nibbles per byte along input dim
  e2m1 LUT (true vals): {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}

Marlin's effective formula:
    gs_internal = 1.0 / weight_global_scale_on_disk      # = amax/2688
    w_dequant   = e2m1[nibble] * scale_block * gs_internal

rvllm's kernel formula (mistral35_w4a16_gemv_bf16.cu:49):
    alpha       = 1.0 / weight_global_scale_on_disk      # = amax/2688
    combined    = scale_block * alpha * (1.0/6.0)
    w_dequant   = e2m1[nibble] * combined
              = e2m1[nibble] * scale_block * (amax/2688) / 6

Difference: rvllm has an extra `/6` factor. If that matches what the
kernel's e2m1 LUT returns (i.e., LUT returns normalized [-1..+1] instead
of the true [-6..+6]), the /6 is correct. Otherwise the kernel produces
6× too-small weights.

This script settles it by computing q_out two ways and comparing to
rvllm's actual q_out dump:

    Hypothesis A (Marlin-style, true e2m1 LUT):
        w = e2m1_true[nibble] * scale * (1/gs_disk)
    Hypothesis B (rvllm-style with normalized LUT):
        w = (e2m1_true[nibble]/6) * scale * (1/gs_disk)
        = e2m1_true[nibble] * scale * (1/gs_disk) / 6

Whichever one matches rvllm's dump within bf16 ULP tells us which
formula the kernel actually implements. Then we compare against the
expected Marlin-style answer.

Usage:
    python v3/tools/mistral35_marlin_oracle_check.py
"""
from __future__ import annotations
import numpy as np
import os
import sys
from safetensors import safe_open

MODEL_DIR = "/home/r00t/mistral-3.5"
DUMP_DIR = "/tmp/rvllm-mistral35-dump"
LAYER = 0
HIDDEN = 12288
N_Q_HEADS = 96
HEAD_DIM = 128
Q_OUT_DIM = N_Q_HEADS * HEAD_DIM  # 12288

# True NVFP4 e2m1 lookup table.
# Nibble layout: bit3=sign, bits2..0=magnitude index into 8 values.
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def load_q_proj_layer0() -> tuple[np.ndarray, np.ndarray, float]:
    """Load layer-0 q_proj weights. Returns (packed, scale_block, gs_disk)."""
    # Hunt across all shards for the three keys.
    target_prefix = f"model.language_model.layers.{LAYER}.self_attn.q_proj"
    target_keys = {
        f"{target_prefix}.weight_packed": None,
        f"{target_prefix}.weight_scale": None,
        f"{target_prefix}.weight_global_scale": None,
    }
    for fname in sorted(os.listdir(MODEL_DIR)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(MODEL_DIR, fname), framework="pt") as f:
            for key in list(target_keys):
                if target_keys[key] is None and key in f.keys():
                    target_keys[key] = f.get_tensor(key)
        if all(v is not None for v in target_keys.values()):
            break
    for key, val in target_keys.items():
        assert val is not None, f"missing {key}"
    packed = target_keys[f"{target_prefix}.weight_packed"].numpy()
    scale_block = target_keys[f"{target_prefix}.weight_scale"]
    gs = target_keys[f"{target_prefix}.weight_global_scale"]
    # Convert fp8_e4m3fn → fp32. PyTorch supports `.float()`.
    scale_block_f32 = scale_block.float().numpy()
    gs_f32 = float(gs.float().item())
    return packed, scale_block_f32, gs_f32


def unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8[N, K/2] → float32[N, K] of TRUE e2m1 values in [-6..+6]."""
    # Each byte holds two nibbles. Convention: low nibble first, then high.
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Interleave: out[..., 2k] = low_nibble(byte_k), out[..., 2k+1] = high_nibble(byte_k)
    nibbles = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
    sign_bit = (nibbles >> 3) & 1
    mag_idx = nibbles & 0x07
    mag = E2M1_MAG[mag_idx]
    return np.where(sign_bit == 1, -mag, mag)


def expand_block_scales(scale_block: np.ndarray, k_full: int, group_size: int = 16) -> np.ndarray:
    """[N, K/16] → [N, K] by repeating each block scale 16×."""
    return np.repeat(scale_block, group_size, axis=1)[:, :k_full]


def load_bf16(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()))


def main():
    print(f"== Mistral 3.5 W4A16 NVFP4 oracle vs rvllm dump ==\n")
    packed, scale_block, gs_disk = load_q_proj_layer0()
    print(f"  weight_packed:        {packed.shape} {packed.dtype}")
    print(f"  weight_scale_block:   {scale_block.shape} (fp8→f32, max={scale_block.max():.3f})")
    print(f"  weight_global_scale:  {gs_disk:.3f}  (CT convention: 2688/amax)")
    print(f"  → derived amax = 2688/{gs_disk:.3f} = {2688/gs_disk:.5f}\n")

    # Step 1: unpack nibbles to TRUE e2m1 values [-6..+6].
    e2m1_true = unpack_e2m1(packed)  # [12288, 12288]
    print(f"  unpacked nibbles:     {e2m1_true.shape}, range=[{e2m1_true.min():.1f}..{e2m1_true.max():.1f}]")

    # Step 2: per-block scale broadcast to per-element.
    scale_full = expand_block_scales(scale_block, e2m1_true.shape[1])
    assert scale_full.shape == e2m1_true.shape
    print(f"  expanded scales:      {scale_full.shape}")

    # Step 3: dequant under both hypotheses.
    gs_internal = 1.0 / gs_disk  # = amax/2688

    print(f"\n  HYP A (Marlin-style):  w = e2m1 * scale * (amax/2688)")
    w_marlin = e2m1_true * scale_full * gs_internal

    print(f"  HYP B (rvllm-style):   w = e2m1 * scale * (amax/2688) / 6")
    w_rvllm = w_marlin / 6.0

    print(f"  HYP A weight rms:     {rms(w_marlin):.6f}")
    print(f"  HYP B weight rms:     {rms(w_rvllm):.6f}\n")

    # Step 4: load post-rmsnorm input (rvllm dump).
    post_rms_path = os.path.join(DUMP_DIR, "post_rmsnorm.f32")
    if not os.path.isfile(post_rms_path):
        print(f"  MISSING: {post_rms_path}")
        sys.exit(1)
    post_rms = np.fromfile(post_rms_path, dtype=np.float32)
    print(f"  post_rmsnorm input:   {post_rms.shape} ({post_rms.size/HIDDEN:.1f} tokens)")

    # The dump may be one or many tokens. Take the LAST token (the one
    # that produced rvllm's q_out we'll compare against).
    n_tok = post_rms.size // HIDDEN
    if post_rms.size % HIDDEN != 0:
        print(f"  WARN: post_rmsnorm size {post_rms.size} not multiple of {HIDDEN}")
    x_last = post_rms[-HIDDEN:].astype(np.float32)  # [12288]
    print(f"  using last token, x rms={rms(x_last):.6f}")

    # Step 5: matmul.
    q_marlin = w_marlin @ x_last  # [12288]
    q_rvllm = w_rvllm @ x_last
    print(f"\n  q_marlin rms:         {rms(q_marlin):.6f}")
    print(f"  q_rvllm  rms:         {rms(q_rvllm):.6f}")

    # Step 6: compare to rvllm's actual q_out dump.
    # Prefer pre-RoPE if available, else q_out.f32.
    cand = [
        ("boundary_qproj_pre_rope.bf16", "bf16"),
        ("q_out_layer0.bf16", "bf16"),
        ("q_out.f32", "f32"),
    ]
    rvllm_q = None
    for fname, fmt in cand:
        p = os.path.join(DUMP_DIR, fname)
        if os.path.isfile(p):
            rvllm_q = load_bf16(p) if fmt == "bf16" else np.fromfile(p, dtype=np.float32)
            print(f"\n  rvllm dump:           {fname}  shape={rvllm_q.shape}")
            break
    if rvllm_q is None:
        print("\n  no rvllm q dump found")
        sys.exit(1)

    # Possibly multi-token; take last.
    if rvllm_q.size > Q_OUT_DIM and rvllm_q.size % Q_OUT_DIM == 0:
        rvllm_q = rvllm_q[-Q_OUT_DIM:]
    if rvllm_q.size != Q_OUT_DIM:
        print(f"  WARN: rvllm dump size {rvllm_q.size} ≠ {Q_OUT_DIM}")
    print(f"  rvllm q rms:          {rms(rvllm_q):.6f}")

    print(f"\n  ====== verdict ======")
    cos_a = cos(q_marlin, rvllm_q)
    cos_b = cos(q_rvllm, rvllm_q)
    rA = rms(q_marlin) / max(rms(rvllm_q), 1e-30)
    rB = rms(q_rvllm) / max(rms(rvllm_q), 1e-30)
    print(f"  HYP A (Marlin)  vs rvllm dump:  cos={cos_a:.6f}  rms_ratio={rA:.4f}")
    print(f"  HYP B (rvllm /6) vs rvllm dump: cos={cos_b:.6f}  rms_ratio={rB:.4f}")

    print(f"\n  Interpretation:")
    print(f"    rms_ratio ≈ 1.0 + cos ≈ 1.0 → that's the formula rvllm executes")
    print(f"    rms_ratio ≈ 6.0           → rvllm path is /6 too small (vs Marlin)")
    print(f"    rms_ratio ≈ 1/6 = 0.167   → rvllm path is 6× too big (vs Marlin)")


if __name__ == "__main__":
    main()
