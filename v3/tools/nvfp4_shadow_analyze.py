#!/usr/bin/env python3
# Analyze /tmp/nvfp4_shadow dumps from RVLLM_NVFP4_SHADOW_F16=1 +
# RVLLM_NVFP4_SHADOW_LAYERS=<csv>.
#
# Per dumped layer: dequant NVFP4 K and V back to f32, compare against
# the f16 shadow ground truth, report RMSE / relative error / max-abs
# error. Used to identify which layers carry the highest quantization
# noise so they can be FP8-overridden via RVLLM_FP8_KV_LAYERS.
#
# NVFP4 layout: each 16-elem block has one E4M3 scale.
# Packed bytes: 2 elems per byte (low nibble first, high second).
# E2M1 magnitude set: {0, 0.5, 1, 1.5, 2, 3, 4, 6}; sign bit on top.

import json
import pathlib
import sys

import numpy as np

DUMP = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/nvfp4_shadow")
META = json.loads((DUMP / "meta.json").read_text())

# E2M1 magnitude lookup, indexed by magnitude bits 0..7
# (sign + 3-bit magnitude packed into the nibble; bit 3 = sign).
FP4_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def dequant_nvfp4(packed: np.ndarray, scale_e4m3: np.ndarray, n_elems: int) -> np.ndarray:
    """Returns f32 array of length n_elems."""
    # Unpack nibbles.
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nibbles = np.empty(packed.size * 2, dtype=np.uint8)
    nibbles[0::2] = lo
    nibbles[1::2] = hi
    nibbles = nibbles[:n_elems]
    sign = (nibbles >> 3) & 1
    mag = FP4_MAG[nibbles & 0x07]
    vals = np.where(sign == 1, -mag, mag)
    # Decode E4M3 scale: S(1)|E(4)|M(3), bias=7.
    # Cast to int16 BEFORE arithmetic to avoid uint8 underflow when
    # exp_s - 7 goes negative.
    s_bits = scale_e4m3.astype(np.uint8)
    sign_s = ((s_bits >> 7) & 1).astype(np.int16)
    exp_s = ((s_bits >> 3) & 0x0F).astype(np.int16)
    man_s = (s_bits & 0x07).astype(np.int16)
    # Normal: (1 + man/8) * 2^(exp-7); subnormal exp=0: (man/8) * 2^(-6).
    pow_normal = np.power(2.0, (exp_s - 7).astype(np.float32))
    f_scale = np.where(
        exp_s == 0,
        (man_s.astype(np.float32) / 8.0) * (2.0 ** -6),
        (1.0 + man_s.astype(np.float32) / 8.0) * pow_normal,
    )
    f_scale = np.where(sign_s == 1, -f_scale, f_scale)
    # Each 16-elem block uses one scale.
    f_scale_per_elem = np.repeat(f_scale, 16)[:n_elems]
    return vals * f_scale_per_elem


def analyze_side(L: int, side: str, n_used: int) -> dict:
    """side ∈ {'k','v'}. Computes per-layer error over `n_used` first elements."""
    shadow = np.fromfile(DUMP / f"layer_{L}_{side}_shadow.bin", dtype=np.float16).astype(np.float32)
    packed = np.fromfile(DUMP / f"layer_{L}_{side}.bin", dtype=np.uint8)
    scale = np.fromfile(DUMP / f"layer_{L}_{side}_scale.bin", dtype=np.uint8)
    deq = dequant_nvfp4(packed, scale, shadow.size)
    s = shadow[:n_used]
    d = deq[:n_used]
    err = d - s
    rmse = float(np.sqrt(np.mean(err ** 2)))
    s_rms = float(np.sqrt(np.mean(s ** 2)))
    rel = rmse / max(s_rms, 1e-9)
    max_abs = float(np.max(np.abs(err)))
    s_max = float(np.max(np.abs(s)))
    # Cosine sim per-token-head: reshape to (-1, head_dim), avg cos.
    return {
        "rmse": rmse,
        "rel_err": rel,
        "max_abs": max_abs,
        "shadow_rms": s_rms,
        "shadow_max": s_max,
        "n_used": n_used,
    }


def main():
    ctx = META["context_len"]
    block_size = META["block_size"]
    # Number of FULL blocks worth of tokens populated. Trailing partial
    # block has zero in the unused tail of the shadow but the same
    # content for the populated head — comparing aggregates is fine.
    rows = []
    for L_meta in META["layers"]:
        L = L_meta["layer"]
        nkvh = L_meta["num_kv_heads"]
        hd = L_meta["head_dim"]
        # f16 shadow: full cache = num_blocks * block_size * nkvh * hd elements.
        # Used elements = ctx * nkvh * hd (block-paged but block_table is sequential
        # per meta — confirmed above). Compare just the populated prefix.
        n_used = ctx * nkvh * hd
        try:
            kk = analyze_side(L, "k", n_used)
            vv = analyze_side(L, "v", n_used)
        except FileNotFoundError as e:
            print(f"[skip] layer {L}: {e}", file=sys.stderr)
            continue
        rows.append((L, L_meta["layer_type"], hd, kk, vv))
    print(f"# context_len={ctx}, n_layers_dumped={len(rows)}")
    print(f"{'L':>3} {'type':>17} {'hd':>4}  "
          f"{'K rmse':>10} {'K rel':>8}  {'V rmse':>10} {'V rel':>8}  "
          f"{'K maxabs':>10} {'V maxabs':>10}")
    # Sort by combined K+V rel_err descending.
    rows_sorted = sorted(rows, key=lambda r: -(r[3]["rel_err"] + r[4]["rel_err"]))
    for L, lt, hd, kk, vv in rows_sorted:
        print(f"{L:>3} {lt:>17} {hd:>4}  "
              f"{kk['rmse']:>10.5f} {kk['rel_err']:>8.4f}  "
              f"{vv['rmse']:>10.5f} {vv['rel_err']:>8.4f}  "
              f"{kk['max_abs']:>10.4f} {vv['max_abs']:>10.4f}")


if __name__ == "__main__":
    main()
