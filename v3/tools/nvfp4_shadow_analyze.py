#!/usr/bin/env python3
"""
NVFP4 shadow-KV diagnostic analyzer.

Reads the dump written by the `RVLLM_NVFP4_SHADOW_F16` runtime path
(see gemma4_bring_up.rs "NVFP4 SHADOW DIAGNOSTIC" block) and computes
per-(layer, kv_head) metrics comparing the ground-truth f16 shadow
against the dequantized NVFP4 storage. Goal: locate the layer where
NVFP4 degrades enough to corrupt attention on mixed German/code/JSON
prompts.

Usage:
    python3 v3/tools/nvfp4_shadow_analyze.py [dump_dir]
    python3 v3/tools/nvfp4_shadow_analyze.py [dump_dir] --sentinel

Dump layout expected in `dump_dir` (default `/tmp/nvfp4_shadow`):
    meta.json
    layer_{L}_k.bin            -- NVFP4 packed u8, 2 values/byte
    layer_{L}_v.bin            -- NVFP4 packed u8
    layer_{L}_k_scale.bin      -- E4M3 u8, 1 scale per 16 elems
    layer_{L}_v_scale.bin      -- E4M3 u8
    layer_{L}_k_shadow.bin     -- f16, ground truth
    layer_{L}_v_shadow.bin     -- f16, ground truth
    layer_{L}_q.bin            -- f16, post-RoPE Q for first decoded token
                                  (padded slot: num_heads * max_head_dim * 2
                                   bytes; tail zero when head_dim<max_head_dim)
    q_last_layer.bin           -- fp8 e4m3, Q of last exec'd layer (legacy)

No external deps beyond numpy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------
# NVFP4 / E4M3 helpers (pure numpy, no CUDA)
# -----------------------------------------------------------------------

# E2M1 LUT — used by unpack AND by the sentinel re-quantizer.
E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
E2M1_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def unpack_e2m1_u8(packed: np.ndarray) -> np.ndarray:
    """
    Unpack a [..., N/2] u8 array of NVFP4 (E2M1) values into [..., N] f32.

    Layout (matches CUDA-side pack): low nibble = even element, high = odd.
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float32)
    out[..., 0::2] = E2M1_LUT[lo]
    out[..., 1::2] = E2M1_LUT[hi]
    return out


def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    """Decode E4M3 (bias=7) u8 -> f32. No NaN/Inf handling for scales."""
    b = b.astype(np.uint32)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    sub = (exp == 0) & (mant != 0)
    norm = exp > 0
    out = np.zeros(b.shape, dtype=np.float32)
    normal_vals = (1.0 + mant[norm].astype(np.float32) / 8.0) * (
        2.0 ** (exp[norm].astype(np.int32) - 7)
    )
    out[norm] = normal_vals
    sub_vals = (mant[sub].astype(np.float32) / 8.0) * (2.0 ** -6)
    out[sub] = sub_vals
    out = np.where(sign == 1, -out, out)
    return out


def dequant_nvfp4(packed: np.ndarray, scales_e4m3: np.ndarray, num_elems: int) -> np.ndarray:
    """Dequantize NVFP4 back to f32."""
    vals = unpack_e2m1_u8(packed)
    scales = e4m3_to_f32(scales_e4m3)
    scales_per_elem = np.repeat(scales, 16)[:num_elems]
    return (vals[:num_elems] * scales_per_elem).astype(np.float32)


def quant_e2m1_nibble(x: float, scale: float) -> int:
    """Nearest-magnitude E2M1 nibble for `x / scale`. Returns 0..15."""
    if scale == 0.0:
        return 0
    v = x / scale
    sign = 1 if v < 0 else 0
    mag = abs(v)
    idx = int(np.argmin(np.abs(E2M1_MAGS - mag)))
    return (sign << 3) | idx


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a)) + 1e-30
    return num / den


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    k = min(k, a.size)
    ia = set(np.argpartition(-a, k - 1)[:k].tolist())
    ib = set(np.argpartition(-b, k - 1)[:k].tolist())
    return len(ia & ib) / float(k)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-30)


def load_layer_q(dump_dir: Path, meta: dict, layer_info: dict) -> np.ndarray | None:
    """Load post-RoPE f16 Q for `layer_info["layer"]`. Returns [num_heads, head_dim] f32,
    or None when the per-layer Q file is missing (backward compat)."""
    l = layer_info["layer"]
    p = dump_dir / f"layer_{l}_q.bin"
    if not p.exists():
        return None
    num_heads = meta["num_heads"]
    max_hd = meta.get("max_head_dim", layer_info["head_dim"])
    hd = layer_info["head_dim"]
    raw = np.fromfile(p, dtype=np.float16).astype(np.float32)
    # Slot is num_heads * max_head_dim; head stride is max_head_dim.
    raw = raw.reshape(num_heads, max_hd)
    return raw[:, :hd].copy()


# -----------------------------------------------------------------------
# Per-layer analysis
# -----------------------------------------------------------------------

def analyze_layer(dump_dir: Path, meta: dict, layer_info: dict) -> dict:
    l = layer_info["layer"]
    hd = layer_info["head_dim"]
    nkvh = layer_info["num_kv_heads"]
    num_blocks = layer_info["num_blocks"]
    block_size = meta["block_size"]
    ctx = meta["context_len"]
    num_heads = meta["num_heads"]

    k_packed = np.fromfile(dump_dir / f"layer_{l}_k.bin", dtype=np.uint8)
    v_packed = np.fromfile(dump_dir / f"layer_{l}_v.bin", dtype=np.uint8)
    k_scale_e4m3 = np.fromfile(dump_dir / f"layer_{l}_k_scale.bin", dtype=np.uint8)
    v_scale_e4m3 = np.fromfile(dump_dir / f"layer_{l}_v_scale.bin", dtype=np.uint8)
    k_shadow = np.fromfile(dump_dir / f"layer_{l}_k_shadow.bin", dtype=np.float16).astype(np.float32)
    v_shadow = np.fromfile(dump_dir / f"layer_{l}_v_shadow.bin", dtype=np.float16).astype(np.float32)

    total_elems = num_blocks * block_size * nkvh * hd
    k_deq = dequant_nvfp4(k_packed, k_scale_e4m3, total_elems)
    v_deq = dequant_nvfp4(v_packed, v_scale_e4m3, total_elems)

    shape = (num_blocks, block_size, nkvh, hd)
    k_shadow_r = k_shadow.reshape(shape)
    v_shadow_r = v_shadow.reshape(shape)
    k_deq_r = k_deq.reshape(shape)
    v_deq_r = v_deq.reshape(shape)

    # Q for this layer (optional; enables logit/out metrics).
    q = load_layer_q(dump_dir, meta, layer_info)
    has_q = q is not None
    group = num_heads // nkvh if nkvh > 0 else 1
    inv_sqrt_hd = 1.0 / math.sqrt(hd)

    per_head = []
    for h in range(nkvh):
        # Slot layout: [num_blocks, block_size, nkvh, hd]; flatten (block,
        # slot) and truncate to ctx valid positions.
        k_gt = k_shadow_r[:, :, h, :].reshape(-1, hd)[:ctx]
        k_nv = k_deq_r[:, :, h, :].reshape(-1, hd)[:ctx]
        v_gt = v_shadow_r[:, :, h, :].reshape(-1, hd)[:ctx]
        v_nv = v_deq_r[:, :, h, :].reshape(-1, hd)[:ctx]
        entry = {
            "head": h,
            "rel_err_K": rel_err(k_gt, k_nv),
            "rel_err_V": rel_err(v_gt, v_nv),
        }
        if has_q and ctx > 0:
            # GQA: heads [h*group..(h+1)*group) share this kv_head.
            q_group = q[h * group:(h + 1) * group]  # [group, hd]
            # Logits [group, ctx]
            logits_gt = (q_group @ k_gt.T) * inv_sqrt_hd
            logits_nv = (q_group @ k_nv.T) * inv_sqrt_hd
            # Aggregate across the group (average the rel_err / overlap).
            logit_errs = [rel_err(logits_gt[g], logits_nv[g]) for g in range(group)]
            top8 = [topk_overlap(logits_gt[g], logits_nv[g], 8) for g in range(group)]
            top16 = [topk_overlap(logits_gt[g], logits_nv[g], 16) for g in range(group)]
            # Output err: softmax(logits) @ V, per group member.
            attn_gt = softmax(logits_gt) @ v_gt  # [group, hd]
            attn_nv = softmax(logits_nv) @ v_nv
            out_errs = [rel_err(attn_gt[g], attn_nv[g]) for g in range(group)]
            entry["logit_err"] = float(np.mean(logit_errs))
            entry["topk8"] = float(np.mean(top8))
            entry["topk16"] = float(np.mean(top16))
            entry["out_err"] = float(np.mean(out_errs))
        per_head.append(entry)

    overall = {
        "layer": l,
        "head_dim": hd,
        "num_kv_heads": nkvh,
        "ctx": ctx,
        "has_q": has_q,
        "mean_rel_err_K": float(np.mean([h["rel_err_K"] for h in per_head])),
        "max_rel_err_K": float(np.max([h["rel_err_K"] for h in per_head])),
        "mean_rel_err_V": float(np.mean([h["rel_err_V"] for h in per_head])),
        "max_rel_err_V": float(np.max([h["rel_err_V"] for h in per_head])),
        "per_head": per_head,
    }
    if has_q:
        overall["mean_logit_err"] = float(np.mean([h["logit_err"] for h in per_head]))
        overall["max_logit_err"] = float(np.max([h["logit_err"] for h in per_head]))
        overall["mean_topk8"] = float(np.mean([h["topk8"] for h in per_head]))
        overall["min_topk8"] = float(np.min([h["topk8"] for h in per_head]))
        overall["mean_topk16"] = float(np.mean([h["topk16"] for h in per_head]))
        overall["min_topk16"] = float(np.min([h["topk16"] for h in per_head]))
        overall["mean_out_err"] = float(np.mean([h["out_err"] for h in per_head]))
        overall["max_out_err"] = float(np.max([h["out_err"] for h in per_head]))
    else:
        overall["note"] = "logit/topk/out_err skipped — layer_{L}_q.bin not found"
    return overall


# -----------------------------------------------------------------------
# Sentinel: verify E2M1 nibble-pack + E4M3 scale layout against CUDA dump
# -----------------------------------------------------------------------

def sentinel(dump_dir: Path, meta: dict) -> int:
    """Pick a layer, 3 arbitrary slots, re-quantize f16 shadow value and
    compare to the packed byte. Returns process exit code."""
    layers = meta["layers"]
    if not layers:
        print("SENTINEL FAIL — no layers in meta.json", file=sys.stderr)
        return 2
    li = layers[0]
    l = li["layer"]
    hd = li["head_dim"]
    nkvh = li["num_kv_heads"]
    num_blocks = li["num_blocks"]
    block_size = meta["block_size"]
    ctx = meta["context_len"]

    k_packed = np.fromfile(dump_dir / f"layer_{l}_k.bin", dtype=np.uint8)
    k_scale_e4m3 = np.fromfile(dump_dir / f"layer_{l}_k_scale.bin", dtype=np.uint8)
    k_shadow = np.fromfile(dump_dir / f"layer_{l}_k_shadow.bin", dtype=np.float16).astype(np.float32)

    shape = (num_blocks, block_size, nkvh, hd)
    k_shadow_r = k_shadow.reshape(shape)
    # Flat element layout matches shape row-major; scales group every 16
    # consecutive elements along that flat order.

    # Pick 3 slots inside the populated region.
    if ctx <= 0:
        print("SENTINEL FAIL — ctx=0, no populated slots", file=sys.stderr)
        return 2
    slots = [
        (0, 0, 0),
        (0, min(1, block_size - 1), min(1, nkvh - 1)),
        ((ctx - 1) // block_size, (ctx - 1) % block_size, min(nkvh - 1, 0)),
    ]
    total_elems = num_blocks * block_size * nkvh * hd
    k_deq_flat = dequant_nvfp4(k_packed, k_scale_e4m3, total_elems)
    scales = e4m3_to_f32(k_scale_e4m3)
    k_unpacked = unpack_e2m1_u8(k_packed)[:total_elems]

    mismatches = []
    for (b, t, h) in slots:
        if b >= num_blocks or t >= block_size or h >= nkvh:
            continue
        # Per-channel check across head_dim for this slot.
        flat_base = ((b * block_size + t) * nkvh + h) * hd
        for c in range(hd):
            flat_idx = flat_base + c
            byte_idx = flat_idx // 2
            nibble_hi = (flat_idx & 1) == 1
            raw = k_packed[byte_idx]
            actual_nib = (int(raw) >> 4) & 0x0F if nibble_hi else int(raw) & 0x0F
            scale_idx = flat_idx // 16
            scale = float(scales[scale_idx])
            gt = float(k_shadow_r[b, t, h, c])
            expected_nib = quant_e2m1_nibble(gt, scale)
            if expected_nib != actual_nib:
                mismatches.append({
                    "slot": (b, t, h, c),
                    "flat_idx": flat_idx,
                    "expected_nib": expected_nib,
                    "actual_nib": actual_nib,
                    "shadow_val": gt,
                    "scale": scale,
                    "nvfp4_val": float(k_deq_flat[flat_idx]),
                    "unpacked_val": float(k_unpacked[flat_idx]),
                })
                if len(mismatches) >= 1:
                    break
        if mismatches:
            break

    if not mismatches:
        print(f"SENTINEL OK — layout matches (layer {l}, checked 3 slots × {hd} channels)")
        return 0

    m = mismatches[0]
    # Classify the mismatch.
    exp_nib = m["expected_nib"]
    act_nib = m["actual_nib"]
    diag = "scale-value or quantization mismatch"
    # Nibble-order flipped: check the OTHER nibble in the same byte.
    flat_idx = m["flat_idx"]
    byte_idx = flat_idx // 2
    raw = int(k_packed[byte_idx])
    lo_nib = raw & 0x0F
    hi_nib = (raw >> 4) & 0x0F
    other_nib = hi_nib if (flat_idx & 1) == 0 else lo_nib
    if other_nib == exp_nib:
        diag = "nibble-order flipped (lo<->hi)"
    elif abs(int(exp_nib) - int(act_nib)) == 1:
        diag = "off-by-one-nibble (rounding direction or adjacent LUT entry)"
    # Scale-index mismatch: if neighbouring scale would match.
    if diag == "scale-value or quantization mismatch":
        scale_idx = flat_idx // 16
        for delta in (-1, 1):
            ni = scale_idx + delta
            if 0 <= ni < scales.size:
                neigh_nib = quant_e2m1_nibble(m["shadow_val"], float(scales[ni]))
                if neigh_nib == act_nib:
                    diag = f"scale-index mismatch (off-by-{delta})"
                    break

    print(f"SENTINEL FAIL — {diag}", file=sys.stderr)
    print(f"  layer={l} slot={m['slot']} (block, token, kv_head, channel)", file=sys.stderr)
    print(f"  flat_idx={m['flat_idx']}  byte=0x{raw:02x}  nibble_hi={(flat_idx & 1) == 1}", file=sys.stderr)
    print(f"  expected nibble=0x{exp_nib:X}  actual nibble=0x{act_nib:X}", file=sys.stderr)
    print(f"  shadow_val={m['shadow_val']:+.6f}  scale={m['scale']:+.6e}", file=sys.stderr)
    print(f"  dequant_val={m['nvfp4_val']:+.6f}  unpacked_raw={m['unpacked_val']:+.6f}", file=sys.stderr)
    return 2


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default="/tmp/nvfp4_shadow")
    ap.add_argument("--sentinel", action="store_true",
                    help="verify E2M1 / E4M3 layout by re-quantizing shadow values")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        print(f"dump dir not found: {dump_dir}", file=sys.stderr)
        sys.exit(1)
    meta = json.loads((dump_dir / "meta.json").read_text())

    if args.sentinel:
        sys.exit(sentinel(dump_dir, meta))

    print(f"[shadow-analyze] dump_dir = {dump_dir}")
    print(f"[shadow-analyze] prompt_len = {meta['prompt_len']}, ctx = {meta['context_len']}")
    print(f"[shadow-analyze] shadow layers: {meta['shadow_layer_indices']}")
    print()

    results = []
    for layer_info in meta["layers"]:
        try:
            r = analyze_layer(dump_dir, meta, layer_info)
            results.append(r)
        except Exception as e:  # pragma: no cover (diagnostic path)
            print(f"layer {layer_info['layer']}: FAILED: {e}")

    # Summary table — KV-only view.
    print(f"{'layer':>5} {'hd':>4} {'nkvh':>4} {'meanK':>10} {'maxK':>10} {'meanV':>10} {'maxV':>10}")
    for r in results:
        print(
            f"{r['layer']:>5d} {r['head_dim']:>4d} {r['num_kv_heads']:>4d} "
            f"{r['mean_rel_err_K']:>10.4f} {r['max_rel_err_K']:>10.4f} "
            f"{r['mean_rel_err_V']:>10.4f} {r['max_rel_err_V']:>10.4f}"
        )

    # Extended table — attention-logit view (only when per-layer Q was dumped).
    if any(r["has_q"] for r in results):
        print()
        print(f"{'layer':>5} {'mean_logit':>11} {'max_logit':>11} {'min_top8':>9} {'min_top16':>10} {'max_out':>10}")
        for r in results:
            if not r["has_q"]:
                continue
            print(
                f"{r['layer']:>5d} {r['mean_logit_err']:>11.4f} {r['max_logit_err']:>11.4f} "
                f"{r['min_topk8']:>9.3f} {r['min_topk16']:>10.3f} "
                f"{r['max_out_err']:>10.4f}"
            )

        # Collapse locator: first layer where attention structure breaks down.
        kv_threshold = 0.3
        print()
        for r in results:
            if not r["has_q"]:
                continue
            if r["min_topk8"] < 0.5 or r["max_logit_err"] > 0.1:
                print(
                    f"[shadow-analyze] COLLAPSE LOCATOR — layer {r['layer']}: "
                    f"min_topk8={r['min_topk8']:.3f}, max_logit_err={r['max_logit_err']:.4f}, "
                    f"max_out_err={r['max_out_err']:.4f}"
                )
                break
        else:
            print("[shadow-analyze] no collapse layer found under thresholds "
                  "(min_topk8<0.5 or max_logit_err>0.1)")

        # KV-only fallback flag (pre-existing behaviour).
        for r in results:
            if r["max_rel_err_K"] > kv_threshold or r["max_rel_err_V"] > kv_threshold:
                print(
                    f"[shadow-analyze] candidate KV-rel-err collapse at layer {r['layer']} "
                    f"(max_rel_err_K={r['max_rel_err_K']:.3f}, max_rel_err_V={r['max_rel_err_V']:.3f})"
                )
                break
    else:
        # No Q dumps — keep the legacy flag behaviour.
        threshold = 0.3
        for r in results:
            if r["max_rel_err_K"] > threshold or r["max_rel_err_V"] > threshold:
                print(
                    f"\n[shadow-analyze] candidate collapse at layer {r['layer']} "
                    f"(max_rel_err_K={r['max_rel_err_K']:.3f}, max_rel_err_V={r['max_rel_err_V']:.3f})"
                )
                break


if __name__ == "__main__":
    main()
