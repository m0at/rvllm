"""Cosine + rms-ratio compare between rvllm-serve dumps and the HF
Pixtral reference dumps from `pixtral_hf_reference.py`.

Phase 3-test (b) of the Round-12 Pixtral vision integration.
Walks both dump directories, finds matching `*.bin` files, loads
them as little-endian BF16, normalises shapes (vendors flatten /
unsqueeze inconsistently), and reports per-stage:

  cos       row-flat dot product cosine vs HF reference
  rms_ratio rvllm.rms / hf.rms
  rms_diff  rms(rvllm - hf) / max(hf.rms, 1e-12)
  shape_rvllm shape_hf  (raw element counts; mismatched shapes
                         normalise via flatten before cosine)

Usage:
  python v3/tools/pixtral_hf_compare.py \\
    --rvllm /tmp/rvllm_dump \\
    --hf    /tmp/pixtral_hf_ref

Threshold for "kernel is correct vs HF": cos ≥ 0.999, rms_diff <
1e-2. ViT activations sit in BF16 throughout the 48-block stack so
some accumulator drift is expected.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np


def load_bf16(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    n = min(af.size, bf.size)
    af, bf = af[:n], bf[:n]
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()))


def normalise_for_compare(arr: np.ndarray, fname: str, hidden: int = 1664,
                          text_hidden: int = 12288) -> np.ndarray:
    """Reshape HF / rvllm dumps to a common [N, hidden] layout.

    HF's PixtralVisionModel.patch_conv hook fires PRE-flatten so its
    `post_patch_conv` dump is NCHW [1, hidden, h, w]. rvllm's dump
    is [N, hidden] (already flattened). Permute HF dumps to match
    rvllm's layout when n_elems and shape allow.
    """
    n = arr.size
    h = hidden
    th = text_hidden
    if fname == "post_patch_conv.bin" and n % h == 0:
        # HF [hidden, h*w] → rvllm [h*w, hidden]
        # Try interpreting as [hidden, hw] first.
        hw = n // h
        # If size allows, transpose. We don't know which one is which,
        # but the rvllm dump never has the channel-major layout, so the
        # canonical is [hw, hidden]. If `arr` is [hidden, hw] in
        # row-major, reshape+transpose; if already [hw, hidden] this
        # would silently corrupt — guard via a heuristic: rvllm side
        # always lays out [N, hidden] so reshape there is safe.
        return arr  # callers pass HF/rvllm explicitly, see main.
    return arr


def reshape_stage(arr: np.ndarray, fname: str, source: str,
                  hidden: int = 1664, text_hidden: int = 12288) -> np.ndarray:
    """Reshape `arr` to a canonical [N, hidden] layout depending on
    which stage / source it came from.

    `source` is "rvllm" or "hf".

    rvllm dumps are always row-major [N, hidden] (or [N, text_hidden]
    after the projector). HF dumps for `post_patch_conv` are NCHW
    [1, hidden, h, w] which we permute to [h*w, hidden]. All other
    HF dumps are already [N, hidden] (post the .flatten(1).T inside
    PixtralVisionModel.forward) so a flat reshape suffices.
    """
    n = arr.size
    if source == "hf" and fname == "post_patch_conv.bin":
        # NCHW [1, hidden, h, w] → permute to [h*w, hidden].
        if n % hidden != 0:
            return arr
        hw = n // hidden
        # The bytes are in [hidden, h*w] order (row-major within NCHW
        # [1, hidden, h, w] is `c outer, hw inner`). Transpose to
        # [h*w, hidden].
        return arr.reshape(hidden, hw).T.copy()
    if fname.startswith("output") or fname.startswith("post_linear_2"):
        h = text_hidden
    elif fname.startswith("post_linear_1"):
        h = text_hidden
    else:
        h = hidden
    if n % h == 0:
        return arr.reshape(n // h, h)
    return arr.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvllm", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    args = ap.parse_args()

    rvllm_dir = Path(args.rvllm)
    hf_dir = Path(args.hf)

    rvllm_files = {p.name for p in rvllm_dir.glob("*.bin")}
    hf_files = {p.name for p in hf_dir.glob("*.bin")}
    shared = sorted(rvllm_files & hf_files)
    if not shared:
        print("no overlapping *.bin files between dirs", flush=True)
        print(f"  rvllm: {sorted(rvllm_files)}")
        print(f"  hf:    {sorted(hf_files)}")
        return 2

    print(f"{'stage':<24} {'cos':>10}  {'rms_ratio':>10}  {'rms_diff':>10}  shapes")
    fail = 0
    for fname in shared:
        a_raw = load_bf16(rvllm_dir / fname)
        b_raw = load_bf16(hf_dir / fname)
        a = reshape_stage(a_raw, fname, "rvllm")
        b = reshape_stage(b_raw, fname, "hf")
        ra = rms(a)
        rb = rms(b)
        ratio = ra / max(rb, 1e-30)
        n = min(a.size, b.size)
        d = a.flatten()[:n].astype(np.float64) - b.flatten()[:n].astype(np.float64)
        rd = float(np.sqrt((d * d).mean())) / max(rb, 1e-30)
        c = cos(a, b)
        ok = c >= args.cos_thresh and rd < 1e-2
        flag = " " if ok else "✗"
        print(f"  {flag} {fname:<24} {c:>10.6f}  {ratio:>10.4f}  {rd:>10.4e}  {a.size} vs {b.size}")
        if not ok:
            fail += 1
    print()
    if fail:
        print(f"FAIL: {fail} stage(s) below cos≥{args.cos_thresh} or rms_diff<1e-2")
        return 2
    print("OK: all stages pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
