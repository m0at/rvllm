#!/usr/bin/env python3
"""Per-sub-step cosine compare of an rvllm `dump_substep` directory
against an HF reference dump produced by gemma_vision_substep_hf_dump.py.

Both sides are f16 little-endian, [N, D] row-major. Filename pattern:
g4v_blk{B}_{step}.bin. The script auto-discovers matching pairs and
prints per-row cosine mean / min, plus a "first divergence" pointer
once cos drops below a threshold.

Usage:
  python3 v3/tools/cmp_g4v_substep.py \\
    --rvllm /tmp/g4v_audit \\
    --hf    /tmp/hf_g4v_blk0 \\
    --block 0
"""
import argparse, os
from pathlib import Path
import numpy as np


SUBSTEPS = [
    "input_ln", "q_proj", "k_proj", "v_proj",
    "q_norm", "k_norm", "v_norm",
    "q_rot", "k_rot",
    "attn_out", "o_proj",
    "post_attn_ln", "post_attn_resid",
    "pre_ff_ln", "gate_proj", "up_proj", "gelu_mul",
    "down_proj", "post_ff_ln",
]


def load(path: Path, ncols: int) -> np.ndarray:
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return arr.reshape(-1, ncols)


def cos_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa = a.reshape(-1, a.shape[-1])
    fb = b.reshape(-1, b.shape[-1])
    return (fa * fb).sum(-1) / (
        np.linalg.norm(fa, axis=-1) * np.linalg.norm(fb, axis=-1) + 1e-9
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvllm", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    rv = Path(args.rvllm)
    hf = Path(args.hf)
    print(f"{'step':<20} {'shape':<14} {'cos mean':>10} {'cos min':>10}")
    first_div = None
    for step in SUBSTEPS:
        rv_p = rv / f"g4v_blk{args.block}_{step}.bin"
        hf_p = hf / f"g4v_blk{args.block}_{step}.bin"
        if not rv_p.exists() or not hf_p.exists():
            print(f"{step:<20} {'(missing)':<14}")
            continue
        # Determine ncols from HF file size (reference is ground truth).
        # Common widths: 1152 (HIDDEN), 4304 (INTERMEDIATE).
        sz = hf_p.stat().st_size // 2  # f16 elements
        for cand in (1152, 4304):
            if sz % cand == 0:
                ncols = cand
                break
        else:
            print(f"{step:<20} weird size {sz}; skip")
            continue
        a = load(rv_p, ncols)
        b = load(hf_p, ncols)
        if a.shape != b.shape:
            print(f"{step:<20} shape mismatch rv={a.shape} hf={b.shape}")
            continue
        mask = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        if not mask.any():
            print(f"{step:<20} all-inf rows; skip"); continue
        c = cos_per_row(a[mask], b[mask])
        m, mn = c.mean(), c.min()
        marker = " ← first drop" if first_div is None and mn < args.threshold else ""
        if first_div is None and mn < args.threshold:
            first_div = step
        print(f"{step:<20} {str(a.shape):<14} {m:>10.4f} {mn:>10.4f}{marker}")
    if first_div:
        print(f"\nfirst divergence below cos<{args.threshold}: {first_div}")
    else:
        print(f"\nall sub-steps stay cos>={args.threshold}")


if __name__ == "__main__":
    main()
