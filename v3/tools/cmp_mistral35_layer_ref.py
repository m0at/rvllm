#!/usr/bin/env python3
"""Diff a `probe-mistral35-layer-ref` CPU baseline dir against a
device-side dump dir, file-by-file.

The CPU baseline is produced by:

    cargo run --bin probe-mistral35-layer-ref --release -- \\
        --out-dir /tmp/rvllm-mistral35-ref \\
        --prompt-len 4 --seed 42

This script walks every `*.bin` under `--cpu-dir`, finds the
matching file in `--device-dir`, loads both as f32 LE 1-D arrays
(reshaped to a flat row), and reports cosine similarity +
max-abs-error per file. A summary line at the bottom flags any
file whose cosine drops below `--threshold` (default 0.999).

Files dumped by the probe binary follow this naming convention:

  layer0_hidden_in_pos{N}.bin     [hidden_size]
  layer0_hidden_out_pos{N}.bin    [hidden_size]
  layer0_kv_k.bin                 [seq_len, kv_dim]
  layer0_kv_v.bin                 [seq_len, kv_dim]

Once the Mistral 3.5 CUDA forward (Mistral35Bringup::run_generate)
lands and gains a `RVLLM_MISTRAL35_LAYER_DUMP_DIR=...` knob
matching this layout, the comparison closes the validation loop:

  - cosine ≥ 0.9999 per file → kernel matches the reference.
  - first file under threshold → bisect target for the kernel
    bug.

Usage:

  python3 v3/tools/cmp_mistral35_layer_ref.py \\
      --cpu-dir /tmp/rvllm-mistral35-ref \\
      --device-dir /tmp/rvllm-mistral35-device \\
      [--threshold 0.999]

Both sides MUST be f32 little-endian raw byte streams. The CPU
side already is; the future CUDA dump path needs to write its
intermediate hidden states in the same layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"{path}: byte length {len(raw)} not a multiple of 4")
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    n_a = float(np.linalg.norm(a))
    n_b = float(np.linalg.norm(b))
    if n_a == 0.0 or n_b == 0.0:
        # Both zero is a perfect match; one zero one not is a
        # complete miss — return 0 either way is misleading. Use
        # NaN so the summary surfaces it clearly.
        return float("nan") if n_a != n_b else 1.0
    return float(np.dot(a, b) / (n_a * n_b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-dir", required=True, type=Path,
                    help="probe-mistral35-layer-ref --out-dir")
    ap.add_argument("--device-dir", required=True, type=Path,
                    help="future CUDA layer-dump output dir")
    ap.add_argument("--threshold", type=float, default=0.999,
                    help="cosine below this value flags the file as drifting")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-file lines that pass the threshold")
    args = ap.parse_args()

    if not args.cpu_dir.is_dir():
        print(f"error: cpu-dir not a directory: {args.cpu_dir}", file=sys.stderr)
        return 2
    if not args.device_dir.is_dir():
        print(f"error: device-dir not a directory: {args.device_dir}",
              file=sys.stderr)
        return 2

    cpu_files = sorted(p for p in args.cpu_dir.glob("*.bin") if p.is_file())
    if not cpu_files:
        print(f"error: no *.bin files under {args.cpu_dir}", file=sys.stderr)
        return 2

    print(f"compare: cpu={args.cpu_dir} device={args.device_dir}")
    print(f"threshold: cos >= {args.threshold}")
    print(f"{'file':<40} {'shape':>14}  {'cos':>10}  {'max_abs_err':>12}  status")
    print("-" * 92)

    failed: list[str] = []
    missing: list[str] = []
    for cpu_path in cpu_files:
        dev_path = args.device_dir / cpu_path.name
        if not dev_path.exists():
            missing.append(cpu_path.name)
            print(f"{cpu_path.name:<40} {'(missing)':>14}  "
                  f"{'-':>10}  {'-':>12}  MISSING")
            continue
        try:
            a = load_f32(cpu_path)
            b = load_f32(dev_path)
        except Exception as e:
            print(f"{cpu_path.name:<40}  load error: {e}")
            failed.append(cpu_path.name)
            continue
        if a.shape != b.shape:
            print(f"{cpu_path.name:<40}  shape mismatch: cpu={a.shape} device={b.shape}")
            failed.append(cpu_path.name)
            continue
        c = cosine(a, b)
        mae = float(np.max(np.abs(a - b))) if a.size > 0 else 0.0
        status = "ok" if c >= args.threshold else "DRIFT"
        if status == "DRIFT":
            failed.append(cpu_path.name)
        if status != "ok" or not args.quiet:
            print(f"{cpu_path.name:<40} {str(a.shape):>14}  "
                  f"{c:>10.6f}  {mae:>12.4e}  {status}")

    print("-" * 92)
    total = len(cpu_files)
    n_ok = total - len(failed) - len(missing)
    print(f"summary: {n_ok}/{total} ok, {len(failed)} drifting, "
          f"{len(missing)} missing on device side")
    if failed or missing:
        print("first drift / missing (bisect target):",
              (failed + missing)[0] if (failed or missing) else "-")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
