#!/usr/bin/env python3
"""Round-24 audit harness: compare per-token vs batched-prefill
qwen36 layer dumps and report per-layer cosine + first divergence.

Usage:
  python3 cmp_qwen36_prefill_layers.py <ref_dir> <test_dir>

Both dirs must contain matching `embed.f16`, `layer_NN_attn.f16`,
`layer_NN_moe.f16` files written by the per-token (token-major) and
batched (`RVLLM_QWEN36_BATCH_LINEAR_PREFILL=1`) paths respectively.
"""

import sys
import os
import struct
import math


def load_f16(path):
    with open(path, "rb") as f:
        b = f.read()
    n = len(b) // 2
    out = []
    for i in range(n):
        u16 = struct.unpack_from("<H", b, i * 2)[0]
        sign = (u16 >> 15) & 1
        exp = (u16 >> 10) & 0x1F
        mant = u16 & 0x3FF
        if exp == 0:
            val = (mant / 1024.0) * (2 ** -14)
        elif exp == 0x1F:
            val = float("nan") if mant else float("inf")
        else:
            val = (1 + mant / 1024.0) * (2 ** (exp - 15))
        if sign:
            val = -val
        out.append(val)
    return out


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return float("nan")
    return dot / (na * nb)


def max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ref_dir, test_dir = sys.argv[1], sys.argv[2]
    files = ["embed.f16"]
    for layer in range(40):
        files.append(f"layer_{layer:02}_attn.f16")
        files.append(f"layer_{layer:02}_moe.f16")

    print(f"{'file':32s}  {'cos':>10s}  {'max_abs':>10s}  {'len':>6s}")
    print("-" * 64)
    first_div = None
    for fname in files:
        rp = os.path.join(ref_dir, fname)
        tp = os.path.join(test_dir, fname)
        if not (os.path.exists(rp) and os.path.exists(tp)):
            continue
        a = load_f16(rp)
        b = load_f16(tp)
        if len(a) != len(b):
            print(f"{fname:32s}  LEN MISMATCH ({len(a)} vs {len(b)})")
            continue
        c = cosine(a, b)
        m = max_abs(a, b)
        marker = ""
        if c < 0.999999:
            if first_div is None:
                first_div = fname
                marker = "  <-- FIRST DIVERGENCE"
        print(f"{fname:32s}  {c:10.6f}  {m:10.4e}  {len(a):6d}{marker}")
    print()
    if first_div is None:
        print("OK: all layers byte-equivalent (cos >= 0.999999)")
    else:
        print(f"DIVERGENCE: first file with cos < 0.999999 = {first_div}")


if __name__ == "__main__":
    main()
