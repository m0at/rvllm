#!/usr/bin/env python3
"""Round-26 audit: compare per-(layer, phase, tok) qwen36 prefill dumps.

Usage:
  python3 cmp_qwen36_prefill_layers.py <ref_dir> <test_dir>

Both dirs contain:
  embed_tok{TT}.f16
  layer_{LL}_attn_tok{TT}.f16
  layer_{LL}_moe_tok{TT}.f16

for L in 0..40, TT = num prompt tokens.

Reports per-row cosine + max-abs diff. Highlights the first
(L, phase, t) where cos < 0.999999.
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


def discover_tokens(ref_dir):
    """Find max tok suffix in ref_dir (e.g. 21 → 22 prompt tokens)."""
    max_tok = -1
    for f in os.listdir(ref_dir):
        if f.startswith("embed_tok") and f.endswith(".f16"):
            n = int(f[len("embed_tok"):-len(".f16")])
            if n > max_tok:
                max_tok = n
    return max_tok + 1


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ref_dir, test_dir = sys.argv[1], sys.argv[2]
    n_tokens = discover_tokens(ref_dir)
    if n_tokens == 0:
        print(f"no embed_tok*.f16 files in {ref_dir}")
        sys.exit(1)

    files = []
    for t in range(n_tokens):
        files.append(f"embed_tok{t:02}.f16")
    for layer in range(40):
        for t in range(n_tokens):
            files.append(f"layer_{layer:02}_attn_tok{t:02}.f16")
        for t in range(n_tokens):
            files.append(f"layer_{layer:02}_moe_tok{t:02}.f16")

    print(f"# n_tokens = {n_tokens}")
    print(f"{'file':36s}  {'cos':>10s}  {'max_abs':>10s}")
    print("-" * 64)

    first_div = None
    by_layer = {}
    for fname in files:
        rp = os.path.join(ref_dir, fname)
        tp = os.path.join(test_dir, fname)
        if not (os.path.exists(rp) and os.path.exists(tp)):
            continue
        a = load_f16(rp)
        b = load_f16(tp)
        if len(a) != len(b):
            print(f"{fname:36s}  LEN MISMATCH ({len(a)} vs {len(b)})")
            continue
        c = cosine(a, b)
        m = max_abs(a, b)
        if c < 0.999999:
            if first_div is None:
                first_div = fname
                marker = "  <-- FIRST DIVERGENCE"
            else:
                marker = ""
            print(f"{fname:36s}  {c:10.6f}  {m:10.4e}{marker}")
        # Aggregate per layer-phase
        if fname.startswith("layer_"):
            key = fname[:len("layer_NN_attn")] if "_attn_" in fname else fname[:len("layer_NN_moe")]
            by_layer.setdefault(key, []).append((c, m))
    print()
    print("# layer-phase summary (worst-row cos, mean cos, worst max_abs):")
    print("-" * 64)
    for key, rows in sorted(by_layer.items()):
        worst = min(c for c, _ in rows)
        mean = sum(c for c, _ in rows) / len(rows)
        wm = max(m for _, m in rows)
        flag = " *" if worst < 0.999999 else ""
        print(f"{key:36s}  worst={worst:.6f}  mean={mean:.6f}  max_abs={wm:.4e}{flag}")

    print()
    if first_div is None:
        print("OK: all (layer, phase, tok) rows byte-equivalent (cos >= 0.999999)")
    else:
        print(f"DIVERGENCE: first file with cos < 0.999999 = {first_div}")


if __name__ == "__main__":
    main()
