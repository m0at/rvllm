#!/usr/bin/env python3
import argparse
import json
import math
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def ppl_value(doc):
    ppl = (doc.get("ppl") or {}).get("ppl")
    if not isinstance(ppl, (int, float)) or not math.isfinite(ppl):
        raise ValueError("missing finite ppl.ppl")
    return float(ppl)


def gen_text(doc):
    text = (doc.get("generation") or {}).get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("missing generation.text")
    return text


def common_prefix_len(a, b):
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def control_fraction(text):
    if not text:
        return 1.0
    bad = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
    return bad / len(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--ppl-rel-tol", type=float, default=0.03)
    p.add_argument("--ppl-abs-tol", type=float, default=0.10)
    p.add_argument("--min-prefix-chars", type=int, default=80)
    p.add_argument("--max-control-frac", type=float, default=0.01)
    args = p.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    failures = []
    base_ppl = ppl_value(base)
    cand_ppl = ppl_value(cand)
    ppl_abs = cand_ppl - base_ppl
    ppl_rel = ppl_abs / base_ppl if base_ppl else float("inf")
    if ppl_abs > args.ppl_abs_tol and ppl_rel > args.ppl_rel_tol:
        failures.append(
            f"PPL regressed: baseline={base_ppl:.6g} candidate={cand_ppl:.6g} "
            f"abs={ppl_abs:.6g} rel={ppl_rel:.3%}"
        )

    base_text = gen_text(base)
    cand_text = gen_text(cand)
    prefix = common_prefix_len(base_text, cand_text)
    if prefix < args.min_prefix_chars:
        failures.append(
            f"generation prefix drift: common_prefix={prefix} chars "
            f"required={args.min_prefix_chars}"
        )
    for name, text in (("baseline", base_text), ("candidate", cand_text)):
        frac = control_fraction(text)
        if frac > args.max_control_frac:
            failures.append(
                f"{name} generation has too many control chars: "
                f"{frac:.2%} > {args.max_control_frac:.2%}"
            )

    report = {
        "baseline_ppl": base_ppl,
        "candidate_ppl": cand_ppl,
        "ppl_abs_delta": ppl_abs,
        "ppl_rel_delta": ppl_rel,
        "generation_common_prefix_chars": prefix,
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"correctness gate error: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(2)
