#!/usr/bin/env python3
"""
Logit-margin analyzer for rvllm-serve diagnostic dumps.

Pairs with the `RVLLM_DUMP_LOGITS=1` gate in `gemma4_bring_up.rs`.
That gate writes per-decode-step top-10 logit dumps to
`$RVLLM_DUMP_LOGITS_DIR/logits_<ns>.json`. This tool reads the
latest such dump (or a specified one), pulls the model's tokenizer,
and prints per-step margin/rank breakdown so we can localize why
specific decode steps produced wrong tokens (e.g. R1's "Termिने" or
R2's "Ich verwal {").

Decision aid (per GPT-5.5 + our discriminator scheme):
  - emitted token = top-1 with comfortable margin     → quant noise
                                                        unlikely to
                                                        flip it
  - emitted = top-1 but <0.5 margin to top-2          → near-tied
                                                        logit; rotation
                                                        helps
  - emitted = top-3+ in this dump but the dump path  → directional
    ran a different config than the one with bug        drift; rotation
                                                        may not help
  - sudden cliff in top-1 confidence at a specific    → that's where
    decode step                                          the corruption
                                                          starts

Usage:
  v3/tools/logit_margin_analyze.py [--dir DIR] [--file FILE]
                                    [--top K] [--show-text]
                                    [--model MODEL_DIR]

  --dir       directory containing logits_*.json (default $RVLLM_DUMP_LOGITS_DIR
              or /tmp/rvllm_logits)
  --file      specific logits dump to inspect; overrides --dir scan
  --top       how many of the top-K alternates to show per step (default 5)
  --show-text decode token IDs to text (requires tokenizer; --model dir)
  --model     model dir for tokenizer (default
              /home/r00t/.vllm/models/gemma-4-31b-it-fp8-block)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys


def load_dump(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def latest_dump(dir_: pathlib.Path) -> pathlib.Path:
    files = sorted(dir_.glob("logits_*.json"))
    if not files:
        raise FileNotFoundError(f"no logits_*.json in {dir_}")
    return files[-1]


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def try_load_tokenizer(model_dir: pathlib.Path):
    """Best-effort: try transformers, then huggingface_hub. None if nothing works."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:
        print(f"[warn] transformers load failed: {e}", file=sys.stderr)
    try:
        import sentencepiece as spm
        for f in ("tokenizer.model", "spiece.model"):
            p = model_dir / f
            if p.exists():
                sp = spm.SentencePieceProcessor()
                sp.load(str(p))
                class _Spm:
                    def __init__(self, sp): self.sp = sp
                    def decode(self, ids):
                        if isinstance(ids, int): ids = [ids]
                        return self.sp.decode(list(ids))
                return _Spm(sp)
    except Exception as e:
        print(f"[warn] sentencepiece load failed: {e}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get(
                        "RVLLM_DUMP_LOGITS_DIR", "/tmp/rvllm_logits")))
    ap.add_argument("--file", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--show-text", action="store_true",
                    help="decode token IDs via tokenizer (requires transformers/sentencepiece)")
    ap.add_argument("--model", type=pathlib.Path,
                    default=pathlib.Path(
                        "/home/r00t/.vllm/models/gemma-4-31b-it-fp8-block"))
    ap.add_argument("--steps", type=int, default=64,
                    help="how many steps to print (default 64)")
    args = ap.parse_args()

    if args.file:
        dump_path = args.file
    else:
        dump_path = latest_dump(args.dir)
    print(f"[reading] {dump_path}")

    d = load_dump(dump_path)
    print(f"  prompt_tokens: {d.get('prompt_tokens')}")
    print(f"  output_tokens: {len(d.get('output_ids', []))}")
    print(f"  step records:  {len(d.get('steps', []))}")

    tok = None
    if args.show_text:
        tok = try_load_tokenizer(args.model)
        if tok is None:
            print("[warn] tokenizer unavailable; printing IDs only",
                  file=sys.stderr)

    print()
    print(f"{'step':>4} {'emitted':>8} {'top1':>10} {'top2':>10} "
          f"{'log-margin':>10} {'p(emit)':>8} {'p(top1)':>8}  {'rank-of-emitted':>5} | top-{args.top}")

    n_print = min(args.steps, len(d['steps']))
    suspicious = []
    for entry in d['steps'][:n_print]:
        step = entry['step']
        emitted = entry['emitted']
        top = entry['top10']
        ids = [t[0] for t in top]
        scores = [t[1] for t in top]
        rank = ids.index(emitted) if emitted in ids else None
        rank_str = str(rank + 1) if rank is not None else ">10"
        # Margin in raw f32 logit space: top1 - top2
        margin = scores[0] - (scores[1] if len(scores) > 1 else 0.0)
        # softmax over top-10 (approximation; full softmax would need full vocab)
        sm = softmax(scores)
        p_top1 = sm[0]
        p_emit = sm[rank] if rank is not None else 0.0

        flag = ""
        if rank is not None and rank > 0:
            flag = " *FLIPPED*"
            suspicious.append(entry)
        elif margin < 0.5:
            flag = " *low-margin*"
            suspicious.append(entry)

        if args.show_text and tok is not None:
            try:
                tok_str = tok.decode([emitted]).replace('\n', '\\n')[:24]
            except Exception:
                tok_str = "?"
            print(f"{step:>4} {emitted:>8} {ids[0]:>10} {ids[1]:>10} "
                  f"{margin:>10.4f} {p_emit:>8.4f} {p_top1:>8.4f}  {rank_str:>5} | "
                  f"{tok_str!r}{flag}")
        else:
            print(f"{step:>4} {emitted:>8} {ids[0]:>10} {ids[1]:>10} "
                  f"{margin:>10.4f} {p_emit:>8.4f} {p_top1:>8.4f}  {rank_str:>5}{flag}")

    print()
    print(f"=== summary ===")
    print(f"  {len(suspicious)} step(s) flagged "
          f"(emitted-not-top-1 OR margin < 0.5)")
    if suspicious:
        print(f"  first flagged step: #{suspicious[0]['step']}")
        if args.show_text and tok is not None:
            try:
                ctx = " ".join(tok.decode([t]).strip() or '_'
                               for t in d['output_ids'][:suspicious[0]['step']])
                print(f"  output context up to first flag: {ctx[-100:]}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
