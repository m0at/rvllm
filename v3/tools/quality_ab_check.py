#!/usr/bin/env python3
# Direct-mode quality A/B harness for KV-dtype comparison at long
# context. Sends a SINGLE long prompt to /v1/chat/completions with
# temperature=0 (greedy), logs reply + optional logprobs + timing.
#
# Designed to be run 3× with the same prompt under 3 different rvllm
# KV configurations (NVFP4 default, FP8 via RVLLM_FP8_KV=1, F16 via
# RVLLM_F16_KV=1), then the JSON outputs compared externally.
#
# Purpose: distinguish KV-precision garbling from softmax/engine
# numerical issues, and answer the "does F16 also garble at 15k?"
# question the split-KV review flagged as uninstrumented.
#
# Usage:
#   # Start rvllm-serve under the mode you want to test, then:
#   python3 quality_ab_check.py --mode-label nvfp4-default \
#       --target-tokens 15000 --max-tokens 128 --out /tmp/q_nvfp4.json
#
# The script doesn't manage the service — restart rvllm-serve with the
# desired env vars between runs. A typical sequence:
#   1. (default config)                    → --mode-label nvfp4
#   2. systemd drop-in: RVLLM_FP8_KV=1     → --mode-label fp8
#   3. systemd drop-in: RVLLM_F16_KV=1     → --mode-label f16
#
# If the service doesn't support a mode (e.g. F16 KV decode currently
# returns FeatureNotAvailable on sm_121), the harness records that
# cleanly in the JSON rather than crashing.

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import sys
import time
import urllib.request
import urllib.error


PARAGRAPH = (
    "Der Satz des Pythagoras besagt, dass in einem rechtwinkligen Dreieck "
    "das Quadrat der Hypotenuse gleich der Summe der Quadrate der beiden "
    "Katheten ist. "
)
DEFAULT_QUESTION = (
    "Antworte in einem einzigen Satz: wie heißt die Hauptstadt von "
    "Frankreich?"
)


def make_prompt(target_tokens: int, question: str) -> str:
    """Pad with the fixed paragraph until ~`target_tokens`, append the
    question. Rough rule: ~4.5 chars/token for mixed German prose."""
    target_chars = max(int(target_tokens * 4.5) - len(question), 0)
    padding = ""
    while len(padding) < target_chars:
        padding += PARAGRAPH
    return padding + question


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def send_request(port: int, prompt: str, max_tokens: int,
                 timeout_s: int, want_logprobs: bool) -> dict:
    payload = {
        "model": "gemma-4-31b-it",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp is not None else str(e)
        status = e.code
    elapsed = time.perf_counter() - t0

    result: dict = {"elapsed_s": elapsed, "http_status": status}
    try:
        j = json.loads(body)
    except json.JSONDecodeError:
        result["parse_error"] = body[:500]
        return result

    try:
        choice = j["choices"][0]
        result["reply"] = choice["message"]["content"] or ""
        result["finish_reason"] = choice.get("finish_reason")
        usage = j.get("usage", {})
        result["prompt_tokens"] = usage.get("prompt_tokens", 0)
        result["completion_tokens"] = usage.get("completion_tokens", 0)
        if want_logprobs and "logprobs" in choice and choice["logprobs"]:
            # OpenAI-compat `logprobs.content` is a list of per-token dicts;
            # keep the token IDs / strings / logprobs for diffing.
            lp = choice["logprobs"].get("content", []) or []
            result["tokens"] = [
                {"token": t.get("token"), "logprob": t.get("logprob")}
                for t in lp
            ]
    except (KeyError, IndexError, TypeError) as e:
        result["parse_error"] = f"{e.__class__.__name__}: {e}"
        result["raw_head"] = body[:500]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Direct-mode KV-dtype quality A/B harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode-label", required=True,
                    help="tag for this run (e.g. 'nvfp4', 'fp8', 'f16')")
    ap.add_argument("--target-tokens", type=int, default=15000,
                    help="approximate prompt length in tokens (default 15000)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout-secs", type=int, default=900)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--question", default=DEFAULT_QUESTION,
                    help="final question appended to the padding")
    ap.add_argument("--logprobs", action="store_true",
                    help="request per-token logprobs (engine-dependent)")
    ap.add_argument("--out", default=None,
                    help="JSON output path (default: stdout)")
    args = ap.parse_args()

    if not port_open("127.0.0.1", args.port):
        sys.exit(f"port {args.port} not open — start rvllm-serve first.")

    prompt = make_prompt(args.target_tokens, args.question)
    record = {
        "mode_label": args.mode_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_tokens": args.target_tokens,
        "prompt_chars": len(prompt),
        "prompt_head": prompt[:80],
        "prompt_tail": prompt[-80:],
        "max_tokens": args.max_tokens,
    }
    print(f"[{args.mode_label}] prompt_chars={len(prompt)}  "
          f"target_tokens≈{args.target_tokens}")

    r = send_request(args.port, prompt, args.max_tokens, args.timeout_secs,
                     args.logprobs)
    record.update(r)

    print(f"  http_status   = {r.get('http_status')}")
    print(f"  prompt_tokens = {r.get('prompt_tokens', 0)}")
    print(f"  completion    = {r.get('completion_tokens', 0)}")
    print(f"  elapsed       = {r.get('elapsed_s', 0):.1f}s")
    print(f"  finish_reason = {r.get('finish_reason')}")
    reply = r.get("reply", "") or r.get("parse_error", "")
    print(f"  reply[:200]   = {reply[:200]!r}")
    if "tokens" in r:
        n_tok = len(r["tokens"])
        print(f"  logprob-tokens= {n_tok}")

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(f"  → wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
