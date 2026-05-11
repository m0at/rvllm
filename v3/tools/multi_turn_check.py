#!/usr/bin/env python3
"""Multi-turn stability harness — 5 turns each, two stages.

Stage 1 (`direct`): hits rvllm-server /v1/chat/completions directly with a
growing chat history (turn N sees turns 1..N-1, so context grows each turn;
turn 5 is roughly in the 12-15k-token regime).

Stage 2 (`zeroclaw`): hits zeroclaw /webhook (localhost:42617). Webhook is
stateless — each turn is independent — so this tests end-to-end including
persona injection / tool plumbing / response formatting, not just raw
chat-completion throughput.

For each turn we record: prompt_tokens, completion_tokens, total elapsed,
tokens/sec (rough), and the first ~80 chars of the reply.

rvllm-server must already be running. This harness does NOT manage the
service — run `sudo systemctl start rvllm-serve` first.

Usage:
  ~/.venv/bin/python3 v3/tools/multi_turn_check.py \
      --stage direct       # default: both
      --turns 5
      --port 8010
      --webhook-port 42617
      --timeout-secs 300
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request


QUESTION_SEED = (
    "Ich habe eine längere technische Frage zu Transformer-Decodern auf "
    "GB10 Hardware. Antworte bitte auf Deutsch in genau einem kurzen Satz. "
    "Frage: was ist der Hauptgrund dafür, dass GQA-Decoding auf sm_121 "
    "sinnvoll ist?"
)

FOLLOWUPS = [
    "Gut. Fasse deine letzte Antwort in genau einem Wort zusammen.",
    "Nenne mir drei potentielle Fallstricke der NVFP4 KV-Quantisierung. "
    "Ein Satz pro Fallstrick, keine Einleitung.",
    "Welcher dieser drei Punkte ist bei einem 15k-Kontext am kritischsten "
    "und warum? Zwei Sätze.",
    "Zum Abschluss: nenne die Hauptstadt von Frankreich in einem Wort.",
]


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def post_json(url: str, payload: dict, timeout_s: int) -> tuple[float, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = r.read().decode()
    elapsed = time.perf_counter() - t0
    return elapsed, json.loads(body)


def run_direct(port: int, turns: int, timeout_s: int, max_tokens: int) -> int:
    print(f"\n== STAGE 1: direct /v1/chat/completions (port {port}) ==")
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    messages: list[dict] = []
    any_fail = False
    for t in range(1, turns + 1):
        if t == 1:
            prompt = QUESTION_SEED
        else:
            idx = min(t - 2, len(FOLLOWUPS) - 1)
            prompt = FOLLOWUPS[idx]
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": "gemma-4-31b-it",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        # approx chars — rough proxy for growing context
        ctx_chars = sum(len(m["content"]) for m in messages)
        print(f"  turn {t}: ctx_chars={ctx_chars}, user={prompt[:60]!r}")
        try:
            elapsed, resp = post_json(url, payload, timeout_s)
        except Exception as e:
            print(f"    -> FAIL: {e.__class__.__name__}: {e}")
            any_fail = True
            break
        try:
            reply = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
        except (KeyError, IndexError) as e:
            print(f"    -> FAIL: malformed response: {e} body={str(resp)[:200]}")
            any_fail = True
            break
        tps = (ct / elapsed) if elapsed > 0 else 0.0
        print(f"    -> PASS prompt={pt} gen={ct} time={elapsed:.1f}s "
              f"~{tps:.1f}tok/s reply={reply[:80]!r}")
        messages.append({"role": "assistant", "content": reply})
    return 0 if not any_fail else 1


def run_zeroclaw(port: int, turns: int, timeout_s: int) -> int:
    print(f"\n== STAGE 2: zeroclaw /webhook (port {port}) ==")
    url = f"http://127.0.0.1:{port}/webhook"
    prompts = [QUESTION_SEED] + FOLLOWUPS
    any_fail = False
    for t in range(1, turns + 1):
        prompt = prompts[min(t - 1, len(prompts) - 1)]
        print(f"  turn {t}: {prompt[:60]!r}")
        try:
            elapsed, resp = post_json(url, {"message": prompt}, timeout_s)
        except Exception as e:
            print(f"    -> FAIL: {e.__class__.__name__}: {e}")
            any_fail = True
            break
        reply = ""
        if isinstance(resp, dict):
            reply = (resp.get("reply") or resp.get("message")
                     or resp.get("text") or "")
            if not reply:
                reply = json.dumps(resp)[:200]
        else:
            reply = str(resp)[:200]
        print(f"    -> PASS time={elapsed:.1f}s reply={reply[:120]!r}")
    return 0 if not any_fail else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["direct", "zeroclaw", "both"],
                    default="both")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--webhook-port", type=int, default=42617)
    ap.add_argument("--timeout-secs", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    rc = 0
    if args.stage in ("direct", "both"):
        if not port_open("127.0.0.1", args.port):
            sys.exit(f"port {args.port} not open — "
                     f"`sudo systemctl start rvllm-serve` first.")
        rc |= run_direct(args.port, args.turns, args.timeout_secs,
                         args.max_tokens)
    if args.stage in ("zeroclaw", "both"):
        if not port_open("127.0.0.1", args.webhook_port):
            sys.exit(f"port {args.webhook_port} not open — zeroclaw down.")
        rc |= run_zeroclaw(args.webhook_port, args.turns, args.timeout_secs)
    return rc


if __name__ == "__main__":
    sys.exit(main())
