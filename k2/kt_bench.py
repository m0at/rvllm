#!/usr/bin/env python3
"""Benchmark a KTransformers+SGLang Kimi K2.6 endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp

PROMPTS = [
    "Write a short Rust function that parses a CSV row with escaped quotes.",
    "List five practical steps to debug a CUDA OOM during model serving.",
    "Explain when mmap is better than read()+copy for large model weights.",
    "Write three concise bullet points about MoE CPU offload tradeoffs.",
]

DEFAULT_PPL_TEXT = (
    "In the beginning was the Word, and the Word was with God, and the Word was God. "
    "The same was in the beginning with God. All things were made by him; and without "
    "him was not any thing made that was made."
)
DEFAULT_PPL_CHAR_LIMIT = 131072


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, math.ceil(len(values) * p / 100) - 1))
    return values[idx]


async def wait_ready(base_url: str, model: str, timeout_s: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        while time.monotonic() < deadline:
            try:
                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        return {"ready": True, "body": json.loads(body)}
                    last = body
            except Exception as exc:  # noqa: BLE001
                last = repr(exc)
            await asyncio.sleep(2)
    return {"ready": False, "last_error": last}


async def smoke_chat(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            body = await resp.json()
    elapsed = time.perf_counter() - start
    choice = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tok_s": completion_tokens / elapsed if elapsed > 0 and completion_tokens > 0 else 0.0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "text_preview": choice[:240],
    }


async def stream_ttft(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    first_token_at = None
    completion_tokens = 0
    pieces: list[str] = []
    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=600, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                evt = json.loads(data)
                choices = evt.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text", "")
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(text)
                usage = evt.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
    end = time.perf_counter()
    if completion_tokens == 0:
        joined = "".join(pieces)
        completion_tokens = max(0, len(joined.split()))
    return {
        "ttft_ms": (first_token_at - start) * 1000 if first_token_at is not None else None,
        "elapsed_s": end - start,
        "completion_tokens": completion_tokens,
        "tok_s": completion_tokens / max(1e-9, end - start),
        "text_preview": "".join(pieces)[:240],
    }


async def one_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any] | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                return None
            body = await resp.json()
    except Exception:  # noqa: BLE001
        return None
    elapsed = time.perf_counter() - start
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    if completion_tokens <= 0:
        return None
    return {
        "latency_ms": elapsed * 1000,
        "completion_tokens": completion_tokens,
        "tok_s": completion_tokens / elapsed,
    }


async def throughput_bench(
    base_url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    target_completion_tokens: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    per_req_tps: list[float] = []
    total_completion_tokens = 0
    errors = 0
    prompt_idx = 0
    lock = asyncio.Lock()

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal total_completion_tokens, errors, prompt_idx
        while True:
            async with lock:
                if total_completion_tokens >= target_completion_tokens:
                    return
                prompt = PROMPTS[prompt_idx % len(PROMPTS)]
                prompt_idx += 1
            result = await one_completion(session, base_url, model, prompt, max_tokens)
            async with lock:
                if result is None:
                    errors += 1
                else:
                    latencies.append(result["latency_ms"])
                    per_req_tps.append(result["tok_s"])
                    total_completion_tokens += result["completion_tokens"]

    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await asyncio.gather(*[worker(session) for _ in range(concurrency)])
    elapsed = time.perf_counter() - start
    latencies.sort()
    return {
        "elapsed_s": elapsed,
        "completion_tokens": total_completion_tokens,
        "throughput_tok_s": total_completion_tokens / elapsed if elapsed > 0 else 0.0,
        "requests": len(latencies),
        "errors": errors,
        "avg_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p50_latency_ms": percentile(latencies, 50) if latencies else None,
        "p95_latency_ms": percentile(latencies, 95) if latencies else None,
        "avg_req_tok_s": statistics.fmean(per_req_tps) if per_req_tps else None,
    }


async def ppl_probe(base_url: str, text: str) -> dict[str, Any]:
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 1,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async with session.post(f"{base_url}/generate", json=payload) as resp:
            body = await resp.json()
    meta = body.get("meta_info", {})
    token_logprobs = meta.get("input_token_logprobs")
    if not token_logprobs:
        return {"ok": False, "body": body}

    vals: list[float] = []
    for item in token_logprobs:
        if not item:
            continue
        lp = item[0]
        if lp is None:
            continue
        vals.append(float(lp))

    if not vals:
        return {"ok": False, "body": body}

    avg_nll = -sum(vals) / len(vals)
    ppl = math.exp(avg_nll)
    return {
        "ok": True,
        "tokens": len(vals),
        "avg_nll": avg_nll,
        "perplexity": ppl,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    if args.ppl_text_file:
        ppl_text = Path(args.ppl_text_file).read_text()
    else:
        default_ppl_path = Path("/workspace/wiki.test.raw")
        ppl_text = default_ppl_path.read_text() if default_ppl_path.exists() else DEFAULT_PPL_TEXT
    if args.ppl_char_limit > 0 and len(ppl_text) > args.ppl_char_limit:
        ppl_text = ppl_text[: args.ppl_char_limit]
    concurrencies = [int(x) for x in str(args.concurrency).split(",") if x.strip()]

    ready = await wait_ready(base_url, args.model, args.ready_timeout)
    if not ready.get("ready"):
        raise RuntimeError(f"server not ready: {ready.get('last_error')}")

    smoke = await smoke_chat(base_url, args.model, args.prompt, args.smoke_max_tokens)
    stream = await stream_ttft(base_url, args.model, args.prompt, args.stream_max_tokens)
    throughput = []
    for concurrency in concurrencies:
        throughput.append(
            {
                "concurrency": concurrency,
                **(
                    await throughput_bench(
                        base_url,
                        args.model,
                        concurrency,
                        args.bench_max_tokens,
                        args.target_completion_tokens,
                    )
                ),
            }
        )
    ppl = await ppl_probe(base_url, ppl_text)

    return {
        "base_url": base_url,
        "model": args.model,
        "smoke": smoke,
        "stream": stream,
        "throughput": throughput,
        "ppl": ppl,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a KTransformers Kimi K2.6 server")
    p.add_argument("--base-url", default="http://127.0.0.1:31245")
    p.add_argument("--model", default="Kimi-K2.6")
    p.add_argument(
        "--prompt",
        default="Write a compact Rust function that trims trailing zeros from a decimal string.",
    )
    p.add_argument("--ready-timeout", type=float, default=1800.0)
    p.add_argument("--smoke-max-tokens", type=int, default=96)
    p.add_argument("--stream-max-tokens", type=int, default=128)
    p.add_argument("--bench-max-tokens", type=int, default=128)
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--target-completion-tokens", type=int, default=2048)
    p.add_argument("--ppl-text-file")
    p.add_argument("--ppl-char-limit", type=int, default=DEFAULT_PPL_CHAR_LIMIT)
    p.add_argument("--output", default="-")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(main_async(args))
    out = json.dumps(result, indent=2)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).write_text(out)
        print(out)


if __name__ == "__main__":
    main()
