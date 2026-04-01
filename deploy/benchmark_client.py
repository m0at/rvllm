#!/usr/bin/env python3
"""Benchmark client for rvllm and Python vLLM comparison.

Non-streaming mode: each request returns a complete JSON response.
Wall-clock latency = actual generation time. No TCP buffering artifacts.

Measures:
- Throughput (tokens/sec) = total completion tokens / total wall time
- Request latency (end-to-end per request)
- TTFT approximated as full-request latency / completion_tokens (first token)
"""

import asyncio
import aiohttp
import json
import time
import argparse
import sys

PROMPTS = [
    "Continue the following sequence of lowercase letters separated by spaces, with no punctuation and no explanation:\nalpha beta gamma delta epsilon",
    "Continue this comma-separated list with more common nouns only, no period and no explanation:\nchair, table, window, door, floor",
    "Continue writing short lowercase words separated by spaces. Do not stop early:\nred blue green yellow",
    "Write more entries in this numbered list, continuing the format exactly:\n1. one\n2. two\n3. three\n4.",
    "Repeat the word token separated by spaces. Keep going:\ntoken token token token",
]

JSON_HEADERS = {
    "Content-Type": "application/json",
    "Connection": "keep-alive",
}


async def send_request(
    session,
    url,
    payload_bytes,
):
    """Send a non-streaming completion request and measure wall-clock latency."""
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/completions",
            data=payload_bytes,
            headers=JSON_HEADERS,
        ) as resp:
            body = await resp.read()
            if resp.status != 200:
                return None
            result = json.loads(body)
    except Exception:
        return None

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    if completion_tokens == 0:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "tokens_per_sec": completion_tokens / (latency_ms / 1000),
    }


async def run_benchmark(
    url,
    num_prompts,
    concurrency,
    max_tokens=128,
    model="default",
    temperature=0.0,
    top_p=1.0,
    target_completion_tokens=0,
):
    results = []
    errors = 0
    dispatched_requests = 0
    completion_tokens_so_far = 0
    state_lock = asyncio.Lock()

    def make_payload(index: int) -> bytes:
        return json.dumps(
            {
                "model": model,
                "prompt": PROMPTS[index % len(PROMPTS)],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "ignore_eos": True,
                "stream": False,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    async def worker(session):
        nonlocal errors, dispatched_requests, completion_tokens_so_far
        while True:
            async with state_lock:
                if target_completion_tokens > 0:
                    if completion_tokens_so_far >= target_completion_tokens:
                        return
                elif dispatched_requests >= num_prompts:
                    return
                payload = make_payload(dispatched_requests)
                dispatched_requests += 1
            result = await send_request(session, url, payload)
            async with state_lock:
                if result is None:
                    errors += 1
                else:
                    results.append(result)
                    completion_tokens_so_far += result["completion_tokens"]

    start = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=3600)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600),
        connector=connector,
    ) as session:
        tasks = [worker(session) for _ in range(concurrency)]
        await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    if not results:
        print(f"ERROR: All {num_prompts} requests failed!")
        return None

    latencies = sorted([r["latency_ms"] for r in results])
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)

    def percentile(arr, p):
        idx = int(len(arr) * p / 100)
        return arr[min(idx, len(arr) - 1)]

    return {
        "server_url": url,
        "num_requests": dispatched_requests,
        "successful_requests": len(results),
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "target_completion_tokens": target_completion_tokens,
        "total_time_sec": total_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "throughput_tok_per_sec": total_completion_tokens / total_time,
        "requests_per_sec": len(results) / total_time,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "min_latency_ms": latencies[0],
        "max_latency_ms": latencies[-1],
        "avg_tokens_per_request": total_completion_tokens / len(results),
        "avg_tps_per_request": sum(r["tokens_per_sec"] for r in results) / len(results),
        "num_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark client for rvllm and Python vLLM (non-streaming)"
    )
    parser.add_argument("--url", required=True, help="Server URL")
    parser.add_argument("--model", default="default", help="Model name for request payload")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--target-completion-tokens", type=int, default=0)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    print(f"Benchmarking {args.url} (non-streaming)")
    print(f"  Model: {args.model}")
    print(
        f"  Prompts: {args.num_prompts}, Concurrency: {args.concurrent}, "
        f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}, Top-p: {args.top_p}, "
        f"Target completion tokens: {args.target_completion_tokens}"
    )

    result = asyncio.run(
        run_benchmark(
            args.url,
            args.num_prompts,
            args.concurrent,
            args.max_tokens,
            args.model,
            args.temperature,
            args.top_p,
            args.target_completion_tokens,
        )
    )

    if result:
        print(f"\nResults:")
        print(f"  Total time:  {result['total_time_sec']:.2f}s")
        print(f"  Throughput:  {result['throughput_tok_per_sec']:.1f} tok/s")
        print(f"  Requests/s:  {result['requests_per_sec']:.1f}")
        print(f"  Avg latency: {result['avg_latency_ms']:.1f} ms")
        print(f"  P50 latency: {result['p50_latency_ms']:.1f} ms")
        print(f"  P95 latency: {result['p95_latency_ms']:.1f} ms")
        print(f"  P99 latency: {result['p99_latency_ms']:.1f} ms")
        print(f"  Avg tok/req: {result['avg_tokens_per_request']:.1f}")
        print(f"  Errors:      {result['num_errors']}")

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
