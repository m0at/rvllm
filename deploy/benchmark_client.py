#!/usr/bin/env python3
"""Benchmark client for rvllm and Python vLLM comparison.

Sends concurrent inference requests and measures:
- Throughput (tokens/sec)
- Time to first token (TTFT)
- Inter-token latency (ITL)
- Request latency (end-to-end)
- Tokens per second per request
"""

import asyncio
import aiohttp
import json
import time
import argparse
import numpy as np
from dataclasses import dataclass, asdict

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list of integers.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a short story about a robot learning to paint.",
    "Explain how a transformer neural network works.",
    "What are the advantages of Rust over C++?",
    "Describe the water cycle in detail.",
    "Write a haiku about machine learning.",
    "Explain the concept of recursion with an example.",
    "What is the difference between a stack and a queue?",
    "Describe how HTTPS encryption works.",
    "Write a SQL query to find duplicate records in a table.",
    "Explain the CAP theorem in distributed systems.",
    "What are the main principles of object-oriented programming?",
    "Describe the architecture of a modern CPU.",
    "Write a regular expression to validate email addresses.",
    "Explain how garbage collection works in Java.",
    "What is the difference between concurrency and parallelism?",
    "Describe the MapReduce programming model.",
    "Explain how a B-tree index works in databases.",
    "What are the trade-offs between microservices and monoliths?",
    "Describe the process of DNS resolution.",
    "Write pseudocode for the A* pathfinding algorithm.",
]


@dataclass
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    ttft_ms: float  # time to first token (for streaming)
    tokens_per_sec: float


@dataclass
class BenchmarkResult:
    server_url: str
    num_requests: int
    concurrency: int
    total_time_sec: float
    # Throughput
    total_tokens: int
    tokens_per_sec: float
    requests_per_sec: float
    # Latency stats
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    # Per-request token throughput
    avg_tps: float
    # Errors
    num_errors: int


async def send_request(session, url, prompt, max_tokens=128):
    """Send a single completion request and measure timing."""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    completion_tokens = 0

    try:
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        completion_tokens += 1
                    except json.JSONDecodeError:
                        pass
    except Exception:
        return None  # Error

    end = time.perf_counter()
    total_ms = (end - start) * 1000
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else total_ms

    # Fallback: if not streaming, use non-streaming endpoint
    if completion_tokens == 0:
        payload["stream"] = False
        start = time.perf_counter()
        try:
            async with session.post(f"{url}/v1/completions", json=payload) as resp:
                result = await resp.json()
                first_token_time = time.perf_counter()
                completion_tokens = result.get("usage", {}).get(
                    "completion_tokens", max_tokens
                )
        except Exception:
            return None
        end = time.perf_counter()
        total_ms = (end - start) * 1000
        ttft_ms = (first_token_time - start) * 1000

    prompt_tokens = len(prompt.split()) * 2  # rough estimate
    tps = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0

    return RequestResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_latency_ms=total_ms,
        ttft_ms=ttft_ms,
        tokens_per_sec=tps,
    )


async def run_benchmark(url, num_prompts, concurrency, max_tokens=128):
    """Run the full benchmark."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    errors = 0

    async def limited_request(session, prompt):
        nonlocal errors
        async with semaphore:
            result = await send_request(session, url, prompt, max_tokens)
            if result is None:
                errors += 1
            else:
                results.append(result)

    start = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        tasks = [limited_request(session, p) for p in prompts]
        await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    if not results:
        print(f"ERROR: All {num_prompts} requests failed!")
        return None

    latencies = [r.total_latency_ms for r in results]
    ttfts = [r.ttft_ms for r in results]
    total_tokens = sum(r.completion_tokens for r in results)

    return BenchmarkResult(
        server_url=url,
        num_requests=num_prompts,
        concurrency=concurrency,
        total_time_sec=total_time,
        total_tokens=total_tokens,
        tokens_per_sec=total_tokens / total_time,
        requests_per_sec=len(results) / total_time,
        avg_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        avg_ttft_ms=float(np.mean(ttfts)),
        p50_ttft_ms=float(np.percentile(ttfts, 50)),
        p95_ttft_ms=float(np.percentile(ttfts, 95)),
        avg_tps=float(np.mean([r.tokens_per_sec for r in results])),
        num_errors=errors,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark client for rvllm and Python vLLM"
    )
    parser.add_argument("--url", required=True, help="Server URL (e.g. http://localhost:8000)")
    parser.add_argument("--num-prompts", type=int, default=200, help="Number of prompts to send")
    parser.add_argument("--concurrent", type=int, default=16, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per completion")
    parser.add_argument("--output", default="results.json", help="Output JSON file path")
    args = parser.parse_args()

    print(f"Benchmarking {args.url}")
    print(
        f"  Prompts: {args.num_prompts}, Concurrency: {args.concurrent}, Max tokens: {args.max_tokens}"
    )

    result = asyncio.run(
        run_benchmark(args.url, args.num_prompts, args.concurrent, args.max_tokens)
    )

    if result:
        print(f"\nResults:")
        print(f"  Throughput: {result.tokens_per_sec:.1f} tok/s")
        print(f"  Requests/s: {result.requests_per_sec:.1f}")
        print(f"  Avg latency: {result.avg_latency_ms:.1f} ms")
        print(f"  P50 latency: {result.p50_latency_ms:.1f} ms")
        print(f"  P95 latency: {result.p95_latency_ms:.1f} ms")
        print(f"  Avg TTFT: {result.avg_ttft_ms:.1f} ms")
        print(f"  Errors: {result.num_errors}")

        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
