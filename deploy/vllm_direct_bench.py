#!/usr/bin/env python3
"""Direct engine benchmark for Python vLLM (no HTTP).

Matches rvLLM's direct engine measurement: call the engine API directly,
bypass all HTTP/networking overhead. Measures pure generation throughput.

Usage:
    python3 deploy/vllm_direct_bench.py --model Qwen/Qwen2.5-7B --max-tokens 128
"""

import time
import json
import argparse

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


def bench_concurrency(
    llm,
    sampling_params,
    n,
    warmup=True,
    warmup_concurrency=None,
    warmup_sampling_params=None,
):
    """Benchmark at concurrency N by batching N prompts."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]

    if warmup:
        warmup_n = n if warmup_concurrency is None else min(n, warmup_concurrency)
        llm.generate(
            prompts[:warmup_n],
            warmup_sampling_params or sampling_params,
        )

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_tokens / elapsed

    return {
        "n": n,
        "total_tokens": total_tokens,
        "elapsed_ms": round(elapsed * 1000),
        "tok_per_sec": round(tok_per_sec, 1),
        "avg_tokens_per_req": round(total_tokens / n, 1),
        "failed": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM direct engine benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", default="/root/results_vllm_direct.json")
    parser.add_argument("--concurrency", type=str, default="1,4,16,32,64,128",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=4,
        help="Warmup prompt count. Use 0 to disable warmup.",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=None,
        help="Warmup max tokens. Defaults to --max-tokens.",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    load_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        max_model_len=args.max_model_len,
    )
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    warmup_sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=(
            args.max_tokens
            if args.warmup_max_tokens is None
            else args.warmup_max_tokens
        ),
        ignore_eos=args.ignore_eos,
    )

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    results = []

    print(f"\nDirect engine benchmark (no HTTP)")
    print(f"Model: {args.model}, max_tokens={args.max_tokens}")
    print(f"{'N':>6} | {'tok/s':>10} | {'tokens':>8} | {'elapsed':>8}")
    print("-" * 45)

    for n in concurrency_levels:
        r = bench_concurrency(
            llm,
            sampling_params,
            n,
            warmup=(n == concurrency_levels[0] and args.warmup_concurrency != 0),
            warmup_concurrency=args.warmup_concurrency,
            warmup_sampling_params=warmup_sampling_params,
        )
        results.append(r)
        print(f"{r['n']:>6} | {r['tok_per_sec']:>10,.1f} | {r['total_tokens']:>8,} | {r['elapsed_ms'] / 1000:>7.2f}s")

    output = {
        "engine": "vllm",
        "model": args.model,
        "output_len": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "warmup_concurrency": args.warmup_concurrency,
        "warmup_max_tokens": warmup_sampling_params.max_tokens,
        "load_time_sec": round(load_time, 1),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
