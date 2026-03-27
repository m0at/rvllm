#!/usr/bin/env python3
"""Compare rvllm vs Python vLLM benchmark results."""

import json
import argparse


def format_val(v, unit=""):
    if isinstance(v, float):
        return f"{v:.1f}{unit}"
    return f"{v}{unit}"


def speedup_str(rust_val, python_val, lower_is_better=True):
    if lower_is_better:
        ratio = python_val / rust_val if rust_val > 0 else 0
    else:
        ratio = rust_val / python_val if python_val > 0 else 0

    if ratio >= 1.0:
        return f"\033[32m{ratio:.2f}x Rust\033[0m"  # Green
    else:
        return f"\033[31m{1/ratio:.2f}x Python\033[0m"  # Red


def main():
    parser = argparse.ArgumentParser(
        description="Compare rvllm vs Python vLLM benchmark results"
    )
    parser.add_argument("--rust", required=True, help="Path to Rust benchmark results JSON")
    parser.add_argument("--python", required=True, help="Path to Python benchmark results JSON")
    args = parser.parse_args()

    with open(args.rust) as f:
        rust = json.load(f)
    with open(args.python) as f:
        python = json.load(f)

    print("=" * 80)
    print("  rvllm (Rust) vs vLLM (Python) -- A100 80GB Benchmark Comparison")
    print("=" * 80)
    print()

    metrics = [
        ("Throughput (tok/s)", "tokens_per_sec", False),
        ("Requests/sec", "requests_per_sec", False),
        ("Avg Latency (ms)", "avg_latency_ms", True),
        ("P50 Latency (ms)", "p50_latency_ms", True),
        ("P95 Latency (ms)", "p95_latency_ms", True),
        ("P99 Latency (ms)", "p99_latency_ms", True),
        ("Avg TTFT (ms)", "avg_ttft_ms", True),
        ("P50 TTFT (ms)", "p50_ttft_ms", True),
        ("P95 TTFT (ms)", "p95_ttft_ms", True),
        ("Avg TPS/request", "avg_tps", False),
    ]

    print(f"  {'Metric':<25s} {'Rust':>12s} {'Python':>12s} {'Winner':>20s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*20}")

    for name, key, lower_better in metrics:
        rv = rust.get(key, 0)
        pv = python.get(key, 0)
        winner = speedup_str(rv, pv, lower_better)
        print(f"  {name:<25s} {format_val(rv):>12s} {format_val(pv):>12s} {winner:>20s}")

    print()
    print(f"  Rust errors: {rust.get('num_errors', 0)}")
    print(f"  Python errors: {python.get('num_errors', 0)}")
    print()

    # Resource usage (if captured)
    resource_metrics = [
        ("Startup (ms)", "startup_ms", True),
        ("CPU RSS (MB)", "cpu_rss_mb", True),
        ("GPU VRAM (GB)", "gpu_vram_gb", True),
    ]

    has_resources = any(rust.get(key) for _, key, _ in resource_metrics)
    if has_resources:
        print(f"  {'Resource Metric':<25s} {'Rust':>12s} {'Python':>12s} {'Winner':>20s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*20}")
        for name, key, lower_better in resource_metrics:
            rv = rust.get(key)
            pv = python.get(key)
            if rv is not None and pv is not None:
                winner = speedup_str(rv, pv, lower_better)
                print(f"  {name:<25s} {format_val(rv):>12s} {format_val(pv):>12s} {winner:>20s}")
        print()

    # Overall summary
    tok_ratio = rust.get("tokens_per_sec", 0) / max(python.get("tokens_per_sec", 1), 1)
    lat_ratio = max(python.get("p50_latency_ms", 1), 1) / max(
        rust.get("p50_latency_ms", 1), 1
    )

    print("  SUMMARY:")
    print(
        f"    Throughput: Rust is {tok_ratio:.2f}x {'faster' if tok_ratio > 1 else 'slower'}"
    )
    print(
        f"    Latency:   Rust is {lat_ratio:.2f}x {'faster' if lat_ratio > 1 else 'slower'}"
    )

    if has_resources:
        rs = rust.get("startup_ms", 0)
        ps = python.get("startup_ms", 0)
        if rs > 0:
            print(f"    Startup:   Rust is {ps/rs:.1f}x faster")
        rc = rust.get("cpu_rss_mb", 0)
        pc = python.get("cpu_rss_mb", 0)
        if rc > 0:
            print(f"    CPU RSS:   Rust uses {pc/rc:.1f}x less memory")
        rg = rust.get("gpu_vram_gb", 0)
        pg = python.get("gpu_vram_gb", 0)
        if rg > 0:
            print(f"    GPU VRAM:  Rust uses {pg/rg:.1f}x less VRAM")

    print("=" * 80)


if __name__ == "__main__":
    main()
