#!/usr/bin/env python3
"""
Compare Rust (criterion) vs Python benchmark results.

Reads:
  - Rust results from target/criterion/*/new/estimates.json
  - Python results from benches/python_results.json

Run: python3 benches/compare.py
"""

import json
import os
import sys
import math
from pathlib import Path

# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).resolve().parent.parent
CRITERION_DIR = BASE_DIR / "target" / "criterion"
PYTHON_RESULTS = BASE_DIR / "benches" / "python_results.json"

# Map from criterion directory names to canonical benchmark names.
# Criterion stores results under group_name/benchmark_id/new/estimates.json
# We build: "softmax/rust/32000" -> median_ns from criterion

def load_criterion_results():
    """Walk criterion output and extract median point estimates."""
    results = {}
    if not CRITERION_DIR.exists():
        print(f"{YELLOW}Warning: {CRITERION_DIR} not found. Run Rust benchmarks first:{RESET}")
        print(f"  cd {BASE_DIR} && cargo bench -p rvllm-bench")
        return results

    for estimates_path in CRITERION_DIR.rglob("new/estimates.json"):
        # Path like: target/criterion/softmax/rust 32000/new/estimates.json
        rel = estimates_path.relative_to(CRITERION_DIR)
        parts = list(rel.parts)
        # parts: [group, bench_id, "new", "estimates.json"]
        if len(parts) < 4:
            continue
        group = parts[0]
        bench_id = parts[1]

        try:
            with open(estimates_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        # criterion stores median.point_estimate in nanoseconds
        median_ns = None
        if "median" in data:
            median_ns = data["median"].get("point_estimate")
        elif "mean" in data:
            median_ns = data["mean"].get("point_estimate")

        if median_ns is None:
            continue

        # Reconstruct canonical name: group/bench_id
        # bench_id has spaces like "rust 32000" -> "rust/32000"
        canonical = f"{group}/{bench_id.replace(' ', '/')}"
        results[canonical] = median_ns

    return results


def load_python_results():
    """Load Python benchmark results from JSON."""
    if not PYTHON_RESULTS.exists():
        print(f"{YELLOW}Warning: {PYTHON_RESULTS} not found. Run Python benchmarks first:{RESET}")
        print(f"  python3 benches/bench_python.py")
        return {}

    with open(PYTHON_RESULTS) as f:
        data = json.load(f)

    results = {}
    for entry in data:
        results[entry["name"]] = entry["median_ns"]
    return results


def build_match_table(rust_results, python_results):
    """
    Match Rust and Python benchmarks by canonical name pattern.

    Rust names:  "softmax/rust/32000"
    Python names: "softmax/python/32000"

    We strip the implementation tag (rust/python/torch/python_k50/rust_k50 etc)
    to produce a common key, then pair them up.
    """
    def normalize_key(name):
        """Extract (group, variant, size) from benchmark name."""
        parts = name.split("/")
        if len(parts) < 2:
            return None, None, name

        group = parts[0]

        # Last part is usually the size/param or the full id
        # Middle parts are implementation identifiers
        # e.g. "softmax/rust/32000" -> group=softmax, impl=rust, size=32000
        # e.g. "top_k/rust_k50/32000" -> group=top_k, impl=rust_k50, size=32000
        # e.g. "repetition_penalty/rust_32k_500past" -> group=repetition_penalty, impl=..., size=None

        impl_tag = parts[1] if len(parts) >= 2 else ""
        size = parts[2] if len(parts) >= 3 else ""

        return group, impl_tag, size

    # Group by (group, normalized_impl_suffix, size)
    # Rust: rust, rust_k50, rust_p0.9, rust_top5, rust_32k_500past, rust_rayon
    # Python: python, python_k50, python_p0.9, python_top5, python_32k_500past, torch

    # Build a simpler approach: for each rust result, try to find a matching python one
    rows = []

    for rust_name, rust_ns in sorted(rust_results.items()):
        rparts = rust_name.split("/")
        group = rparts[0]
        rust_impl = rparts[1] if len(rparts) > 1 else ""
        size_part = "/".join(rparts[2:]) if len(rparts) > 2 else ""

        # Try to find python equivalent:
        # Replace "rust" prefix with "python" in the impl tag
        python_impl = rust_impl.replace("rust", "python")
        python_key = f"{group}/{python_impl}"
        if size_part:
            python_key += f"/{size_part}"

        py_ns = python_results.get(python_key)

        # Also try torch equivalent
        torch_impl = rust_impl.replace("rust", "torch")
        torch_key = f"{group}/{torch_impl}"
        if size_part:
            torch_key += f"/{size_part}"
        torch_ns = python_results.get(torch_key)

        # Build display name
        display = f"{group}"
        if size_part:
            # Extract suffix from impl tag (e.g. _k50, _p0.9)
            suffix = rust_impl.replace("rust", "").strip("_")
            if suffix:
                display += f"/{suffix}"
            display += f"/{size_part}"
        else:
            suffix = rust_impl.replace("rust", "").strip("_")
            if suffix:
                display += f"/{suffix}"

        rows.append({
            "name": display,
            "rust_ns": rust_ns,
            "python_ns": py_ns,
            "torch_ns": torch_ns,
        })

    return rows


def format_us(ns):
    """Format nanoseconds as microseconds string."""
    if ns is None:
        return "-"
    us = ns / 1000.0
    if us < 1.0:
        return f"{ns:.0f}ns"
    elif us < 1000.0:
        return f"{us:.1f}"
    elif us < 1_000_000.0:
        return f"{us/1000:.1f}ms"
    else:
        return f"{us/1_000_000:.2f}s"


def speedup_str(rust_ns, other_ns):
    if rust_ns is None or other_ns is None or rust_ns == 0:
        return "-", ""
    ratio = other_ns / rust_ns
    if ratio >= 1.0:
        return f"{ratio:.1f}x", GREEN
    else:
        return f"{1.0/ratio:.1f}x slower", RED


def print_table(rows):
    """Print formatted comparison table."""
    # Header
    hdr = f"{'Benchmark':<38s} | {'Rust (us)':>12s} | {'NumPy (us)':>12s} | {'Speedup':>14s} | {'Torch (us)':>12s} | {'vs Torch':>14s}"
    sep = "-" * len(hdr)

    print(f"\n{BOLD}{'='*len(hdr)}{RESET}")
    print(f"{BOLD}  RUST vs PYTHON BENCHMARK COMPARISON{RESET}")
    print(f"{BOLD}{'='*len(hdr)}{RESET}")
    print()
    print(f"  {BOLD}{hdr}{RESET}")
    print(f"  {sep}")

    speedups_numpy = []
    speedups_torch = []

    for row in rows:
        name = row["name"]
        rust_us = format_us(row["rust_ns"])
        py_us = format_us(row["python_ns"])
        torch_us = format_us(row["torch_ns"])

        sp_py, color_py = speedup_str(row["rust_ns"], row["python_ns"])
        sp_torch, color_torch = speedup_str(row["rust_ns"], row["torch_ns"])

        if row["python_ns"] is not None and row["rust_ns"] is not None and row["rust_ns"] > 0:
            speedups_numpy.append(row["python_ns"] / row["rust_ns"])

        if row["torch_ns"] is not None and row["rust_ns"] is not None and row["rust_ns"] > 0:
            speedups_torch.append(row["torch_ns"] / row["rust_ns"])

        line = f"  {name:<38s} | {rust_us:>12s} | {py_us:>12s} | {color_py}{sp_py:>14s}{RESET} | {torch_us:>12s} | {color_torch}{sp_torch:>14s}{RESET}"
        print(line)

    print(f"  {sep}")

    # Summary stats
    print(f"\n{BOLD}  Summary Statistics{RESET}")
    print(f"  {'-'*50}")

    if speedups_numpy:
        geo_mean = math.exp(sum(math.log(s) for s in speedups_numpy) / len(speedups_numpy))
        best = max(speedups_numpy)
        worst = min(speedups_numpy)
        color_geo = GREEN if geo_mean >= 1.0 else RED
        print(f"  vs NumPy:")
        print(f"    Geometric mean speedup:  {color_geo}{geo_mean:.2f}x{RESET}")
        print(f"    Best case:               {GREEN}{best:.1f}x{RESET}")
        if worst >= 1.0:
            print(f"    Worst case:              {GREEN}{worst:.2f}x{RESET}")
        else:
            print(f"    Worst case:              {RED}{1.0/worst:.2f}x slower{RESET}")
        wins = sum(1 for s in speedups_numpy if s >= 1.0)
        print(f"    Rust wins:               {wins}/{len(speedups_numpy)}")

    if speedups_torch:
        geo_mean = math.exp(sum(math.log(s) for s in speedups_torch) / len(speedups_torch))
        best = max(speedups_torch)
        worst = min(speedups_torch)
        color_geo = GREEN if geo_mean >= 1.0 else RED
        print(f"  vs Torch (CPU):")
        print(f"    Geometric mean speedup:  {color_geo}{geo_mean:.2f}x{RESET}")
        print(f"    Best case:               {GREEN}{best:.1f}x{RESET}")
        if worst >= 1.0:
            print(f"    Worst case:              {GREEN}{worst:.2f}x{RESET}")
        else:
            print(f"    Worst case:              {RED}{1.0/worst:.2f}x slower{RESET}")
        wins = sum(1 for s in speedups_torch if s >= 1.0)
        print(f"    Rust wins:               {wins}/{len(speedups_torch)}")

    print()


def main():
    print(f"{DIM}Loading Rust results from {CRITERION_DIR}{RESET}")
    rust_results = load_criterion_results()
    print(f"{DIM}  Found {len(rust_results)} Rust benchmarks{RESET}")

    print(f"{DIM}Loading Python results from {PYTHON_RESULTS}{RESET}")
    python_results = load_python_results()
    print(f"{DIM}  Found {len(python_results)} Python benchmarks{RESET}")

    if not rust_results and not python_results:
        print(f"\n{RED}No results found. Run benchmarks first:{RESET}")
        print(f"  cd {BASE_DIR} && cargo bench -p rvllm-bench")
        print(f"  cd {BASE_DIR} && python3 benches/bench_python.py")
        sys.exit(1)

    if not rust_results:
        print(f"\n{YELLOW}No Rust results. Showing Python-only results:{RESET}")
        for name, ns in sorted(python_results.items()):
            print(f"  {name:<55s}  {format_us(ns):>12s}")
        sys.exit(0)

    rows = build_match_table(rust_results, python_results)
    print_table(rows)


if __name__ == "__main__":
    main()
