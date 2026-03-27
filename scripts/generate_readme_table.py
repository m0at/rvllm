#!/usr/bin/env python3
"""Read final_benchmark.json and output markdown tables for the README.

Usage:
    python3 scripts/generate_readme_table.py --input results/final_benchmark.json
    python3 scripts/generate_readme_table.py --input results/final_benchmark.json --output-txt results/report.txt
    python3 scripts/generate_readme_table.py --input results/final_benchmark.json --markdown  # README-ready
"""

import argparse
import json
import sys
from pathlib import Path


def ratio_str(rust_val, python_val, lower_is_better=True):
    """Format improvement ratio. Returns e.g. '3.3x faster' or '3.0x less'."""
    if rust_val == 0 and python_val == 0:
        return "--"
    if lower_is_better:
        if rust_val == 0:
            return "inf"
        r = python_val / rust_val
        if r >= 1:
            return f"**{r:.1f}x**"
        return f"{1/r:.1f}x worse"
    else:
        # higher is better (throughput)
        if python_val == 0:
            return "inf"
        r = rust_val / python_val
        if r >= 1:
            return f"**{r:.1f}x**"
        return f"{1/r:.1f}x worse"


def fmt_num(val, suffix="", decimals=1):
    if val is None or val == 0:
        return "--"
    if val >= 1000:
        return f"{val:,.0f}{suffix}"
    return f"{val:.{decimals}f}{suffix}"


def build_resource_table(data):
    rust = data["rust"]
    py = data["python"]

    rows = []
    rows.append("## Resource Usage\n")
    rows.append(f"Model: `{data['model']}` | GPU: {data.get('gpu', 'unknown')}\n")
    rows.append("| Metric | rvLLM (Rust) | Python vLLM | Improvement |")
    rows.append("|---|---:|---:|---:|")

    # Startup
    rs = rust.get("startup_ms", 0)
    ps = py.get("startup_ms", 0)
    rows.append(
        f"| Startup time | {rs/1000:.1f}s | {ps/1000:.1f}s | {ratio_str(rs, ps)} faster |"
    )

    # CPU RSS
    rc = rust.get("cpu_rss_mb", 0)
    pc = py.get("cpu_rss_mb", 0)
    rows.append(
        f"| CPU RSS | {fmt_num(rc, ' MB')} | {fmt_num(pc, ' MB')} | {ratio_str(rc, pc)} less |"
    )

    # GPU VRAM
    rg = rust.get("gpu_vram_mb", 0)
    pg = py.get("gpu_vram_mb", 0)
    rows.append(
        f"| GPU VRAM | {fmt_num(rg, ' MB')} | {fmt_num(pg, ' MB')} | {ratio_str(rg, pg)} less |"
    )

    return "\n".join(rows)


def build_throughput_table(data, max_tokens):
    """Build a concurrency sweep table for a given max_tokens setting."""
    rust_tests = data["rust"].get("load_tests", {})
    py_tests = data["python"].get("load_tests", {})
    concurrency_levels = data.get("concurrency_levels", [1, 4, 8, 16, 32])

    rows = []
    rows.append(f"\n## Throughput (max_tokens={max_tokens})\n")
    rows.append(
        "| Concurrency | rvLLM tok/s | Python tok/s | Speedup "
        "| rvLLM P50 (ms) | Python P50 (ms) | Latency ratio |"
    )
    rows.append("|---:|---:|---:|---:|---:|---:|---:|")

    for conc in concurrency_levels:
        label = f"c{conc}_t{max_tokens}"
        rt = rust_tests.get(label, {})
        pt = py_tests.get(label, {})

        r_tps = rt.get("tokens_per_sec", 0)
        p_tps = pt.get("tokens_per_sec", 0)
        r_p50 = rt.get("p50_latency_ms", 0)
        p_p50 = pt.get("p50_latency_ms", 0)

        tps_ratio = ratio_str(r_tps, p_tps, lower_is_better=False)
        lat_ratio = ratio_str(r_p50, p_p50, lower_is_better=True)

        rows.append(
            f"| {conc} "
            f"| {fmt_num(r_tps)} "
            f"| {fmt_num(p_tps)} "
            f"| {tps_ratio} "
            f"| {fmt_num(r_p50)} "
            f"| {fmt_num(p_p50)} "
            f"| {lat_ratio} |"
        )

    return "\n".join(rows)


def build_latency_detail_table(data, max_tokens):
    """Detailed latency percentiles for a given max_tokens."""
    rust_tests = data["rust"].get("load_tests", {})
    py_tests = data["python"].get("load_tests", {})
    concurrency_levels = data.get("concurrency_levels", [1, 4, 8, 16, 32])

    rows = []
    rows.append(f"\n## Latency Percentiles (max_tokens={max_tokens})\n")
    rows.append(
        "| Conc | rvLLM P50 | rvLLM P95 | rvLLM P99 "
        "| Python P50 | Python P95 | Python P99 "
        "| rvLLM TTFT | Python TTFT |"
    )
    rows.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for conc in concurrency_levels:
        label = f"c{conc}_t{max_tokens}"
        rt = rust_tests.get(label, {})
        pt = py_tests.get(label, {})

        rows.append(
            f"| {conc} "
            f"| {fmt_num(rt.get('p50_latency_ms', 0))} ms "
            f"| {fmt_num(rt.get('p95_latency_ms', 0))} ms "
            f"| {fmt_num(rt.get('p99_latency_ms', 0))} ms "
            f"| {fmt_num(pt.get('p50_latency_ms', 0))} ms "
            f"| {fmt_num(pt.get('p95_latency_ms', 0))} ms "
            f"| {fmt_num(pt.get('p99_latency_ms', 0))} ms "
            f"| {fmt_num(rt.get('avg_ttft_ms', 0))} ms "
            f"| {fmt_num(pt.get('avg_ttft_ms', 0))} ms |"
        )

    return "\n".join(rows)


def build_error_table(data):
    """Show error counts per test."""
    rust_tests = data["rust"].get("load_tests", {})
    py_tests = data["python"].get("load_tests", {})

    rust_errors = sum(t.get("num_errors", 0) for t in rust_tests.values())
    py_errors = sum(t.get("num_errors", 0) for t in py_tests.values())

    if rust_errors == 0 and py_errors == 0:
        return "\n## Errors\n\nZero errors across all test configurations for both servers."

    rows = ["\n## Errors\n"]
    rows.append("| Config | rvLLM errors | Python errors |")
    rows.append("|---|---:|---:|")
    for label in sorted(set(list(rust_tests.keys()) + list(py_tests.keys()))):
        re = rust_tests.get(label, {}).get("num_errors", 0)
        pe = py_tests.get(label, {}).get("num_errors", 0)
        if re > 0 or pe > 0:
            rows.append(f"| {label} | {re} | {pe} |")

    if len(rows) == 3:
        return "\n## Errors\n\nZero errors across all test configurations for both servers."
    return "\n".join(rows)


def build_summary_table(data):
    """One-glance summary: pick the highest concurrency, max_tokens=128 as the headline."""
    concurrency_levels = data.get("concurrency_levels", [1, 4, 8, 16, 32])
    max_conc = max(concurrency_levels)
    headline_label = f"c{max_conc}_t128"

    # Fall back to whatever max_tokens is available
    rust_tests = data["rust"].get("load_tests", {})
    py_tests = data["python"].get("load_tests", {})
    if headline_label not in rust_tests:
        max_tokens_list = data.get("max_tokens_list", [128])
        headline_label = f"c{max_conc}_t{max(max_tokens_list)}"

    rt = rust_tests.get(headline_label, {})
    pt = py_tests.get(headline_label, {})

    rust = data["rust"]
    py = data["python"]

    rows = []
    rows.append("## Headline Results\n")
    rows.append(
        f"Measured on {data.get('gpu', 'A100')}, `{data['model']}`, "
        f"{data['num_prompts']} prompts, concurrency {max_conc}.\n"
    )
    rows.append("| Metric | rvLLM (Rust) | Python vLLM | Improvement |")
    rows.append("|---|---:|---:|---:|")

    # Throughput
    r_tps = rt.get("tokens_per_sec", 0)
    p_tps = pt.get("tokens_per_sec", 0)
    rows.append(
        f"| Throughput | {fmt_num(r_tps, ' tok/s', 0)} "
        f"| {fmt_num(p_tps, ' tok/s', 0)} "
        f"| {ratio_str(r_tps, p_tps, lower_is_better=False)} faster |"
    )

    # Request rate
    r_rps = rt.get("requests_per_sec", 0)
    p_rps = pt.get("requests_per_sec", 0)
    rows.append(
        f"| Request rate | {fmt_num(r_rps, ' req/s', 0)} "
        f"| {fmt_num(p_rps, ' req/s', 0)} "
        f"| {ratio_str(r_rps, p_rps, lower_is_better=False)} faster |"
    )

    # P50 latency
    r_p50 = rt.get("p50_latency_ms", 0)
    p_p50 = pt.get("p50_latency_ms", 0)
    rows.append(
        f"| P50 Latency | {fmt_num(r_p50, ' ms')} "
        f"| {fmt_num(p_p50, ' ms')} "
        f"| {ratio_str(r_p50, p_p50)} lower |"
    )

    # P95 latency
    r_p95 = rt.get("p95_latency_ms", 0)
    p_p95 = pt.get("p95_latency_ms", 0)
    rows.append(
        f"| P95 Latency | {fmt_num(r_p95, ' ms')} "
        f"| {fmt_num(p_p95, ' ms')} "
        f"| {ratio_str(r_p95, p_p95)} lower |"
    )

    # TTFT
    r_ttft = rt.get("avg_ttft_ms", 0)
    p_ttft = pt.get("avg_ttft_ms", 0)
    rows.append(
        f"| Avg TTFT | {fmt_num(r_ttft, ' ms')} "
        f"| {fmt_num(p_ttft, ' ms')} "
        f"| {ratio_str(r_ttft, p_ttft)} lower |"
    )

    # Startup
    rs = rust.get("startup_ms", 0)
    ps = py.get("startup_ms", 0)
    rows.append(
        f"| Startup | {rs/1000:.0f}s | {ps/1000:.0f}s "
        f"| {ratio_str(rs, ps)} faster |"
    )

    # CPU RSS
    rc = rust.get("cpu_rss_mb", 0)
    pc = py.get("cpu_rss_mb", 0)
    rows.append(
        f"| CPU memory (RSS) | {fmt_num(rc, ' MB', 0)} "
        f"| {fmt_num(pc, ' MB', 0)} "
        f"| {ratio_str(rc, pc)} less |"
    )

    # GPU VRAM
    rg = rust.get("gpu_vram_mb", 0)
    pg = py.get("gpu_vram_mb", 0)
    rows.append(
        f"| GPU VRAM | {fmt_num(rg, ' MB', 0)} "
        f"| {fmt_num(pg, ' MB', 0)} "
        f"| {ratio_str(rg, pg)} less |"
    )

    # Errors
    r_err = rt.get("num_errors", 0)
    p_err = pt.get("num_errors", 0)
    rows.append(f"| Errors | {r_err} | {p_err} | -- |")

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison tables")
    parser.add_argument(
        "--input", required=True, help="Path to final_benchmark.json"
    )
    parser.add_argument(
        "--output-txt", default=None, help="Write plain text report to file"
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Output README-ready markdown (default: prints to stdout)"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    max_tokens_list = data.get("max_tokens_list", [128])

    sections = []
    sections.append(build_summary_table(data))
    sections.append(build_resource_table(data))

    for mt in max_tokens_list:
        sections.append(build_throughput_table(data, mt))

    for mt in max_tokens_list:
        sections.append(build_latency_detail_table(data, mt))

    sections.append(build_error_table(data))

    report = "\n\n".join(sections) + "\n"

    if args.output_txt:
        Path(args.output_txt).write_text(report)
        print(f"Report written to {args.output_txt}", file=sys.stderr)

    print(report)


if __name__ == "__main__":
    main()
