#!/usr/bin/env python3

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary.json"
RV_KERN = ROOT / "rvllm_n64.kern.txt"
VV_KERN = ROOT / "vllm_n64.kern.txt"


def load_summary():
    return json.loads(SUMMARY.read_text())


def parse_kern_table(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 2:
            continue
        try:
            pct = float(parts[0])
        except ValueError:
            continue
        name = parts[-1]
        rows.append((pct, name))
    return rows


def categorize_kernel(name: str):
    lower = name.lower()
    if "fa3_v3_decode" in lower or "flashattnfwd" in name or "flash_attention_3" in lower:
        return "Attention"
    if "silu" in lower or "swiglu" in lower:
        return "FFN pointwise"
    if "rmsnorm" in lower or "pow_rsqrt" in lower:
        return "Norm"
    if "reshape_and_cache" in lower or "rotary" in lower or "rope" in lower or "deinterleave_qkv" in lower:
        return "KV / rope / layout"
    if name.startswith("nvjet_hsh") or name.startswith("nvjet_tst") or name.startswith("nvjet_hss") or "cublaslt::splitkreduce" in lower:
        return "GEMM"
    if "argmax" in lower or "topk" in lower or "softmax" in lower or "lm_head" in lower:
        return "Sampling / LM head"
    return "Other"


def aggregate_categories(rows):
    totals = {
        "GEMM": 0.0,
        "Attention": 0.0,
        "FFN pointwise": 0.0,
        "Norm": 0.0,
        "KV / rope / layout": 0.0,
        "Sampling / LM head": 0.0,
        "Other": 0.0,
    }
    for pct, name in rows[:80]:
        totals[categorize_kernel(name)] += pct
    return totals


def build_throughput_figure(summary):
    ns = summary["n_values"]
    rv = [summary["engines"]["rvllm"]["benchmark"][str(n)]["tok_per_sec"] for n in ns]
    vv = [summary["engines"]["vllm"]["benchmark"][str(n)]["tok_per_sec"] for n in ns]
    gaps = [(1.0 - r / v) * 100.0 for r, v in zip(rv, vv)]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(10.5, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ns, rv, color="#0F4C81", marker="o", linewidth=2.5, label="rvLLM")
    ax1.plot(ns, vv, color="#CC5A2A", marker="o", linewidth=2.5, label="vLLM 0.19")
    ax1.fill_between(ns, rv, vv, color="#E9EEF5", alpha=0.9)
    ax1.set_title("Fair Throughput Comparison on H100", fontsize=16, weight="bold")
    ax1.set_ylabel("tokens / second")
    ax1.set_xticks(ns)
    ax1.grid(axis="y", color="#D8DEE9", linewidth=0.8)
    ax1.legend(frameon=False, loc="upper left")

    for n, r, v in zip(ns, rv, vv):
        ax1.text(n, r - max(rv) * 0.03, f"{r:.0f}", color="#0F4C81", ha="center", va="top", fontsize=8)
        ax1.text(n, v + max(vv) * 0.02, f"{v:.0f}", color="#CC5A2A", ha="center", va="bottom", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar([str(n) for n in ns], gaps, color=["#C44E52" if g > 15 else "#DD8452" if g > 10 else "#55A868" for g in gaps])
    ax2.set_ylabel("% behind vLLM")
    ax2.set_ylim(0, max(gaps) * 1.25)
    ax2.grid(axis="y", color="#E5E9F0", linewidth=0.8)
    for bar, gap in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{gap:.1f}%", ha="center", va="bottom", fontsize=9)

    out = ROOT / "throughput_gap.pdf"
    fig.savefig(out)
    plt.close(fig)


def build_kernel_category_figure(rv_rows, vv_rows):
    rv = aggregate_categories(rv_rows)
    vv = aggregate_categories(vv_rows)
    cats = ["GEMM", "Attention", "FFN pointwise", "Norm", "KV / rope / layout", "Sampling / LM head"]

    x = np.arange(len(cats))
    width = 0.38

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    ax.bar(x - width / 2, [rv[c] for c in cats], width=width, color="#0F4C81", label="rvLLM")
    ax.bar(x + width / 2, [vv[c] for c in cats], width=width, color="#CC5A2A", label="vLLM 0.19")
    ax.set_title("N=64 Kernel Time Composition by Functional Category", fontsize=16, weight="bold")
    ax.set_ylabel("share of GPU kernel time (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["GEMM", "Attention", "FFN\npointwise", "Norm", "KV / rope\n/ layout", "Sampling /\nLM head"])
    ax.grid(axis="y", color="#E5E9F0", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    out = ROOT / "kernel_categories.pdf"
    fig.savefig(out)
    plt.close(fig)


def build_exec_summary(summary):
    ns = summary["n_values"]
    rv = {n: summary["engines"]["rvllm"]["benchmark"][str(n)]["tok_per_sec"] for n in ns}
    vv = {n: summary["engines"]["vllm"]["benchmark"][str(n)]["tok_per_sec"] for n in ns}
    lines = []
    lines.append(f"Model: {summary['model']}")
    lines.append("Matched settings: output_len=128, max_model_len=4096, gpu_memory_utilization=0.85")
    lines.append("")
    lines.append("Throughput table")
    for n in ns:
        gap = (1.0 - rv[n] / vv[n]) * 100.0
        lines.append(f"N={n:>3}: rvLLM {rv[n]:>7.1f} tok/s | vLLM {vv[n]:>7.1f} tok/s | gap {gap:>5.1f}%")
    lines.append("")
    lines.append("Profiler read")
    lines.append("rvLLM: attention decode kernel, rvLLM GEMMs, RMSNorm, and SiLU pointwise still dominate.")
    lines.append("vLLM: larger nvjet_tst GEMMs, Triton fused SiLU*mul, FlashAttention forward, and cache update are tighter.")
    lines.append("")
    lines.append("Interpretation")
    lines.append("The remaining gap is compute-side. The main losses are FFN/GEMM/fused pointwise quality, not benchmark harness overhead.")
    (ROOT / "exec_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    summary = load_summary()
    rv_rows = parse_kern_table(RV_KERN)
    vv_rows = parse_kern_table(VV_KERN)
    build_throughput_figure(summary)
    build_kernel_category_figure(rv_rows, vv_rows)
    build_exec_summary(summary)


if __name__ == "__main__":
    main()
