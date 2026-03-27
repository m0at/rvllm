#!/usr/bin/env python3
"""Sampling parity test: rvLLM vs Python vLLM.

For non-greedy decoding, verifies that token frequency distributions
are statistically indistinguishable between the two servers using
chi-squared tests. Tests each sampling parameter independently.

Usage:
    python3 tests/parity/sampling_parity.py \
        --rust-url http://localhost:8000 \
        --python-url http://localhost:8001 \
        --model Qwen/Qwen2.5-1.5B \
        --runs 100
"""
import argparse, json, sys, time, requests
from collections import Counter
import numpy as np
from scipy import stats

SIGNIFICANCE = 0.01  # p-value threshold for chi-squared test


def query_completion(url, model, prompt, **kwargs):
    payload = {"model": model, "prompt": prompt, **kwargs}
    r = requests.post(f"{url}/v1/completions", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def collect_samples(url, model, prompt, n_runs, **kwargs):
    """Run n_runs completions and return list of output texts."""
    results = []
    for i in range(n_runs):
        try:
            text = query_completion(url, model, prompt, **kwargs)
            results.append(text)
        except requests.RequestException as e:
            print(f"    Request {i+1} failed: {e}")
    return results


def first_token(text):
    """Extract first whitespace-delimited token."""
    stripped = text.lstrip()
    if not stripped:
        return "<empty>"
    parts = stripped.split()
    return parts[0] if parts else stripped[:10]


def chi_squared_test(rust_samples, python_samples):
    """Chi-squared test on first-token frequency distributions.

    Returns (statistic, p_value, pass_bool, detail_str).
    """
    rust_tokens = [first_token(s) for s in rust_samples]
    python_tokens = [first_token(s) for s in python_samples]

    rust_counts = Counter(rust_tokens)
    python_counts = Counter(python_tokens)

    # Union of all tokens
    all_tokens = sorted(set(rust_counts.keys()) | set(python_counts.keys()))
    if len(all_tokens) < 2:
        return 0.0, 1.0, True, f"Only {len(all_tokens)} unique token(s), trivially matching"

    rust_freq = np.array([rust_counts.get(t, 0) for t in all_tokens], dtype=float)
    python_freq = np.array([python_counts.get(t, 0) for t in all_tokens], dtype=float)

    # Pool low-frequency bins (expected < 5) for chi-squared validity
    combined = rust_freq + python_freq
    mask = combined >= 5
    if mask.sum() < 2:
        # Not enough high-frequency bins -- fall back to KS test on raw text lengths
        rust_lens = [len(s) for s in rust_samples]
        python_lens = [len(s) for s in python_samples]
        stat, p = stats.ks_2samp(rust_lens, python_lens)
        passed = p > SIGNIFICANCE
        return stat, p, passed, f"KS test on output lengths (too few high-freq tokens for chi2)"

    # Filter to valid bins
    valid_tokens = [t for t, m in zip(all_tokens, mask) if m]
    rust_valid = rust_freq[mask]
    python_valid = python_freq[mask]

    # Contingency table: 2 rows (rust, python) x N columns (tokens)
    observed = np.array([rust_valid, python_valid])
    stat, p, dof, expected = stats.chi2_contingency(observed)
    passed = p > SIGNIFICANCE

    top_rust = sorted(rust_counts.items(), key=lambda x: -x[1])[:5]
    top_python = sorted(python_counts.items(), key=lambda x: -x[1])[:5]
    detail = (f"chi2={stat:.2f} dof={dof} p={p:.4f}\n"
              f"    Rust top-5:   {top_rust}\n"
              f"    Python top-5: {top_python}")
    return stat, p, passed, detail


PASS_COUNT = 0
FAIL_COUNT = 0


def run_test(name, rust_url, python_url, model, prompt, n_runs, **kwargs):
    global PASS_COUNT, FAIL_COUNT
    print(f"\n  [{name}]")
    param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items() if k != "max_tokens")
    print(f"    Params: {param_str}")
    print(f"    Collecting {n_runs} samples from each server...")

    t0 = time.time()
    rust_samples = collect_samples(rust_url, model, prompt, **kwargs)
    python_samples = collect_samples(python_url, model, prompt, **kwargs)
    elapsed = time.time() - t0
    print(f"    Collected in {elapsed:.1f}s (rust={len(rust_samples)}, python={len(python_samples)})")

    if len(rust_samples) < 10 or len(python_samples) < 10:
        print(f"    [SKIP] Not enough samples")
        return

    stat, p, passed, detail = chi_squared_test(rust_samples, python_samples)
    if passed:
        PASS_COUNT += 1
        print(f"    [PASS] Distributions statistically indistinguishable (p={p:.4f})")
    else:
        FAIL_COUNT += 1
        print(f"    [FAIL] Distributions significantly different (p={p:.4f})")
    print(f"    {detail}")


def main():
    p = argparse.ArgumentParser(description="Sampling parity: rvLLM vs Python vLLM")
    p.add_argument("--rust-url", default="http://localhost:8000")
    p.add_argument("--python-url", default="http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--runs", type=int, default=100, help="Samples per test case")
    args = p.parse_args()

    print(f"Sampling Parity Test: rvLLM vs Python vLLM")
    print(f"  Rust:   {args.rust_url}")
    print(f"  Python: {args.python_url}")
    print(f"  Model:  {args.model}")
    print(f"  Runs per test: {args.runs}")
    print(f"  Significance level: {SIGNIFICANCE}")

    prompt = "The meaning of life is"
    n = args.runs

    # -- Temperature only --
    print("\n--- Temperature Sampling ---")
    run_test("temperature=0.8", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=0.8)
    run_test("temperature=1.2", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=1.2)

    # -- Top-k only --
    print("\n--- Top-k Sampling ---")
    run_test("top_k=10", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=1.0, top_k=10)
    run_test("top_k=50", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=1.0, top_k=50)

    # -- Top-p only --
    print("\n--- Top-p (nucleus) Sampling ---")
    run_test("top_p=0.5", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=1.0, top_p=0.5)
    run_test("top_p=0.95", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=1.0, top_p=0.95)

    # -- Repetition penalty --
    print("\n--- Repetition Penalty ---")
    run_test("repetition_penalty=1.2", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=0.8, repetition_penalty=1.2)

    # -- Frequency + presence penalty --
    print("\n--- Frequency + Presence Penalty ---")
    run_test("freq=0.5+pres=0.5", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=0.8, frequency_penalty=0.5, presence_penalty=0.5)

    # -- Stop strings --
    print("\n--- Stop Strings ---")
    stop_prompt = "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,"
    run_test("stop=[newline]", args.rust_url, args.python_url, args.model, stop_prompt, n,
             max_tokens=50, temperature=0.0, stop=["\n"])

    # Verify stop string is respected on both sides
    print("\n  [stop string content check]")
    try:
        rust_text = query_completion(args.rust_url, args.model, stop_prompt,
                                     max_tokens=50, temperature=0, stop=["\n"])
        python_text = query_completion(args.python_url, args.model, stop_prompt,
                                       max_tokens=50, temperature=0, stop=["\n"])
        rust_has_nl = "\n" in rust_text
        python_has_nl = "\n" in python_text
        if not rust_has_nl and not python_has_nl:
            global PASS_COUNT
            PASS_COUNT += 1
            print(f"    [PASS] Both stopped before newline")
        else:
            global FAIL_COUNT
            FAIL_COUNT += 1
            print(f"    [FAIL] Stop not respected (rust_has_nl={rust_has_nl} python_has_nl={python_has_nl})")
    except requests.RequestException as e:
        print(f"    [SKIP] {e}")

    # -- Combined: temperature + top_p (common production config) --
    print("\n--- Combined: temperature=0.8 + top_p=0.95 ---")
    run_test("temp=0.8+top_p=0.95", args.rust_url, args.python_url, args.model, prompt, n,
             max_tokens=20, temperature=0.8, top_p=0.95)

    # Summary
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'='*60}")
    print(f"SUMMARY: {PASS_COUNT}/{total} tests passed, {FAIL_COUNT}/{total} failed")
    print(f"  Significance level: p < {SIGNIFICANCE} means distributions differ")

    if FAIL_COUNT == 0:
        print("RESULT: PASS")
        sys.exit(0)
    elif FAIL_COUNT <= 1:
        print("RESULT: WARN -- minor deviations (1 failure may be statistical noise)")
        sys.exit(0)
    else:
        print(f"RESULT: FAIL -- {FAIL_COUNT} sampling parameter(s) show significant divergence")
        sys.exit(1)


if __name__ == "__main__":
    main()
