#!/usr/bin/env python3
"""Equivalent Python/numpy/torch benchmarks for comparison with Rust rvllm."""

import time
import numpy as np
import json
import sys

SEED = 42

# Try importing torch -- optional
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("torch not available, skipping torch benchmarks")

# ---------------------------------------------------------------------------
# numpy implementations
# ---------------------------------------------------------------------------

def softmax(logits):
    m = np.max(logits)
    e = np.exp(logits - m)
    return e / np.sum(e)

def softmax_inplace(logits):
    """In-place softmax on a numpy array."""
    m = np.max(logits)
    np.subtract(logits, m, out=logits)
    np.exp(logits, out=logits)
    s = np.sum(logits)
    logits /= s
    return logits

def log_softmax(logits):
    m = np.max(logits)
    lse = m + np.log(np.sum(np.exp(logits - m)))
    return logits - lse

def apply_temperature(logits, temperature):
    return logits / temperature

def apply_top_k(logits, k):
    if k == 0 or k >= len(logits):
        return logits
    logits = logits.copy()
    indices = np.argpartition(logits, -k)[-k:]
    mask = np.ones(len(logits), dtype=bool)
    mask[indices] = False
    logits[mask] = float('-inf')
    return logits

def apply_top_p(logits, p):
    probs = softmax(logits)
    sorted_idx = np.argsort(-probs)
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, p) + 1
    keep = sorted_idx[:cutoff]
    logits = np.full_like(logits, float('-inf'))
    logits[keep] = probs[keep]
    return logits

def apply_min_p(logits, min_p):
    probs = softmax(logits)
    max_prob = np.max(probs)
    threshold = max_prob * min_p
    logits = logits.copy()
    logits[probs < threshold] = float('-inf')
    return logits

def apply_repetition_penalty(logits, past_tokens, penalty):
    logits = logits.copy()
    past = np.array(past_tokens, dtype=np.int64)
    past = past[past < len(logits)]
    pos_mask = logits[past] > 0
    logits[past] = np.where(pos_mask, logits[past] / penalty, logits[past] * penalty)
    return logits

def apply_frequency_presence_penalty(logits, token_counts, freq_penalty, presence_penalty):
    logits = logits.copy()
    for tok, count in token_counts.items():
        if tok < len(logits) and count > 0:
            logits[tok] -= freq_penalty * count + presence_penalty
    return logits

def apply_combined_penalties(logits, past_tokens, token_counts, rep_penalty, freq_penalty, presence_penalty):
    logits = apply_repetition_penalty(logits, past_tokens, rep_penalty)
    logits = apply_frequency_presence_penalty(logits, token_counts, freq_penalty, presence_penalty)
    return logits

def multinomial_sample(probs, rng):
    r = rng.random()
    cumsum = np.cumsum(probs)
    return int(np.searchsorted(cumsum, r))

def greedy_sample(logits):
    return int(np.argmax(logits))

def top_logprobs(logits, n):
    lp = log_softmax(logits)
    idx = np.argpartition(-lp, n)[:n]
    idx = idx[np.argsort(-lp[idx])]
    return list(zip(idx.tolist(), lp[idx].tolist()))

def full_pipeline(logits, past):
    l = apply_temperature(logits, 0.8)
    l = apply_top_k(l, 50)
    l = apply_top_p(l, 0.9)
    l = apply_repetition_penalty(l, past, 1.1)
    probs = softmax(l)
    return int(np.argmax(probs))

def dequantize_q4_0(data, scales, total):
    group_size = 32
    num_groups = (total + group_size - 1) // group_size
    output = np.empty(total, dtype=np.float32)
    for g in range(num_groups):
        scale = scales[g]
        start = g * group_size
        end = min(start + group_size, total)
        byte_offset = g * (group_size // 2)
        for i in range(start, end):
            local = i - start
            byte_idx = byte_offset + local // 2
            if local % 2 == 0:
                nibble = int(data[byte_idx]) & 0x0F
            else:
                nibble = (int(data[byte_idx]) >> 4) & 0x0F
            output[i] = (nibble - 8.0) * scale
    return output

def dequantize_q4_0_vectorized(data, scales, total):
    group_size = 32
    num_groups = total // group_size
    data_np = np.frombuffer(data, dtype=np.uint8)[:num_groups * 16]
    low = (data_np & 0x0F).astype(np.float32)
    high = ((data_np >> 4) & 0x0F).astype(np.float32)
    unpacked = np.empty(num_groups * group_size, dtype=np.float32)
    unpacked[0::2] = low[:num_groups * 16]
    unpacked[1::2] = high[:num_groups * 16]
    scales_expanded = np.repeat(scales[:num_groups], group_size)
    return (unpacked[:total] - 8.0) * scales_expanded[:total]

# ---------------------------------------------------------------------------
# torch implementations
# ---------------------------------------------------------------------------

def torch_softmax(logits_t):
    return torch.softmax(logits_t, dim=-1)

def torch_log_softmax(logits_t):
    return torch.log_softmax(logits_t, dim=-1)

def torch_top_k(logits_t, k):
    v, idx = torch.topk(logits_t, k)
    out = torch.full_like(logits_t, float('-inf'))
    out.scatter_(-1, idx, v)
    return out

def torch_multinomial(probs_t):
    return torch.multinomial(probs_t, 1)

def torch_temperature(logits_t, temp):
    return logits_t / temp

# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def random_logits(vocab_size, seed=SEED):
    rng = np.random.RandomState(seed)
    return rng.uniform(-10, 10, vocab_size).astype(np.float32)

def bench(name, fn, iterations=100, warmup=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    times.sort()
    median_ns = times[len(times) // 2]
    mean_ns = sum(times) / len(times)
    p5 = times[int(len(times) * 0.05)]
    p95 = times[int(len(times) * 0.95)]
    return {
        "name": name,
        "median_ns": median_ns,
        "mean_ns": int(mean_ns),
        "p5_ns": p5,
        "p95_ns": p95,
        "iterations": iterations,
    }

def format_time(ns):
    if ns < 1_000:
        return f"{ns:.0f} ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:.1f} us"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f} ms"
    else:
        return f"{ns/1_000_000_000:.3f} s"

def main():
    results = []

    # --- Softmax ---
    for vocab in [1000, 32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"softmax/python/{vocab}", lambda: softmax(logits), iterations=200)
        results.append(r)

    # --- Softmax in-place ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"softmax_inplace/python/{vocab}", lambda: softmax_inplace(logits.copy()), iterations=200)
        results.append(r)

    # --- Log softmax ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"log_softmax/python/{vocab}", lambda: log_softmax(logits), iterations=200)
        results.append(r)

    # --- Multinomial ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        probs = softmax(logits)
        rng = np.random.RandomState(123)
        r = bench(f"multinomial/python/{vocab}", lambda: multinomial_sample(probs, rng), iterations=200)
        results.append(r)

    # --- Temperature ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"temperature/python/{vocab}", lambda: apply_temperature(logits, 0.8), iterations=200)
        results.append(r)

    # --- Top-k ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"top_k/python_k50/{vocab}", lambda: apply_top_k(logits, 50), iterations=200)
        results.append(r)

    # --- Top-p ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"top_p/python_p0.9/{vocab}", lambda: apply_top_p(logits, 0.9), iterations=200)
        results.append(r)

    # --- Min-p ---
    logits = random_logits(32000)
    r = bench("min_p/python_32k", lambda: apply_min_p(logits, 0.1), iterations=200)
    results.append(r)

    # --- Repetition penalty ---
    logits = random_logits(32000)
    past = list(range(500))
    r = bench("repetition_penalty/python_32k_500past", lambda: apply_repetition_penalty(logits, past, 1.1), iterations=200)
    results.append(r)

    # --- Repetition penalty large ---
    logits = random_logits(32000)
    past_large = list(range(2000))
    r = bench("repetition_penalty_large/python_32k_2000past", lambda: apply_repetition_penalty(logits, past_large, 1.1), iterations=200)
    results.append(r)

    # --- Combined penalties ---
    logits = random_logits(32000)
    past = list(range(500))
    token_counts = {}
    for t in past:
        token_counts[t] = token_counts.get(t, 0) + 1
    for t in range(100):
        token_counts[t] = token_counts.get(t, 0) + 3
    r = bench("combined_penalties/python_32k_freq_pres_rep", lambda: apply_combined_penalties(logits, past, token_counts, 1.1, 0.5, 0.5), iterations=200)
    results.append(r)

    # --- Full pipeline ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        past = list(range(200))
        r = bench(f"full_pipeline/python/{vocab}", lambda: full_pipeline(logits, past), iterations=200)
        results.append(r)

    # --- Greedy ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"greedy_sample/python/{vocab}", lambda: greedy_sample(logits), iterations=500)
        results.append(r)

    # --- Top logprobs ---
    for vocab in [32000, 128000]:
        logits = random_logits(vocab)
        r = bench(f"top_logprobs/python_top5/{vocab}", lambda: top_logprobs(logits, 5), iterations=200)
        results.append(r)

    # --- Dequant Q4 (loop) ---
    for size in [1_000_000, 10_000_000]:
        num_groups = size // 32
        data = bytes([(i % 256) for i in range(size // 2)])
        scales = np.array([(i + 1) * 0.001 for i in range(num_groups)], dtype=np.float32)
        iters = 5 if size >= 10_000_000 else 20
        r = bench(f"dequant_q4_0/python_loop/{size}", lambda: dequantize_q4_0(data, scales, size), iterations=iters, warmup=2)
        results.append(r)

    # --- Dequant Q4 (vectorized) ---
    for size in [1_000_000, 10_000_000]:
        num_groups = size // 32
        data = bytes([(i % 256) for i in range(size // 2)])
        scales = np.array([(i + 1) * 0.001 for i in range(num_groups)], dtype=np.float32)
        r = bench(f"dequant_q4_0/python_vectorized/{size}", lambda: dequantize_q4_0_vectorized(data, scales, size), iterations=100)
        results.append(r)

    # --- Batch sampling ---
    for batch_size in [8, 32, 64]:
        logits_batch = [random_logits(32000, seed=i) for i in range(batch_size)]
        past_batch = [list(range(100 + i * 10)) for i in range(batch_size)]
        def do_batch(lb=logits_batch, pb=past_batch):
            for l, p in zip(lb, pb):
                full_pipeline(l, p)
        r = bench(f"batch_sampling/python/{batch_size}", do_batch, iterations=50)
        results.append(r)

    # -----------------------------------------------------------------------
    # Torch benchmarks (CPU)
    # -----------------------------------------------------------------------
    if HAS_TORCH:
        print("\nRunning torch (CPU) benchmarks...")

        for vocab in [32000, 128000]:
            logits_t = torch.from_numpy(random_logits(vocab))
            r = bench(f"softmax/torch/{vocab}", lambda: torch_softmax(logits_t), iterations=200)
            results.append(r)

        for vocab in [32000, 128000]:
            logits_t = torch.from_numpy(random_logits(vocab))
            r = bench(f"log_softmax/torch/{vocab}", lambda: torch_log_softmax(logits_t), iterations=200)
            results.append(r)

        for vocab in [32000, 128000]:
            logits_t = torch.from_numpy(random_logits(vocab))
            r = bench(f"top_k/torch_k50/{vocab}", lambda: torch_top_k(logits_t, 50), iterations=200)
            results.append(r)

        for vocab in [32000, 128000]:
            logits_t = torch.from_numpy(random_logits(vocab))
            r = bench(f"temperature/torch/{vocab}", lambda: torch_temperature(logits_t, 0.8), iterations=200)
            results.append(r)

        for vocab in [32000, 128000]:
            logits_t = torch.from_numpy(random_logits(vocab))
            probs_t = torch.softmax(logits_t, dim=-1)
            r = bench(f"multinomial/torch/{vocab}", lambda: torch_multinomial(probs_t), iterations=200)
            results.append(r)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PYTHON BENCHMARK RESULTS")
    print("=" * 80)
    for r in results:
        print(f"  {r['name']:55s}  median: {format_time(r['median_ns']):>12s}  mean: {format_time(r['mean_ns']):>12s}")

    # Save JSON
    out_path = "/Users/andy/hermes-lite/vllm-rs/benches/python_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
