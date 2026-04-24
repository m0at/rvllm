"""Benchmark + perplexity harness for MiniMax-M2.7-NVFP4 on TPU v6e-8.

Throughput mode: warmup + timed iterations, reports min/mean/max ms/step + tok/s.
Perplexity mode: chunked 2048-token sliding windows over a text file,
computes NLL per token and reports perplexity.

Writes results JSON to tpu/out/m2_bench_B<batch>_C<ctx>.json.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))

from m2_tpu_infer import (
    load_config_m2,
    load_model_m2,
    forward_step_m2,
)
from m2_attention import precompute_rope_m2
from m2_kv_cache import make_kv_caches
from m2_mesh import make_mesh_v6e8


OUT_DIR = "/Users/andy/rvllm/tpu/out"


def _stats_ms(timings_s):
    ms = [t * 1000.0 for t in timings_s]
    return {
        "min_ms": round(min(ms), 4),
        "mean_ms": round(sum(ms) / len(ms), 4),
        "max_ms": round(max(ms), 4),
    }


def run_throughput(args, mesh, state):
    batch = args.batch
    ctx = args.ctx
    iters = args.iters
    warmup = args.warmup

    print(f"warmup={warmup} iters={iters} batch={batch} ctx={ctx}", file=sys.stderr)

    dummy_tok = jnp.zeros((batch,), dtype=jnp.int32)
    pos = jnp.int32(0)
    state['token_ids'] = dummy_tok
    state['pos'] = pos
    state['ctx'] = jnp.int32(1)

    for i in range(warmup):
        t0 = time.perf_counter()
        out = forward_step_m2(state, dummy_tok, pos, mode="token")
        try:
            out.block_until_ready()
        except AttributeError:
            pass
        dt = time.perf_counter() - t0
        print(f"warmup {i+1}/{warmup}: {dt*1000:.3f} ms", file=sys.stderr)

    timings = []
    t_all = time.perf_counter()
    for i in range(iters):
        t0 = time.perf_counter()
        out = forward_step_m2(state, dummy_tok, pos, mode="token")
        try:
            out.block_until_ready()
        except AttributeError:
            pass
        dt = time.perf_counter() - t0
        timings.append(dt)
        print(f"iter {i+1}/{iters}: {dt*1000:.3f} ms", file=sys.stderr)
    total_elapsed = time.perf_counter() - t_all

    stats = _stats_ms(timings)
    mean_s = stats["mean_ms"] / 1000.0
    tok_s = (batch / mean_s) if mean_s > 0 else 0.0

    result = {
        "mode": "throughput",
        "model_dir": args.model_dir,
        "batch": batch,
        "ctx": ctx,
        "iters": iters,
        "warmup": warmup,
        "min_ms": stats["min_ms"],
        "mean_ms": stats["mean_ms"],
        "max_ms": stats["max_ms"],
        "tok_s": round(tok_s, 3),
        "total_elapsed_s": round(total_elapsed, 3),
    }
    print(json.dumps(result))
    return result


def run_perplexity(args, mesh, state):
    if not args.ppl_file:
        print("--ppl requires --ppl-file", file=sys.stderr)
        sys.exit(2)

    with open(args.ppl_file, "r") as f:
        text = f.read()

    tokenize = state["tokenize"] if isinstance(state, dict) and "tokenize" in state else None
    if tokenize is None:
        from m2_chat import load_tokenizer_m2
        tok = load_tokenizer_m2(args.model_dir)
        tokenize = tok.encode

    all_ids = tokenize(text)
    if not isinstance(all_ids, list):
        all_ids = list(all_ids)
    print(f"total tokens: {len(all_ids)}", file=sys.stderr)

    window = 2048
    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    num_chunks = max(1, len(all_ids) // window)
    for ci in range(num_chunks):
        start = ci * window
        end = start + window
        chunk = all_ids[start:end]
        if len(chunk) < 2:
            continue
        positions = list(range(len(chunk)))
        nll, n = forward_step_m2(
            state,
            token_ids=chunk,
            positions=positions,
            mode="nll",
        )
        total_nll += float(nll)
        total_tokens += int(n)
        running_ppl = math.exp(total_nll / total_tokens) if total_tokens else 0.0
        dt = time.perf_counter() - t0
        print(
            f"chunk {ci+1}/{num_chunks}: running_ppl={running_ppl:.4f} "
            f"({total_tokens/dt:.1f} tok/s)",
            file=sys.stderr,
        )

    ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
    elapsed = time.perf_counter() - t0
    result = {
        "mode": "perplexity",
        "model_dir": args.model_dir,
        "ppl_file": args.ppl_file,
        "window": window,
        "perplexity": round(ppl, 4),
        "tokens": total_tokens,
        "elapsed_s": round(elapsed, 3),
    }
    print(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ppl", action="store_true")
    ap.add_argument("--ppl-file", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading config...", file=sys.stderr)
    load_config_m2(args.model_dir)

    print("building mesh...", file=sys.stderr)
    mesh = make_mesh_v6e8()
    if isinstance(mesh, tuple):
        mesh = mesh[0]

    print("loading model...", file=sys.stderr)
    model = load_model_m2(args.model_dir, mesh, args.ctx)

    print("building kv caches + rope...", file=sys.stderr)
    caches = make_kv_caches(args.batch, args.ctx, mesh)
    from m2_tpu_infer import ROPE_THETA, ROTARY_DIM
    cos, sin = precompute_rope_m2(ROPE_THETA, ROTARY_DIM, args.ctx)
    cos = jnp.asarray(cos)
    sin = jnp.asarray(sin)

    state = {
        'model': model,
        'caches': caches,
        'cos': cos,
        'sin': sin,
        'mesh': mesh,
    }

    if args.ppl:
        result = run_perplexity(args, mesh, state)
    else:
        result = run_throughput(args, mesh, state)

    out_path = os.path.join(
        OUT_DIR, f"m2_bench_B{args.batch}_C{args.ctx}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
