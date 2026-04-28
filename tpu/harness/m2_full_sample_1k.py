"""Run a real MiniMax-M2.7 full-model chat sample on TPU.

This uses the complete JAX M2 path from /workspace/runs/m2bench/tpu/harness:
attention, RoPE, full NVFP4 MoE, final norm, and lm_head.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HARNESS = os.environ.get("M2_BENCH_HARNESS", "/workspace/runs/m2bench/tpu/harness")
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from m2_full_bench import decode_tokens, load_model_stacked, load_tokenizer, make_batched_empty_cache
from m2_synth_bench import DTYPE, NL_FULL, forward_logits, make_mesh_v6e8

EOS_ID = 200020
DEFAULT_SYSTEM = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."


def render_chat(user, system=DEFAULT_SYSTEM, think=True, empty_think=False, assistant_prefix=""):
    if empty_think:
        suffix = "<think>\n</think>\n\n"
    else:
        suffix = "<think>\n" if think else ""
    return (
        "]~!b[]~b]system\n"
        + system
        + "[e~[\n]~b]user\n"
        + user
        + "[e~[\n]~b]ai\n"
        + suffix
        + assistant_prefix
    )


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def apply_decode_controls(logits, generated, banned, repetition_penalty, recent_window):
    logits = logits.astype(np.float32, copy=True)
    for token_id in banned:
        if 0 <= token_id < logits.shape[0]:
            logits[token_id] = -np.inf
    if repetition_penalty and repetition_penalty > 1.0:
        for token_id in set(generated[-recent_window:]):
            if 0 <= token_id < logits.shape[0]:
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
    return logits


def sample_top_p(logits, rng, temperature, top_p, top_k):
    logits = logits.astype(np.float32) / max(temperature, 1e-6)
    if top_k and 0 < top_k < logits.shape[0]:
        idx = np.argpartition(-logits, top_k)[:top_k]
        vals = logits[idx]
        order = np.argsort(-vals)
        idx = idx[order]
        vals = vals[order]
    else:
        idx = np.argsort(-logits)
        vals = logits[idx]
    probs = softmax(vals)
    cdf = np.cumsum(probs)
    last = np.searchsorted(cdf, top_p, side="left")
    keep = np.zeros_like(probs, dtype=bool)
    keep[: max(1, last + 1)] = True
    idx = idx[keep]
    probs = probs[keep]
    probs = probs / probs.sum()
    return int(rng.choice(idx, p=probs))


def clean_visible(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>\n</think>", "").strip()


def should_stop_sentence(text, min_tokens, generated_len):
    if min_tokens <= 0 or generated_len < min_tokens:
        return False
    stripped = text.rstrip()
    if not stripped.endswith((".", "!", "?")):
        return False
    return len(stripped.split()) >= 45


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--direct-answer", action="store_true")
    p.add_argument("--empty-think", action="store_true")
    p.add_argument("--suppress-reasoning", action="store_true")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--recent-window", type=int, default=256)
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--stop-after-sentence-min-tokens", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    rng = np.random.default_rng(args.seed)
    mesh = make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)

    t_load = time.perf_counter()
    model_state = load_model_stacked(args.model_dir, mesh, args.ctx, B=1, n_workers=args.workers)
    load_s = time.perf_counter() - t_load
    print(f"loaded model in {load_s:.2f}s", file=sys.stderr)

    embed, final_norm, lm_head, stacked, _, _, cos, sin = model_state
    tokenizer = load_tokenizer(args.model_dir)
    prompt_text = render_chat(
        args.prompt,
        think=not args.direct_answer,
        empty_think=args.empty_think,
        assistant_prefix=args.assistant_prefix,
    )
    prompt_ids = tokenizer.encode(prompt_text).ids
    if len(prompt_ids) >= args.ctx:
        raise SystemExit(f"prompt too long: {len(prompt_ids)} >= ctx {args.ctx}")
    print(f"prompt_tokens={len(prompt_ids)} max_new={args.max_new_tokens}", file=sys.stderr)

    k_cache, v_cache = make_batched_empty_cache(mesh, NL_FULL, 1, args.ctx, DTYPE)

    def run_logits(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        return forward_logits(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head, mesh)

    fwd = jax.jit(run_logits)

    logits = None
    t_prefill = time.perf_counter()
    for pos, token_id in enumerate(prompt_ids):
        x = embed[jnp.array([int(token_id)], dtype=jnp.int32)]
        logits, k_cache, v_cache = fwd(
            x, stacked, k_cache, v_cache, jnp.int32(pos), cos, sin, final_norm, lm_head
        )
    jax.block_until_ready(logits)
    prefill_s = time.perf_counter() - t_prefill
    print(f"prefill_s={prefill_s:.2f}", file=sys.stderr)

    generated = []
    token_times_ms = []
    stopped_sentence = False
    banned = {200050, 200051} if args.suppress_reasoning else set()
    first_logits = apply_decode_controls(
        np.asarray(logits[0]), generated, banned, args.repetition_penalty, args.recent_window
    )
    next_id = sample_top_p(first_logits, rng, args.temperature, args.top_p, args.top_k)
    for i in range(args.max_new_tokens):
        generated.append(next_id)
        if next_id == EOS_ID:
            print(f"eos at generated token {i + 1}", file=sys.stderr)
            break
        current_visible = clean_visible(args.assistant_prefix + decode_tokens(tokenizer, generated))
        if should_stop_sentence(
            current_visible, args.stop_after_sentence_min_tokens, len(generated)
        ):
            stopped_sentence = True
            print(f"sentence stop at generated token {i + 1}", file=sys.stderr)
            break
        if i + 1 == args.max_new_tokens:
            break
        x = embed[jnp.array([next_id], dtype=jnp.int32)]
        t0 = time.perf_counter()
        logits, k_cache, v_cache = fwd(
            x,
            stacked,
            k_cache,
            v_cache,
            jnp.int32(len(prompt_ids) + i),
            cos,
            sin,
            final_norm,
            lm_head,
        )
        jax.block_until_ready(logits)
        token_times_ms.append((time.perf_counter() - t0) * 1000.0)
        controlled_logits = apply_decode_controls(
            np.asarray(logits[0]), generated, banned, args.repetition_penalty, args.recent_window
        )
        next_id = sample_top_p(controlled_logits, rng, args.temperature, args.top_p, args.top_k)
        if (i + 1) % 50 == 0:
            print(f"generated={i + 1}", file=sys.stderr)

    completion = decode_tokens(tokenizer, generated)
    visible = clean_visible(completion)
    full_text = decode_tokens(tokenizer, prompt_ids + generated)
    mean_ms = float(np.mean(token_times_ms)) if token_times_ms else None
    result = {
        "schema": "rvllm.m2.full_model_1k_sample.v1",
        "arch": "MiniMax-M2.7-NVFP4",
        "runtime": "jax_full_model",
        "layers": NL_FULL,
        "kv_cache": os.environ.get("RVLLM_M2_KV", "bf16"),
        "moe_impl": os.environ.get("M2_MOE", "shardmap"),
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": len(generated),
        "stopped_eos": bool(generated and generated[-1] == EOS_ID),
        "stopped_sentence": stopped_sentence,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "direct_answer": args.direct_answer,
        "empty_think": args.empty_think,
        "suppress_reasoning": args.suppress_reasoning,
        "repetition_penalty": args.repetition_penalty,
        "assistant_prefix": args.assistant_prefix,
        "seed": args.seed,
        "load_seconds": load_s,
        "prefill_seconds": prefill_s,
        "mean_ms_per_generated_token": mean_ms,
        "tok_per_s": float(1000.0 / mean_ms) if mean_ms else None,
        "generated_token_ids": generated,
        "completion_text": completion,
        "visible_answer": clean_visible(args.assistant_prefix + completion),
        "full_text": full_text,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    out.with_suffix(".txt").write_text(visible + "\n")
    print(
        json.dumps(
            {
                "generated_tokens": result["generated_tokens"],
                "stopped_eos": result["stopped_eos"],
                "mean_ms_per_generated_token": result["mean_ms_per_generated_token"],
                "tok_per_s": result["tok_per_s"],
            },
            indent=2,
        )
    )
    print("\n--- visible answer ---\n" + visible[:5000] + "\n--- end ---")


if __name__ == "__main__":
    main()
