"""MiniMax-M2.7-NVFP4 full benchmark on TPU v6e-8.

Produces:
  1. Batch sweep tok/s at B=1, 8, 16, 32, 64, 128 (decode-only, fresh empty KV)
  2. Perplexity on a fixed text sample
  3. Coherent 2048-token generation from a user prompt

Reuses load_model_stacked from m2_real_bench.py.

Usage:
    python3 m2_full_bench.py --model-dir /dev/shm/m2-nvfp4 \\
        --batches 1,8,16,32,64,128 --ctx 2048 \\
        --prompt "Explain angular momentum." --gen-tokens 2048 \\
        --ppl-text <path-to-text> --out /tmp/m2_full.json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_synth_bench import (
    NL_FULL, NH, NKV, HEAD_DIM, H, MOE_INTER, NUM_EXPERTS, TOP_K,
    VOCAB, ROPE_THETA, ROTARY_DIM, DTYPE,
    make_mesh_v6e8, precompute_rope, forward_step,
)
from m2_real_bench import load_model_stacked


def load_tokenizer(model_dir):
    """Prefer HF tokenizers library (pure Python), fast enough for our needs."""
    from tokenizers import Tokenizer
    tk_path = Path(model_dir) / "tokenizer.json"
    if not tk_path.exists():
        raise FileNotFoundError(f"tokenizer.json missing at {tk_path}")
    tk = Tokenizer.from_file(str(tk_path))
    return tk


def decode_tokens(tk, ids):
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    return tk.decode(list(map(int, ids)))


def encode_text(tk, text):
    enc = tk.encode(text)
    return np.array(enc.ids, dtype=np.int32)


def make_batched_empty_cache(mesh, nl, B, ctx, dtype):
    from jax.sharding import NamedSharding, PartitionSpec as P
    rep = NamedSharding(mesh, P())
    k = jax.device_put(jnp.zeros((nl, B, ctx, NKV, HEAD_DIM), dtype=dtype), rep)
    v = jax.device_put(jnp.zeros((nl, B, ctx, NKV, HEAD_DIM), dtype=dtype), rep)
    return k, v


def bench_batch(mesh, model_state, B, ctx, iters, warmup):
    """Run decode at batch B, timing steady-state ms/step and tok/s."""
    (embed, final_norm, lm_head, stacked, _k_ignored, _v_ignored, cos, sin) = model_state

    # Fresh KV for this batch size.
    k_cache, v_cache = make_batched_empty_cache(mesh, NL_FULL, B, ctx, DTYPE)

    def _forward(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        return forward_step(x, stacked, k_cache, v_cache, pos, cos, sin,
                          final_norm, lm_head, mesh)
    forward_jit = jax.jit(_forward)

    # Seed with zero token.
    tok = jnp.zeros((B,), dtype=jnp.int32)
    x = embed[tok]

    # Warmup (triggers compile for this B).
    print(f"  B={B} warmup ({warmup} iters)...", file=sys.stderr)
    for it in range(warmup):
        tok, k_cache, v_cache = forward_jit(x, stacked, k_cache, v_cache,
                                            jnp.int32(it), cos, sin,
                                            final_norm, lm_head)
        x = embed[tok]
    jax.block_until_ready(tok)

    print(f"  B={B} measure ({iters} iters)...", file=sys.stderr)
    times = []
    for it in range(iters):
        t0 = time.perf_counter()
        tok, k_cache, v_cache = forward_jit(x, stacked, k_cache, v_cache,
                                            jnp.int32(warmup + it), cos, sin,
                                            final_norm, lm_head)
        jax.block_until_ready(tok)
        times.append((time.perf_counter() - t0) * 1000)
        x = embed[tok]
    times = np.array(times)
    return {
        'batch': B, 'ctx': ctx, 'iters': iters,
        'ms_min': float(times.min()),
        'ms_mean': float(times.mean()),
        'ms_max': float(times.max()),
        'ms_p50': float(np.median(times)),
        'tok_per_s': float(1000.0 * B / times.mean()),
    }


def generate_sample(mesh, model_state, tk, prompt, gen_tokens, ctx):
    """Generate `gen_tokens` tokens from prompt. Returns decoded text + ms/tok."""
    (embed, final_norm, lm_head, stacked, _, _, cos, sin) = model_state

    prompt_ids = encode_text(tk, prompt)
    print(f"  prompt: {len(prompt_ids)} tokens", file=sys.stderr)

    # Fresh KV for B=1.
    k_cache, v_cache = make_batched_empty_cache(mesh, NL_FULL, 1, ctx, DTYPE)

    def _forward(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        return forward_step(x, stacked, k_cache, v_cache, pos, cos, sin,
                          final_norm, lm_head, mesh)
    forward_jit = jax.jit(_forward)

    # Prefill: feed prompt tokens one at a time (ideally use prefill kernel — B=1 is fine serial for 20 tokens)
    out_ids = list(prompt_ids)
    for i, tid in enumerate(prompt_ids):
        x = embed[jnp.array([int(tid)], dtype=jnp.int32)]
        tok, k_cache, v_cache = forward_jit(
            x, stacked, k_cache, v_cache, jnp.int32(i), cos, sin,
            final_norm, lm_head)
    jax.block_until_ready(tok)
    # Append the very-first generated token
    out_ids.append(int(tok[0]))

    # Decode loop
    print(f"  generating {gen_tokens} tokens...", file=sys.stderr)
    t0 = time.perf_counter()
    for i in range(1, gen_tokens):
        x = embed[tok]
        tok, k_cache, v_cache = forward_jit(
            x, stacked, k_cache, v_cache,
            jnp.int32(len(prompt_ids) + i), cos, sin, final_norm, lm_head)
        out_ids.append(int(tok[0]))
    jax.block_until_ready(tok)
    dt = time.perf_counter() - t0
    gen_ms_per_tok = dt * 1000 / gen_tokens
    gen_tok_s = gen_tokens / dt

    text = decode_tokens(tk, out_ids)
    return {
        'prompt_len': len(prompt_ids),
        'gen_tokens': gen_tokens,
        'ms_per_tok': gen_ms_per_tok,
        'tok_per_s': gen_tok_s,
        'text': text,
        'prompt': prompt,
    }


def compute_ppl(mesh, model_state, tk, text, ctx):
    """Compute perplexity on `text`. Uses non-autoregressive teacher-forced pass
    via the same forward_step (one token at a time decode) for simplicity."""
    (embed, final_norm, lm_head, stacked, _, _, cos, sin) = model_state

    ids = encode_text(tk, text)
    if len(ids) < 2:
        return {'skipped': 'text too short'}
    if len(ids) > ctx:
        ids = ids[:ctx]
    n = len(ids)

    k_cache, v_cache = make_batched_empty_cache(mesh, NL_FULL, 1, ctx, DTYPE)

    def _forward_logits(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        from m2_synth_bench import rms_norm
        # Replicate forward_step internals but return logits, not argmax.
        def layer_body(carry, layer_w):
            x, k_cache, v_cache, i = carry
            k_i = jax.lax.dynamic_index_in_dim(k_cache, i, axis=0, keepdims=False)
            v_i = jax.lax.dynamic_index_in_dim(v_cache, i, axis=0, keepdims=False)
            from m2_synth_bench import attn_layer, rms_norm as rn
            from m2_moe import moe_block_nvfp4
            att_out, k_i_new, v_i_new = attn_layer(
                x, layer_w['attn_q'], layer_w['attn_k'], layer_w['attn_v'], layer_w['attn_o'],
                layer_w['attn_qn'], layer_w['attn_kn'], layer_w['ln1'],
                k_i, v_i, pos, cos, sin)
            x = x + att_out
            h = rn(x, layer_w['ln2'])
            moe_out = moe_block_nvfp4(
                h, layer_w['rg'], layer_w['rb'],
                (layer_w['w1_p'], layer_w['w1_s'], layer_w['w1_s2']),
                (layer_w['w2_p'], layer_w['w2_s'], layer_w['w2_s2']),
                (layer_w['w3_p'], layer_w['w3_s'], layer_w['w3_s2']),
                mesh,
            )
            x = x + moe_out
            k_cache = jax.lax.dynamic_update_index_in_dim(k_cache, k_i_new, i, axis=0)
            v_cache = jax.lax.dynamic_update_index_in_dim(v_cache, v_i_new, i, axis=0)
            return (x, k_cache, v_cache, i + 1), None
        (x, k_cache, v_cache, _), _ = jax.lax.scan(
            layer_body, (x, k_cache, v_cache, jnp.int32(0)), stacked)
        h = rms_norm(x, final_norm)
        logits = h @ lm_head.T
        return logits, k_cache, v_cache

    forward_jit = jax.jit(_forward_logits)

    nll_sum = 0.0
    n_scored = 0
    for i in range(n - 1):
        x = embed[jnp.array([int(ids[i])], dtype=jnp.int32)]
        logits, k_cache, v_cache = forward_jit(
            x, stacked, k_cache, v_cache,
            jnp.int32(i), cos, sin, final_norm, lm_head)
        logp = jax.nn.log_softmax(logits[0].astype(jnp.float32), axis=-1)
        nll = -float(logp[int(ids[i + 1])])
        nll_sum += nll
        n_scored += 1
    ppl = float(np.exp(nll_sum / n_scored))
    return {'n_tokens_scored': n_scored, 'avg_nll': nll_sum / n_scored, 'ppl': ppl}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--batches', default='1,8,16,32,64,128',
                   help='comma-separated batch sizes to sweep')
    p.add_argument('--ctx', type=int, default=2048)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--prompt', default='Explain angular momentum.')
    p.add_argument('--gen-tokens', type=int, default=2048)
    p.add_argument('--ppl-text', default=None, help='path to a text file for PPL')
    p.add_argument('--skip-sweep', action='store_true')
    p.add_argument('--skip-gen', action='store_true')
    p.add_argument('--skip-ppl', action='store_true')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    mesh = make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)

    t0 = time.time()
    model_state = load_model_stacked(
        args.model_dir, mesh, args.ctx, B=1, n_workers=args.workers)
    load_s = time.time() - t0
    print(f">> load: {load_s:.1f}s", file=sys.stderr)

    tk = load_tokenizer(args.model_dir)
    print(f">> tokenizer: vocab {tk.get_vocab_size()}", file=sys.stderr)

    out = {
        'arch': 'MiniMax-M2.7-NVFP4',
        'slice': 'v6e-8',
        'nl': NL_FULL,
        'ctx': args.ctx,
        'load_seconds': load_s,
        'sweep': None,
        'ppl': None,
        'generation': None,
    }

    # 1. Batch sweep
    if not args.skip_sweep:
        print("\n### Batch sweep ###", file=sys.stderr)
        sweep_results = []
        for B in [int(b) for b in args.batches.split(',')]:
            try:
                r = bench_batch(mesh, model_state, B, args.ctx, args.iters, args.warmup)
                print(f"  B={B:4d} ms/step={r['ms_mean']:.2f} tok/s={r['tok_per_s']:.1f}")
                sweep_results.append(r)
            except Exception as e:
                print(f"  B={B} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                sweep_results.append({'batch': B, 'error': str(e)})
        out['sweep'] = sweep_results

    # 2. Perplexity
    if not args.skip_ppl and args.ppl_text:
        print("\n### Perplexity ###", file=sys.stderr)
        text = Path(args.ppl_text).read_text()
        r = compute_ppl(mesh, model_state, tk, text, args.ctx)
        print(f"  PPL: {r}")
        out['ppl'] = r

    # 3. Coherent generation
    if not args.skip_gen:
        print(f"\n### Generation ({args.gen_tokens} tokens) ###", file=sys.stderr)
        r = generate_sample(mesh, model_state, tk, args.prompt, args.gen_tokens, args.ctx)
        print(f"  prompt_len={r['prompt_len']}  gen={r['gen_tokens']}  tok/s={r['tok_per_s']:.1f}")
        print(f"\n--- generated text ---\n{r['text']}\n--- end ---")
        out['generation'] = r

    out_path = args.out or f"/tmp/m2_full_bench_{int(time.time())}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
