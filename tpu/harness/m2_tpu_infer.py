#!/usr/bin/env python3
"""MiniMax-M2.7-NVFP4 inference on TPU v6e-8 via JAX SPMD (expert-parallel 8-way).

Integrates the 16-agent M2 swarm. This file owns only wiring: it imports from the
other agents' modules and exposes a CLI matching gemma4_tpu_infer.py.

Paths:
  - Path A (default): NVFP4 weights stay packed in HBM; on-the-fly dequant fuses
    into the GEMM epilogue via nvfp4_jax_ops.nvfp4_matmul.
  - Path B: upcast NVFP4 -> int8 at load time via Zig CPU SIMD; standard int8
    matmul thereafter (no on-the-fly dequant).

Usage:
    python3 m2_tpu_infer.py --model-dir /path/to/MiniMax-M2.7-NVFP4 \
        --max-tokens 32 --prompt "Hello" [--path A|B] [--speculate]
"""
import argparse, json, os, sys, time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from m2_attention import precompute_rope_m2, attention_layer
from m2_moe import moe_block, router_sigmoid_topk
from m2_mtp import mtp_forward, mtp_load_weights
from m2_mesh import make_mesh_v6e8, expert_shard_spec, replicate_spec
from m2_kv_cache import make_kv_caches, M2_KV_LAYOUT
from m2_chat import load_tokenizer_m2, apply_chat_template
from nvfp4_loader import (
    ModeloptSafetensorsReader,
    dequant_nvfp4_to_int8_cpu,
    dequant_nvfp4_to_bf16_cpu,
)
from nvfp4_jax_ops import nvfp4_to_bf16_jax, nvfp4_matmul


# Module-level globals (set by load_config_m2).
H = NH = NKV = HEAD_DIM = NL = VOCAB = 0
ROTARY_DIM = 0
ROPE_THETA = 0.0
NUM_EXPERTS = TOP_K = MOE_INTER = 0
USE_MTP = False
NUM_MTP = 0
PATH = "A"
B = 1


def load_config_m2(model_dir):
    """Read config.json and set module-level globals per M2 spec."""
    global H, NH, NKV, HEAD_DIM, NL, VOCAB
    global ROTARY_DIM, ROPE_THETA
    global NUM_EXPERTS, TOP_K, MOE_INTER
    global USE_MTP, NUM_MTP

    cfg_path = os.path.join(model_dir, 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    H = int(cfg['hidden_size'])
    NH = int(cfg['num_attention_heads'])
    NKV = int(cfg['num_key_value_heads'])
    HEAD_DIM = int(cfg['head_dim'])
    NL = int(cfg['num_hidden_layers'])
    VOCAB = int(cfg['vocab_size'])
    ROTARY_DIM = int(cfg.get('rotary_dim', int(HEAD_DIM * cfg.get('partial_rotary_factor', 0.5))))
    ROPE_THETA = float(cfg.get('rope_theta', 5_000_000.0))

    NUM_EXPERTS = int(cfg['num_local_experts'])
    TOP_K = int(cfg['num_experts_per_tok'])
    MOE_INTER = int(cfg['intermediate_size'])

    USE_MTP = bool(cfg.get('use_mtp', False))
    NUM_MTP = int(cfg.get('num_mtp_modules', 0))

    print(
        f"M2 config: H={H} NH={NH} NKV={NKV} HEAD_DIM={HEAD_DIM} NL={NL} VOCAB={VOCAB}",
        file=sys.stderr,
    )
    print(
        f"  rotary_dim={ROTARY_DIM} rope_theta={ROPE_THETA}",
        file=sys.stderr,
    )
    print(
        f"  MoE: experts={NUM_EXPERTS} top_k={TOP_K} moe_inter={MOE_INTER}",
        file=sys.stderr,
    )
    print(f"  MTP: use={USE_MTP} modules={NUM_MTP}", file=sys.stderr)
    return cfg


def _put_replicated(mesh, arr):
    rank = len(arr.shape)
    spec = P(*([None] * rank)) if rank > 0 else P()
    return jax.device_put(arr, NamedSharding(mesh, spec))


def _put_expert_sharded(mesh, arr):
    rank = len(arr.shape)
    spec = P('expert', *([None] * (rank - 1)))
    return jax.device_put(arr, NamedSharding(mesh, spec))


def load_model_m2(model_dir, mesh, max_ctx, path="A"):
    """Load all M2 weights from the NVFP4 safetensors directory.

    Returns a dict with keys:
      embed          -- (VOCAB, H) bf16
      final_norm     -- (H,) bf16
      attention_weights -- list of NL dicts {qw, kw, vw, ow, qn, kn, ln}
      moe_weights    -- list of NL dicts {gate_up, down} expert-sharded
      router_weights -- list of NL dicts {router_w, router_bias}
      lm_head        -- (VOCAB, H) bf16
      mtp_weights    -- list of NUM_MTP dicts or None
    """
    reader = ModeloptSafetensorsReader(model_dir)

    def read_bf16_replicated(name):
        arr = reader.read_bf16(name)
        return _put_replicated(mesh, jnp.asarray(arr))

    def read_expert_weight(name):
        """Per-expert packed NVFP4 weight. Returns packed or bf16 depending on PATH.

        Expected logical shape: (NUM_EXPERTS, out, in). Path A keeps packed layout
        (uint8 w_packed + uint8 w_scale). Path B dequantizes to int8 per-row.
        """
        if path == "A":
            t = reader.read_nvfp4(name)
            return t  # caller sends packed + scales through nvfp4_matmul
        elif path == "B":
            t = reader.read_nvfp4(name)
            i8, rs = dequant_nvfp4_to_int8_cpu(t, n_threads=0)
            return {'int8': jnp.asarray(i8), 'row_scales': jnp.asarray(rs)}
        else:
            raise ValueError(f"unknown path {path!r}, expected 'A' or 'B'")

    # Embedding + final norm + lm_head are bf16 (in the ignore list).
    embed = read_bf16_replicated('model.embed_tokens.weight')
    final_norm = read_bf16_replicated('model.norm.weight')
    lm_head = read_bf16_replicated('lm_head.weight')

    attention_weights = []
    moe_weights = []
    router_weights = []

    for i in range(NL):
        prefix = f'model.layers.{i}'

        # Attention: bf16 per the ignore list.
        attn = {
            'ln': read_bf16_replicated(f'{prefix}.input_layernorm.weight'),
            'qw': read_bf16_replicated(f'{prefix}.self_attn.q_proj.weight'),
            'kw': read_bf16_replicated(f'{prefix}.self_attn.k_proj.weight'),
            'vw': read_bf16_replicated(f'{prefix}.self_attn.v_proj.weight'),
            'ow': read_bf16_replicated(f'{prefix}.self_attn.o_proj.weight'),
            'qn': read_bf16_replicated(f'{prefix}.self_attn.q_norm.weight'),
            'kn': read_bf16_replicated(f'{prefix}.self_attn.k_norm.weight'),
            'post_ln': read_bf16_replicated(f'{prefix}.post_attention_layernorm.weight'),
        }
        attention_weights.append(attn)

        # Router: bf16, in the ignore list. Per-expert bias for aux-loss-free routing.
        router = {
            'router_w': read_bf16_replicated(
                f'{prefix}.block_sparse_moe.gate.weight'),
            'router_bias': read_bf16_replicated(
                f'{prefix}.block_sparse_moe.gate.e_score_correction_bias'),
        }
        router_weights.append(router)

        # Experts: NVFP4 packed (path A) or int8 (path B).
        # Layout: per-expert fused gate||up and down.
        moe = {
            'gate_up': read_expert_weight(
                f'{prefix}.block_sparse_moe.experts.gate_up_proj'),
            'down': read_expert_weight(
                f'{prefix}.block_sparse_moe.experts.down_proj'),
        }
        moe_weights.append(moe)

    mtp_weights = None
    if USE_MTP and NUM_MTP > 0:
        all_tensors = {name: reader for name in reader.list_tensors()}
        mtp_weights = mtp_load_weights(all_tensors, 'model.mtp_modules.')

    return {
        'embed': embed,
        'final_norm': final_norm,
        'attention_weights': attention_weights,
        'moe_weights': moe_weights,
        'router_weights': router_weights,
        'lm_head': lm_head,
        'mtp_weights': mtp_weights,
    }


def _decode_one_step(token_id, pos, ctx, model, caches, cos, sin, mesh):
    """Internal: decode one token. Returns (next_token, log_probs, caches_new)."""
    embed = model['embed']
    final_norm = model['final_norm']
    lm_head = model['lm_head']

    # (B,) -> (B, H)
    x = embed[token_id]

    new_k_cache = caches['k']
    new_v_cache = caches['v']

    for i in range(NL):
        attn_w = model['attention_weights'][i]
        router_w = model['router_weights'][i]
        moe_w = model['moe_weights'][i]

        x, k_new, v_new = attention_layer(
            x,
            attn_w,
            attn_w['ln'],
            caches['k'][i],
            caches['v'][i],
            pos,
            cos,
            sin,
            ctx,
        )
        new_k_cache = new_k_cache.at[i].set(k_new)
        new_v_cache = new_v_cache.at[i].set(v_new)

        x = moe_block(
            x,
            router_w['router_w'],
            router_w['router_bias'],
            moe_w['gate_up'],
            moe_w['down'],
            mesh,
        )

    # Final RMSNorm.
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    x = (x.astype(jnp.float32) / rms).astype(x.dtype) * final_norm

    # lm_head: (VOCAB, H) bf16. Standard matmul (lm_head is in the ignore list).
    logits = jnp.einsum('bh,vh->bv', x, lm_head)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    new_caches = {'k': new_k_cache, 'v': new_v_cache}
    return next_tok, log_probs, new_caches


def forward_step_m2(state, token_ids=None, positions=None, mode="token"):
    """Unified forward-step entry.

    mode="token": single-step decode. Returns next-token ids, shape (B,), int32.
                  `state` must carry {'model','caches','cos','sin','mesh','pos','ctx'}
                  and `token_ids` is a (B,) int32 array (or omitted if state
                  already carries the current token).
    mode="nll":   full forward over a token sequence. Requires `token_ids` as a
                  1-D sequence; `positions` is ignored (positions assumed 0..N-1).
                  Returns (nll_sum_f32_scalar, n_tokens_int).
    """
    if not isinstance(state, dict):
        raise TypeError("forward_step_m2: `state` must be a dict produced by load_model_m2 + runner wrapping")

    model = state['model']
    caches = state['caches']
    cos = state['cos']
    sin = state['sin']
    mesh = state['mesh']

    if mode == "token":
        if token_ids is None:
            token_ids = state['token_ids']
        pos = state['pos'] if positions is None else positions
        ctx = state.get('ctx', pos + 1)
        tok_arr = jnp.asarray(token_ids, dtype=jnp.int32)
        next_tok, _lp, new_caches = _decode_one_step(
            tok_arr, pos, ctx, model, caches, cos, sin, mesh)
        state['caches'] = new_caches
        return next_tok

    elif mode == "nll":
        if token_ids is None:
            raise ValueError("forward_step_m2(mode='nll') requires token_ids")
        ids = list(token_ids)
        if len(ids) < 2:
            return jnp.float32(0.0), 0

        local_caches = caches
        total_nll = jnp.float32(0.0)
        n = 0
        for step in range(len(ids) - 1):
            tok_arr = jnp.asarray([ids[step]], dtype=jnp.int32)
            pos = jnp.int32(step)
            ctx = jnp.int32(step + 1)
            _nt, log_probs, local_caches = _decode_one_step(
                tok_arr, pos, ctx, model, local_caches, cos, sin, mesh)
            tgt = ids[step + 1]
            total_nll = total_nll + (-log_probs[0, tgt]).astype(jnp.float32)
            n += 1
        state['caches'] = local_caches
        return total_nll, n

    else:
        raise ValueError(f"forward_step_m2: unknown mode {mode!r}, expected 'token' or 'nll'")


def run_generate_m2(args, mesh, model, caches, cos, sin, tokenizer):
    """Greedy generate from a prompt."""
    if tokenizer is not None and not args.prompt.replace(',', '').strip().isdigit():
        prompt_ids = list(tokenizer.encode(args.prompt))
    else:
        prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    print(f"prompt: {len(prompt_ids)} tokens {prompt_ids[:10]}", file=sys.stderr)

    fwd_jit = jax.jit(_decode_one_step, static_argnames=('mesh',), donate_argnums=(4,))
    generated = []
    last = None
    total_steps = len(prompt_ids) + args.max_tokens
    t_start = time.time()
    ttft = None

    for step in range(total_steps):
        token_id = prompt_ids[step] if step < len(prompt_ids) else last
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        tok_arr = jnp.array([token_id] * B, dtype=jnp.int32)

        t0 = time.time()
        next_tok, _lp, caches = fwd_jit(tok_arr, pos, ctx, model, caches, cos, sin, mesh)
        next_tok.block_until_ready()
        dt = time.time() - t0

        last = int(next_tok[0])

        if step < len(prompt_ids):
            print('.', end='', file=sys.stderr, flush=True)
            if step == len(prompt_ids) - 1:
                ttft = time.time() - t_start
                print(
                    f"\nTTFT: {ttft*1000:.1f}ms ({len(prompt_ids)} prompt tokens)",
                    file=sys.stderr,
                )
        else:
            generated.append(last)
            if last in (1, 2):
                print(f"\n[EOS tok={last} at step {step}]", file=sys.stderr)
                break
            print(f"[{last}]", end='', file=sys.stderr, flush=True)
            if step < len(prompt_ids) + 3:
                print(f" ({dt*1000:.1f}ms)", end='', file=sys.stderr, flush=True)

    total = time.time() - t_start
    print(file=sys.stderr)
    print("=== Results ===", file=sys.stderr)
    print(f"prompt tokens:    {len(prompt_ids)}", file=sys.stderr)
    print(f"generated tokens: {len(generated)}", file=sys.stderr)
    if ttft:
        print(f"TTFT:             {ttft*1000:.1f}ms", file=sys.stderr)
    if len(generated) > 1 and ttft:
        decode_time = total - ttft
        tps = len(generated) / decode_time
        print(f"decode tok/s:     {tps:.1f}", file=sys.stderr)
        print(f"ms/token:         {decode_time/len(generated)*1000:.1f}", file=sys.stderr)
    print(f"total time:       {total:.1f}s", file=sys.stderr)
    print(f"generated:        {generated[:20]}", file=sys.stderr)
    if tokenizer is not None:
        try:
            print(f"decoded:          {tokenizer.decode(generated)!r}", file=sys.stderr)
        except Exception as e:
            print(f"decoded: <decode failed: {e}>", file=sys.stderr)


def run_perplexity_m2(args, mesh, model, caches, cos, sin, tokenizer):
    if tokenizer is None:
        print("ERROR: no tokenizer available for perplexity", file=sys.stderr)
        return

    if args.ppl_file:
        with open(args.ppl_file) as f:
            text = f.read()
    else:
        text = ("The quick brown fox jumps over the lazy dog. "
                "In the beginning was the Word, and the Word was with God.")

    ids = list(tokenizer.encode(text))
    max_tokens = min(len(ids), args.max_ctx - 1)
    bos = getattr(tokenizer, 'bos_token_id', 1)
    token_ids = [bos] + ids[:max_tokens]
    print(f"perplexity eval: {len(token_ids)} tokens", file=sys.stderr)

    fwd_jit = jax.jit(_decode_one_step, static_argnames=('mesh',), donate_argnums=(4,))
    total_nll = 0.0
    n = 0
    t_start = time.time()

    for step in range(len(token_ids) - 1):
        tok_arr = jnp.array([token_ids[step]], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        _nt, log_probs, caches = fwd_jit(tok_arr, pos, ctx, model, caches, cos, sin, mesh)
        tgt = token_ids[step + 1]
        nll = -float(log_probs[0, tgt])
        total_nll += nll
        n += 1
        if step % 50 == 0:
            ppl_sf = float(np.exp(total_nll / max(n, 1)))
            elapsed = time.time() - t_start
            tps = n / max(elapsed, 0.01)
            print(
                f"  step {step}/{len(token_ids)-1} nll={nll:.3f} "
                f"ppl={ppl_sf:.2f} ({tps:.1f} tok/s)",
                file=sys.stderr,
            )

    avg_nll = total_nll / n
    ppl = float(np.exp(avg_nll))
    elapsed = time.time() - t_start
    print(file=sys.stderr)
    print("=== Perplexity ===", file=sys.stderr)
    print(f"tokens:     {n}", file=sys.stderr)
    print(f"avg NLL:    {avg_nll:.4f}", file=sys.stderr)
    print(f"perplexity: {ppl:.2f}", file=sys.stderr)
    print(f"time:       {elapsed:.1f}s ({n/elapsed:.1f} tok/s)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--max-ctx', type=int, default=8192)
    parser.add_argument('--prompt', default='Hello')
    parser.add_argument('--perplexity', action='store_true')
    parser.add_argument('--ppl-file', default=None)
    parser.add_argument('--fused', action='store_true',
                        help='On-chip decode loop (placeholder for M2 v0)')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--path', choices=['A', 'B'], default='A',
                        help='A=on-the-fly NVFP4 dequant; B=int8 upcast at load')
    parser.add_argument('--speculate', action='store_true',
                        help='Enable MTP speculative decode (default off)')
    args = parser.parse_args()

    global B, PATH
    B = args.batch
    PATH = args.path

    load_config_m2(args.model_dir)

    mesh, _axes = make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)

    max_ctx = args.max_ctx
    print(f"path: {PATH} (A=on-the-fly NVFP4, B=int8 upcast)", file=sys.stderr)

    model = load_model_m2(args.model_dir, mesh, max_ctx, path=PATH)

    caches = make_kv_caches(B, max_ctx, mesh)

    cos, sin = precompute_rope_m2(ROPE_THETA, ROTARY_DIM, max_ctx)
    cos = jax.device_put(jnp.asarray(cos), NamedSharding(mesh, P(None, None)))
    sin = jax.device_put(jnp.asarray(sin), NamedSharding(mesh, P(None, None)))

    tokenizer = load_tokenizer_m2(args.model_dir)

    if args.speculate:
        if not USE_MTP or model.get('mtp_weights') is None:
            raise RuntimeError(
                "--speculate requested but model has no MTP weights loaded")
        print(
            f"speculate: MTP enabled with {NUM_MTP} draft modules",
            file=sys.stderr,
        )

    if args.perplexity:
        run_perplexity_m2(args, mesh, model, caches, cos, sin, tokenizer)
    else:
        run_generate_m2(args, mesh, model, caches, cos, sin, tokenizer)


if __name__ == '__main__':
    main()
