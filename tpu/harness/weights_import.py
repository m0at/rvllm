#!/usr/bin/env python3
"""Import pre-quantized int8 weights from npz shards. Skips CPU quantization.

Usage:
    # Pull from HF first:
    ./weights_pull.sh
    # Then use in inference:
    python3 weights_import.py --weights-dir ~/weights-int8 --max-tokens 128 --fused
"""
import argparse, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes

sys.path.insert(0, os.path.dirname(__file__))
from gemma4_tpu_infer import (
    make_mesh, _sharded_zeros, precompute_rope,
    forward_step, make_decode_loop, load_tokenizer,
    H, S_HD, S_KV, G_KV, S_KVH, G_KVH, WINDOW, BLOCK_K,
    N_SLIDING, N_GLOBAL, N_GROUPS, NL, SOFTCAP_VAL,
    rms_norm,
)

def load_weights_fast(weights_dir, mesh, max_ctx):
    """Load pre-quantized weights directly. No CPU quantization."""
    t0 = time.time()

    def load(name):
        return np.load(os.path.join(weights_dir, f"{name}.npy"), allow_pickle=True)

    def put(arr, spec):
        return jax.device_put(jnp.array(arr), NamedSharding(mesh, spec))

    def put_i8(arr, spec):
        return jax.device_put(jnp.array(arr, dtype=jnp.int8), NamedSharding(mesh, spec))

    print("loading pre-quantized weights...", file=sys.stderr)
    embed = put(load("embed"), P(None, None))
    final_norm = put(load("final_norm"), P(None))

    sl_i8_sh = {
        'qw': P(None, 'tp', None), 'kw': P(None, 'tp', None), 'vw': P(None, 'tp', None),
        'ow': P(None, None, 'tp'), 'gw': P(None, 'tp', None), 'uw': P(None, 'tp', None),
        'dw': P(None, None, 'tp'),
    }
    sl_sc_sh = {
        'qw_s': P(None, 'tp'), 'kw_s': P(None, 'tp'), 'vw_s': P(None, 'tp'),
        'ow_s': P(None, None), 'gw_s': P(None, 'tp'), 'uw_s': P(None, 'tp'),
        'dw_s': P(None, None),
    }

    sl_weights = {}
    for k in ['qw','kw','vw','ow','gw','uw','dw']:
        arr = load(f"sl.{k}")
        if arr.dtype == np.int8:
            sl_weights[k] = put_i8(arr, sl_i8_sh[k])
        else:
            sl_weights[k] = put(arr, sl_i8_sh[k])
        print(f"  sl.{k}: {arr.shape}", file=sys.stderr)
    for k in ['qw_s','kw_s','vw_s','ow_s','gw_s','uw_s','dw_s']:
        sl_weights[k] = put(load(f"sl.{k}"), sl_sc_sh[k])
    for k in ['qn','kn','ln1','ln2','ln3','ln4','ls']:
        sl_weights[k] = put(load(f"sl.{k}"), P(None, None))

    gl_weights = {}
    for k in ['qw','kw','vw','ow','gw','uw','dw']:
        arr = load(f"gl.{k}")
        if arr.dtype == np.int8:
            gl_weights[k] = put_i8(arr, sl_i8_sh[k])
        else:
            gl_weights[k] = put(arr, sl_i8_sh[k])
        print(f"  gl.{k}: {arr.shape}", file=sys.stderr)
    for k in ['qw_s','kw_s','vw_s','ow_s','gw_s','uw_s','dw_s']:
        gl_weights[k] = put(load(f"gl.{k}"), sl_sc_sh[k])
    for k in ['qn','kn','ln1','ln2','ln3','ln4','ls']:
        gl_weights[k] = put(load(f"gl.{k}"), P(None, None))

    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    kvs_sh = NamedSharding(mesh, P(None, None, None))
    sl_caches = {
        'kc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
    }
    gl_caches = {
        'kc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
    }

    cos_s = put(load("cos_s"), P(None, None))
    sin_s = put(load("sin_s"), P(None, None))
    cos_g = put(load("cos_g"), P(None, None))
    sin_g = put(load("sin_g"), P(None, None))

    elapsed = time.time() - t0
    print(f"loaded in {elapsed:.1f}s (skipped quantization)", file=sys.stderr)
    return embed, final_norm, sl_weights, gl_weights, sl_caches, gl_caches, cos_s, sin_s, cos_g, sin_g

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--max-ctx', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--prompt', default='2,3257')
    args = parser.parse_args()

    mesh = make_mesh()
    embed, final_norm, sl_w, gl_w, sl_c, gl_c, cos_s, sin_s, cos_g, sin_g = \
        load_weights_fast(args.weights_dir, mesh, args.max_ctx)

    prompt_ids = [int(x) for x in args.prompt.split(',')]
    fwd_jit = jax.jit(forward_step)

    print("generating...", file=sys.stderr)
    t0 = time.time()
    for step in range(len(prompt_ids) + args.max_tokens):
        if step < len(prompt_ids):
            token_id = prompt_ids[step]
        else:
            token_id = last_tok
        tok = jnp.array([token_id], dtype=jnp.int32)
        next_tok, sl_c, gl_c = fwd_jit(
            tok, jnp.int32(step), jnp.int32(step + 1),
            embed, final_norm, sl_w, gl_w, sl_c, gl_c,
            cos_s, sin_s, cos_g, sin_g)
        last_tok = int(next_tok[0])
        if step >= len(prompt_ids):
            if last_tok in (1, 2, 107):
                break
    elapsed = time.time() - t0
    print(f"done: {args.max_tokens} tokens in {elapsed:.1f}s", file=sys.stderr)
