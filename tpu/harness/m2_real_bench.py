"""Real MiniMax-M2.7-NVFP4 bench on TPU v6e-8.

Loads the actual checkpoint from a local directory (typically /dev/shm/m2-nvfp4),
parallelizes the 190K-tensor read via ThreadPoolExecutor over safetensors shards,
stacks weights along a leading NL axis, and runs the same scan-based forward
pass from m2_synth_bench.py.

Usage:
    python3 m2_real_bench.py --model-dir /dev/shm/m2-nvfp4 \\
                             --batch 1 --ctx 2048 --iters 5 --warmup 2
"""

import argparse
import os
import sys
import time
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_synth_bench import (
    H, NH, NKV, HEAD_DIM, ROTARY_DIM, ROPE_THETA,
    MOE_INTER, NUM_EXPERTS, TOP_K, NL_FULL, VOCAB, RMS_EPS, GROUP_SIZE, DTYPE,
    make_mesh_v6e8, precompute_rope, forward_step,
)
from nvfp4_loader import ModeloptSafetensorsReader


def _shard_tensor_list(reader, names):
    """Group tensor names by their safetensors shard file."""
    by_shard = defaultdict(list)
    for n in names:
        path = reader._name_to_shard.get(n)
        if path is None:
            raise KeyError(f"tensor not found in any shard: {n}")
        by_shard[path].append(n)
    return by_shard


def _read_shard_group(reader, shard_path, names, is_nvfp4):
    """Read all tensors from one shard. Called from a worker thread."""
    reader._open_shard(shard_path)
    out = {}
    for n in names:
        if is_nvfp4(n):
            out[n] = reader.read_nvfp4(n)
        else:
            out[n] = reader.read_bf16(n)
    return out


def parallel_read(reader, names, n_workers=16):
    """Read 190K tensors in parallel by sharding work across shard files."""
    by_shard = _shard_tensor_list(reader, names)

    def is_nvfp4(n):
        # NVFP4 tensors: pairs have `<base>.weight` uint8 + `<base>.weight_scale`.
        return reader.is_nvfp4(n)

    out = {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=min(n_workers, len(by_shard))) as ex:
        futs = [ex.submit(_read_shard_group, reader, p, ns, is_nvfp4)
                for p, ns in by_shard.items()]
        done = 0
        total = len(by_shard)
        for f in futs:
            out.update(f.result())
            done += 1
            if done % 4 == 0 or done == total:
                elapsed = time.time() - start
                print(f"  loaded {done}/{total} shards ({elapsed:.1f}s)",
                      file=sys.stderr)
    return out


def _bf16_stack(arrays):
    """Stack a list of (already-numpy, bf16) arrays along a new axis 0."""
    return np.stack(arrays, axis=0)


def _nvfp4_stack(tensors):
    """Stack a list of NvFp4Tensor (packed+scales+global_scale) along axis 0.

    Returns (packed_stacked, scales_stacked, scale2_stacked) as numpy arrays.
    """
    packed = np.stack([t.packed for t in tensors], axis=0)       # (NL, E, M, N/2)
    scales = np.stack([t.scales for t in tensors], axis=0)       # (NL, E, M, N/16)
    scale2 = np.stack(
        [np.asarray(getattr(t, 'global_scale', 1.0), dtype=np.float32)
         for t in tensors],
        axis=0,
    )  # (NL,) — but we want per-expert, so broadcast below if needed
    # If scale2 is scalar per tensor, broadcast to (NL, E) with same value.
    if scale2.ndim == 1:  # (NL,)
        # Need per-expert; since the real checkpoint has one scale2 per tensor
        # (which is per-expert weight), expand to (NL, E) by duplicating.
        # Each NvFp4Tensor IS for one expert — so stacking NL of them gives (NL,).
        # But we have 256 experts per layer, so we need to stack by layer.
        pass  # Caller handles the per-expert dimension via its own loop.
    return packed, scales, scale2


def load_model_stacked(model_dir, mesh, max_ctx, B, n_workers=16):
    """Load real MiniMax-M2.7-NVFP4 checkpoint in scan-compatible stacked layout.

    Returns the same pytree shape that m2_synth_bench.random_weights returns:
        embed, final_norm, lm_head, stacked, k_cache, v_cache, cos, sin
    """
    reader = ModeloptSafetensorsReader(model_dir)
    expert_spec = NamedSharding(mesh, P(None, 'expert', None, None))
    scale2_spec = NamedSharding(mesh, P(None, 'expert'))
    rep_spec = NamedSharding(mesh, P())

    # Step 1: build list of all tensor names we need.
    nl = NL_FULL

    # Backbone
    bb_names = [
        'model.embed_tokens.weight',
        'model.norm.weight',
        'lm_head.weight',
    ]

    # Per-layer (62 layers)
    layer_bf16_names = []
    for i in range(nl):
        p = f'model.layers.{i}'
        layer_bf16_names.extend([
            f'{p}.input_layernorm.weight',
            f'{p}.post_attention_layernorm.weight',
            f'{p}.self_attn.q_proj.weight',
            f'{p}.self_attn.k_proj.weight',
            f'{p}.self_attn.v_proj.weight',
            f'{p}.self_attn.o_proj.weight',
            f'{p}.self_attn.q_norm.weight',
            f'{p}.self_attn.k_norm.weight',
            f'{p}.block_sparse_moe.gate.weight',
            f'{p}.block_sparse_moe.e_score_correction_bias',
        ])

    # Per-layer per-expert NVFP4 (62 * 256 experts * 3 projections)
    expert_nvfp4_names = []
    for i in range(nl):
        p = f'model.layers.{i}'
        for e in range(NUM_EXPERTS):
            for w in ('w1', 'w2', 'w3'):
                expert_nvfp4_names.append(
                    f'{p}.block_sparse_moe.experts.{e}.{w}.weight')

    all_names = bb_names + layer_bf16_names + expert_nvfp4_names
    print(f">> reading {len(all_names)} tensors across {len(reader.shards)} shards "
          f"with {n_workers} workers", file=sys.stderr)

    t0 = time.time()
    raw = parallel_read(reader, all_names, n_workers=n_workers)
    print(f">> read done in {time.time()-t0:.1f}s", file=sys.stderr)

    # Step 2: assemble into the scan pytree. Stack per layer.
    print(">> stacking + device-placing", file=sys.stderr)

    def to_bf16(a):
        if a.dtype == np.dtype(ml_dtypes.bfloat16):
            return a
        return a.astype(ml_dtypes.bfloat16)

    # Backbone
    embed = jax.device_put(jnp.asarray(to_bf16(raw['model.embed_tokens.weight']), dtype=DTYPE), rep_spec)
    final_norm = jax.device_put(jnp.asarray(to_bf16(raw['model.norm.weight']), dtype=DTYPE), rep_spec)
    lm_head = jax.device_put(jnp.asarray(to_bf16(raw['lm_head.weight']), dtype=DTYPE), rep_spec)

    # Per-layer stacked
    def stack_by_layer(name_fmt):
        return np.stack([to_bf16(raw[name_fmt.format(i=i)]) for i in range(nl)], axis=0)

    # Attn shapes: Q (NH*HEAD_DIM, H), K (NKV*HEAD_DIM, H), V (NKV*HEAD_DIM, H), O (H, NH*HEAD_DIM)
    stacked = {
        'ln1': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.input_layernorm.weight'), dtype=DTYPE),
            rep_spec),
        'ln2': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.post_attention_layernorm.weight'), dtype=DTYPE),
            rep_spec),
        'attn_q': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.self_attn.q_proj.weight'), dtype=DTYPE),
            rep_spec),
        'attn_k': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.self_attn.k_proj.weight'), dtype=DTYPE),
            rep_spec),
        'attn_v': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.self_attn.v_proj.weight'), dtype=DTYPE),
            rep_spec),
        'attn_o': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.self_attn.o_proj.weight'), dtype=DTYPE),
            rep_spec),
        'attn_qn': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.self_attn.q_norm.weight'), dtype=DTYPE),
            rep_spec),
        'attn_kn': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.self_attn.k_norm.weight'), dtype=DTYPE),
            rep_spec),
        'rg': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.block_sparse_moe.gate.weight'), dtype=DTYPE),
            rep_spec),
        'rb': jax.device_put(
            jnp.asarray(stack_by_layer('model.layers.{i}.block_sparse_moe.e_score_correction_bias'), dtype=DTYPE),
            rep_spec),
    }

    # Per-expert NVFP4: for each projection, stack (NL, E, rows, cols/2) packed + (NL, E, rows, cols/16) scale.
    def stack_experts(proj_name):
        """proj_name in {'w1', 'w2', 'w3'}. Returns (packed_stack, scale_stack, scale2_stack)."""
        packed_all, scale_all, scale2_all = [], [], []
        for i in range(nl):
            p = f'model.layers.{i}.block_sparse_moe.experts'
            packed_L, scale_L, scale2_L = [], [], []
            for e in range(NUM_EXPERTS):
                t = raw[f'{p}.{e}.{proj_name}.weight']
                packed_L.append(t.packed)
                scale_L.append(t.scales)
                scale2_L.append(np.asarray(t.global_scale, dtype=np.float32))
            packed_all.append(np.stack(packed_L, axis=0))  # (E, rows, cols/2)
            scale_all.append(np.stack(scale_L, axis=0))
            scale2_all.append(np.stack(scale2_L, axis=0))   # (E,)
        return (
            np.stack(packed_all, axis=0),   # (NL, E, rows, cols/2)
            np.stack(scale_all, axis=0),
            np.stack(scale2_all, axis=0),   # (NL, E)
        )

    for pn, key_prefix in [('w1', 'w1'), ('w2', 'w2'), ('w3', 'w3')]:
        print(f">> stacking experts {pn}", file=sys.stderr)
        p_arr, s_arr, s2_arr = stack_experts(pn)
        stacked[f'{key_prefix}_p'] = jax.device_put(jnp.asarray(p_arr), expert_spec)
        stacked[f'{key_prefix}_s'] = jax.device_put(jnp.asarray(s_arr), expert_spec)
        stacked[f'{key_prefix}_s2'] = jax.device_put(jnp.asarray(s2_arr), scale2_spec)

    # KV caches
    k_cache = jax.device_put(jnp.zeros((nl, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), rep_spec)
    v_cache = jax.device_put(jnp.zeros((nl, B, max_ctx, NKV, HEAD_DIM), dtype=DTYPE), rep_spec)

    cos, sin = precompute_rope(ROPE_THETA, ROTARY_DIM, max_ctx)
    cos = jax.device_put(jnp.array(cos), rep_spec)
    sin = jax.device_put(jnp.array(sin), rep_spec)

    return embed, final_norm, lm_head, stacked, k_cache, v_cache, cos, sin


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--ctx', type=int, default=2048)
    p.add_argument('--iters', type=int, default=5)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    mesh = make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)

    t_load0 = time.time()
    embed, final_norm, lm_head, stacked, k_cache, v_cache, cos, sin = load_model_stacked(
        args.model_dir, mesh, args.ctx, args.batch, n_workers=args.workers)
    load_s = time.time() - t_load0
    print(f">> total load+place: {load_s:.1f}s", file=sys.stderr)

    def _forward(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        return forward_step(x, stacked, k_cache, v_cache, pos, cos, sin,
                          final_norm, lm_head, mesh)
    forward_jit = jax.jit(_forward)

    tok = jnp.zeros((args.batch,), dtype=jnp.int32)
    x = embed[tok]

    print(f">> warmup ({args.warmup} iters) — first compile may be long", file=sys.stderr)
    for it in range(args.warmup):
        tok, k_cache, v_cache = forward_jit(x, stacked, k_cache, v_cache,
                                            jnp.int32(it), cos, sin,
                                            final_norm, lm_head)
        x = embed[tok]
    jax.block_until_ready(tok)

    print(f">> measure ({args.iters} iters)", file=sys.stderr)
    times = []
    tokens = []
    for it in range(args.iters):
        t0 = time.perf_counter()
        tok, k_cache, v_cache = forward_jit(x, stacked, k_cache, v_cache,
                                            jnp.int32(args.warmup + it), cos, sin,
                                            final_norm, lm_head)
        jax.block_until_ready(tok)
        times.append((time.perf_counter() - t0) * 1000)
        tokens.append(int(tok[0]))
        x = embed[tok]

    times = np.array(times)
    tok_s = 1000.0 * args.batch / times.mean()
    result = {
        'arch': 'MiniMax-M2.7-NVFP4 real',
        'slice': 'v6e-8',
        'batch': args.batch, 'ctx': args.ctx, 'nl': NL_FULL,
        'iters': args.iters,
        'load_seconds': load_s,
        'ms_min': float(times.min()),
        'ms_mean': float(times.mean()),
        'ms_max': float(times.max()),
        'ms_p50': float(np.median(times)),
        'tok_per_s': float(tok_s),
        'first_tokens': tokens[:5],
    }
    print(f"\nB={args.batch} ctx={args.ctx} NL={NL_FULL}")
    print(f"  load      : {load_s:.1f}s")
    print(f"  ms/step   : {result['ms_mean']:.2f} (min={result['ms_min']:.2f})")
    print(f"  tok/s     : {tok_s:.2f}")
    print(f"  tokens    : {tokens[:5]}")

    out_path = args.out or f"/tmp/m2_real_bench_{int(time.time())}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
