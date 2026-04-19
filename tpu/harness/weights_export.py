#!/usr/bin/env python3
"""Export pre-quantized int8 weights + split caches layout to npz shards.

Run on TPU after load_model to save the sharded weight arrays.
On next deploy, weights_import.py loads them directly (skips quantization).

Usage:
    python3 weights_export.py --model-dir ~/models/gemma-4-31B-it --output-dir ~/weights-int8
    # Then push to HF:
    ./weights_push.sh ~/weights-int8
"""
import argparse, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from gemma4_tpu_infer import load_model, make_mesh, precompute_rope, S_HD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-ctx', type=int, default=2048)
    args = parser.parse_args()

    mesh = make_mesh()
    print(f"mesh: {mesh}", file=sys.stderr)

    print("loading + quantizing model...", file=sys.stderr)
    t0 = time.time()
    embed, final_norm, sl_weights, gl_weights, sl_caches, gl_caches = \
        load_model(args.model_dir, mesh, args.max_ctx)
    print(f"loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)

    def save_array(name, arr):
        path = os.path.join(args.output_dir, f"{name}.npy")
        np.save(path, np.array(arr))
        mb = os.path.getsize(path) / 1e6
        print(f"  {name}: {arr.shape} {arr.dtype} -> {mb:.1f}MB", file=sys.stderr)

    print("saving embed + final_norm...", file=sys.stderr)
    save_array("embed", embed)
    save_array("final_norm", final_norm)

    print("saving sliding weights (50 layers)...", file=sys.stderr)
    for k, v in sl_weights.items():
        save_array(f"sl.{k}", v)

    print("saving global weights (10 layers)...", file=sys.stderr)
    for k, v in gl_weights.items():
        save_array(f"gl.{k}", v)

    cos_s, sin_s = precompute_rope(10000.0, S_HD, args.max_ctx)
    cos_g, sin_g = precompute_rope(1000000.0, 128, args.max_ctx)
    save_array("cos_s", cos_s)
    save_array("sin_s", sin_s)
    save_array("cos_g", cos_g)
    save_array("sin_g", sin_g)

    meta = {
        'max_ctx': args.max_ctx,
        'n_sliding': 50,
        'n_global': 10,
        'window': 1024,
        'block_k': 8192,
    }
    np.save(os.path.join(args.output_dir, "meta.npy"), meta)

    total_mb = sum(os.path.getsize(os.path.join(args.output_dir, f))
                   for f in os.listdir(args.output_dir) if f.endswith('.npy')) / 1e6
    print(f"\ntotal: {total_mb:.0f}MB in {args.output_dir}", file=sys.stderr)

if __name__ == '__main__':
    main()
