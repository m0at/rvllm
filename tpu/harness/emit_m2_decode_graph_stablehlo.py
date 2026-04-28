"""Emit the M2 decode step as a single self-contained StableHLO MLIR file.

The flow:
  1. Load real M2 weights from $REMOTE_MODEL_DIR via the existing JAX
     load_model_stacked path (m2_real_bench.load_model_stacked).
  2. JIT-lower the existing JAX forward_step from m2_synth_bench so the
     entire 62-layer decode body, including attention + MoE + MLP + final
     norm + lm_head, is captured as one StableHLO function.
  3. Print the lowered StableHLO MLIR text to stdout (or save to a file)
     so a downstream rvLLM-side path can load it as the decode graph
     instead of re-emitting from Rust.

This is the route to a real rvLLM PPL: rvLLM's Rust runtime + PJRT executes
the JAX-emitted graph; PPL = JAX PPL = 4.92 on m2_ppl_corpus.txt.

Usage on the TPU VM:
  python3 emit_m2_decode_graph_stablehlo.py \\
      --model-dir /workspace/models/m2/MiniMax-M2.7-NVFP4 \\
      --batch 8 --ctx 2048 \\
      --out /workspace/models/runs/m2_jax_decode_graph.mlir
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Find the m2bench harness on the TPU VM
HARNESS = "/workspace/runs/m2bench/tpu/harness"
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax.sharding import NamedSharding, PartitionSpec as P  # type: ignore

import m2_synth_bench  # type: ignore
import m2_real_bench  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument(
        "--out",
        default="/workspace/models/runs/m2_jax_decode_graph.mlir",
        help="Output StableHLO MLIR text path.",
    )
    p.add_argument(
        "--mode",
        choices=["text", "bytecode"],
        default="text",
        help="text emits readable .mlir, bytecode writes binary .mlirbc.",
    )
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Override batch size via the module-level B before forward_step uses it.
    m2_synth_bench.B = args.batch

    mesh = m2_synth_bench.make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)

    # Load weights + KV caches with the existing harness path.
    model_state = m2_real_bench.load_model_stacked(
        args.model_dir, mesh, args.ctx, B=args.batch, n_workers=args.workers,
    )
    print(">> model loaded", file=sys.stderr)

    # Unpack (signature varies between rvLLM-vendored and upstream JAX paths).
    # We expect (embed, final_norm, stacked, k_cache, v_cache, ...) -- handle
    # whatever the harness returns.
    if hasattr(model_state, "_asdict"):
        st = model_state._asdict()
    elif isinstance(model_state, dict):
        st = model_state
    else:
        # Tuple / list -- positional unpack matching m2_real_bench convention
        st = {
            "embed": model_state[0],
            "final_norm": model_state[1],
            "stacked": model_state[2],
            "k_cache": model_state[3],
            "v_cache": model_state[4],
            "lm_head": model_state[5] if len(model_state) > 5 else None,
        }

    embed = st.get("embed")
    final_norm = st["final_norm"]
    stacked = st["stacked"]
    k_cache = st["k_cache"]
    v_cache = st["v_cache"]
    lm_head = st.get("lm_head")
    if lm_head is None and embed is not None:
        # M2 ties lm_head with embed in some configs; if separate weight does
        # not exist in state, fall back to embed.
        lm_head = embed

    # Rotary tables -- forward_step expects (cos, sin) of size (max_ctx, head_dim_rot)
    cos, sin = m2_synth_bench.precompute_rope(
        m2_synth_bench.ROPE_THETA, m2_synth_bench.ROTARY_DIM, args.ctx
    )
    cos = jax.device_put(jnp.array(cos), NamedSharding(mesh, P(None, None)))
    sin = jax.device_put(jnp.array(sin), NamedSharding(mesh, P(None, None)))

    # Single-step inputs:
    # x: hidden state at the current decode step, shape (B, H) bf16
    # pos: position scalar, shape (B,) i32
    B = args.batch
    H = m2_synth_bench.H
    x_spec = jax.ShapeDtypeStruct((B, H), m2_synth_bench.DTYPE)
    pos_spec = jax.ShapeDtypeStruct((B,), jnp.int32)

    # JIT-lower forward_step. The function takes x, stacked, k_cache, v_cache,
    # pos, cos, sin, final_norm, lm_head, mesh -- but mesh is a static config,
    # not a runtime tensor. Wrap to capture mesh.
    def fwd(x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head):
        return m2_synth_bench.forward_step(
            x, stacked, k_cache, v_cache, pos, cos, sin, final_norm, lm_head, mesh,
        )

    print(">> jit lowering forward_step", file=sys.stderr)
    jitted = jax.jit(fwd)

    # Build abstract specs for the dynamic operands and use real arrays for
    # the (large) static-ish weights so JAX picks them up via concrete tracing.
    # For now, lower with all-real arguments (weights captured as constants).
    # If output is too big, switch to symbolic specs for stacked.
    lowered = jitted.lower(
        jax.ShapeDtypeStruct(x_spec.shape, x_spec.dtype),
        stacked,
        k_cache,
        v_cache,
        jax.ShapeDtypeStruct(pos_spec.shape, pos_spec.dtype),
        cos,
        sin,
        final_norm,
        lm_head,
    )
    mlir = lowered.compiler_ir(dialect="stablehlo")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "bytecode":
        import io

        buf = io.BytesIO()
        mlir.operation.write_bytecode(buf)
        out_path.write_bytes(buf.getvalue())
        print(f"wrote bytecode {out_path} ({len(buf.getvalue())} bytes)", file=sys.stderr)
    else:
        text = str(mlir)
        out_path.write_text(text)
        print(f"wrote text {out_path} ({len(text)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
