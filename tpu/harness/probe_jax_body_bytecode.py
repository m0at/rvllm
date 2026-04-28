"""Probe: can a JAX-emitted StableHLO bytecode .mlirbc slot into the rvLLM
tpu_custom_call body slot in place of a Mosaic body?

The validator at m2_tpu_custom_call.rs:26 only checks the MLIR-bytecode magic
(ML\xefR), not the dialect. This script emits a trivial M2-layer-shaped JAX
function (currently identity) as a StableHLO bytecode .mlirbc and writes it
to /tmp. Next step is to scp it to the TPU VM and link it via the existing
m2_rust_decode_bench --decode-layer-body-format serde path. If PJRT accepts
the body and the full-graph compile passes, we have the unblock for an
end-to-end JAX-extracted real-MLP body.

Operand contract mirrors the existing Mosaic body args in
v3/crates/rvllm-xla/src/m2_decode_body.rs (single-tile N=128 shape):

  hidden:           bf16 [8, 3072]
  positions:        i32  [8]
  kv_in:            i8   [33554432]
  layer_offsets:    i32  [34]
  expert_directory: i32  [256, 25]
  w1_block_t:       i8   [3072, 128]
  w1_row_scales:    f32  [128]
Returns:
  hidden_out:       bf16 [8, 3072]
  kv_out:           i8   [33554432]
"""

from __future__ import annotations

import io
import pathlib
import sys

import jax
import jax.numpy as jnp


def m2_layer_identity(hidden, positions, kv_in, layer_offsets,
                      expert_directory, w1_block_t, w1_row_scales):
    """Identity passthrough -- minimal probe of the JAX → bytecode → rvLLM body path."""
    # Reference operands so XLA does not optimize them out of the signature.
    _ = positions, layer_offsets, expert_directory, w1_block_t, w1_row_scales
    return hidden, kv_in


def main(out_path: str = "/tmp/m2_layer_probe_jax.mlirbc"):
    arg_specs = [
        jax.ShapeDtypeStruct((8, 3072), jnp.bfloat16),
        jax.ShapeDtypeStruct((8,), jnp.int32),
        jax.ShapeDtypeStruct((33554432,), jnp.int8),
        jax.ShapeDtypeStruct((34,), jnp.int32),
        jax.ShapeDtypeStruct((256, 25), jnp.int32),
        jax.ShapeDtypeStruct((3072, 128), jnp.int8),
        jax.ShapeDtypeStruct((128,), jnp.float32),
    ]
    jitted = jax.jit(m2_layer_identity)
    lowered = jitted.lower(*arg_specs)
    mlir = lowered.compiler_ir(dialect="stablehlo")
    buf = io.BytesIO()
    mlir.operation.write_bytecode(buf)
    bc = buf.getvalue()
    out = pathlib.Path(out_path)
    out.write_bytes(bc)
    head = bc[:4].hex()
    print(f"wrote {out} ({len(bc)} bytes), magic={head}")
    if not bc.startswith(b"ML\xefR"):
        print("FAIL: bytecode does not start with ML xefR magic", file=sys.stderr)
        sys.exit(1)
    print("magic ok; ready for rvLLM body-slot probe")


if __name__ == "__main__":
    main(*sys.argv[1:])
