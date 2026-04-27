# MiniMax-M2.7 int8 arena for rvllm

This dataset contains an rvllm-specific flat int8 arena exported from `lukealonso/MiniMax-M2.7-NVFP4`.

It is **not** a standard Hugging Face Transformers checkpoint. It is a boot cache for the Rust/PJRT rvllm MiniMax-M2 runtime.

## Contents

- `manifest.json`: arena layout, shard list, timings, and caveats.
- `arena-*.bin`: sharded flat int8 arena bytes.

## Current export

- Source: `lukealonso/MiniMax-M2.7-NVFP4`
- Shape: B=8, ctx=2048 rvllm decode ABI
- Format: `rvllm_flat_int8_arena`
- Export path: Rust only, no Python/JAX runtime
- Conversion: NVFP4 expert weights -> int8 with f32 row scales
- Parallelism: Rayon on the TPU VM host

The first measured Rayon export materialized a 233.1 GB arena in 214.0 seconds using 180 Rayon worker threads.

## Caveat

This export currently uses `copy_dense_tensors=false`: dense slots are reserved and zero-filled while MoE expert NVFP4 weights/scales are actually converted to int8. This is useful for rvllm int8-kernel development and boot-cache testing, not a complete standalone production model.
