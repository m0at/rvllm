# MiniMax-M2 Optimization Tasks

Hard order. Do not start later work until the current item has a correctness
gate and a saved result JSON.

## 0. B=8/B=16 MoE Dispatch

- [x] Replace padded all-to-all buckets for B=8/B=16 with replicate-token MoE.
- [x] Skip inactive local experts with `lax.cond` so empty experts do not run
  gate/up/down NVFP4 matmuls.
- [x] Verify B=8 next-token output matches legacy all-to-all.
- [x] Record B=8: 146.2 ms/step, 54.7 tok/s.
- [x] Record B=16: 154.9 ms/step, 103.3 tok/s.
- [x] Sweep B=32 with replicate-token forced: 186.8 ms/step, 171.3 tok/s.
- [x] Include B=32 in `RVLLM_M2_MOE_IMPL=auto`.

## 1. Int8 + Flat Architecture

- [x] Collapse bench/PPL/gen onto one scanned forward core so correctness paths
  cannot drift from throughput paths.
- [x] Add opt-in int8 KV cache for M2 decode (`RVLLM_M2_KV=int8`) using Gemma4's
  per-head vector scale pattern.
- [x] Run B=8 int8-KV compile/runtime smoke and save result JSON.
- [x] Add correctness gate: bf16 KV versus int8 KV on fixed prompt, PPL delta,
  and generation prefix. Saved JSONs under `tpu/out/m2/`.
- [x] Re-bench B=16/B=32 after int8 KV and record whether B=32 is unlocked.
- [x] Decide whether int8 KV should become default for B>=16 only or remain
  opt-in. Decision: keep opt-in until API/infer wrappers use the same flat
  cache surface.
- [x] Decide whether NVFP4-to-int8 weight upcast is worth keeping as an offline
  experiment only; do not make it default unless PPL/coherence survives.
  Decision: offline experiment only. It bloats the 130 GB NVFP4 checkpoint and
  is not on the default path.

Recorded int8-KV results:

- B=8: 145.2 ms/step, 55.1 tok/s (`tpu/out/m2/m2_int8_b8.json`)
- B=16: 154.6 ms/step, 103.5 tok/s (`tpu/out/m2/m2_int8_b16.json`)
- B=32: 186.8 ms/step, 171.3 tok/s (`tpu/out/m2/m2_int8_b32_replicate.json`)
- Correctness: PPL 5.60 on 318 scored tokens + coherent 64-token generation
  (`tpu/out/m2/m2_int8_correct.json`)

## 2. Custom Mosaic/MLIR Fused NVFP4 Matmul

- [x] Write a minimal Mosaic/MLIR custom-call scaffold outside the hot path.
- [x] Build the first kernel that accepts packed `uint8` NVFP4 weights and FP8
  scales.
- [x] Pin the Rust-side B=8 Mosaic tile contract (`BM=8, BN=512, BK=1024`),
  including packed/scales/decoded-RHS/accumulator VMEM working-set accounting.
- [x] Match the Rust NVFP4 matmul reference exactly against expanded dequant on
  deterministic small tensors.
- [ ] Decode FP4/FP8 in VMEM/registers and feed bf16 RHS tiles to TPU matmul
  without high-level Pallas layout inference.
- [ ] Match TPU/JAX `nvfp4_matmul` exactly on small random tensors.
- [ ] Replace only one projection behind an env flag and verify B=8 output.
- [ ] If faster, wire all MoE projections behind a correctness gate.

Rust artifacts:

- `rvllm_fused::m2_nvfp4`: exact Rust NVFP4 decode + matmul reference.
- `rvllm_fused::M2Nvfp4MosaicTilePlan`: B=8/B=16 tile contract and VMEM
  working-set accounting for fused decode -> bf16 RHS tile -> TPU matmul.
- `m2_emit_nvfp4_mosaic`: emits the M2 Mosaic custom-call MLIR signature plus
  explicit `BM/BN/BK` tile metadata.
- `tpu/out/m2/rvllm_m2_nvfp4_matmul.mlir`: generated B=8 gate/up kernel
  scaffold (`x_bf16`, packed NVFP4 `uint8`, FP8 scales, global scale, bf16 out;
  `BM=8, BN=512, BK=1024`, 1.38 MB per-tile working set).

Rust verification:

- `cargo test -p rvllm-fused m2_nvfp4 --release` (6 tests passing)

## 3. Batched Prefill

- [ ] Add a prefill path that processes prompt length T as one compiled
  `lax.scan` over positions instead of T serial host calls.
- [ ] Preserve KV cache writes for every prompt position.
- [ ] Verify generated text matches serial prefill for the angular-momentum
  prompt.
- [ ] Record TTFT before/after for 20-token and 128-token prompts.

## 4. EAGLE-3 / Spec Decode

- [ ] Reuse the working v6e-4 EAGLE scaffolding as reference.
- [ ] Port target-model feature capture to M2.
- [ ] Train or load a draft head; no speculative numbers without an acceptance
  rate report.
- [ ] Verify greedy losslessness against the target model.
- [ ] Record accepted tokens/cycle and end-to-end tok/s.
