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
- [ ] Sweep B=32 with replicate-token forced, then decide whether `auto` should
  include B=32.

## 1. Int8 + Flat Architecture

- [ ] Define "flat" runtime layout: one direct M2 inference surface with no
  dense/gather/Pallas experiment branches on the hot path.
- [ ] Add int8 KV cache for M2 using the Gemma4 split/unified cache pattern as
  reference.
- [ ] Add correctness gate: bf16 KV versus int8 KV on fixed prompt, PPL delta,
  and generation prefix.
- [ ] Re-bench B=16/B=32 after int8 KV.
- [ ] Decide whether NVFP4-to-int8 weight upcast is worth keeping as an offline
  experiment only; do not make it default unless PPL/coherence survives.

## 2. Custom Mosaic/MLIR Fused NVFP4 Matmul

- [ ] Build a minimal Mosaic custom-call kernel that accepts packed `uint8`
  NVFP4 weights and FP8 scales.
- [ ] Decode FP4/FP8 in VMEM/registers and feed bf16 RHS tiles to TPU matmul
  without high-level Pallas layout inference.
- [ ] Match `nvfp4_matmul` exactly on small random tensors.
- [ ] Replace only one projection behind an env flag and verify B=8 output.
- [ ] If faster, wire all MoE projections behind a correctness gate.

## 3. Batched Prefill

- [ ] Add a prefill path that processes prompt length T as one compiled forward
  over positions instead of T serial host calls.
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
