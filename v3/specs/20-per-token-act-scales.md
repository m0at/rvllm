# Spec 20: Per-token activation scales on cuBLASLt (B5)

## Target

+1–2% decode throughput on QKV, O, gate_up, down GEMMs (4 of the 5 FP8 linears — lm_head already uses per-token scales). 2 days of work.

## Current state

In `v3/crates/rvllm-cutlass/src/cublaslt.rs`, all FP8 matmuls use per-tensor A-scale (`A_SCALE_POINTER` = address of single f32 scalar broadcast across all rows). The activation comes in FP8 with a single per-tensor scale that was produced by the preceding `fused_*_fp8_quant` kernel (which computes a *per-token* scale but currently writes it into a per-row layout that we ignore).

Concretely:
- `fused_add_rmsnorm_fp8_quant` writes `output_scales[row]` — per-token scale vector of length `num_tokens`.
- `fused_silu_mul_fp8_quant` does the same.
- `quantize_fp8_per_token` writes `output_scale[row]` the same way.

All three produce per-token scales. cuBLASLt is being told to use per-tensor.

**Impact of the loss:** when activations have outliers clustered on a few tokens (common in pre-fill, less so in decode), per-tensor scaling gives those outliers their dynamic range at the cost of crushing the other tokens into the FP8 E4M3 bottom few bins. Per-token preserves precision on all rows.

## Required API: vector A-scale

cuBLASLt has supported per-output-row scaling via `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` pointed at a vector (not scalar) since CUDA 12.2. We need to flag the descriptor with `CUBLASLT_MATMUL_DESC_A_SCALE_VEC_MODE` (or equivalent — confirm with cuBLASLt docs for our CUDA version).

Implementation steps:

1. **Check cuBLASLt version support.** On H100 box: `cublasLtGetVersion()` must be ≥ 12020. If not, upgrade CUDA toolkit first.
2. **Descriptor flag.** Set an attribute on the matmul descriptor to tell cuBLASLt the A-scale pointer is a vector of length M (not a scalar). Exact attribute name: `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT` — verify; fallback is to check `CUBLASLT_MATMUL_DESC_A_SCALE_MODE`.
3. **Pass the vector.** In `fp8_gemm`, `fp8_gemm_bias`, `fp8_gemm_residual`, add a new parameter `a_scale_is_vec: bool` OR a new sibling method. Prefer the sibling method (cleaner contract): `fp8_gemm_pertok(a_fp8, b_fp8, d, m, n, k, a_scale_vec_ptr, b_scale_ptr, stream)`.
4. **Rewire call sites.** In `layer_exec.rs`:
   - QKV GEMM: A = `hidden_fp8`, A-scale vec = `hidden_scale`.
   - O GEMM: A = `attn_out_fp8`, A-scale vec = `attn_out_scale`.
   - gate_up: A = `hidden_fp8` (second one), A-scale vec = `hidden_scale` (second one, post-MLP norm).
   - down: A = `mlp_out_fp8`, A-scale vec = `mlp_out_scale`.
5. **Re-autotune.** cuBLASLt heuristics may pick different algos with vector scaling; our per-shape algo cache keys on shape + epilogue + bias flag. Extend the `AlgoKey` to include `a_scale_vec: bool`.
6. **Correctness bench.** N=1 greedy, 32 tokens, compare token IDs against the per-tensor path. Cosine on logits ≥ 0.9999.
7. **Bench.** N=128/256/512. Expect ≥ +1% throughput, no MMLU regression.

## Risks

- **cuBLASLt may not support per-token A-scale in combination with EPILOGUE_BIAS or β=1.** If so, fall back to per-tensor on those specific GEMMs or pre-apply the bias in a tiny kernel. Test before wiring broadly.
- **Algo heuristic may pick a noticeably slower algo when vector scaling is on.** We measure and keep per-tensor if the per-token algo isn't competitive on our shapes.

## Success criteria

- N=128 tok/s ≥ 22,500 (+2% floor from current 22,069).
- No change in greedy output for 32-token prompts.
- compute-sanitizer memcheck clean.

## Files touched

- `v3/crates/rvllm-cutlass/src/cublaslt.rs` — add `fp8_gemm_pertok*` variants + extend `AlgoKey`.
- `v3/crates/rvllm-runtime/src/layer_exec.rs` — pass per-token scale pointers where available.
- Unit test in `v3/crates/rvllm-cutlass/tests/` that compares per-tensor vs per-token output cosine.
