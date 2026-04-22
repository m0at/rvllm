# Phase F5 — unified prefill profile + MMA correctness regression

**Date:** 2026-04-23, GB10 / driver 595.58.03, CUDA 13.2.
**Branch:** `rusty_sm121_unified_prefill_mma`.
**Goal:** measure what's left after F4 and identify where the remaining
4× gap to vLLM lives.

## 1. TTFT headline — what the user sees

One curl through `rvllm-serve` against the 1836-token persona chat
prompt (system + user turn, 3-token reply):

| config                                      | TTFT    | output        |
|---------------------------------------------|---------|---------------|
| production default (per-token outer loop)   | 61.7 s  | `"Computing's"` |
| unified kernel, `UNIFIED_PREFILL_MMA=0`     | 11.7 s  | `"Computing's"` |
| unified kernel, `UNIFIED_PREFILL_MMA=1` (F4)|  5.6 s  | **`"L1."` — broken** |
| vLLM Triton reference                       |  1.8 s  | correct        |

Short-prompt sanity check — `"The capital of France is"`, max_tokens=6:

| config            | output                      |
|-------------------|-----------------------------|
| `MMA=0` scalar    | `"Paris."`                  |
| `MMA=1` full F4   | `"Bởi vì bạn đã cung"`     |
| `MMA=1` + force `pv_use_mma=false` (Q·Kᵀ MMA only) | `"Hadr**France**."` |

Both the Q·Kᵀ-only (F3) and Q·Kᵀ + P·V (F4) tensor-core paths produce
incoherent output on the live 60-layer Gemma 4 31B model, despite
passing the single-layer NumPy harness to `scale_rel.max` bit-identity
against the scalar reference.

## 2. Nsight-Compute breakdown

`v3/target/release/probe-gemma4-load` with `RVLLM_PROMPT_LEN=1024`,
`MAX_NEW=1`, scalar unified kernel. `ncu --metrics gpu__time_duration.sum
--launch-skip 500 --launch-count 1000`. Sampled 884 kernel launches /
2.98 s of kernel time; captures the steady-state prefill stride
(model-load launches skipped).

    time_ms  count   %    name
    2343.20     38  78.6  flash_attention_2_prefill_fp8kv_unified_kernel
     145.83     45   4.9  nvjet_sm121_qqsss_mma_192x176x128_...  (cuBLASLt)
     118.33     77   4.0  cutlass_fp8_gemm_blockscale_sm120     (weight GEMMs)
      82.87     77   2.8  f32_to_f16_sat_kernel
      74.35     76   2.5  scale_rows_f32_ratio_kernel
      74.10     77   2.5  scale_cols_f32_kernel
      47.13     31   1.6  nvjet_sm121_qqsss_mma_192x160x128_...
      22.97     39   0.8  fused_gelu_mul_fp8_quant_kernel
      15.25     38   0.5  fused_qkv_rmsnorm_kernel
      11.80     77   0.4  fused_norm_add_residual_f16in_kernel
      11.21     38   0.4  fused_rope_partial_fp8kv_kernel
       9.23     76   0.3  fused_rmsnorm_fp8_quant_kernel
       6.02     38   0.2  quantize_fp8_per_token_kernel

**Takeaway.** The unified prefill attention kernel dominates — 78.6%
of kernel time. Every GEMM / fused op combined is under 20%. The
secondary kernels we already tuned (CUTLASS blockwise FP8 at 4.0%,
cuBLASLt at 4.9%, fused quant / norm at fractions of a percent) are
nowhere near the hot path. The path to ≥2× more speedup runs through
the attention kernel itself, which is exactly what F3 / F4 target —
but the current MMA implementation is numerically unusable.

Raw CSV: [`ncu_unified_scalar_1024_20260423.csv`](ncu_unified_scalar_1024_20260423.csv).

## 3. Why F3 / F4 break the live model (hypothesis)

Single-layer harness (`v3/tools/fa2_unified_prefill_check.py`) shows
MMA vs scalar `scale_rel.max` bit-identical at every shape tested
(prompt_len ∈ {64, 128, 256, 512} × head_dim ∈ {256, 512}). So the
per-layer arithmetic is correct.

Across 60 layers with softmax in between, the picture changes. Two
sources of per-layer drift in the current MMA path:

1. **Summation order inside the tensor core** differs from the
   scalar loop's left-to-right `dot += qr[d] * kr[d]`. f32 is not
   associative; tile-parallel reductions inside the MMA compute the
   same value to within one ULP, but the softmax then *amplifies*
   the delta non-linearly. Over 60 layers a few ULPs of drift per
   layer is enough to flip token-level argmax on the LM head.
2. **P → FP8 re-quant** (F4 only) throws away all mantissa below
   `row_max / 128`. For peaked softmax distributions (the common case
   post-attention) most of P is zeroed out. Single-layer bounded
   deviation compounds across many tiles × layers.

vLLM's Triton kernel uses the same tensor cores and doesn't hit this,
so the bug isn't fundamental to MMA — it's in the specific reduction
/ re-quant choices our port makes. Resolving it needs either:

* **Keep P in fp16 for the P·V matmul.** `mma.sync.m16n8k32.row.col
  .f32.fp16.fp16.f32` exists; skip the FP8 re-quant entirely.
  Smem cost: P becomes 2 B/elem (1 KB extra), V still FP8 so still
  needs a dequant pass before the MMA.
* **Don't use MMA for Q·Kᵀ.** Our numerical drift shows up even with
  only F3 engaged. Keep the scalar Q·Kᵀ and only use MMA for P·V
  with an fp16-P variant. Loses some speedup vs the F4 target but
  stays correct.
* **Investigate reduction-order choice** (`mma.sync` with different
  `.shape` / `.atom` variants, or manual multi-stage accumulation)
  until the per-layer drift is small enough that softmax doesn't
  amplify it past the argmax boundary.

Any of these is a full kernel rewrite on top of F4. Out of scope for
this phase.

## 4. Production state today

* `rvllm-serve` runs the scalar unified kernel
  (`RVLLM_BATCH_PREFILL=1`, `RVLLM_UNIFIED_PREFILL=1`,
  `RVLLM_UNIFIED_PREFILL_MMA=0`). TTFT on the 1836-token prompt:
  11.7 s (vs 61.7 s baseline). **5.3× production speedup, shippable.**
* `RVLLM_UNIFIED_PREFILL_MMA=1` opt-in exists but currently produces
  garbage output on Gemma 4 31B — kept for F6 debugging, not for
  live use.

## 5. Next steps (F6 candidates, in priority order)

1. **Reproduce the MMA drift under `rvllm-ppl`** to quantify it
   layer-by-layer. If the per-layer delta is below a known tolerance
   we can point to (say, `scale_rel < 1e-4`), the path to correct
   MMA is clearer than debugging end-to-end argmax flips.
2. **fp16-P variant of the P·V MMA.** Drop the FP8 re-quant step;
   expect modest perf loss (smem cost + dequant pass) but correct
   output.
3. **Compare reduction orders** by emitting a Triton-generated
   unified-attention kernel for reference and diffing per-layer
   state against it.
4. **Only then**: attempt an FP8 tensor-core path that's actually
   correct across 60 layers. Probably ends up looking more like
   vLLM's Triton output than our current CUDA.

The F3 / F4 tensor-core commits are preserved on branches
`rusty_sm121_unified_prefill_mma` + `rusty_sm121_inference_server_mma`.
`rusty_sm121` is back to the scalar unified kernel (F3/F4 not merged).
