# Spec 19: LM-head FP8 from pre-quantized activation (B4)

## Target

+3–4% decode throughput at N=128 by removing one fused_rmsnorm_fp8_quant launch + one HBM round-trip on the sampling tail. 1 day of work.

## Current tail (pre-Phase F + Phase F)

```
layer 27 down_proj (cuBLASLt residual epilogue) → residual (f16, max_tokens × hidden)
fused_rmsnorm_fp8_quant(model.norm γ)           → hidden_fp8 + hidden_scale (per-token f32)
cuBLASLt FP8 lm_head(hidden_fp8 → logits_f16)
argmax_kernel(logits_f16 → sampled_tokens i32)
```

The rmsnorm+quant kernel is *already* how every layer's input to QKV is produced, so we have that quantized tensor once per layer. But the LM-head's input is NOT the output of any layer's rmsnorm — it's the final `model.norm` applied to the residual. So we have to apply `model.norm` somehow.

## Observation

The FP8 quantization inside `fused_rmsnorm_fp8_quant` writes per-token scales to `hidden_scale`. The cuBLASLt FP8 lm_head GEMM then re-uses those scales via `A_SCALE_POINTER`. **Both kernels are already correct — the +3–4% doesn't come from removing them on decode.**

Rethink: on **batched decode**, num_tokens = num_seqs = N. One call of `fused_rmsnorm_fp8_quant` with num_tokens=N is cheap. The +3–4% claim in Task #7 was originally about eliminating the LM-head's separate f16→FP8 quantize — which we **already do** in v3 via `fused_rmsnorm_fp8_quant(model.norm)`. So B4 as originally scoped is **already shipped**.

## Revised B4: real lever is removing the second pass

Look at `layer_exec.rs`: after the final layer, step 1 of the *next* layer is `fused_add_rmsnorm_fp8_quant(attn_norm_gamma)`. On the last layer we skip that, but we *do* run `fused_rmsnorm_fp8_quant(model.norm)` before the LM head. These are two rmsnorm kernels that both read the residual and both write FP8 hidden, with different gamma tensors.

If we stored the FP8 hidden from the last layer's *input normalization* and applied a delta-γ correction in the LM-head GEMM's prologue, we'd save one rmsnorm kernel. But this is a 5-day job (EVT prologue, same machinery as B2) — not 1 day.

**Conclusion:** B4 as described ("1 day, +3-4%") is already done. Drop this task. Redirect the budget toward B5 per-token scales (which we may only be doing on some of the 5 linears today).

## Action

Verify by reading `v3/crates/rvllm-runtime/src/bring_up.rs` LM-head tail and `layer_exec.rs` decode path. Check that:
1. `fused_rmsnorm_fp8_quant(model.norm)` runs once between layer 27 and lm_head.
2. cuBLASLt FP8 lm_head uses the per-token scale from that kernel.
3. No redundant quantize between them.

If any of these is false, fix it (1–2 hours) and ship. If all are true, mark B4 as completed (already shipped) and move to B5.
