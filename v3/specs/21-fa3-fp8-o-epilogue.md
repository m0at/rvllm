# Spec 21: FA3 FP8 O-output epilogue patch (B3)

## Target

+2–4% decode throughput at N=128 by having FA3's hdim128 E4M3 paged kernel write FP8 E4M3 output directly, eliminating `quantize_fp8_per_token` + one HBM round-trip of `attn_out` per layer × 28. 4 days.

## Current path

```
FA3 fa3_sm90_paged_decode_fp8(q_fp8, k_fp8, v_fp8)
  → attn_out_f16 (stored f16 per kernel: T_out = cutlass::bfloat16_t per template logic,
                   cast to __half by our wrapper on write-out)
→ quantize_fp8_per_token_kernel(attn_out_f16) → attn_out_fp8 + attn_out_scale
→ cuBLASLt FP8 O-proj(attn_out_fp8, W_o) → residual += ...
```

Two kernels + one 2× HBM traffic round-trip (write f16, read f16, write fp8).

## Target path

```
FA3 fa3_sm90_paged_decode_fp8_o_fp8(q_fp8, k_fp8, v_fp8)
  → attn_out_fp8 + per-tensor O-scale (OR per-token O-scale if feasible)
→ cuBLASLt FP8 O-proj(attn_out_fp8, W_o) → residual += ...
```

One kernel + 1× HBM round-trip.

## Upstream status

Per our research, FA3 upstream's E4M3 paged kernels **use `T_out = cutlass::bfloat16_t` internally** (see `flash_fwd_launch_template.h:207`). There is no flag to change that. The epilogue stores `T_out` via TMA.

We have two options:

### Option A: patch FA3 source (chosen)

Maintain `kernels/fa3_patches/fp8_o_output.patch`. The patch:
1. Changes `T_out = cutlass::float_e4m3_t` when `Is_FP8 && params.o_is_fp8 == true`.
2. Threads an `o_scale_ptr: float*` param through `Flash_fwd_params`.
3. In the epilogue store path, applies `o_scale_inv * val` before `__nv_fp8_e4m3` cast.
4. CUTLASS copy-atom for E4M3 output (TMA store of 1-byte elements) — CUTLASS 3.x supports this; we need to pick the right `SM90_TMA_STORE` variant.

Estimated ~400–600 LOC touched across 3–5 FA3 files. Apply patch in `kernels/build_fa3.sh` before `nvcc` invocation. Upstream pin: whatever SHA we're on today (record in the patch header).

### Option B: write a tiny post-attention "fp16 → fp8 + scale" fused with a memory coalesce

Skip — this is exactly what `quantize_fp8_per_token` does today. No win.

**Decision: Option A.**

## Implementation

### 1. FA3 patch

Files likely patched (verify on H100 box, FA3 tree at `/root/flash-attention`):
- `hopper/flash.h` — add `void* o_fp8_ptr; float* o_descale_ptr;` to `Flash_fwd_params`.
- `hopper/epilogue_fwd.hpp` — in the `CollectiveEpilogueFwd` specialization for Sm90, when `params.o_is_fp8`, use `ElementOut = cutlass::float_e4m3_t` and scale the accumulator by `o_descale_inv` before the TMA store.
- `hopper/flash_fwd_launch_template.h` — propagate `params.o_is_fp8` into the `ElementOut` type alias.
- `hopper/instantiations/flash_fwd_hdim128_e4m3_paged_sm90.cu` — add the FP8-O instantiation.

Keep f16-O path as default; new FP8-O path gated on a template bool. That way we avoid breaking the prefill path (which still writes f16) in the same .so.

### 2. Wrapper changes

`kernels/fa3_sm90_wrapper.cu`:
- New entry: `fa3_sm90_paged_decode_fp8_o_fp8(...)` that sets `params.o_is_fp8 = true` and `params.o_descale_ptr = o_scale_dev_ptr` before dispatching to the FP8-O template.
- Keep existing `fa3_sm90_paged_decode_fp8` for fallback.

### 3. Rust FFI

`v3/crates/rvllm-attention/src/lib.rs`:
- Add `fn_paged_decode_fp8_o_fp8: Option<PagedDecodeFp8OFp8Fn>` (Option so the engine doesn't refuse to start on older .so).

`v3/crates/rvllm-attention/src/decode.rs`:
- Add `PagedDecodeFp8OFp8Launcher` mirroring `PagedDecodeFp8Launcher` with an extra `o_scale_ptr` param.

### 4. Wire in layer_exec

When `Fa3Kernels::fn_paged_decode_fp8_o_fp8.is_some()`:
- Swap `PagedDecodeFp8Launcher` → `PagedDecodeFp8OFp8Launcher`.
- Pass `attn_out_fp8` and `attn_out_scale` directly as outputs (skip the pre-existing `attn_out` f16 scratch).
- Delete the `QuantizeFp8PerTokenLaunch` call (step 7 in `forward_phase`).

### 5. Scale source

The O-scale is per-tensor for simplicity (one f32 per call). Compute it at calibration time or default to the same static seed (`1/448`) used elsewhere. If quality suffers, upgrade to per-token: FA3 would need to *write* the per-token scale alongside the FP8 output, which means a second TMA store — doable but doubles epilogue complexity.

## Risks

1. **FA3 upstream churn.** If we rebase onto a newer FA3, the patch may not apply cleanly. Mitigation: pin a SHA in `kernels/build_fa3.sh` and bump deliberately.
2. **CUTLASS 3.x FP8 TMA store.** Verify the epilogue path can emit E4M3 without layout issues. FA3 Hopper already uses TMA for f16 store; swapping to 1-byte store should reuse the same `CollectiveEpilogue` template with `ElementOut = float_e4m3_t`.
3. **Per-tensor O-scale may clip attention outputs.** Attention outputs after softmax × V have softer dynamic range than raw activations, so per-tensor is usually fine — but measure.

## Success criteria

- N=128 tok/s ≥ 22,500 (+2% floor from current 22,069, or more if stacked with B5).
- Cosine of final logits vs f16-O path ≥ 0.9995 at N=1, 32 tokens.
- Greedy N=1 output matches f16-O for the first 16 tokens.
- compute-sanitizer memcheck clean.

## Files touched

- `kernels/fa3_patches/fp8_o_output.patch` — NEW (~400 LOC)
- `kernels/build_fa3.sh` — apply patch before build
- `kernels/fa3_sm90_wrapper.cu` — new entry point
- `v3/crates/rvllm-attention/src/lib.rs` — new fn pointer type + dlsym
- `v3/crates/rvllm-attention/src/decode.rs` — new launcher
- `v3/crates/rvllm-runtime/src/layer_exec.rs` — swap + delete quantize step
- `v3/crates/rvllm-runtime/src/bring_up.rs` — add o_scale_region scalar
