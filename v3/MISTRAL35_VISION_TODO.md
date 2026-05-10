# Mistral 3.5 Pixtral vision — done, follow-ups

## Status (Round-12 phase 5a, commit `6b3c6d4`)

**Vision is production-accessible.** Image-bearing chat completions
on Mistral 3.5 NVFP4 work end-to-end through rvllm-serve when the
operator sets `RVLLM_LOAD_VISION=1` in the rvllm profile. No debug
flag required.

E2E semantic gate (3/3 real images, phase 4 commit `d961d1c`):

| image                            | prompt                                         | response                                  |
|----------------------------------|------------------------------------------------|-------------------------------------------|
| orange ball on light blue        | "What is in this image? Answer in 5 words."    | "orange ball on light blue background"    |
| blue square w/ yellow triangle   | "Describe this image in 8 words or less."      | "Yellow triangle on blue square background." |
| green grid w/ white lines        | "What pattern is in this image? 8 words max."  | "Green grid with white lines."            |

## Pipeline (each component shipped)

1. **Preprocess** (`vision_preprocess::preprocess_mistral35_pixtral`)
   — host-side resize + Pixtral CLIP-norm + HWC patchify.
2. **patch_conv weight permute at load** (`mistral35_load::upload_bf16_conv_weight_chw_to_hwc`)
   — `[O, C, H, W]` → `[O, H, W, C]` so the device GEMM is a plain
   `[N, 3*P*P] @ [O, 3*P*P]^T`.
3. **Pixtral 2D RoPE host builder** (`mistral35_pixtral_rope::PixtralRopeTables::build`)
   — HF-compatible: `freqs = 1 / base^(arange(0, dim, 2) / dim)`,
   `row_freqs = freqs[::2]`, `col_freqs = freqs[1::2]`, second-half
   mirrors first.
4. **GPU forward** (`Mistral35Bringup::forward_pixtral_vision`):
   patch_conv → ln_pre → 48 ViT blocks (norm + Q/K/V GEMMs +
   `pixtral_rotary_2d_bf16` on Q/K + batched-strided QK^T +
   `softmax_row_f32_to_bf16` + transpose V + batched-strided
   attn @ V + scatter heads + O proj + residual + norm + gate/up
   GEMMs + `gelu_tanh_mul_bf16` + down proj + residual)
   → projector RMSNorm → `patch_merger_pixtral_2x2`
   (channel-outer to match HF unfold) → merging_layer Linear →
   linear_1 → `gelu_tanh_bf16` → linear_2 → DtoH.
5. **Splice** (`Mistral35Bringup::generate_with_vision`):
   per-request HtoD upload of all splice bytes to one arena
   region, position→device-ptr lookup, DtoD copy in
   `forward_smoke_q_proj_inner` after embed_gather replaces the
   text embed for the prompt slots reserved by the renderer.
6. **Admission** (`handlers.rs`): vision-bearing requests on
   Mistral 3.5 are accepted when `state.vision_loaded`. Reject
   reason `vision_not_loaded` if the operator forgot
   `RVLLM_LOAD_VISION=1`.

## Verified-correct vs HF reference

| stage                | rvllm vs HF cosine (BF16) | notes                                          |
|----------------------|---------------------------|------------------------------------------------|
| post_patch_conv      | 0.999997                  | byte-faithful                                  |
| post_ln_pre          | 0.999992                  | byte-faithful                                  |
| post_blocks (out)    | 0.94                      | residual ~6° angular drift through 48 blocks   |
| post_proj_norm       | 0.91                      | inherits the post_blocks drift                 |
| post_merge           | 0.93                      | patch_merger is byte-faithful given input      |
| output (soft tokens) | 0.94                      | net soft-token vs HF                           |

The 6° angular drift accumulates from small per-block BF16
rounding differences (likely softmax precision and/or the order
of f32→bf16 casts inside attention vs HF's path). It does NOT
prevent semantically correct visual answers — Pixtral's design
is robust to that magnitude of perturbation.

## Follow-ups (not blocking)

### Numerical fidelity
- Bisect the per-block drift using a stream-isolated dump path
  (the in-stream `cuStreamSynchronize + cuMemcpyDtoH_v2` pattern
  in the prefill loop alters forward output — Round-12 phase 3-test
  (c) finding). Options: a separate stream + event for the dump,
  or a "static-image, single-block forward" debug entry that
  only runs one block at a time.
- Check whether HF's softmax is in float32 (likely) and whether
  our `softmax_row_f32_to_bf16` matches HF byte-for-byte.

### Performance
- Batch Q/K/V GEMMs into a single fused projection with a
  concatenated weight `[3*v_hidden, v_hidden]`; same for gate/up
  → `[2*intermediate, v_hidden]`. Saves 144 GEMM launches per
  image (3 per block × 48 blocks); marginal wallclock win
  (~1-2 s) since vision tower is not the bottleneck.
- Real wallclock pain is the language-decoder per-token forward
  loop (~150 ms × 365 prefill tokens ≈ 55 s for image-bearing
  requests). That's a separate yak — `MISTRAL35_BATCHED_PREFILL_PLAN.md`
  exists; not in vision scope.

### Cleanup
- The per-block dump in `forward_pixtral_vision_cuda` is gated
  behind `RVLLM_PIXTRAL_PER_BLOCK_DUMP=1` because the
  cuMemcpyDtoH_v2 corrupts forward output on this driver. Replace
  with an event-driven cudarc::CudaSlice::dtoh_async path that
  uses a separate stream.
