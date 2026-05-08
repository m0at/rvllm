# Mistral 3.5 Pixtral vision — remaining work

Status as of commit `b2e6147` (loop session):

## Done
- All 438 vision/projector BF16 tensors upload to device on bring-up.
- Image preprocessing (host-side): `vision_preprocess::preprocess_mistral35_pixtral`
  produces `[N_patches, 3*14*14]` f32 patches from raw RGB.
- Arena bumped 80→100 GiB to fit lang weights (~75 GiB) + KV cache
  (1.4 GiB) + vision (~5 GiB).
- Architecture parsed: 48 ViT layers, hidden=1664, head_dim=104,
  intermediate=8192, patch_size=14, image_size=1540, spatial_merge=2,
  image_token_index=10.

## Not done (forward path)

### Phase B — patch embedding (~50 LOC)
Pixtral's `patch_conv [1664, 3, 14, 14]` is mathematically a matmul:
`patches[N, 588] @ patch_conv_flat[1664, 588]^T → out[N, 1664]`.
Cast preprocess output F32 → BF16 host-side, then call
`cublaslt.bf16_gemm_f32` (already used by lm_head). No new kernel.

### Phase C — ViT forward (~400 LOC + integration)
Each of 48 layers: pre-RMSNorm → MHA (no causal mask, n_heads=16,
no GQA so gqa_ratio=1) → +residual → pre-RMSNorm → SwiGLU(gate,up)
→ down → +residual.

Reusable BF16 kernels already wired:
- `rmsnorm_inplace_bf16_gbf16` (attention_norm, ffn_norm)
- `silu_mul_bf16` (SwiGLU)
- `vector_add_bf16` (residuals)

Need new kernels:
- 2D RoPE for Pixtral (per-patch (row, col) angle, NOT the
  text-side YaRN+split-half kernel).
- Full self-attention (m=N_patches, no past KV) — the
  `mistral35_qk_dot_bf16` + `softmax_v_bf16` pair were sized for
  m=1 single-token decode and need an m=N variant. Could write a
  proper flash-attention-style kernel OR reuse the existing
  `flash_attention_2_f16kv_kernel` after BF16↔F16 casts.
- BF16 GEMM for q/k/v/o + gate/up/down (square 1664×1664 and
  rectangular 8192×1664). Use cuBLASLt `bf16_gemm_f32` with a
  cast-down at the end, OR add `bf16_gemm_bf16` to the cublaslt
  module.

### Phase D — patch merger (~30 LOC)
Spatial 2×2 merge: reshape `[N, 1664]` → `[N/4, 4, 1664]` →
`[N/4, 6656]`, then matmul with `patch_merger [1664, 6656]` →
`[N/4, 1664]`. One BF16 GEMM + a reshape (no kernel needed if the
strides are arranged at allocation time).

### Phase E — projector (~50 LOC)
`norm[1664]` → `linear_1[12288, 1664]` → activation (GELU? SiLU?
Mistral's projector activation is unconfirmed; check
config.json#projector_hidden_act) → `linear_2[12288, 12288]` →
text-embedding space.

### Phase F — admission + splice (~150 LOC)
- `cuda_worker` mistral branch: parse `image_url` parts of the
  request (data URI / http(s) fetch), bound the same way the Qwen
  / Gemma path does (`RVLLM_VISION_MAX_*`).
- Tokenize: replace each image's `image_token_index=10` with
  `num_vision_tokens` placeholder copies (= patches after
  spatial_merge).
- Forward: run vision tower per image, store per-image vision
  embeddings.
- In `forward_smoke_q_proj`: between embed_gather and the first
  layer, copy each vision embedding row over the corresponding
  token slot in `h_residual`.

## Validation gaps

The current text path produces `predicted_token=101484 (' Greco')`
deterministically for the chat-templated `"hi"` prompt (364 tokens
through 88 layers + lm_head). Whether that matches HF Mistral 3.5
inference is unverified — needs a side-by-side comparison against
the reference inference (vllm or HF transformers running the same
prompt) to confirm or expose math errors.

Until that validation lands, vision integration risks compounding
any latent text-path bug. Recommended order for the next session:
1. HF reference comparison for text path (~1 token validation).
2. Patch conv + ViT layer 0 forward (1 kernel + 1 BF16 GEMM,
   diff against HF Pixtral layer 0).
3. Stack 48 layers + projector.
4. Admission + splice in handler.
