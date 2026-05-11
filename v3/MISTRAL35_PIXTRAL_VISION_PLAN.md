# Mistral 3.5 Pixtral vision GPU forward — strategy doc

**Status:** design / scaffolding (Step 7 GPU half of `mistral-35-
integration.md`).
**Goal:** map the Pixtral ViT forward onto rvllm's existing
`v3/crates/rvllm-runtime` vision infrastructure, identify the
exact kernel reuse vs new-variant boundary, and document the
splice into `mistral_layer_step` so the GPU implementation has a
clear plan when it lands.

## Pixtral arch (from `Mistral35Arch::vision`)

| Field | Value | Notes |
|---|---|---|
| `model_type` | `pixtral` | parsed + asserted at startup |
| `num_hidden_layers` | 48 | the deepest of the three vision towers we serve |
| `hidden_size` | 1664 | bigger than Gemma 4 (1152), Qwen 3.6 (1280) |
| `num_attention_heads` | 16 | divides `hidden_size` evenly: `head_dim = 1664 / 16 = 104` |
| `head_dim` | **104** | NOT 64/72/128 — Pixtral-specific, register pressure check needed |
| `intermediate_size` | 8192 | MLP up-proj |
| `patch_size` | 14 | grid-aligned: every input must round to a multiple of `patch_size * spatial_merge_size = 28` |
| `image_size` | 1540 | longest-edge resize target |
| `num_channels` | 3 | RGB |
| `rope_theta` | 10000.0 | standard |
| `spatial_merge_size` | 2 | `merged_h = grid_h / 2`, `merged_w = grid_w / 2` |

Host preprocess (`preprocess_mistral35_pixtral`, already landed)
produces `[num_patches, 3 * 14 * 14]` f32 row-major + an
`(num_h, num_w)` patch grid. The GPU forward picks up from there.

## Per-block forward sequence

```
0. patch_conv:  [num_patches, 3*14*14] · weight[1664, 3*14*14]
                 -> [num_patches, 1664]                             (1 GEMM)
1. pre-norm:    rms_norm                                            (existing kernel)
2. for block in 0..48:
     2a. norm1
     2b. q/k/v = x · w_qkv                                          (3 GEMMs)
     2c. apply_rope_pixtral_2d on Q + K  (per-head_dim=104)         <- NEW kernel
     2d. flash_attention_2_unified_prefill (BF16 KV)                (existing FA2 path)
     2e. attn_out · w_o                                             (1 GEMM)
     2f. residual1
     2g. norm2
     2h. ff_in · w_gate || w_up
     2i. gelu_tanh_mul                                              (existing kernel)
     2j. ff_out · w_down
     2k. residual2
3. patch_merger:  [grid_h, grid_w, 1664] -> [merged_h * merged_w, 6656]
                  (concat 2x2 spatial neighbours)                   <- NEW kernel
4. projector:
     4a. projector_norm: rms_norm                                   (existing kernel)
     4b. linear_1: [merged_tokens, 6656] · [6656, 12288] -> bf16    (1 GEMM)
     4c. gelu_tanh                                                  (existing kernel)
     4d. linear_2: [merged_tokens, 12288] · [12288, 12288] -> bf16  (1 GEMM)
5. splice into prefill embed buffer at slot.token_start             (existing splice)
```

## Kernel reuse map

### Reusable as-is from the Qwen 3.6 / Gemma 4 vision paths

| Kernel | Path | Notes |
|---|---|---|
| `rmsnorm_inplace_*` | `kernels/rmsnorm_*` | f16 + bf16 variants. Pixtral norm is RMSNorm (same as Mistral text decoder). |
| `gelu_tanh_*`, `gelu_tanh_mul_*` | `kernels/gelu_tanh*.cu` | Pixtral MLP activation matches Gemma 4's. |
| `silu_mul_*` | not used — Pixtral uses GeLU, not SiLU | (decoder side uses SiLU) |
| `softmax_row_*` | `kernels/softmax_row_*.cu` | Per-row softmax inside FA2; reused. |
| `extract_head_*`, `scatter_heads_*` | `kernels/extract_head_*.cu`, `kernels/scatter_heads_*.cu` | Per-head Q/K/V split + recombine. Generic in shape. |
| `vit_avgpool_*` | `kernels/vit_avgpool*.cu` | Used by the patch merger pool path. |
| `vit_standardize_*` | `kernels/vit_standardize*.cu` | Pixtral CLIP-style normalisation done host-side already. Not needed on GPU. |
| `f16_gemm_f32_batched_strided` / `bf16_*` | `crates/rvllm-cutlass/src/cublaslt.rs` | All ViT GEMMs are dense BF16 / F16 — no NVFP4 here. |
| Vision splice | `qwen36_bring_up::forward_qwen36_decode` / `gemma4_bring_up::run_generate` | Mistral text decoder reuses the same per-(seq_idx, slot) splice mechanism. |

### Need new variants (Pixtral-specific)

| Kernel | Why | Plan |
|---|---|---|
| **`vit_rotary_pixtral_2d_*`** | Existing `vit_rotary_2d_f16.cu` (Qwen) and `vit_rotary_gemma4_2d_*.cu` are head_dim=64/128. Pixtral's `head_dim = 104` means `head_dim/2 = 52` element pairs per head, NOT a power of 2. The kernel's smem layout + warp-level shuffle pattern likely assumes `head_dim/2 % 32 == 0`. | New variant `kernels/vit_rotary_pixtral_2d_bf16.cu` (and f16 sibling). Loop over `head_dim/2 = 52` sin/cos pairs explicitly; no warp-shuffle reduction. Register impact: 52 f32 sin + 52 f32 cos = ~13 lanes × 8 regs/thread, fits comfortably. |
| **`patch_merger_2x2`** | Pixtral merges 2×2 spatial-neighbour patches into one `4 * 1664 = 6656` token. Existing Qwen `PatchMerger` does a similar concat but with different spatial-merge size + token shape. | New variant `kernels/patch_merger_pixtral_2x2.cu`. Grid: `(merged_h * merged_w, 1, 1)`. Block: 256 threads. Each thread copies one `(c, src_y, src_x)` triple from the 4-sample neighbourhood into the merged output. Pure permutation + concat. |

### Drop entirely (no Pixtral analogue)

- `vit_pos_emb_lookup_2d_*`: Pixtral uses 2D RoPE rather than learned 2D position embeddings.
- `vit_pos_embed_interp_f16.cu` (Qwen bilinear): no learned-table interpolation needed.

## Activation dtype convention

All Pixtral ViT activations stay in BF16 throughout the 48-block
stack. The decoder side runs NVFP4 on its projections; the vision
side keeps BF16 because the ViT weights ship BF16 (validated by
`Mistral35WeightInventory`'s 434 vision BF16 + 4 projector BF16
counts). No NVFP4 dequant kernels needed in the vision path.

## Splice integration

`forward_mistral35_vision(image_bytes) -> VisionForwardOutput`:

```rust
struct VisionForwardOutput {
    /// `[num_image_tokens, hidden_size = 12288]` BF16 device buffer.
    /// Length = `merged_h * merged_w` from preprocess.
    pub data: u64,            // device pointer
    pub num_tokens: usize,
}
```

The decoder's prefill embed step copies this buffer in at
`slot.token_start * row_bytes` after the embedding gather and
before any RMSNorm / FA2 launch (mirrors the Qwen / Gemma path).
Vision-bearing requests force `common_prefix_len = 0` and the
chunked-prefill batch path; F16-KV is incompatible with vision
on existing paths and the same restriction applies to Mistral.

## Scratch budget per image

Per image at the production max (`image_size = 1540`):

```
patches      = (1540/14)² = 110² = 12 100   (after factor-28 round
                                             to 1540×1540: 110×110)
merged       = 12 100 / 4 = 3 025 tokens
hidden BF16  = 12 100 × 1664 × 2 = ~38 MiB working set
attn scores  = 12 100² × 16 × 4 = ~9.4 GiB f32 (per-head; reduced
                                                via FA2 streaming)
```

`RVLLM_VISION_MAX_TOTAL_TOKENS` (default 8192 in
`MISTRAL35_PROFILE_TEMPLATE.env`) caps the merged-token budget so
a single max-resolution image (3025 tokens) leaves room for ~2.7
images per request before admission rejects.

The 9.4 GiB raw attention-scores upper bound is purely a
worst-case sanity number — FA2 streams in tile-sized chunks
(~64 rows × full-N × 16 heads × 4 bytes = ~12 MiB scratch live at
any moment), which fits comfortably in the existing arena.

## Bench expectations

Pixtral ViT at the production input shape:

* Patches → 12 100 tokens (1540×1540 input)
* 48 blocks × ~50 GEMMs / block (FA2 inline + 4 projection GEMMs)
  = ~2 400 launches per image
* Wall-clock target on GB10: < 800 ms for one max-res image.

## Open questions deferred to GPU-iteration phase

1. **head_dim=104 register pressure.** Pixtral's odd head_dim
   means `q_reg[FA2_BC][104]` consumes 832 bytes/thread of
   register space. FA2's existing path likely allocates
   `q_reg[FA2_BC][128]` worst-case; verify it isn't padded to 128
   silently (which would waste 24/128 = 19% of the register
   budget per element) before bench.
2. **Pixtral 2D RoPE convention.** Mistral's reference impl uses
   `freqs = (theta ** -i/d for i in 0..d/2)` × spatial coords.
   Validate the exact formula against the HF Pixtral processor
   numpy ref before sealing the new `vit_rotary_pixtral_2d`
   kernel.
3. **Patch merger spatial order.** Does Pixtral concat in
   `[c, h, w]`, `[h, w, c]`, or some swizzle? Cosine vs HF must
   gate before this is wired into a forward.

## Implementation order

When the device-side work begins:

1. **patch_conv**: 1 BF16 GEMM, no new kernels needed (existing
   cublaslt path covers it).
2. **vit_rotary_pixtral_2d_bf16.cu**: small new kernel.
3. **single-block forward**: norm1 → QKV → RoPE → FA2 → O →
   residual → norm2 → gate/up → gelu_mul → down → residual.
4. **patch_merger_pixtral_2x2.cu**: small new kernel.
5. **projector**: 2 BF16 GEMMs + RMSNorm + GeLU.
6. **forward_mistral35_vision**: composes 1-5 + splice.

Each step has a CPU reference (in
`rvllm-runtime::vision_preprocess` for host preprocess; the
ViT-side reference would extend `mistral35_layer_ref.rs` if a
deeper validation harness is needed) plus a numpy fixture diff
gate via the existing `cmp_g4v_substep.py` template.
