# Gemma 4 Vision Implementation Plan (Phase 3b)

Status as of session end (2026-05-04):

✅ **Phase 3a (loader, commit `75daff9`)** — `Gemma4Vision` struct + 27 blocks
+ patch_embedder + position_embedding_table [2, 10240, 1152] +
embed_vision projection loaded via `load_gemma_vision()`.

❌ **Phase 3b (forward, this doc)** — NOT STARTED.

❌ **Phase 3c (cuda_worker wiring + E2E)** — NOT STARTED.

## HF Reference (transformers Gemma4VisionModel)

Verified via `transformers/models/gemma4/modeling_gemma4.py`:

### Patch embedder
```
input_proj: Linear [hidden=1152, 768] no bias
position_embedding_table: Parameter [2, 10240, 1152]   # (axis row/col, pos, hidden)
```
forward:
1. `pixel = 2 * (pixel - 0.5)` — pre-normalize
2. `hidden = input_proj(pixel)` → [N, 1152]
3. position embedding via one-hot lookup:
   ```
   one_hot[pos]  in [N, 2, 10240]  (per-axis row/col one-hot)
   pos_emb = einsum("npa, apl -> npl", one_hot, table)   # [N, 2, 1152]
   pos_emb = pos_emb.sum(dim=1)                          # [N, 1152]
   # zero out padding positions (we won't have padding for single image)
   ```
4. `hidden = hidden + pos_emb`

### Block (sandwich norm — different from Qwen)
```
res = x
x = input_layernorm(x)        # RMSNorm, Gemma +1 shift
x = self_attn(x, cos, sin)    # see below
x = post_attention_layernorm(x)
x = res + x                   # residual

res = x
x = pre_feedforward_layernorm(x)
x = mlp(x)                    # SwiGLU: down(silu(gate(x)) * up(x))
x = post_feedforward_layernorm(x)
x = res + x
```

### Attention (separate q_proj/k_proj/v_proj, NOT fused QKV)
- num_heads=16, head_dim=72 (= same as Qwen ViT)
- q_proj/k_proj/v_proj: Linear [hidden, num_heads*head_dim], NO bias
- q_norm: RMSNorm head_dim, Gemma +1 shift
- k_norm: RMSNorm head_dim, Gemma +1 shift
- v_norm: RMSNorm head_dim, **with_scale=False** (just rsqrt + 1, no gamma multiply!)
- scaling=**1.0** (no 1/sqrt(d) — already baked into norms)
- multidim rotary applied to Q and K
- Standard attention with `eager_attention_forward` (= Q*K^T → softmax → @V)
- o_proj: Linear [num_heads*head_dim, hidden], NO bias

### MLP (SwiGLU, NOT 2-layer with GELU like Qwen)
- gate_proj: Linear [hidden=1152, intermediate=4304] no bias
- up_proj:   same shape
- down_proj: Linear [4304, 1152] no bias
- act_fn = `gelu_pytorch_tanh` (NOT silu — see config `hidden_activation`)
- forward: `down(act_fn(gate(x)) * up(x))`

WAIT — Gemma 4 vision config typically has `hidden_activation='gelu_pytorch_tanh'`,
but the formula `down(act_fn(gate) * up)` is SwiGLU pattern with GELU
substituted for SiLU. Verify by reading config when implementing.

### Pooler (`Gemma4VisionPooler`)
- `output_length=256` (default, configurable)
- avg_pool with `k = sqrt(input_seq_len // output_length)` (typically k=2 or 3)
- Then `hidden *= sqrt(hidden_size)` ← Gemma scaling

### embed_vision (post-encoder projector to text-hidden)
- Linear [1152, 5376] = projects to text hidden
- + RMSNorm[5376]

## Implementation steps

### Step 1: Add kernel handles to `Gemma4FusedModules`
Need:
- `layernorm_inplace_f16` (already exists in kernels — need to load)
- `softmax_row_f16` (exists — need to load)
- `silu_mul_f16` (exists)
- `vit_avgpool_f16` (exists)
- `transpose_2d_f16` (NEW from Qwen vision fix)
- `scale_inplace_f16` (NEW from Qwen vision fix)
- `add_bias_f16` (exists)
- `gelu_tanh_f16` (exists)

Mirror the loader pattern from Qwen36OutsideKernels (qwen36_bring_up.rs).

### Step 2: New kernel — 2D position-embedding lookup
File: `kernels/vit_pos_emb_lookup_f16.cu`

```cuda
// Add 2D position embedding to hidden_states.
// pos_table: [2, num_pos, hidden] (row table + col table)
// row_pos[N], col_pos[N] = per-token row/col indices in [0, num_pos)
// hidden_states[N, hidden] += pos_table[0, row_pos[i], :] + pos_table[1, col_pos[i], :]
```

Or alternatively: do as 2 separate add_bias-style kernels (one for row, one
for col), reusing existing `add_bias_f16`-style logic with index lookup.

### Step 3: forward_gemma_vision (~600 LOC)
File: `crates/rvllm-runtime/src/gemma4_bring_up.rs`

```rust
pub fn forward_gemma_vision(&self, image_bytes: &[u8]) -> Result<VisionForwardOutput> {
    // 1. Preprocess (Rust: vision_preprocess::preprocess_gemma)
    // 2. Upload patches [N, 768] f16 to device
    // 3. patch_embed: linear (no bias) [768→1152]
    // 4. Add 2D position embeddings (2 lookups: row + col)
    // 5. 27 blocks: sandwich-norm (input_ln → attn → post_attn_ln → +res
    //               then pre_ffw_ln → mlp → post_ffw_ln → +res)
    //    - attn: q_proj/k_proj/v_proj (separate); q_norm/k_norm/v_norm;
    //      multidim rotary; standard attn; o_proj
    //    - mlp: gate_proj/up_proj → act → mul → down_proj
    // 6. Avg-pool to ≤280 tokens
    // 7. Multiply by sqrt(hidden_size=1152)
    // 8. embed_vision linear [1152→5376] + RMSNorm
    // 9. DtoH and return
}
```

### Step 4: Wire to cuda_worker (gemma4 path)
Mirror what Qwen does at `cuda_worker.rs:138-187`:
- collect_vision_items
- For each: forward_gemma_vision, store device-side embedding
- Pass `vision_splice: Vec<(usize, &[u8])>` to gemma's `run_generate`

### Step 5: Update Gemma chat template handling for image tokens
Gemma uses BOI (begin-of-image) + image-token×N + EOI (end-of-image)
markers. Tokenizer-aware. Check `tokenize.rs` for Gemma path.

### Step 6: Test E2E with HF dump comparison
Use `/tmp/qwen_vision_dump_stages.py` as template, swap to
`Gemma4VisionModel`. Compare layer-by-layer like we did for Qwen.

## Estimated effort
~700 LOC new code + 1 new kernel. 4-6 hours focused work
+ debugging.

## Reference commits (Qwen vision foundation)
- `9f0ea52` — V-transpose fix (GEMM input @ weight^T semantics)
- `10f6e0d` — partial-rotary + pos_embed bilinear interp
- `24b9651` — attention scale 1/sqrt(d)
- `bac5788` — preprocess min_pixels fix

The same pattern (HF dump → stage cmp → fix divergence) works for
Gemma; expect similar bug surface (GEMM layouts, normalization
shifts, rotary semantics).
