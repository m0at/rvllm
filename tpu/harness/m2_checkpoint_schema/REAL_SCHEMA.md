# MiniMax-M2.7-NVFP4 actual checkpoint schema (lukealonso variant)

Ground truth extracted from `lukealonso/MiniMax-M2.7-NVFP4` on HuggingFace hub.
191,069 total tensors mapped into shards.

## Canonical tensor names (26 unique, dedup by layer-id and expert-id)

### Backbone (bf16, NOT quantized)
- `model.embed_tokens.weight` — vocab x hidden
- `model.norm.weight` — hidden
- `lm_head.weight` — vocab x hidden

### Per layer (62 layers, bf16, NOT quantized)
- `model.layers.L.input_layernorm.weight` — hidden
- `model.layers.L.post_attention_layernorm.weight` — hidden
- `model.layers.L.self_attn.q_proj.weight` — (NH * HEAD_DIM) x hidden  (48*128=6144 x 3072)
- `model.layers.L.self_attn.k_proj.weight` — (NKV * HEAD_DIM) x hidden (8*128=1024 x 3072)
- `model.layers.L.self_attn.v_proj.weight` — (NKV * HEAD_DIM) x hidden (8*128=1024 x 3072)
- `model.layers.L.self_attn.o_proj.weight` — hidden x (NH * HEAD_DIM)  (3072 x 6144)
- `model.layers.L.self_attn.q_norm.weight` — HEAD_DIM (per-layer, 128)
- `model.layers.L.self_attn.k_norm.weight` — HEAD_DIM (per-layer, 128)

### MoE router (bf16)
- `model.layers.L.block_sparse_moe.gate.weight` — NUM_EXPERTS x hidden (256 x 3072)
- `model.layers.L.block_sparse_moe.e_score_correction_bias` — NUM_EXPERTS (256)

Note the `.gate.` wraps only the gate projection. The bias sits at
`block_sparse_moe.e_score_correction_bias`, NOT at `block_sparse_moe.gate.e_score_correction_bias`.

### Per-expert (62 layers * 256 experts = 15,872 experts, NVFP4 quantized)

Each expert has THREE projections using SwiGLU: `w1=gate`, `w3=up`, `w2=down`.
They are STORED AS SEPARATE TENSORS (not fused `gate_up`).

For each of {w1, w2, w3}, the modelopt NVFP4 format stores **4 tensors**:
- `...experts.N.wX.weight` — packed uint8, 2 FP4 values per byte (E2M1 signed)
- `...experts.N.wX.weight_scale` — per-16-element block scale, FP8 E4M3 (one byte per 16 weights)
- `...experts.N.wX.weight_scale_2` — **per-tensor global scale**, FP32 scalar (NEW, agents missed this)
- `...experts.N.wX.input_scale` — per-tensor static activation scale, FP32 scalar

**Dequant formula (modelopt NVFP4 two-level):**
```
decoded_bf16 = fp4_e2m1_lut[nibble] * fp8_e4m3_block_scale * global_scale_f32
```

Shapes for layer L, expert N:
- `w1.weight`: (MOE_INTER, hidden) = (1536, 3072) → packed uint8 shape (1536, 1536) = 1536 * 3072 / 2 bytes per row
- `w3.weight`: same as w1
- `w2.weight`: (hidden, MOE_INTER) = (3072, 1536) → packed uint8 shape (3072, 768)
- For each, `weight_scale` shape is (rows, cols / 16)
- `weight_scale_2` and `input_scale` are scalar FP32 per tensor

### NOT present in this checkpoint
- No MTP tensors (`model.mtp_modules.M.*`). Despite config `use_mtp=True`, the repo has no MTP weights. Treat `use_mtp` as False at runtime until weights materialize.
- No shared expert. Matches config `shared_intermediate_size=0`.
- No sliding window. Matches config `sliding_window=null`.

## Impact on rvllm code

`tpu/harness/nvfp4_loader.py` needs:
1. Recognize modelopt-style NVFP4: four tensors (`weight`, `weight_scale`, `weight_scale_2`, `input_scale`) instead of assumed (`weight_packed`, `weight_scale`).
2. Add a FP32 `global_scale` field to `NvFp4Tensor`.
3. `dequant_nvfp4_to_bf16_cpu` and `..._int8_cpu` must multiply by both block scale AND global scale.

`tpu/harness/nvfp4_jax_ops.py` needs:
1. `nvfp4_to_bf16_jax` and `nvfp4_matmul` accept `global_scale: jnp.ndarray (scalar)` as an extra arg.
2. Fold the global scale into the dequant multiply.

`tpu/harness/m2_tpu_infer.py` needs:
1. Fix all tensor names to match above. Notably: router bias = `block_sparse_moe.e_score_correction_bias` (no `.gate.`).
2. Expert loader should read `w1`, `w2`, `w3` as separate NVFP4 tensors; skip `input_scale` for now (only needed for static activation quant — path A uses bf16 activations).
3. Set `USE_MTP=False` unconditionally, regardless of config (real checkpoint has no MTP).

`tpu/harness/m2_moe.py` needs:
1. `moe_block_nvfp4` signature should take three pytrees: `(w1_packed, w1_scale, w1_scale2)`, `(w2_*)`, `(w3_*)` rather than fused `gate_up`.
2. Expert FFN becomes:
     gate = nvfp4_matmul(x, w1_packed, w1_scale, w1_scale2, MOE_INTER, hidden)
     up   = nvfp4_matmul(x, w3_packed, w3_scale, w3_scale2, MOE_INTER, hidden)
     h    = silu(gate) * up
     out  = nvfp4_matmul(h, w2_packed, w2_scale, w2_scale2, hidden, MOE_INTER)

## Non-changes
- Attention backbone (`m2_attention.py`): tensor names match. No change.
- Mesh (`m2_mesh.py`), KV cache (`m2_kv_cache.py`): no change.
- Chat template (`m2_chat.py`): no change.
- Deploy script: no change.
