# Spec 3: Model Architectures

## Summary

rvLLM supports 10 causal LM architecture groups plus embedding models (23 registered architecture strings total mapping to 12 distinct structs). vLLM supports ~260. This spec identifies the highest-priority models to add based on popularity and implementation complexity, and provides per-model architectural analysis against the existing rvLLM layer library.

rvLLM's architecture pattern is well-established: implement the `Architecture` trait (`pub trait Architecture: Send + Sync` with a single `forward()` method returning `Result<GpuBuffer<f32>>`), register the HuggingFace architecture string in `create_model()`, and reuse shared layers (RMSNorm, LinearLayer, RotaryEmbedding, MoELayer, etc.).

## Existing rvLLM Models

| Architecture String | File | Notes |
|---|---|---|
| `LlamaForCausalLM` | `llama.rs` | Reference implementation |
| `MistralForCausalLM` | `mistral.rs` | Llama variant with sliding window |
| `Qwen2ForCausalLM` | `qwen2.rs` | Llama variant with optional QKV bias |
| `GemmaForCausalLM` | `gemma.rs` | +1 norm offset, GeGLU, embedding scaling |
| `Gemma2ForCausalLM` | `gemma.rs` | Gemma + logit softcap + sliding window alternation (even layers, currently computed but unused in forward pass) |
| `PhiForCausalLM` / `Phi3ForCausalLM` / `Phi3SmallForCausalLM` | `phi.rs` | Parallel attn+MLP (Phi2), sequential (Phi3) with QK LayerNorm |
| `MixtralForCausalLM` | `mixtral.rs` | Sparse MoE |
| `DeepSeekV2ForCausalLM` / `DeepseekV2ForCausalLM` | `deepseek.rs` | MLA + MoE + shared expert |
| `CohereForCausalLM` | `cohere.rs` | LayerNorm, MQA, per-head QK normalization, parallel attn+MLP |
| `GPTNeoXForCausalLM` / `StableLmForCausalLM` | `gpt_neox.rs` | Parallel attn+MLP, LayerNorm, GELU |
| `BertModel`, `RobertaModel`, etc. (8 strings) | `embedding.rs` | Embedding models (non-causal) |

### Config Mechanism Note

`ModelRunnerConfig` has generic fields (`num_layers`, `hidden_size`, `num_heads`, `num_kv_heads`, `head_dim`, `intermediate_size`, `vocab_size`, `max_position`, `rope_theta`, `dtype`, `architecture`). Model-specific config values (e.g., MoE expert count, QK-Norm flags) are not in the struct. Models like DeepSeek work around this by hardcoding defaults in `new()`. New models will either need config extension or per-architecture config parsing from the HuggingFace `config.json`.

## Priority Models to Add

### Priority 1: Qwen3 (`Qwen3ForCausalLM`)

**Difficulty**: Low (~200 lines)
**Why**: Qwen3 (0.6B through 235B) is one of the most widely deployed model families.

**Architectural differences from Qwen2 (which rvLLM already has)**:

| Feature | Qwen2 | Qwen3 |
|---|---|---|
| QKV bias | Optional (per config) | Optional (per config) |
| QK-Norm | None | **RMSNorm on Q and K per-head** |
| MLP | SiLU gate/up/down | Same |
| Norm | RMSNorm | Same |
| RoPE theta | Model-dependent | Default 1,000,000 |

**The only structural change is QK-Norm**: after Q and K projections, reshape to `[num_tokens, num_heads, head_dim]`, apply RMSNorm independently per head, then reshape back. This is 2 extra RMSNorm calls per layer.

**Implementation**:
1. Copy `qwen2.rs` to `qwen3.rs`
2. Add two weight buffers per layer: `q_norm_weight: GpuBuffer<f16>` and `k_norm_weight: GpuBuffer<f16>` (shape: `[head_dim]`)
3. Add QK-Norm: after Q/K projection, before RoPE. Note: the existing `RMSNorm::forward()` takes `(&GpuBuffer, &GpuBuffer, f32)` and returns a new `Result<GpuBuffer<f16>>`. Per-head QK-Norm requires a new helper (see "New Shared Components" below) since the existing `RMSNorm` operates on the full hidden dimension, not per-head:
   ```rust
   // q shape: [num_tokens * num_heads * head_dim]
   // Apply RMSNorm per head (treat as [num_tokens * num_heads, head_dim])
   // This needs a new qk_norm_forward() helper or a CUDA kernel for efficiency
   q = qk_norm_forward(&q, &q_norm_weight, head_dim, eps)?;
   k = qk_norm_forward(&k, &k_norm_weight, head_dim, eps)?;
   ```
4. Load `q_norm.weight` and `k_norm.weight` from safetensors (per-layer)
5. Register `"Qwen3ForCausalLM"` in `mod.rs`

**vLLM reference**: `vllm/model_executor/models/qwen3.py` -- inherits Qwen2, only replaces attention module to add QK-Norm.

**Follow-up**: `Qwen3MoeForCausalLM` uses the existing `MoELayer` pattern with a shared expert.

---

### Priority 2: Gemma3 (`Gemma3ForCausalLM`)

**Difficulty**: Medium (~350 lines)
**Why**: Google's latest Gemma family, widely used for fine-tuning.

**Architectural differences from Gemma2 (which rvLLM already has)**:

| Feature | Gemma2 | Gemma3 |
|---|---|---|
| Attn logit softcap | 50.0 | **Removed** |
| Final logit softcap | 30.0 | Configurable (may be 0) |
| QK-Norm | None | **GemmaRMSNorm on Q and K per-head** (with +1 offset) |
| RoPE theta | Same for all layers | **Different theta for local vs global layers** |
| Weight tying | Always tied | Configurable via `tie_word_embeddings` |
| Sliding window | Alternating layers | Configured via `layer_types` array from config |

**New requirements**:
1. **QK-Norm with Gemma's +1 offset**: Same as Qwen3 QK-Norm but using `(1 + weight) * normalized_x` instead of `weight * normalized_x`. Note: `GemmaRMSNorm` is currently a **private** struct inside `gemma.rs`. It needs to be made `pub` or extracted to `layers/norm.rs` for reuse by Gemma3.
2. **Per-layer RoPE theta**: Local (sliding window) layers use `rope_local_base_freq`, global layers use `rope_theta`. Use `RotaryEmbedding::new(head_dim, max_seq_len, base)` (the instance method, which caches cos/sin tables) — NOT the static `forward()` which hardcodes `base: 10000.0`. Create two persistent instances with different `base` values.
3. **Read `layer_types` from config**: Replace even/odd heuristic with the actual config array. Requires parsing `layer_types` and `rope_local_base_freq` from the HuggingFace `config.json` (not currently in `ModelRunnerConfig`).

**Implementation**:
1. Copy `gemma.rs` Gemma2 implementation to a new `Gemma3ForCausalLM`
2. Remove attention logit softcapping (`attn_logit_softcap`)
3. Extract `GemmaRMSNorm` to `layers/norm.rs` (or make it `pub`) for QK-Norm reuse
4. Add QK-Norm with Gemma's +1 offset variant per head
5. Create two `RotaryEmbedding::new()` instances (local + global) with different `base` params, select per layer based on `layer_types`
6. Parse `layer_types`, `rope_local_base_freq`, `tie_word_embeddings` from model config JSON
7. Register `"Gemma3ForCausalLM"` in `mod.rs`

**vLLM reference**: `vllm/model_executor/models/gemma3.py`

**Note**: Gemma3 also has a multimodal variant (`Gemma3ForConditionalGeneration`) with a vision encoder. This spec covers text-only.

---

### Priority 3: Llama4 (`Llama4ForCausalLM`)

**Difficulty**: High (~600-800 lines)
**Why**: Meta's latest model family with significant architectural innovations.

**Architectural differences from Llama (which rvLLM already has)**:

| Feature | Llama | Llama4 |
|---|---|---|
| RoPE | All layers | **Conditional**: `no_rope_layers` array determines which layers get RoPE |
| QK-Norm | None | **RMSNorm on Q/K** (only on RoPE layers, no learnable weight) |
| Attention | Standard | **Chunked local attention** on RoPE layers (`attention_chunk_size`) |
| Temperature | None | **Position-dependent attention temperature** on NoPE layers |
| FFN | Dense MLP (all layers) | **Interleaved MoE**: every Nth layer uses MoE (configurable step) |
| MoE routing | N/A | **Sigmoid routing** (not softmax) |
| MoE shared expert | N/A | Always-on shared expert alongside routed experts |

**New requirements**:

1. **Conditional RoPE**: Per-layer boolean array. Some layers skip RoPE entirely ("NoPE" layers).
2. **QK-Norm (no weights)**: RMSNorm on Q/K without learnable parameters -- just normalization. Only on RoPE layers.
3. **Chunked local attention**: A new attention variant where RoPE layers attend only within a fixed chunk window (`attention_chunk_size`). This is distinct from sliding window -- it's a hard boundary, not a sliding one.
4. **Attention temperature tuning**: On NoPE layers, Q is scaled by `floor_scale = floor((position+1) / 8192)`, `scale = ln(floor_scale + 1) * 0.1 + 1`. This requires new logic in the forward pass.
5. **Sigmoid MoE routing**: `router_scores = sigmoid(top_k_logits)` instead of `softmax(top_k_logits)`. The existing `MoELayer` uses softmax; needs a routing mode flag.
6. **Interleaved MoE/dense layers**: Every `interleave_moe_layer_step` layers uses MoE FFN; others use dense MLP.
7. **Weight permutation for rotary**: Q/K weights from HuggingFace use interleaved layout; must be permuted to sequential during loading.

**Implementation**:
1. Create `crates/rvllm-model-runner/src/architectures/llama4.rs`
2. Each layer struct tracks: `is_rope_layer: bool`, `is_moe_layer: bool`
3. Forward pass per layer:
   - RMSNorm
   - Q/K/V projection
   - If RoPE layer: QK-Norm, RoPE, chunked local attention
   - If NoPE layer: temperature-scaled attention (standard attention with Q scaling)
   - Output projection, residual
   - RMSNorm
   - If MoE layer: MoE FFN (with sigmoid routing + shared expert)
   - If dense layer: standard SiLU MLP
   - Residual
4. Add `routing_mode: RoutingMode` enum to `MoELayer` (Softmax vs Sigmoid)
5. Implement chunked local attention (mask builder that zeros out cross-chunk attention)
6. Register `"Llama4ForCausalLM"` in `mod.rs`

**vLLM reference**: `vllm/model_executor/models/llama4.py`

**Warning**: This is the most complex model. The chunked local attention and attention temperature tuning have no precedent in the existing rvLLM codebase. Consider implementing after Qwen3 and Gemma3 are solid.

**Note**: `rvllm-attention/src/sliding_window.rs` has `SlidingWindowConfig` with `layer_mask: Option<Vec<bool>>` for per-layer behavior, which could serve as a reference for the chunked local attention implementation.

---

## Existing Layer Library Reuse

| Layer | Qwen3 | Gemma3 | Llama4 |
|---|---|---|---|
| RMSNorm | Reuse | Reuse (+1 offset variant exists) | Reuse |
| LinearLayer | Reuse | Reuse | Reuse |
| RotaryEmbedding | Reuse | Need per-layer theta | Need conditional enable |
| SiLU MLP | Reuse | N/A (uses GeGLU) | Reuse for dense layers |
| GeGLU MLP | N/A | Reuse from Gemma2 | N/A |
| MoELayer | N/A | N/A | Reuse + add sigmoid routing |
| Attention | Reuse | Reuse | Need chunked local variant |

## New Shared Components Needed

1. **QK-Norm helper** (used by Qwen3, Gemma3, Llama4): Per-head RMSNorm on Q and K tensors. Should be a shared utility in `layers/norm.rs` since 3 models need it. Requires extracting `GemmaRMSNorm` (currently private in `gemma.rs`) for the +1 offset variant, and adding a standard variant for Qwen3/Llama4. The helper should operate on `[num_tokens * num_heads, head_dim]` shaped tensors.
2. **Configurable MoE routing** (used by Llama4): Add `RoutingMode::Sigmoid` to `MoELayer`. Note: the existing `MoELayer.renormalize: bool` field is dead code (softmax on top-k already normalizes to sum=1). Clarify renormalization semantics when adding sigmoid mode, where sum ≠ 1 by default.
3. **Chunked local attention** (used by Llama4): New attention mask builder in `rvllm-attention`. Reference `SlidingWindowConfig` in `sliding_window.rs` for per-layer behavior patterns.

**Note on `AttentionBackend`**: There are two trait definitions — the real one in `crates/rvllm-attention/src/backend.rs` (cache-based: `query, key_cache, value_cache, block_tables, ...`) and a simplified bridge in `crates/rvllm-model-runner/src/bridge.rs` (used by architecture forward passes: `query, key, value, attention_metadata, layer_idx`). Chunked local attention for Llama4 may need implementation at both layers.

## Testing Strategy

For each new model:
1. **Smoke test**: Load model weights (mock or real), run a single forward pass, verify output shape.
2. **Coherency**: Generate text from 5 diverse prompts, verify output is coherent English (not degenerate repetition).
3. **Parity check** (if practical): Compare token-by-token output against Python vLLM for the same model and prompt with greedy decoding (`temperature=0`). This is the gold standard but requires both servers running.
4. **Weight loading**: Verify all expected weight tensors are loaded and none are missing (log warnings for unrecognized weights).

## Implementation Order

```
1. Extract GemmaRMSNorm to layers/norm.rs, add standard QK-Norm helper (shared dependency)
2. Qwen3                              (simplest, validates QK-Norm)
3. Gemma3                             (validates per-layer RoPE theta + Gemma QK-Norm)
4. MoE sigmoid routing in layers/moe.rs (shared dependency for Llama4)
5. Llama4                             (most complex, builds on everything above)
```

**Line estimates** (implementation + tests, given ~100 test lines per existing model):
- Qwen3: ~250-300 lines total
- Gemma3: ~400-450 lines total
- Llama4: ~700-900 lines total

## Files Changed

| File | Change |
|------|--------|
| `crates/rvllm-model-runner/src/architectures/qwen3.rs` | New file |
| `crates/rvllm-model-runner/src/architectures/gemma3.rs` | New file (or extend gemma.rs) |
| `crates/rvllm-model-runner/src/architectures/llama4.rs` | New file |
| `crates/rvllm-model-runner/src/architectures/mod.rs` | Register new models |
| `crates/rvllm-model-runner/src/layers/norm.rs` | Add QK-Norm helper |
| `crates/rvllm-model-runner/src/layers/moe.rs` | Add sigmoid routing mode |
| `crates/rvllm-attention/src/` | Chunked local attention (for Llama4) |
