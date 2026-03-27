# Vision-Language Model Support

## Status: Research Phase

rvLLM currently supports text-only causal language models. This document outlines the plan for adding vision-language model (VLM) support.

## Architecture Overview

VLMs typically consist of three components:

```
Image -> [Vision Encoder] -> [Projection] -> [Language Model]
                                                 ^
Text  -> [Tokenizer] ----------------------------|
```

1. **Vision Encoder**: A ViT (Vision Transformer) that converts image patches into hidden state vectors
2. **Projection Layer**: Maps vision hidden states into the language model's embedding space (linear, MLP, or cross-attention)
3. **Language Model**: Standard causal LM that processes interleaved image+text tokens

## Target Architectures

| Model | Vision Encoder | Projection | LM Backbone | Priority |
|---|---|---|---|---|
| LLaVA 1.5/1.6 | CLIP ViT-L/14 | 2-layer MLP | Llama/Vicuna | High (simplest) |
| Qwen-VL / Qwen2-VL | ViT with dynamic resolution | Cross-attention | Qwen2 | High |
| InternVL 2 | InternViT-6B | MLP | InternLM2 / Llama | Medium |
| Phi-3-Vision | CLIP ViT | Linear | Phi-3 | Medium |
| Llama 3.2 Vision | ViT | Cross-attention layers | Llama 3.2 | High |

## Implementation Plan

### Phase 1: Image Preprocessing (no GPU)

```
crates/rvllm-model-runner/src/vision/
  mod.rs           -- module root
  preprocessor.rs  -- image loading, resize, normalize, patch extraction
  clip.rs          -- CLIP-specific preprocessing (224x224, bicubic, normalize)
```

- Load images from URL or base64 (in the API layer)
- Resize to model-specific resolution
- Normalize with model-specific mean/std
- Extract patches (14x14 or 16x16)
- Output: tensor of shape `[num_patches, 3, patch_size, patch_size]`

### Phase 2: Vision Encoder (GPU)

```
crates/rvllm-model-runner/src/vision/
  vit.rs           -- Vision Transformer forward pass
  position.rs      -- 2D position embeddings for patches
```

- ViT forward pass: patch embedding -> N transformer layers -> output
- This is a standard transformer, reuse existing layer implementations (linear, norm, attention)
- But: vision attention is NOT paged (fixed sequence length), so use dense attention
- Output: `[num_patches, hidden_dim]`

### Phase 3: Projection + Interleaving

```
crates/rvllm-model-runner/src/vision/
  projection.rs    -- map vision tokens to LM embedding space
```

- Project vision hidden states into LM dimension
- Insert projected image tokens at the correct positions in the text token sequence
- Handle special tokens: `<image>`, `<|image_pad|>`, etc.

### Phase 4: API Integration

```
crates/rvllm-api/src/routes/chat.rs  -- handle image_url in messages
```

OpenAI vision API format:
```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
    ]
  }]
}
```

### Phase 5: KV Cache for Vision Tokens

Vision tokens are computed once (during prefill) and cached. They don't change during decoding. This means:
- Vision token KV cache is static after prefill
- Can be shared across multiple requests with the same image (prefix caching)
- Memory: vision tokens are typically 576 tokens (LLaVA) to 2048 tokens (Qwen2-VL)

## How to Contribute

### Starting Point: LLaVA 1.5

LLaVA is the simplest VLM to implement because:
1. Vision encoder is a standard CLIP ViT (well-documented)
2. Projection is a 2-layer MLP (trivial)
3. LM backbone is Llama (already implemented)
4. No special attention patterns (just concatenate vision + text tokens)

```rust
// Pseudocode for LLaVA forward pass
fn forward(&self, text_tokens: &[u32], image: Option<&Tensor>) -> Tensor {
    let text_embeds = self.embed_tokens(text_tokens);

    if let Some(img) = image {
        let vision_embeds = self.vision_encoder.forward(img);  // [576, 4096]
        let projected = self.projection.forward(vision_embeds);  // [576, 4096]
        let combined = interleave(text_embeds, projected, image_positions);
        self.llm_forward(combined)
    } else {
        self.llm_forward(text_embeds)
    }
}
```

### Reference Implementations

Study these for implementation details:

- **mistral.rs**: Has multi-modal support built on Candle. Look at their vision encoder and projection layer implementations for patterns.
- **Python vLLM**: `vllm/model_executor/models/llava.py` -- the reference implementation
- **transformers**: `transformers/models/llava/modeling_llava.py` -- HuggingFace reference

### Key Decisions

1. **Image loading library**: Use `image` crate for decoding (JPEG, PNG, WebP). No Python dependency.
2. **HTTP image fetching**: Use `reqwest` to download images from URLs in the API layer.
3. **Preprocessing on CPU vs GPU**: CPU is fine for preprocessing (resize, normalize). The ViT forward pass runs on GPU.
4. **Dynamic resolution**: Qwen2-VL and LLaVA-NeXT support variable image sizes. Start with fixed resolution (LLaVA 1.5), add dynamic later.

### File Structure

```
crates/rvllm-model-runner/src/
  vision/
    mod.rs
    preprocessor.rs      -- image loading, resize, patch extraction
    clip.rs              -- CLIP ViT encoder
    vit.rs               -- generic ViT forward pass
    position.rs          -- 2D position embeddings
    projection.rs        -- vision -> LM space mapping
  architectures/
    llava.rs             -- LLaVA 1.5/1.6 (first target)
    qwen_vl.rs           -- Qwen-VL / Qwen2-VL
    internvl.rs          -- InternVL 2

crates/rvllm-api/src/
  routes/
    chat.rs              -- handle image_url content type
  types/
    request.rs           -- ContentPart enum (text | image_url)
```

### Dependencies to Add

```toml
# In rvllm-model-runner/Cargo.toml
image = "0.25"           # Image decoding
reqwest = { version = "0.12", features = ["blocking"] }  # URL fetching
```

## Timeline

This is a significant feature requiring ~3000-5000 lines of new code. Estimated phases:
1. Image preprocessing: 1 agent
2. CLIP ViT encoder: 1 agent
3. LLaVA integration: 1 agent
4. API + testing: 1 agent
5. Additional architectures: community contributions

## Related Issues

- Model architecture PRs welcome -- see [CONTRIBUTING.md](../CONTRIBUTING.md)
- Multi-modal models tracked in project roadmap
