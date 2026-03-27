# Contributing to rvLLM

We're building the Rust standard for LLM inference. Here's how to help.

## Architecture

Every feature lives in its own crate. You can work on any crate independently -- all tests run with `mock-gpu` (no CUDA needed for development).

```bash
# Run all tests (no GPU required)
cargo test --workspace

# Check compilation
cargo check --workspace

# Run your crate's tests
cargo test -p rvllm-model-runner
```

## Open Feature Tracks

These are well-scoped features with clear specs. Each one is a meaningful contribution. Pick one, implement it, open a PR.

---

### 1. LoRA Adapter Serving

**Crate:** `rvllm-model-runner`, `rvllm-model-loader`, `rvllm-api`
**Difficulty:** Hard
**Impact:** Huge -- lets users serve 100+ fine-tuned models on one GPU

**What to build:**
- `crates/rvllm-model-runner/src/lora.rs` -- LoRA weight container: base_weight + low-rank A,B matrices per layer
- Fused forward: `output = x @ (W + alpha * A @ B)` -- either materialize the merged weight or compute the delta on the fly
- Hot-swap: load/unload LoRA adapters without restarting the server
- API: `POST /v1/lora/load`, `POST /v1/lora/unload`, `model` field in request selects adapter
- Store LoRA weights separately from base model, share base KV cache across adapters

**Reference:** [Python vLLM LoRA](https://docs.vllm.ai/en/latest/models/lora.html), [LoRA paper](https://arxiv.org/abs/2106.09685)

**Key files to read:**
- `crates/rvllm-model-runner/src/layers/linear.rs` -- where weight multiplication happens
- `crates/rvllm-model-runner/src/gpu_runner.rs` -- forward pass orchestration
- `crates/rvllm-model-loader/src/gpu_loader.rs` -- weight loading

---

### 2. Beam Search and Best-of-N

**Crate:** `rvllm-engine`, `rvllm-block-manager`, `rvllm-sequence`
**Difficulty:** Medium
**Impact:** Medium -- needed for summarization, translation, and quality-sensitive applications

**What to build:**
- `crates/rvllm-engine/src/beam_search.rs` -- maintain K beams per request
- Each beam is a `Sequence` that shares prompt KV cache blocks via copy-on-write
- At each step: expand each beam by top-K tokens, score, prune to K best
- Best-of-N: run N independent samples, return highest cumulative logprob
- Use `BlockManager::fork()` for CoW KV cache sharing between beams

**The data structures already exist:**
- `SequenceGroup` supports multiple `Sequence` objects (beam candidates)
- `BlockManager` has `fork()` and reference counting on physical blocks
- `SamplingParams` has `best_of` and `use_beam_search` fields

**What's missing:** wiring these into the `GpuLLMEngine::step()` loop.

---

### 3. Batch Processing API

**Crate:** `rvllm-api`, `rvllm-engine`
**Difficulty:** Medium
**Impact:** Medium -- needed for offline batch processing workloads

**What to build:**
- `crates/rvllm-api/src/routes/batch.rs` -- OpenAI Batch API endpoints
- `POST /v1/batches` -- accept JSONL file of requests, return batch ID
- `GET /v1/batches/{id}` -- check status (pending, in_progress, completed, failed)
- `GET /v1/batches/{id}/output` -- download results as JSONL
- Background processing: feed requests through the engine, store results on disk
- Support cancellation: `POST /v1/batches/{id}/cancel`

**Reference:** [OpenAI Batch API](https://platform.openai.com/docs/api-reference/batch)

---

### 4. Embedding Model Support

**Crate:** `rvllm-model-runner`, `rvllm-api`
**Difficulty:** Medium
**Impact:** Medium -- embeddings are a major API use case

**What to build:**
- `crates/rvllm-model-runner/src/architectures/embedding.rs` -- `EmbeddingModel` trait
- Forward pass returns hidden states instead of logits
- Pooling strategies: mean pooling, CLS token, last token
- Normalization option (L2 normalize embeddings)
- `crates/rvllm-api/src/routes/embeddings.rs` -- `POST /v1/embeddings` endpoint
- Support sentence-transformers, E5, GTE, BGE model families

**Key difference from causal models:** no autoregressive generation, single forward pass per request. Much simpler execution path.

---

### 5. Vision-Language Models

**Crate:** `rvllm-model-runner`, `rvllm-tokenizer`, `rvllm-api`
**Difficulty:** Very Hard
**Impact:** High -- VLMs are growing fast

**What to build:**
- Image preprocessing: resize, normalize, patch extraction
- Vision encoder (ViT): separate forward pass for image patches
- Cross-attention or concatenated image+text tokens
- Support LLaVA, Qwen-VL, InternVL architectures
- API: accept image URLs or base64 in chat messages
- `crates/rvllm-model-runner/src/vision/` -- vision encoder, image preprocessing

**This is the largest feature.** Consider starting with LLaVA (simplest architecture: ViT encoder + linear projection + Llama decoder).

---

### 6. Pipeline Parallelism

**Crate:** `rvllm-executor`, `rvllm-worker`
**Difficulty:** Hard
**Impact:** Medium -- useful for very large models that don't fit with TP alone

**What to build:**
- Split transformer layers across GPUs (layers 0-15 on GPU0, 16-31 on GPU1)
- Point-to-point communication between pipeline stages (NCCL send/recv)
- Microbatching to keep all GPUs busy
- Combine with tensor parallelism for hybrid TP+PP

**Prerequisites:** Agent 18 (Multi-GPU Tensor Parallelism) should be done first.

---

## How to Contribute

### Setup

```bash
git clone https://github.com/m0at/hermes-lite.git
cd hermes-lite/vllm-rs
cargo test --workspace  # verify everything passes
```

### Adding a Model Architecture

The most common contribution. Here's the pattern:

```bash
# 1. Create your architecture file
touch crates/rvllm-model-runner/src/architectures/my_model.rs
```

```rust
// crates/rvllm-model-runner/src/architectures/my_model.rs
use crate::bridge::*;

pub struct MyModelForCausalLM {
    config: ModelRunnerConfig,
}

impl Architecture for MyModelForCausalLM {
    fn forward(&self, input: &ModelInput, cache: &CacheEngine) -> Result<GpuBuffer<f32>> {
        // Your forward pass here
        // See llama.rs for the reference implementation
        todo!()
    }

    fn name(&self) -> &str { "MyModelForCausalLM" }
}
```

```rust
// Register in crates/rvllm-model-runner/src/architectures/mod.rs
pub fn create_model(architecture: &str, ...) -> Result<Box<dyn Architecture>> {
    match architecture {
        // ... existing models ...
        "MyModelForCausalLM" => Ok(Box::new(MyModelForCausalLM::new(...))),
        _ => Err(LLMError::ModelError(...)),
    }
}
```

### Adding a CUDA Kernel

```bash
# 1. Write the kernel
cat > kernels/my_kernel.cu << 'EOF'
extern "C" __global__ void my_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // your operation
    }
}
EOF

# 2. Compile to PTX
cd kernels && nvcc -ptx -arch=sm_80 -O3 -o my_kernel.ptx my_kernel.cu

# 3. Load in Rust via KernelLoader (see crates/rvllm-gpu/src/kernel_loader.rs)
```

### Adding an API Endpoint

```rust
// crates/rvllm-api/src/routes/my_endpoint.rs
use axum::{Json, extract::State};

pub async fn my_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MyRequest>,
) -> Result<Json<MyResponse>, ApiError> {
    // Your handler
}

// Register in crates/rvllm-api/src/server.rs build_router()
```

### Testing

```bash
# Run all tests (mock-gpu, no CUDA needed)
cargo test --workspace

# Run specific crate
cargo test -p rvllm-sampling

# Run with CUDA (requires GPU)
cargo test --workspace --features cuda

# API compatibility tests
VLLM_RS_URL=http://localhost:8000 python3 -m pytest tests/api_compat/ -v
```

### Code Style

- `cargo fmt` before committing
- `cargo clippy --workspace` should be warning-free
- All public items need `///` doc comments
- Use `tracing::{info, debug, warn, error}` for logging, never `println!`
- Error handling: return `Result<T>` with `LLMError`, never `unwrap()` in library code
- CUDA code behind `#[cfg(feature = "cuda")]`

## Project Principles

- **Correctness first** -- match Python vLLM output token-for-token before optimizing
- **Test without hardware** -- every feature should be testable with `mock-gpu`
- **Minimal dependencies** -- prefer standard library, avoid crate bloat
- **Direct GPU access** -- no PyTorch, no Python, just cuBLAS and CUDA kernels
- **One binary** -- everything ships as a single static executable
