# rvLLM: High-performance LLM inference in Rust

A from-scratch Rust rewrite of [vLLM](https://github.com/vllm-project/vllm) -- the most popular open-source LLM serving engine. Drop-in replacement for the OpenAI-compatible API with dramatically better resource efficiency.

**30 CUDA kernels. Pure f16 end-to-end. CUDA graph replay. 27,555 tok/s on H100. 20x faster startup. 31x smaller binary.**

## The gap (and how we're closing it)

Honest head-to-head on H100 80GB. Qwen2.5-1.5B, f16, greedy decoding, 100 tokens/request. Direct engine benchmarks (no HTTP). Both engines on their native optimized environments. Measured 2026-03-29.

| N | rvLLM (FA2) | rvLLM (FA3) | Improvement | vLLM 0.18 (full) | vs vLLM |
|---:|---:|---:|---|---:|---|
| 1 | 272 | **317** | +17% | 453 | 0.70x |
| 4 | 1,022 | **1,173** | +15% | 1,713 | 0.68x |
| 16 | 3,520 | **4,031** | +15% | 6,896 | 0.58x |
| 64 | 11,032 | **13,521** | +23% | 25,664 | 0.53x |
| 128 | 15,053 | **20,328** | +35% | 41,051 | 0.50x |
| 256 | 18,578 | **27,555** | +48% | 61,922 | 0.45x |

vLLM (torch.compile + FlashAttention 3 + piecewise CUDA graphs) is faster. The gap comes from optimizations that took hundreds of engineers years to build. rvLLM uses hand-written CUDA kernels and cuBLAS without compile-time fusion.

The FA3-style kernel (256 threads, vectorized half2 loads, warp-parallel reductions) closed 15-48% of the gap in one iteration. Fused kernels (RMSNorm+GEMV, SiLU+Down projection) are wired and being validated.

**Coherence verified** -- all prompts produce correct, coherent English text:
```
"The capital of France is" -> " Paris. The capital of France is Paris."
"Write a Python function"  -> " that takes a list of integers and returns the sum..."
"2 + 2 ="                  -> " 4"
```

### What rvLLM does well

| Metric | rvLLM | Python vLLM |
|---|---|---|
| Startup time | 6 sec | ~120 sec |
| Binary size | 16 MB | ~500 MB |
| CPU memory | 348 MB | ~1 GB |
| Dependencies | 0 (static binary) | PyTorch + 500MB |
| Scaling | Near-linear N=1 to N=256 | Same |

### Path to close the gap

1. **FA3 decode kernel** -- DONE. 256 threads, vectorized half2, warp-parallel. +15-48%.
2. **Fused SiLU+Down GEMV** -- DONE. Eliminates intermediate buffer + 1 launch per layer.
3. **Fused RMSNorm+GEMV** -- DONE (kernel written). Eliminates normed buffer + 1 launch per layer.
4. **Piecewise CUDA graphs** -- captures separate graph pieces for variable-length prefill/decode.
5. **FP8/INT8 quantization** -- halves weight reads, doubles effective memory bandwidth.
6. **True Hopper TMA/WGMMA** -- async tensor memory access + warp group matrix multiply.

See [docs/arch.md](docs/arch.md) for the full forward pass trace and [docs/update-log.md](docs/update-log.md) for optimization history.

## Install

```bash
# From crates.io
cargo install rvllm

# From PyPI
pip install rvllm
```

Or build from source -- see [Quick Start](#quick-start) below.

### CPU Component Benchmarks (sampling, logit processing)

Operations that run on CPU between GPU forward passes. Measured on both Apple M5 and A100 Xeon.

| Operation | Rust | Python (numpy) | Speedup | Notes |
|---|---|---|---|---|
| Combined penalties (rep+freq+pres) | 2.6 us | 63 us | **24x** | Pure iteration, zero alloc |
| Repetition penalty (2K tokens) | 3.1 us | 34 us | **11x** | In-place mutation |
| Multinomial sampling (32K vocab) | 12 us | 66 us | **5.5x** | Cumulative sum + early exit |
| Top-P nucleus (128K vocab) | 1.6 ms | 6.9 ms | **4.3x** | Partial sort + threshold |
| Q4 dequantization (10M elements) | 7.1 ms | 9.7 ms | **1.4x** | Chunk-based autovectorization |
| Batch sampling (64 seqs, Rayon) | 4.3 ms | 36.4 ms | **8.5x** | Rayon across 10 M5 cores |

## Tested GPUs

| Compute Capability | GPUs | Status |
|---|---|---|
| sm_70 | V100 | Supported |
| sm_75 | T4, RTX 2080 | Supported |
| sm_80 | **A100**, A30 | Tested, benchmarked |
| sm_86 | RTX 3090, A40 | Supported |
| sm_89 | RTX 4090, L40S | Supported |
| sm_90 | **H100**, H200 | Supported |
| sm_100 | **B100**, **B200** | Supported (requires CUDA 12.8+) |
| sm_120 | **RTX 5090**, RTX 6000 Blackwell | Supported (requires CUDA 13.0+) |
| sm_122 | RTX 5080, RTX 5070 | Supported (requires CUDA 13.0+) |

Kernels are compiled to PTX for all architectures by default (`cd kernels && bash build.sh`). To build for a specific GPU:
```bash
CUDA_ARCH=sm_90 bash kernels/build.sh   # H100 only
bash kernels/build.sh sm_89             # RTX 4090 only
```

**Want to add support for a new GPU?** Add the `sm_XX` target to `kernels/build.sh` and verify the kernels compile. If a kernel uses architecture-specific features (tensor cores, etc.), submit a PR with the optimized variant. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Why Rust over Python

### The GIL problem

Python's Global Interpreter Lock means vLLM's scheduler, tokenizer, and output processing all run single-threaded. When you have 256 concurrent requests, the scheduling loop itself becomes a bottleneck. Rust has no GIL -- scheduling, sampling, and tokenization run truly parallel across all cores.

### No garbage collector

Python's garbage collector can pause inference at unpredictable times. With large batch sizes, GC pauses grow as Python tracks millions of tensor metadata objects. Rust's ownership model means deterministic deallocation with zero GC pauses. Memory is freed the instant it goes out of scope.

### 15MB vs 500MB

Python vLLM requires PyTorch (~2GB), transformers, numpy, and dozens of other packages. A fresh `pip install vllm` pulls ~500MB of dependencies. rvLLM compiles to a single 15MB static binary with zero runtime dependencies. Deploy by copying one file.

### Direct GPU access

Python vLLM talks to the GPU through PyTorch, which adds overhead for tensor creation, memory management, and kernel dispatch. rvLLM calls cuBLAS and CUDA kernels directly through cudarc, eliminating the middle layer. FP16 hgemm with tensor cores for matrix multiplies and f16 KV cache keep memory bandwidth and compute on the fast path.

### Startup time

Python vLLM takes 30-60 seconds to start (importing PyTorch, JIT compiling Triton kernels, initializing NCCL). rvLLM starts serving in ~7 seconds -- load model weights and go.

### Memory efficiency

Python objects carry ~50 bytes of overhead each. A running vLLM server with thousands of sequences creates millions of Python objects for metadata tracking. Rust structs are laid out exactly as you define them -- an 8-byte sequence ID is 8 bytes, not 58.

## Quick Start

### Build from source

```bash
# Mac/Linux (no GPU needed, uses mock-gpu backend for development)
cargo build --release -p rvllm-server

# Linux + NVIDIA GPU (requires CUDA toolkit)
cargo build --release --features cuda -p rvllm-server

# Compile CUDA kernels (only needed for GPU inference)
cd kernels && bash build.sh
```

### Serve a model

```bash
# Start serving (downloads model from HuggingFace automatically)
./target/release/rvLLM serve --model Qwen/Qwen2.5-1.5B

# With options
./target/release/rvLLM serve \
  --model meta-llama/Llama-3.2-1B \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

### Send requests

```bash
# Completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The theory of relativity states that","max_tokens":100}'

# Chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain quantum computing"}],"max_tokens":200}'

# Responses
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B","input":"Explain quantum computing","max_output_tokens":200}'

# Responses with custom function tools
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B","input":"What is the weather in Boston?","tools":[{"type":"function","name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}],"tool_choice":"auto"}'

# Streaming
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":100,"stream":true}'
```

### Docker

```bash
# Build image
make docker

# Run with GPU
docker run --gpus all -p 8000:8000 rvllm:latest \
  serve --model Qwen/Qwen2.5-1.5B

# Docker Compose (starts both Rust and Python vLLM for comparison)
MODEL_NAME=Qwen/Qwen2.5-1.5B docker compose up
```

## API Compatibility

rvLLM implements the same OpenAI-compatible API as Python vLLM. Existing clients work unchanged -- just point them at the Rust server.

| Endpoint | Method | Status |
|----------|--------|--------|
| `/v1/completions` | POST | Working (streaming + non-streaming) |
| `/v1/chat/completions` | POST | Working (streaming + non-streaming) |
| `/v1/responses` | POST | Working (text, stored retrieval, custom function tools, tool streaming; built-in tools not yet supported) |
| `/v1/responses/{id}` | GET | Working for stored responses |
| `/v1/responses/{id}/input_items` | GET | Working for stored responses |
| `/v1/models` | GET | Working |
| `/health` | GET | Working |
| `/metrics` | GET | Working (Prometheus format) |

### Using with the OpenAI Python client

```python
from openai import OpenAI

# Just change the base_url -- everything else stays the same
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Completions
response = client.completions.create(
    model="Qwen/Qwen2.5-1.5B",
    prompt="The meaning of life is",
    max_tokens=50,
    temperature=0.8,
    top_p=0.95,
)
print(response.choices[0].text)

# Chat
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B",
    messages=[{"role": "user", "content": "Write a haiku about Rust"}],
    max_tokens=50,
)
print(response.choices[0].message.content)

# Responses
response = client.responses.create(
    model="Qwen/Qwen2.5-1.5B",
    input="Write a haiku about Rust",
    max_output_tokens=50,
)
print(response.output[0].content[0].text)

# Streaming
stream = client.completions.create(
    model="Qwen/Qwen2.5-1.5B",
    prompt="In the beginning",
    max_tokens=100,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].text, end="", flush=True)
```

### Using with LiteLLM

```python
import litellm

response = litellm.completion(
    model="hosted_vllm/Qwen/Qwen2.5-1.5B",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:8000/v1",
)
```

### Using with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",
    model="Qwen/Qwen2.5-1.5B",
)
response = llm.invoke("Explain transformers in one paragraph")
```

### Supported sampling parameters

All standard OpenAI parameters work:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Randomness (0 = greedy) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `top_k` | int | -1 | Top-K filtering (-1 = disabled) |
| `max_tokens` | int | 256 | Maximum tokens to generate |
| `stop` | string[] | null | Stop sequences |
| `stream` | bool | false | Enable SSE streaming |
| `presence_penalty` | float | 0.0 | Penalize repeated topics |
| `frequency_penalty` | float | 0.0 | Penalize repeated tokens |
| `seed` | int | null | Deterministic generation |
| `n` | int | 1 | Number of completions |

### Beam search controls

rvLLM also accepts the beam-search controls used by vLLM:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_beam_search` | bool | false | Enable beam search instead of independent sampling |
| `best_of` | int | `n` | Beam width when beam search is enabled |
| `length_penalty` | float | 1.0 | Length normalization factor for beam ranking |
| `early_stopping` | bool | false | Stop once completed beams dominate active beams |

Notes:
- Beam width is `best_of.unwrap_or(n)`.
- Beam search is non-streaming only.
- Non-beam `best_of` remains best-of-N sampling, not beam search.

```bash
# Beam-search completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-1.5B",
    "prompt":"Summarize Rust ownership in one sentence.",
    "max_tokens":64,
    "n":2,
    "best_of":4,
    "use_beam_search":true,
    "length_penalty":0.6,
    "early_stopping":true
  }'

# Beam-search chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-1.5B",
    "messages":[{"role":"user","content":"Explain beam search in plain English."}],
    "max_tokens":96,
    "n":2,
    "use_beam_search":true,
    "length_penalty":0.8
  }'
```

## Reproducible Benchmarking

### Fresh-instance benchmark script

Run a bounded, reproducible benchmark on any CUDA machine:

```bash
bash bench/run.sh
```

This will:
1. Verify CUDA/GPU presence
2. Build rvLLM with `--features cuda`
3. Start the server, wait for health
4. Run 16 prompts at concurrency 1 and 4
5. Report startup time, RSS, VRAM, latency percentiles, throughput
6. Clean up the server on exit (PID-based, with trap)

Environment variables: `MODEL`, `PORT`, `MAX_TOKENS`, `NUM_PROMPTS`, `CONCURRENCY_LEVELS`.

### One-command A100 benchmark (vast.ai)

Requires a [vast.ai](https://vast.ai) account with API key configured.

```bash
make a100-bench
```

This will:
1. Provision an A100 80GB on vast.ai (~$1.10/hr)
2. Upload and build rvLLM with CUDA
3. Install Python vLLM 0.18.0
4. Run both servers on the same model
5. Benchmark throughput, latency, TTFT, memory usage
6. Print a side-by-side comparison
7. Tear down the instance

### Manual deployment

```bash
# 1. Provision
bash deploy/vastai-provision.sh

# 2. Build on the instance
bash deploy/vastai-deploy.sh

# 3. Run benchmarks
bash deploy/vastai-benchmark.sh

# 4. Tear down
bash deploy/vastai-teardown.sh
```

### Local CPU benchmarks (no GPU needed)

Compare Rust vs Python/numpy/torch on sampling and logit processing:

```bash
make bench-compare
# or
bash scripts/benchmark.sh
```

### Run API compatibility tests

```bash
# Start server, then:
RVLLM_URL=http://localhost:8000 python3 -m pytest tests/api_compat/ -v
```

### Beam-search smoke test on CUDA

After starting a CUDA-backed server, run:

```bash
RVLLM_URL=http://localhost:8000 bash tests/api_compat/run_beam_search_smoke.sh
```

## Video Demo

Record a side-by-side terminal demo comparing rvLLM vs Python vLLM inference speed:

```bash
bash bench/video_demo.sh
```

Uses tmux split panes to show both servers receiving identical prompts simultaneously. Records output as an asciinema `.cast` file. See `bench/video/README.md` for details.

## Paper / Technical Report

An arXiv-style technical paper describing the architecture, CUDA integration, and design decisions is available in two formats:

**LaTeX sources** (under `docs/paper/`):
```bash
cd docs/paper
pdflatex rvllm.tex && bibtex rvllm && pdflatex rvllm.tex && pdflatex rvllm.tex   # color
pdflatex rvllm-bw.tex && bibtex rvllm-bw && pdflatex rvllm-bw.tex && pdflatex rvllm-bw.tex  # B&W
```

**GitHub Pages version** with B&W/Color toggle: enable GitHub Pages on the `/docs` folder in repo Settings. No download button -- the paper is rendered inline as HTML.

## Architecture

23 Rust crates organized in a dependency tree from low-level GPU primitives to the HTTP API surface.

```
rvllm-server (binary, 16MB)
  |
  +-- rvllm-api                  HTTP layer: axum, SSE streaming, OpenAI routes
  |     +-- rvllm-engine         Async inference loop: scheduler + executor + tokenizer
  |     |     +-- rvllm-scheduler       Continuous batching, FCFS/priority/SJF policies
  |     |     +-- rvllm-executor        Single/multi-GPU worker orchestration
  |     |     |     +-- rvllm-worker    Per-GPU execution: forward pass + sampling
  |     |     +-- rvllm-speculative     Draft-model speculative decoding
  |     +-- rvllm-telemetry      Prometheus metrics, structured tracing
  |
  +-- rvllm-model-runner         Transformer forward pass, layer implementations
  |     +-- rvllm-attention      PagedAttention, FlashAttention backends
  |     +-- rvllm-kv-cache       Paged key-value cache, block tables
  |     +-- rvllm-model-loader   SafeTensors/GGUF loading, HF hub, sharding
  |     +-- rvllm-quant          GPTQ/AWQ/FP8 dequantization
  |
  +-- rvllm-sampling             Logit processing, top-k/p, multinomial, Rayon batching
  +-- rvllm-block-manager        Block allocation, copy-on-write, prefix sharing
  +-- rvllm-memory               GPU/CPU memory pools, swap manager
  +-- rvllm-gpu                  CUDA/mock abstraction, cuBLAS, kernel loader
  +-- rvllm-tokenizer            HuggingFace tokenizers, chat templates
  +-- rvllm-sequence             Sequence state, request groups, metadata
  +-- rvllm-config               CLI args, TOML config, validation
  +-- rvllm-python               PyO3 Python bindings
  +-- rvllm-core                 Shared types, error hierarchy, prelude
```

### CUDA Kernels

15 hand-written CUDA kernels compiled to PTX, loaded at runtime via cudarc:

| Kernel | File | Purpose |
|--------|------|---------|
| PagedAttention V2 | `paged_attention.cu` | Attention with block-table indirection, online softmax |
| FlashAttention-2 | `flash_attention.cu` | Fused prefill + decode attention with causal masking |
| RMSNorm | `rms_norm.cu` | Shared-memory parallel reduction for normalization |
| RMSNorm FP16 | `rms_norm_f16.cu` | Half-precision RMSNorm variant |
| Fused Residual+RMSNorm | `fused_residual_rmsnorm.cu` | Fused residual add + normalize in one kernel |
| Rotary Embedding | `rotary_embedding.cu` | RoPE with GQA support |
| Activations | `activation.cu` | SiLU, GELU, fused SiLU*mul for MLP |
| Activations FP16 | `activation_f16.cu` | Half-precision activation variants |
| Softmax | `softmax.cu` | Warp-level numerically stable softmax |
| Argmax | `argmax.cu` | GPU-side greedy sampling (avoids D2H transfer) |
| Embedding Gather | `embedding_gather.cu` | GPU-resident token embedding lookup |
| Reshape and Cache | `reshape_and_cache.cu` | Write QKV into paged KV cache |
| Block Copy | `copy_blocks.cu` | KV cache block copy for beam search |
| Add Bias | `add_bias.cu` | Fused bias addition for QKV projections |
| FP8 KV Cache | `fp8_kv.cu` | E4M3 quantization/dequantization for KV cache |

### Design decisions

**Why not wrap PyTorch from Rust?** PyTorch's C++ API (libtorch) is 2GB and brings its own CUDA runtime, memory allocator, and threading model. We'd inherit all of Python vLLM's overhead. Going direct to cuBLAS/CUDA means we control every allocation and kernel launch.

**Why cudarc?** Safe Rust bindings to the CUDA driver API. No need for a C++ build step. PTX kernels loaded at runtime, not linked at compile time. The `mock-gpu` feature compiles everywhere without CUDA.

**Why not Triton?** Triton requires Python and a JIT compiler. Our CUDA kernels are pre-compiled to PTX -- zero runtime compilation, deterministic startup.

**Why separate crates?** Each crate has a clear responsibility and can be tested independently. The mock-gpu feature means all scheduling, sampling, and API logic is tested without a GPU. Only the forward pass requires real hardware.

## Migrating from Python vLLM

### For API consumers (zero code changes)

If you call vLLM's OpenAI-compatible API, rvLLM is a drop-in replacement. Same endpoints, same request format, same response format.

```bash
# Before (Python vLLM)
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B

# After (Rust rvLLM)
rvLLM serve --model meta-llama/Llama-3-8B
```

Your client code doesn't change at all.

### For server operators

Same CLI flags:

| Python vLLM | Rust rvLLM | Notes |
|---|---|---|
| `--model` | `--model` | Same |
| `--port` | `--port` | Same (default 8000) |
| `--host` | `--host` | Same (default 0.0.0.0) |
| `--gpu-memory-utilization` | `--gpu-memory-utilization` | Same (default 0.90) |
| `--max-model-len` | `--max-model-len` | Same |
| `--tensor-parallel-size` | `--tensor-parallel-size` | Same |
| `--enforce-eager` | (default) | Rust has no graph compilation step |
| `--dtype auto` | `--dtype auto` | Same |

### Supported model architectures

| Architecture | Models | Status |
|---|---|---|
| LlamaForCausalLM | Llama 2/3, CodeLlama, Vicuna | Working |
| MistralForCausalLM | Mistral 7B, Mistral Nemo | Working |
| Qwen2ForCausalLM | Qwen2, Qwen2.5 | Working |
| PhiForCausalLM | Phi-2, Phi-3, Phi-3.5 | Implemented |
| GemmaForCausalLM | Gemma, Gemma 2 | Implemented |
| MixtralForCausalLM | Mixtral 8x7B, 8x22B | Implemented |
| DeepseekV2ForCausalLM | DeepSeek-V2, DeepSeek-V2.5 | Implemented |
| GPTNeoXForCausalLM | Pythia, GPT-NeoX-20B | Implemented |
| StableLmForCausalLM | StableLM-3B, StableLM-2 | Implemented |
| CohereForCausalLM | Command-R, Command-R+ | Implemented |

**Want to add a model?** See [CONTRIBUTING.md](CONTRIBUTING.md#1-adding-a-model-architecture) -- it's a single file implementing the `Architecture` trait. We're tracking community-requested architectures in [issues](https://github.com/m0at/hermes-lite/issues).

### Python bindings

```bash
pip install maturin
cd rvllm && maturin develop --release
```

```python
import rvllm

# Fast sampling (Rayon parallelism, no server needed)
sampler = rvllm.Sampler()
result = sampler.sample(logits=[1.0, 2.0, 3.0], temperature=0.8, top_k=50)

# Tokenizer
tok = rvllm.Tokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
ids = tok.encode("Hello world")

# Parallel batch sampling (8x faster than sequential Python)
results = rvllm.sample_batch(
    logits_batch=[[1.0, 2.0] * 16000] * 64,
    temperature=0.8, top_p=0.95, seed=42,
)
```

## CLI Reference

```
rvLLM serve [OPTIONS]

Options:
  --model <MODEL>                   Model name or path (HuggingFace hub or local)
  --host <HOST>                     Bind address [default: 0.0.0.0]
  --port <PORT>                     Port [default: 8000]
  --dtype <DTYPE>                   Data type [default: auto]
  --max-model-len <LEN>            Max sequence length [default: 2048]
  --gpu-memory-utilization <FRAC>  GPU memory fraction [default: 0.90]
  --tensor-parallel-size <N>       Number of GPUs [default: 1]
  --max-num-seqs <N>               Max concurrent sequences [default: 256]
  --tokenizer <PATH>               Custom tokenizer path
  --log-level <LEVEL>              Log level [default: info]
  --disable-telemetry              Disable Prometheus metrics

rvLLM info                        Show GPU and system info
rvLLM benchmark --model <MODEL>   Run offline throughput benchmark
```

## Project Status

### Working
- GPU inference on A100 via cuBLAS HGEMM (FP16, tensor cores) + CUDA kernels (RMSNorm, SiLU, residual, embedding on GPU)
- RoPE + f16 KV cache for coherent text generation
- Continuous batching scheduler with preemption
- Full sampling pipeline (temperature, top-k/p/min-p, penalties, multinomial, Rayon parallel)
- Guided decoding / JSON mode / JSON schema / regex grammar
- Tool/function calling (Hermes-style, JSON parsing)
- Beam search and best-of-N sampling
- Logprobs in GPU path
- OpenAI-compatible API (completions, chat, streaming, embeddings, batch)
- 10 model architectures (Llama, Mistral, Qwen2, Phi, Gemma, GPT-NeoX, StableLM, Cohere, Mixtral MoE, DeepSeek MoE)
- FlashAttention-2 (CPU reference + CUDA kernel)
- CUDA graph capture/replay (working end-to-end on A100)
- FP8 KV cache (E4M3 quantization with per-head scaling)
- Prefix caching with LRU eviction
- Sliding window attention
- Tensor parallelism primitives (NCCL bindings, column/row parallel)
- Prometheus metrics (forward time, TTFT, ITL, queue gauges)
- Embedding model support (/v1/embeddings)
- Batch processing API (/v1/batches)
- PyO3 Python bindings (`import rvllm`)
- SafeTensors loading from HuggingFace Hub
- Mock-GPU backend for development without hardware
- Docker deployment with CUDA 12.4
- vast.ai automated provisioning and benchmarking
- Token-level parity test suite
- 790 tests across 23 crates

### Completed Optimizations
- Full f16 forward path (zero casts, all f16 kernels)
- Fused QKV + gate+up weight concatenation (5 GEMMs -> 2 per layer)
- Cross-layer residual+RMSNorm fusion (-28 kernel launches)
- In-place RoPE, packed metadata HtoD, memset elimination
- CUDA graph capture/replay with cuBLAS workspace
- Dedicated GPU thread (async loop stays responsive during compute)
- Async DtoH with pinned host memory

### Roadmap
- INT8/FP8 quantization (halve weight reads -> ~2ms/tok -> ~500 tok/s)
- Speculative decoding (amortize weight reads across draft tokens)
- Async engine overlap with new request arrival processing
- LoRA adapter hot-swapping
- Vision-language models
- Pipeline parallelism
- Production hardening (fuzz testing, load testing at 1000 concurrent)

## Development Cost

What it actually costs to build and benchmark an LLM inference engine from scratch, for anyone considering a similar project.

### Compute (vast.ai GPU rentals)

| GPU | Use | Rate | Est. total |
|-----|-----|------|-----------|
| A100 80GB SXM4 | Primary dev/benchmark instance | $0.96-1.15/hr | ~$800 |
| B200 (4x, 733GB VRAM) | High-concurrency scaling tests | $12.08/hr | ~$500 |
| A100 (spot instances) | Short-lived kernel debugging, CI | $0.91-2.94/hr | ~$200 |
| **Total vast.ai** | | | **~$1,500** |

### AI assistance (Claude Code)

Heavy use of Claude Code with Claude Opus for architecture design, CUDA kernel writing, debugging, and code review. Base subscription covers most usage; ~$280 in extra usage charges for intensive multi-agent swarm sessions during the final performance push.

### Total

Roughly **$1,780** in compute and AI overage costs to go from zero to a working Rust LLM server with verified **3,467 tok/s at N=32 on A100 FP16**, CUDA graph capture/replay, and end-to-end benchmark coverage. No salaries, no team -- one developer (Andy Norris, San Francisco) with Claude and rented GPUs over 22 hours.

## Optimization History

| Phase | N=1 tok/s | N=32 tok/s | Key change |
|---|---:|---:|---|
| Phase 4 | 130 | 3,467 | CUDA graph capture working (3 root causes fixed) |
| Phase 5 | 174 | 4,276 | 10-agent swarm: cast reduction, fused ops, engine optimization |
| Full f16 | 200 | - | Zero casts, all f16 kernels, f16io attention kernel |
| 9-agent kernel | 236 | 5,123 | Cross-layer fusion, memset elimination, pool tuning |
| GPU thread | **218** | **6,098** | Dedicated OS thread for GPU, async loop stays responsive |

See **[docs/update-log.md](docs/update-log.md)** for the full chronological record with technical details, timing breakdowns, and agent descriptions.

## Changelog

### v0.1.0

- Initial release
- OpenAI-compatible API: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/batches`
- Streaming (SSE) and non-streaming responses
- 10 model architectures: Llama, Mistral, Qwen2, Phi, Gemma, Mixtral MoE, DeepSeek MoE, GPT-NeoX, StableLM, Cohere
- Continuous batching scheduler with FCFS/priority/SJF policies and preemption
- PagedAttention with block-table KV cache management
- 15 hand-written CUDA kernels (PagedAttention V2, FlashAttention-2, RMSNorm, RoPE, SiLU, GELU, softmax, argmax, embedding gather, reshape_and_cache, block copy, add_bias, FP8 KV, fused residual+RMSNorm)
- Full sampling pipeline: temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, multinomial, beam search
- Guided decoding: JSON mode, JSON schema, regex grammar
- Tool/function calling (Hermes-style)
- FP8 KV cache with E4M3 quantization
- Prefix caching with LRU eviction
- Sliding window attention
- Tensor parallelism primitives (NCCL bindings)
- FlashAttention-2 (CPU reference + CUDA kernel)
- CUDA graph capture/replay (working end-to-end on A100)
- SafeTensors and GGUF model loading from HuggingFace Hub
- PyO3 Python bindings (`import rvllm`)
- Prometheus metrics endpoint (`/metrics`)
- Mock-GPU backend for development without NVIDIA hardware
- Docker deployment with CUDA 12.4
- vast.ai one-command benchmarking (`make a100-bench`)
- 790 tests across 23 crates

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed guides on adding models, kernels, API endpoints, and the open feature tracks (LoRA, beam search, batch API, embeddings, VLMs, pipeline parallelism).

The codebase is organized so you can work on any layer independently:
- **Add a model**: Implement `Architecture` trait in `crates/rvllm-model-runner/src/architectures/`
- **Add a sampling method**: Add to `crates/rvllm-sampling/src/logit_processors.rs`
- **Add an API endpoint**: Add route in `crates/rvllm-api/src/routes/`
- **Add a CUDA kernel**: Write `.cu` in `kernels/`, load via `KernelLoader`

All tests run with `mock-gpu` -- no GPU needed for development:
```bash
cargo test --workspace
```

## License

Apache-2.0
