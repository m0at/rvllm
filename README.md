# rvLLM: High-performance LLM inference in Rust

A from-scratch Rust rewrite of [vLLM](https://github.com/vllm-project/vllm) -- the most popular open-source LLM serving engine. Drop-in replacement for the OpenAI-compatible API with dramatically better resource efficiency.

**50 CUDA kernels. Rust PTX compiler with 2-7.5x faster codegen than nvcc. cuBLAS autotuning. CUDA graph replay. FP8 inference. 20x faster startup. 31x smaller binary.**

## rvLLM vs Python vLLM -- Head-to-Head

All measurements on H100 SXM 80GB, Qwen2.5-7B f16, separate GPU instances per engine. No cherry-picking -- same model, same hardware, same prompts.

### Throughput

| Metric | rvLLM | Python vLLM 0.18 | Ratio |
|---|---:|---:|---|
| **Direct engine tok/s (N=128)** | 12,607 | 14,962 | 0.84x |
| **Direct engine tok/s (N=64)** | 7,280 | 8,807 | 0.83x |
| **Direct engine tok/s (N=16)** | 2,058 | 2,524 | 0.82x |
| **Direct engine tok/s (N=1)** | 108 | 169 | 0.64x |

### JIT Compiler: Our Fused Kernels vs Hand-Written CUDA

rvLLM includes a Rust-native PTX compiler that generates fused GPU kernels at model load time. These JIT kernels are **2-7.5x faster** than our hand-written nvcc-compiled CUDA on H100:

| Fused Kernel | JIT (us) | Hand-written (us) | Speedup |
|---|---:|---:|---|
| Add+RMSNorm+QKV GEMV [1,4608,3584] | 5.5 | 10.6 | **1.92x** |
| Add+RMSNorm+GateUp GEMV [1,37888,3584] | 19.3 | 98.6 | **5.12x** |
| SiLU*Mul+Down GEMV [1,3584,18944] | 9.5 | 70.7 | **7.48x** |
| RMSNorm+QKV GEMV [1,4608,3584] | 5.3 | 10.8 | **2.03x** |

The JIT compiler (`crates/rvllm-fusion/src/ptx_emit.rs`) emits PTX directly from Rust -- no nvcc, no Python, no Triton dependency. It generates shape-specialized kernels with vectorized loads, warp shuffle reductions, and shared memory tiling tuned for the specific model dimensions.

Per-step savings at N=1 (28 layers): **4.2ms** = estimated **1.8x** single-sequence speedup.

### Efficiency

| Metric | rvLLM | Python vLLM 0.18 | Winner |
|---|---:|---:|---|
| **Cold start to first token** | **6 sec** | ~120 sec | rvLLM **20x** |
| **Binary size** | **16 MB** | ~500 MB | rvLLM **31x** |
| **CPU memory at steady state** | **348 MB** | ~1 GB | rvLLM **3x** |
| **Dependencies** | **0** (static binary) | PyTorch + 500MB | rvLLM |
| **P95 latency spread** | **34 ms** (1.4%) | 190 ms (12%) | rvLLM **5.6x tighter** |
| **CUDA graph capture** | **1.7 sec** (35 sizes) | ~60 sec (torch.compile) | rvLLM **35x** |
| **cuBLAS autotuning** | **170 ms** (6 shapes) | ~60 sec (torch.compile) | rvLLM **350x** |

No Python interpreter, no GIL, no garbage collector, no PyTorch tensor allocation. rvLLM's P95 tail is 5.6x tighter than vLLM's because there are no GC pauses, no JIT recompilations, no Python object churn.

### Resource Usage (Qwen2.5-7B f16, H100 80GB)

| Metric | rvLLM | Python vLLM 0.18 |
|---|---:|---:|
| **Model weight VRAM** | 14.0 GB | 14.0 GB |
| **KV cache VRAM (0.9 util)** | 48.5 GB | ~50 GB |
| **Peak GPU memory** | 66.5 GB | ~72 GB |
| **FP8 weight support** | Yes (cublasLt) | Yes |
| **FP8 KV cache** | Yes | Yes |

### CPU-Side Operations

Operations between GPU forward passes, measured on Apple M5 and Xeon:

| Operation | Rust | Python (numpy) | Speedup |
|---|---|---|---|
| Combined penalties (rep+freq+pres) | 2.6 us | 63 us | **24x** |
| Repetition penalty (2K tokens) | 3.1 us | 34 us | **11x** |
| Multinomial sampling (32K vocab) | 12 us | 66 us | **5.5x** |
| Top-P nucleus (128K vocab) | 1.6 ms | 6.9 ms | **4.3x** |
| Batch sampling (64 seqs, Rayon) | 4.3 ms | 36.4 ms | **8.5x** |

### Deployment

| Metric | rvLLM | Python vLLM |
|---|---|---|
| Install | `cargo install rvllm` | `pip install vllm` (+ PyTorch) |
| Container image | ~50 MB | ~15 GB |
| Build from source | 35 sec | N/A |
| Kernel compilation | 30 sec (44 PTX via nvcc) + 0 sec (JIT at runtime) | 0 or ~60s (torch.compile) |
| GPU architectures | sm_80, sm_86, sm_89, sm_90 | Same + ROCm |

## Architecture

### Inference Pipeline

```
Request -> Tokenizer -> Scheduler -> GPU Forward -> Sampler -> Detokenizer -> Response
                            |              |
                     Continuous      CUDA Graph Replay
                     Batching       (35 pre-captured sizes)
                            |              |
                     Block Manager    JIT Fused Kernels
                     (paged KV)      (generated at model load)
```

### Kernel Compiler Stack

Three-tier kernel system:

**Tier 1: JIT-compiled fused kernels (fastest)**
- Rust PTX emitter generates shape-specialized fused kernels at model load
- 2-7.5x faster than hand-written CUDA for M=1 decode
- Patterns: RMSNorm+GEMV, Add+RMSNorm+GEMV, SiLU*Mul+GEMV
- No nvcc dependency -- pure Rust string-based PTX generation

**Tier 2: Hand-written CUDA kernels (50 kernels)**
- Fused decode: add+norm+QKV+bias, RoPE+cache, GQA attention, O-proj+gateup, silu+down
- FP8 E4M3 variants for all projections
- TMA async-prefetch GEMV, WGMMA tensor core GEMV
- Split-KV paged attention for long context

**Tier 3: cuBLAS/cublasLt (batched decode M>1)**
- Autotuned algorithm selection (32 candidates benchmarked per shape at startup)
- Vendored cublaslt type shim for cudarc 0.19 compatibility
- cublasLt for M<=32, cuBLAS for M>32

**LLVM NVPTX backend (experimental)**
- Full compiler: Fusion IR -> LLVM IR -> NVPTX -> PTX via inkwell
- Same backend as Triton (LLVM NVPTX)
- Gated behind `--features llvm` (requires LLVM 20.1)

### Optimization History

| Phase | Change | 7B tok/s (N=128) | Date |
|---|---|---:|---|
| 1 | FP32 baseline | -- | Mar 28 |
| 2 | FP16 inference | 6,360 | Mar 28 |
| 3 | CUDA graph replay + cublasLt | 8,578 | Mar 28 |
| 4 | 8-agent kernel fusion swarm | 12,624 | Mar 29 |
| 5 | Deeper fusion + v4 vectorized loads | 12,800 | Mar 30 |
| 6 | Vendored cublaslt + autotuner | 12,607 | Mar 30 |
| 7 | JIT compiler (2-7.5x faster kernels) | wiring | Mar 30 |

### What's Inside

| Crate | Purpose |
|---|---|
| `rvllm-server` | HTTP API (axum), CLI |
| `rvllm-engine` | Async engine, continuous batching |
| `rvllm-worker` | GPU worker, CUDA graph management |
| `rvllm-model-runner` | Forward pass, weight loading, autotuning |
| `rvllm-gpu` | CUDA abstractions, cuBLAS, kernel loader, vendored cublaslt |
| `rvllm-fusion` | JIT kernel compiler, PTX emitter, LLVM NVPTX backend |
| `rvllm-kv-cache` | Paged KV cache (f16 + FP8) |
| `rvllm-attention` | Attention backends (FA3, GQA, split-KV) |
| `rvllm-speculative` | Speculative decoding (self-draft) |
| `rvllm-tp` | Tensor parallelism (NCCL, Megatron-LM sharding) |
| `rvllm-tokenizer` | HuggingFace tokenizer wrapper |

## Install

```bash
# From crates.io
cargo install rvllm

# From PyPI
pip install rvllm
```

Or build from source:

```bash
git clone https://github.com/m0at/rvllm
cd rvllm
cargo build --release --features cuda
```

## Quick Start

```bash
# Serve Qwen2.5-7B
rvllm serve --model Qwen/Qwen2.5-7B --dtype half

# Benchmark (direct engine, no HTTP)
rvllm benchmark --model Qwen/Qwen2.5-7B --dtype half --n "1,4,16,64,128"

# With FP8 weights (halves VRAM for weights)
RVLLM_FP8_WEIGHTS=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half

# With FP8 KV cache (doubles max sequences)
RVLLM_FP8_KV=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half

# With speculative decoding (faster N=1 latency)
RVLLM_SPECULATIVE=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half
```

## Benchmark Methodology

Both engines serve the same OpenAI-compatible `/v1/completions` endpoint. Direct engine benchmarks use the built-in `rvllm benchmark` command (no HTTP overhead). HTTP benchmarks use `bench/loadtest.py` (async Python client with aiohttp). Head-to-head comparison via `bench/compare_vllm.sh`.

Each engine runs on its own vast.ai H100 SXM 80GB instance -- separate GPUs, clean CUDA state, no cross-contamination.

See [docs/arch.md](docs/arch.md) for the full forward pass trace, [docs/benchmark-history.md](docs/benchmark-history.md) for optimization history, and [docs/cutlass-epilogue-spec.md](docs/cutlass-epilogue-spec.md) for the CUTLASS fusion roadmap.
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
VLLM_RS_URL=http://localhost:8000 python3 -m pytest tests/api_compat/ -v
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
| GptOssForCausalLM | OpenAI GPT-OSS 20B | Working (eager decode) |

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
- 11 model architectures (Llama, Mistral, Qwen2, Phi, Gemma, GPT-NeoX, StableLM, Cohere, Mixtral MoE, DeepSeek MoE, GPT-OSS)
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
