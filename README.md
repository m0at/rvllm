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
