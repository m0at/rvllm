# rvLLM: High-performance LLM inference in Rust

A from-scratch Rust rewrite of [vLLM](https://github.com/vllm-project/vllm) focused on single-card, high-throughput serving with explicit control over kernels, memory, and startup behavior.

310 commits, 31 crates, ~76K lines of Rust, 253 source files. Zero Python in the serving hot path.

## Reproduce the Benchmark

One command spins up a fresh vast.ai H100, builds rvLLM from source, pulls the model from HuggingFace, runs the lifecycle race against stock vLLM, and drops the results locally:

```bash
VASTAI_API_KEY=<your_key> ./race.sh
```

Results land in `bench/combined_results_h100_lifecycle.json`. The instance stays up after the run so you can inspect logs; destroy it with the printed `vastai destroy instance <id>` command when done.

## Current Status

**Near parity with vLLM 0.19.0 at batch 32-64, still behind at batch 1 and batch 128.** Measured on the same H100 SXM 80GB, same Qwen2.5-7B snapshot, same decode length (`output-len=128`), same date (`April 4, 2026`).

What rvLLM does well:
- **`0.99x` vLLM at `N=32` and `1.00x` at `N=64`** on direct engine
- **~50 MB container image** vs ~15 GB for Python vLLM
- **35-second build from source**, no pip, no PyTorch, no `torch.compile`
- **54 CUDA kernels** with no-fallback validation -- silent degradation is treated as a bug
- **JIT fused kernels** 2-7.5x faster than our hand-written CUDA on M=1 decode microbenchmarks
- **Safe-max VRAM control**: `--gpu-memory-reserve-gb` + explicit `--num-gpu-blocks` + `--num-cpu-blocks`

What was wrong before:
1. **Batch-1 normal decode picked the wrong path.** The default `T=1` path was still taking the slower legacy fused decode path instead of `CublasGemvDecode`.
2. **Batched GEMM policy was only half-wired.** The runner thought it had a hybrid policy, but batched QKV could still opportunistically take CUTLASS just because the `.so` existed.
3. **The docs were still describing that older architecture.** This README and the paper now reflect the current dispatch logic and the current benchmark set.

What changed:
1. **Batch-1 default decode now prefers `CublasGemvDecode`.**
2. **Batched GEMM has a real `Hybrid` strategy.**
3. **`Hybrid` now means exactly this**: QKV, O-proj, and down-proj use cuBLAS/cublasLt; GateUp+SiLU uses CUTLASS.
4. **Explicit batched policy override exists**: `RVLLM_BATCHED_GEMM_STRATEGY={cublas,hybrid,cutlass}`.

Where the remaining gap comes from:
1. **Single-stream decode**: `N=1` is still materially behind vLLM.
2. **vLLM's more mature decode stack**: FlashAttention 3 + `torch.compile` + full decode CUDA graphs.
3. **cublasLt autotune stability**: some cached algos are still flaky on a few shapes and need better fallback handling.

## Current H100 Comparison

Qwen2.5-7B f16 on H100 SXM 80GB, 128 output tokens per request, greedy decode (temperature=0). Both engines on the same physical GPU, clean CUDA state.

### Direct Engine (April 4, 2026)

| N | vLLM 0.19.0 tok/s | rvLLM tok/s | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 165.5 | 127.9 | 0.77x |
| 32 | 4,467.7 | 4,407.5 | 0.99x |
| 64 | 7,972.1 | 7,964.0 | 1.00x |
| 128 | 13,903.5 | 13,148.3 | 0.95x |

This is the current honest comparison set. rvLLM is effectively tied at `N=32` and `N=64`, a few percent behind at `N=128`, and still substantially behind at `N=1`.

### Historical Comparison (vLLM 0.6.3, March 2026)

Earlier benchmark against the older vLLM 0.6.3.post1 showed rvLLM at 0.79x-0.99x. Those numbers are retained in [docs/benchmark-history.md](docs/benchmark-history.md) for optimization history but are not the current headline comparison.

### FA3 v3 Attention Kernel

FA3 v3 adds cp.async bulk global-to-shared copies (128-bit, bypasses registers/L1) and split-KV for long context (distributes KV tiles across thread blocks). Combined with no-fallback kernel validation that eliminates silent performance degradation from missing kernels:

| N | v2 tok/s | v3+nofallback tok/s | Change |
|---:|---:|---:|---|
| 1 | 75 | 98 | +31% |
| 16 | 1,537 | 2,122 | +38% |
| 32 | 3,020 | 3,957 | +31% |
| 64 | 5,447 | 7,451 | +37% |
| 128 | 8,652 | 12,312 | +42% |

## Supported Model Architectures

13 architectures with full GPU forward pass support:

| Architecture | Models | Notes |
|---|---|---|
| `LlamaForCausalLM` | Llama 2/3/3.1, CodeLlama, Vicuna | Auto context expansion to 8K |
| `MistralForCausalLM` | Mistral 7B | Sliding window attention |
| `Qwen2ForCausalLM` | Qwen2, Qwen2.5 | Verified benchmark model |
| `CohereForCausalLM` | Command-R | |
| `GPTNeoXForCausalLM` | Pythia, GPT-NeoX, StableLM | |
| `StableLMForCausalLM` | StableLM-2 | |
| `GemmaForCausalLM` | Gemma 1.0 | |
| `Gemma2ForCausalLM` | Gemma 2 | |
| `DeepseekV2ForCausalLM` | DeepSeek-V2, DeepSeek-Coder-V2 | MoE with shared experts |
| `MixtralForCausalLM` | Mixtral 8x7B, 8x22B | Sparse MoE |
| `NemotronHMoE` | Nemotron-H | Hybrid MoE (latest merge) |
| `Phi3SmallForCausalLM` | Phi-3-small | |
| `SentenceTransformer` | BERT, E5, BGE embedding models | Embedding-only forward pass |

Weight formats: SafeTensors (single file and sharded index), GGUF (Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q8_0).

## Decode Paths

Six runtime-selectable decode strategies, each with different performance trade-offs. No fallbacks -- if the selected path's kernels are missing, it fails loud.

| Decode Path | N=1 tok/s | Selection | Notes |
|---|---:|---|---|
| CublasGemvDecode (default) | 127.9 | `T=1, f16 weights` | Current normal batch-1 decode path |
| FusedDecode | legacy | `RVLLM_CUBLAS_DECODE=0` | Older fused f16 GEMV path, now opt-in |
| MegakernelDecode | ~50 | Internal | All 28 layers in 1 kernel launch (instruction tape interpreter) |
| PersistentDecode | ~51 | Internal | SM-DAG cooperative kernel per layer |
| Fp8Decode | auto | `RVLLM_FP8_WEIGHTS=1` | cublasLt FP8 GEMMs (when FP8 weights present) |
| Batched (`Hybrid`) | auto | `T>=2` | QKV/O/down on cuBLAS or cublasLt, GateUp+SiLU on CUTLASS |

For batched decode and prefill, the current default policy is:
- `RVLLM_BATCHED_GEMM_STRATEGY=hybrid` when CUTLASS is available
- `RVLLM_BATCHED_GEMM_STRATEGY=cublas` otherwise

The old bug here was that the runner selected a "hybrid" mode conceptually, but the actual per-op routing was inconsistent. QKV could still slip onto CUTLASS just because the shared library was present. The current code makes the batched policy explicit and stable.

The megakernel packs all 28 transformer layers and the LM head into a single CUDA kernel launch, driven by an instruction tape that sequences GEMV, RMSNorm, RoPE, attention, and activation operations with double-buffered residuals and per-layer KV cache. See [crates/rvllm-model-runner/README.md](crates/rvllm-model-runner/README.md) for the full decode path architecture.

## API Endpoints

Full OpenAI-compatible HTTP API via axum:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/completions` | POST | Text completion (streaming + non-streaming) |
| `/v1/chat/completions` | POST | Chat completion with tool/function calling |
| `/v1/chat/completions/tools` | POST | Dedicated tool calling endpoint |
| `/v1/responses` | POST | Unified Responses API (stored turns, background, streaming) |
| `/v1/responses/:id` | GET | Retrieve a stored response |
| `/v1/responses/:id/input_items` | GET | List response input items |
| `/v1/embeddings` | POST | Compute embeddings |
| `/v1/models` | GET | List available models |
| `/v1/batches` | POST | Submit batch inference (JSONL) |
| `/v1/batches/:id` | GET | Check batch status |
| `/v1/batches/:id/output` | GET | Retrieve batch results (JSONL) |
| `/v1/batches/:id/cancel` | POST | Cancel a running batch |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus exposition |

The Responses API (`/v1/responses`) supports multi-turn conversations via `previous_response_id` and `conversation` references, background execution, stored response retrieval, function tool calling with streaming argument deltas, reasoning configuration, and the `include` parameter for controlling output payloads like logprobs.

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

Three-tier kernel system, with rTriton as the unified kernel layer:

**rTriton: Triton-style JIT compiler + cuBLAS integration (`crates/rtriton/`)**

A standalone Rust reimplementation of OpenAI's Triton GPU kernel compiler, combined with our battle-tested cuBLAS tricks. One crate, one CUDA graph, zero Python:

- **Triton-style builder DSL**: SSA IR with 30+ ops, 7 optimization passes (DCE, constant fold, fusion, coalescing, shared memory planning, software pipelining), PTX codegen targeting sm_80+
- **8 pre-built LLM kernels**: RMSNorm, fused residual+RMSNorm, RoPE, SiLU*mul, tiled GEMM, GEMV, persistent GEMM (stream-K), flash attention decode (online softmax, paged KV, GQA)
- **cuBLAS integration**: FP8 cublasLt plan cache, autotuned algorithm selection (32 candidates/shape), graph workspace pre-allocation, M-threshold routing (cublasLt for M<=32, cuBLAS for M>32)
- **Mixed execution graph**: Triton JIT kernels and cuBLAS GEMMs captured in a single CUDA graph -- zero launch overhead for the full decode layer
- **Decode layer plan**: 9 operations per layer (5 Triton + 4 cuBLAS), buffer allocation with liveness-based interval coloring for memory reuse
- **50 tests passing**, compiles on Mac without CUDA (all GPU code behind `cfg(feature = "cuda")`)

See [crates/rtriton/README.md](crates/rtriton/README.md) for the full rTriton compiler documentation.

A single decode step at c=128 concurrency:
```
[rTriton] fused_residual_rmsnorm     -- 1 kernel, eliminates 2 GMEM round-trips
[cuBLAS]  QKV GEMM (M=128)          -- autotuned cublasLt, FP8 optional
[rTriton] RoPE + KV cache write     -- fused, no intermediate alloc
[rTriton] Flash Attention Decode     -- online softmax, paged KV
[cuBLAS]  O-proj GEMM               -- autotuned
[rTriton] fused_residual_rmsnorm
[cuBLAS]  gate_up GEMM              -- autotuned
[rTriton] SiLU * mul                -- fused activation
[cuBLAS]  down GEMM                 -- autotuned
```

**Tier 1: JIT-compiled fused kernels (current production)**
- Rust PTX emitter generates shape-specialized fused kernels at model load
- 2-7.5x faster than hand-written CUDA for M=1 decode
- Patterns: RMSNorm+GEMV, Add+RMSNorm+GEMV, SiLU*Mul+GEMV
- No nvcc dependency -- pure Rust string-based PTX generation

| Fused Kernel | JIT (us) | Hand-written (us) | Speedup |
|---|---:|---:|---|
| Add+RMSNorm+QKV GEMV [1,4608,3584] | 5.5 | 10.6 | **1.92x** |
| Add+RMSNorm+GateUp GEMV [1,37888,3584] | 19.3 | 98.6 | **5.12x** |
| SiLU*Mul+Down GEMV [1,3584,18944] | 9.5 | 70.7 | **7.48x** |
| RMSNorm+QKV GEMV [1,4608,3584] | 5.3 | 10.8 | **2.03x** |

Per-step savings at N=1 (28 layers): **4.2ms** = estimated **1.8x** single-sequence speedup.

See [crates/rvllm-fusion/README.md](crates/rvllm-fusion/README.md) for the full JIT compiler documentation.

**Tier 2: Hand-written CUDA kernels (54 kernels)**
- Fused decode: add+norm+QKV+bias, RoPE+cache, GQA attention, O-proj+gateup, silu+down
- FP8 E4M3 variants for all projections
- TMA async-prefetch GEMV, WGMMA tensor core GEMV
- Split-KV paged attention for long context
- Megakernel: 28-layer single-launch decode with instruction tape interpreter

**Tier 3: cuBLAS/cublasLt (batched decode M>1)**
- Autotuned algorithm selection (32 candidates benchmarked per shape at startup)
- Vendored cublaslt type shim for cudarc 0.19 compatibility
- cublasLt for M<=32, cuBLAS for M>32

**LLVM NVPTX backend (experimental)**
- Full compiler: Fusion IR -> LLVM IR -> NVPTX -> PTX via inkwell
- Same backend as Triton (LLVM NVPTX)
- Gated behind `--features llvm` (requires LLVM 20.1)

### Crate Map

31 crates organized by layer:

| Layer | Crate | Purpose |
|---|---|---|
| **Binary** | `rvllm-server` | CLI binary (`rvllm serve`, `rvllm benchmark`, `rvllm info`) |
| **API** | `rvllm-api` | axum HTTP server, all OpenAI-compatible routes, SSE streaming, telemetry |
| **Engine** | `rvllm-engine` | Sync `LLMEngine` + async `AsyncLLMEngine` + dedicated GPU thread `AsyncGpuLLMEngine` |
| **Scheduling** | `rvllm-scheduler` | Continuous batching, chunked prefill, FCFS/priority/SJF policies, preemption |
| | `rvllm-sequence` | Sequence, SequenceGroup, status FSM (Waiting/Running/Swapped/Finished) |
| **Execution** | `rvllm-executor` | Executor trait, single-GPU and multi-GPU executors, tensor parallel config |
| | `rvllm-worker` | GPU worker, CUDA graph capture/replay (35 pre-captured batch sizes) |
| **Model** | `rvllm-model-loader` | SafeTensors + GGUF loading, GPU upload, tensor-parallel sharding |
| | `rvllm-model-runner` | Forward pass, 13 architectures, 6 decode paths, megakernel |
| **GPU** | `rvllm-gpu` | CUDA abstractions, cuBLAS/cublasLt, kernel loader (~60 kernel modules), allocator, NCCL |
| | `rvllm-fusion` | JIT PTX compiler (2-7.5x faster than hand-written CUDA) |
| | `rtriton` | Triton-style GPU kernel compiler + cuBLAS integration (research crate) |
| **Attention** | `rvllm-attention` | Flash Attention, paged attention, sliding window, split-KV, GQA |
| **Cache** | `rvllm-kv-cache` | Paged KV cache (f16 + FP8), reshape/cache operations |
| | `rvllm-block-manager` | Ref-counted blocks, copy-on-write, prefix caching, swap management |
| | `rvllm-memory` | GPU/CPU memory pools with free-list allocation |
| **Sampling** | `rvllm-sampling` | Temperature, top-k/p, min-p, repetition/frequency/presence penalties, guided decoding (JSON schema, regex) |
| **Tokenizer** | `rvllm-tokenizer` | HuggingFace tokenizer wrapper, ChatML/Harmony templates, tool call parsing, incremental streaming decode |
| **Quantization** | `rvllm-quant` | GPTQ, AWQ, FP8, GGUF Q4/Q5/Q8 detection and config |
| **Speculative** | `rvllm-speculative` | Self-draft speculative decoding (draft from first N layers of target) |
| **Parallelism** | `rvllm-tp` | Tensor parallelism via NCCL (Megatron-LM column/row sharding) |
| **Observability** | `rvllm-telemetry` | Prometheus metrics (15+ counters/gauges/histograms), structured logging, OTLP traces |
| **Acceleration** | `rvllm-zig` | Zig SIMD backend (softmax, argmax, weight conversion -- NEON + AVX-512) |
| **Vision** | `rvllm-vision` | Image preprocessing (decode/resize/normalize), vision encoder trait, CLIP/SigLIP config |
| **Multimodal** | `rvllm-core` | Shared types, errors, multimodal data types (PixelValues, ImageFeatures, RawImage) |
| **Config** | `rvllm-config` | Engine/model/cache/scheduler/parallel/device/telemetry/vision config, TOML + CLI loading |
| **Bench** | `rvllm-bench` | Criterion benchmarks for sampling hot paths |
| **Bindings** | `rvllm-python` | PyO3 module (`import rvllm`) -- sampler, tokenizer, config |

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
| 5d | FA3 v2 (warp-parallel attention rewrite) | 8,652 | Mar 31 |
| 6 | FA3 v3 (cp.async + split-KV) + no-fallback | 12,312 | Mar 31 |
| 7 | Architecture hardening + INT4 kernel | 12,312 | Apr 1 |
| 8 | Chunked prefill scheduler + lifecycle race | 3,419 (N=32) | Apr 2 |
| 9 | Responses API + Harmony templates + vision types | -- | Apr 3 |
| 10 | Qwen/Nemotron integrations + GGUF loader + mock smoke path | -- | Apr 3 |
| 11 | v0.19 upgrade: zero-bubble scheduling, DBO, MLA, Gemma 4, Blackwell, MXFP8, autotune cache | 4,385 (N=32) | Apr 4 |
| 12 | Batch-1 dispatch fix: default to `CublasGemvDecode` | 127.9 (N=1) | Apr 4 |
| 13 | Batched GEMM `Hybrid` policy fix | 13,148 (N=128) | Apr 4 |

### What Differs from vLLM

Against vLLM 0.19.0 (April 2026), rvLLM now has two different stories:

- **Batched direct engine is close**: `0.99x` at `N=32`, `1.00x` at `N=64`, `0.95x` at `N=128`
- **Single-stream decode is not**: `0.77x` at `N=1`

The biggest architectural difference is that rvLLM's normal batched path is now an explicit per-op hybrid GEMM stack, while vLLM still has the stronger overall decode stack at `N=1` through FlashAttention 3, `torch.compile`, and fuller decode graph integration.

Root causes of the gap (in order of impact):

1. **GEMM tuning**: vLLM uses Triton autotuned GEMMs with 6+ months of per-model config tuning + torch.compile; we use cuBLAS with freshly-implemented disk-cached autotuning.
2. **Attention**: vLLM uses FlashAttention-3 (Tri Dao's official CUDA with TMA + warp specialization); our FA3 v3 uses cp.async + split-KV but lacks TMA.
3. **Scheduler maturity**: vLLM's zero-bubble + DBO has had months of production hardening; ours was just implemented.
4. **Quantization breadth**: vLLM supports GPTQ, AWQ, Marlin, FP8, MXFP8, NVFP4. We have FP8, MXFP8, INT4/W4A16, and GGUF (Q4/Q5/Q8).

What rvLLM does better:

1. **Owns the whole stack** -- Rust server, worker, scheduler, and kernels without a Python runtime in the serving hot path
2. **Deployment footprint** -- ~50 MB container image vs ~15 GB; 35 sec build from source
3. **JIT fused kernels** -- `rvllm-fusion` PTX emission beats the hand-written CUDA versions by 2-7.5x on the measured decode microbenchmarks
4. **Kernel discipline** -- no-fallback validation and 6 decode paths (`FusedDecode`, cuBLAS GEMV, megakernel, persistent, FP8, batched)
5. **Safe-max memory control** -- reserve-based startup sizing plus explicit GPU/CPU block overrides
6. **No GIL, no GC** -- deterministic Rust execution, parallel scheduling via Rayon

## Zig SIMD Acceleration

Hot-path sampling primitives and weight conversion use a Zig SIMD backend (`rvllm-zig`). `@Vector(16, f32)` maps to NEON on aarch64, AVX-512 on x86_64 servers (`-mcpu=x86_64_v4`). Benchmarked on Apple M5 (128K vocab = LLaMA-3 scale):

| Operation | Zig SIMD | Rust (scalar) | Speedup |
|---|---:|---:|---|
| softmax (128K) | 134 us | 192 us | **1.44x** |
| argmax (128K) | 9.2 us | 58 us | **6.31x** |
| argmax+logprob fused (128K) | 131 us | 213 us | **1.62x** |
| scale (128K) | 6.9 us | 6.8 us | 1.0x (memory-bound) |
| bf16->f16 (16M) | 637 us | -- | -- |
| f32->f16 (16M) | 1.07 ms | -- | -- |

The fused `argmax_logprob` kernel computes greedy token selection + log-probability in 2 SIMD passes (argmax+exp-sum) instead of 4 separate scalar passes. `apply_min_p` uses a logit-space threshold (`max + ln(min_p)`) to avoid softmax allocation entirely.

End-to-end sampling improvement (criterion, 128K vocab, vs pure Rust):

| Sampler | Change |
|---|---|
| greedy (128K) | **-17%** (141 us) |
| greedy (32K) | **-10%** (35 us) |
| repetition penalty | **-7%** (143 us) |
| top-p | no change (1.20 ms, sort-dominated) |
| top-k | no change (358 us, quickselect-dominated) |

Weight conversion throughput (16M elements = one 4096x4096 weight matrix):

| Conversion | Throughput |
|---|---|
| bf16 -> f16 | 48.9 GB/s |
| f32 -> f16 | 58.3 GB/s |

Zig is a hard build dependency -- no fallbacks.

## CPU-Side Operations

Operations between GPU forward passes, measured on Apple M5 and Xeon:

| Operation | Rust | Python (numpy) | Speedup |
|---|---|---|---|
| Combined penalties (rep+freq+pres) | 2.6 us | 63 us | **24x** |
| Repetition penalty (2K tokens) | 3.1 us | 34 us | **11x** |
| Multinomial sampling (32K vocab) | 12 us | 66 us | **5.5x** |
| Top-P nucleus (128K vocab) | 1.6 ms | 6.9 ms | **4.3x** |
| Batch sampling (64 seqs, Rayon) | 4.3 ms | 36.4 ms | **8.5x** |

## Guided Decoding

The sampling layer includes a constrained decoding engine (`rvllm-sampling/src/guided.rs`) that enforces output format at the token level:

- **JSON mode**: Forces syntactically valid JSON output
- **JSON Schema**: Compiles JSON schemas (max depth 64) into a `SchemaNode` tree, then computes valid next characters at each generation step
- **Regex**: Pattern-constrained generation
- **VocabTable**: Maps token IDs to their text representations for constraint checking

Request-level control via the `response_format` field in SamplingParams.

## Profile-Guided Autotuning

rvLLM includes a two-stage tuning system: **nsys profiling** identifies which kernels to optimize, then **cublasLt autotuning** finds the fastest algorithm for each GEMM shape.

### Stage 1: Profile with nsys

```bash
NSYS=/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nsys

$NSYS profile --stats=true -o profile_output \
  ./target/release/rvllm benchmark \
  --model /root/models/Qwen2.5-7B --dtype half --n 32 --output-len 32
```

This prints a kernel ranking table showing exactly where GPU time goes:

```
 Time(%)  Total(ms)  Calls   Avg(us)   Name
    5.3     27.9      980    28.5      fa3_v3_decode_gqa_kernel
    4.4     23.2     4089     5.7      fused_residual_rmsnorm_f16_kernel
    2.8     14.8     2044     7.2      silu_mul_interleaved_f16_kernel
    2.6     14.0     6132     2.3      add_bias_f16_kernel        <-- 6132 launches!
    ...
```

The kernel with the most total time and the most launches is your optimization target. In the example above, `add_bias_f16_kernel` has 6,132 separate launches that should be fused into the GEMM epilogue.

See [docs/profiling.md](docs/profiling.md) for the full profiling guide.

### Stage 2: cublasLt Algorithm Autotuning

When built with `--features cuda,cublaslt`, rvLLM benchmarks 32 cublasLt algorithm candidates for each GEMM shape at startup. Results are cached to `~/.cache/rvllm/autotune.json` so subsequent runs skip the benchmarking phase.

```bash
# First run: autotuning takes 1-2 minutes (benchmarks ~24 shapes x 32 algorithms)
cargo build --release --features cuda,cublaslt
./target/release/rvllm serve --model Qwen/Qwen2.5-7B --dtype half

# Second run: instant (reads from cache)
./target/release/rvllm serve --model Qwen/Qwen2.5-7B --dtype half
```

The cache is keyed by `(gpu_name, m, n, k, dtype)` so different GPUs and models get separate tuning results. Override the cache path with `RVLLM_AUTOTUNE_CACHE=/path/to/cache.json`.

The cublasLt path also enables **bias epilogue fusion**: bias-add is folded into the GEMM output instead of launching a separate kernel. This eliminates thousands of kernel launches per benchmark (6,132 `add_bias_f16_kernel` calls become zero).

### Stage 3: CUPTI-Based Full-Pipeline Tuning (rvllm-autotune)

The `rvllm-autotune` crate (`crates/rvllm-autotune/`) goes beyond GEMM algorithm selection to profile and tune every kernel in the inference pipeline:

1. **Profile** (`CuptiProfiler`): captures per-kernel GPU timing via CUPTI activity API with nanosecond precision
2. **Rank** (`KernelRanker`): sorts kernels by total GPU time, classifies as Gemm/Attention/Norm/Activation/Memory, marks which are tunable (ours) vs library internals (cuBLAS)
3. **Sweep** (`ConfigSweeper`): generates alternative launch configs (block sizes, shared memory, tile parameters) and benchmarks each
4. **Cache** (`TuneCache`): persists winning configs to `~/.cache/rvllm/tune.json`

The workflow:
```
nsys profile  -->  identify top kernels  -->  autotune those kernels  -->  cache results
     |                    |                          |                         |
  CUPTI API        KernelRanker              ConfigSweeper               TuneCache
```

## Telemetry

Prometheus-compatible metrics exposed at `/metrics`, plus structured logging via `tracing`:

**Histograms**: request latency, time-to-first-token (TTFT), inter-token latency (ITL), forward pass time, sample time, API request duration

**Gauges**: tokens/sec, running requests, waiting requests, GPU cache usage %, in-flight API requests, worker tokens/sec

**Counters**: preemptions, total requests, finished requests, prompt tokens, generation tokens, forward passes, tokens sampled, engine steps, API requests, API errors

Configurable via `--prometheus-port`, `--otlp-endpoint`, and `--log-level`.

## Configuration

### CLI Arguments

```bash
rvllm serve --model <path_or_repo> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace repo ID or local path |
| `--dtype` | `auto` | `auto`, `float32`, `float16`, `bfloat16` |
| `--max-model-len` | 2048 | Maximum context length |
| `--gpu-memory-utilization` | 0.90 | Fraction of GPU memory for KV cache |
| `--gpu-memory-reserve-gb` | 0.0 | VRAM to leave free for scratch (GiB) |
| `--num-gpu-blocks` | auto | Fixed GPU block count override |
| `--num-cpu-blocks` | auto | Fixed CPU block count override |
| `--tensor-parallel-size` | 1 | Number of GPUs for TP |
| `--max-num-seqs` | 256 | Max concurrent sequences |
| `--max-num-batched-tokens` | 8192 | Max tokens per batch |
| `--max-prefill-chunk` | 128 | Max prompt tokens per prefill step |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Bind port |
| `--log-level` | info | Minimum log level |
| `--disable-telemetry` | false | Turn off metrics collection |

### Environment Variables

| Variable | Description |
|---|---|
| `RVLLM_FP8_WEIGHTS=1` | Quantize all projection weights to FP8 E4M3 at startup |
| `RVLLM_FP8_KV=1` | Store KV cache in FP8 (doubles concurrent sequences) |
| `RVLLM_CUBLAS_DECODE=1` | Use cuBLAS GEMV decode instead of fused kernels |
| `RVLLM_INT4_DECODE=1` | W4A16 GEMV decode path (planned) |
| `RVLLM_SPECULATIVE=1` | Enable self-draft speculative decoding |
| `RVLLM_SPECULATIVE_K=3` | Draft tokens per speculative step |
| `RVLLM_SPECULATIVE_DRAFT_LAYERS=N` | Layers for self-draft (default: total/4) |
| `RVLLM_AUTOTUNE=1` | Enable cublasLt algorithm autotuning at startup |
| `RVLLM_L2_PERSIST=1` | Enable L2 cache persistence hints |
| `RVLLM_PTX_DIR` | Override directory for compiled PTX kernels |
| `HF_TOKEN` | HuggingFace auth token for gated models |
| `VLLM_HOST` / `VLLM_PORT` | Override server bind address/port |
| `VLLM_BATCH_OUTPUT_DIR` | Directory for batch API output files |

### Feature Flags (Cargo)

| Feature | Description |
|---|---|
| `cuda` | Real CUDA GPU support (required for inference) |
| `cublaslt` | cuBLASLt autotuned GEMM plans |
| `zig` | Zig SIMD backend for sampling hot paths |
| `llvm` | LLVM NVPTX backend for fusion compiler (requires LLVM 20.1) |
| `mock-gpu` | CPU-only mock GPU for testing |

## Deployment Tooling

### vast.ai Integration

Full lifecycle automation for H100/B200 instances:

| Script | Purpose |
|---|---|
| `race.sh` | One-command lifecycle race vs stock vLLM |
| `deploy/vastai-provision.sh` | Provision a vast.ai instance |
| `deploy/vastai-deploy.sh` | Build + deploy rvLLM to instance |
| `deploy/vastai-benchmark.sh` | Run benchmark suite |
| `deploy/vastai-teardown.sh` | Destroy instance |
| `deploy/setup_instance.sh` | Instance environment setup |
| `deploy/deploy_and_bench.sh` | Combined deploy + benchmark |
| `deploy/rsync_and_run.sh` | Incremental sync + run |
| `install.sh` | Local install script |

### Benchmark Harnesses

| Script | Purpose |
|---|---|
| `bench/run.sh` | Full benchmark sweep |
| `bench/quick_bench.sh` | Fast smoke test |
| `bench/compare_vllm.sh` | Side-by-side vLLM comparison |
| `bench/loadtest.sh` | HTTP load testing |
| `bench/bench_cutlass.sh` | CUTLASS kernel benchmarks |
| `bench/bench_jit.sh` | JIT compiler benchmarks |
| `bench/bench_long_context.sh` | Long context attention benchmarks |
| `bench/verify_fusion.sh` | Verify fused kernel correctness |
| `deploy/benchmark_client.py` | Python HTTP benchmark client |
| `deploy/compare_results.py` | Result comparison tool |
| `deploy/vllm_direct_bench.py` | Direct vLLM engine benchmark |
| `bench/loadtest.py` | Python load test driver |

### CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`): `cargo check --workspace` + `cargo test --workspace` on every push.

GitHub Pages deployment (`.github/workflows/pages.yml`) for documentation.

## Install

```bash
# From crates.io
cargo install rvllm --features cuda,cublaslt

# From PyPI
uv pip install rvllm
```

Or build from source:

```bash
git clone https://github.com/m0at/rvllm
cd rvllm
cargo build --release --features cuda
```

## Quick Start

```bash
# Serve Qwen2.5-7B with safe-max VRAM sizing
rvllm serve --model Qwen/Qwen2.5-7B --dtype half --gpu-memory-utilization 1.0 --gpu-memory-reserve-gb 2.0

# Benchmark (direct engine, no HTTP)
rvllm benchmark --model Qwen/Qwen2.5-7B --dtype half --n "1,4,8,16,32" --output-len 256
```

### Optional Features

**FP8 Weights** (`RVLLM_FP8_WEIGHTS=1`): Quantizes all projection weights to FP8 E4M3 at startup. Halves weight memory bandwidth for single-stream decode (M=1 GEMV). Does NOT improve batched throughput -- at M>=8, f16 tensor cores already saturate compute and the f16->fp8 cast adds overhead. Use for latency-sensitive single-user workloads, not high-concurrency serving.

**FP8 KV Cache** (`RVLLM_FP8_KV=1`): Stores KV cache in FP8, doubling the number of concurrent sequences at the cost of minor precision loss.

**Speculative Decoding** (`RVLLM_SPECULATIVE=1`): Self-draft speculative decoding using the first N layers of the target model as a draft. Primarily beneficial for large models (70B+) where single-token decode latency is high enough that the draft+verify overhead is worthwhile.

```bash
RVLLM_SPECULATIVE=1 RVLLM_SPECULATIVE_K=3 rvllm serve --model meta-llama/Llama-3-70B --dtype half
```

## Current Operational Notes

| Metric | stock vLLM | rvllm-lite | rvLLM |
|---|---:|---:|---:|
| Idle VRAM after load | 71.9 GiB | 71.9 GiB | 75.2 GiB |
| Post-HTTP VRAM | 72.1 GiB | 72.0 GiB | 75.3 GiB |
| Safe-max startup control | `--gpu-memory-utilization` | `--gpu-memory-utilization` | `--gpu-memory-utilization`, `--gpu-memory-reserve-gb`, `--num-gpu-blocks` |

For `rvLLM`, `--gpu-memory-utilization 1.0` now works with an explicit reserve, which is safer than guessing a fixed fraction and hoping startup scratch allocations fit.

## Deployment

| Metric | rvLLM | Python vLLM |
|---|---|---|
| Install | `cargo install rvllm --features cuda,cublaslt` | `pip install vllm` (+ PyTorch) |
| Container image | ~50 MB | ~15 GB |
| Build from source | 35 sec | N/A |
| Kernel compilation | 30 sec (54 PTX via nvcc) + 0 sec (JIT at runtime) | 0 or ~60s (torch.compile) |
| GPU architectures | sm_80, sm_86, sm_89, sm_90 | Same + ROCm |

## Benchmark Methodology

Both engines serve the same OpenAI-compatible `/v1/completions` endpoint. Direct engine benchmarks use the built-in `rvllm benchmark` command (no HTTP overhead). The latest published HTTP comparison used `deploy/benchmark_client.py`; the repo also includes `bench/loadtest.py` and `bench/compare_vllm.sh` for broader load and side-by-side runs.

Each engine runs on its own vast.ai H100 SXM 80GB instance -- separate GPUs, clean CUDA state, no cross-contamination.

See [docs/arch.md](docs/arch.md) for the full forward pass trace, [docs/benchmark-history.md](docs/benchmark-history.md) for optimization history, and [docs/cutlass-epilogue-spec.md](docs/cutlass-epilogue-spec.md) for the CUTLASS fusion roadmap.

To run the full lifecycle race yourself, see [Reproduce the Benchmark](#reproduce-the-benchmark) at the top.
