# rvLLM: High-performance LLM inference in Rust

A from-scratch Rust rewrite of [vLLM](https://github.com/vllm-project/vllm) focused on single-card, high-throughput serving with explicit control over kernels, memory, and startup behavior.

## Reproduce the Benchmark

One command spins up a fresh vast.ai H100, builds rvLLM from source, pulls the model from HuggingFace, runs the lifecycle race against stock vLLM, and drops the results locally:

```bash
VASTAI_API_KEY=<your_key> ./race.sh
```

Results land in `bench/combined_results_h100_lifecycle.json`. The instance stays up after the run so you can inspect logs; destroy it with the printed `vastai destroy instance <id>` command when done.

## What Is Already Clearly Better

- **Direct engine gets close by `N=32`**: `rvLLM` reaches `3,170 tok/s` vs `3,197 tok/s` for stock `vLLM`.
- **Launch-to-finished lifecycle is faster right now**: in the matched 10k race, `rvLLM` finishes in `30.33s` vs `35.51s` for stock `vLLM`.
- **HTTP steady-state still has obvious headroom**: the same matched race lands at `3,419.4 tok/s` for `rvLLM` vs `4,394.6 tok/s` for stock `vLLM`, so the remaining gap is still serving-path work.
- **`rvllm-lite` cleanly exposes serving overhead**: near-stock direct engine, then `131.8 tok/s` over HTTP.
- **The 10k lifecycle path is verified on real GPU**: `rvLLM` completes `14,080` completion tokens at concurrency `32` in `30.33s` launch-to-finished (`26.04s` to first completion, `4.12s` bench tail, `3,419.4 tok/s`).
- **VRAM startup is safer to drive hard**: reserve-based fill via `--gpu-memory-reserve-gb`, plus explicit `--num-gpu-blocks` and `--num-cpu-blocks`.
- **Kernel behavior is explicit**: 54 CUDA kernels, no-fallback validation, and a Rust PTX fusion path with measured `2-7.5x` decode microbench wins vs our hand-written CUDA equivalents.

## Current H100 Comparison

Qwen2.5-7B f16 on H100 SXM 80GB. Direct engine runs use 256 output tokens. HTTP runs use 200 requests at concurrency 32 with `max_tokens=256`.

![Current H100 comparison](docs/assets/h100-comparison.svg)

All measurements below use the same setup. Stock `vLLM` was benchmarked through its own OpenAI server, `rvllm-lite` is the intermediate Python serving layer, and `rvLLM` is the Rust server.

### Direct Engine

| N | stock vLLM 0.6.3.post1 | rvllm-lite | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|---:|
| 1 | 133.7 | 133.9 | 120.6 | 0.90x |
| 4 | 543.3 | 542.8 | 427.9 | 0.79x |
| 8 | 926.1 | 925.4 | 845.8 | 0.91x |
| 16 | 1,934.5 | 1,664.8 | 1,648.9 | 0.85x |
| 32 | 3,197.1 | 2,994.5 | 3,170.0 | 0.99x |

`rvllm-lite` direct is essentially the stock `vLLM` library path, so the direct-engine gap is really between stock `vLLM` and `rvLLM`. In the latest verified run, `rvLLM` is still behind at low and mid concurrency, but it closes to near-parity by `N=32`.

### HTTP Serving

| Stack | Single request tok/s | 200-req throughput tok/s | Avg latency ms | Idle VRAM |
|---|---:|---:|---:|---:|
| stock vLLM 0.6.3.post1 | 41.0 | 2,861.9 | 2,061.9 | 71.9 GiB |
| rvllm-lite | 128.6 | 131.8 | 43,334.9 | 71.9 GiB |
| rvLLM | 120.2 | 2,723.2 | 2,685.2 | 75.2 GiB |

The key result is that `rvllm-lite` keeps near-stock direct-engine speed but collapses over HTTP, which isolates most of the practical overhead to the Python serving/scheduling layer rather than the underlying `vLLM` engine. `rvLLM`'s server path is in the same practical class as stock `vLLM`, but the direct engine still needs more work to win consistently.

Historical phase-by-phase numbers, including the earlier `12,312 tok/s @ N=128` run, live in [docs/benchmark-history.md](docs/benchmark-history.md).

### 10k Lifecycle Race

Verified on isolated H100 SXM 80GB instances with Qwen2.5-7B f16. This is the matched fixed-length greedy lifecycle harness at `32` concurrency. The first `64`-concurrency `rvLLM` attempt crashed in `gpu-step`, so `32` is the current stable race setting.

| Engine | Concurrency | first_completion_sec | benchmark_wall_sec | shutdown_sec | launch_to_finished_tokens_sec | completion_tokens | throughput_tok_per_sec | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rvLLM | 32 | 26.04 | 4.12 | 0.51 | 30.33 | 14,080 | 3,419.4 | 1,043.3 |
| stock vLLM | 32 | 32.10 | 3.20 | 0.81 | 35.51 | 14,080 | 4,394.6 | 716.1 |

This run matters because it lines up with the direct-engine `0.99x` parity result at `N=32`: the raw card path is already basically there, while the remaining shortfall is in the HTTP + prefill + request-handling path. The headline read right now is simple: `rvLLM` wins launch-to-finished, stock `vLLM` still wins steady-state HTTP throughput.

Artifacts:
- `rvLLM` result: `/root/bench_results/rvllm_result_c32.json`
- `rvLLM` benchmark output: `/root/bench_results/rvllm_bench.json`

### FA3 v3 Attention Kernel

FA3 v3 adds cp.async bulk global-to-shared copies (128-bit, bypasses registers/L1) and split-KV for long context (distributes KV tiles across thread blocks). Combined with no-fallback kernel validation that eliminates silent performance degradation from missing kernels:

| N | v2 tok/s | v3+nofallback tok/s | Change |
|---:|---:|---:|---|
| 1 | 75 | 98 | +31% |
| 16 | 1,537 | 2,122 | +38% |
| 32 | 3,020 | 3,957 | +31% |
| 64 | 5,447 | 7,451 | +37% |
| 128 | 8,652 | 12,312 | +42% |

### N=1 Decode Paths

Multiple decode paths are now available, each with different trade-offs:

| Decode Path | N=1 tok/s | Notes |
|---|---:|---|
| FusedDecode (default) | 121 | Fused f16 GEMV kernels, 55% HBM BW utilization |
| CublasGemvDecode | 118 | Separate norm + cuBLAS HGEMM, 84% BW util standalone |
| MegakernelDecode | ~50 | All 28 layers in 1 kernel launch |
| PersistentDecode | ~51 | Cooperative kernel per layer |
| Fp8Decode | auto | cublasLt FP8 GEMMs (when FP8 weights present) |
| INT4 (planned) | -- | W4A16 GEMV kernel ready, Rust wiring TODO |
| Theoretical ceiling | 222 | 100% HBM BW, f16 weights |

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

### Current Operational Notes

| Metric | stock vLLM | rvllm-lite | rvLLM |
|---|---:|---:|---:|
| Idle VRAM after load | 71.9 GiB | 71.9 GiB | 75.2 GiB |
| Post-HTTP VRAM | 72.1 GiB | 72.0 GiB | 75.3 GiB |
| Safe-max startup control | `--gpu-memory-utilization` | `--gpu-memory-utilization` | `--gpu-memory-utilization`, `--gpu-memory-reserve-gb`, `--num-gpu-blocks` |

For `rvLLM`, `--gpu-memory-utilization 1.0` now works with an explicit reserve, which is safer than guessing a fixed fraction and hoping startup scratch allocations fit.

### Zig SIMD Acceleration

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
| Install | `cargo install rvllm --features cuda,cublaslt` | `pip install vllm` (+ PyTorch) |
| Container image | ~50 MB | ~15 GB |
| Build from source | 35 sec | N/A |
| Kernel compilation | 30 sec (54 PTX via nvcc) + 0 sec (JIT at runtime) | 0 or ~60s (torch.compile) |
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

Three-tier kernel system, with rTriton as the unified kernel layer:

**rTriton: Triton-style JIT compiler + cuBLAS integration (`crates/rtriton/`)**

A standalone Rust reimplementation of OpenAI's Triton GPU kernel compiler, combined with our battle-tested cuBLAS tricks. One crate, one CUDA graph, zero Python:

- **Triton-style builder DSL**: SSA IR with 30+ ops, 7 optimization passes (DCE, constant fold, fusion, coalescing, shared memory planning, software pipelining), PTX codegen targeting sm_80+
- **8 pre-built LLM kernels**: RMSNorm, fused residual+RMSNorm, RoPE, SiLU*mul, tiled GEMM, GEMV, persistent GEMM (stream-K), flash attention decode (online softmax, paged KV, GQA)
- **cuBLAS integration**: FP8 cublasLt plan cache, autotuned algorithm selection (32 candidates/shape), graph workspace pre-allocation, M-threshold routing (cublasLt for M<=32, cuBLAS for M>32)
- **Mixed execution graph**: Triton JIT kernels and cuBLAS GEMMs captured in a single CUDA graph -- zero launch overhead for the full decode layer
- **Decode layer plan**: 9 operations per layer (5 Triton + 4 cuBLAS), buffer allocation with liveness-based interval coloring for memory reuse
- **50 tests passing**, compiles on Mac without CUDA (all GPU code behind `cfg(feature = "cuda")`)

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

**Tier 2: Hand-written CUDA kernels (54 kernels)**
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
| 5d | FA3 v2 (warp-parallel attention rewrite) | 8,652 | Mar 31 |
| 6 | FA3 v3 (cp.async + split-KV) + no-fallback | 12,312 | Mar 31 |
| 7 | Architecture hardening + INT4 kernel | 12,312 | Apr 1 |

Note: Phase 5d and earlier numbers used 512 tok/req. Phase 6+ uses 128 tok/req (same model, same hardware). The Phase 6 improvement comes from FA3 v3 cp.async attention, CUTLASS header integration, and killing all silent kernel fallback paths that were masking missing fused kernels. Phase 7 focused on correctness and portability (RoPE 32K, megakernel param fix, FA3 overflow guard, scheduler anti-thrashing) plus adding INT4 GEMV and cuBLAS decode paths.

### What Differs from vLLM

In the latest verified run, `rvLLM` ranges from `0.79x` to `0.99x` stock `vLLM` on direct engine, wins the matched `10k` launch-to-finished lifecycle race (`30.33s` vs `35.51s`), and still trails stock `vLLM` on steady-state HTTP throughput (`3,419.4` vs `4,394.6 tok/s`). Root causes, in order of impact:

1. **GEMM tuning**: vLLM uses Triton autotuned GEMMs + torch.compile; we use stock cuBLAS heuristics. This is the dominant remaining gap at high concurrency.
2. **Attention**: vLLM uses FlashAttention-3 (Tri Dao's official CUDA, heavily optimized with TMA, warp specialization, pipelining); our FA3 v3 uses cp.async and split-KV but still lacks TMA and full warp specialization.
3. **Scheduler**: vLLM has mature continuous batching with sophisticated prefill/decode interleaving, chunked prefill, and priority preemption. Ours is simpler.
4. **Quantization**: vLLM supports GPTQ, AWQ, SqueezeLLM, Marlin, FP8, etc. We have FP8 and INT4/W4A16 (kernel ready, dispatch wiring in progress).

What rvLLM does better:

1. **Owns the whole stack** -- Rust server, worker, scheduler, and kernels without a Python runtime in the serving hot path
2. **Safe-max memory control** -- reserve-based startup sizing plus explicit GPU/CPU block overrides
3. **JIT fused kernels** -- `rvllm-fusion` PTX emission beats the hand-written CUDA versions by 2-7.5x on the measured decode microbenchmarks
4. **Kernel discipline** -- no-fallback validation and multiple decode paths (`FusedDecode`, cuBLAS GEMV, megakernel, persistent, FP8, planned INT4)
5. **Server overhead vs rvllm-lite** -- the current H100 run shows the Rust server path staying competitive while the intermediate Python layer serializes itself under load

### What's Inside

| Crate | Purpose |
|---|---|
| `rvllm` | HTTP API (axum), CLI |
| `rvllm-engine` | Async engine, continuous batching |
| `rvllm-worker` | GPU worker, CUDA graph management |
| `rvllm-model-runner` | Forward pass, weight loading, autotuning |
| `rvllm-gpu` | CUDA abstractions, cuBLAS, kernel loader, vendored cublaslt |
| `rvllm-fusion` | JIT kernel compiler, PTX emitter, LLVM NVPTX backend |
| `rtriton` (experimental sidecar) | Triton-style GPU kernel compiler + cuBLAS integration research crate |
| `rvllm-zig` | Zig SIMD backend (softmax, argmax, weight conversion) |
| `rvllm-kv-cache` | Paged KV cache (f16 + FP8) |
| `rvllm-attention` | Attention backends (FA3 v3 cp.async + split-KV, GQA) |
| `rvllm-speculative` | Speculative decoding (self-draft) |
| `rvllm-tp` | Tensor parallelism (NCCL, Megatron-LM sharding) |
| `rvllm-tokenizer` | HuggingFace tokenizer wrapper |

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

```bash
RVLLM_FP8_WEIGHTS=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half
```

**FP8 KV Cache** (`RVLLM_FP8_KV=1`): Stores KV cache in FP8, doubling the number of concurrent sequences at the cost of minor precision loss.

```bash
RVLLM_FP8_KV=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half
```

**cuBLAS GEMV Decode** (`RVLLM_CUBLAS_DECODE=1`): Uses separate RMSNorm + cuBLAS HGEMM for N=1 decode instead of fused GEMV kernels. Achieves 84% HBM bandwidth utilization in standalone cuBLAS calls, 118 tok/s end-to-end. Slightly slower than FusedDecode (121 tok/s) due to extra kernel launch overhead, but useful for profiling and as a reference baseline.

```bash
RVLLM_CUBLAS_DECODE=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half
```

**INT4 Decode** (`RVLLM_INT4_DECODE=1`, planned): W4A16 GEMV decode path using per-group asymmetric INT4 quantization. The CUDA kernel (`gemv_int4.cu`, 4 variants: standalone, fused QKV, fused gateup, fused silu+down) is written but not yet wired to Rust dispatch. Will halve weight memory bandwidth vs f16.

**Speculative Decoding** (`RVLLM_SPECULATIVE=1`): Self-draft speculative decoding using the first N layers of the target model as a draft. Produces multiple tokens per step when the draft is accepted. Primarily beneficial for large models (70B+) where single-token decode latency is high enough that the draft+verify overhead is worthwhile. For 7B models, the acceptance rate with self-draft at 1/4 depth is too low to overcome the verify prefill cost. Requires a proper draft KV cache for production use (currently experimental).

```bash
# 70B+ models (recommended)
RVLLM_SPECULATIVE=1 RVLLM_SPECULATIVE_K=3 rvllm serve --model meta-llama/Llama-3-70B --dtype half

# Configuration
RVLLM_SPECULATIVE_K=5          # draft tokens per step (default: 3)
RVLLM_SPECULATIVE_DRAFT_LAYERS=8  # layers for self-draft (default: total_layers/4)
```

## Benchmark Methodology

Both engines serve the same OpenAI-compatible `/v1/completions` endpoint. Direct engine benchmarks use the built-in `rvllm benchmark` command (no HTTP overhead). The latest published HTTP comparison used `deploy/benchmark_client.py`; the repo also includes `bench/loadtest.py` and `bench/compare_vllm.sh` for broader load and side-by-side runs.

Each engine runs on its own vast.ai H100 SXM 80GB instance -- separate GPUs, clean CUDA state, no cross-contamination.

See [docs/arch.md](docs/arch.md) for the full forward pass trace, [docs/benchmark-history.md](docs/benchmark-history.md) for optimization history, and [docs/cutlass-epilogue-spec.md](docs/cutlass-epilogue-spec.md) for the CUTLASS fusion roadmap.

To run the full lifecycle race yourself, see [Reproduce the Benchmark](#reproduce-the-benchmark) at the top.
