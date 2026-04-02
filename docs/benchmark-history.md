# Benchmark History

All results greedy decoding, 512 tokens/request unless noted. (Prior to 2026-03-30: 32 tokens/request.)

## Phase 8 (2026-04-01) -- Stock vLLM vs rvllm-lite vs rvLLM, apples-to-apples H100

This is the current public comparison set used in the README, GitHub Pages, and paper.
Model: Qwen2.5-7B f16 on H100 SXM 80GB. Direct engine runs use 256 output tokens.
HTTP runs use 200 requests at concurrency 32 with `max_tokens=256`.

### Direct engine

| N | stock vLLM 0.6.3.post1 | rvllm-lite | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|---:|
| 1 | 133.7 | 133.9 | 120.6 | 0.90x |
| 4 | 543.3 | 542.8 | 427.9 | 0.79x |
| 8 | 926.1 | 925.4 | 845.8 | 0.91x |
| 16 | 1,934.5 | 1,664.8 | 1,648.9 | 0.85x |
| 32 | 3,197.1 | 2,994.5 | 3,170.0 | 0.99x |

### HTTP serving

| Stack | Single req tok/s | Load tok/s | Avg latency ms | Idle VRAM |
|---|---:|---:|---:|---:|
| stock vLLM 0.6.3.post1 | 41.0 | 2,861.9 | 2,061.9 | 71.9 GiB |
| rvllm-lite | 128.6 | 131.8 | 43,334.9 | 71.9 GiB |
| rvLLM | 120.2 | 2,723.2 | 2,685.2 | 75.2 GiB |

The key diagnostic result is `rvllm-lite`: it stays near stock `vLLM` on direct engine
but collapses under HTTP load, which isolates the practical serving overhead to the
Python server/scheduler layer rather than the underlying `vLLM` engine.

## Phase 9 (2026-04-01) -- Matched 10k lifecycle race, fresh H100 SXM

Launch-to-finished lifecycle timing on clean H100 SXM 80GB boxes with Qwen2.5-7B f16.
This is fixed-length greedy decode, `14,080` completion tokens total, and `32`
concurrency. The stable matched numbers are:

| Engine | first_completion_sec | benchmark_wall_sec | shutdown_sec | launch_to_finished_tokens_sec | completion_tokens | throughput_tok_per_sec | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| rvLLM | 26.04 | 4.12 | 0.51 | 30.33 | 14,080 | 3,419.4 | 1,043.3 |
| stock vLLM | 32.10 | 3.20 | 0.81 | 35.51 | 14,080 | 4,394.6 | 716.1 |

This is the useful read:
- direct engine is already `0.99x` stock `vLLM` at `N=32`
- `rvLLM` wins the full launch-to-finished race because startup is materially shorter
- stock `vLLM` still wins steady-state HTTP throughput
- the remaining gap is in the HTTP + prefill + request-handling path

## Phase 7 (2026-04-01) -- Architecture hardening + INT4 GEMV (H100 SXM 80GB)

Focus: correctness fixes, portability, and new quantization kernel. No throughput regression at high concurrency (12,312 tok/s N=128 unchanged). N=1 improved from 98 to 121 tok/s via fused GEMV + 128-bit vectorized loads.

### N=1 decode path comparison (Qwen2.5-7B f16, 128 tok/req, H100 SXM 80GB)

| Decode Path | N=1 tok/s | HBM BW util | Env var |
|---|---:|---:|---|
| FusedDecode (default) | 121 | 55% | -- |
| CublasGemvDecode | 118 | 84% (standalone) | RVLLM_CUBLAS_DECODE=1 |
| MegakernelDecode | ~50 | -- | RVLLM_MEGAKERNEL=1 |
| PersistentDecode | ~51 | -- | RVLLM_PERSISTENT=1 |
| Theoretical ceiling (f16) | 222 | 100% | -- |

FusedDecode wins end-to-end despite cuBLAS achieving higher standalone BW utilization (84% vs 55%) because the fused kernels eliminate kernel launch overhead and GMEM round-trips between norm and GEMV.

### Correctness and portability fixes

1. **RoPE table cap 8K -> 32K**: RoPE cos/sin precomputed tables were capped at 8192 positions. Qwen2.5 supports 131K context via YaRN, so any sequence beyond 8K silently read zeroed memory for positional embeddings. Raised cap to 32768 (sufficient for all current workloads, dynamic allocation TODO for 131K).
2. **Megakernel portability**: `hidden_size=3584` was hardcoded in the megakernel. Now passed as a kernel parameter, enabling non-Qwen models (Llama 3.1 hidden_size=4096, etc.).
3. **FA3 GQA overflow guard**: `V3_GQA_MAX_HPG` raised from 8 to 16. Qwen2.5-7B has 7 heads/group, Llama 3.1 has 8 -- both fit in 8, but the constant was also used as an array size and was 1 short of the actual max when accounting for loop unrolling. Raising to 16 prevents silent overflow.
4. **Scheduler anti-thrashing**: Preemption capped at 4 sequences per scheduler call with push_back (re-queue at end). Previously, at high concurrency the scheduler could preempt and re-schedule the same sequences in a tight loop, burning CPU without making forward progress.

### INT4/W4A16 GEMV kernel

Added `gemv_int4.cu` with 4 fused variants:
- Standalone INT4 GEMV
- Fused QKV (add+norm+INT4 GEMV)
- Fused gateup (add+norm+INT4 GEMV)
- Fused silu+down (silu*mul+INT4 GEMV)

Per-group asymmetric quantization (group_size=128, zero-point + scale per group). Not yet wired to Rust dispatch -- kernel compilation and unit tests only. Will halve weight memory bandwidth vs f16 at N=1.

### Bandwidth analysis

| Path | N=1 tok/s | BW util | Notes |
|---|---:|---:|---|
| FusedDecode | 121 | 55% | Fused norm+GEMV, 128-bit loads |
| cuBLAS standalone | -- | 84% | HGEMM only (no norm overhead) |
| cuBLAS end-to-end | 118 | ~53% | Separate norm adds launch overhead |
| Theoretical f16 | 222 | 100% | 14GB weights / 3.35 TB/s HBM BW |

The 55% BW utilization gap (vs theoretical 222 tok/s) comes from: kernel launch overhead (~15%), norm/activation kernels (~10%), KV cache writes (~8%), attention kernel (~7%), other (~5%). INT4 would halve the weight read bottleneck, potentially reaching ~180 tok/s at N=1.

## Phase 5d (2026-03-31) -- FA3 v2 kernel optimization (H100 SXM 80GB)

Rewrote FlashAttention-3 decode kernels (both non-GQA and GQA):
1. Warp-parallel QK^T: 8 positions per round via warp shuffle (was 1 sequential)
2. Parallel softmax: block-wide reductions (was single-thread)
3. Bank-conflict-free smem: padded KV stride (+2 half), score stride (+1 float)
4. GQA: all Q heads pre-loaded into registers, V reused across heads in P@V
5. Occupancy: __launch_bounds__(256, 2) for both kernels (GQA was 1)

Syncs per tile (GQA, 7 heads): ~38 vs ~924 (24x fewer).

### A/B test: FA3 v1 vs v2 (direct engine, same codebase, same H100)

| N | FA3 v1 | FA3 v2 | change |
|---|---|---|---|
| 1 | 47 | 75 | +60% |
| 16 | 864 | 1,537 | +78% |
| 32 | 1,716 | 3,020 | +76% |
| 64 | 3,182 | 5,447 | +71% |
| 128 | 5,371 | 8,652 | +61% |

### Qwen2.5-7B f16 -- rvLLM vs vLLM 0.18 (512 tok/req, HTTP steady-state)

| N | rvLLM 5d | vLLM 0.18 (eager) | gap | was (5c) |
|---|---|---|---|---|
| 16 | 1,503 | 1,714 | vLLM 1.14x | 1.95x |
| 32 | 2,902 | 3,431 | vLLM 1.18x | 2.01x |
| 64 | 5,120 | 6,677 | vLLM 1.30x | 2.22x |
| 128 | 8,161 | 12,230 | vLLM 1.50x | 2.22x |

Gap narrowed from ~2x to 1.14-1.50x. Remaining gap is GEMM-bound:
vLLM's Triton autotuned GEMMs + mature continuous batching dominate at
high concurrency. FA3 v2 closed ~60% of the overall gap.

## Phase 5c (2026-03-31) -- Double-buffered scratch + alloc_zeros elimination (H100 SXM 80GB)

Double-buffered scratch in forward_gpu_only (graph-captured path): eliminates 54 per-layer
alloc+D2D copies per step, replaced with 1 copy at end. Changed alloc_zeros to unsafe alloc
in 16 locations (scratch init, linear, activation, norm, softmax, fused_ops).

### Qwen2.5-7B f16 -- rvLLM vs vLLM 0.18 (512 tok/req, same H100)

| N | rvLLM | vLLM 0.18 (eager) | ratio | vs 5b |
|---|---|---|---|---|
| 1 | 53 | -- | -- | -- |
| 16 | 878 | 1,714 | vLLM 1.95x | +1.7% |
| 32 | 1,711 | 3,431 | vLLM 2.01x | +2.5% |
| 64 | 3,006 | 6,677 | vLLM 2.22x | -5.8% |
| 128 | 5,499 | 12,230 | vLLM 2.22x | +7.0% |

Modest +7% at N=128 from graph overhead reduction. The ~2x gap is GEMM-bound:
vLLM's FlashAttention v3 + Triton GEMM autotuning + mature continuous batching
dominate. The per-layer allocation/D2D overhead was <1% of total step time.

## Phase 5b (2026-03-31) -- cublasLt build fix + vLLM comparison (H100 SXM 80GB)

Binary built with `--features cuda,cublaslt`. Fixed cublaslt_raw module registration,
FFI type mismatches, RefCell plan cache. Direct engine benchmark.

CUTLASS headers installed, 4 fused CuTE kernels loaded from JIT cache.
The ~2x gap is real -- vLLM's FlashAttention v3 + Triton fused ops + mature
continuous batching scheduler dominate at high concurrency on 7B.

### Qwen2.5-7B f16 -- rvLLM vs vLLM 0.18 (512 tok/req, same H100)

| N | rvLLM | vLLM 0.18 (eager) | ratio |
|---|---|---|---|
| 16 | 863 | 1,612 | vLLM 1.87x |
| 24 | 1,291 | 2,614 | vLLM 2.02x |
| 32 | 1,669 | 3,231 | vLLM 1.94x |
| 48 | 2,426 | 5,014 | vLLM 2.07x |
| 64 | 3,192 | 6,417 | vLLM 2.01x |
| 96 | 4,193 | 9,611 | vLLM 2.29x |
| 128 | 5,137 | 12,132 | vLLM 2.36x |

vLLM using enforce_eager=True (no CUDA graphs, no torch.compile -- compilation
crashed due to torch version mismatch). With graphs+compile vLLM would be faster still.

### Qwen2.5-1.5B f16 (128 tok/req)

| N | tok/s |
|---|---|
| 128 | 19,551 |

## Phase 5 (2026-03-30) -- Kernel fusion swarm (H100 SXM 80GB)

Direct engine benchmark (no HTTP). Fused kernels: add+norm+QKV GEMV, add+norm+gateup GEMV,
silu+down GEMV, GQA-optimized FA3 attention. Prefill uses fused QKV/gateup for all N.

| N | tok/s | vs Phase 4 (A100) | Notes |
|---|---|---|---|
| 1 | 240 | 1.88x | Fused kernels + cublasLt split-K |
| 4 | 1,201 | 2.22x | |
| 8 | 2,328 | 2.13x | |
| 16 | 4,229 | 2.00x | |
| 32 | 8,575 | 2.47x | |
| 64 | 15,812 | 3.89x | GQA attention 6x less KV bandwidth |
| 128 | 26,161 | 4.11x | |
| 256 | 40,714 | 4.89x | |

### Qwen2.5-7B f16 (H100 SXM 80GB, direct engine)

| N | tok/s | wall_ms |
|---|---|---|
| 1 | 108 | 296 |
| 4 | 544 | 235 |
| 8 | 1,073 | 238 |
| 16 | 2,019 | 253 |
| 32 | 3,911 | 261 |
| 64 | 7,300 | 280 |
| 128 | 12,624 | 324 |

N=256 hits KV cache limits at 0.9 gpu-memory-utilization for 7B.

Note: Phase 4 was on A100, Phase 5 on H100. H100 has ~2x raw bandwidth and ~3x tensor
core FLOPS vs A100. The per-hardware improvement from fusion alone is ~1.5-2x.

## Phase 4 (2026-03-28) -- CUDA graph + cublasLt (A100 80GB SXM4)

Measured with concurrent Python HTTP requests after graph capture fix.

| N | tok/s | ms/tok | Notes |
|---|---|---|---|
| 1 | 128 | 7.7 | 22.7% mem BW utilization |
| 4 | 540 | - | |
| 8 | 1,091 | - | |
| 16 | 2,118 | - | |
| 32 | 3,467 | - | |

Per-token overhead: 5.95ms (77% of total), theoretical peak 574 tok/s.

## Phase 3 (earlier) -- Sampling + attention backend

Previous head-to-head numbers (measured with bench/run.sh batched harness, not reproducible with current code):

| N | rvLLM (tok/s) | vLLM 0.18 (tok/s) |
|---|---|---|
| 1 | 117 | 69 |
| 4 | 882 | 256 |
| 8 | 1,213 | 517 |
| 16 | 1,391 | 1,060 |
| 32 | 1,434 | 1,943 |
| 48 | 3,918 | 2,887 |
| 64 | 4,796 | 3,828 |
| 96 | 5,965 | 5,197 |
| 128 | 7,380 | 6,400 |
| 256 | 9,905 | 9,437 |
| 512 | 10,291 | 10,771 |
| 768 | 10,235 | -- |
| 1024 | 10,051 | 12,740 |

Note: These numbers included optimizations (fused QKV, fused gate+up, vectorized float4, packed HtoD, pre-alloc buffers) that were lost in subsequent code changes and are being re-implemented in Phase 5.

## Phase 2 -- FP16 inference

- 8,339 tok/s peak at N=768
- Matched vLLM at N=48-128

## Phase 1 -- FP32 baseline

- 3,191 tok/s peak at N=512
- 86 tok/s single-sequence

## B200 Results (FP32, earlier)

| N | Tokens | Wall time | tok/s |
|---|---|---|---|
| 1 | 32 | 279ms | 114 |
| 64 | 2,048 | 798ms | 2,566 |
| 256 | 8,192 | 2,106ms | 3,889 |
| 768 | 24,576 | 6,227ms | 3,946 |
| 4,096 | 131,072 | 34,002ms | 3,854 |
