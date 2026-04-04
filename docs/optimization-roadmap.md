# rvLLM Optimization Roadmap

> Historical planning document. Superseded by the April 4, 2026 benchmark set and dispatch-policy fixes.
> Current public direct-engine comparison on H100 / Qwen2.5-7B / `output-len=128`:
> `N=1 127.9 vs 165.5`, `N=32 4407.5 vs 4467.7`, `N=64 7964.0 vs 7972.1`, `N=128 13148.3 vs 13903.5` (rvLLM vs vLLM 0.19.0).

## Current State (April 1, 2026)

Verified coherent: **12,312 tok/s** at N=128 on H100 SXM 80GB, Qwen2.5-7B f16, direct engine.

54 hand-written CUDA kernels. 5 decode paths: FusedDecode (121 tok/s), CublasGemvDecode (118 tok/s, 84% BW util), Megakernel (50), Persistent (51), Fp8Decode. Theoretical N=1 f16 ceiling: 222 tok/s.

### Direct Engine Scaling (H100 SXM 80GB, Qwen2.5-7B f16, 128 tok/req)

```
N=1:    98 tok/s     -- single request, latency-bound
N=4:    548 tok/s    -- low-batch regime
N=16:   2,122 tok/s  -- near parity with vLLM (0.96x)
N=32:   3,957 tok/s  -- 0.86x vLLM
N=64:   7,451 tok/s  -- 0.94x vLLM
N=128:  12,312 tok/s -- 0.85x vLLM, production sweet spot
```

### Python vLLM 0.18 Comparison (H100, direct engine)

```
N=1:    170 tok/s    -- vLLM 1.73x faster
N=4:    665 tok/s    -- vLLM 1.21x faster
N=16:   2,202 tok/s  -- near parity (0.96x)
N=32:   4,585 tok/s  -- vLLM 1.16x faster
N=64:   7,888 tok/s  -- near parity (0.94x)
N=128:  14,528 tok/s -- vLLM 1.18x faster
```

## Bottleneck Analysis by Batch Size Region

### Region 1: N=1 to N=32 (Latency-Bound)

**Observed:** 101-925 tok/s. Linear scaling with N but far below GPU capacity.

**Bottleneck:** GPU SM underutilization. At N=1, a cuBLAS hgemm for Q projection (shape [1, 1536] x [1536, 1536]) produces only 12 output tiles (128x128 each). An A100 has 108 SMs. Only 11% of the GPU is working. K/V projections ([1, 256]) produce just 2 tiles -- 2% utilization.

**What would help:**
- **cublasLt with split-K** (ready, behind feature flag). Split the K dimension across multiple thread blocks to create more parallelism. Expected: 20-40% gain at N=1, tapering to 0% at N=64+. Implementation exists in `cublaslt_ops.rs`, needs wiring into the forward path.
- **Persistent GEMM kernels.** Instead of launching a new kernel per GEMM, keep thread blocks resident and feed them work via global memory queues. Eliminates kernel launch overhead (~10us per launch x 7 GEMMs x 28 layers = ~2ms per step).
- **Speculative decoding.** Draft model generates K=4-8 candidate tokens, target verifies in one forward pass. Multiplies effective throughput by acceptance rate. Scaffolded in `crates/rvllm-speculative/` but not wired.

### Region 2: N=32 to N=64 (Inflection Zone)

**Observed:** 925 to 4,063 tok/s. A 4.4x jump between N=32 and N=64.

**Bottleneck:** The sharp inflection suggests a threshold effect -- likely CUDA graph capture kicking in at a specific padded batch size, or the continuous batching scheduler switching from prefill-dominated to decode-dominated batching.

**What would help:**
- **Profile with RVLLM_PROFILE=1** (implemented) to identify exactly where time is spent at N=32 vs N=64. The profiling infrastructure records CUDA events around each phase (HtoD, embedding, per-layer norm/QKV/attention/MLP, LM head, DtoH).
- **Fix CUDA graph capture/replay** (reverted due to coherency bug). The graph agent's implementation replayed with stale metadata. A correct implementation would update input buffer contents in-place before replay, not re-record. This alone could smooth the inflection and add 10-20% across all N.
- **Fused QKV projection** (3 GEMMs -> 1). Agent wrote the code but it wasn't merged into gpu_layer.rs. The weight concatenation is cached after first call. Reduces kernel launches by 2 per layer.
- **Fused gate+up projection** (2 GEMMs -> 1). Same pattern. Reduces kernel launches by 1 per layer.

### Region 3: N=64 to N=256 (Scaling Region)

**Observed:** 4,063 to 8,316 tok/s. Healthy scaling, GPU becoming well-utilized.

**Bottleneck:** Memory bandwidth for weight reads. At N=128, each decode step reads all 2.5GB of model weights (FP16) from HBM. A100 HBM bandwidth is 2 TB/s. Reading 2.5GB takes ~1.25ms. With 128 tokens produced per step, that's 128/1.25ms = 102,400 tok/s theoretical peak. We achieve 6,360 -- only 6.2% of theoretical. The gap is: attention KV cache reads, activation memory traffic, kernel launch overhead, CPU scheduling time, and PCIe transfers.

**What would help:**
- **FP8 weights** (E4M3). Halves weight bandwidth from 2.5GB to 1.25GB per step. cuBLAS FP8 GEMM via cublasLtMatmul. Existing `fp8_kv.cu` kernel handles FP8 KV cache. Expected: 30-60% throughput gain in this region.
- **Async weight prefetch.** Use `cudaMemPrefetchAsync` or L2 pinning (`cudaAccessPolicyWindow`) to prefetch layer N+1's weights while layer N computes. A100 L2 is 40MB; Qwen2.5-1.5B layer weights are ~90MB so partial prefetch helps.
- **FlashDecoding (split-KV parallelism).** At 512-token context (our benchmark standard), split-KV is now relevant -- splitting KV across multiple thread blocks provides 1.5-3x attention speedup. At 2048-4096 token contexts, gains reach 3-5x. Kernels exist in `split_kv_attention.cu` but wiring had a bug (kernel args mismatch) -- needs careful re-integration.

### Region 4: N=256 to N=1024 (Plateau)

**Observed:** 8,316 to 8,578 tok/s. Nearly flat. This is the compute ceiling.

**Bottleneck:** cuBLAS GEMM throughput. At N=1024, the GEMMs are large enough to fully saturate the GPU's tensor cores. The remaining gap vs vLLM (8,578 vs 12,740 = 0.67x) is pure kernel efficiency.

**What would help:**
- **CuTE/CUTLASS attention kernels.** The current FA2 decode kernel uses scalar FMA for QK dot products, leaving A100's 312 TFLOPS of FP16 tensor cores completely unused. Rewriting with CuTE MMA atoms (`SM80_16x8x16_F32F16F16F32`) would route attention through tensor cores. Expected: 1.3-1.5x for the attention phase.
- **F16 shared memory in attention.** Current kernel promotes f16 KV cache to f32 in shared memory, doubling smem usage (64KB vs 32KB per tile). Keeping f16 in smem halves memory pressure and improves occupancy.
- **cp.async for overlapped KV loads.** Current kernel loads KV tiles synchronously. `cp.async` (sm_80+) enables loading tile N+1 while computing on tile N.
- **Warp-specialized attention.** Separate producer warps (load KV) from consumer warps (compute). Overlaps memory latency with compute within a single thread block. Works on sm_80 via `cp.async` + named barriers.
- **GQA-optimized attention.** Qwen2.5-1.5B has 12 query heads sharing 2 KV heads (6:1 ratio). Current kernel loads KV 6 times for the same 2 KV heads. A GQA-aware kernel loads KV once and processes all 6 query heads, reducing KV bandwidth by 6x.

## Priority-Ranked Optimization List

### Tier 1: Quick Wins (hours each, no kernel changes)

| # | Optimization | Expected Gain | Region | Risk |
|---|---|---|---|---|
| 1 | Fix CUDA graph capture/replay correctly | 10-20% all N | All | Medium (caused coherency bug before) |
| 2 | Wire cublasLt for decode GEMMs | 20-40% at N=1-32 | 1 | Low (behind feature flag) |
| 3 | Fuse QKV projection (code exists) | 5-10% all N | All | Low |
| 4 | Fuse gate+up projection (code exists) | 3-5% all N | All | Low |

### Tier 2: Kernel Optimization (days each)

| # | Optimization | Expected Gain | Region | Risk |
|---|---|---|---|---|
| 5 | GQA-aware attention (load KV once for 6 heads) | 15-30% at high context | 3-4 | Medium |
| 6 | CuTE tensor core MMA for QK/PV | 30-50% attention phase | 3-4 | High (new dependency) |
| 7 | F16 shared memory (no f32 promotion) | 10-20% attention | 3-4 | Medium |
| 8 | cp.async overlapped KV loads | 10-15% attention | 3-4 | Medium |
| 9 | Fix split-KV wiring (kernel args) | 20-40% at long context | 3 | Medium |

### Tier 3: Precision and Architecture (week+)

| # | Optimization | Expected Gain | Region | Risk |
|---|---|---|---|---|
| 10 | FP8 inference (E4M3 weights) | 30-60% all N | 2-4 | Medium |
| 11 | Hopper TMA for H100/H200 | 20-40% on Hopper | All | Low (additive) |
| 12 | Warp specialization (producer/consumer) | 15-30% attention | 3-4 | High |
| 13 | Speculative decoding | 2-4x at N=1 | 1 | Medium |
| 14 | Tensor parallelism for 70B models | Enables new models | N/A | High |

### Tier 4: Production and Scale

| # | Optimization | Expected Gain | Region | Risk |
|---|---|---|---|---|
| 15 | Kernel autotuning (per-GPU, per-shape) | 5-15% all N | All | Low |
| 16 | Larger model benchmarks (Llama-3.1-8B) | Credibility | N/A | Low |
| 17 | FP8 KV cache (wire existing kernel) | 2x KV blocks | 3-4 | Low |
| 18 | INT4 KV cache (KIVI-style) | 4x KV blocks | 3-4 | Medium |
| 19 | Pipeline parallelism (multi-node) | Enables 405B | N/A | High |

## Theoretical Peak Analysis

### A100 80GB, Qwen2.5-1.5B FP16

- **Model weights:** 3 GB (FP16)
- **HBM bandwidth:** 2,039 GB/s
- **Weight read time per step:** 3 GB / 2039 GB/s = 1.47 ms
- **Tokens per step at N=1024:** 1024
- **Theoretical peak (weight-bandwidth-bound):** 1024 / 1.47ms = **696,600 tok/s**
- **Current achieved:** 8,578 tok/s = **1.2% of theoretical**

The massive gap between theoretical and achieved comes from:
1. KV cache reads (~0.5-2 GB depending on context length)
2. Activation intermediates (read/write per layer)
3. Kernel launch overhead (~28 layers x ~9 kernels = 252 launches)
4. CPU scheduling time
5. PCIe metadata transfers
6. Softmax, norm, activation kernel time (non-GEMM compute)

### Realistic Target

With all Tier 1-2 optimizations:
- CUDA graphs eliminate kernel launch overhead: +15%
- Fused GEMMs reduce launches by 40%: +5%
- GQA-aware attention: +15%
- CuTE tensor cores for attention: +20%
- cublasLt split-K at low N: +20% (low N only)

Conservative combined: **8,578 * 1.4 = ~12,000 tok/s**
Optimistic combined: **8,578 * 1.7 = ~14,500 tok/s**

This would match or beat vLLM (12,740) across the full batch range.

## Development History

| Phase | Change | Peak tok/s | Date |
|---|---|---|---|
| Initial FP32 | Baseline | 3,191 | Mar 28, 2026 |
| FP16 hgemm + f16 KV | 5-agent swarm | 8,339 | Mar 28, 2026 |
| cudarc 0.19 + sm_120 | 20-agent swarm | 8,339 | Mar 28, 2026 |
| Kernel optimizations | 17-agent swarm | 8,578 (coherent) | Mar 28, 2026 |
| CUDA graph fix reverted | Bisected bug | 8,578 | Mar 28, 2026 |
| FA3 v3 + split-KV + no-fallback | Phase 6 | 12,312 | Mar 31, 2026 |
| Architecture hardening + INT4 kernel | Phase 7 | 12,312 | Apr 1, 2026 |

## rTriton: Unified Kernel Layer

As of March 30, 2026, `crates/rtriton/` provides a unified approach to closing the gap with vLLM:

**Why rTriton exists:** Research confirmed that vLLM's advantage is NOT from Triton GEMM kernels (cuBLAS wins at all shapes). The advantage comes from torch.compile fusing ~20-30 pointwise/reduction ops between GEMMs into ~6-9 Triton kernels, eliminating kernel launch overhead (~5-10us each) and GMEM round-trips.

**What rTriton does:** Replicates this fusion strategy in Rust:
- Triton-style JIT compiler for fused ops (rmsnorm, rope, silu_mul, attention) -- eliminates GMEM round-trips
- cuBLAS integration with all our tricks (FP8 plan cache, autotuning, graph workspace) -- proven fastest for GEMMs
- Mixed CUDA graph captures both in a single replay -- zero launch overhead

**Expected impact:** The full 9-op decode layer captured as one CUDA graph eliminates:
- 252 kernel launches per step (28 layers x 9 ops) -> 28 graph replays
- GMEM round-trips between pointwise ops (2x bandwidth savings for fused rmsnorm)
- FP8 descriptor recreation (cached plans, <1us per GEMM)

This subsumes Tier 1 (#1 CUDA graph, #3 fused QKV, #4 fused gate_up) and parts of Tier 2 (#5 GQA attention, #8 async loads) from the optimization list above.

**Status:** 50 tests passing, full IR pipeline validated, PTX codegen for all 8 kernels confirmed. P0 remaining: wire real CUDA driver calls and validate PTX through ptxas on GPU.

## Key Architectural Insight

The single largest remaining optimization is **CUDA graph capture/replay done correctly**. The broken implementation showed 10,291 tok/s (23% gain) but produced garbage because it replayed with stale block table pointers. A correct implementation must:

1. Pre-allocate persistent input buffers (token_ids, positions, block_tables, etc.)
2. Update buffer contents in-place before replay (memcpy, not re-allocate)
3. Only replay for decode steps with the same padded batch size
4. Never replay during prefill (variable sequence lengths)
5. Validate output coherency on every change

The graph infrastructure exists in `crates/rvllm-worker/src/graph_runner.rs`. The capture/replay pool with padded batch sizes {1, 2, 4, 8, 16, 32} is implemented. What broke was the metadata update path -- block tables and positions need to be updated in the captured graph's input buffers, not in newly allocated buffers.

## Phase 7: Architecture Hardening (April 1, 2026)

Phase 7 focused on robustness, kernel coverage, and N=1 decode path diversity. No throughput regression at N=128 (still 12,312 tok/s).

### Changes

- **RoPE cap 8K -> 32K** -- extended rotary position embedding maximum from 8K to 32K context length
- **Megakernel hidden_size param** -- parameterized hidden dimension instead of compile-time constant, enabling multi-model support
- **FA3 MAX_HPG 8 -> 16** -- doubled maximum heads-per-group in FlashAttention3, supporting wider GQA ratios
- **Scheduler anti-thrash** -- hysteresis in the continuous batching scheduler to prevent rapid prefill/decode mode switching
- **INT4 W4A16 GEMV kernel** (`gemv_int4.cu`) -- 4 fused variants for weight-only INT4 quantized inference. Not yet wired to Rust engine; CUDA kernels validated standalone
- **cuBLAS GEMV decode path** -- separate norm + cuBLAS HGEMM achieving 84% HBM bandwidth utilization at N=1

### Five Decode Paths (N=1)

| Path | tok/s | Notes |
|---|---|---|
| FusedDecode | 121 | JIT fused norm+GEMV, fastest path |
| CublasGemvDecode | 118 | Separate norm + cuBLAS HGEMM, 84% BW util |
| Megakernel | 50 | Single-launch, hidden_size parameterized |
| Persistent | 51 | Resident thread blocks, work queue |
| Fp8Decode | -- | FP8 E4M3 weight path |

Theoretical N=1 f16 ceiling on H100: **222 tok/s** (3.35 TB/s effective BW / 15.1 GB model weights).

### Kernel Count: 50 -> 54

New kernels added in Phase 7:
1. `gemv_int4.cu` -- 4 fused INT4 W4A16 GEMV variants (norm+gemv, gemv, silu_mul+gemv, add+norm+gemv)

## What's Next

### Near-term (Phase 8)

- **Wire INT4 GEMV to Rust** -- integrate `gemv_int4.cu` kernels into the model runner decode path, enabling W4A16 quantized serving
- **L2 persistence** -- pin hot layer weights in H100 L2 cache (50 MB) using `cudaAccessPolicyWindow` to reduce HBM reads for repeated decode steps
- **Swap preemption** -- evict low-priority sequences to host memory under memory pressure, re-prefetch on demand, enabling longer context without OOM

### Medium-term

- **Close N=1 gap** -- FusedDecode at 121 tok/s vs theoretical 222 tok/s leaves 45% headroom; profile and eliminate remaining bottlenecks
- **INT4 + FP8 mixed precision** -- combine W4A16 GEMV for projections with FP8 KV cache for maximum memory efficiency
- **Multi-model megakernel** -- leverage parameterized hidden_size to serve multiple model architectures from a single binary
