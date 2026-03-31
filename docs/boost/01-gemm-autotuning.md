# 01: GEMM Autotuning

Gap analysis and implementation plan for closing the GEMM performance gap between rvLLM and vLLM at high concurrency.

---

## 1. Current State Analysis

### 1.1 The Layered GEMM Stack in rvLLM

rvLLM has five GEMM dispatch paths, selected by the M-threshold router in `crates/rtriton/src/kernels/gemm_dispatch.rs` (lines 36-58):

- **M=1 + FP8**: `CublasFp8` -- cublasLt with FP8 E4M3 input, F16 output, cached plan per (m,n,k)
- **M <= 32**: `CublasLt` -- cublasLt with heuristic algo selection, 4 MiB workspace for split-K
- **M > 32**: `CublasStandard` -- stock cuBLAS `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- **Triton JIT** (opt-in via `prefer_triton=true`): tiled GEMM, GEMV, persistent GEMM via rTriton IR -> PTX
- **CUTLASS 3.x** (sm_90): `cutlass_gemm.cu` with TMA + WGMMA, loaded as `.so` via FFI in `cutlass_ffi.rs`

### 1.2 The Autotuner (`crates/rvllm-gpu/src/cublas_autotune.rs`)

The existing autotuner (lines 119-345) does the following at model load time:
- Requests up to `MAX_ALGOS = 32` candidate algorithms from `cublasLtMatmulAlgoGetHeuristic`
- Allocates scratch buffers (uninitialized) for A[m,k], B[n,k], C[m,n]
- Runs `WARMUP_ITERS = 3` iterations per candidate, then `BENCH_ITERS = 10` timed iterations
- Uses CUDA events for timing
- Caches the best `cublasLtMatmulAlgo_t` per (m,n,k) triple
- Workspace: fixed 4 MiB (`FP8_WORKSPACE_SIZE` in `cublaslt_ops.rs` line 22)
- Only autotunes at M=1 and M=128 (line 357-369), missing all intermediate batch sizes

### 1.3 Critical Limitation: the `hgemm_a_bt_into` path (line 127-236)

The `hgemm_a_bt_into` function, which is the workhorse for sub-slice writes, requests only `ret=1` from `cublasLtMatmulAlgoGetHeuristic` (line 194). It takes the single top heuristic without benchmarking. This means the most-used code path never benefits from the autotuner. The autotuner results stored in `CublasAutotuner` are never wired into this function.

### 1.4 The cuBLAS Warmup Path (`crates/rvllm-gpu/src/cublas.rs`, lines 89-152)

For the standard cuBLAS path (M > 32), `warmup_gemm_shapes` runs 3 dummy `cublasGemmEx` calls per shape to populate cuBLAS's internal algorithm cache. This relies entirely on NVIDIA's built-in heuristics -- no exploration of the algorithm space.

### 1.5 Workspace Budget

All paths use 4 MiB workspace. On H100, cuBLAS recommends 32 MiB for best split-K performance. The `CUBLASLT_M_THRESHOLD` is set to 32 (line 20 of `cublaslt_ops.rs`), but the comment on line 19 notes this is conservative: "cublasLt at M=128 is slightly slower than cuBLAS without autotuning."

---

## 2. What vLLM / Triton Does Differently

### 2.1 Triton Autotuner Architecture

Triton's autotuner (`triton.autotune`) generates GEMM kernels with different configurations at JIT compile time and benchmarks them on the actual hardware. The key parameters tuned:

| Parameter | Triton Range | rvLLM Current |
|-----------|-------------|---------------|
| BLOCK_M | 16, 32, 64, 128, 256 | Fixed by cuBLAS |
| BLOCK_N | 32, 64, 128, 256 | Fixed by cuBLAS |
| BLOCK_K | 32, 64, 128 | Fixed by cuBLAS |
| num_warps | 2, 4, 8 | Fixed by cuBLAS |
| num_stages | 2, 3, 4, 5 | Fixed by cuBLAS |
| split_k | 1, 2, 4, 8, 16 | cuBLAS heuristic only |
| GROUP_M (swizzle) | 4, 8, 16 | Fixed 8 in persistent_gemm.cu |

The critical insight: Triton autotuner generates entirely different kernels (different PTX) for each configuration, while cuBLAS selects from a fixed set of pre-compiled algorithms. Triton can specialize tile shapes to the exact problem dimensions.

### 2.2 torch.compile Integration

vLLM uses `torch.compile` which:
1. Traces the Python forward pass into a FX graph
2. Fuses contiguous elementwise/reduction ops into Triton kernels
3. For GEMM ops, generates `triton.ops.matmul` with autotuning
4. The Triton matmul kernel uses persistent scheduling on Hopper (sm_90)
5. Each unique (M,N,K) shape gets its own best configuration

### 2.3 Why Triton Wins at High Concurrency

At high concurrency (M=128-1024), the GEMM shapes are compute-bound. The performance gap comes from:

1. **Tile shape optimization**: For QKV projection at M=128, N=12288, K=4096 on H100:
   - cuBLAS selects a generic 128x128x64 tile (its best heuristic for "large" GEMMs)
   - Triton autotunes and often selects 128x256x64 or 256x128x32, which better match the N:K ratio

2. **Split-K parallelism for tall-skinny shapes**: For O-proj at M=128, N=4096, K=4096:
   - cuBLAS heuristic may or may not choose split-K=2
   - Triton autotunes split_k=4 or split_k=8, creating 4-8x more thread blocks

3. **L2 cache-aware swizzle**: Triton's `GROUP_M` parameter controls the tile traversal order. For tall-skinny shapes (large N, moderate M), grouping tiles along the M dimension improves L2 hit rate for the A operand.

4. **Persistent kernel scheduling**: On H100 sm_90, Triton uses persistent kernels (fixed grid = NUM_SM) that loop over tiles. This eliminates tile dispatch overhead and enables better L2 locality through cooperative prefetching.

5. **Epilogue fusion**: Triton kernels fuse bias-add, activation functions, and residual additions into the GEMM epilogue, avoiding separate kernel launches and HBM round-trips. vLLM's `torch.compile` automatically detects and fuses these patterns.

---

## 3. Quantitative Gap Analysis

From the optimization roadmap (`docs/optimization-roadmap.md`):

| Batch Size | rvLLM (tok/s) | vLLM (tok/s) | Ratio |
|-----------|---------------|--------------|-------|
| N=128 | 6,360 | 6,400 | 0.99x |
| N=256 | 8,316 | 9,437 | 0.88x |
| N=512 | 8,528 | 10,771 | 0.79x |
| N=1024 | 8,578 | 12,740 | 0.67x |

The gap widens monotonically with batch size. At N=1024, vLLM is 1.49x faster. This is the GEMM gap: at high batch sizes, compute is dominated by large GEMMs (M=1024 prefill/decode batches), and Triton's autotuned kernels extract more TFLOPS from the tensor cores.

---

## 4. Concrete Implementation Plan

### Phase 1: Enhanced cublasLt Autotuning (1-2 days, expected 5-10% at N>=256)

Modifications:

**File: `crates/rvllm-gpu/src/cublas_autotune.rs`**
- Expand `autotune_model` to cover batch sizes [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
- Increase `MAX_ALGOS` from 32 to 128 (cuBLAS returns more candidates when workspace is larger)
- Increase `BENCH_ITERS` from 10 to 20 for more stable measurements
- Increase workspace from 4 MiB to 32 MiB (H100 recommendation)
- Add persistent disk cache (JSON) for autotuned results, keyed by (GPU model, M, N, K, dtype)

**File: `crates/rvllm-gpu/src/cublaslt_ops.rs`**
- Wire autotuned algo into `hgemm_a_bt_into` (currently takes top-1 heuristic without benchmarking)
- Accept an `Option<&CublasAutotuner>` parameter; when present, use the cached algo instead of requesting `ret=1` from heuristic
- Increase `FP8_WORKSPACE_SIZE` to 32 MiB
- Raise `CUBLASLT_M_THRESHOLD` from 32 to 256 (cublasLt with autotuned algo beats stock cuBLAS at all M <= 256)

**File: `crates/rvllm-model-runner/src/layers/linear_cuda.rs`**
- Thread the `CublasAutotuner` through `forward_once_f16_lt` and all GEMM call sites
- Add a `forward_autotuned` method that uses the autotuned algo for all batch sizes

### Phase 2: CUTLASS Autotuning Integration (3-5 days, expected 10-20% at N>=256)

The existing CUTLASS integration (`kernels/cutlass_gemm.cu`) uses a fixed 128x128x64 tile shape. CUTLASS 3.x supports runtime tile selection:

**File: `kernels/cutlass_gemm.cu`**
- Generate multiple instantiations with different tile shapes:
  - 128x128x64 (current)
  - 128x256x64 (better for large N like gate_up: N=37888)
  - 256x128x32 (better for large M)
  - 64x256x64 (better for tall-skinny N >> M)
  - 256x256x32 (maximum output tile for compute-bound GEMMs)
- Each instantiation is a separate `extern "C"` function
- Add cluster shapes: 1x1x1 (current), 2x1x1, 1x2x1 for SM clustering on H100

**File: `crates/rvllm-gpu/src/cutlass_ffi.rs`**
- Extend `CutlassKernels` to hold function pointers for each tile variant
- Add a `select_best_tile` method that benchmarks each variant at model load time
- Cache results per (M, N, K) shape

### Phase 3: rTriton JIT GEMM with Autotuning (1-2 weeks, expected 15-30% at N>=256)

The rTriton crate already has the autotuner infrastructure (`crates/rtriton/src/autotune.rs`) with config search and persistent cache. The missing piece is actual MMA codegen in `crates/rtriton/src/codegen.rs` (line 386: `Op::Dot` emits a placeholder comment, not real MMA instructions).

**File: `crates/rtriton/src/codegen.rs`**
- Implement proper `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` emission for `Op::Dot`
- For sm_90: `wgmma.mma_async.sync.aligned.m64n256k16.f16.f16`
- Register allocation for accumulator fragments: 4x4 grid of m16n8k16 tiles per warp = 32 registers for C fragment
- Shared memory layout with bank conflict avoidance: pad A tiles to stride=33 instead of stride=32 (avoids 4-way bank conflicts when 32 threads access the same column)

**File: `crates/rtriton/src/kernels/gemm.rs`**
- Rewrite `build_tiled_gemm` with proper K-loop, multi-stage pipelining, and swizzled tile indexing
- Add `build_hopper_gemm` using TMA descriptors (`cp.async.bulk.tensor`)
- Software pipelining: N stages of cp.async for A/B tiles, with `cp.async.wait_group N-2` to overlap load and compute

**File: `crates/rtriton/src/autotune.rs`**
- Wire the benchmark runner to actually compile, load, and time JIT kernels on GPU
- Add split-K configs: for each (bm, bn, bk, warps, stages) combo, also try split_k in {1, 2, 4, 8}
- Implement the actual `autotune_kernel` function that:
  1. Generates all Config candidates
  2. For each: compile IR -> PTX -> cubin -> load module -> benchmark
  3. Store the best config in `PersistentCache`

### Phase 4: Persistent GEMM with Stream-K (2-3 weeks, expected 5-15% additional)

The existing `kernels/persistent_gemm.cu` is a prototype without real software pipelining. A production persistent GEMM on H100:

- **Tile decomposition on sm_90**: 128x256x64 output tile, 4 warp groups (each owns 128x64 of the output)
- **Stream-K partitioning**: Instead of data-parallel tile assignment, partition the total work (M/BM * N/BN * K/BK iterations) evenly across NUM_SM=132 blocks
- **Partial tile accumulation**: Blocks that share an output tile contribute partial results via atomic add (or a two-phase reduce: partial -> full tile owners)
- **TMA-based loads**: `cp.async.bulk.tensor.2d` for A/B tiles, 128-byte aligned, to bypass L1 and go directly to shared memory
- **Warp specialization**: 1 warp group produces (loads tiles), 3 warp groups consume (compute MMA). The producer uses TMA descriptors, consumers use `wgmma.mma_async`

---

## 5. H100 SM90 Architecture-Specific Details

### 5.1 Tensor Core Throughput

H100 SXM: 132 SMs, each with 4th-gen tensor cores.
- FP16: 4 `wgmma.mma_async.m64n256k16` per SM per cycle = 64*256*16*2 = 524,288 FMA ops/cycle/SM
- At 1.83 GHz boost: 132 * 524,288 * 1.83e9 = 126.8 TFLOP/s peak MMA throughput
- Practical peak with overhead: ~90 TFLOP/s (70% utilization is excellent)

### 5.2 Register Pressure Analysis

For a 128x256x64 tile with 4 warp groups:
- Each warp group owns 128x64 output in f32: 128 * 64 * 4 bytes = 32,768 bytes
- 4 warps per warp group, 32 threads per warp: 128 threads
- Per-thread accumulator registers: 32,768 / 128 / 4 = 64 registers
- A operand fragment: m64k16 = 16 registers
- B operand fragment: n256k16 = 128 registers (reused across K)
- Pointer/index/mask overhead: ~10 registers
- Total: ~218 registers per thread
- H100 register file: 65,536 registers per SM, max 255 per thread
- At 218 registers/thread: floor(65536/218) = 300 threads = 9.3 warps per SM
- With 4 warp groups = 16 warps needed per CTA: requires register spilling or smaller tile
- Practical approach: use m64n128k16 per warp group = 64 accumulator registers, total ~142 registers, allows 1 CTA of 16 warps = 51% occupancy

### 5.3 Shared Memory Bank Conflict Analysis

H100 shared memory: 32 banks, 4-byte bank width.
- A tile [128, 64] f16 in shared memory: each row = 64 * 2 bytes = 128 bytes = 32 banks
- When 32 threads in a warp load column 0: all access bank 0 -> 32-way conflict
- Fix: pad row stride to 65 elements (130 bytes per row) or use swizzled addressing
- Swizzle pattern: XOR row index with column group to distribute accesses across banks
- CUTLASS uses `SmemLayoutAtom` with `Swizzle<3,3,3>` for 128-bit accesses -> zero bank conflicts

### 5.4 Warp Scheduler Utilization

H100 has 4 warp schedulers per SM, each can issue 1 instruction per cycle.
- For persistent GEMM with 16 warps per CTA: 16/4 = 4 warps per scheduler
- With 2-3 pipeline stages, each warp alternates between MMA (4 cycles latency) and load/store
- Theoretical scheduler utilization: ~75-85% (limited by MMA latency bubbles)
- With warp specialization (producer/consumer): producer warps handle all loads, consumer warps issue back-to-back MMA -> ~90% utilization

---

## 6. Expected Speedup Summary

| Phase | Effort | Expected Gain (N>=256) | Risk |
|-------|--------|----------------------|------|
| Phase 1: Enhanced cublasLt autotuning | 1-2 days | 5-10% | Low |
| Phase 2: CUTLASS tile variants | 3-5 days | 10-20% | Medium |
| Phase 3: rTriton JIT GEMM + autotune | 1-2 weeks | 15-30% | High |
| Phase 4: Stream-K persistent GEMM | 2-3 weeks | 5-15% additional | Very High |
| **Combined** | **3-6 weeks** | **30-50%** | |

At N=1024, closing a 1.49x gap requires 49% improvement. The combined phases target 30-50%, which would bring rvLLM to 11,150-12,870 tok/s -- roughly matching vLLM's 12,740.

---

## 7. Risk Assessment

**Low Risk (Phase 1):**
cuBLAS autotuning is a known technique; the infrastructure exists. The main risk is that cuBLAS's 32-128 candidate algorithms may all be within 1-3% of each other on H100 for large shapes, limiting the upside.

**Medium Risk (Phase 2):**
CUTLASS template instantiation is well-understood but compile times are extreme (~5 min per variant). Need to limit variants to 5-8 per projection type. The FFI boundary adds complexity for error handling and debugging.

**High Risk (Phase 3):**
rTriton's codegen currently emits placeholder MMA. Implementing correct `wgmma.mma_async` emission with proper register allocation, pipeline stage management, and shared memory barriers is a substantial compiler engineering effort. The PostMortem (`POSTMORTEM_rtriton.md`) explicitly notes this as the P1 gap: "Dot op currently emits a placeholder comment, not real MMA instructions."

**Very High Risk (Phase 4):**
Stream-K requires partial tile accumulation with correct synchronization. The atomicAdd approach has non-deterministic FP rounding. The two-phase reduce approach requires a global memory workspace and a separate "fixup" kernel. Neither approach is simple to get right and numerically validate.

---

## 8. Recommended Sequencing

1. **Immediate (today)**: Phase 1 -- wire autotuned algos into `hgemm_a_bt_into`, increase workspace to 32 MiB, autotune more batch sizes. This is the highest ROI: lowest effort, guaranteed improvement.

2. **This week**: Phase 2 -- generate 4-5 CUTLASS tile variants, benchmark at model load. The CUTLASS integration already works (`cutlass_ffi.rs` loads and dispatches correctly); only need to add more instantiations.

3. **Next 2 weeks**: Phase 3 -- implement real MMA codegen in rTriton. Start with sm_80 `mma.sync.m16n8k16`, validate against cuBLAS output, then add sm_90 `wgmma`. This is the long-term strategic investment.

4. **After Phase 3 works**: Phase 4 -- stream-K is only worth pursuing once the basic tiled GEMM matches cuBLAS quality, as it adds complexity on top.

---

## Critical Files for Implementation

- `crates/rvllm-gpu/src/cublas_autotune.rs` -- The autotuner needs wider shape coverage, larger workspace, wiring into the execution path
- `crates/rvllm-gpu/src/cublaslt_ops.rs` -- The `hgemm_a_bt_into` function takes top-1 heuristic without autotuning; needs to accept and use autotuned algo
- `crates/rtriton/src/codegen.rs` -- The `Op::Dot` case (line 386) emits placeholder MMA; needs real `mma.sync`/`wgmma` emission
- `crates/rtriton/src/autotune.rs` -- Config space and persistent cache exist; needs `autotune_kernel()` function that compiles/benchmarks JIT kernels
- `kernels/cutlass_gemm.cu` -- Single tile shape (128x128x64); needs multiple instantiations for shape-dependent dispatch
