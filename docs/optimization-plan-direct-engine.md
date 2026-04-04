# rvLLM Direct Engine Optimization Plan

> Historical planning document. Superseded by the April 4, 2026 benchmark set and the current batch-1 / batched GEMM routing fixes.
> Current public direct-engine comparison on H100 / Qwen2.5-7B / `output-len=128`:
> `N=1 127.9 vs 165.5`, `N=32 4407.5 vs 4467.7`, `N=64 7964.0 vs 7972.1`, `N=128 13148.3 vs 13903.5` (rvLLM vs vLLM 0.19.0).

End-to-end token throughput optimization for the direct engine path (no HTTP overhead). Target: match or exceed vLLM 0.18 at all batch sizes on H100 SXM, then scale to B200.

Current state: 12,312 tok/s at N=128 (0.85x vLLM). The gap is not one thing -- it is death by a thousand cuts across the entire pipeline. This document orders every optimization by expected impact, implementation difficulty, and dependencies.

---

## Where We Are

```
                    rvLLM       vLLM 0.18     ratio
N=1                 98          170           0.58x    <-- memory-bound
N=4                 548         665           0.82x
N=16                2,122       2,202         0.96x    <-- near parity
N=32                3,957       4,585         0.86x
N=64                7,451       7,888         0.94x
N=128               12,312      14,528        0.85x    <-- compute-bound, GEMM gap
```

The pattern is clear: we are competitive at medium batch (0.94-0.96x at N=16/64) but lose at both extremes. Low N is memory-bandwidth limited (weight reads dominate). High N is compute-limited (GEMM throughput dominates). These are fundamentally different problems requiring different solutions.

---

## Critical Bugs Found (Fix Before Optimizing)

### Bug 1: FA3 v3 cp.async pipeline is NOT double-buffered

Comment at `flash_attention_3_v3.cu:7` says "Double-buffered KV tiles overlap next-tile K prefetch with current-tile PV computation." This is a lie. The actual code:

```c
// Line 262-264: Load K, WAIT FOR ALL, barrier
v3_cp_async_commit();
v3_cp_async_wait_all();   // <-- STALLS ALL WARPS until K arrives
__syncthreads();

// ... compute QK^T + softmax ...

// Line 314-316: Load V, WAIT FOR ALL, barrier
v3_cp_async_commit();
v3_cp_async_wait_all();   // <-- STALLS ALL WARPS again
__syncthreads();
```

Both K and V loads block all 8 warps before any compute begins. There is zero overlap between data movement and computation. This is single-buffered, sequential execution.

**Fix:** True double-buffered pipeline:
1. Load K[tile] + commit group 0
2. Load K[tile+1] + commit group 1 (prefetch next tile)
3. `wait_group 1` (oldest group = K[tile] done, K[tile+1] still in flight)
4. Compute QK^T on K[tile] while K[tile+1] transfers
5. Load V[tile] (reuse K[tile] buffer since consumed) + commit group
6. `wait_group 1` to get V[tile] while K[tile+1] finishes

This is ~15-25% throughput gain on attention kernel alone.

### Bug 2: V tile load waits for QK^T to finish (no overlap)

After QK^T + softmax (lines 266-308), the V tile load starts at line 311. But the V load could have been issued BEFORE softmax if we had a second shared memory buffer. Since we're single-buffered (line 17-18 says "Single-buffered to minimize smem footprint"), V cannot be loaded until K is consumed.

The comment justifies this: "~18KB vs 34KB double-buffered -- critical for L1 cache at high occupancy." At `__launch_bounds__(256, 2)` occupancy, each block uses up to 114 KB smem. The current single buffer is `V3_BC * head_dim * 2 = 64 * 128 * 2 = 16 KB`. Double buffer would be 32 KB. Still well under 114 KB. The L1 argument doesn't hold -- double-buffer is fine here.

### Bug 3: GQA Q reload per group is wasteful

In `fa3_v3_decode_gqa_kernel` lines 231-247, Q vectors for all heads in the group are loaded from global memory into `q_regs[g][]`. This is done once before the tile loop (good). But each group loads from a different Q head offset:

```c
int g_head = kv_head_idx * heads_per_group + g;
int q_base = (seq_idx * num_heads + g_head) * head_dim;
```

This is correct but the load pattern is stride-128-bytes (head_dim * 2 = 256 bytes apart). For 8 heads per group, that is 8 global memory transactions. These could be coalesced by loading all Q heads in a single pass using cooperative thread loading, then distributing to per-thread registers.

### Bug 4: Prefill path has no fusion at all

`gpu_layer.rs:834-1070` (scratch path for T>1) uses completely unfused kernels:
- Separate RMSNorm
- Separate cuBLAS/cublasLt GEMM for QKV
- Separate bias adds (up to 3 kernels for Q/K/V bias)
- Separate RoPE kernel
- Separate cache write kernel
- Separate attention
- Separate O-proj GEMM
- Separate add+norm
- Separate gate+up GEMM
- Separate SiLU*mul
- Separate down GEMM

That is 12-15 kernels per layer for prefill vs 6 for decode. For a batch of 128 tokens across 28 layers, this is 336-420 kernels that could be ~168 if we fused the pointwise ops around the GEMMs.

### Bug 5: rvllm-fusion dispatch is completely disconnected

`crates/rvllm-fusion/src/dispatch.rs` (680 lines) builds kernel templates but is never called from `gpu_layer.rs`. The entire JIT infrastructure (codegen.rs, ptx_emit.rs, llvm_backend.rs, matcher.rs) totaling ~5,000 lines is dead code. The forward pass loads pre-compiled PTX kernels by name via `KernelLoader::get_func()`.

Similarly, `crates/rtriton/src/` (~3,200 lines) is completely dormant -- none of its kernel builders or codegen are wired into the runtime.

### Bug 6: persistent_layer_decode is disabled

`gpu_layer.rs:211`: `if false && num_tokens == 1 && !input.is_prefill {`

This was disabled because "cooperative grid sync overhead outweighs kernel launch savings." But the cooperative kernel reduces 6-8 kernel launches to 1 per layer. With CUDA graphs already capturing the launch sequence, the comparison is unfair -- the cooperative kernel should be compared against the graphed sequence, not individual launches. Worth re-benchmarking on H100.

---

## The Forward Pass: Where Time Actually Goes

Per decode step for Qwen2.5-7B on H100 (28 layers, N=128):

| Operation | Kernel Count | % of Step Time | Bound By |
|-----------|-------------|-----------------|----------|
| GEMM (QKV, O, GateUp, Down) | 112 (4 per layer) | ~55% | Compute (tensor cores) |
| Attention (FA3 v3) | 28 | ~20% | Memory BW (KV cache reads) |
| Fused norms + residuals | 56 | ~8% | Memory BW |
| RoPE | 28 | ~3% | Memory BW |
| KV cache write | 28 | ~3% | Memory BW |
| Embedding + LM head | 2 | ~5% | Compute (LM head GEMM) |
| Sampling | 1 (CPU) | ~2% | CPU |
| Metadata HtoD | 6 uploads | ~2% | PCIe latency |
| CUDA graph overhead | 1 launch | ~2% | Driver |

Total: ~284 kernels captured in 1 CUDA graph launch (decode). ~400+ for prefill (no graph).

The 55% GEMM share at N=128 means GEMM optimization has the highest ceiling. But the 20% attention share means FA improvements compound with GEMM gains rather than competing.

---

## Tier 1: Fix What's Broken (15-30% total gain expected)

### 1.1 FA3 v3: Implement Real Double-Buffered cp.async Pipeline

The single biggest attention performance bug. Current kernel is sequential:
load K -> wait -> compute QK^T -> load V -> wait -> compute PV. No overlap.

Target kernel pipeline per tile:

```
Tile 0:  [load K0] [wait K0] [QK^T_0 + load V0] [wait V0] [PV_0 + load K1]
Tile 1:  [wait K1] [QK^T_1 + load V1] [wait V1] [PV_1 + load K2]
...
```

Implementation changes to `flash_attention_3_v3.cu`:
1. Allocate double-buffer in smem: `s_kv_a` and `s_kv_b` (32 KB total for head_dim=128)
   - Still under 114 KB per-block limit at occupancy 2
2. Before tile loop: prefetch K[0] into buf_a, commit group 0
3. Tile loop body:
   - `wait_group 1` (completes K[tile], K[tile+1] still in flight)
   - QK^T from current buffer
   - Issue V[tile] load into other buffer, commit group
   - Softmax
   - `wait_group 1` (completes V[tile])
   - PV accumulation from V buffer
   - Issue K[tile+1] load into other buffer, commit group
   - Swap buffer pointers

Smem budget: 2 * 64 * 128 * 2 = 32 KB (vs current 16 KB). Occupancy stays at 2 blocks/SM since 228 KB / 2 = 114 KB >> 32 KB.

Expected gain: 15-25% on attention kernel = 3-5% end-to-end. More at long context.

### 1.2 Wire rvllm-fusion / rTriton Into the Forward Pass

5,000+ lines of fusion JIT code (`rvllm-fusion/`) and 3,200+ lines of Triton-style compiler (`rtriton/`) are completely dead. The forward pass hardcodes kernel names via `KernelLoader::get_func()`.

The minimum viable connection:
1. At model load, `rvllm-fusion/matcher.rs` identifies which fused patterns the model needs
2. `rvllm-fusion/dispatch.rs` compiles those patterns via `jit.rs` (nvcc) or `ptx_emit.rs` (pure Rust)
3. `KernelLoader` registers the compiled kernels alongside hand-written ones
4. `gpu_layer.rs` dispatches to fused kernels when available, pre-compiled otherwise

This unlocks arbitrary fusion without writing new CUDA for each pattern. Priority patterns:
- RMSNorm + cuBLAS GEMM prologue (eliminates 28 HBM round-trips for T>1 norm->GEMM)
- GateUp GEMM + SiLU*mul epilogue (eliminates 28 kernel launches)
- Down GEMM + residual add epilogue (eliminates 28 kernel launches)

Expected gain: 5-15% from reduced kernel launches + eliminated HBM round-trips.

### 1.3 Fix Prefill Fusion (T>1 Path)

The scratch path (gpu_layer.rs:834-1070) runs 12-15 separate kernels per layer. For prefill of 512 tokens across 28 layers, that is ~400 kernel launches with no CUDA graph.

Quick wins without new kernels:
1. **Fuse bias into QKV GEMM** via cuBLASLt `CUBLASLT_EPILOGUE_BIAS` (eliminates 3 bias kernels per layer)
2. **Fuse RoPE + cache write** -- the `fused_rope_cache_f16_kernel` exists and works for T>1 but the scratch path calls them separately. Wire it in.
3. **Fuse SiLU*mul into GateUp GEMM** via CUTLASS EVT epilogue:
   ```cpp
   using Epilogue = cute::tuple<
       Compute<homogeneous_multiply_add>,
       Compute<activation<SiLU>>,
   >;
   ```
4. **CUDA graph capture for prefill** -- bucket prefill sizes to powers of 2 and capture graphs. Currently only decode has graphs.

Expected gain: 5-10% on prefill throughput, plus graph capture saves 10-15% from eliminating launch overhead.

### 1.4 Autotuner: Persist Results to Disk

The autotuner sweeps 88 shapes x deep enumeration at every cold start. That is 30-60 seconds on H100 that adds to cold start time. vLLM's torch.compile caching is worse (~60s) but we can do better.

Serialize `CublasAutotuner::results` to `~/.cache/rvllm/autotune_{model_hash}_{gpu_uuid}.bin`. On subsequent starts, load from cache and skip autotuning entirely. Invalidate cache when cuBLAS version changes (embed library version in cache key).

Expected gain: 0% runtime, but saves 30-60s cold start (currently ~6s without autotune, ~36-66s with).

---

## Tier 2: Medium Impact, Moderate Complexity (10-25% total gain expected)

### 2.1 TMA-Based FlashAttention (SM90 Warp-Specialized)

Our FA3 v3 uses `cp.async` bulk copies (128-bit) + split-KV for long context. This is good but not optimal on H100. The official FlashAttention-3 from Tri Dao uses:

1. **TMA (Tensor Memory Accelerator):** Hardware DMA unit that moves entire tiles from global to shared memory without using any threads or registers. Our `cp.async` still uses threads to issue the copies.

2. **Warp specialization:** Producer warps handle TMA loads (Q, K, V tiles into shared memory) while consumer warps execute WGMMA tensor core ops. The two groups overlap completely via `mbarrier` producer-consumer synchronization. Our current kernel has all warps doing both loads and compute sequentially.

3. **WGMMA (Warp Group MMA):** Operates on 4 warps (128 threads) and reads B operand directly from shared memory (no register stage). Our kernel uses per-thread loads into registers then `hmma` instructions. WGMMA gives ~1.3x throughput for FP16 attention.

4. **Multi-stage pipeline:** 3-7 stages of K/V tiles prefetched via TMA while current tile is being computed. Our double-buffering (2 stages) is the minimum.

Implementation path:
- Write `flash_attention_4_tma.cu` using CUTLASS 3.x CuTe abstractions
- Use `SM90_TMA_LOAD` for Q/K/V tile loads
- Use `SM90_16x8x16_F16F16F16F16_TN` WGMMA for QK^T and PV
- `PipelineTmaAsync` for 4-stage pipeline
- Grid: `(num_seqs, num_kv_heads)` with Q heads processed in the inner loop (GQA-optimized)

```cpp
// CuTe-based attention kernel sketch
using TiledMma = TiledMMA<SM90_16x8x16_F16F16F16F16_TN,
                           Layout<Shape<_2, _1, _1>>>;
using GmemCopyQKV = SM90_TMA_LOAD;
using SmemLayout = GMMA::Layout_MN_SW128_Atom<half_t>;

// Pipeline: producer warps load tiles, consumer warps compute
cute::PipelineTmaAsync<4> pipeline;
```

Expected gain: 15-25% on attention kernel (which is ~20% of total step), so 3-5% end-to-end. More at long context (4K+) where attention dominates.

### 2.2 Persistent GEMM with Stream-K Scheduling

For compute-bound GEMMs at high N (M>=128), the standard tile-based GEMM launches `ceil(M/TILE_M) * ceil(N/TILE_N)` thread blocks. Some blocks finish early (uneven tile shapes at edges), leaving SMs idle. This tail effect is significant when the grid is small relative to SM count.

Stream-K persistent GEMMs fix this: launch exactly `num_SMs` thread blocks, each loops over tiles. Work is distributed evenly. The last tiles of one GEMM can overlap with the first tiles of the next GEMM on the same SM.

On H100 with 132 SMs, for QKV projection at M=128, N=12288, K=4096 with 128x128 tiles: that is 1x96=96 tiles, meaning 36 SMs sit idle (27% waste). Stream-K would use all 132 SMs.

Implementation:
- rTriton already has a `persistent_gemm` kernel stub -- needs real WMMA codegen
- Alternative: CUTLASS 3.x `KernelTmaWarpSpecializedCooperative` with `StreamKScheduler`
- The CUTLASS path is more mature; the rTriton path is strategic (no external dependency)

Priority: CUTLASS path first (ship in days), rTriton path as follow-up (weeks).

Expected gain: 5-15% at N>=128 (where GEMM is compute-bound and tail effects matter).

### 2.3 Scheduler: Prefill/Decode Disaggregation

The current scheduler interleaves prefill and decode tokens in the same batch. This creates problems:

1. **Shape variance:** A batch with 1 prefill (512 tokens) + 127 decode (1 token each) has M=639, but the GEMM is only useful for M=128 of those (the decode tokens generate next tokens). The prefill tokens are doing work that could be deferred.

2. **CUDA graph incompatibility:** Graphs are captured per exact batch size. Mixed prefill+decode batches have variable sizes, causing frequent graph misses and fallback to individual kernel launches.

3. **KV cache pressure:** Long prefills consume many blocks at once, potentially preempting decode sequences that are closer to completion.

Better approach (what vLLM does):
- **Chunked prefill:** Limit prefill to `max_prefill_chunk` tokens per step (e.g., 256). We have this partially implemented but the chunk size is not tuned.
- **Prefill priority inversion:** When decode queue is deep, prioritize decode (lower latency) over prefill (higher throughput). The current FCFS policy doesn't account for this.
- **Batch size bucketing:** Round batch sizes to powers of 2 (or specific values matching CUDA graph capture sizes) to maximize graph hits.

The scheduler is 944 lines and well-structured. The changes are surgical:

```rust
// In schedule_running():
// 1. Separate prefill and decode sequences
let (prefill_seqs, decode_seqs): (Vec<_>, Vec<_>) =
    running.iter().partition(|sg| sg.is_prefilling());

// 2. Budget tokens separately
let decode_budget = max_num_batched_tokens - prefill_chunk_size;
let prefill_budget = prefill_chunk_size;

// 3. Build separate metadata for each group
// (enables different CUDA graph paths)
```

Expected gain: 5-10% at mixed workloads. At pure decode (our benchmark), minimal impact. But real-world workloads are always mixed.

### 2.4 Cross-Layer Vertical Fusion

The per-layer boundary forces an HBM round-trip between every layer's down-projection output and the next layer's RMSNorm input. The data flow is:

```
down_proj output [N, H] f16  ->  write HBM
                                    |
residual add [N, H] f16     <-  read HBM (previous residual)
                                    |
RMSNorm [N, H] f16          ->  write HBM (norm output)
                                    |
QKV GEMM input               <-  read HBM
```

That is 4 HBM touches for data that should stay in registers. The JIT fused kernels (`fused_add_norm_qkv_gemv.cu`) already fuse residual+norm+GEMV for M=1. But for M>1, the GEMM is done by cuBLAS which requires input in HBM.

Options:
1. **cuBLASLt with bias epilogue on previous layer + RMSNorm on next layer:** Doesn't help because RMSNorm is a reduction (needs all elements before normalizing).
2. **CUTLASS EVT with custom epilogue on down_proj:** Write `residual + down_proj_output` to HBM, then fuse RMSNorm into a custom CUTLASS prologue on the QKV GEMM. CUTLASS 3.x supports custom prologues via `CollectiveMainloop` customization. This is advanced but the CuTe abstractions make it tractable.
3. **Register-level fusion via rTriton:** The rTriton compiler can fuse residual_add + RMSNorm + GEMV into a single PTX kernel for M=1 (already working, 2-7.5x speedup). Extending to small M (2-16) is feasible since the GEMV approach scales linearly.

For M>16, option 2 is the path. For M<=16, option 3 is already working.

Expected gain: 3-5% at all batch sizes (eliminates 56 HBM round-trips per step).

---

## Tier 3: High Impact but High Complexity (20-50% gain at specific regimes)

### 3.1 Speculative Decoding

At N=1 (single-stream decode), we are at 98 tok/s. The hardware can do ~170 tok/s (vLLM proves this). Even with FP8, we might reach 140-160 tok/s. Speculative decoding can push effective throughput to 200-400 tok/s by generating multiple tokens per step.

Architecture:
- **Draft model:** Use the same model with a smaller layer count or a separate 1.5B model
- **Self-draft:** Use early-exit from layer 8 of the 7B model as a draft (no separate model, shares KV cache)
- **Verification:** Run full 7B model on K draft tokens, accept the longest prefix that matches

For self-draft with K=4 draft tokens and ~70% acceptance rate:
- Step 1: Run layers 1-8, predict 4 draft tokens (fast, ~30% of full model cost)
- Step 2: Run full 28 layers on all 4 draft tokens in parallel (M=4 GEMM, better utilization)
- Step 3: Accept ~2.8 tokens on average
- Effective: 2.8 tokens per full-model step, ~1.3 full steps per unit time = 3.6x throughput

The `rvllm-speculative` crate exists as a scaffold. Implementation:
1. Implement `SelfDraftSpeculator` that runs first K layers as draft
2. Modify `GpuModelRunner::forward` to accept multiple candidate sequences
3. Add verification logic (parallel scoring + longest-prefix acceptance)
4. Wire into scheduler: speculative sequences get priority since they batch well

Expected gain: 2-4x at N=1-4 (the latency-bound regime). Diminishing returns at N>32 where batching already fills the GPU.

### 3.2 FP4 on B200 (Blackwell)

B200 has native FP4 E2M1 support at 9 PFLOPS (4x FP8, 8x FP16). For decode at N=1-32 (memory-bound), FP4 weights would quarter the bandwidth requirement. This is the single biggest potential gain for low-batch inference.

Challenges:
- FP4 quantization has higher precision loss -- need per-block scaling (groups of 32-128 elements)
- cuBLASLt FP4 support via TCGEN05 tensor cores on SM100
- Weight packing: 2 FP4 values per byte, with scale factors stored alongside
- No cuBLASLt FP4 matmul API yet (as of CUDA 13.1) -- will need CUTLASS 3.x with SM100 TCGEN05 MMA or wait for NVIDIA's API

Implementation plan:
1. Add FP4 quantization to `rvllm-quant` (per-group E2M1 with f16 scales)
2. Add CUTLASS SM100 GEMM kernel with FP4 input, FP16 accumulator
3. Autotune FP4 shapes (same infrastructure as FP16/FP8)
4. Quality validation: perplexity regression on standard benchmarks

Expected gain: 50-80% at N=1-16 on B200. Not applicable on H100 (no FP4 hardware).

### 3.3 rTriton Full Stack: Shape-Specialized Kernel Codegen

The strategic play. rTriton already generates PTX for fused kernels (2-7.5x faster than hand-written CUDA at M=1). The gap: real GEMM codegen using WMMA/WGMMA instructions instead of cuBLAS calls.

If rTriton can generate shape-specialized GEMMs:
- No external library dependency (cuBLAS, CUTLASS)
- Per-shape tile optimization (like Triton's autotuner but in Rust)
- Fuse GEMM + epilogue + next-op in a single generated kernel
- The CUDA graph captures one massive fused kernel per layer instead of 10 separate ones

The missing pieces in rTriton's codegen:
1. **WMMA/WGMMA instruction emission:** `Dot` op currently emits a placeholder. Need to lower to `wmma.mma.sync` (SM80) or `wgmma.mma_async` (SM90) PTX instructions.
2. **2D tile mapping:** Currently maps everything to `threadIdx.x`. Need `threadIdx.x/y` for M/N tile dimensions.
3. **Shared memory staging:** The `SharedMemoryPlanning` pass identifies candidates but codegen doesn't emit `ld.shared`/`st.shared`.
4. **Software pipelining:** The pass exists but codegen doesn't emit `cp.async` + commit/wait groups.
5. **Loop lowering:** The `Loop` IR op is stubbed. Need actual loop codegen with `bra` + phi nodes.

This is months of work but the payoff is transformative: a single Rust binary that generates optimal GPU code for any model on any GPU architecture, with zero Python or C++ dependencies.

Priority: Continue in parallel with Tier 1/2 work. Ship rTriton for M=1 fused kernels now (working), extend to M<=16 next, then tackle GEMM codegen.

Expected gain: 10-30% over cuBLAS at all batch sizes (matching or exceeding Triton's autotuned GEMMs, with tighter fusion).

---

## Tier 4: Infrastructure and Compounding Optimizations

### 4.1 CUDA Graph Pool Optimization

Current state: CUDA graphs captured per exact batch size. With 35 captured sizes, capture takes 1.7s (35x faster than vLLM). But there are edge cases:

- **Graph miss on new batch size:** Falls back to individual kernel launches (10-20% slower)
- **Memory overhead:** Each graph snapshot pins its GPU memory allocations. 35 graphs x ~100MB each = 3.5 GB wasted
- **Stale graphs:** After autotuner updates, old graphs use stale algo pointers

Improvements:
1. **Batch size bucketing:** Round to nearest power of 2. Instead of 35 unique sizes, capture 8 graphs (1, 2, 4, 8, 16, 32, 64, 128). Pad unused slots with dummy tokens.
2. **Graph memory sharing:** Use `cudaGraphInstantiateWithParams` with `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` to release memory between launches.
3. **Lazy capture:** Don't capture at startup. Capture on first encounter of each bucket, then cache. Amortize capture cost over the first few requests.

Expected gain: 2-5% from fewer graph misses + less memory pressure.

### 4.2 Metadata Upload Consolidation

Currently 6 separate HtoD uploads per step:
- token_ids: [N] i32
- positions: [N] i32
- context_lens: [N] i32
- block_tables: [N, max_blocks] i32
- slot_mapping: [N] i32
- is_prefill flags

Each upload is a tiny PCIe transfer with ~5us latency overhead. 6 x 5us = 30us per step.

Consolidation: Pack all metadata into a single contiguous buffer on CPU, upload once. Kernels index into offsets within the buffer.

```rust
struct PackedMetadata {
    // Layout: [token_ids | positions | context_lens | slot_mapping | block_tables]
    buf: Vec<i32>,
    token_ids_offset: usize,
    positions_offset: usize,
    // ...
}
```

Expected gain: ~25us per step. At N=128 this is <1% but at N=1 it is 2-3%.

### 4.3 L2 Cache Persistence for Weights

H100 has 50 MB L2 cache. A 7B model has ~14 GB weights. Obviously the weights don't fit. But during decode at N=1, each layer's weights are read once and evicted. The next layer's weights cold-start from HBM.

However: if we set L2 persistence policy on the most-reused weights (embedding table, final LayerNorm, LM head), these stay hot across steps. The embedding table (vocab_size * hidden = 151936 * 3584 * 2 = ~1 GB) doesn't fit, but the LM head weight is the same tensor (tied embeddings in Qwen2.5).

More useful: set L2 persistence on the KV cache blocks for the hottest sequences (longest context = most reuse).

```c
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = kv_cache_ptr;
attr.accessPolicyWindow.num_bytes = hot_bytes;
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

Expected gain: 2-5% at N=1-8 (attention is memory-bound, KV cache L2 hits reduce HBM reads).

### 4.4 Async Weight Prefetch

During attention computation (which doesn't touch weights), the next layer's GEMM weights could be prefetched into L2/shared memory. This overlaps weight load latency with attention compute.

Implementation via TMA prefetch hints:
```
// PTX: prefetch weight tile into L2 during attention
prefetch.global.L2 [weight_ptr];
```

Or via a separate CUDA stream that issues weight loads while the compute stream runs attention.

Expected gain: 3-8% at N=1-16 (where weight loads are the bottleneck and attention gives a natural prefetch window).

### 4.5 CPU-Side: Async Sampling Pipeline

Sampling is CPU-bound (top-k/p/min-p on logits). Currently sampling blocks the GPU thread until the CPU finishes. At N=128, CPU sampling takes ~4ms (batch of 128 sequences, each with top-p over 152K vocab).

Already fast in Rust (4-11x faster than Python/numpy for top-p, repetition penalty, batch sampling). But can overlap with next GPU step:

1. GPU finishes step N, copies logits to pinned buffer
2. CPU starts sampling step N tokens
3. GPU starts step N+1 immediately (doesn't wait for sampling)
4. CPU finishes sampling, feeds token IDs for step N+2

This requires double-buffering the logits/token_id pinned buffers and a 1-step pipeline.

Warning: this was attempted before and reverted because "scheduler needs all tokens from step N before scheduling N+1." The fix: the scheduler pre-allocates KV blocks for the next token of all running sequences before step N completes. Since we know exactly which sequences are running, the block allocation is deterministic.

Expected gain: 2-5% at N>=64 (where CPU sampling time becomes non-negligible).

---

## Tier 5: Architecture-Level Changes (B200 / Multi-GPU)

### 5.1 B200 Native Optimization

B200 changes the calculus:
- 8 TB/s HBM3e (2.4x H100): Memory-bound ops speed up proportionally
- 192 SMs (1.45x H100): More parallelism
- 96 MB L2 (1.9x H100): More weight caching
- TCGEN05 tensor cores: New MMA shapes, potentially faster attention
- FP4 native: 9 PFLOPS

Migration checklist:
1. Recompile all CUDA kernels with `-arch=sm_100a`
2. Retune autotuner (new algo space, different optimal configs)
3. Increase pipeline stages (more SMs = more tiles in flight)
4. Enable FP4 weight path
5. Increase L2 persistence window (96MB = ~6x more persistent data)
6. Autotuner already handles B200: `workspace_bytes(100) = 64MB`, `compute_type = FAST_TF32`

### 5.2 Multi-GPU Tensor Parallelism Optimization

The `rvllm-tp` crate implements Megatron-style TP with NCCL. For 70B+ models that don't fit on a single GPU, TP is required. But TP introduces all-reduce communication after every GEMM.

On NVLink 5 (B200): 1.8 TB/s bidirectional per GPU. For 2-GPU TP with a 70B model, each all-reduce moves ~hidden_dim * 2 bytes * 2 (reduce-scatter + all-gather) = ~16 KB per token. At N=128: ~2 MB per all-reduce, ~16 all-reduces per layer, ~450 per step. Total: ~900 MB/step, well within NVLink bandwidth.

The optimization: overlap all-reduce with the next GEMM. Standard approach:
1. Split the all-reduce into reduce-scatter + all-gather
2. Start GEMM on the local shard immediately after reduce-scatter
3. All-gather runs on a separate NCCL stream, overlapping with GEMM compute

This requires the GEMM to be decomposable into independent shards, which is true for the QKV and GateUp projections (column-parallel) but not for O-proj and down-proj (row-parallel, which need the full result).

Expected gain: 10-20% on multi-GPU configurations by hiding communication latency.

---

## Execution Priority

Ordered by (expected throughput gain) / (implementation effort). Bugs first, then optimizations.

| Priority | Optimization | Gain | Effort | Regime |
|----------|-------------|------|--------|--------|
| **BUG** | FA3 v3 real double-buffer pipeline | 3-5% e2e | 2 days | All N, more at long ctx |
| **BUG** | Wire rvllm-fusion into forward pass | 5-15% | 1 week | All N (prefill especially) |
| **BUG** | Fix prefill fusion (T>1 scratch path) | 5-10% | 3 days | Prefill throughput |
| 1 | cuBLASLt epilogue fusion (bias, SiLU) | 5-8% | 2 days | All N |
| 2 | Persist autotuned algos to disk | 0% runtime, 30-60s startup | 1 day | All |
| 3 | TMA FlashAttention (SM90 WGMMA) | 3-5% | 1 week | All N, more at long ctx |
| 4 | Metadata upload consolidation | 1-3% | 1 day | N=1-8 |
| 5 | CUDA graph for prefill (bucketed) | 5-10% | 2 days | Prefill |
| 6 | Persistent/Stream-K GEMM | 5-15% | 1 week | N>=128 |
| 7 | Cross-layer vertical fusion | 3-5% | 1 week | All N |
| 8 | Scheduler disaggregation | 5-10% | 3 days | Mixed workloads |
| 9 | L2 cache persistence (KV hot blocks) | 2-5% | 1 day | N=1-8 |
| 10 | Re-benchmark persistent_layer_decode | 0-15% | 1 day | N=1 decode |
| 11 | Speculative decoding | 2-4x | 2 weeks | N=1-4 |
| 12 | Async sampling pipeline | 2-5% | 3 days | N>=64 |
| 13 | rTriton GEMM codegen | 10-30% | 2 months | All N |
| 14 | B200 FP4 | 50-80% | 2 weeks | N=1-16 on B200 |

---

## Projected Throughput After All Tier 1+2 Optimizations

Conservative estimates (gains don't fully compound due to Amdahl's law):

```
                    Current     Projected     vLLM 0.18     Projected Ratio
N=1                 98          160-200       170           0.94-1.18x
N=4                 548         750-900       665           1.13-1.35x
N=16                2,122       2,600-2,900   2,202         1.18-1.32x
N=32                3,957       4,800-5,400   4,585         1.05-1.18x
N=64                7,451       8,900-10,000  7,888         1.13-1.27x
N=128               12,312      15,000-17,000 14,528        1.03-1.17x
```

The goal is not 1.0x -- it is >1.0x everywhere. rvLLM's architectural advantages (no Python, no GIL, 20x faster startup, 3x less memory, 5.6x tighter P95 latency) become even more compelling when the raw throughput matches or exceeds vLLM.

---

## Measurement Methodology

Every optimization must be measured before and after with:

1. **Direct engine benchmark** (`bench/run.sh`): N=1,4,8,16,32,64,128 with Qwen2.5-7B on H100 SXM
2. **ncu profile** for the top 5 kernels by time: verify the optimization actually reduced the targeted bottleneck
3. **Memory bandwidth utilization** via `ncu --metrics sm__throughput,dram__throughput`: ensure we are not trading compute for memory stalls
4. **Quality check**: perplexity on WikiText-2 for any quantization change
5. **P95 latency** at sustained load: ensure optimization doesn't hurt tail latency

The benchmark script should run automatically after every PR that touches kernel code. Regression = revert.

---

## What Not To Do

- **Don't add fallbacks.** Every fallback path is a silent performance cliff. The no-fallback validation from Phase 6 was critical -- extend this principle. If a kernel isn't available, error at startup, don't silently fall back to a 10x slower path.
- **Don't try FP8 weight quantization again.** It failed before with quality/stability issues. Revisit only after the engine-level bugs above are fixed and measured.
- **Don't over-fuse.** Fusion that tanks occupancy (too many registers/smem) can be net negative. Always benchmark. The persistent_layer_decode disable (gpu_layer.rs:211) is an example of fusion that was slower.
- **Don't optimize the HTTP path.** The direct engine gap is the real gap. HTTP overhead is fixed cost that becomes negligible at high throughput.
- **Don't add GPTQ/AWQ/Marlin kernel zoo.** FP8 is the right quantization for H100. FP4 for B200. The others are legacy formats for older GPUs.
- **Don't pipeline steps before fixing the scheduler.** The previous attempt at engine-level pipelining cratered throughput (1787->154 tok/s) because the scheduler needs token outputs before scheduling the next step.
- **Don't write a custom memory allocator.** The current pool-based allocator with CoW is correct and fast. GPU memory allocation is not in the top 10 bottlenecks.
- **Don't write more dead infrastructure.** 8,000+ lines of fusion/rtriton code are disconnected from the runtime. Wire what exists before writing more.
