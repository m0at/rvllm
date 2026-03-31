# 03: Warp Specialization

Research document covering warp specialization techniques for rvLLM's CUDA kernels -- splitting warps into producer (memory) and consumer (compute) roles to hide HBM latency behind arithmetic, targeting 15-30% throughput improvement in attention and GEMV kernels.

## Table of Contents

1. [Current Kernel Execution Model](#current-kernel-execution-model)
2. [Warp Specialization Theory](#warp-specialization-theory)
3. [H100 SM90 Architecture Specifics](#h100-sm90-architecture-specifics)
4. [How FlashAttention-3 Uses Warp Specialization](#how-flashattention-3-uses-warp-specialization)
5. [How CUTLASS 3.x Uses Warp Specialization](#how-cutlass-3x-uses-warp-specialization)
6. [rvLLM Attention Kernel Analysis](#rvllm-attention-kernel-analysis)
7. [Implementation Plan: Attention](#implementation-plan-attention)
8. [Implementation Plan: GEMV](#implementation-plan-gemv)
9. [Register File Partitioning](#register-file-partitioning)
10. [Barrier Synchronization Patterns](#barrier-synchronization-patterns)
11. [Pipeline Depth Analysis](#pipeline-depth-analysis)
12. [Shared Memory Allocation Per Stage](#shared-memory-allocation-per-stage)
13. [Expected Performance Impact](#expected-performance-impact)
14. [Code-Level Changes](#code-level-changes)

---

## Current Kernel Execution Model

### All Warps Do the Same Thing

Every rvLLM kernel today uses a **homogeneous warp model**: all warps in a thread block execute identical code, differing only by their assigned data partition. There is no role differentiation.

**FA3 v3 decode kernel** (`flash_attention_3_v3.cu`):
- 256 threads = 8 warps
- Grid: `(num_seqs, num_kv_heads, num_splits)`, Block: `(256, 1, 1)`
- `__launch_bounds__(256, 2)` -- targets 2 blocks/SM occupancy
- All 8 warps cooperate on: cp.async KV tile load -> wait -> QK^T dot products -> softmax -> cp.async V load -> wait -> PV accumulation
- Synchronization: `__syncthreads()` between every phase (load, compute, load, compute)
- Single-buffered shared memory (~18KB) to preserve L1 cache partition at high occupancy

**Current pipeline structure** (per tile iteration):

```
Time -->

All 8 warps:  [Load K]  [sync]  [QK^T + softmax]  [sync]  [Load V]  [sync]  [PV accum]  [sync]
              ^^^^^^^^                                       ^^^^^^^^
              HBM stall                                      HBM stall
```

The two HBM stalls (K load, V load) are **fully serialized** with compute. Every warp blocks on `cp.async.wait_group 0` then `__syncthreads()` before any computation begins. This means the GPU's memory system and compute units are never simultaneously utilized.

**GEMV kernels** (`gemv_f16.cu`, `wgmma_gemv.cu`):
- 256 threads, one block per output row (scalar) or 128 output rows (WGMMA)
- All threads load weight elements, multiply, then warp-reduce
- Weight loads from HBM completely serialize with FMA compute
- No pipelining, no double-buffering, no async loads

**Persistent GEMM** (`persistent_gemm.cu`):
- 128 threads = 4 warps, persistent grid of `NUM_SMS` blocks
- Uses wmma 16x16x16 fragments but loads A/B tiles synchronously
- K-tile loop is: load A tile -> load B tile -> `__syncthreads()` -> wmma MMA -> `__syncthreads()`
- Has 3-stage constant defined (`STAGES 3`) but the actual K-tile loop uses only 1 stage (no pipelining implemented yet)

**Fused kernels** (`fused_add_norm_qkv_gemv.cu`):
- Two-phase: (1) all threads cooperate on add+norm, (2) warp-per-row GEMV
- Phase 2 does warp-per-row dot products with no async overlap

### What This Means

The current model leaves significant performance on the table:

1. **Memory latency is fully exposed.** HBM read latency on A100/H100 is 400-600 cycles. Each tile load stalls all warps for the full duration.
2. **Compute units idle during loads.** The FP16 tensor cores (312 TFLOPS on A100, 990 TFLOPS on H100) sit completely idle during KV cache reads.
3. **Memory units idle during compute.** The HBM controller (2 TB/s on A100, 3.35 TB/s on H100) is unused during QK^T and PV accumulation.
4. **No overlap between phases.** The sequential load->compute->load->compute pattern means at best 50% hardware utilization even in a bandwidth-saturated kernel.

---

## Warp Specialization Theory

### The Core Idea

Warp specialization divides warps within a thread block into two roles:

- **Producer warps**: dedicated to memory operations (global -> shared memory loads via cp.async or TMA)
- **Consumer warps**: dedicated to compute operations (tensor core MMA, softmax, reductions)

Producers and consumers operate **concurrently** on different pipeline stages, synchronized via lightweight barriers (named barriers or mbarrier). This overlaps memory latency with useful computation.

### Why It Works: The Dual-Issue Window

An SM can simultaneously:
1. Issue memory instructions (LD/ST units, TMA, cp.async) from one warp
2. Issue arithmetic instructions (FMA, HMMA, SFU) from a different warp
3. Issue barrier/sync instructions from yet another warp

The warp schedulers are independent -- they select ready warps regardless of type. By partitioning warps by role, we guarantee that:
- There is always a memory warp ready to issue loads (it never stalls on compute dependencies)
- There is always a compute warp ready to issue MMA (it never stalls on memory dependencies)
- The two never contend for the same execution units

### Pipelining Visualization

Without warp specialization (current rvLLM):
```
Stage:    [Load K_0] [Compute QK_0] [Load V_0] [Compute PV_0] [Load K_1] ...
Memory:   ####.......                ####.......                ####.......
Compute:  ........#####              ........####               ........####

Utilization: ~50% each unit (never overlap)
```

With warp specialization (target):
```
Producer: [Load K_0] [Load V_0] [Load K_1] [Load V_1] [Load K_2] ...
Consumer:            [QK_0 + softmax_0] [PV_0] [QK_1 + softmax_1] [PV_1] ...

Memory:   ####       ####       ####       ####       ####
Compute:       ######     ######     ######     ######

Utilization: ~85-95% each unit (steady-state overlap)
```

The producer runs 1-2 stages ahead of the consumer. The consumer works on data from a completed earlier stage while the producer fetches data for the next stage.

### Mathematical Model

Let:
- `T_load` = time to load one KV tile from HBM to shared memory
- `T_compute` = time to compute QK^T + softmax + PV for one tile
- `N_tiles` = total number of tiles
- `P` = pipeline depth (number of stages ahead the producer runs)

**Without specialization:**
```
T_total = N_tiles * (T_load_K + T_compute_QK + T_load_V + T_compute_PV)
        = N_tiles * (2 * T_load + 2 * T_compute)
```

**With specialization (steady state):**
```
T_total = P * T_load  +  N_tiles * max(T_load, T_compute)  +  drain
```

If `T_load ~= T_compute` (which it often is for decode attention with moderate context lengths), the speedup approaches **2x** for the tile loop. In practice, 1.3-1.7x is realistic after accounting for barrier overhead, fill/drain, and register pressure from buffering.

---

## H100 SM90 Architecture Specifics

### SM Structure

Each H100 SM contains:
- **4 warp schedulers**, each managing up to 16 warps (64 warps per SM max, not 128 -- the 128 figure is for the full SM partition including sub-partitions)
- **4 processing blocks** (quadrants), each with:
  - 1 warp scheduler
  - 1 dispatch unit
  - 16 INT32 cores
  - 16 FP32 cores
  - 1 tensor core (FP16/BF16/FP8/INT8 MMA)
  - 1 SFU (special function unit: sin/cos/exp/rsqrt)
  - 8 LD/ST units
- **256 KB register file** (65,536 x 32-bit registers)
- **228 KB shared memory** (configurable with L1 cache, max shared = 228 KB)
- **256 KB L1 data cache / shared memory** (combined, configurable split)

### Warp Scheduling on SM90

Key scheduler behaviors relevant to warp specialization:

1. **Independent quadrant scheduling.** Each quadrant's scheduler independently picks a ready warp every cycle. Quadrants do not coordinate.
2. **Warp residency.** A warp is resident on exactly one quadrant for its entire lifetime. At launch, warps 0-15 go to quadrant 0, 16-31 to quadrant 1, etc.
3. **Maximum warps per SM.** 64 warps = 2048 threads. With 256-thread blocks, that's 8 resident warps per block, max 8 blocks per SM (if register/smem allows).
4. **`setmaxnreg` instruction (SM90+).** Allows setting maximum registers per warp at runtime. Producer warps can release registers (they need few), giving more to consumer warps for MMA accumulation.

### TMA (Tensor Memory Accelerator) -- SM90 Only

TMA is the hardware unit that replaces `cp.async` for bulk data movement on Hopper:

- **Asynchronous, fire-and-forget.** One instruction initiates a multi-dimensional copy. No warp cycles consumed during transfer.
- **Tensor descriptor.** A 128-byte descriptor defines the source tensor's shape, strides, and data type. The TMA unit handles address calculation.
- **Multicast.** One TMA load can deliver data to shared memory of multiple SMs simultaneously (for tensor parallelism).
- **mbarrier integration.** TMA loads arrive at an mbarrier, which consumer warps wait on. This is the native synchronization primitive.
- **Bandwidth.** TMA saturates the full HBM bandwidth (3.35 TB/s) with fewer warps than cp.async requires.

For rvLLM today, we target **cp.async on SM80+** (A100, RTX 3090, etc.) which is already partially implemented in FA3 v3. The TMA path is a future SM90-specific upgrade.

### Async Copy Pipeline (cp.async) -- SM80+

Already used in `flash_attention_3_v3.cu`:

```cuda
// 128-bit bulk copy, bypasses L1 and registers
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;
cp.async.commit_group;      // commit current group
cp.async.wait_group N;      // wait until at most N groups pending
```

Key properties:
- Each `cp.async` is 16 bytes (8 f16 values)
- Groups are FIFO -- `wait_group 1` means "wait until the oldest group completes"
- Multiple groups can be in flight simultaneously (up to 8 on A100)
- Bypass mode (`cg` = cache-global) skips L1, useful for streaming access patterns

### Named Barriers (SM90 mbarrier)

The SM90 introduces hardware `mbarrier` (memory barrier) objects in shared memory:

```cuda
// Initialize: N threads expected to arrive
mbarrier.init.shared.b64 [smem_addr], N;

// Producer signals completion
mbarrier.arrive.shared.b64 [smem_addr];
// or with TMA:
cp.async.bulk.tensor... , [mbarrier_addr];  // TMA auto-arrives

// Consumer waits
mbarrier.try_wait.parity.shared.b64 [smem_addr], phase;
```

On SM80 (A100), we use `bar.sync` with barrier IDs (0-15):
```cuda
// Named barrier (not __syncthreads)
// Only threads that arrive at this barrier_id are synchronized
asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(thread_count));
```

This is critical: `__syncthreads()` synchronizes ALL threads in the block. Named barriers synchronize only the threads that participate, allowing producer and consumer warps to have independent synchronization domains.

### Register Pressure on SM90

| Configuration | Registers/thread | Warps/SM | Threads/SM | Occupancy |
|---|---|---|---|---|
| 256 regs/thread | 256 | 4 | 128 | 6.25% |
| 128 regs/thread | 128 | 8 | 256 | 12.5% |
| 64 regs/thread | 64 | 16 | 512 | 25% |
| 32 regs/thread | 32 | 32 | 1024 | 50% |

SM90 has 65,536 registers per SM. The tension:
- MMA consumer warps want many registers (accumulator fragments are register-hungry)
- Memory producer warps need very few registers (just pointers and loop variables)
- `setmaxnreg` on SM90 lets us give producers 24-32 regs and consumers 128-256 regs

On SM80 (A100), we cannot dynamically reassign registers. The compiler assigns a fixed count per kernel. But we can still benefit from:
- Keeping producer code simple (few live variables -> compiler uses fewer regs)
- Using `__launch_bounds__` to guide register allocation
- Accepting slightly lower occupancy in exchange for full overlap

---

## How FlashAttention-3 Uses Warp Specialization

Tri Dao's FlashAttention-3 (the official CUDA implementation, not rvLLM's FA3) uses a sophisticated 3-warpgroup design on SM90:

### Warpgroup Layout

A "warpgroup" on SM90 is 4 consecutive warps (128 threads) that execute wgmma instructions together. FA3 uses 3 warpgroups = 12 warps = 384 threads:

```
Warpgroup 0 (warps 0-3):   PRODUCER -- TMA loads of K/V tiles
Warpgroup 1 (warps 4-7):   CONSUMER 1 -- WGMMA for QK^T and PV
Warpgroup 2 (warps 8-11):  CONSUMER 2 -- softmax computation and rescaling
```

Or in the 2-consumer variant:
```
Warpgroup 0 (warps 0-3):   PRODUCER -- issues TMA descriptors, manages mbarriers
Warpgroup 1 (warps 4-7):   MMA CONSUMER -- wgmma.mma_async for QK^T and PV
                            (during softmax, this warpgroup is yielded)
```

### Pipeline Flow

```
             Stage i        Stage i+1        Stage i+2
Producer:    [TMA K_i]      [TMA V_i]        [TMA K_{i+1}]    [TMA V_{i+1}]
             arrive(bar_K)  arrive(bar_V)     arrive(bar_K)     arrive(bar_V)

Consumer 1:  wait(bar_K)              wait(bar_V)     wait(bar_K)
             [QK^T via     [PV via           [QK^T via
              wgmma]        wgmma]            wgmma]

Consumer 2:  [softmax on   [rescale          [softmax on
              QK^T_{i-1}]   acc_{i-1}]        QK^T_i]
```

Key details:
- Producer uses `setmaxnreg.inc.sync.aligned.u32 24` -- only 24 registers per thread
- Consumers use `setmaxnreg.inc.sync.aligned.u32 240` -- 240 registers for MMA fragments
- mbarrier per stage: producer arrives, consumers wait
- Pipeline depth = 2 (producer is 1-2 tiles ahead)
- Shared memory: 2 KV tile buffers (double-buffered) at ~164 KB total for d=128

### Why FA3 Is Faster

On H100 with d=128, ctx=2048:
- FA3 (warp-specialized): ~750 TFLOPS effective (75% of peak)
- FA2 (homogeneous): ~500 TFLOPS effective (50% of peak)
- rvLLM FA3 v3 (cooperative, no specialization): estimated ~350-400 TFLOPS

The difference is almost entirely the overlap. FA3's producer warps keep the TMA unit busy while consumers churn through WGMMA instructions. There's never a cycle where both memory and compute are idle.

---

## How CUTLASS 3.x Uses Warp Specialization

CUTLASS 3.x (the NVIDIA template library for high-performance GEMMs) introduced warp specialization as the default execution model for SM90:

### Mainloop Architecture

```
CollectiveMainloop {
    Producer (1 warpgroup = 4 warps):
        for each K-tile:
            acquire(smem_buffer[stage])        // wait for consumer to release
            TMA_load(A_tile[k] -> smem_A[stage])
            TMA_load(B_tile[k] -> smem_B[stage])
            arrive(load_barrier[stage])        // signal to consumer
            stage = (stage + 1) % NUM_STAGES

    Consumer (1-2 warpgroups = 4-8 warps):
        for each K-tile:
            wait(load_barrier[stage])          // wait for producer
            wgmma.mma_async(smem_A[stage], smem_B[stage], acc_regs)
            release(smem_buffer[stage])        // signal to producer
            stage = (stage + 1) % NUM_STAGES

        // Epilogue: store accumulator to global memory
        store(acc_regs -> C_global)
}
```

### Pipeline Stages

CUTLASS typically uses 2-4 pipeline stages (double to quadruple buffering):

```
Producer:  [Load A0,B0] [Load A1,B1] [Load A2,B2] [Load A3,B3] ...
                        [Load A0,B0] ...
                         (refill)
Consumer:               [MMA stage0] [MMA stage1] [MMA stage2] [MMA stage3] ...
```

With `NUM_STAGES=3`:
- 3 shared memory buffers for A tiles and 3 for B tiles
- Producer can run up to 2 stages ahead
- Consumer processes stage 0 while producer loads stages 1 and 2
- When consumer finishes stage 0, it releases that buffer for reuse

### Register File Management

CUTLASS 3.x on SM90:
```
Producer warpgroup:    setmaxnreg 24    // ~6 KB per warp
Consumer warpgroup:    setmaxnreg 232   // ~58 KB per warp (MMA accumulators)
```

For a 128x128x64 tile with FP16 accumulators:
- Each consumer warp holds 128x128 / (4 warps * 32 threads) = 128 FP32 values = 128 registers
- Plus temporaries, loop variables: ~160-200 registers total
- Producer needs: 2 TMA descriptors, loop counter, stage index: ~16-24 registers

### Why This Matters for rvLLM

rvLLM's `persistent_gemm.cu` uses the CUTLASS *philosophy* (persistent grid, tile looping) but not the CUTLASS *execution model* (warp specialization). The current kernel loads tiles synchronously:

```cuda
// Current: ALL warps do both load and compute
for (int ki = 0; ki < k_tiles; ki++) {
    // All warps load A tile
    for (int idx = tid; idx < smem_a_size; idx += THREADS)
        smem_a[idx] = A[...];
    // All warps load B tile
    for (int idx = tid; idx < smem_b_size; idx += THREADS)
        smem_b[idx] = B[...];
    __syncthreads();
    // All warps do WMMA
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
}
```

The CUTLASS approach would be:
```cuda
// Target: producer warps load, consumer warps compute, overlapped
if (warp_id < NUM_PRODUCER_WARPS) {
    // Producer: cp.async/TMA loads with barrier signaling
    for (int ki = 0; ki < k_tiles; ki++) {
        wait_for_empty_buffer(stage);
        async_load(A_tile, B_tile, smem[stage]);
        signal_full(stage);
        stage = (stage + 1) % NUM_STAGES;
    }
} else {
    // Consumer: MMA with barrier waiting
    for (int ki = 0; ki < k_tiles; ki++) {
        wait_for_full(stage);
        wmma::mma_sync(acc, smem_a[stage], smem_b[stage], acc);
        signal_empty(stage);
        stage = (stage + 1) % NUM_STAGES;
    }
}
```

---

## rvLLM Attention Kernel Analysis

### Current FA3 v3 Decode Kernel Breakdown

Per tile iteration in `fa3_v3_decode_gqa_kernel`:

**Phase 1: Load K tile (cp.async)**
```cuda
v3_async_load_tile(s_kv, key_cache, ...);  // all 256 threads issue cp.async
v3_cp_async_commit();
v3_cp_async_wait_all();                     // STALL: wait for ALL copies
__syncthreads();                            // STALL: wait for ALL threads
```
- Bytes loaded: `tile_len * head_dim * 2` = `64 * 128 * 2` = 16,384 bytes
- Bandwidth at 2 TB/s (A100): 16KB / 2TB/s = 8 us
- But paged KV cache means scattered reads: effective ~1.0-1.5 TB/s = 11-16 us

**Phase 2: QK^T + softmax**
```cuda
// 8 warps each handle 1 KV position per round, 8 rounds for 64 positions
for (base_t = 0; base_t < tile_len; base_t += 8) {
    dot = q_reg * s_kv[t]  // 128 FMA per position
    dot = warp_sum(dot)     // 5 shuffle-xor rounds
}
// block_reduce_max, exp, block_reduce_sum
```
- FLOPs per tile: `64 * 128 * 2` (QK^T) + `64 * 3` (exp+max+sum) = ~16,640 FLOPs
- At 312 TFLOPS (A100 FP16): 16.6K / 312T = 0.053 us
- But these are scalar FMA, not tensor core: effective ~20 TFLOPS = 0.83 us
- Plus 8 `__syncthreads()` in warp-parallel loop + 2 for reductions: ~1-2 us overhead

**Phase 3: Load V tile (cp.async)**
- Same as Phase 1: 11-16 us

**Phase 4: PV accumulation**
```cuda
for (r = 0; r < acc_dims; r++) {
    for (t = 0; t < tile_len; t++)
        acc[r] += s_scores[t] * s_kv[t * head_dim + d];
}
```
- FLOPs: `128 * 64 * 2` = 16,384 FLOPs
- Same scalar FMA issue: ~0.83 us

**Total per tile: ~25-35 us** (dominated by two memory loads)

### Opportunity: What Specialization Would Change

If we split warps 0-1 (64 threads) as producers and warps 2-7 (192 threads) as consumers:

```
Tile i:
  Producer (2 warps): [Load K_i] [Load V_i]  [Load K_{i+1}] [Load V_{i+1}]
  Consumer (6 warps): ............[QK^T_i + softmax_i] [PV_i] [QK^T_{i+1}]...
```

The consumer starts work as soon as K_i is available (no waiting for V_i). Softmax completes while V_i loads. PV accumulation starts immediately when V_i arrives.

Estimated per-tile time in steady state: `max(T_load, T_compute)` = max(~13 us, ~5 us) = ~13 us. Down from ~30 us. **~2.3x speedup for the tile loop.**

But attention is only ~15-25% of total decode time (GEMMs dominate). The end-to-end impact is **15-30% throughput improvement** -- significant but not transformative alone.

---

## Implementation Plan: Attention

### Design: Cooperative Specialization (SM80 Compatible)

The FA3 v3 kernel comment already notes the tradeoff:
> "Full producer/consumer warp specialization would reduce occupancy on this memory-bound kernel (2 blocks/SM -> 1), so cooperative is better."

This is correct for the current single-buffered design. But with double-buffering and proper register budgeting, we can maintain 2 blocks/SM with specialization.

**Target configuration:**
- Block: 256 threads = 8 warps
- Producer warps: warps 0-1 (64 threads) -- cp.async loads
- Consumer warps: warps 2-7 (192 threads) -- QK^T, softmax, PV
- Shared memory: 2 KV tile buffers (double-buffered) = 2 * 64 * 128 * 2 = 32 KB
- Plus scores + scratch: ~2 KB
- Total: ~34 KB per block
- At 2 blocks/SM: 68 KB shared memory (fits in A100's 164 KB shared memory partition)

### Synchronization: Named Barriers on SM80

We use CUDA's `bar.sync` with barrier IDs to create separate synchronization domains:

```cuda
#define BAR_K_READY   1   // producer signals K tile loaded
#define BAR_V_READY   2   // producer signals V tile loaded
#define BAR_K_FREE    3   // consumer signals K buffer released
#define BAR_V_FREE    4   // consumer signals V buffer released

// Producer arrives at a barrier
__device__ void bar_arrive(int bar_id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(bar_id), "r"(count));
}

// Wait at a barrier
__device__ void bar_wait(int bar_id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(count));
}
```

Barrier participant counts:
- `BAR_K_READY`: producer arrives (64 threads), consumer waits (192 threads) -> total 256
- `BAR_V_READY`: same
- `BAR_K_FREE`: consumer arrives (192 threads), producer waits (64 threads) -> total 256
- `BAR_V_FREE`: same

### Kernel Skeleton

```cuda
extern "C"
__global__ void __launch_bounds__(256, 2)
fa3_v4_decode_warp_specialized_kernel(
    __half* __restrict__ output,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_splits
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int is_producer = (warp_id < 2);  // warps 0-1

    // Double-buffered shared memory
    extern __shared__ char smem[];
    __half* s_kv[2];      // two KV tile buffers
    s_kv[0] = (__half*)smem;
    s_kv[1] = s_kv[0] + V3_BC * head_dim;
    float* s_scores = (float*)(s_kv[1] + V3_BC * head_dim);
    float* s_warp = s_scores + V3_BC + 1;

    // ... setup: load Q into registers, init accumulators ...

    if (is_producer) {
        // ===== PRODUCER PATH =====
        // Warps 0-1: dedicated to async KV tile loading

        for (int tile = start_tile; tile < end_tile; tile++) {
            int buf = tile % 2;
            int tile_start = tile * V3_BC;
            int tile_len = min(V3_BC, context_len - tile_start);

            // Wait for consumer to release this buffer
            if (tile >= start_tile + 2) {
                bar_wait(BAR_K_FREE + buf, 256);
            }

            // Load K tile (only 64 threads, but cp.async is bandwidth-limited anyway)
            v3_async_load_tile(s_kv[buf], key_cache, block_tables,
                               seq_idx, max_blocks_per_seq,
                               tile_start, tile_len, num_kv_heads, kv_head_idx,
                               head_dim, block_size, tid);
            v3_cp_async_commit();
            v3_cp_async_wait_all();

            // Signal: K tile ready
            bar_arrive(BAR_K_READY, 64);

            // Load V tile into the SAME buffer (consumer already read K)
            // Wait for consumer to finish with K first
            bar_wait(BAR_V_FREE + buf, 256);

            v3_async_load_tile(s_kv[buf], value_cache, block_tables,
                               seq_idx, max_blocks_per_seq,
                               tile_start, tile_len, num_kv_heads, kv_head_idx,
                               head_dim, block_size, tid);
            v3_cp_async_commit();
            v3_cp_async_wait_all();

            // Signal: V tile ready
            bar_arrive(BAR_V_READY, 64);
        }

    } else {
        // ===== CONSUMER PATH =====
        // Warps 2-7: QK^T + softmax + PV accumulation

        for (int tile = start_tile; tile < end_tile; tile++) {
            int buf = tile % 2;
            int tile_start = tile * V3_BC;
            int tile_len = min(V3_BC, context_len - tile_start);

            // Wait for K tile
            bar_wait(BAR_K_READY, 256);

            // ---- QK^T (warp-parallel, 6 warps) ----
            int consumer_warp = warp_id - 2;  // 0..5
            for (int base_t = 0; base_t < tile_len; base_t += 6) {
                int t = base_t + consumer_warp;
                if (t < tile_len) {
                    float dot = 0.0f;
                    // ... dot product with q_regs and s_kv[buf] ...
                    dot = warp_sum(dot);
                    if (lane_id == 0) s_scores[t] = dot;
                }
            }

            // Signal: done reading K (producer can reuse buffer for V)
            bar_arrive(BAR_V_FREE + buf, 192);

            // ---- Online softmax (6-warp reduction) ----
            // ... block_reduce_max, exp, block_reduce_sum among consumer warps ...

            // Wait for V tile
            bar_wait(BAR_V_READY, 256);

            // ---- PV accumulation ----
            int consumer_tid = tid - 64;  // 0..191
            for (int r = 0; r < acc_dims; r++) {
                int d = consumer_tid + r * 192;
                if (d < head_dim) {
                    for (int t = 0; t < tile_len; t++)
                        acc[r] += s_scores[t] * __half2float(s_kv[buf][t * head_dim + d]);
                }
            }

            // Signal: done reading V (producer can reuse buffer for K)
            bar_arrive(BAR_K_FREE + buf, 192);
        }

        // ---- Write output (consumer warps only) ----
        // ... normalize and write to output/partials ...
    }
}
```

### Barrier Protocol Summary

```
Barrier IDs (using buf = tile % 2, so effectively 4 barriers):

BAR_K_READY:  Producer arrives (64)  after K loaded
              Consumer waits  (192)  before QK^T

BAR_V_FREE:   Consumer arrives (192) after QK^T done (K buffer can be overwritten with V)
              Producer waits  (64)   before loading V into same buffer

BAR_V_READY:  Producer arrives (64)  after V loaded
              Consumer waits  (192)  before PV accum

BAR_K_FREE:   Consumer arrives (192) after PV done (buffer fully released)
              Producer waits  (64)   before loading next K
```

With double buffering:
```
              buf 0                     buf 1
Producer:  [Load K0] -> arrive K_RDY -> [Load K1] -> arrive K_RDY
           wait V_FREE                  wait V_FREE
           [Load V0] -> arrive V_RDY -> [Load V1] -> arrive V_RDY
           wait K_FREE                  wait K_FREE

Consumer:  wait K_RDY -> [QK0] ->       wait K_RDY -> [QK1]
           arrive V_FREE                arrive V_FREE
           wait V_RDY -> [PV0] ->       wait V_RDY -> [PV1]
           arrive K_FREE                arrive K_FREE
```

---

## Implementation Plan: GEMV

The GEMV kernels (`gemv_f16.cu`, `wgmma_gemv.cu`) can also benefit from warp specialization, though the pattern is different because GEMV is purely memory-bandwidth-bound (arithmetic intensity < 1).

### Current GEMV Bottleneck

`gemv_f16_kernel`: one block per output row, 256 threads.
- Each thread loads `k/256` weight elements from HBM and `k/256` input elements from shared memory
- Weight load is the bottleneck: `k * 2` bytes per row from HBM
- For Qwen2.5-1.5B down_proj: k=8960, so 17.9 KB per row from HBM
- One block reads 17.9 KB. At 2 TB/s: 9 ns. But launch overhead + warp scheduling dominates.

`wgmma_gemv_f16_kernel`: one block per 128 output rows, 256 threads.
- Loads input to shared memory, then tiles K dimension with wmma
- Per K-tile: build A tile (broadcast input), build B tile (weight chunk), wmma MMA
- Weight loading is the bottleneck: `128 * k * 2` bytes per block = 128 * 1536 * 2 = 384 KB for QKV
- All warps participate in loading B tiles AND computing MMA

### Warp Specialization for GEMV

For the WGMMA GEMV kernel, split into:
- **Producer warps (0-1):** prefetch weight tiles into shared memory via cp.async
- **Consumer warps (2-7):** wmma MMA on the previously loaded tiles

```
K-tile pipeline:

Producer:  [Load B_0]  [Load B_1]  [Load B_2]  [Load B_3]  ...
Consumer:              [MMA_0]     [MMA_1]     [MMA_2]     [MMA_3]  ...
```

Since GEMV is bandwidth-bound, the benefit is smaller than for attention. But it eliminates the `__syncthreads()` between load and compute phases, and keeps the memory pipeline constantly fed.

**Expected gain: 5-15% for GEMV kernels.** The gain is modest because:
1. GEMV has very low arithmetic intensity (1 FMA per 2 bytes loaded)
2. The memory system is already the bottleneck regardless of overlap
3. Reducing producer thread count means fewer cp.async in flight, potentially reducing bandwidth

The better approach for GEMV is actually **pipelined K-tile loading** where all warps participate in loading the next tile while consuming the current one (cooperative pipelining, which is what the comment in FA3 v3 refers to):

```cuda
// Cooperative pipeline for GEMV (no role split)
// Stage 0: preload first K-tile
async_load_b_tile(smem_b[0], k=0);
commit(); wait_all(); sync();

for (int kt = 0; kt < k_tiles; kt++) {
    // Overlap: start loading NEXT tile while computing CURRENT
    if (kt + 1 < k_tiles) {
        async_load_b_tile(smem_b[(kt+1) % 2], k=(kt+1)*16);
        commit();
    }

    // Compute on current tile (all warps)
    wmma::mma_sync(acc, a_frag, smem_b[kt % 2], acc);

    // Wait for next tile if we started one
    if (kt + 1 < k_tiles) {
        wait_all(); sync();
    }
}
```

This cooperative approach is better for GEMV because:
1. All 8 warps issuing cp.async maximizes memory bandwidth utilization
2. MMA compute is so fast it fits within the load latency window
3. No register waste from idle producer warps

---

## Register File Partitioning

### Budget Analysis: FA3 v4 Attention (Warp-Specialized)

**Producer warps (warps 0-1):**
| Variable | Registers |
|---|---|
| `tid, warp_id, lane_id` | 3 |
| `seq_idx, kv_head_idx, split_idx` | 3 |
| `tile, buf, tile_start, tile_len` | 4 |
| `s_kv pointer (x2 for double-buf)` | 2 |
| `cp.async address calculation` | 6 |
| `loop variables (c, t, ch, page_idx, page_off, phys_block)` | 6 |
| **Total** | **~24 registers** |

**Consumer warps (warps 2-7):**
| Variable | Registers |
|---|---|
| `tid, warp_id, lane_id, consumer_warp, consumer_tid` | 5 |
| `q_regs[HPG][4]` (GQA: 8 heads * 4 regs) | 32 |
| `head_acc[HPG][4]` (accumulator: 8 * 4) | 32 |
| `head_row_max[HPG]` | 8 |
| `head_row_sum[HPG]` | 8 |
| `dot, tile_max, prev_max, new_max, correction, my_exp` | 6 |
| `loop variables, buf, tile, base_t, t, d, r, g` | 8 |
| `s_kv, s_scores, s_warp pointers` | 3 |
| **Total** | **~102 registers** |

### Occupancy Calculation (A100 SM80)

A100: 65,536 registers per SM.

With specialization (mixed register usage):
- Per block: 2 producer warps * 32 threads * 24 regs = 1,536 regs
- Per block: 6 consumer warps * 32 threads * 104 regs = 19,968 regs
- Total per block: 21,504 registers
- 2 blocks/SM: 43,008 registers (65.6% of register file) -- fits

Without specialization (uniform register usage):
- Per block: 8 warps * 32 threads * 104 regs = 26,624 regs
- 2 blocks/SM: 53,248 registers (81.2% of register file) -- tight but fits

Specialization saves **21% of the register file** by not wasting 104 registers on each producer thread. This margin can be used to:
1. Increase `HPG` (heads per group) for wider GQA
2. Increase tile size
3. Allow 3 blocks/SM (if shared memory allows)

### SM90 with `setmaxnreg`

On H100, we can explicitly set:
```cuda
if (is_producer) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 32;\n");
    // Producer code with 32 regs
} else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    // Consumer code with 232 regs
}
```

This gives consumers 232 registers (enough for d=256 heads with full unrolling) while producers use only 32. The register file is dynamically partitioned.

---

## Barrier Synchronization Patterns

### SM80 Named Barriers (A100)

CUDA supports 16 named barriers per thread block (IDs 0-15). Barrier 0 is reserved for `__syncthreads()`. IDs 1-15 are available.

```cuda
// Named barrier: only threads that call this are synchronized
// barrier_id: 1-15
// thread_count: must be a multiple of 32 (warp granularity)
__device__ void named_bar_sync(int barrier_id, int thread_count) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(thread_count));
}

// Split arrive/wait for maximum concurrency:
__device__ void named_bar_arrive(int barrier_id, int thread_count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(thread_count));
}
```

**Important constraint:** `bar.sync` requires that EXACTLY `thread_count` threads arrive before any thread can proceed. If thread_count = 256, ALL warps must participate (equivalent to `__syncthreads()`). To have separate domains:

```cuda
// Producer-consumer pattern with named barriers:
#define N_PRODUCER 64   // 2 warps
#define N_CONSUMER 192  // 6 warps
#define N_ALL 256       // all warps

// Producer signals data ready (only producers arrive)
// Consumer waits until all producers have arrived
// thread_count = N_ALL means both must participate
// To work correctly: producer calls bar.arrive, consumer calls bar.sync

// Alternative: use bar.arrive + bar.wait separately
// bar.arrive: non-blocking, just registers arrival
// bar.sync: blocks until all thread_count threads have arrived
```

Actually, the correct pattern for split-domain synchronization on SM80 is:

```cuda
// Both producer and consumer must call bar.sync with the SAME thread_count
// thread_count is the total that must arrive, not per-role

// Option 1: Full block barriers (simplest, slight overhead)
if (is_producer) {
    // Load K tile
    async_load_k(...);
    cp_async_wait_all();
    __syncthreads();  // barrier 0: all 256 threads
    // K is ready, consumers can proceed

    // Wait for consumers to finish K
    __syncthreads();  // barrier 0: all 256 threads

    // Load V tile
    async_load_v(...);
    cp_async_wait_all();
    __syncthreads();
} else {
    __syncthreads();  // wait for K
    // QK^T + softmax
    __syncthreads();  // signal K consumed
    __syncthreads();  // wait for V
    // PV accumulation
}

// Option 2: Named barriers with arrive/sync split (SM80)
if (is_producer) {
    async_load_k(...);
    cp_async_wait_all();
    asm volatile("bar.arrive 1, 256;");  // I'm done loading
    // Don't wait -- start loading V immediately
    // But need to know V buffer is free...
    asm volatile("bar.sync 2, 256;");    // wait for K consumed
    async_load_v(...);
    cp_async_wait_all();
    asm volatile("bar.arrive 3, 256;");  // V ready
    asm volatile("bar.sync 4, 256;");    // wait for V consumed
} else {
    asm volatile("bar.sync 1, 256;");    // wait for K ready
    // QK^T
    asm volatile("bar.arrive 2, 256;");  // K consumed
    asm volatile("bar.sync 3, 256;");    // wait for V ready
    // PV
    asm volatile("bar.arrive 4, 256;");  // V consumed
}
```

With `bar.arrive`, the thread continues without blocking. With `bar.sync`, it blocks until all `thread_count` threads have arrived at that barrier ID. This gives us the fine-grained control needed.

### SM90 mbarrier (H100)

On SM90, mbarrier objects reside in shared memory and support phases:

```cuda
// In shared memory
__shared__ uint64_t mbar_k[2];  // K ready, double-buffered
__shared__ uint64_t mbar_v[2];  // V ready, double-buffered

// Initialize (once, by tid 0)
if (tid == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(cvta(mbar_k[0])), "r"(expected_count));
    // ... init all 4 mbarriers ...
}
__syncthreads();

// Producer: arrive (non-blocking)
asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(cvta(mbar_k[buf])));

// Consumer: wait
uint32_t phase = 0;
asm volatile(
    "{\n"
    ".reg .pred P;\n"
    "WAIT:\n"
    "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
    "@!P bra WAIT;\n"
    "}\n"
    :: "r"(cvta(mbar_k[buf])), "r"(phase)
);
```

The mbarrier approach is more efficient than `bar.sync` because:
1. It supports **phases** -- no need for multiple barrier IDs
2. **TMA auto-arrives** -- the TMA hardware itself signals the mbarrier when a copy completes
3. **try_wait** -- non-blocking poll, can do useful work while waiting
4. No fixed thread_count -- flexible participation

---

## Pipeline Depth Analysis

### How Many Stages to Overlap?

The optimal pipeline depth depends on the ratio `T_load / T_compute`:

| `T_load / T_compute` | Optimal Stages | Steady-State Utilization |
|---|---|---|
| 1.0 | 2 (double buffer) | ~100% both units |
| 2.0 | 2-3 | ~100% memory, ~50% compute |
| 0.5 | 2 | ~50% memory, ~100% compute |
| 3.0+ | 3-4 | ~100% memory, ~33% compute |

For rvLLM FA3 decode attention:
- `T_load` (K or V tile): ~11-16 us (paged KV, scattered access)
- `T_compute` (QK^T + softmax, or PV): ~2-5 us (scalar FMA, not tensor core)
- Ratio: ~3-5x memory-bound

This means **2 stages (double buffering) is sufficient.** Adding a 3rd stage would only help if we could make the compute phase longer (e.g., by switching QK^T to tensor core MMA, increasing tile size, or processing more heads per tile).

### Shared Memory Cost Per Stage

**Per KV tile buffer:**
```
tile_len * head_dim * sizeof(__half) = 64 * 128 * 2 = 16,384 bytes = 16 KB
```

**Double-buffered (2 stages):**
```
2 * 16 KB = 32 KB for KV tiles
+ scores:  8 * (64+1) * 4 = 2,080 bytes (GQA, 8 heads)
+ scratch: 8 * 4 = 32 bytes
Total: ~34.1 KB per block
```

**At 2 blocks/SM:** 68.2 KB. A100 can configure up to 164 KB shared memory. H100 up to 228 KB. Both easily fit.

**Triple-buffered (3 stages):**
```
3 * 16 KB = 48 KB for KV tiles
+ scores + scratch: ~2.1 KB
Total: ~50.1 KB per block
```

At 2 blocks/SM: 100.2 KB. Still fits but cuts into L1 cache partition on A100 (the shared mem / L1 split is configurable). For decode attention where the access pattern is streaming (not reuse-heavy), 34 KB is sufficient.

**Recommendation: 2 stages (double buffering).** The memory-bound nature of decode attention means a 3rd stage adds shared memory cost without proportional benefit. The producer is already bandwidth-limited, not latency-limited.

---

## Shared Memory Allocation Per Stage

### Detailed Layout for FA3 v4 (Warp-Specialized, Double-Buffered)

```
Address Map (for GQA kernel, head_dim=128, tile_len=64, HPG=8):

Offset 0:     s_kv_buf0[64 * 128]   = 16,384 bytes (K then V, stage 0)
Offset 16384: s_kv_buf1[64 * 128]   = 16,384 bytes (K then V, stage 1)
Offset 32768: s_scores[8 * 65]      =  2,080 bytes (f32, 8 heads * (64+1) stride)
Offset 34848: s_warp[8]             =     32 bytes  (f32, warp reduction scratch)
                                     ----------
Total dynamic shared memory:          34,880 bytes  (~34.1 KB)
```

Versus current FA3 v3 (single-buffered):
```
Offset 0:     s_kv[64 * 128]        = 16,384 bytes
Offset 16384: s_scores[8 * 65]      =  2,080 bytes
Offset 18464: s_warp[8]             =     32 bytes
                                     ----------
Total:                                 18,496 bytes  (~18.1 KB)
```

The double-buffered version uses **1.84x more shared memory** (34.1 KB vs 18.1 KB). At 2 blocks/SM, that's 68 KB vs 36 KB. Both fit comfortably.

### Alignment Requirements

cp.async requires 16-byte aligned source and destination:
- `s_kv_buf0` at offset 0: aligned (start of smem)
- `s_kv_buf1` at offset 16384: aligned (16384 is divisible by 16)
- `s_scores` at offset 32768: aligned
- `s_warp` at offset 34848: 34848 / 16 = 2178. Aligned.

### Non-GQA Variant

For non-GQA (1 head per block):
```
s_kv_buf0[64 * 128]  = 16,384 bytes
s_kv_buf1[64 * 128]  = 16,384 bytes
s_score[64]           =    256 bytes
s_warp[8]             =     32 bytes
Total:                  33,056 bytes (~32.3 KB)
```

---

## Expected Performance Impact

### Attention Kernel Speedup

**Decode attention (FA3 v3 -> v4 warp-specialized):**

| Metric | FA3 v3 (current) | FA3 v4 (target) | Improvement |
|---|---|---|---|
| K load time per tile | 13 us | 13 us (same) | 0% |
| V load time per tile | 13 us | 13 us (same) | 0% |
| QK^T + softmax per tile | 3 us | 4 us (fewer warps) | -33% |
| PV accum per tile | 3 us | 4 us (fewer warps) | -33% |
| **Total per tile** | **32 us** (serial) | **17 us** (overlapped) | **47% faster** |
| Steady-state per tile | 32 us | max(13, 8) = 13 us | 59% faster |

Note: consumer compute gets slightly slower (6 warps instead of 8) but the overlap more than compensates.

**Impact on end-to-end throughput at different context lengths:**

| Context | Tiles | Attention % of step | Speedup (attention) | Speedup (overall) |
|---|---|---|---|---|
| 128 | 2 | 5% | 1.3x | 1.5% |
| 512 | 8 | 15% | 1.5x | 7.5% |
| 2048 | 32 | 30% | 1.6x | 18% |
| 8192 | 128 | 50% | 1.7x | 35% |

At rvLLM's current benchmark (128 tok/req, ~512 context), the expected end-to-end improvement is **7-15%** from attention warp specialization alone.

### GEMV Kernel Speedup

For GEMV, cooperative pipelining (not full specialization) is recommended:

| Metric | Current | Cooperative Pipeline | Improvement |
|---|---|---|---|
| Per K-tile (load+compute) | 2.5 us | 1.8 us | 28% faster |
| QKV projection (k=1536) | 240 us | 180 us | 25% faster |
| Down proj (k=8960) | 560 us | 420 us | 25% faster |

Overall GEMV improvement: 15-25%. Since GEMVs are ~60% of decode time at low N, this translates to **9-15% end-to-end at N=1-32**.

### Combined Impact Estimate

| Batch size | Attention gain | GEMV gain | Combined | New tok/s (from 12,312) |
|---|---|---|---|---|
| N=1 | 1% | 12% | 13% | ~14,000 |
| N=16 | 3% | 10% | 13% | ~14,000 |
| N=64 | 8% | 5% | 13% | ~14,000 |
| N=128 | 12% | 3% | 15% | ~14,200 |
| N=256 | 15% | 2% | 17% | ~14,400 |

These numbers assume the current throughput baseline of 12,312 tok/s (FA3 v3, 128 tok/req).

---

## Code-Level Changes

### Files to Modify

1. **`kernels/flash_attention_3_v4.cu`** (new file)
   - Warp-specialized decode kernel with double-buffered cp.async
   - GQA and non-GQA variants
   - Same combine kernel as v3 (no changes needed)

2. **`kernels/wgmma_gemv.cu`** (modify)
   - Add cooperative pipelining to the K-tile loop
   - Double-buffer the B tile staging area
   - Use cp.async for weight tile loads

3. **`kernels/build.sh`** (modify)
   - Add `flash_attention_3_v4.cu` to the build list
   - Ensure sm_80 minimum for cp.async + named barriers

4. **`crates/rvllm-gpu/src/kernel_loader.rs`** (modify)
   - Register new kernel functions: `fa3_v4_decode_kernel`, `fa3_v4_decode_gqa_kernel`
   - Keep v3 kernels as fallback for verification

5. **`crates/rvllm-model-runner/src/gpu_layer.rs`** (modify)
   - Add runtime selection between v3 and v4 attention kernels
   - Calculate new shared memory sizes for double-buffered variant
   - Feature flag: `RVLLM_WARP_SPEC=1` to enable v4

### Implementation Order

1. **Phase 1: Cooperative pipeline for GEMV** (lowest risk, fastest to implement)
   - Double-buffer B tile in wgmma_gemv
   - Overlap next-tile cp.async with current-tile wmma
   - No barrier changes needed (still use __syncthreads)
   - Validate: output matches to f16 precision

2. **Phase 2: Warp-specialized FA3 v4 decode (non-GQA)**
   - Implement producer/consumer split with named barriers
   - Double-buffered KV tile loading
   - Start with 2 producer warps, 6 consumer warps
   - Validate: output matches FA3 v3 to f16 precision
   - Benchmark: nsight compute roofline analysis

3. **Phase 3: Warp-specialized FA3 v4 decode (GQA)**
   - Extend to GQA with multi-head scoring
   - Tune producer/consumer warp ratio
   - Consider 1 producer warp + 7 consumer warps for GQA (more compute per KV load)

4. **Phase 4: SM90 upgrades (H100 only)**
   - Replace cp.async with TMA loads
   - Replace bar.sync with mbarrier
   - Use setmaxnreg for register partitioning
   - wgmma instructions instead of wmma

### Testing Strategy

- **Correctness:** Compare v4 output against v3 output for all context lengths (1, 64, 512, 2048, 8192)
- **Precision:** Max absolute error should be < 1e-3 for f16 decode
- **Performance:** nsight compute metrics:
  - `smsp__warps_active.avg.pct_of_peak_sustained_active` should increase
  - `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` should increase
  - `dram__bytes_read.sum` should stay the same (we're not reducing memory traffic, just overlapping it)
  - `gpu__time_duration.avg` should decrease by 15-30% for attention kernels

### Kernel Launch Configuration

```rust
// In gpu_layer.rs, attention dispatch:

let use_warp_spec = std::env::var("RVLLM_WARP_SPEC").is_ok();

let (kernel_name, smem_size) = if use_warp_spec {
    let smem = 2 * tile_len * head_dim * 2  // double-buffered KV
             + heads_per_group * (tile_len + 1) * 4  // scores
             + 8 * 4;  // warp scratch
    if is_gqa {
        ("fa3_v4_decode_gqa_kernel", smem)
    } else {
        ("fa3_v4_decode_kernel", smem)
    }
} else {
    // Existing v3 path
    // ...
};
```

---

## Appendix: Warp Scheduling Diagram

### Per-SM Warp Schedule (2 blocks, FA3 v4)

```
SM with 2 blocks (Block A and Block B), 4 quadrants:

Quadrant 0 (warps 0-15):
  Block A warp 0 (producer): LDST LDST LDST LDST ...   (cp.async K tile)
  Block A warp 1 (producer): LDST LDST LDST LDST ...   (cp.async K tile)
  Block B warp 0 (producer): LDST LDST LDST LDST ...   (cp.async K tile)
  Block B warp 1 (producer): LDST LDST LDST LDST ...   (cp.async K tile)

Quadrant 1 (warps 16-31):
  Block A warp 2 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)
  Block A warp 3 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)
  Block B warp 2 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)
  Block B warp 3 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)

Quadrant 2 (warps 32-47):
  Block A warp 4 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)
  Block A warp 5 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)
  Block B warp 4 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)
  Block B warp 5 (consumer): FMA  FMA  FMA  FMA  ...   (QK^T dot product)

Quadrant 3 (warps 48-63):
  Block A warp 6 (consumer): SFU  FMA  SFU  FMA  ...   (softmax exp + PV)
  Block A warp 7 (consumer): SFU  FMA  SFU  FMA  ...   (softmax exp + PV)
  Block B warp 6 (consumer): SFU  FMA  SFU  FMA  ...   (softmax exp + PV)
  Block B warp 7 (consumer): SFU  FMA  SFU  FMA  ...   (softmax exp + PV)

Execution unit utilization:
  LD/ST units: ~80% (producer warps keep them fed)
  FMA units:   ~70% (consumer warps doing dot products)
  SFU:         ~20% (intermittent softmax exp calls)
  Tensor core: 0%   (scalar FMA path -- future: move QK^T to HMMA)
```

### Future: Tensor Core QK^T (Additional Optimization)

The largest remaining gain after warp specialization is moving QK^T from scalar FMA to tensor core HMMA. This is orthogonal to warp specialization but compounds with it:

Current: 6 consumer warps * 32 threads * 2 FMA/cycle = 384 FLOPs/cycle
Target:  6 consumer warps * 1 HMMA(16x8x16)/cycle = 6 * 2048 = 12,288 FLOPs/cycle

That's **32x more compute throughput** for QK^T, making the kernel truly memory-bound rather than compute-bound. At that point, warp specialization's overlap becomes even more critical because compute is so fast that any memory stall is proportionally devastating.

---

## References

1. Tri Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024). Source: github.com/Dao-AILab/flash-attention
2. NVIDIA CUTLASS 3.x documentation: Warp-Specialized Persistent GEMM. Source: github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md
3. NVIDIA H100 Whitepaper: SM90 Architecture. Source: resources.nvidia.com/en-us-tensor-core
4. CUDA C++ Programming Guide: Asynchronous Data Copies, Named Barriers. Source: docs.nvidia.com/cuda/cuda-c-programming-guide
5. rvLLM kernels: `flash_attention_3_v3.cu`, `wgmma_gemv.cu`, `persistent_gemm.cu`
6. rvLLM docs: `optimization-roadmap.md`, `throughput-optimization-spec.md`
