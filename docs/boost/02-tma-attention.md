# 02: TMA Attention

## TMA-Accelerated FlashAttention-3: Migration Plan for rvLLM

## 1. Current Attention Kernel Architecture

### 1.1 Kernel Hierarchy

rvLLM implements a four-tier attention stack, from oldest to newest:

| Kernel | File | SM Requirement | Key Feature |
|---|---|---|---|
| PagedAttention V2 | `kernels/paged_attention.cu` | SM 7.0+ | Baseline, 1 thread per dim |
| FlashAttention-2 | `kernels/flash_attention.cu` | SM 7.0+ | Tiled online softmax, FA2 algorithm |
| FA3 v2 | `kernels/flash_attention_3.cu` | SM 8.0+ | Warp-parallel QKT, bank-free smem |
| **FA3 v3** | `kernels/flash_attention_3_v3.cu` | SM 8.0+ | cp.async, split-KV, cooperative pipelining |

The production path for decode on SM 9.0 (H100) is **FA3 v3** (`fa3_v3_decode_gqa_kernel` / `fa3_v3_decode_kernel`), dispatched by `crates/rvllm-attention/src/backend.rs` via `SplitKvAttention`. The prefill path uses `flash_attention_3_prefill_f16io_kernel` from `kernels/flash_attention_3_prefill.cu`.

### 1.2 FA3 v3 Data Flow (Decode)

The GQA decode kernel (`fa3_v3_decode_gqa_kernel`) processes one KV head per thread block, with all query heads in the GQA group sharing the loaded KV data:

```
Grid: (num_seqs, num_kv_heads, num_splits)
Block: 256 threads (8 warps)
Launch bounds: __launch_bounds__(256, 2)  -- target 2 blocks/SM
```

**Per-tile iteration (BC=64 positions per tile):**

```
Phase 1: cp.async load K[tile] -> s_kv[64 * head_dim]     (all 256 threads)
Phase 2: cp.async.commit_group + cp.async.wait_group 0    (barrier)
Phase 3: __syncthreads()                                   (make smem visible)
Phase 4: Warp-parallel QKT: 8 warps x 1 position each     (warp_id selects KV row)
Phase 5: Block-wide reduce max, online softmax update      (shuffle + smem)
Phase 6: cp.async load V[tile] -> s_kv[64 * head_dim]     (reuse same buffer)
Phase 7: cp.async.commit_group + cp.async.wait_group 0    (barrier)
Phase 8: __syncthreads()                                   (make smem visible)
Phase 9: P@V accumulation (tid-strided over head_dim)      (V reused across GQA heads)
Phase 10: __syncthreads()                                  (prepare for next tile)
```

### 1.3 Current cp.async Implementation

The cp.async path is defined in `flash_attention_3_v3.cu` lines 36-58:

```c
// 16-byte (128-bit) async copy, cache-global policy (bypasses L1)
__device__ __forceinline__ void v3_cp_async_16(void* smem, const void* gmem) {
    unsigned smem_addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem)
    );
}
```

Key characteristics:
- **Granularity**: 16 bytes per instruction (8 f16 values), V3_CHUNK=8
- **Synchronization**: `cp.async.commit_group` + `cp.async.wait_group 0/1`
- **Buffering**: Single-buffered (not double-buffered) to keep smem at ~18KB for 2 blocks/SM occupancy
- **Address computation**: Each thread independently computes page table lookup (line 128-141)
- **Throughput**: 256 threads issue `cp.async.cg` instructions strided over `total_chunks = tile_len * (head_dim / 8)`

### 1.4 Shared Memory Layout

For the GQA kernel with head_dim=128, BC=64:

```
s_kv:     64 * 128 * 2 bytes = 16,384 bytes  (f16, no padding -- dense for cp.async alignment)
s_scores: 8 * 65 * 4 bytes  = 2,080 bytes    (8 GQA heads, stride BC+1 for bank conflicts)
s_warp:   8 * 4 bytes        = 32 bytes       (warp reduction scratch)
Total:    ~18,496 bytes per block
```

At 2 blocks/SM, this consumes ~37KB of the 228KB available shared memory on H100, leaving 191KB for L1 cache.

### 1.5 KV Cache Physical Layout

From `kernels/reshape_and_cache.cu` and `crates/rvllm-kv-cache/src/cache.rs`:

```
Cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
Element access: cache[(phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d]
```

The paged KV cache is non-contiguous: each logical KV position maps to `block_tables[seq_idx * max_blocks + page_idx]`, which gives a physical block number. Within a block, `block_size` consecutive tokens are contiguous. Typical block_size=16.

**This is the critical constraint for TMA**: TMA descriptors assume a contiguous tensor with known strides. Paged KV crosses page boundaries non-contiguously.

## 2. TMA vs cp.async: Hardware-Level Comparison

### 2.1 SM90 Memory Hierarchy

The H100 (SM 9.0a) introduced the **Tensor Memory Accelerator (TMA)**, a dedicated hardware unit for bulk asynchronous data movement between global memory and shared memory. TMA is a separate functional unit from the SM's load/store unit; it operates as an autonomous DMA engine.

```
                  +---------+
                  | SM Core |
                  |  (CUDA  |
                  |  cores) |
                  +----+----+
                       |
          +------------+-------------+
          |                          |
     +----v-----+            +------v------+
     | LD/ST    |            | TMA Unit    |
     | Unit     |            | (HW DMA)    |
     +----+-----+            +------+------+
          |                          |
     +----v--------------------------v----+
     |        Shared Memory (228 KB)      |
     +------------------------------------+
          |
     +----v----+
     | L2 Cache|  (50 MB on H100)
     +----+----+
          |
     +----v----+
     | HBM3    |  (80 GB, 3.35 TB/s)
     +----+----+
```

### 2.2 Instruction-Level Comparison

| Property | cp.async (SM 8.0+) | TMA (SM 9.0+) |
|---|---|---|
| **Instruction** | `cp.async.cg.shared.global [smem], [gmem], 16` | `cp.async.bulk.tensor.Nd.shared::cluster.global.tile [smem], [desc, coords], [mbar]` |
| **Granularity** | 4/8/16 bytes per instruction | Up to 128 bytes per instruction; tiles up to 256x256 elements |
| **Address gen** | Per-thread: each thread computes gmem addr | Per-CTA: one thread issues, TMA unit handles address gen |
| **Synchronization** | `cp.async.commit_group` / `cp.async.wait_group N` | mbarrier (arrive/wait), async proxy fence |
| **Threads consumed** | ALL threads issue loads cooperatively | 1 thread (or warp) issues; others free for compute |
| **Descriptor** | None (raw pointers) | `CUtensorMap` descriptor: base ptr + dims + strides + swizzle |
| **L2 residency** | Cache-global (bypasses L1, uses L2) | Cache-global + prefetch hints + multicast to cluster |
| **Swizzle** | Manual padding (e.g., stride+2) | HW swizzle modes: 32B, 64B, 128B (bank-conflict-free) |
| **Multicast** | N/A | Cluster-wide multicast: one load serves multiple SMs |
| **Occupancy cost** | Threads busy during load | Threads free during load -> warp specialization |

### 2.3 Bandwidth Analysis

**cp.async throughput on H100:**
- Each `cp.async.cg` copies 16 bytes
- 256 threads issuing simultaneously: 256 * 16 = 4,096 bytes per instruction cycle
- With 64 KV positions x 128 dims x 2 bytes/f16 = 16,384 bytes per tile
- Requires 16,384 / 16 = 1,024 cp.async instructions across 256 threads = 4 instructions per thread
- At ~1 instruction per 4 cycles (memory latency hidden by ILP), ~16 cycles per tile load
- **Effective bandwidth**: limited by L2 to ~2.5 TB/s on H100 (observed), ~80% of peak 3.35 TB/s

**TMA throughput on H100:**
- TMA unit processes 128 bytes per transaction (single instruction)
- For a 64x128 f16 tile: 16,384 bytes = 128 TMA transactions
- One thread issues all 128 transactions sequentially, but TMA unit is pipelined
- TMA achieves ~95-98% of peak HBM bandwidth (measured by NVIDIA in FA3 paper)
- **Effective bandwidth**: ~3.2 TB/s (vs ~2.5 TB/s for cp.async)
- **Thread savings**: 255 of 256 threads freed for compute during load

### 2.4 Why TMA is Faster

1. **Address generation offloaded**: cp.async requires each thread to compute `block_tables[...] * block_size + offset * num_kv_heads * head_dim + ...`. This is 5-7 integer operations per thread per 16-byte copy. TMA computes addresses in hardware from a descriptor.

2. **Bank-conflict-free writes**: TMA hardware swizzle modes (32B/64B/128B) eliminate shared memory bank conflicts during the store. Our current v3 kernel removed the +2 padding from v2 (stride=head_dim, not head_dim+2) for cp.async alignment, but this means bank conflicts exist in the QKT read phase.

3. **Warp specialization enables overlap**: With cp.async, all 256 threads participate in loads, then all 256 threads participate in compute. Phases are serialized. With TMA, 1-2 warps can be dedicated "producer" warps that only issue TMA loads, while 6-7 warps are "consumer" warps that only do compute. The phases overlap.

4. **Deeper pipelining**: TMA with mbarrier supports N-stage pipelining (typically 3-4 stages). The current v3 kernel is single-buffered. TMA enables:
   ```
   Stage 0: Load K[tile+2], Compute QKT[tile], Load V[tile-1]  (3-deep)
   ```

## 3. How Tri Dao's FlashAttention-3 Uses TMA

Tri Dao's official FlashAttention-3 (published 2024, integrated into vLLM) uses TMA as a core component. Key design elements, reconstructed from the FA3 paper and CUDA source:

### 3.1 TMA Descriptor Setup (Host-Side)

```cpp
// Pseudocode for FA3 TMA descriptor creation (host-side, before kernel launch)
CUtensorMap tma_desc_K;
CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;

// K cache is [seq_len, num_kv_heads, head_dim] -- contiguous
uint64_t dims[3] = {head_dim, num_kv_heads, seq_len};
uint64_t strides[2] = {head_dim * sizeof(half), num_kv_heads * head_dim * sizeof(half)};

// Box dimensions: tile_K x head_dim (e.g., 64 x 128)
uint32_t box_dim[2] = {head_dim, TILE_K};
uint32_t elem_strides[2] = {1, 1};

cuTensorMapEncodeTiled(
    &tma_desc_K,
    dtype,
    2,              // num dimensions
    (void*)K_base,  // global base pointer
    dims,           // global dims
    strides,        // global strides (in bytes)
    box_dim,        // tile dims
    elem_strides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,     // 128-byte swizzle for bank-conflict-free
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

### 3.2 Kernel-Side TMA Load

```c
// In-kernel TMA load (single thread issues, rest compute)
// Only executed by the "producer" warp (warp 0)
if (warp_id == PRODUCER_WARP && lane_id == 0) {
    // Arrive at the mbarrier to signal load completion
    uint64_t* mbar = &smem_barriers[stage];

    // TMA 2D tile load: K[kv_offset : kv_offset+TILE_K, head_offset : head_offset+HEAD_DIM]
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr),          // shared memory destination
           "l"(tma_desc_ptr),       // TMA descriptor (in constant memory)
           "r"(coord_head),         // coordinate 0: head_dim offset
           "r"(coord_seq),          // coordinate 1: sequence position offset
           "r"(mbar_addr)           // mbarrier to signal on completion
    );
}
```

### 3.3 Async Barrier (mbarrier) Pattern

FA3 uses SM 9.0's mbarrier for producer-consumer synchronization:

```c
// ---- Producer warp (warp 0): prefetch next tile ----
if (warp_id == 0) {
    // Initialize barrier for this pipeline stage
    if (lane_id == 0) {
        // Set expected arrive count = 1 (TMA completes once)
        // Plus expected_bytes from TMA (the HW tracks bytes automatically)
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(mbar_smem_addr), "r"(arrive_count)
        );
    }
    __syncwarp();

    // Issue TMA load
    if (lane_id == 0) {
        // cp.async.bulk.tensor... (as above)
    }

    // Arrive at the barrier (producer side)
    if (lane_id == 0) {
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(mbar_smem_addr), "r"(expected_bytes)
        );
    }
}

// ---- Consumer warps (warps 1-7): wait for data, then compute ----
if (warp_id > 0) {
    // Wait for the TMA load to complete
    uint32_t phase_bit = stage & 1;
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra WAIT_LOOP;\n"
        "}\n"
        :: "r"(mbar_smem_addr), "r"(phase_bit)
    );
}
```

### 3.4 Warp Specialization Pattern

FA3 uses 1 producer warp + 7 consumer warps (for 256-thread blocks):

```
Warp 0 (Producer):       Warps 1-7 (Consumers):
  Load K[tile N+1]         Compute QKT[tile N]
  Load V[tile N]           Softmax[tile N]
  Signal mbarrier          Wait mbarrier
  Wait consumer done       Compute PV[tile N]
  Repeat                   Signal consumer done
                           Repeat
```

This achieves full overlap: while consumers compute on tile N, the producer has already loaded tile N+1's K and is loading tile N's V. The pipeline is 2-3 stages deep.

### 3.5 FA3 WGMMA Integration

Official FA3 uses WGMMA (Warp Group MMA) for the QKT and PV matmuls:

```c
// QKT: [Br, d] x [d, Bc]^T -> [Br, Bc]
// Uses wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
// Operand A (Q) in registers, operand B (K) in shared memory
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, "
    " %8, %9, %10, %11, %12, %13, %14, %15}, "
    "%16, %17, "
    "1, 1, 1;\n"   // scale_d, imm_a, imm_b
    : ... // accumulator registers (D)
    : ... // A descriptor register, B shared memory address
);
```

rvLLM's current FA3 v3 does NOT use WGMMA -- it uses scalar FMA for QKT dot products and PV accumulation. This is a separate optimization from TMA but is synergistic: TMA feeds data to smem, WGMMA consumes it from smem.

## 4. Step-by-Step Migration Plan: cp.async to TMA

### 4.1 Phase 0: Infrastructure Prerequisites

**4.1.1 TMA Descriptor Host-Side API**

Create `kernels/tma_utils.cuh` with host-side descriptor creation:

```cpp
// Must be called from host code (Rust FFI or CUDA host function)
// cuTensorMapEncodeTiled is a driver API function
extern "C" void create_tma_descriptor_2d(
    CUtensorMap* desc,
    void* base_ptr,
    uint64_t dim0,       // head_dim
    uint64_t dim1,       // total_tokens (block_size * num_blocks)
    uint64_t stride0,    // head_dim * sizeof(half)  -- inner stride (bytes)
    uint32_t box_dim0,   // head_dim (tile width)
    uint32_t box_dim1,   // TILE_K (tile height, e.g., 64)
    CUtensorMapSwizzle swizzle  // CU_TENSOR_MAP_SWIZZLE_128B
);
```

**4.1.2 Rust-Side Descriptor Management**

In `crates/rvllm-attention/src/tma.rs` (new file):

```rust
/// Manages TMA descriptors for paged KV cache.
/// Since KV cache is paged (non-contiguous), we need one descriptor
/// per physical block, or use the linearized approach (see Section 6).
pub struct TmaDescriptorPool {
    k_descriptors: Vec<CUtensorMap>,  // one per physical block
    v_descriptors: Vec<CUtensorMap>,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
}
```

**4.1.3 Compile Flags**

The kernel must be compiled with `-arch=sm_90a` (note the `a` suffix for TMA + cluster support). The existing CUTLASS kernels already use this flag.

### 4.2 Phase 1: TMA for Contiguous Prefill (Low Risk)

Start with the prefill kernel (`flash_attention_3_prefill.cu`) where Q/K/V are contiguous tensors (not paged). This validates TMA plumbing without the paged-cache complication.

**New file**: `kernels/flash_attention_3_tma_prefill.cu`

**Changes:**
1. Create TMA descriptors for Q [num_tokens, num_heads, head_dim], K [num_tokens, num_kv_heads, head_dim], V same
2. Replace the explicit half2 load loops with TMA 2D tile loads
3. Add mbarrier initialization in shared memory
4. Keep all 8 warps as consumers (no warp specialization yet -- just replace load mechanism)

**PTX sequence for tile load:**

```asm
// Initialize mbarrier (once at kernel start)
mbarrier.init.shared::cta.b64 [mbar_smem], 1;

// For each tile:
// Thread 0 issues TMA load
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    [smem_K_addr], [tma_desc_K, {0, tile_start}], [mbar_smem];

// Thread 0 sets expected transaction bytes
mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar_smem], expected_bytes;

// All threads wait
mbarrier.try_wait.parity.shared::cta.b64 pred, [mbar_smem], phase;
```

**Expected gain from Phase 1**: 5-10% bandwidth improvement on prefill, primarily from HW address generation and 128B swizzle.

### 4.3 Phase 2: TMA with Linearized KV for Decode (Medium Risk)

For decode, the KV cache is paged. We solve this with a **per-block TMA descriptor** approach:

**Strategy**: Create one TMA descriptor per physical KV block. Each descriptor describes a contiguous [block_size, num_kv_heads, head_dim] region. At kernel launch, pass the block table and an array of TMA descriptors. Inside the kernel, for each tile, look up the physical block and use the corresponding descriptor.

**New file**: `kernels/flash_attention_3_tma_v4.cu`

**Shared memory layout (double-buffered):**

```
Buffer 0: K tile [64, 128] f16 = 16,384 bytes   (swizzled)
Buffer 1: K tile [64, 128] f16 = 16,384 bytes   (swizzled)
mbarrier[2]: 2 * 8 bytes = 16 bytes
scores[8 * 65]: 2,080 bytes
warp_scratch[8]: 32 bytes
Total: ~34,896 bytes per block
```

At 34,896 bytes, we can still fit 2 blocks/SM on H100 (228KB smem total, ~70KB used, 158KB for L1).

**Key implementation detail -- crossing block boundaries within a tile:**

A 64-position tile may span 4 physical blocks (at block_size=16). Since TMA descriptors describe a single contiguous block, we issue **multiple TMA loads per tile**, one per physical block:

```c
// Load a tile that spans potentially multiple pages
for (int t = 0; t < tile_len; ) {
    int kv_pos = tile_start + t;
    int page_idx = kv_pos / block_size;
    int page_off = kv_pos % block_size;
    int phys_block = __ldg(&block_tables[seq_idx * max_blocks + page_idx]);

    // How many positions we can load from this page
    int remaining_in_page = block_size - page_off;
    int load_len = min(remaining_in_page, tile_len - t);

    // Issue TMA for this contiguous chunk
    if (threadIdx.x == 0) {
        // 2D TMA: load [load_len, head_dim] from position page_off in block phys_block
        // Descriptor indexes into the per-block descriptor array
        CUtensorMap* desc = &tma_descs[phys_block * num_kv_heads + kv_head_idx];
        // Coordinate: (0, page_off) -- head_dim offset 0, position offset page_off
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(smem_addr + t * head_dim * 2),
               "l"(desc),
               "r"(0),          // head_dim coordinate (start at 0)
               "r"(page_off),   // position coordinate within block
               "r"(mbar_addr)
        );
    }
    t += load_len;
}
```

**Descriptor creation (host-side, per block):**

```c
for (int blk = 0; blk < num_physical_blocks; blk++) {
    for (int kv_head = 0; kv_head < num_kv_heads; kv_head++) {
        CUtensorMap desc;
        half* base = &cache[blk * block_size * num_kv_heads * head_dim
                            + kv_head * head_dim];
        uint64_t dims[2] = {head_dim, block_size};
        uint64_t strides[1] = {num_kv_heads * head_dim * sizeof(half)};
        uint32_t box[2] = {head_dim, block_size};  // load entire block at once

        cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2, base, dims, strides, box, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,  // prefetch hint
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

        tma_descs[blk * num_kv_heads + kv_head] = desc;
    }
}
```

**Expected gain from Phase 2**: 15-25% overall decode throughput improvement from:
- HW address generation (saves 5-7 int ops per thread per 16B copy)
- 128B swizzle (eliminates bank conflicts in QKT reads)
- L2 prefetch hints
- Double-buffering overlap (K[tile+1] loads while computing on K[tile])

### 4.4 Phase 3: Warp Specialization (High Impact)

Restructure the kernel into producer/consumer warps:

```c
#define PRODUCER_WARPS 1
#define CONSUMER_WARPS 7
#define PIPELINE_STAGES 3

__shared__ __align__(128) uint64_t mbarriers[PIPELINE_STAGES];

if (warp_id < PRODUCER_WARPS) {
    // ---- PRODUCER WARP ----
    for (int tile = start_tile; tile < end_tile; tile++) {
        int stage = tile % PIPELINE_STAGES;

        // Wait for consumers to finish with this stage's buffer
        mbarrier_wait_consumer(mbarriers[stage]);

        // Issue TMA loads for K tile
        tma_load_kv_tile(s_K[stage], tma_desc_K, tile, ...);
        mbarrier_arrive_tx(mbarriers[stage], K_bytes);

        // Issue TMA loads for V tile into separate buffer
        tma_load_kv_tile(s_V[stage], tma_desc_V, tile, ...);
        mbarrier_arrive_tx(mbarriers[stage], V_bytes);
    }
} else {
    // ---- CONSUMER WARPS (7 warps) ----
    for (int tile = start_tile; tile < end_tile; tile++) {
        int stage = tile % PIPELINE_STAGES;

        // Wait for K data to arrive
        mbarrier_wait_producer(mbarriers[stage]);

        // QKT computation (7 warps, each handles ~9 of 64 positions)
        // More warps per position = faster intra-warp reduction
        parallel_qkt(q_regs, s_K[stage], s_scores, tile_len);

        // Online softmax
        online_softmax(s_scores, tile_len, ...);

        // Wait for V data (may already be here if producer was fast)
        // (V loads overlapped with QKT compute)

        // PV accumulation
        accumulate_pv(s_scores, s_V[stage], head_acc, tile_len);

        // Signal producer that this stage buffer is free
        mbarrier_arrive_consumer(mbarriers[stage]);
    }
}
```

**Shared memory layout (3-stage pipeline, separate K and V buffers):**

```
K buffer[3]: 3 * 64 * 128 * 2 =  49,152 bytes  (3 stages of K tiles)
V buffer[3]: 3 * 64 * 128 * 2 =  49,152 bytes  (3 stages of V tiles)
mbarrier[3]: 3 * 8            =       24 bytes
scores[8 * 65]:               =    2,080 bytes
warp_scratch[8]:              =       32 bytes
Total:                          ~100,440 bytes per block
```

At ~100KB, we drop to **1 block/SM**. This is acceptable because warp specialization compensates: the producer warp saturates HBM bandwidth while 7 consumer warps saturate compute.

**Alternative: shared K/V buffer (2-stage)**

If 100KB is too much, share the K/V buffer (load K, compute QKT, then load V into same buffer, compute PV):

```
KV buffer[2]: 2 * 64 * 128 * 2 = 32,768 bytes  (2 stages, K or V)
mbarrier[2]: 2 * 8             =      16 bytes
scores[8 * 65]:                =   2,080 bytes
warp_scratch[8]:               =      32 bytes
Total:                           ~34,896 bytes per block -> 2 blocks/SM possible
```

The FA3 v3 comment (line 13-14) explicitly notes: "Full producer/consumer warp specialization would reduce occupancy on this memory-bound kernel (2 blocks/SM -> 1), so cooperative is better." This was true for cp.async where the load mechanism requires all threads. With TMA, the producer warp is nearly free -- it issues a single instruction and the TMA unit does the rest. The bandwidth gain from warp specialization typically outweighs the occupancy loss.

**Decision point**: Profile both 1-block warp-specialized and 2-block cooperative-TMA. On H100 for decode (memory-bound), the warp-specialized version should win because:
- Peak HBM BW requires fewer SMs than available
- 1 block/SM x 132 SMs = 132 active blocks (still full GPU)
- The freed compute warps can process QKT+softmax+PV faster

**Expected gain from Phase 3**: Additional 20-35% decode throughput from overlapped load/compute.

### 4.5 Phase 4: WGMMA for QKT and PV (Compute Optimization)

This is synergistic with TMA but not dependent on it. Replace scalar FMA dot products with WGMMA:

```c
// Current: scalar dot product per warp
float dot = 0.0f;
for (int r = 0; r < half2_iters; r++) {
    int d = lane_id * 2 + r * 64;
    if (d + 1 < head_dim) {
        dot += q_reg[r*2] * __half2float(s_kv[t * head_dim + d]);
        dot += q_reg[r*2+1] * __half2float(s_kv[t * head_dim + d + 1]);
    }
}
dot = v3_warp_sum(dot);

// Proposed: WGMMA (warp group = 4 warps = 128 threads)
// Reshape Q[1, 128] as [64, 128] (broadcast) and K[64, 128] -> S[64, 64]
// wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
// A = Q tile (in registers), B = K tile (in shared memory with swizzle)
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
    "{...accumulators...}, %desc_a, %desc_b, 1, 1, 1;\n"
    : ... : ...
);
```

WGMMA requires TMA's 128B swizzle layout in shared memory -- this is why TMA migration naturally enables WGMMA.

## 5. Expected Performance Gains

### 5.1 Theoretical Bandwidth Model

For decode with Qwen2.5-7B (head_dim=128, 4 KV heads, block_size=16, context_len=512):

**Current FA3 v3 (cp.async):**
- KV bytes per tile: 64 * 128 * 2 = 16,384 bytes (K) + 16,384 bytes (V) = 32,768 bytes
- Tiles per sequence: ceil(512 / 64) = 8
- Total KV bytes per seq per KV head: 8 * 32,768 = 262,144 bytes
- Total KV bytes per seq (4 KV heads): 1,048,576 bytes = 1 MB
- At 128 sequences: 128 MB
- cp.async effective BW: ~2.5 TB/s -> load time: 128 MB / 2.5 TB/s = 51.2 us
- Compute (FMA): 128 seqs * 512 positions * 128 dim * 2 (QKT + PV) = ~16.8M FLOPs per head
- At ~990 TFLOPS f16: ~17 ns (completely compute-bound if overlap works)

**With TMA + warp specialization:**
- TMA effective BW: ~3.2 TB/s -> load time: 128 MB / 3.2 TB/s = 40.0 us
- Overlapped compute hides latency -> effective time approaches max(load, compute)
- load time dominates -> ~40 us vs ~51 us = **22% faster**

**With TMA + warp specialization + WGMMA:**
- WGMMA processes QKT as [64, 128] x [128, 64] matmul per tile
- At ~990 TFLOPS: trivially fast (< 1 us per tile)
- Fully overlap with TMA loads -> approach TMA bandwidth limit
- Expected: **25-30% faster** than current cp.async

### 5.2 Projected Throughput

Based on current Phase 6 numbers (12,312 tok/s at N=128, 128 tok/req on H100):

| Configuration | Expected tok/s | Improvement |
|---|---|---|
| FA3 v3 (current cp.async) | 12,312 | baseline |
| + TMA (no warp spec) | ~13,500 | +10% |
| + TMA + warp specialization | ~15,400 | +25% |
| + TMA + warp spec + WGMMA | ~16,000 | +30% |
| Tri Dao FA3 (reference) | ~17,000-18,000 | -- |

The remaining gap to Tri Dao's FA3 after TMA+warpspec+WGMMA would be:
- Cluster-level multicast (TMA multicast across SM cluster for GQA)
- Advanced software pipelining (ping-pong buffers with 4+ stages)
- FP8 WGMMA for KV (halves bandwidth requirement)

## 6. TMA Descriptor Management for Paged KV Cache

### 6.1 The Core Challenge

TMA descriptors encode a **contiguous** tensor's base address, dimensions, and strides. The paged KV cache is **non-contiguous** -- logical position N may map to physical block A, while position N+1 maps to physical block B (if they cross a page boundary).

Three approaches, in order of increasing complexity:

### 6.2 Approach A: Per-Block Descriptor Array (Recommended)

Create one TMA descriptor per (physical_block, kv_head) pair. At kernel launch, pass the entire descriptor array to the kernel via constant memory or global memory.

**Host-side setup:**
```cpp
// Allocate descriptor array: num_physical_blocks * num_kv_heads descriptors
// Each descriptor covers [block_size, head_dim] for one KV head in one physical block
std::vector<CUtensorMap> descriptors(num_blocks * num_kv_heads);

for (int blk = 0; blk < num_blocks; blk++) {
    for (int kv = 0; kv < num_kv_heads; kv++) {
        CUtensorMap& desc = descriptors[blk * num_kv_heads + kv];

        // Base pointer for this block/head
        half* base = &kv_cache[blk * block_size * num_kv_heads * head_dim
                               + kv * head_dim];

        // Tensor shape: [head_dim, block_size]
        //   dim0 = head_dim (innermost, contiguous)
        //   dim1 = block_size (strided by num_kv_heads * head_dim)
        uint64_t global_dim[2] = {(uint64_t)head_dim, (uint64_t)block_size};
        uint64_t global_stride[1] = {(uint64_t)(num_kv_heads * head_dim * sizeof(half))};
        uint32_t box_dim[2] = {(uint32_t)head_dim, (uint32_t)block_size};
        uint32_t elem_stride[2] = {1, 1};

        cuTensorMapEncodeTiled(&desc,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2, base, global_dim, global_stride, box_dim, elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }
}
```

**Kernel-side tile load:**
```c
__device__ void tma_load_kv_tile_paged(
    __half* smem_buf,
    const CUtensorMap* __restrict__ tma_descs,  // [num_blocks * num_kv_heads]
    const int* __restrict__ block_tables,
    int seq_idx, int max_blocks,
    int tile_start, int tile_len,
    int num_kv_heads, int kv_head_idx,
    int head_dim, int block_size,
    uint64_t* mbar
) {
    if (threadIdx.x != 0) return;  // Only thread 0 issues TMA

    int smem_offset = 0;
    for (int t = 0; t < tile_len; ) {
        int kv_pos = tile_start + t;
        int page_idx = kv_pos / block_size;
        int page_off = kv_pos % block_size;
        int phys_block = __ldg(&block_tables[seq_idx * max_blocks + page_idx]);

        int remaining_in_page = block_size - page_off;
        int chunk_len = min(remaining_in_page, tile_len - t);

        // TMA descriptor for this physical block and KV head
        const CUtensorMap* desc = &tma_descs[phys_block * num_kv_heads + kv_head_idx];

        // Issue TMA load: [head_dim, chunk_len] starting at position page_off
        uint32_t smem_addr = __cvta_generic_to_shared(smem_buf + smem_offset);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(smem_addr),
               "l"(desc),
               "r"(0),          // head_dim coord (always 0 -- load full row)
               "r"(page_off),   // position coord within this block
               "r"((uint32_t)__cvta_generic_to_shared(mbar))
        );

        smem_offset += chunk_len * head_dim;
        t += chunk_len;
    }

    // Signal expected transaction bytes
    uint32_t expected_bytes = tile_len * head_dim * sizeof(__half);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(expected_bytes)
    );
}
```

**Descriptor array size**: For 10,000 physical blocks and 4 KV heads, we need 40,000 descriptors. Each `CUtensorMap` is 128 bytes, so total = 5 MB. This fits easily in GPU global memory and can be prefetched to L2.

**Descriptor invalidation**: When physical blocks are freed/reallocated (KV eviction), the corresponding descriptors must be updated. This happens at the scheduler level between batches, not during kernel execution. The update is host-side: `cuTensorMapEncodeTiled()` with the new base pointer.

### 6.3 Approach B: Gather-then-TMA (Hybrid)

Use a fast gather kernel to copy scattered KV pages into a contiguous staging buffer, then use TMA from the staging buffer.

**Pros**: Single TMA descriptor per sequence, simpler kernel
**Cons**: Extra copy (bandwidth overhead), extra GPU memory for staging buffer

Rejected: the gather copy costs the same bandwidth we're trying to save.

### 6.4 Approach C: Restructure KV Cache Layout

Allocate KV cache as a single contiguous buffer and use a custom allocator that keeps pages physically contiguous per sequence.

**Pros**: TMA works perfectly with a single descriptor per sequence
**Cons**: Requires rewriting the entire paged KV cache system, defeats purpose of paging (fragmentation), loses flexibility

Rejected: too invasive, paging exists for a reason.

### 6.5 Recommended Approach

**Approach A (per-block descriptors)** is the right choice because:
1. Minimal changes to existing KV cache layout
2. TMA's address generation HW handles the within-block striding
3. Multiple TMA issues per tile (one per page crossing) is still far cheaper than per-thread cp.async
4. Typical tile (64 positions, block_size=16) crosses 4 pages -> 4 TMA issues from 1 thread vs 1,024 cp.async from 256 threads

## 7. SM90 Hardware Details: TMA Unit Capabilities

### 7.1 TMA Architecture

The TMA unit on H100 is a dedicated DMA controller embedded in each SM. It operates independently from the SM's warp schedulers and can issue requests to the L2 cache / HBM while the SM processes other instructions.

**Key capabilities:**
- **Bulk async copy**: `cp.async.bulk.tensor.{1-5}d.shared::cluster.global.{tile,im2col}` -- loads multi-dimensional tiles from global memory to shared memory
- **Multicast**: Load once from global, broadcast to multiple SMs in a cluster (CU_TENSOR_MAP_MULTICAST_ENABLE)
- **Prefetch**: `cp.async.bulk.prefetch.tensor.Nd.L2.global.tile` -- prefetch to L2 without shared memory destination
- **Store**: `cp.async.bulk.tensor.Nd.global.shared::cta.tile` -- store from shared to global (for output)
- **Swizzle modes**: 32B, 64B, 128B -- rearranges bytes within each tile row to avoid bank conflicts
- **OOB fill**: Out-of-bounds coordinates return 0 (configurable) -- useful for partial tiles at sequence boundaries

### 7.2 TMA Transaction Flow

```
1. Thread issues cp.async.bulk.tensor.2d instruction
   -> Encodes: descriptor ptr, coordinates, mbarrier ptr, smem destination

2. TMA unit decodes the descriptor:
   -> Base address, global dimensions, global strides, box dimensions
   -> Computes actual global addresses for each element in the tile

3. TMA unit generates L2 cache requests:
   -> Coalesced 128-byte cache line requests
   -> L2 residency hint applied (if L2_256B promotion enabled)

4. L2 serves from cache or fetches from HBM:
   -> TMA unit applies swizzle on write to shared memory
   -> Writes are 128-byte aligned, bank-conflict-free

5. When all bytes written, TMA unit decrements mbarrier:
   -> mbarrier transaction count tracks bytes, not operations
   -> When transaction count reaches 0, barrier is released
```

### 7.3 mbarrier (Async Barrier) Architecture

SM 9.0 introduces hardware-accelerated barriers in shared memory:

```
// Barrier layout in shared memory (8 bytes per barrier)
// [63:32] = transaction count (bytes remaining)
// [31:0]  = arrival count (threads/phases)
struct mbarrier_t {
    uint32_t arrival_count;   // decremented by mbarrier.arrive
    uint32_t tx_count;        // decremented by TMA completions
};
```

**Key PTX instructions:**

```asm
// Initialize barrier with arrive count and tx count
mbarrier.init.shared::cta.b64 [mbar], arrive_count;

// Producer: set expected transaction bytes (before TMA issue)
mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], expected_bytes;

// Consumer: wait for all transactions and arrivals to complete
// Uses phase bit for ping-pong (avoids resetting barrier)
mbarrier.try_wait.parity.shared::cta.b64 pred, [mbar], phase_bit;
// @!pred bra RETRY;  (spin loop)

// Alternative: blocking wait (with timeout)
mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 pred, [mbar], phase_bit, timeout;
```

### 7.4 Async Proxy and Fence

TMA operations go through the "async proxy" -- a separate address space that does not participate in the SM's memory consistency model. To make TMA results visible to the SM's load/store unit, a fence is required:

```asm
// After mbarrier wait succeeds, fence to make TMA writes visible
fence.proxy.async.shared::cta;
// Now shared memory reads from the SM will see TMA-written data
```

In practice, `mbarrier.try_wait.parity.acquire` includes an implicit acquire fence, so the explicit `fence.proxy.async` is not always needed. But for correctness, the pattern is:

```c
// Producer issues TMA
// Consumer waits on mbarrier (acquire semantics)
// Consumer reads shared memory (guaranteed to see TMA writes)
```

### 7.5 Swizzle Modes

TMA swizzle rearranges bytes within each 128-byte row of the tile to distribute bank accesses:

| Mode | Description | Use Case |
|---|---|---|
| NONE | No swizzle | When reads are already bank-conflict-free |
| 32B | XOR within 32-byte groups | Narrow tiles (head_dim <= 32) |
| 64B | XOR within 64-byte groups | Medium tiles |
| **128B** | XOR within 128-byte groups | **Optimal for head_dim=128, f16** (128*2=256 bytes/row) |

For head_dim=128 with f16 (256 bytes per row), 128B swizzle is optimal: it ensures that when 32 threads in a warp read consecutive elements from a shared memory row, the accesses are spread across all 32 banks.

**Swizzle layout for head_dim=128, f16, 128B mode:**

```
Logical row [0..127] f16 = [0..255] bytes:
  Physical bytes [0..127]:   XOR with (row_in_tile & 7) << 4
  Physical bytes [128..255]: XOR with (row_in_tile & 7) << 4

Result: row 0 is identity, row 1 has bytes shifted by 16, etc.
When warp reads column j from row r, the bank = (j * 2 XOR (r & 7) * 16) % 128 / 4
-> No bank conflicts for any access pattern where lanes read same column across rows.
```

## 8. Code-Level Implementation Plan

### 8.1 New Files to Create

| File | Purpose |
|---|---|
| `kernels/tma_attention_common.cuh` | TMA intrinsics, mbarrier helpers, swizzle utils |
| `kernels/flash_attention_3_tma_prefill.cu` | Phase 1: TMA prefill kernel |
| `kernels/flash_attention_3_tma_v4.cu` | Phase 2-3: TMA decode kernel with warp spec |
| `crates/rvllm-attention/src/tma.rs` | Rust-side TMA descriptor pool management |
| `crates/rvllm-attention/src/tma_attention.rs` | TMA attention backend (AttentionBackend impl) |

### 8.2 `kernels/tma_attention_common.cuh`

```cpp
#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

// ============================================================
// mbarrier helpers
// ============================================================

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(arrive_count));
}

__device__ __forceinline__ void mbar_arrive_tx(uint64_t* mbar, uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(tx_bytes));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
}

__device__ __forceinline__ void mbar_wait_parity(uint64_t* mbar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase));
}

// ============================================================
// TMA load wrapper
// ============================================================

__device__ __forceinline__ void tma_load_2d(
    void* smem_dst,
    const void* tma_desc,  // CUtensorMap*
    uint32_t coord0,       // inner dim (head_dim offset)
    uint32_t coord1,       // outer dim (position offset)
    uint64_t* mbar
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(tma_desc),
           "r"(coord0), "r"(coord1), "r"(mbar_addr)
        : "memory"
    );
}

// ============================================================
// Async proxy fence
// ============================================================

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
```

### 8.3 Kernel Skeleton: `flash_attention_3_tma_v4.cu`

```cpp
#include "tma_attention_common.cuh"
#include <float.h>

#define TMA_BC 64
#define TMA_THREADS 256
#define TMA_WARPS 8
#define TMA_PRODUCER_WARPS 1
#define TMA_CONSUMER_WARPS 7
#define TMA_GQA_MAX_HPG 8
#define TMA_SCORE_STRIDE (TMA_BC + 1)
#define TMA_STAGES 2

extern "C"
__global__ void __launch_bounds__(TMA_THREADS, 2)
fa3_tma_decode_gqa_kernel(
    __half* __restrict__ output,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const __half* __restrict__ query,
    const CUtensorMap* __restrict__ k_tma_descs,  // [num_phys_blocks * num_kv_heads]
    const CUtensorMap* __restrict__ v_tma_descs,  // [num_phys_blocks * num_kv_heads]
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_splits
) {
    const int seq_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // ... (context_len check, split-KV range, same as v3) ...

    // Shared memory: double-buffered KV + mbarriers + scores
    extern __shared__ char smem_raw[];
    __half* s_kv[TMA_STAGES];
    s_kv[0] = (__half*)smem_raw;
    s_kv[1] = s_kv[0] + TMA_BC * head_dim;

    uint64_t* s_mbar = (uint64_t*)(s_kv[1] + TMA_BC * head_dim);
    float* s_scores  = (float*)(s_mbar + TMA_STAGES);
    float* s_warp    = s_scores + TMA_GQA_MAX_HPG * TMA_SCORE_STRIDE;

    // Initialize mbarriers (once)
    if (tid == 0) {
        for (int s = 0; s < TMA_STAGES; s++) {
            mbar_init(&s_mbar[s], 1);
        }
    }
    __syncthreads();

    // Load Q into registers (same as v3)
    // ... q_regs[HPG][4], head_row_max, head_row_sum, head_acc ...

    // ---- Main tile loop with producer/consumer split ----
    if (warp_id < TMA_PRODUCER_WARPS) {
        // PRODUCER: issue TMA loads
        for (int tile = start_tile; tile < end_tile; tile++) {
            int stage = (tile - start_tile) % TMA_STAGES;

            // Issue TMA for K tile (paged)
            if (lane_id == 0) {
                int tile_start = tile * TMA_BC;
                int tile_len = min(TMA_BC, context_len - tile_start);
                uint32_t tx_bytes = tile_len * head_dim * sizeof(__half);

                mbar_arrive_tx(&s_mbar[stage], tx_bytes);

                // Multi-page TMA issue
                int smem_off = 0;
                for (int t = 0; t < tile_len; ) {
                    int kv_pos = tile_start + t;
                    int page_idx = kv_pos / block_size;
                    int page_off = kv_pos % block_size;
                    int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                    int chunk = min(block_size - page_off, tile_len - t);

                    const CUtensorMap* desc = &k_tma_descs[phys_block * num_kv_heads + kv_head_idx];
                    tma_load_2d(
                        &s_kv[stage][smem_off],
                        desc, 0, page_off, &s_mbar[stage]
                    );

                    smem_off += chunk * head_dim;
                    t += chunk;
                }
            }

            // Wait for consumer to finish with previous use of this stage
            // (For stage 0 on first iteration, this is a no-op)
            // ... (mbarrier consumer-done signal) ...
        }
    } else {
        // CONSUMER: compute on arrived data
        for (int tile = start_tile; tile < end_tile; tile++) {
            int stage = (tile - start_tile) % TMA_STAGES;
            int tile_start = tile * TMA_BC;
            int tile_len = min(TMA_BC, context_len - tile_start);

            // Wait for K data
            uint32_t phase = ((tile - start_tile) / TMA_STAGES) & 1;
            mbar_wait_parity(&s_mbar[stage], phase);

            // QKT + softmax (same algorithm as v3, but only consumer warps)
            // ... (7 warps compute) ...

            // Wait for V data (producer loads V after K)
            // ... (second mbarrier or reuse pattern) ...

            // PV accumulation
            // ...
        }
    }

    // ... (output write, same as v3) ...
}
```

### 8.4 Rust-Side Integration

In `crates/rvllm-attention/src/tma.rs`:

```rust
use cudarc::driver::{CudaContext, CudaSlice};

/// Pool of TMA descriptors for paged KV cache.
/// Updated when physical blocks are allocated/freed.
pub struct TmaDescriptorPool {
    /// Device memory holding CUtensorMap array for K cache
    k_descs: CudaSlice<[u8; 128]>,  // CUtensorMap is 128 bytes
    /// Device memory holding CUtensorMap array for V cache
    v_descs: CudaSlice<[u8; 128]>,
    /// Number of physical blocks
    num_blocks: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Block size
    block_size: usize,
    /// Head dimension
    head_dim: usize,
}

impl TmaDescriptorPool {
    pub fn new(
        ctx: &CudaContext,
        k_cache_ptr: u64,
        v_cache_ptr: u64,
        num_blocks: usize,
        num_kv_heads: usize,
        block_size: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let total_descs = num_blocks * num_kv_heads;
        // Allocate host-side descriptor array, fill via cuTensorMapEncodeTiled,
        // then copy to device
        // ...
    }

    /// Update descriptor for a specific physical block (after reallocation)
    pub fn update_block(&mut self, phys_block: usize, new_base_ptr: u64) -> Result<()> {
        // Re-encode descriptors for all kv_heads of this block
        // Copy updated descriptors to device
        // ...
    }
}
```

## 9. Risk Analysis: TMA with Paged KV Cache

### 9.1 Risk: Multiple TMA Issues Per Tile

**Problem**: With block_size=16 and tile_size=64, each tile crosses 4 page boundaries. Each crossing requires a separate TMA load instruction with a different descriptor. This partially negates TMA's advantage of "one instruction per tile."

**Mitigation**:
1. **Increase block_size**: block_size=64 means tiles never cross page boundaries. One TMA issue per tile.
2. **Align tile size to block_size**: Use TMA_BC=16 (matching block_size) for single-issue loads. But 16-position tiles increase overhead from more tile iterations.
3. **Hybrid**: Use block_size=64 for decode (where TMA matters most) and block_size=16 for prefill (where cp.async is fine).

**Recommended**: Increase block_size to 64 for the TMA path. This means 64 * num_kv_heads * head_dim * 2 = 64 * 4 * 128 * 2 = 65,536 bytes per page. Memory fragmentation increases but is acceptable for H100's 80GB.

### 9.2 Risk: Descriptor Array Memory

**Problem**: 10,000 blocks * 4 kv_heads * 128 bytes/desc = 5 MB in GPU memory.

**Mitigation**: 5 MB is negligible vs 80 GB. Not a real concern.

### 9.3 Risk: Descriptor Invalidation on Block Reallocation

**Problem**: When the KV cache evicts blocks and reallocates them to different sequences, the TMA descriptors contain stale base pointers. Using a stale descriptor would cause silent data corruption.

**Mitigation**:
1. Block table updates happen between forward passes (in the scheduler), never during kernel execution.
2. After any block table update, refresh the affected TMA descriptors before the next kernel launch.
3. Use a generation counter: if the descriptor pool's generation doesn't match the block table's generation, force a full refresh.

### 9.4 Risk: Occupancy Regression

**Problem**: Double-buffered smem + mbarriers increases smem from ~18KB to ~35KB per block. With warp specialization (3-stage, separate K/V), it's ~100KB. H100 has 228KB per SM.

| Configuration | smem/block | blocks/SM | Total active blocks (132 SMs) |
|---|---|---|---|
| Current v3 (single-buf) | 18 KB | 2 | 264 |
| TMA double-buf | 35 KB | 2 | 264 |
| TMA warp-spec 3-stage | 100 KB | 2 | 264 |
| TMA warp-spec 3-stage+sep K/V | 100 KB | 2 | 264 |

At 100KB per block, we can still fit 2 blocks/SM (200KB < 228KB). The `__launch_bounds__(256, 2)` directive holds.

If we need 3-stage with separate K and V (150KB), we drop to 1 block/SM = 132 active blocks. This is still enough for full GPU utilization in the memory-bandwidth-bound regime.

### 9.5 Risk: Correctness of Multi-Issue TMA with mbarrier

**Problem**: When a single tile requires 4 TMA issues (4 page crossings), the mbarrier must track the total byte count correctly. If one TMA fails (e.g., OOB), the barrier may never complete.

**Mitigation**:
1. Use `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` in the descriptor -- OOB coordinates return 0, and the TMA still completes the transaction.
2. Set `mbarrier.arrive.expect_tx` with the **total** expected bytes (tile_len * head_dim * sizeof(half)), not per-issue bytes. Each TMA issue contributes its portion to the running byte count.
3. Boundary check: ensure `tile_len * head_dim * sizeof(half)` matches the sum of all individual TMA issue sizes.

### 9.6 Risk: Head Dimension Alignment for Swizzle

**Problem**: TMA 128B swizzle requires the innermost dimension to be a multiple of 128 bytes / sizeof(element). For f16, this means head_dim must be a multiple of 64.

**Supported head dimensions**: head_dim=64 (64*2=128 bytes, matches exactly), head_dim=128 (128*2=256 bytes, OK), head_dim=96 (96*2=192 bytes, NOT a power-of-2 -- swizzle may not work optimally).

**Mitigation**: For head_dim=96, use 64B swizzle instead of 128B. Or pad to 128 in shared memory.

## 10. Summary: Implementation Sequence

| Phase | What | Est. Effort | Risk | Expected Gain |
|---|---|---|---|---|
| **0** | TMA infrastructure (descriptors, Rust FFI, compile flags) | 2 days | Low | 0% (plumbing) |
| **1** | TMA prefill (contiguous tensors, no paging) | 2 days | Low | 5-10% prefill |
| **2** | TMA decode with per-block descriptors (paged KV) | 3 days | Medium | 15-25% decode |
| **3** | Warp specialization (producer/consumer) | 3 days | Medium | +20-35% decode |
| **4** | WGMMA for QKT and PV | 3 days | Medium | +5-10% decode |
| **5** | Block_size=64 for aligned TMA | 1 day | Low | +5% (cleaner) |
| **6** | Cluster multicast for GQA | 2 days | High | +10% GQA |

**Total estimated gain: 45-80% decode throughput improvement**, closing the gap from 0.67x to ~0.90-0.95x of Tri Dao's official FA3.

---

### Critical Files for Implementation

- `/Users/andy/rvllm/kernels/flash_attention_3_v3.cu` -- The current production FA3 v3 decode kernel with cp.async; this is the primary file to study and the template for the TMA rewrite.
- `/Users/andy/rvllm/kernels/tma_gemv_fp16.cu` -- Existing cp.async double-buffer GEMV kernel; demonstrates the project's async copy patterns and naming conventions.
- `/Users/andy/rvllm/kernels/cutlass_gemm.cu` -- CUTLASS 3.x SM90 GEMM with TMA and WGMMA via the collective builder; proves TMA toolchain works in the build system.
- `/Users/andy/rvllm/crates/rvllm-kv-cache/src/cache.rs` -- KV cache layout definition (`[num_blocks, block_size, num_kv_heads, head_dim]`); critical for TMA descriptor stride calculation.
- `/Users/andy/rvllm/crates/rvllm-attention/src/backend.rs` -- Attention backend dispatch; the TMA backend must integrate here as a new `AttentionBackend` implementation for SM >= 9.0.
