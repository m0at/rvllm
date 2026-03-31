# 06: Memory Bandwidth Optimization

## 1. H100 Memory Hierarchy Reference

| Level | Capacity | Bandwidth | Latency | Scope |
|-------|----------|-----------|---------|-------|
| HBM3 | 80 GB | 3.35 TB/s (theoretical), ~2.68 TB/s achievable | ~400 ns | Global |
| L2 Cache | 50 MB | ~12 TB/s | ~100 ns | Shared across all 132 SMs |
| L1/SMEM | 228 KB/SM (configurable split) | ~19 TB/s per SM, ~2.5 PB/s aggregate | ~30 ns | Per-SM |
| Register File | 256 KB/SM (65536 x 32-bit) | Unlimited (operand access) | 0 | Per-SM |

Key H100 (SXM5) numbers for roofline analysis:
- Peak FP16 tensor core: 989.4 TFLOPS (with sparsity: 1978.9 TFLOPS)
- HBM3 bandwidth: 3.35 TB/s theoretical, practical peak ~2.68 TB/s (80% efficiency)
- L2 cache lines: 128 bytes, 32 sectors of 32 bytes each
- Memory transactions: 32 bytes per sector, minimum 128 bytes per cache line
- Warp memory coalescing: 32 threads x 4 bytes = 128 bytes per coalesced transaction

Arithmetic intensity crossover (roofline):
- Compute-bound when FLOP/Byte > 989.4 TFLOPS / 3.35 TB/s = 295 FLOP/Byte
- At M=1 decode, GEMV has 2 FLOP per weight byte (f16), so AI = 2 / 2 = 1.0 FLOP/Byte
- This is 295x below the roofline knee -- decode is profoundly memory-bandwidth bound

## 2. Current Bandwidth Utilization Estimate Per Kernel

### Reference model: Qwen2.5-1.5B (28 layers)
- hidden_size = 1536, num_heads = 12, num_kv_heads = 2, head_dim = 128
- intermediate_size = 8960, vocab_size = 151936
- q_dim = 1536, kv_dim = 256, qkv_dim = 2048, gate_up_dim = 17920

### Per-layer weight bytes (f16):

| Weight Matrix | Shape | Bytes (f16) | Bytes (FP8) |
|--------------|-------|------------|-------------|
| QKV fused | [2048, 1536] | 6,291,456 (6.0 MB) | 3,145,728 (3.0 MB) |
| O projection | [1536, 1536] | 4,718,592 (4.5 MB) | 2,359,296 (2.25 MB) |
| Gate+Up fused | [17920, 1536] | 55,050,240 (52.5 MB) | 27,525,120 (26.25 MB) |
| Down projection | [1536, 8960] | 27,525,120 (26.25 MB) | 13,762,560 (13.125 MB) |
| Norm weights (x2) | [1536] x 2 | 6,144 (6 KB) | - |
| **Layer total** | | **93,591,552 (89.3 MB)** | **46,792,704 (44.6 MB)** |

**Full model weights (28 layers):** 2.49 GB (f16) / 1.25 GB (FP8)
**LM head:** [151936, 1536] x 2 = 466.8 MB (f16)

### Per-kernel bandwidth analysis (M=1 decode, single token):

#### 2a. Fused Add+Norm+QKV GEMV (`fused_cute_add_norm_qkv_gemv`)

**Bytes read:**
- input hidden state: 1536 x 2 = 3,072 B
- add_vec (prev MLP): 1536 x 2 = 3,072 B
- norm_weight: 1536 x 2 = 3,072 B
- QKV weight matrix: 2048 x 1536 x 2 = 6,291,456 B (f16 variant)
- QKV weight matrix: 2048 x 1536 x 1 = 3,145,728 B (FP8 variant) + scales: 2048 x 2 = 4,096 B

**Bytes written:**
- QKV output: 2048 x 2 = 4,096 B
- residual_out: 1536 x 2 = 3,072 B (block 0 only)

**Total traffic (f16):** ~6.30 MB read + ~7 KB write = 6.31 MB
**Total traffic (FP8):** ~3.16 MB read + ~7 KB write = 3.16 MB

**Achieved bandwidth estimate:**
- Grid: (2048+7)/8 = 256 blocks, 256 threads each
- The GEMV phase uses warp-per-row (8 warps = 8 rows per block)
- Each warp reads one full weight row: 1536 x 2 = 3,072 bytes via 128-bit loads (int4)
- With 32 lanes, each lane reads 3072/32 = 96 bytes = 6 int4 loads
- Weight reads dominate: 6.0 MB weight / total 6.3 MB traffic = 95%
- **Estimated achieved BW: ~55-65% of HBM peak** (1.8-2.2 TB/s)
- Bottleneck: The RMSNorm phase requires a full __syncthreads barrier between the sum-of-squares reduction and the normalization pass. This serialization is unavoidable but adds ~2-3 us of dead time.

**Opportunities:**
1. The f16 variant uses 128-bit loads (int4) for weight reads -- good
2. The FP8 variant uses 32-bit loads (uint) for 4 bytes at a time -- should upgrade to 128-bit (16 bytes = 16 FP8 values)
3. Input vectors (3 KB each) will be cached in L1 after first access, so redundant reads across blocks are nearly free
4. All 256 blocks redundantly compute the norm phase: 6 KB x 2 reads = negligible since it fits in L2

#### 2b. Fused RoPE + Cache Write (`fused_rope_cache_f16_kernel`)

**Bytes read/written:**
- Q vector: 1536 x 2 = 3,072 B (read + write in-place)
- K vector: 256 x 2 = 512 B (read + write in-place + write to cache)
- V vector: 256 x 2 = 512 B (read + write to cache)
- cos/sin tables: 128 x 4 = 512 B each (read; cached in L2 across tokens)
- positions, slot_mapping: 8 B total

**Total traffic:** ~8.7 KB
**Achieved BW:** Irrelevant -- this kernel is latency-bound, not bandwidth-bound.
Grid is (1, max(12,2), 1) = 12 blocks x 64 threads = 768 threads total.
The entire kernel likely completes in < 1 us. Not worth optimizing for bandwidth.

#### 2c. FlashAttention-3 GQA Decode (`flash_attention_3_decode_gqa_f16io_kernel`)

**Bytes read (for context length C):**
- Query: num_kv_heads x head_dim x 2 per group = 12 x 128 x 2 = 3,072 B
- K cache: C x num_kv_heads x head_dim x 2 = C x 512 B
- V cache: C x num_kv_heads x head_dim x 2 = C x 512 B
- Block tables: (C/block_size) x 4 B
- context_lens: 4 B

For C=512: K+V = 512 x 1024 = 512 KB
For C=2048: K+V = 2048 x 1024 = 2 MB
For C=8192: K+V = 8192 x 1024 = 8 MB

**Bytes written:**
- Output: num_heads x head_dim x 2 = 12 x 128 x 2 = 3,072 B

**Achieved bandwidth estimate:**
- Grid: (num_seqs=1, num_kv_heads=2) = 2 blocks (GQA mode)
- Only 2 blocks active on 132 SMs -- SM utilization = 1.5%
- Each block processes all 6 query heads sharing one KV head
- KV tile size: 64 positions x 128 dims x 2 bytes = 16 KB per tile
- Tiles loaded via half2 coalesced reads, then processed in smem
- **Estimated achieved BW: ~15-25% of peak** at C=512 (0.5-0.8 TB/s)
- **Root cause:** Only 2 thread blocks total for 132 SMs. Massive SM underutilization.

**Opportunities:**
1. Split-KV parallelism (already implemented in `flash_attention_3_v3.cu`): distribute tiles across multiple blocks per KV head. At C=2048, 4 splits = 8 blocks = better SM coverage
2. Page table indirection: each KV position requires `page_idx = kv_pos / block_size` then `block_tables[seq * max_blocks + page_idx]`, which is an integer divide + indirect load. The block table load is scattered (different physical blocks), causing L2 misses. With block_size=16, the indirection overhead is ~6% of KV load time.
3. cp.async in v3 kernel overlaps next K tile load with current P@V computation -- already implemented but not enabled as default path.

#### 2d. Fused Oproj+Add+Norm+GateUp GEMV (`fused_cute_oproj_add_norm_gateup_gemv`)

**Bytes read:**
- attn_out: q_dim x 2 = 3,072 B
- o_weight: hidden x q_dim x 2 = 4,718,592 B (4.5 MB)
- residual: hidden x 2 = 3,072 B
- norm_weight: hidden x 2 = 3,072 B
- gateup_weight: gate_up_dim x hidden x 2 = 55,050,240 B (52.5 MB)

**Total weight traffic:** 57.0 MB

**O-proj phase:** Every block (2240 total) redundantly computes all hidden_size=1536 output elements of O-proj. This means o_weight is read once globally and served from L2 (4.5 MB fits in 50 MB L2). The attn_out vector (3 KB) is read once per block but cached in L1. Cost: 4.5 MB from HBM (first block) + ~3 KB x 2239 from L2 = effectively 4.5 MB HBM.

**GateUp GEMV phase:** Each block reads 8 rows of gateup_weight. With 2240 blocks reading 8 rows each, the full 52.5 MB is streamed once. Input is in smem. Weight reads are 128-bit vectorized (half2).

**Estimated achieved BW: ~55-65% of peak** (same as QKV GEMV phase)

**Critical insight:** The O-proj phase is computed sequentially (one output element at a time, all-threads-reduce). For hidden_size=1536, this is 1536 serial reduction steps, each with a __syncthreads. This is extremely inefficient. The O-proj phase dominates latency despite small data size because of serialization. This kernel trades compute efficiency for fusion benefit.

#### 2e. Fused SiLU + Down GEMV (`fused_cute_silu_down_gemv`)

**Bytes read:**
- gate vector: intermediate x 2 = 17,920 B
- up vector: intermediate x 2 = 17,920 B
- down_weight: hidden x intermediate x 2 = 27,525,120 B (26.25 MB)

**Total traffic:** 26.3 MB

**Achieved BW:** ~55-65% of peak. Same analysis as other GEMV kernels. 8 warps per block (RPB=8), warp-per-row, half2 vectorized loads.

**Note:** Gate and up vectors (18 KB each) fit comfortably in L1 and are reused across all 8 rows in a block.

#### 2f. Embedding Gather (`embedding_gather_f16_kernel`)

**Bytes read:** token_id lookup + 1536 x 2 = 3,072 B per token
**At M=1:** 3 KB total. Pure latency-bound. Not bandwidth relevant.

#### 2g. Fused LM Head + Argmax (`fused_lm_head_argmax_f16_kernel`)

**Bytes read:**
- hidden_state: 1536 x 4 = 6,144 B (f32, loaded into smem)
- lm_head weight: 151936 x 1536 x 2 = 466,828,288 B (445.3 MB)

**Total traffic:** 445.3 MB -- this is the SINGLE LARGEST bandwidth consumer per step.

**Achieved BW:** Each block has blockDim.x threads, one row per thread. Each row is a full dot product over 1536 elements via half2 vectorized loads. With 151936 rows and typical 256-thread blocks, grid = 593 blocks. Each block processes 256 vocab rows.

**Estimated achieved BW: ~50-60% of peak.** The hidden state is shared via smem, but each thread independently streams its weight row. Sequential dot product per row means no cross-thread cooperation for the dot product itself -- each thread reads 1536 x 2 = 3 KB for its row.

**Opportunity:** For greedy decode, we need only argmax. The current kernel fuses this (no logits materialization). But the weight read is unavoidable. FP8 LM head weights would save 222.6 MB -- a massive win. Weight sharing with embed_tokens (tied weights) is already implemented.

### 2h. Total bytes per token per decode step (M=1, 28 layers, Qwen2.5-1.5B f16)

| Component | Bytes (f16) | Bytes (FP8) | % of Total |
|-----------|-------------|-------------|------------|
| QKV weight (x28) | 176.2 MB | 88.1 MB | 5.9% |
| O-proj weight (x28) | 126.0 MB | 63.0 MB | 4.2% |
| GateUp weight (x28) | 1,470.0 MB | 735.0 MB | 49.5% |
| Down weight (x28) | 735.0 MB | 367.5 MB | 24.7% |
| LM head weight | 445.3 MB | 222.6 MB | 15.0% |
| KV cache reads (C=512) | 14.3 MB | 7.2 MB | 0.5% |
| Activations/norms | ~3.5 MB | ~3.5 MB | 0.1% |
| **Total** | **2,970 MB** | **1,487 MB** | **100%** |

**Time at peak HBM BW (2.68 TB/s achievable):**
- f16: 2,970 MB / 2,680 MB/ms = 1.11 ms
- FP8: 1,487 MB / 2,680 MB/ms = 0.55 ms

**Observed decode latency (A100, N=1):** ~9.9 ms/token = ~7-10% bandwidth utilization
**Observed decode latency (A100, N=1, fused path):** ~5-6 ms extrapolated

The gap is due to: (a) kernel launch overhead (5-6 kernel launches per layer), (b) __syncthreads barriers within fused kernels, (c) L2/L1 miss penalties on weight streaming, (d) serial O-proj computation in mega-fused kernel, (e) attention SM underutilization.

## 3. Weight Layout Optimization: Row-Major vs Column-Major

### Current layout:
All weight matrices are stored **row-major** `[out_dim, in_dim]`:
- GEMV reads one complete row per output element: `output[n] = sum_k(weight[n,k] * input[k])`
- Row-major is optimal for GEMV because threads within a warp read consecutive elements along the K dimension, yielding perfectly coalesced 128-byte transactions.

### Analysis per GEMM shape:

| GEMM | M | N | K | Layout | Optimal? |
|------|---|---|---|--------|----------|
| QKV | 1 | 2048 | 1536 | Row-major B[N,K] | YES for GEMV, YES for cuBLAS (NN layout) |
| O-proj | 1 | 1536 | 1536 | Row-major B[N,K] | YES |
| GateUp | 1 | 17920 | 1536 | Row-major B[N,K] | YES |
| Down | 1 | 1536 | 8960 | Row-major B[N,K] | YES |
| LM head | 1 | 151936 | 1536 | Row-major B[N,K] | YES |
| Prefill GEMM | T>1 | varies | varies | Row-major A[M,K], B[N,K] | Need transpose for cuBLAS col-major convention |

**Conclusion:** Row-major is correct for decode GEMV. No layout change needed.

**For the persistent GEMM kernel (`persistent_gemm.cu`):** A is [M,K] row-major, B is [N,K] row-major. The inner K-tile loads use `smem_a[row * BLOCK_K + col]` and `smem_b[row * BLOCK_K + col]`. For wmma, A needs row-major and B needs col-major loading. The current code loads B as col-major by using `BLOCK_K` as the leading dimension and `wmma::col_major` tag. This works because B[N,K] row-major with leading dim K is equivalent to B_T[K,N] col-major with leading dim K. **Layout is correct.**

## 4. Vectorized Loads: Current Usage and Opportunities

### Current 128-bit (int4) load usage:

| Kernel | Load Width | Status | Detail |
|--------|-----------|--------|--------|
| `fused_cute_add_norm_qkv_gemv` (f16) | **128-bit (int4)** | GOOD | `int4 packed = w4[i]` decodes 8 half values |
| `fused_cute_add_norm_qkv_gemv` (FP8) | 32-bit (uint) | **SUBOPTIMAL** | `unsigned int packed` = 4 FP8 bytes; should use int4 = 16 FP8 bytes |
| `gemv_f16_kernel` | 32-bit (half2) | **SUBOPTIMAL** | Only 2 half per load; should use int4 = 8 half |
| `fused_cute_silu_down_gemv` (f16) | 32-bit (half2) | **SUBOPTIMAL** | Same issue |
| `fused_cute_silu_down_gemv` (FP8) | 32-bit (uint) | **SUBOPTIMAL** | 4 FP8 bytes per load |
| `fused_cute_oproj_add_norm_gateup_gemv` (f16) | 32-bit (half2) | **SUBOPTIMAL** | O-proj uses half2 for weight reads |
| `fused_cute_oproj_add_norm_gateup_gemv` (FP8) | 32-bit (uint) | MODERATE | Phase 3 uses uint for 4 bytes |
| `fused_cute_add_norm_gateup_gemv` | 32-bit (half2) | **SUBOPTIMAL** | Weight reads use half2 |
| `tma_gemv_fp16_kernel` | 128-bit (cp.async) | GOOD | 16-byte cp.async copies |
| `flash_attention_3` KV loads | 32-bit (half2) | MODERATE | KV tile loads use half2; could use int4 |
| `fused_lm_head_argmax_f16_kernel` | 32-bit (half2) | **SUBOPTIMAL** | Each thread reads full row with half2 |
| `embedding_gather_f16_kernel` | 16-bit (half) | **SUBOPTIMAL** | Scalar half loads; should use int4 |
| `fused_residual_rmsnorm_f16_kernel` | 16-bit (half) | **SUBOPTIMAL** | Scalar half loads for input/add/weight |

### Quantified impact of upgrading to 128-bit loads:

At 32-bit loads, each warp issues 4x more load instructions than necessary. The instruction throughput is not the bottleneck (HBM latency is), but wider loads improve:
1. **L2 sector utilization:** A 32-bit load from a 32-byte sector wastes 28 bytes of fetched data if adjacent threads don't access the same sector. With 128-bit loads, each thread touches 16 bytes = half a sector, so 2 adjacent threads fill one sector perfectly.
2. **Instruction count reduction:** 4x fewer load instructions means the scheduler can issue more compute instructions between loads, improving latency hiding.
3. **Expected improvement:** 5-10% bandwidth utilization improvement per kernel that currently uses half2.

### Priority upgrade list:
1. `fused_cute_silu_down_gemv` (f16 variant) -- reads 26.25 MB of down_weight with half2. Upgrade to int4 saves ~2 us per layer.
2. `fused_cute_add_norm_gateup_gemv` -- reads 52.5 MB of gateup_weight with half2. Largest single weight read. Upgrade to int4 saves ~5 us per layer.
3. `fused_cute_oproj_add_norm_gateup_gemv` (f16 variant) -- both O-proj and gateup phases use half2.
4. `fused_lm_head_argmax_f16_kernel` -- reads 445 MB with half2. Upgrade saves ~20 us total.
5. All FP8 kernels -- upgrade from uint (32-bit = 4 bytes) to uint4 (128-bit = 16 bytes).

## 5. L2 Cache Residency Analysis

### H100 L2 Cache: 50 MB

| Tensor | Size | Fits in L2? | Access Pattern |
|--------|------|-------------|----------------|
| Input activation (hidden) | 3 KB | YES (100x over) | Read by all blocks, cached after first |
| Norm weights | 3 KB each | YES | Read by all blocks |
| RoPE cos/sin tables | 512 B per token | YES | Read once per token |
| O-proj weight | 4.5 MB | YES | Redundantly read by all gateup blocks |
| QKV weight | 6.0 MB | YES | Streamed once; next-layer QKV evicts |
| Down weight | 26.25 MB | YES (52.5%) | Streamed once, partial L2 residence |
| GateUp weight | 52.5 MB | NO (105% of L2) | Streams through L2; causes eviction of everything |
| LM head weight | 445 MB | NO (890%) | Completely streaming |
| KV cache (C=512) | 0.5 MB per layer | YES (1%) | Random access via page tables |
| KV cache (C=8192) | 8 MB per layer | YES (16%) | Fills significant L2 fraction |

### L2 Cache Partitioning Strategy

**Problem:** GateUp weight (52.5 MB) exceeds L2 capacity and evicts everything. When the down projection follows, its weight reads all miss L2.

**Strategy 1: CUDA L2 Access Policy Windows (cudaAccessPolicyWindow)**
Pin the next layer's QKV weight in L2 while the current layer's MLP runs. When the current layer finishes and the next layer's norm+QKV GEMV starts, QKV weight is already warm in L2.

```
Layer N: [Norm+QKV] -> [Attn] -> [Oproj+Norm+GateUp] -> [SiLU+Down]
                                                          ^-- prefetch Layer N+1 QKV weight into L2
Layer N+1: [Norm+QKV] -> ...
                ^-- QKV weight is L2-warm, saving ~6 MB of HBM reads
```

**Expected impact:** Save 6 MB x 28 layers = 168 MB of HBM reads per step. At 2.68 TB/s, this saves ~63 us. Modest but free.

**Strategy 2: Reorder weight reads within MLP**
Currently: GateUp (52.5 MB) then Down (26.25 MB). The GateUp read evicts Down from L2.

Alternative: Interleave GateUp and Down computation. Split GateUp into chunks that fit in L2 alongside Down weight. This requires rearchitecting the SiLU+Down kernel to operate on partial intermediate results.

**Strategy 3: Persistent kernels for L2 weight reuse across batch elements**
At batch size B>1, the same weight rows are read B times. With persistent kernels, one thread block can process multiple batch elements before moving to the next weight tile, keeping the tile in L2.

Current persistent GEMM kernel (`persistent_gemm.cu`) already implements GROUP_M=8 swizzled tile ordering for L2 locality. But it's only used for M>1 (prefill). For M=1 decode with batching (M=B), this provides:
- At B=8: each 128x128 weight tile is read once and used for 8 batch elements
- L2 amplification: 8x reduction in effective HBM weight reads
- **Expected improvement at B=8:** 2-3x throughput gain for GEMV

## 6. Persistent Kernels for Cross-Batch Weight Reuse

### Current state:
The existing `persistent_gemm.cu` implements a persistent grid with swizzled tile ordering for prefill (M>1). For decode (M=1), the code uses separate GEMV kernels with no cross-batch reuse.

### Opportunity: Batched decode with persistent weight tiles

For batch size B (multiple concurrent sequences decoded together), the weight matrix is the same across all B tokens. A persistent kernel can:

1. Load a weight tile [TILE_N, TILE_K] into shared memory
2. Process all B input vectors against this tile
3. Move to the next tile

**Arithmetic intensity improvement:**
- Without reuse: 2B FLOP / (TILE_N x TILE_K x 2) bytes = 2B / (TILE_N x TILE_K x 2) FLOP/Byte
- Amortized weight load: each byte of weight produces B * 2 FLOPs instead of 2 FLOPs
- At B=8, AI increases from 1.0 to 8.0 FLOP/Byte -- still below roofline but significantly better

**Implementation sketch:**
```
Grid: (NUM_SMS=132, 1, 1)
for tile in assigned_tiles:
    load weight_tile[TILE_N, TILE_K] into smem
    for b in 0..B:
        dot_product(input[b], weight_tile) -> partial_acc[b]
    reduce and write output[b, tile_n_range]
```

The `tma_gemv_fp16.cu` kernel already loads the input vector into shared memory and processes 8 rows per block. Extending this to process multiple batch elements per weight tile is straightforward.

## 7. Software Prefetching with cp.async.bulk

### Current cp.async usage:

| Kernel | cp.async? | Stage | Detail |
|--------|-----------|-------|--------|
| `tma_gemv_fp16.cu` | YES | Double-buffered | Weight tiles prefetched via `cp.async.cg.shared.global` 16-byte chunks |
| `flash_attention_3_v3.cu` | YES | Single-buffered | KV tiles loaded via `cp.async.cg.shared.global` |
| `persistent_gemm.cu` | Mentioned but not implemented | - | K-tile pipelining described but uses synchronous loads |
| All fused GEMV kernels | NO | - | Direct global loads, no prefetching |

### Opportunity: Add cp.async to fused GEMV kernels

The fused add+norm+GEMV kernels currently read weight rows directly from global memory. Since the norm phase takes significant time (reduction + barrier), we could prefetch the first weight tile during the norm computation:

```
Phase 1: Residual add + sum-of-squares reduction
  [prefetch: first 8 weight rows -> smem buffer A]
Phase 2: RMSNorm scale computation (barrier)
  [cp.async completes during barrier]
Phase 3: Norm weight application
Phase 4: GEMV with weight from buffer A
  [prefetch: next 8 weight rows -> smem buffer B]
Phase 5: GEMV with weight from buffer B
  [prefetch: next 8 rows -> buffer A]
  ... (double-buffered pipeline)
```

**Challenge for GEMV:** Each warp processes one row independently. The weight data per row is 1536 x 2 = 3 KB. With 8 warps, a double buffer needs 2 x 8 x 3 KB = 48 KB of shared memory, which fits in the 228 KB per-SM budget but competes with the `s_normed` array (1536 x 4 = 6 KB).

**Expected improvement:** The current kernels achieve ~60% BW because global loads stall the pipeline. cp.async hides global memory latency by overlapping loads with computation. Expected: 5-15% improvement in GEMV phase, or 3-10% overall kernel speedup.

### cp.async.bulk (TMA descriptor-based, sm_90+)

H100 supports TMA (Tensor Memory Accelerator) with `cp.async.bulk` which can load entire 2D tiles from global to shared memory with a single instruction, no per-thread address computation. This is strictly better than the per-thread `cp.async.cg` used in current kernels.

**Not yet used anywhere in rvLLM.** The CUTLASS GEMM kernel (`cutlass_gemm.cu`) uses TMA via CUTLASS abstractions, but custom kernels do not.

## 8. KV Cache Access Pattern Analysis

### Cache layout:
```
key_cache:   [num_blocks, block_size, num_kv_heads, head_dim] in f16
value_cache: [num_blocks, block_size, num_kv_heads, head_dim] in f16
```

### Page table indirection:
```
For KV position `pos`:
  page_idx = pos / block_size           // integer divide
  page_off = pos % block_size           // modulo
  phys_block = block_tables[seq_idx * max_blocks + page_idx]  // indirect load
  base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d
```

### Coalescing analysis:

**Within a tile:** For a single KV position, threads 0..head_dim read consecutive addresses:
```
key_cache[base + 0], key_cache[base + 1], ..., key_cache[base + head_dim-1]
```
With head_dim=128 and f16, this is 256 bytes = 2 cache lines. Perfectly coalesced.

**Across positions in a tile:** Different positions map to different physical blocks (potentially non-contiguous). Two consecutive positions might be:
```
pos=0: phys_block=5, offset within block=0
pos=1: phys_block=5, offset within block=1  // same physical block, contiguous
...
pos=15: phys_block=5, offset within block=15
pos=16: phys_block=12, offset within block=0  // DIFFERENT physical block, non-contiguous
```

Within a page (block_size positions), accesses are contiguous. Across pages, there is a jump to a different physical block. The stride between page boundaries is `block_size * num_kv_heads * head_dim * 2` bytes = `16 * 2 * 128 * 2` = 8,192 bytes for Qwen2.5-1.5B with block_size=16.

**Page boundary penalty:** At each page boundary, L2 must fetch a new cache line set from a different HBM region. With 50 MB L2 and typical KV cache sizes, most pages fit in L2 for moderate context lengths (C <= ~50,000 tokens x 2 heads x 128 dims x 2 bytes = 25 MB per layer for K or V). For C=512, total KV = 0.5 MB, trivially fits in L2.

**Integer division overhead:** `pos / block_size` and `pos % block_size` require integer division. On H100, integer divide is 30+ cycles. For 64 positions per tile with 256 threads, each thread computes ~0.25 divides. Negligible compared to HBM latency.

### Optimization: Contiguous KV layout within attention tiles

Current: Each position requires independent page table lookup.
Alternative: Precompute physical addresses for the tile and load via cp.async with precomputed pointers.
Expected improvement: Minimal (< 2%) since the page table is small and cached in L1.

### FP8 KV cache:

`fp8_kv.cu` implements FP8 E4M3 quantization with per-head scale factors. This halves KV cache bandwidth:
- C=512: 512 KB -> 256 KB (+ 4 KB scales)
- C=8192: 8 MB -> 4 MB (+ 64 KB scales)

At long contexts where KV cache dominates, FP8 KV provides up to 2x attention speedup.

## 9. Activation Recomputation vs Storage Tradeoff

### Current activation memory per token per layer:

| Activation | Shape | Bytes | Lifetime |
|-----------|-------|-------|----------|
| normed hidden | [hidden] | 3,072 | Phases 1-2 |
| QKV output | [qkv_dim] | 4,096 | Phases 2-4 |
| Attention output | [q_dim] | 3,072 | Phases 4-5 |
| O-proj output | [hidden] | 3,072 | Phases 5-6 |
| Residual | [hidden] | 3,072 | Phases 1-end |
| normed2 | [hidden] | 3,072 | Phases 6-7 |
| Gate+Up output | [gate_up_dim] | 35,840 | Phases 7-8 |
| SiLU output | [intermediate] | 17,920 | Phase 8 |
| Down output | [hidden] | 3,072 | Phase 8-end |
| **Peak concurrent** | | **~50 KB** | |

**At M=1 decode, activations are negligible** (50 KB vs 89 MB weights per layer).

### Recomputation tradeoffs:

For M=1 decode, activation recomputation makes zero sense -- the activations are tiny compared to weight bandwidth. Every additional byte of weight NOT read saves far more time than any activation recomputation could.

For prefill (M=T tokens), the tradeoff changes:
- Activations scale with T: T x hidden x 2 per activation
- At T=2048: normed = 6 MB, QKV = 8 MB, gate_up = 70 MB
- Weight reads are T-independent: same 89 MB per layer
- Recomputing normed (3 KB activation) to avoid storing it costs one extra hidden_size reduction -- acceptable
- Recomputing QKV to avoid storing it costs re-reading 6 MB of QKV weight -- NOT acceptable

**Recommendation:** Fused kernels already minimize activation storage by keeping intermediates in shared memory (smem). The current approach of fusing norm+GEMV is optimal for decode: the normed hidden state lives entirely in smem and never touches HBM.

## 10. Weight Compression: Structured Sparsity (2:4)

### H100 2:4 structured sparsity support:

The H100 tensor cores natively support 2:4 structured sparsity via sparse tensor core instructions. In each group of 4 consecutive elements, exactly 2 must be zero. The hardware:
1. Stores only the 2 non-zero values (50% compression) plus a 2-bit metadata index
2. Performs the sparse MMA in one instruction (same throughput as dense)

**Effective bandwidth reduction: 2x for weight reads.**

### Impact on rvLLM decode:

| Component | Dense (f16) | 2:4 Sparse (f16) | Savings |
|-----------|-------------|-------------------|---------|
| QKV weight (28 layers) | 176 MB | 88 MB + 5.5 MB meta | 47% |
| O-proj weight (28 layers) | 126 MB | 63 MB + 3.9 MB meta | 47% |
| GateUp weight (28 layers) | 1,470 MB | 735 MB + 46 MB meta | 47% |
| Down weight (28 layers) | 735 MB | 367.5 MB + 23 MB meta | 47% |
| LM head | 445 MB | 222.5 MB + 14 MB meta | 47% |
| **Total weight traffic** | **2,952 MB** | **1,476 MB + 92 MB** | **47%** |

**Combined with FP8:** FP8 + 2:4 sparsity = 4x total weight bandwidth reduction.
- Dense f16: 2,952 MB
- FP8 + 2:4: 738 MB + 46 MB meta = 784 MB
- Time at 2.68 TB/s: 2,952 / 2,680 = 1.10 ms -> 784 / 2,680 = 0.29 ms

**Challenges:**
1. Model quality: 2:4 pruning requires fine-tuning to maintain accuracy. rvLLM does not currently perform pruning.
2. GEMV kernels: The current custom GEMV kernels do not support sparse tensor core instructions. Would need new kernels using `mma.sp` instructions.
3. cuBLAS/CUTLASS support: CUTLASS 3.x supports structured sparsity via `SparseTensorOp`. cuSPARSELt provides drop-in sparse GEMM.

**Recommendation:** Implement as a future optimization after FP8 weights are fully deployed. Requires upstream model providers to release 2:4-pruned checkpoints, or rvLLM to implement its own pruning pipeline.

## 11. Multi-Level Tiling Strategy

### Decode GEMV (M=1) tiling hierarchy:

Current approach: one weight row per warp, streamed from HBM through L2 -> L1 -> registers.

**Optimal tiling for L2 -> SMEM -> Registers:**

**Level 1: L2 tile (GROUP_M rows):**
Process GROUP_M consecutive weight rows on the same SM. Since the input vector is in SMEM, the only HBM traffic is weight rows. GROUP_M rows share L2 locality if they are within the same 128-byte-aligned memory region.

Current RPB=8 (rows per block) already achieves this implicitly: 8 consecutive weight rows occupy 8 x 1536 x 2 = 24 KB, which spans ~188 L2 cache lines. These lines will be L2-warm due to spatial locality of consecutive row reads.

**Level 2: SMEM tile (TILE_K columns):**
Process the K dimension in tiles of TILE_K. Load TILE_K columns of 8 weight rows into SMEM, multiply against corresponding TILE_K elements of the input vector (also in SMEM).

The `tma_gemv_fp16.cu` kernel implements this with TILE_K=256, double-buffered. Each tile:
- Weight tile: 8 x 256 x 2 = 4 KB per warp
- Input tile: 256 x 2 = 512 B (shared across warps)
- Total SMEM per buffer: 8 x 4 KB + 512 B = ~33 KB
- Double buffer: ~66 KB, fits in 228 KB per SM

**Level 3: Register tile (per-lane accumulation):**
Each lane accumulates its partial dot product across TILE_K elements. With half2 loads, each lane processes 2 elements per iteration. With int4 (128-bit) loads, each lane processes 8 elements per iteration.

**Current gap:** The fused GEMV kernels (fused_cute_add_norm_qkv_gemv, etc.) do NOT tile the K dimension. Each warp reads the entire K dimension in one pass. For K=1536, this means 1536 x 2 / 32 lanes = 96 bytes per lane per row, with no SMEM buffering of weights.

**Improvement:** Adding K-tiling to fused kernels would enable:
1. Double-buffered weight loads via cp.async (hide HBM latency)
2. Better register reuse across the K tile
3. Reduced instruction pressure (fewer outstanding global loads)

**Expected improvement: 10-20% for GEMV phase** from K-tiling with cp.async overlap.

## 12. Memory Access Coalescing Audit

### Per-kernel coalescing status:

| Kernel | Read Coalescing | Write Coalescing | Issues |
|--------|----------------|------------------|--------|
| `fused_cute_add_norm_qkv_gemv` (f16) | PERFECT | PERFECT | 128-bit loads, lane-stride = 128 bits |
| `fused_cute_add_norm_qkv_gemv` (FP8) | GOOD (32-bit) | PERFECT | Could upgrade to 128-bit |
| `gemv_f16_kernel` | GOOD (32-bit half2) | PERFECT | Lane-stride half2, coalesced |
| `fused_cute_silu_down_gemv` (f16) | GOOD (32-bit half2) | PERFECT | Lane-stride half2 |
| `fused_cute_oproj_add_norm_gateup_gemv` | GOOD (32-bit) | PERFECT | O-proj sequential issue noted |
| `flash_attention_3` KV loads | GOOD (half2) | N/A (smem) | `fa3_load_kv_tile` uses half2 with element-based indexing |
| `paged_attention_v2_f16kv_kernel` | **POOR** | PERFECT | Thread `dim_idx` reads `key_cache[...+dim_idx]` -- coalesced BUT one token at a time (serial inner loop) |
| `fused_residual_rmsnorm_f16_kernel` | **POOR** | **POOR** | Scalar half loads/stores, not vectorized |
| `embedding_gather_f16_kernel` | **POOR** | **POOR** | Scalar half loads, no vectorization |
| `reshape_and_cache_kernel` | MODERATE | MODERATE | Scalar loads but threads are consecutive |
| `fused_rope_cache_f16_kernel` | MODERATE | MODERATE | Thread per pair (stride=2), coalesced within pair |
| `fused_lm_head_argmax_f16` | GOOD (half2) | N/A | Each thread reads own row; adjacent threads read adjacent rows -- partially coalesced at row start |

### Critical finding: `paged_attention_v2_f16kv_kernel`

```c
for (int t = 0; t < tokens_in_block; t++) {
    const int k_offset = ((physical_block * block_size + t) * num_heads + head_idx) * head_dim + dim_idx;
    float k_val = (dim_idx < head_dim) ? __half2float(key_cache[k_offset]) : 0.0f;
```

Each thread reads one element for one token, then all threads __syncthreads, then the next token. This is an **N-pass sequential scan** over tokens. Thread 0 reads `key_cache[base + 0]`, thread 1 reads `key_cache[base + 1]`, etc. This IS coalesced within a token (adjacent threads read adjacent elements). But processing tokens one-at-a-time means the full tile bandwidth is:
- head_dim x 2 bytes per token x block_size tokens = 128 x 2 x 16 = 4 KB per block
- With 1 thread block per (seq, head), only 1 block issues these loads
- Bandwidth: ~4 KB / ~0.5 us = ~8 GB/s = 0.24% of peak

**This kernel is 400x below peak bandwidth.** However, it is already replaced by `flash_attention_3` in the decode path, which tiles 64 positions at once and uses warp-parallel dot products.

### Critical finding: `fused_residual_rmsnorm_f16_kernel`

```c
for (int i = tid; i < hidden_size; i += stride) {
    float val = __half2float(input[row_offset + i]) + __half2float(add[row_offset + i]);
```

Scalar half loads. Each thread reads one f16 element (2 bytes). With 1024 threads, a warp of 32 threads reads 32 x 2 = 64 bytes per transaction, wasting half of a 128-byte cache line. At hidden_size=1536, thread 0 reads element 0, thread 1 reads element 1, etc. -- this IS coalesced within a warp (adjacent threads, adjacent addresses). But 2-byte loads mean only 64 bytes per 128-byte transaction, 50% L2 sector utilization.

**Fix:** Use half2 or int4 vectorized loads:
```c
const int h2 = hidden_size / 2;
const half2* in2 = (const half2*)input;
// or better:
const int h8 = hidden_size / 8;
const int4* in4 = (const int4*)input;
```

**The f16 variant of fused_residual_rmsnorm already does this with half2. But the general rmsnorm_f16_kernel does not.**

## 13. Quantitative Roofline Model

### M=1 Decode, Qwen2.5-1.5B, f16 weights, C=512

**Total bytes per token:**
- Weight reads: 2,952 MB
- KV cache reads: 14.3 MB (K+V, 28 layers, C=512)
- Activation reads/writes: ~3.5 MB
- Metadata (block tables, positions, etc.): < 0.1 MB
- **Total: 2,970 MB**

**Total FLOPs per token:**
- GEMV: 2 x weight_elements = 2 x 1,476 M = 2,952 MFLOP = 2.95 GFLOP
- Attention QK dot: 28 x 2 x 512 x 128 x 12 = 44 MFLOP
- Attention PV: 28 x 2 x 512 x 128 x 12 = 44 MFLOP
- Norm/activation: negligible
- **Total: ~3.04 GFLOP**

**Arithmetic intensity: 3.04 GFLOP / 2.97 GB = 1.02 FLOP/Byte**
**Roofline knee: 295 FLOP/Byte**
**Conclusion: 289x below compute saturation. PURE memory bandwidth bound.**

**Theoretical minimum latency:**
- At 3.35 TB/s: 2,970 MB / 3,350 = 0.886 ms
- At 2.68 TB/s (80%): 2,970 MB / 2,680 = 1.108 ms

### M=1 Decode, FP8 weights:

**Total bytes:** 1,487 MB + 14.3 MB KV + 3.5 MB act = 1,505 MB
**Theoretical minimum:** 1,505 / 2,680 = 0.562 ms
**AI:** 3.04 GFLOP / 1.505 GB = 2.02 FLOP/Byte (still 146x below roofline)

### M=1 Decode, FP8 weights + 2:4 sparsity:

**Total bytes:** ~784 MB
**Theoretical minimum:** 784 / 2,680 = 0.293 ms
**AI:** 3.04 GFLOP / 0.784 GB = 3.88 FLOP/Byte (still 76x below roofline)

### Batch decode (M=B tokens):

Weight reads are constant (B-independent). KV reads scale with B. Activations scale with B.

| Batch | Weights | KV (C=512) | Act | Total | AI | Min Latency | Tok/s |
|-------|---------|------------|-----|-------|----|-------------|-------|
| B=1 | 2,952 MB | 14 MB | 4 MB | 2,970 MB | 1.0 | 1.11 ms | 901 |
| B=8 | 2,952 MB | 115 MB | 28 MB | 3,095 MB | 7.9 | 1.15 ms | 6,929 |
| B=32 | 2,952 MB | 459 MB | 113 MB | 3,524 MB | 27.5 | 1.31 ms | 24,352 |
| B=128 | 2,952 MB | 1,835 MB | 451 MB | 5,238 MB | 74.3 | 1.95 ms | 65,508 |
| B=256 | 2,952 MB | 3,670 MB | 903 MB | 7,525 MB | 103 | 2.81 ms | 91,141 |

At B=256, KV cache bandwidth exceeds weight bandwidth. The crossover point where KV traffic equals weight traffic is at B x C x 1,024 = 2,952 MB -> B = 2,952,000 / (512 x 1,024) = 5.6. So at C=512, even B=6 makes KV bandwidth comparable to weight bandwidth.

## 14. Expected Improvement Summary

| Optimization | Applies To | Estimated Gain | Difficulty | Dependencies |
|-------------|-----------|---------------|------------|-------------|
| **Upgrade half2 -> int4 loads in all fused GEMV kernels** | silu_down, gateup, oproj+gateup, lm_head | 5-10% decode latency | LOW | None |
| **FP8 weights for all projections** | All GEMV | 40-50% decode latency | MEDIUM | Partially implemented (fused_add_norm_fp8_gemv exists) |
| **K-tiling + cp.async in fused GEMV** | QKV, GateUp, Down GEMV | 10-20% GEMV phase | MEDIUM | cp.async intrinsics |
| **L2 cache pinning for next-layer QKV** | Cross-layer | 3-5% decode latency | LOW | cudaAccessPolicyWindow API |
| **Split-KV attention (already in v3 kernel)** | Attention at C>512 | 50-200% attention phase | LOW | Wire v3 into forward path |
| **Batched persistent GEMV** | Batch B>1 decode | 30-100% at B=4-32 | HIGH | New kernel |
| **Vectorized embedding gather** | Embedding | < 1% | TRIVIAL | None |
| **Vectorized fused_residual_rmsnorm** | All norm kernels | 2-3% norm phase | LOW | None |
| **FP8 LM head weights** | LM head | 8-10% decode latency | LOW | Cast LM head to FP8 |
| **2:4 structured sparsity** | All weights | 40-50% decode latency | HIGH | Pruned model checkpoint, new kernels |
| **TMA descriptor-based loads (cp.async.bulk)** | All sm_90 kernels | 5-10% | HIGH | H100-specific, new kernel arch |
| **Fix O-proj serialization in mega-fused kernel** | oproj+gateup fused | 15-25% for that kernel | MEDIUM | Restructure to warp-parallel O-proj |

### Priority ordering for maximum impact:

1. **FP8 weights everywhere** (+40-50%) -- partially done, extend to all paths
2. **int4 vectorized loads** (+5-10%) -- mechanical change in ~6 kernels
3. **Wire split-KV v3 attention** (+50-200% attention) -- kernel exists
4. **cp.async in fused GEMV** (+10-20% GEMV) -- moderate effort
5. **Batched persistent GEMV** (+30-100% batched) -- new kernel needed
6. **Fix O-proj serialization** (+15-25% that kernel) -- restructure needed
7. **FP8 LM head** (+8-10%) -- trivial extension of FP8 infrastructure
8. **L2 cache pinning** (+3-5%) -- API call, low effort
9. **2:4 sparsity** (+40-50%) -- requires pruned models, long-term
10. **TMA descriptors** (+5-10%) -- H100-specific, research

**Combined near-term (items 1-4+7+8):** Estimated 60-80% decode latency reduction for M=1.
**Theoretical floor:** 0.55 ms/token (FP8) vs current ~5-6 ms/token. Factor of ~10x improvement ceiling from pure bandwidth optimization.

---

### Critical Files for Implementation

- `/Users/andy/rvllm/kernels/fused_add_norm_qkv_gemv.cu` -- Primary fused GEMV kernel; needs int4 vectorization in the gateup/silu_down/oproj variants and cp.async prefetching
- `/Users/andy/rvllm/kernels/fused_silu_down_gemv.cu` -- Down projection GEMV; needs upgrade from half2 to int4 vectorized loads (second-largest weight read per layer)
- `/Users/andy/rvllm/kernels/fused_oproj_add_norm_gateup_gemv.cu` -- Mega-fused kernel with serial O-proj bottleneck; needs restructuring to warp-parallel O-proj and int4 loads
- `/Users/andy/rvllm/crates/rvllm-model-runner/src/gpu_layer.rs` -- Forward pass orchestrator; controls which kernel path executes and where L2 cache pinning/split-KV wiring would be added
- `/Users/andy/rvllm/kernels/flash_attention_3_v3.cu` -- Split-KV + cp.async attention kernel already implemented but not wired as default decode path; wiring this is a high-impact low-effort win