// FlashAttention-2 kernel with paged KV cache support.
//
// Implements the tiled FlashAttention-2 algorithm (Dao, 2023):
//   - Tiled Q*K computation entirely in SRAM (shared memory)
//   - Online softmax (Milakov & Gimelshein) -- no full NxN attention matrix
//   - Causal masking built-in
//   - Backward-compatible with PagedAttention block tables
//
// Supports head_dim: 64, 96, 128
//
// Launch config:
//   Grid:  (num_seqs, num_heads, 1)
//   Block: (THREADS_PER_BLOCK, 1, 1)  -- typically 128 or 256
//   Shared memory: see smem sizing below
//
// Each thread block computes attention for one (sequence, head) pair.
// The Q tile is loaded once; K/V tiles are streamed from paged KV cache.
//
// Provides both f32 and f16-KV variants:
//   - flash_attention_2_kernel / flash_attention_2_decode_kernel: f32 cache
//   - flash_attention_2_f16kv_kernel / flash_attention_2_decode_f16kv_kernel: f16 cache
// The f16kv variants load half-precision K/V from the paged cache and promote
// to f32 in shared memory. All computation (QK dot, softmax, PV accum) is f32.

#include <float.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// Configuration constants
// ============================================================================

// Tile sizes for K/V streaming. Br = rows of Q per tile, Bc = cols of K per tile.
// For decode (single query token), Br=1 is optimal.
// For prefill, Br=64 or 128 is typical.
//
// Bc default is 64 (SM80 / SM89 / SM90). Blackwell consumer targets (sm_100,
// sm_121) have tighter static-smem budgets per block; head_dim=256 with Bc=64
// overflows the 48 KB default limit on sm_121 (PR#28 bring-up). Halve Bc to
// 32 on those arches so the per-block smem footprint stays within limits
// without changing the SM90 production path. An explicit `-DFA2_BC=<n>` on
// the nvcc command line still wins — the build system can override per-arch
// if future tuning wants a different value.
#ifndef FA2_BC
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#    define FA2_BC 32      // sm_100 / sm_121 / sm_122 — smem-budget safe
#  else
#    define FA2_BC 64      // sm_80 / sm_89 / sm_90 — unchanged baseline
#  endif
#endif
#define FA2_THREADS 128    // Threads per block

// ============================================================================
// Utility: warp-level reductions
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Broadcast from lane 0 of each warp
__device__ __forceinline__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

// ============================================================================
// Block-level reduce via shared memory
// ============================================================================

__device__ float block_reduce_max(float val, float* smem_reduce, int tid, int num_threads) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (num_threads + 31) / 32;

    val = warp_reduce_max(val);
    if (lane_id == 0) {
        smem_reduce[warp_id] = val;
    }
    __syncthreads();

    if (tid < num_warps) {
        val = smem_reduce[tid];
    } else {
        val = -FLT_MAX;
    }
    if (tid < 32) {
        val = warp_reduce_max(val);
    }
    return val; // valid in lane 0 of warp 0
}

__device__ float block_reduce_sum(float val, float* smem_reduce, int tid, int num_threads) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (num_threads + 31) / 32;

    val = warp_reduce_sum(val);
    if (lane_id == 0) {
        smem_reduce[warp_id] = val;
    }
    __syncthreads();

    if (tid < num_warps) {
        val = smem_reduce[tid];
    } else {
        val = 0.0f;
    }
    if (tid < 32) {
        val = warp_reduce_sum(val);
    }
    return val; // valid in lane 0 of warp 0
}

// ============================================================================
// FlashAttention-2 forward kernel with paged KV cache
// ============================================================================
//
// For each (seq, head), the kernel:
//   1. Loads the query row(s) into registers/shared memory.
//   2. Iterates over KV positions in tiles of size Bc.
//      For each tile:
//        a. Loads K tile from paged cache into shared memory.
//        b. Computes S = Q * K^T (tiled matmul in SRAM).
//        c. Applies causal mask if enabled.
//        d. Updates online softmax running max and sum.
//        e. Rescales previous accumulator.
//        f. Loads V tile from paged cache into shared memory.
//        g. Accumulates P * V into output.
//   3. Final normalization: output /= softmax_sum.

extern "C"
__global__ void flash_attention_2_kernel(
    float* __restrict__ output,            // [num_tokens, num_heads, head_dim]
    const float* __restrict__ query,       // [num_tokens, num_heads, head_dim]
    const float* __restrict__ key_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ value_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int* __restrict__ seq_start_pos, // [num_seqs+1] -- cumulative start positions for prefill (sentinel at end)
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int num_query_tokens,      // total query tokens (1 per seq for decode)
    int causal                 // 1 = apply causal mask, 0 = no mask
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // GQA: map query head to KV head
    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx : (head_idx / (num_heads / num_kv_heads));

    // Query start position for this sequence (for multi-token prefill).
    // seq_start_pos has num_seqs+1 entries (sentinel at end = num_query_tokens)
    // so seq_start_pos[seq_idx+1] is always a valid read.
    const int q_start = (seq_start_pos != nullptr) ? seq_start_pos[seq_idx] : seq_idx;
    const int q_len = (seq_start_pos != nullptr)
                      ? (seq_start_pos[seq_idx + 1] - q_start)
                      : (num_query_tokens - q_start);

    // ---- Shared memory layout ----
    // We partition shared memory as:
    //   1. K tile:    [Bc, head_dim]  floats
    //   2. V tile:    [Bc, head_dim]  floats
    //   3. S tile:    [Bc]            floats (attention scores for current tile)
    //   4. Reduce:    [FA2_THREADS/32] floats (for block-level reductions)
    extern __shared__ float smem[];
    float* s_key   = smem;                                         // [Bc * head_dim]
    float* s_val   = smem + FA2_BC * head_dim;                     // [Bc * head_dim]
    float* s_score = smem + 2 * FA2_BC * head_dim;                 // [Bc]
    float* s_reduce = smem + 2 * FA2_BC * head_dim + FA2_BC;       // [FA2_THREADS/32]

    // Number of KV tiles
    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;

    // Process each query position sequentially (typically q_len=1 for decode)
    for (int qi = 0; qi < q_len; qi++) {
        const int q_pos = q_start + qi;
        const int q_global_pos = (seq_start_pos != nullptr) ? q_pos : seq_idx;

        // Load query vector into registers -- each thread handles multiple dimensions
        // Strategy: thread tid handles dimensions tid, tid+THREADS, tid+2*THREADS, ...
        const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
        float q_reg[8]; // max 8 dims per thread (head_dim <= 128, threads >= 128 => 1)
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                q_reg[r] = query[(q_global_pos * num_heads + head_idx) * head_dim + d] * scale;
            } else {
                q_reg[r] = 0.0f;
            }
        }

        // Online softmax accumulators (per-thread output dims)
        float row_max = -FLT_MAX;
        float row_sum = 0.0f;
        float acc[8]; // accumulator for output dimensions
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            acc[r] = 0.0f;
        }

        // Iterate over KV tiles
        for (int tile = 0; tile < num_kv_tiles; tile++) {
            const int tile_start = tile * FA2_BC;
            const int tile_len = min(FA2_BC, context_len - tile_start);

            // ---- Load K tile from paged cache into shared memory ----
            // Each thread cooperatively loads elements
            for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
                int t = idx / head_dim;
                int d = idx % head_dim;
                int kv_pos = tile_start + t;

                // Resolve paged address
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int k_offset = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;

                s_key[t * head_dim + d] = key_cache[k_offset];
            }
            __syncthreads();

            // ---- Compute S = Q * K^T for this tile ----
            // Each thread computes partial dot products for all tile_len positions
            for (int t = 0; t < tile_len; t++) {
                float dot = 0.0f;
                for (int r = 0; r < dims_per_thread && r < 8; r++) {
                    int d = tid + r * FA2_THREADS;
                    if (d < head_dim) {
                        dot += q_reg[r] * s_key[t * head_dim + d];
                    }
                }

                // Block-level sum reduction for this dot product
                dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);

                if (tid == 0) {
                    int kv_pos = tile_start + t;
                    if (causal && kv_pos > (context_len - q_len + qi)) {
                        s_score[t] = -FLT_MAX;
                    } else {
                        s_score[t] = dot;
                    }
                }
                __syncthreads();
            }

            // ---- Online softmax: find tile max and update running max ----
            float tile_max = -FLT_MAX;
            if (tid == 0) {
                for (int t = 0; t < tile_len; t++) {
                    tile_max = fmaxf(tile_max, s_score[t]);
                }
            }
            // Broadcast tile_max from thread 0
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);
            // For threads not in warp 0, broadcast via shared memory
            if (tid == 0) {
                s_reduce[0] = tile_max;
            }
            __syncthreads();
            tile_max = s_reduce[0];
            __syncthreads();

            // Update running max and rescale
            float prev_max = row_max;
            float new_max = fmaxf(row_max, tile_max);

            // Rescale previous accumulator
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                for (int r = 0; r < dims_per_thread && r < 8; r++) {
                    acc[r] *= correction;
                }
                row_sum *= correction;
            }
            row_max = new_max;

            // Exponentiate scores and compute tile sum
            if (tid == 0) {
                float tsum = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    float val = (s_score[t] > -FLT_MAX + 1.0f) ? expf(s_score[t] - row_max) : 0.0f;
                    s_score[t] = val;
                    tsum += val;
                }
                // Store tile sum in reduce slot for broadcast
                s_reduce[0] = tsum;
            }
            __syncthreads();
            float tile_sum = s_reduce[0];
            row_sum += tile_sum;
            __syncthreads();

            // ---- Load V tile from paged cache into shared memory ----
            for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
                int t = idx / head_dim;
                int d = idx % head_dim;
                int kv_pos = tile_start + t;

                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int v_offset = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;

                s_val[t * head_dim + d] = value_cache[v_offset];
            }
            __syncthreads();

            // ---- Accumulate P * V ----
            // Each thread handles its assigned output dimensions
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * FA2_THREADS;
                if (d < head_dim) {
                    float val_acc = 0.0f;
                    for (int t = 0; t < tile_len; t++) {
                        val_acc += s_score[t] * s_val[t * head_dim + d];
                    }
                    acc[r] += val_acc;
                }
            }
            __syncthreads();
        }

        // ---- Final normalization and write output ----
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                int out_idx = (q_global_pos * num_heads + head_idx) * head_dim + d;
                output[out_idx] = acc[r] * inv_sum;
            }
        }
    }
}

// ============================================================================
// Decode-optimized variant: single query token per sequence.
// Uses a simpler path with no multi-token loop.
// ============================================================================

extern "C"
__global__ void flash_attention_2_decode_kernel(
    float* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    const float* __restrict__ query,       // [num_seqs, num_heads, head_dim]
    const float* __restrict__ key_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ value_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx : (head_idx / (num_heads / num_kv_heads));

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + FA2_BC * head_dim;
    float* s_score  = smem + 2 * FA2_BC * head_dim;
    float* s_reduce = smem + 2 * FA2_BC * head_dim + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;

    // Load query into registers
    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            q_reg[r] = query[(seq_idx * num_heads + head_idx) * head_dim + d] * scale;
        } else {
            q_reg[r] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        acc[r] = 0.0f;
    }

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int tile_start = tile * FA2_BC;
        const int tile_len = min(FA2_BC, context_len - tile_start);

        // Load K tile
        for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
            int t = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            s_key[t * head_dim + d] = key_cache[((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d];
        }
        __syncthreads();

        // Q * K^T
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * FA2_THREADS;
                if (d < head_dim) {
                    dot += q_reg[r] * s_key[t * head_dim + d];
                }
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) {
                s_score[t] = dot;
            }
            __syncthreads();
        }

        // Online softmax update
        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; t++) {
                tile_max = fmaxf(tile_max, s_score[t]);
            }
            s_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_reduce[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                acc[r] *= correction;
            }
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                float val = expf(s_score[t] - row_max);
                s_score[t] = val;
                tsum += val;
            }
            s_reduce[0] = tsum;
        }
        __syncthreads();
        row_sum += s_reduce[0];
        __syncthreads();

        // Load V tile
        for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
            int t = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            s_val[t * head_dim + d] = value_cache[((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d];
        }
        __syncthreads();

        // Accumulate P * V
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    val_acc += s_score[t] * s_val[t * head_dim + d];
                }
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    // Normalize and write
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] = acc[r] * inv_sum;
        }
    }
}

// ============================================================================
// F16 KV cache variants
//
// Identical algorithms to the f32 kernels above, but key_cache and value_cache
// are __half*. Each load from the paged cache converts f16 -> f32 via
// __half2float(). All shared memory and computation remains f32.
// Q, output, and all scalar accumulators are f32.
// ============================================================================

extern "C"
__global__ void flash_attention_2_f16kv_kernel(
    float* __restrict__ output,            // [num_tokens, num_heads, head_dim]
    const float* __restrict__ query,       // [num_tokens, num_heads, head_dim]
    const __half* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] f16
    const __half* __restrict__ value_cache,// [num_blocks, block_size, num_kv_heads, head_dim] f16
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int* __restrict__ seq_start_pos, // [num_seqs+1]
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int num_query_tokens,
    int causal
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx : (head_idx / (num_heads / num_kv_heads));

    const int q_start = (seq_start_pos != nullptr) ? seq_start_pos[seq_idx] : seq_idx;
    const int q_len = (seq_start_pos != nullptr)
                      ? (seq_start_pos[seq_idx + 1] - q_start)
                      : (num_query_tokens - q_start);

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + FA2_BC * head_dim;
    float* s_score  = smem + 2 * FA2_BC * head_dim;
    float* s_reduce = smem + 2 * FA2_BC * head_dim + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;

    for (int qi = 0; qi < q_len; qi++) {
        const int q_pos = q_start + qi;
        const int q_global_pos = (seq_start_pos != nullptr) ? q_pos : seq_idx;

        const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
        float q_reg[8];
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                q_reg[r] = query[(q_global_pos * num_heads + head_idx) * head_dim + d] * scale;
            } else {
                q_reg[r] = 0.0f;
            }
        }

        float row_max = -FLT_MAX;
        float row_sum = 0.0f;
        float acc[8];
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            acc[r] = 0.0f;
        }

        for (int tile = 0; tile < num_kv_tiles; tile++) {
            const int tile_start = tile * FA2_BC;
            const int tile_len = min(FA2_BC, context_len - tile_start);

            // Load K tile from f16 paged cache -> f32 shared memory
            for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
                int t = idx / head_dim;
                int d = idx % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int k_offset = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                s_key[t * head_dim + d] = __half2float(key_cache[k_offset]);
            }
            __syncthreads();

            // Q * K^T
            for (int t = 0; t < tile_len; t++) {
                float dot = 0.0f;
                for (int r = 0; r < dims_per_thread && r < 8; r++) {
                    int d = tid + r * FA2_THREADS;
                    if (d < head_dim) {
                        dot += q_reg[r] * s_key[t * head_dim + d];
                    }
                }
                dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
                if (tid == 0) {
                    int kv_pos = tile_start + t;
                    if (causal && kv_pos > (context_len - q_len + qi)) {
                        s_score[t] = -FLT_MAX;
                    } else {
                        s_score[t] = dot;
                    }
                }
                __syncthreads();
            }

            // Online softmax
            float tile_max = -FLT_MAX;
            if (tid == 0) {
                for (int t = 0; t < tile_len; t++) {
                    tile_max = fmaxf(tile_max, s_score[t]);
                }
            }
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);
            if (tid == 0) { s_reduce[0] = tile_max; }
            __syncthreads();
            tile_max = s_reduce[0];
            __syncthreads();

            float prev_max = row_max;
            float new_max = fmaxf(row_max, tile_max);
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                for (int r = 0; r < dims_per_thread && r < 8; r++) {
                    acc[r] *= correction;
                }
                row_sum *= correction;
            }
            row_max = new_max;

            if (tid == 0) {
                float tsum = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    float val = (s_score[t] > -FLT_MAX + 1.0f) ? expf(s_score[t] - row_max) : 0.0f;
                    s_score[t] = val;
                    tsum += val;
                }
                s_reduce[0] = tsum;
            }
            __syncthreads();
            float tile_sum = s_reduce[0];
            row_sum += tile_sum;
            __syncthreads();

            // Load V tile from f16 paged cache -> f32 shared memory
            for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
                int t = idx / head_dim;
                int d = idx % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int v_offset = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                s_val[t * head_dim + d] = __half2float(value_cache[v_offset]);
            }
            __syncthreads();

            // Accumulate P * V
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * FA2_THREADS;
                if (d < head_dim) {
                    float val_acc = 0.0f;
                    for (int t = 0; t < tile_len; t++) {
                        val_acc += s_score[t] * s_val[t * head_dim + d];
                    }
                    acc[r] += val_acc;
                }
            }
            __syncthreads();
        }

        // Final normalization and write output (f32)
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                int out_idx = (q_global_pos * num_heads + head_idx) * head_dim + d;
                output[out_idx] = acc[r] * inv_sum;
            }
        }
    }
}

// ============================================================================
// Decode-optimized F16 KV variant
// ============================================================================

extern "C"
__global__ void flash_attention_2_decode_f16kv_kernel(
    float* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    const float* __restrict__ query,       // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] f16
    const __half* __restrict__ value_cache,// [num_blocks, block_size, num_kv_heads, head_dim] f16
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx : (head_idx / (num_heads / num_kv_heads));

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + FA2_BC * head_dim;
    float* s_score  = smem + 2 * FA2_BC * head_dim;
    float* s_reduce = smem + 2 * FA2_BC * head_dim + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;

    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            q_reg[r] = query[(seq_idx * num_heads + head_idx) * head_dim + d] * scale;
        } else {
            q_reg[r] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        acc[r] = 0.0f;
    }

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int tile_start = tile * FA2_BC;
        const int tile_len = min(FA2_BC, context_len - tile_start);

        // Load K tile: f16 cache -> f32 shared memory
        for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
            int t = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            s_key[t * head_dim + d] = __half2float(key_cache[((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d]);
        }
        __syncthreads();

        // Q * K^T
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * FA2_THREADS;
                if (d < head_dim) {
                    dot += q_reg[r] * s_key[t * head_dim + d];
                }
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) {
                s_score[t] = dot;
            }
            __syncthreads();
        }

        // Online softmax update
        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; t++) {
                tile_max = fmaxf(tile_max, s_score[t]);
            }
            s_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_reduce[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                acc[r] *= correction;
            }
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                float val = expf(s_score[t] - row_max);
                s_score[t] = val;
                tsum += val;
            }
            s_reduce[0] = tsum;
        }
        __syncthreads();
        row_sum += s_reduce[0];
        __syncthreads();

        // Load V tile: f16 cache -> f32 shared memory
        for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
            int t = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            s_val[t * head_dim + d] = __half2float(value_cache[((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d]);
        }
        __syncthreads();

        // Accumulate P * V
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    val_acc += s_score[t] * s_val[t * head_dim + d];
                }
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    // Normalize and write (f32 output)
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] = acc[r] * inv_sum;
        }
    }
}

// ============================================================================
// FP8 E4M3 paged-decode variant — matches the PagedDecodeFp8Launcher ABI
// ============================================================================
//
// Drop-in replacement for `fa3_sm90_paged_decode_fp8` on targets where
// FA3 doesn't apply (sm_121). Q, K, V are all FP8 E4M3 with per-tensor
// scalar descales; output is f16 to match the FA3 .so ABI.
//
// Internal math stays in f32 for numerical stability. On-load dequant
// per byte:
//     val_f32 = fp8_e4m3_to_float(byte) * *descale
// `window_size_left` is accepted but treated as `-1` (full context) for
// now — sliding-window attention on the FP8-KV path is a follow-up.

__device__ __forceinline__ float fp8kv_decode_byte(unsigned char b) {
#if __CUDA_ARCH__ >= 1000
    // Native Blackwell path: PTX `cvt.rn.f16.e4m3x2` emitted for
    // sm_100+. On GB10 this is a single-cycle hardware conversion
    // (measured ~2× faster than the branchless scalar path in the
    // fp8_gemv bench, which is the analogous hot loop).
    __half_raw hr = __nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(__half(hr));
#else
    // Pre-Blackwell path: branchless closed-form decode (avoids the
    // divergent if/else in the original scalar path). Mirrors
    // `fp8e4m3_to_float` in kernels/fp8_gemv.cu.
    unsigned int s = (b >> 7) & 1u;
    unsigned int e = (b >> 3) & 0xFu;
    unsigned int m = b & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
#endif
}

extern "C"
__global__ void flash_attention_2_decode_fp8kv_kernel(
    __half* __restrict__ output,                 // [num_seqs, num_heads, head_dim] f16
    const unsigned char* __restrict__ query,     // [num_seqs, num_heads, head_dim] fp8
    const unsigned char* __restrict__ key_cache, // [num_blocks, block_size, num_kv_heads, head_dim] fp8
    const unsigned char* __restrict__ value_cache,
    const float* __restrict__ k_scale_cache,     // per-slot f32, OR nullptr -> use k_descale_fallback
    const float* __restrict__ v_scale_cache,     // per-slot f32, OR nullptr -> use v_descale_fallback
    const float* __restrict__ q_scale_cache,     // per-(seq,head) f32, OR nullptr -> use q_descale
    const float* __restrict__ k_descale_fallback,// scalar, used only if k_scale_cache==nullptr
    const float* __restrict__ v_descale_fallback,// scalar, used only if v_scale_cache==nullptr
    const int* __restrict__ block_tables,        // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,        // [num_seqs]
    const float* __restrict__ q_descale,         // single scalar
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int window_size_left
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx
                                                         : (head_idx / (num_heads / num_kv_heads));

    const float q_scale = (q_scale_cache != nullptr)
        ? __ldg(&q_scale_cache[seq_idx * num_heads + head_idx])
        : *q_descale;
    // Per-slot KV scales (Gemma 4) or scalar fallback (Llama/Qwen).
    // Branch is on a per-launch parameter so the compiler can hoist
    // it out of the inner tile loop.
    const bool k_perslot = (k_scale_cache != nullptr);
    const bool v_perslot = (v_scale_cache != nullptr);
    const float k_scale_scalar = k_perslot ? 0.0f : __ldg(k_descale_fallback);
    const float v_scale_scalar = v_perslot ? 0.0f : __ldg(v_descale_fallback);

    // Sliding-window left boundary. `window_size_left < 0` = full
    // context (global attention). Otherwise the decode query at
    // absolute KV position `context_len - 1` can only attend back
    // to `window_start = max(0, context_len - 1 - window_size_left)`.
    // kv_pos < window_start → score = -FLT_MAX (masked).
    const int decode_q_abs_pos = context_len - 1;
    const int window_start = (window_size_left < 0)
        ? 0
        : max(0, decode_q_abs_pos - window_size_left);

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + FA2_BC * head_dim;
    float* s_score  = smem + 2 * FA2_BC * head_dim;
    float* s_reduce = smem + 2 * FA2_BC * head_dim + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;

    // Load Q row with descale + attention scale folded in.
    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            unsigned char qb = query[(seq_idx * num_heads + head_idx) * head_dim + d];
            q_reg[r] = fp8kv_decode_byte(qb) * q_scale * scale;
        } else {
            q_reg[r] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] = 0.0f;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int tile_start = tile * FA2_BC;
        const int tile_len = min(FA2_BC, context_len - tile_start);

        // Load K tile: fp8 cache -> f32 shared mem (apply per-slot K
        // scale on load). Vectorized by 8: each thread pulls one
        // u64 = 8 FP8 bytes. head_dim is always a multiple of 8 for
        // the shapes we support (128/256/512), so the
        // tile_len*head_dim/8 work divides cleanly across the 128
        // threads.
        {
            const int vec_total = tile_len * (head_dim / 8);
            for (int vi = tid; vi < vec_total; vi += FA2_THREADS) {
                int t = vi / (head_dim / 8);
                int d_base = (vi % (head_dim / 8)) * 8;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int slot = phys_block * block_size + page_off;
                float k_scale = k_perslot
                    ? __ldg(&k_scale_cache[slot * num_kv_heads + kv_head_idx])
                    : k_scale_scalar;
                const unsigned char* k_row = key_cache
                    + (slot * num_kv_heads + kv_head_idx) * head_dim;
                unsigned long long k8 = __ldg(
                    reinterpret_cast<const unsigned long long*>(k_row + d_base));
                float* s = s_key + t * head_dim + d_base;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    s[b] = fp8kv_decode_byte((unsigned char)(k8 >> (b * 8))) * k_scale;
                }
            }
        }
        __syncthreads();

        // Q * K^T per tile column, with left-window mask applied
        // before softmax. `window_size_left < 0` (global) → window_start
        // is 0 → kv_pos >= 0 always, no-op mask.
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * FA2_THREADS;
                if (d < head_dim) {
                    dot += q_reg[r] * s_key[t * head_dim + d];
                }
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) {
                int kv_pos = tile_start + t;
                s_score[t] = (kv_pos < window_start) ? -FLT_MAX : dot;
            }
            __syncthreads();
        }

        // Online softmax update. Masked columns carry -FLT_MAX so
        // their exp is 0 and they contribute nothing to the row sum.
        // The `s_score[t] > -FLT_MAX + 1.0f` guard on the exp pass
        // keeps the sum numerically clean (otherwise exp(-FLT_MAX -
        // row_max) can become NaN on some architectures).
        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; t++) tile_max = fmaxf(tile_max, s_score[t]);
            s_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_reduce[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                // Masked entries (window-left) carry -FLT_MAX; skip
                // the exp to keep the sum exact and avoid the
                // platform-specific NaN that expf(-FLT_MAX) can
                // produce on some archs.
                float v = (s_score[t] > -FLT_MAX + 1.0f)
                            ? expf(s_score[t] - row_max) : 0.0f;
                s_score[t] = v;
                tsum += v;
            }
            s_reduce[0] = tsum;
        }
        __syncthreads();
        row_sum += s_reduce[0];
        __syncthreads();

        // Load V tile with per-slot V scale (see K load).
        {
            const int vec_total = tile_len * (head_dim / 8);
            for (int vi = tid; vi < vec_total; vi += FA2_THREADS) {
                int t = vi / (head_dim / 8);
                int d_base = (vi % (head_dim / 8)) * 8;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int slot = phys_block * block_size + page_off;
                float v_scale = v_perslot
                    ? __ldg(&v_scale_cache[slot * num_kv_heads + kv_head_idx])
                    : v_scale_scalar;
                const unsigned char* v_row = value_cache
                    + (slot * num_kv_heads + kv_head_idx) * head_dim;
                unsigned long long v8 = __ldg(
                    reinterpret_cast<const unsigned long long*>(v_row + d_base));
                float* s = s_val + t * head_dim + d_base;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    s[b] = fp8kv_decode_byte((unsigned char)(v8 >> (b * 8))) * v_scale;
                }
            }
        }
        __syncthreads();

        // Accumulate P * V.
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    val_acc += s_score[t] * s_val[t * head_dim + d];
                }
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    // Normalize + cast f32 -> f16 output.
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] =
                __float2half(acc[r] * inv_sum);
        }
    }
}

// ============================================================================
// FP8 E4M3 paged-decode variant — BC=16 for head_dim=512 (smem-budget fit)
// ============================================================================
//
// Same kernel as above but with `FA2_BC = 16` instead of the
// arch-default 32. Needed on sm_121 for Gemma 4 *global-attention*
// layers (head_dim=512): BC=32 gives a 128 KB smem request, beyond the
// 99 KB `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`
// ceiling on Blackwell consumer silicon. BC=16 drops the tile to
// 64 KB which fits.
//
// Dispatcher selects between the two at launch time based on
// `params.head_dim`. Duplicated rather than templated because all the
// smem offsets use the BC macro directly and the kernel is already
// long — one copy-paste-and-rename is the lower-entropy fix.

extern "C"
__global__ void flash_attention_2_decode_fp8kv_bc16_kernel(
    __half* __restrict__ output,
    const unsigned char* __restrict__ query,
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    const float* __restrict__ k_scale_cache,
    const float* __restrict__ v_scale_cache,
    const float* __restrict__ q_scale_cache,
    const float* __restrict__ k_descale_fallback,
    const float* __restrict__ v_descale_fallback,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float* __restrict__ q_descale,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int window_size_left
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx
                                                         : (head_idx / (num_heads / num_kv_heads));

    const float q_scale = (q_scale_cache != nullptr)
        ? __ldg(&q_scale_cache[seq_idx * num_heads + head_idx])
        : *q_descale;
    const bool k_perslot = (k_scale_cache != nullptr);
    const bool v_perslot = (v_scale_cache != nullptr);
    const float k_scale_scalar = k_perslot ? 0.0f : __ldg(k_descale_fallback);
    const float v_scale_scalar = v_perslot ? 0.0f : __ldg(v_descale_fallback);

    // Sliding-window boundary — see BC=32 sibling for the full note.
    const int decode_q_abs_pos = context_len - 1;
    const int window_start = (window_size_left < 0)
        ? 0
        : max(0, decode_q_abs_pos - window_size_left);

    extern __shared__ float smem[];
    // BC=16 layout: [16 * head_dim K tile, 16 * head_dim V tile, 16 scores, smem reduce]
    float* s_key    = smem;
    float* s_val    = smem + 16 * head_dim;
    float* s_score  = smem + 2 * 16 * head_dim;
    float* s_reduce = smem + 2 * 16 * head_dim + 16;

    const int num_kv_tiles = (context_len + 16 - 1) / 16;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;

    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            unsigned char qb = query[(seq_idx * num_heads + head_idx) * head_dim + d];
            q_reg[r] = fp8kv_decode_byte(qb) * q_scale * scale;
        } else {
            q_reg[r] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] = 0.0f;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int tile_start = tile * 16;
        const int tile_len = min(16, context_len - tile_start);

        // Vectorized-by-8 K load (see BC=32 kernel comment).
        {
            const int vec_total = tile_len * (head_dim / 8);
            for (int vi = tid; vi < vec_total; vi += FA2_THREADS) {
                int t = vi / (head_dim / 8);
                int d_base = (vi % (head_dim / 8)) * 8;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int slot = phys_block * block_size + page_off;
                float k_scale = k_perslot
                    ? __ldg(&k_scale_cache[slot * num_kv_heads + kv_head_idx])
                    : k_scale_scalar;
                const unsigned char* k_row = key_cache
                    + (slot * num_kv_heads + kv_head_idx) * head_dim;
                unsigned long long k8 = __ldg(
                    reinterpret_cast<const unsigned long long*>(k_row + d_base));
                float* s = s_key + t * head_dim + d_base;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    s[b] = fp8kv_decode_byte((unsigned char)(k8 >> (b * 8))) * k_scale;
                }
            }
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * FA2_THREADS;
                if (d < head_dim) {
                    dot += q_reg[r] * s_key[t * head_dim + d];
                }
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) {
                int kv_pos = tile_start + t;
                s_score[t] = (kv_pos < window_start) ? -FLT_MAX : dot;
            }
            __syncthreads();
        }

        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; t++) tile_max = fmaxf(tile_max, s_score[t]);
            s_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_reduce[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                float v = (s_score[t] > -FLT_MAX + 1.0f)
                            ? expf(s_score[t] - row_max) : 0.0f;
                s_score[t] = v;
                tsum += v;
            }
            s_reduce[0] = tsum;
        }
        __syncthreads();
        row_sum += s_reduce[0];
        __syncthreads();

        // Vectorized-by-8 V load with per-slot V scale.
        {
            const int vec_total = tile_len * (head_dim / 8);
            for (int vi = tid; vi < vec_total; vi += FA2_THREADS) {
                int t = vi / (head_dim / 8);
                int d_base = (vi % (head_dim / 8)) * 8;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int slot = phys_block * block_size + page_off;
                float v_scale = v_perslot
                    ? __ldg(&v_scale_cache[slot * num_kv_heads + kv_head_idx])
                    : v_scale_scalar;
                const unsigned char* v_row = value_cache
                    + (slot * num_kv_heads + kv_head_idx) * head_dim;
                unsigned long long v8 = __ldg(
                    reinterpret_cast<const unsigned long long*>(v_row + d_base));
                float* s = s_val + t * head_dim + d_base;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    s[b] = fp8kv_decode_byte((unsigned char)(v8 >> (b * 8))) * v_scale;
                }
            }
        }
        __syncthreads();

        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    val_acc += s_score[t] * s_val[t * head_dim + d];
                }
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] =
                __float2half(acc[r] * inv_sum);
        }
    }
}

// Fully f16 I/O variant: f16 query, f16 KV cache, f16 output.
// All internal math remains f32 for numerical stability.
// This eliminates the f32<->f16 casts around the attention kernel.
extern "C"
__global__ void flash_attention_2_decode_f16io_kernel(
    __half* __restrict__ output,           // [num_seqs, num_heads, head_dim] f16
    const __half* __restrict__ query,      // [num_seqs, num_heads, head_dim] f16
    const __half* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] f16
    const __half* __restrict__ value_cache,// [num_blocks, block_size, num_kv_heads, head_dim] f16
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx : (head_idx / (num_heads / num_kv_heads));

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + FA2_BC * head_dim;
    float* s_score  = smem + 2 * FA2_BC * head_dim;
    float* s_reduce = smem + 2 * FA2_BC * head_dim + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;

    // Load Q (f16 -> f32 registers)
    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            q_reg[r] = __half2float(query[(seq_idx * num_heads + head_idx) * head_dim + d]) * scale;
        } else {
            q_reg[r] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] = 0.0f;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int tile_start = tile * FA2_BC;
        const int tile_len = min(FA2_BC, context_len - tile_start);

        for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
            int t = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            s_key[t * head_dim + d] = __half2float(key_cache[((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d]);
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * FA2_THREADS;
                if (d < head_dim) dot += q_reg[r] * s_key[t * head_dim + d];
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) s_score[t] = dot;
            __syncthreads();
        }

        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; t++) tile_max = fmaxf(tile_max, s_score[t]);
            s_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_reduce[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                float val = expf(s_score[t] - row_max);
                s_score[t] = val;
                tsum += val;
            }
            s_reduce[0] = tsum;
        }
        __syncthreads();
        row_sum += s_reduce[0];
        __syncthreads();

        for (int idx = tid; idx < tile_len * head_dim; idx += FA2_THREADS) {
            int t = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            s_val[t * head_dim + d] = __half2float(value_cache[((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d]);
        }
        __syncthreads();

        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * FA2_THREADS;
            if (d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; t++) val_acc += s_score[t] * s_val[t * head_dim + d];
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    // Normalize and write (f16 output)
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * FA2_THREADS;
        if (d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] = __float2half(acc[r] * inv_sum);
        }
    }
}

