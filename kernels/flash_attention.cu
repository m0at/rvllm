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

#include <float.h>

// ============================================================================
// Configuration constants
// ============================================================================

// Tile sizes for K/V streaming. Br = rows of Q per tile, Bc = cols of K per tile.
// For decode (single query token), Br=1 is optimal.
// For prefill, Br=64 or 128 is typical.
#define FA2_BC 64          // K/V tile width (number of KV positions per tile)
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
    const int* __restrict__ seq_start_pos, // [num_seqs] -- cumulative start positions for prefill
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

    // Query start position for this sequence (for multi-token prefill)
    const int q_start = (seq_start_pos != nullptr) ? seq_start_pos[seq_idx] : seq_idx;
    const int q_len = (seq_start_pos != nullptr && seq_idx + 1 < gridDim.x)
                      ? (seq_start_pos[seq_idx + 1] - q_start)
                      : 1;

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
                    // Apply causal mask: if q_pos < kv_pos, mask out
                    int kv_pos = tile_start + t;
                    if (causal && kv_pos > (q_global_pos - (context_len - q_len) + qi)) {
                        // For prefill: q position in the sequence is qi,
                        // kv position is kv_pos. Causal means qi < kv_pos => mask.
                        // The absolute position of query token qi is:
                        //   (context_len - q_len) + qi  [offset of first query token + qi]
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
