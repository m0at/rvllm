// Split-KV paged attention: distributes KV blocks across multiple thread blocks
// per (seq, head), then combines partial results.
//
// Inspired by b12x (lukealonso/b12x) split-KV design for Blackwell.
// Portable to SM 7.0+ (no TMA dependency).
//
// Two kernels:
//   1. split_kv_decode_f16kv_kernel  -- each block processes a chunk of KV tiles
//   2. split_kv_combine_kernel       -- reduces partial outputs across splits
//
// Forward kernel launch:
//   Grid:  (num_seqs, num_heads, num_splits)
//   Block: (SPLIT_KV_THREADS, 1, 1)
//
// Combine kernel launch:
//   Grid:  (num_seqs, num_heads, 1)
//   Block: (head_dim, 1, 1)    -- one thread per output dimension
//
// Workspace layout:
//   partial_out: [num_splits, num_seqs, num_heads, head_dim]  f32
//   partial_max: [num_splits, num_seqs, num_heads]            f32
//   partial_sum: [num_splits, num_seqs, num_heads]            f32

#include <float.h>
#include <cuda_fp16.h>

#define SPLIT_KV_THREADS 128
#define SPLIT_KV_BC 64  // KV tile width (positions per tile)

// ============================================================================
// Warp/block reductions (same as flash_attention.cu)
// ============================================================================

__device__ __forceinline__ float skv_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float skv_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float skv_block_reduce_sum_broadcast(float val, float* smem_reduce, int tid, int num_threads) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (num_threads + 31) / 32;

    val = skv_warp_reduce_sum(val);
    if (lane_id == 0) smem_reduce[warp_id] = val;
    __syncthreads();

    if (tid < 32) {
        float v = (tid < num_warps) ? smem_reduce[tid] : 0.0f;
        v = skv_warp_reduce_sum(v);
        if (tid == 0) smem_reduce[0] = v;
    }
    __syncthreads();
    return smem_reduce[0];
}

__device__ float skv_block_reduce_max_broadcast(float val, float* smem_reduce, int tid, int num_threads) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (num_threads + 31) / 32;

    val = skv_warp_reduce_max(val);
    if (lane_id == 0) smem_reduce[warp_id] = val;
    __syncthreads();

    if (tid < 32) {
        float v = (tid < num_warps) ? smem_reduce[tid] : -FLT_MAX;
        v = skv_warp_reduce_max(v);
        if (tid == 0) smem_reduce[0] = v;
    }
    __syncthreads();
    return smem_reduce[0];
}

// ============================================================================
// Split-KV decode kernel with f16 KV cache
//
// Each thread block handles a subset of KV tiles for one (seq, head).
// Writes unnormalized partial output + (max, sum) to workspace for combine.
// ============================================================================

extern "C"
__global__ void split_kv_decode_f16kv_kernel(
    float* __restrict__ partial_out,       // [num_splits, num_seqs, num_heads, head_dim]
    float* __restrict__ partial_max,       // [num_splits, num_seqs, num_heads]
    float* __restrict__ partial_sum,       // [num_splits, num_seqs, num_heads]
    const float* __restrict__ query,       // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ value_cache,// [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    float scale,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_splits
) {
    const int seq_idx   = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx
                          : (head_idx / (num_heads / num_kv_heads));

    // Determine which KV tiles this split handles
    const int total_tiles = (context_len + SPLIT_KV_BC - 1) / SPLIT_KV_BC;
    const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    const int start_tile = split_idx * tiles_per_split;
    const int end_tile = min(start_tile + tiles_per_split, total_tiles);

    // Workspace offset for this split
    const int ws_base = ((split_idx * num_seqs + seq_idx) * num_heads + head_idx);

    if (start_tile >= total_tiles) {
        // This split has no work -- write sentinel values
        if (tid == 0) {
            partial_max[ws_base] = -FLT_MAX;
            partial_sum[ws_base] = 0.0f;
        }
        const int dims_per_thread = (head_dim + SPLIT_KV_THREADS - 1) / SPLIT_KV_THREADS;
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * SPLIT_KV_THREADS;
            if (d < head_dim) {
                partial_out[ws_base * head_dim + d] = 0.0f;
            }
        }
        return;
    }

    // Shared memory layout: K tile + V tile + scores + reduce scratch
    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + SPLIT_KV_BC * head_dim;
    float* s_score  = smem + 2 * SPLIT_KV_BC * head_dim;
    float* s_reduce = smem + 2 * SPLIT_KV_BC * head_dim + SPLIT_KV_BC;

    const int dims_per_thread = (head_dim + SPLIT_KV_THREADS - 1) / SPLIT_KV_THREADS;

    // Load query into registers
    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
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

    // Iterate over this split's KV tiles
    for (int tile = start_tile; tile < end_tile; tile++) {
        const int tile_start = tile * SPLIT_KV_BC;
        const int tile_len = min(SPLIT_KV_BC, context_len - tile_start);

        // Load K tile: f16 -> f32 shared memory (half2 vectorized)
        {
            const int hd4 = head_dim >> 2;
            const int total_vec = tile_len * hd4;
            for (int idx = tid; idx < total_vec; idx += SPLIT_KV_THREADS) {
                int t = idx / hd4;
                int v = idx % hd4;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int k_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + v * 4;
                half2 h01 = __ldg(reinterpret_cast<const half2*>(&key_cache[k_base]));
                half2 h23 = __ldg(reinterpret_cast<const half2*>(&key_cache[k_base + 2]));
                int s_base = t * head_dim + v * 4;
                s_key[s_base]     = __half2float(h01.x);
                s_key[s_base + 1] = __half2float(h01.y);
                s_key[s_base + 2] = __half2float(h23.x);
                s_key[s_base + 3] = __half2float(h23.y);
            }
        }
        __syncthreads();

        // Q * K^T with broadcast reduction
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * SPLIT_KV_THREADS;
                if (d < head_dim) {
                    dot += q_reg[r] * s_key[t * head_dim + d];
                }
            }
            dot = skv_block_reduce_sum_broadcast(dot, s_reduce, tid, SPLIT_KV_THREADS);
            if (tid == 0) {
                s_score[t] = dot;
            }
        }
        __syncthreads();

        // Online softmax update
        {
            float local_max = -FLT_MAX;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS) {
                local_max = fmaxf(local_max, s_score[t]);
            }
            float tile_max = skv_block_reduce_max_broadcast(local_max, s_reduce, tid, SPLIT_KV_THREADS);

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

            float local_sum = 0.0f;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS) {
                float val = expf(s_score[t] - row_max);
                s_score[t] = val;
                local_sum += val;
            }
            __syncthreads();
            float tile_sum = skv_block_reduce_sum_broadcast(local_sum, s_reduce, tid, SPLIT_KV_THREADS);
            row_sum += tile_sum;
        }

        // Load V tile: f16 -> f32 shared memory (half2 vectorized)
        {
            const int hd4 = head_dim >> 2;
            const int total_vec = tile_len * hd4;
            for (int idx = tid; idx < total_vec; idx += SPLIT_KV_THREADS) {
                int t = idx / hd4;
                int v = idx % hd4;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int v_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + v * 4;
                half2 h01 = __ldg(reinterpret_cast<const half2*>(&value_cache[v_base]));
                half2 h23 = __ldg(reinterpret_cast<const half2*>(&value_cache[v_base + 2]));
                int s_base = t * head_dim + v * 4;
                s_val[s_base]     = __half2float(h01.x);
                s_val[s_base + 1] = __half2float(h01.y);
                s_val[s_base + 2] = __half2float(h23.x);
                s_val[s_base + 3] = __half2float(h23.y);
            }
        }
        __syncthreads();

        // Accumulate P * V (unnormalized)
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * SPLIT_KV_THREADS;
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

    // Write partial results to workspace (unnormalized output + max + sum)
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
        if (d < head_dim) {
            partial_out[ws_base * head_dim + d] = acc[r];
        }
    }
    if (tid == 0) {
        partial_max[ws_base] = row_max;
        partial_sum[ws_base] = row_sum;
    }
}

// ============================================================================
// Split-KV combine kernel
//
// Reduces partial outputs across splits using online softmax correction.
// Grid:  (num_seqs, num_heads, 1)
// Block: (head_dim, 1, 1)
// ============================================================================

extern "C"
__global__ void split_kv_combine_kernel(
    float* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    const float* __restrict__ partial_out, // [num_splits, num_seqs, num_heads, head_dim]
    const float* __restrict__ partial_max, // [num_splits, num_seqs, num_heads]
    const float* __restrict__ partial_sum, // [num_splits, num_seqs, num_heads]
    const int* __restrict__ context_lens,  // [num_seqs]
    int num_seqs,
    int num_heads,
    int head_dim,
    int num_splits
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx  = threadIdx.x;

    if (dim_idx >= head_dim) return;
    if (context_lens[seq_idx] == 0) return;

    // Find global max across all splits
    float global_max = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        int idx = (s * num_seqs + seq_idx) * num_heads + head_idx;
        float m = partial_max[idx];
        if (m > global_max) global_max = m;
    }

    // Skip if all splits had no data
    if (global_max <= -FLT_MAX + 1.0f) {
        output[(seq_idx * num_heads + head_idx) * head_dim + dim_idx] = 0.0f;
        return;
    }

    // Combine: rescale each split's output and sum by exp(split_max - global_max)
    float combined_out = 0.0f;
    float combined_sum = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int ws_base = (s * num_seqs + seq_idx) * num_heads + head_idx;
        float m = partial_max[ws_base];
        float sm = partial_sum[ws_base];

        if (sm <= 0.0f) continue;  // skip empty splits

        float correction = expf(m - global_max);
        combined_out += correction * partial_out[ws_base * head_dim + dim_idx];
        combined_sum += correction * sm;
    }

    float result = (combined_sum > 0.0f) ? (combined_out / combined_sum) : 0.0f;
    output[(seq_idx * num_heads + head_idx) * head_dim + dim_idx] = result;
}

// ============================================================================
// Split-KV decode kernel with FP8 E4M3 KV cache
//
// Reads unsigned char from KV cache and dequantizes via per-head scale factors.
// fp8_value = (float)byte * kv_scale[head]
// ============================================================================

// FP8 E4M3 to float conversion via bit manipulation
__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char bits) {
    if (bits == 0) return 0.0f;
    // E4M3: 1 sign, 4 exponent, 3 mantissa, bias=7, no inf/nan
    int sign = (bits >> 7) & 1;
    int exp_bits = (bits >> 3) & 0xF;
    int mant_bits = bits & 0x7;

    float mantissa;
    int exponent;
    if (exp_bits == 0) {
        // subnormal: value = (-1)^s * 2^(-6) * (0.mant)
        mantissa = (float)mant_bits / 8.0f;
        exponent = -6;
    } else {
        // normal: value = (-1)^s * 2^(exp-7) * (1.mant)
        mantissa = 1.0f + (float)mant_bits / 8.0f;
        exponent = exp_bits - 7;
    }

    float val = ldexpf(mantissa, exponent);
    return sign ? -val : val;
}

extern "C"
__global__ void split_kv_decode_fp8kv_kernel(
    float* __restrict__ partial_out,        // [num_splits, num_seqs, num_heads, head_dim]
    float* __restrict__ partial_max,        // [num_splits, num_seqs, num_heads]
    float* __restrict__ partial_sum,        // [num_splits, num_seqs, num_heads]
    const float* __restrict__ query,        // [num_seqs, num_heads, head_dim]
    const unsigned char* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] u8
    const unsigned char* __restrict__ value_cache, // same
    const float* __restrict__ kv_scale,     // [num_kv_heads] per-head dequant scale
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    float scale,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_splits
) {
    const int seq_idx   = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx
                          : (head_idx / (num_heads / num_kv_heads));
    const float kv_dequant = kv_scale[kv_head_idx];

    const int total_tiles = (context_len + SPLIT_KV_BC - 1) / SPLIT_KV_BC;
    const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    const int start_tile = split_idx * tiles_per_split;
    const int end_tile = min(start_tile + tiles_per_split, total_tiles);

    const int ws_base = ((split_idx * num_seqs + seq_idx) * num_heads + head_idx);

    if (start_tile >= total_tiles) {
        if (tid == 0) {
            partial_max[ws_base] = -FLT_MAX;
            partial_sum[ws_base] = 0.0f;
        }
        const int dims_per_thread = (head_dim + SPLIT_KV_THREADS - 1) / SPLIT_KV_THREADS;
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * SPLIT_KV_THREADS;
            if (d < head_dim) partial_out[ws_base * head_dim + d] = 0.0f;
        }
        return;
    }

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + SPLIT_KV_BC * head_dim;
    float* s_score  = smem + 2 * SPLIT_KV_BC * head_dim;
    float* s_reduce = smem + 2 * SPLIT_KV_BC * head_dim + SPLIT_KV_BC;

    const int dims_per_thread = (head_dim + SPLIT_KV_THREADS - 1) / SPLIT_KV_THREADS;

    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
        q_reg[r] = (d < head_dim) ? query[(seq_idx * num_heads + head_idx) * head_dim + d] * scale : 0.0f;
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] = 0.0f;

    for (int tile = start_tile; tile < end_tile; tile++) {
        const int tile_start = tile * SPLIT_KV_BC;
        const int tile_len = min(SPLIT_KV_BC, context_len - tile_start);

        // Load K tile: FP8 -> f32 shared memory (4-byte vectorized reads)
        {
            const int total_elems = tile_len * head_dim;
            for (int idx = tid; idx < total_elems; idx += SPLIT_KV_THREADS) {
                int t = idx / head_dim;
                int d = idx % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int k_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                s_key[t * head_dim + d] = fp8_e4m3_to_float(key_cache[k_base]) * kv_dequant;
            }
        }
        __syncthreads();

        // Q * K^T
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * SPLIT_KV_THREADS;
                if (d < head_dim) dot += q_reg[r] * s_key[t * head_dim + d];
            }
            dot = skv_block_reduce_sum_broadcast(dot, s_reduce, tid, SPLIT_KV_THREADS);
            if (tid == 0) s_score[t] = dot;
        }
        __syncthreads();

        // Online softmax
        {
            float local_max = -FLT_MAX;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS)
                local_max = fmaxf(local_max, s_score[t]);
            float tile_max = skv_block_reduce_max_broadcast(local_max, s_reduce, tid, SPLIT_KV_THREADS);

            float prev_max = row_max;
            float new_max = fmaxf(row_max, tile_max);
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] *= correction;
                row_sum *= correction;
            }
            row_max = new_max;

            float local_sum = 0.0f;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS) {
                float val = expf(s_score[t] - row_max);
                s_score[t] = val;
                local_sum += val;
            }
            __syncthreads();
            float tile_sum = skv_block_reduce_sum_broadcast(local_sum, s_reduce, tid, SPLIT_KV_THREADS);
            row_sum += tile_sum;
        }

        // Load V tile: FP8 -> f32 shared memory
        {
            const int total_elems = tile_len * head_dim;
            for (int idx = tid; idx < total_elems; idx += SPLIT_KV_THREADS) {
                int t = idx / head_dim;
                int d = idx % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int v_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                s_val[t * head_dim + d] = fp8_e4m3_to_float(value_cache[v_base]) * kv_dequant;
            }
        }
        __syncthreads();

        // Accumulate P * V
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * SPLIT_KV_THREADS;
            if (d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; t++)
                    val_acc += s_score[t] * s_val[t * head_dim + d];
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    // Write partial results
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
        if (d < head_dim) partial_out[ws_base * head_dim + d] = acc[r];
    }
    if (tid == 0) {
        partial_max[ws_base] = row_max;
        partial_sum[ws_base] = row_sum;
    }
}

// ============================================================================
// Single-split fast path: skip workspace, write directly to output.
// Equivalent to flash_attention_2_decode_f16kv but with the same interface
// for consistent dispatch from Rust.
// ============================================================================

extern "C"
__global__ void split_kv_decode_single_f16kv_kernel(
    float* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    const float* __restrict__ query,       // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ value_cache,// [num_blocks, block_size, num_kv_heads, head_dim]
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

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx
                          : (head_idx / (num_heads / num_kv_heads));

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + SPLIT_KV_BC * head_dim;
    float* s_score  = smem + 2 * SPLIT_KV_BC * head_dim;
    float* s_reduce = smem + 2 * SPLIT_KV_BC * head_dim + SPLIT_KV_BC;

    const int num_kv_tiles = (context_len + SPLIT_KV_BC - 1) / SPLIT_KV_BC;
    const int dims_per_thread = (head_dim + SPLIT_KV_THREADS - 1) / SPLIT_KV_THREADS;

    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
        q_reg[r] = (d < head_dim) ? query[(seq_idx * num_heads + head_idx) * head_dim + d] * scale : 0.0f;
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] = 0.0f;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        const int tile_start = tile * SPLIT_KV_BC;
        const int tile_len = min(SPLIT_KV_BC, context_len - tile_start);

        {
            const int hd4 = head_dim >> 2;
            const int total_vec = tile_len * hd4;
            for (int idx = tid; idx < total_vec; idx += SPLIT_KV_THREADS) {
                int t = idx / hd4;
                int v = idx % hd4;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int k_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + v * 4;
                half2 h01 = __ldg(reinterpret_cast<const half2*>(&key_cache[k_base]));
                half2 h23 = __ldg(reinterpret_cast<const half2*>(&key_cache[k_base + 2]));
                int s_base = t * head_dim + v * 4;
                s_key[s_base]     = __half2float(h01.x);
                s_key[s_base + 1] = __half2float(h01.y);
                s_key[s_base + 2] = __half2float(h23.x);
                s_key[s_base + 3] = __half2float(h23.y);
            }
        }
        __syncthreads();

        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * SPLIT_KV_THREADS;
                if (d < head_dim) dot += q_reg[r] * s_key[t * head_dim + d];
            }
            dot = skv_block_reduce_sum_broadcast(dot, s_reduce, tid, SPLIT_KV_THREADS);
            if (tid == 0) s_score[t] = dot;
        }
        __syncthreads();

        {
            float local_max = -FLT_MAX;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS)
                local_max = fmaxf(local_max, s_score[t]);
            float tile_max = skv_block_reduce_max_broadcast(local_max, s_reduce, tid, SPLIT_KV_THREADS);

            float prev_max = row_max;
            float new_max = fmaxf(row_max, tile_max);
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                for (int r = 0; r < dims_per_thread && r < 8; r++) acc[r] *= correction;
                row_sum *= correction;
            }
            row_max = new_max;

            float local_sum = 0.0f;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS) {
                float val = expf(s_score[t] - row_max);
                s_score[t] = val;
                local_sum += val;
            }
            __syncthreads();
            float tile_sum = skv_block_reduce_sum_broadcast(local_sum, s_reduce, tid, SPLIT_KV_THREADS);
            row_sum += tile_sum;
        }

        {
            const int hd4 = head_dim >> 2;
            const int total_vec = tile_len * hd4;
            for (int idx = tid; idx < total_vec; idx += SPLIT_KV_THREADS) {
                int t = idx / hd4;
                int v = idx % hd4;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int v_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + v * 4;
                half2 h01 = __ldg(reinterpret_cast<const half2*>(&value_cache[v_base]));
                half2 h23 = __ldg(reinterpret_cast<const half2*>(&value_cache[v_base + 2]));
                int s_base = t * head_dim + v * 4;
                s_val[s_base]     = __half2float(h01.x);
                s_val[s_base + 1] = __half2float(h01.y);
                s_val[s_base + 2] = __half2float(h23.x);
                s_val[s_base + 3] = __half2float(h23.y);
            }
        }
        __syncthreads();

        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * SPLIT_KV_THREADS;
            if (d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; t++)
                    val_acc += s_score[t] * s_val[t * head_dim + d];
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
        if (d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] = acc[r] * inv_sum;
        }
    }
}

// ============================================================================
// Split-KV decode kernel with f16 I/O (query is __half, KV cache is __half)
//
// Identical to split_kv_decode_f16kv_kernel but accepts __half query instead
// of float, matching the FA3 kernel interface for consistent Rust dispatch.
// Workspace buffers remain f32 for numerical stability during partial reduction.
//
// Grid:  (num_seqs, num_heads, num_splits)
// Block: (SPLIT_KV_THREADS, 1, 1)
// ============================================================================

extern "C"
__global__ void split_kv_decode_f16io_kernel(
    float* __restrict__ partial_out,       // [num_splits, num_seqs, num_heads, head_dim]
    float* __restrict__ partial_max,       // [num_splits, num_seqs, num_heads]
    float* __restrict__ partial_sum,       // [num_splits, num_seqs, num_heads]
    const __half* __restrict__ query,      // [num_seqs, num_heads, head_dim]  <-- f16
    const __half* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ value_cache,// [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    float scale,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_splits
) {
    const int seq_idx   = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads) ? head_idx
                          : (head_idx / (num_heads / num_kv_heads));

    const int total_tiles = (context_len + SPLIT_KV_BC - 1) / SPLIT_KV_BC;
    const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    const int start_tile = split_idx * tiles_per_split;
    const int end_tile = min(start_tile + tiles_per_split, total_tiles);

    const int ws_base = ((split_idx * num_seqs + seq_idx) * num_heads + head_idx);

    if (start_tile >= total_tiles) {
        if (tid == 0) {
            partial_max[ws_base] = -FLT_MAX;
            partial_sum[ws_base] = 0.0f;
        }
        const int dims_per_thread = (head_dim + SPLIT_KV_THREADS - 1) / SPLIT_KV_THREADS;
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * SPLIT_KV_THREADS;
            if (d < head_dim) {
                partial_out[ws_base * head_dim + d] = 0.0f;
            }
        }
        return;
    }

    extern __shared__ float smem[];
    float* s_key    = smem;
    float* s_val    = smem + SPLIT_KV_BC * head_dim;
    float* s_score  = smem + 2 * SPLIT_KV_BC * head_dim;
    float* s_reduce = smem + 2 * SPLIT_KV_BC * head_dim + SPLIT_KV_BC;

    const int dims_per_thread = (head_dim + SPLIT_KV_THREADS - 1) / SPLIT_KV_THREADS;

    // Load query into registers (f16 -> f32 with scale)
    float q_reg[8];
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
        if (d < head_dim) {
            q_reg[r] = __half2float(query[(seq_idx * num_heads + head_idx) * head_dim + d]) * scale;
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

    for (int tile = start_tile; tile < end_tile; tile++) {
        const int tile_start = tile * SPLIT_KV_BC;
        const int tile_len = min(SPLIT_KV_BC, context_len - tile_start);

        // Load K tile: f16 -> f32 shared memory (half2x2 vectorized)
        {
            const int hd4 = head_dim >> 2;
            const int total_vec = tile_len * hd4;
            for (int idx = tid; idx < total_vec; idx += SPLIT_KV_THREADS) {
                int t = idx / hd4;
                int v = idx % hd4;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int k_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + v * 4;
                half2 h01 = __ldg(reinterpret_cast<const half2*>(&key_cache[k_base]));
                half2 h23 = __ldg(reinterpret_cast<const half2*>(&key_cache[k_base + 2]));
                int s_base = t * head_dim + v * 4;
                s_key[s_base]     = __half2float(h01.x);
                s_key[s_base + 1] = __half2float(h01.y);
                s_key[s_base + 2] = __half2float(h23.x);
                s_key[s_base + 3] = __half2float(h23.y);
            }
        }
        __syncthreads();

        // Q * K^T with broadcast reduction
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            for (int r = 0; r < dims_per_thread && r < 8; r++) {
                int d = tid + r * SPLIT_KV_THREADS;
                if (d < head_dim) {
                    dot += q_reg[r] * s_key[t * head_dim + d];
                }
            }
            dot = skv_block_reduce_sum_broadcast(dot, s_reduce, tid, SPLIT_KV_THREADS);
            if (tid == 0) {
                s_score[t] = dot;
            }
        }
        __syncthreads();

        // Online softmax update
        {
            float local_max = -FLT_MAX;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS) {
                local_max = fmaxf(local_max, s_score[t]);
            }
            float tile_max = skv_block_reduce_max_broadcast(local_max, s_reduce, tid, SPLIT_KV_THREADS);

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

            float local_sum = 0.0f;
            for (int t = tid; t < tile_len; t += SPLIT_KV_THREADS) {
                float val = expf(s_score[t] - row_max);
                s_score[t] = val;
                local_sum += val;
            }
            __syncthreads();
            float tile_sum = skv_block_reduce_sum_broadcast(local_sum, s_reduce, tid, SPLIT_KV_THREADS);
            row_sum += tile_sum;
        }

        // Load V tile: f16 -> f32 shared memory (half2x2 vectorized)
        {
            const int hd4 = head_dim >> 2;
            const int total_vec = tile_len * hd4;
            for (int idx = tid; idx < total_vec; idx += SPLIT_KV_THREADS) {
                int t = idx / hd4;
                int v = idx % hd4;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = __ldg(&block_tables[seq_idx * max_blocks_per_seq + page_idx]);
                int v_base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + v * 4;
                half2 h01 = __ldg(reinterpret_cast<const half2*>(&value_cache[v_base]));
                half2 h23 = __ldg(reinterpret_cast<const half2*>(&value_cache[v_base + 2]));
                int s_base = t * head_dim + v * 4;
                s_val[s_base]     = __half2float(h01.x);
                s_val[s_base + 1] = __half2float(h01.y);
                s_val[s_base + 2] = __half2float(h23.x);
                s_val[s_base + 3] = __half2float(h23.y);
            }
        }
        __syncthreads();

        // Accumulate P * V (unnormalized)
        for (int r = 0; r < dims_per_thread && r < 8; r++) {
            int d = tid + r * SPLIT_KV_THREADS;
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

    // Write partial results to workspace (unnormalized output + max + sum)
    for (int r = 0; r < dims_per_thread && r < 8; r++) {
        int d = tid + r * SPLIT_KV_THREADS;
        if (d < head_dim) {
            partial_out[ws_base * head_dim + d] = acc[r];
        }
    }
    if (tid == 0) {
        partial_max[ws_base] = row_max;
        partial_sum[ws_base] = row_sum;
    }
}

// ============================================================================
// Split-KV combine kernel with f16 output
//
// Same as split_kv_combine_kernel but writes __half output.
// Grid:  (num_seqs, num_heads, 1)
// Block: (head_dim, 1, 1)
// ============================================================================

extern "C"
__global__ void split_kv_combine_f16io_kernel(
    __half* __restrict__ output,           // [num_seqs, num_heads, head_dim]  <-- f16
    const float* __restrict__ partial_out, // [num_splits, num_seqs, num_heads, head_dim]
    const float* __restrict__ partial_max, // [num_splits, num_seqs, num_heads]
    const float* __restrict__ partial_sum, // [num_splits, num_seqs, num_heads]
    const int* __restrict__ context_lens,  // [num_seqs]
    int num_seqs,
    int num_heads,
    int head_dim,
    int num_splits
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx  = threadIdx.x;

    if (dim_idx >= head_dim) return;
    if (context_lens[seq_idx] == 0) return;

    float global_max = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        int idx = (s * num_seqs + seq_idx) * num_heads + head_idx;
        float m = partial_max[idx];
        if (m > global_max) global_max = m;
    }

    if (global_max <= -FLT_MAX + 1.0f) {
        output[(seq_idx * num_heads + head_idx) * head_dim + dim_idx] = __float2half(0.0f);
        return;
    }

    float combined_out = 0.0f;
    float combined_sum = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int ws_base = (s * num_seqs + seq_idx) * num_heads + head_idx;
        float m = partial_max[ws_base];
        float sm = partial_sum[ws_base];

        if (sm <= 0.0f) continue;

        float correction = expf(m - global_max);
        combined_out += correction * partial_out[ws_base * head_dim + dim_idx];
        combined_sum += correction * sm;
    }

    float result = (combined_sum > 0.0f) ? (combined_out / combined_sum) : 0.0f;
    output[(seq_idx * num_heads + head_idx) * head_dim + dim_idx] = __float2half(result);
}
