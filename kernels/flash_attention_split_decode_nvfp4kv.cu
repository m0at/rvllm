// Split-KV decode + reduce for NVFP4 paged KV cache — vLLM-style
// `paged_attention_v2` adapted to our FP8-Q / NVFP4-KV / f16-out ABI.
//
// Phase 1 (split):  grid=(num_seqs, num_heads, max_num_partitions).
//   Each CTA runs standard FA2 decode over its partition's KV slice
//   `[p*P, (p+1)*P)`, tile size FA2_BC. After the softmax-over-
//   partition, it stores into scratch:
//     tmp_out    [S, H, P, D]  f16 — per-partition attention output,
//                                    DIVIDED by the partition's own
//                                    `exp_sum` (this is the invariant
//                                    the reduce relies on).
//     max_logits [S, H, P]     f32
//     exp_sums   [S, H, P]     f32
//   Empty / fully-sliding-masked partitions write sentinels
//   (max_logit = -INF, exp_sum = 0). The reduce's `isfinite(m)` check
//   handles them.
//
// Phase 2 (reduce): grid=(num_seqs, num_heads). Reads all partitions'
//   tmp_out + max_logits + exp_sums, recombines via the online-
//   softmax formula:
//     global_max = max_p(max_logits[p])
//     w[p]       = exp_sums[p] * exp(max_logits[p] - global_max)
//     inv        = 1 / (sum_p(w[p]) + 1e-6)
//     out[d]     = sum_p(tmp_out[p, d] * w[p] * inv)
//
// PARTITION_SIZE is a runtime scalar (passed by the launcher), must be
// a multiple of FA2_BC. vLLM uses 512 as default. When
// ceil(ctx / PARTITION_SIZE) == 1, the caller should dispatch the
// existing per-head decode kernel instead — this path is only the
// split variant. The reduce kernel still copes with P=1 (copy-only).

// NOTE: this kernel calls `unpack16_nvfp4_to_f16_fast` from
// `nvfp4_utils.cuh`, which emits `cvt.rn.f16x2.e2m1x2` — the
// accelerated-only NVFP4 dequant instruction. `kernels/build.sh`
// greps CU files (not headers) for that string to decide whether to
// upgrade the arch tag from `sm_121` to `sm_121a`, so the mention
// above is load-bearing: without it the PTX lands on `sm_121` and
// the JIT fails with CUDA_ERROR_INVALID_PTX at cuModuleLoadData.

#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "nvfp4_utils.cuh"

#ifndef FA2_BC
#  define FA2_BC 32
#endif
#define FA2_THREADS 128

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(
    float val, float* smem_reduce, int tid, int num_threads
) {
    int warp_id   = tid >> 5;
    int lane_id   = tid & 31;
    int num_warps = (num_threads + 31) >> 5;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem_reduce[warp_id] = val;
    __syncthreads();
    val = (tid < num_warps) ? smem_reduce[tid] : 0.0f;
    if (tid < 32) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ float fp8kv_decode_byte(unsigned char b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    __half_raw hr = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
    return __half2float(__half(hr));
#else
    unsigned int s = (b >> 7) & 1u;
    unsigned int e = (b >> 3) & 0xFu;
    unsigned int m = b & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
#endif
}

// Helper for empty-partition / all-masked-partition writes. CUDA
// device lambdas need `--extended-lambda`; we don't pass it, so this
// is a named function instead.
__device__ __forceinline__ void write_split_empty_partition(
    __half* __restrict__ tmp_out,
    float*  __restrict__ max_logits,
    float*  __restrict__ exp_sums,
    int scratch_idx,
    int tmp_out_row,
    int head_dim,
    int tid
) {
    if (tid == 0) {
        max_logits[scratch_idx] = -FLT_MAX;
        exp_sums  [scratch_idx] = 0.0f;
    }
    for (int d = tid; d < head_dim; d += FA2_THREADS) {
        tmp_out[tmp_out_row + d] = __float2half(0.0f);
    }
}

__device__ __forceinline__ void dequant_nvfp4_row_to_smem(
    const uint8_t*       __restrict__ packed_row,
    const __nv_fp8_e4m3* __restrict__ scale_row,
    __half*              __restrict__ s_dst,
    int tid,
    int head_dim
) {
    const int num_blocks = head_dim >> 4;
    if (tid < num_blocks) {
        const int block_idx = tid;
        const int d_base    = block_idx << 4;
        uint64_t packed8 = *reinterpret_cast<const uint64_t*>(packed_row + (d_base >> 1));
        float    scale   = float(scale_row[block_idx]);
        rvllm_nvfp4::unpack16_nvfp4_to_f16_fast(packed8, scale, s_dst + d_base);
    }
}

// ---------------------------------------------------------------------
// Split-decode kernel.
//   Grid : (num_seqs, num_heads, max_num_partitions)
//   Block: (FA2_THREADS, 1, 1)
//   Smem : 2 * FA2_BC * head_dim * 2  (K+V f16 tiles)
//        + FA2_BC * 4                  (scores f32)
//        + (FA2_THREADS/32) * 4        (reduce scratch)
//   = 2*FA2_BC*head_dim*2 + (FA2_BC + FA2_THREADS/32) * 4
// ---------------------------------------------------------------------
extern "C"
__global__ void flash_attention_2_decode_nvfp4kv_split_kernel(
    __half*              __restrict__ tmp_out,          // [S, H, P, D] f16
    float*               __restrict__ max_logits,       // [S, H, P]    f32
    float*               __restrict__ exp_sums,         // [S, H, P]    f32
    const unsigned char* __restrict__ query,            // [S, H, D]    fp8
    const uint8_t*       __restrict__ key_cache_packed,
    const uint8_t*       __restrict__ value_cache_packed,
    const __nv_fp8_e4m3* __restrict__ key_cache_scale,
    const __nv_fp8_e4m3* __restrict__ value_cache_scale,
    const int*           __restrict__ block_tables,
    const int*           __restrict__ context_lens,
    const float*         __restrict__ q_descale,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int window_size_left,
    int partition_size,
    int max_num_partitions
) {
    const int seq_idx      = blockIdx.x;
    const int head_idx     = blockIdx.y;
    const int partition_id = blockIdx.z;
    const int tid          = threadIdx.x;

    const int context_len = context_lens[seq_idx];

    // Global addressing into scratch tensors. Every partition writes
    // its own slot; empty partitions still write sentinels so the
    // reduce kernel's `num_partitions = ceil(ctx/P)` loop doesn't
    // need to know which specific partitions were active.
    const int scratch_idx = (seq_idx * num_heads + head_idx) * max_num_partitions
                          + partition_id;
    const int tmp_out_row = scratch_idx * head_dim;

    // (lambda removed — nvcc not compiled with --extended-lambda;
    //  use `write_split_empty_partition(...)` calls instead.)

    // Empty context.
    if (context_len == 0) {
        write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid);
        return;
    }

    // Partition's token range [part_start, part_end).
    const int part_start = partition_id * partition_size;
    const int part_end   = min(part_start + partition_size, context_len);
    if (part_start >= context_len) {
        write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid);
        return;
    }

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const float q_scale = *q_descale;

    // Sliding-window bounds (same formula as the non-split kernel).
    const int decode_q_abs_pos = context_len - 1;
    const int window_start = (window_size_left < 0)
        ? 0
        : max(0, decode_q_abs_pos - window_size_left);

    // Whole-partition fast mask: if even the last element of the
    // partition is before `window_start`, the partition contributes
    // nothing.
    if (part_end <= window_start) {
        write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid);
        return;
    }

    // Smem layout — same as non-split decode.
    extern __shared__ unsigned char smem_u8[];
    __half* s_key   = reinterpret_cast<__half*>(smem_u8);
    __half* s_val   = s_key + FA2_BC * head_dim;
    float*  s_score  = reinterpret_cast<float*>(s_val + FA2_BC * head_dim);
    float*  s_reduce = s_score + FA2_BC;

    const int tile_lo = part_start / FA2_BC;
    const int tile_hi = (part_end + FA2_BC - 1) / FA2_BC;
    // If the sliding-window prunes leading tiles inside this
    // partition, start past them. This mirrors the non-split kernel's
    // tile_start_idx logic.
    const int ws_tile_lo = window_start / FA2_BC;
    const int first_tile = max(tile_lo, ws_tile_lo);

    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
    const int half_D      = head_dim >> 1;
    const int scales_per_D = head_dim >> 4;

    // Load Q row (same as non-split).
    float q_reg[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = tid + r * FA2_THREADS;
        if (r < dims_per_thread && d < head_dim) {
            unsigned char qb = query[(seq_idx * num_heads + head_idx) * head_dim + d];
            q_reg[r] = fp8kv_decode_byte(qb) * q_scale * scale;
        } else {
            q_reg[r] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = 0.0f;

    for (int tile = first_tile; tile < tile_hi; ++tile) {
        const int tile_start = tile * FA2_BC;
        const int tile_end   = min((tile + 1) * FA2_BC, part_end);
        const int tile_len   = tile_end - tile_start;
        if (tile_len <= 0) continue;

        // K tile dequant.
        for (int t = 0; t < tile_len; ++t) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const uint8_t*       k_packed = key_cache_packed
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D;
            const __nv_fp8_e4m3* k_scale  = key_cache_scale
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D;
            dequant_nvfp4_row_to_smem(k_packed, k_scale, s_key + t * head_dim,
                                      tid, head_dim);
        }
        __syncthreads();

        // Q·K^T per column (with sliding-window mask on edge).
        for (int t = 0; t < tile_len; ++t) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = tid + r * FA2_THREADS;
                if (r < dims_per_thread && d < head_dim) {
                    dot += q_reg[r] * __half2float(s_key[t * head_dim + d]);
                }
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) {
                int kv_pos = tile_start + t;
                s_score[t] = (kv_pos < window_start) ? -FLT_MAX : dot;
            }
            __syncthreads();
        }

        // Online-softmax update inside partition.
        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; ++t)
                tile_max = fmaxf(tile_max, s_score[t]);
            s_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_reduce[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max  = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            #pragma unroll
            for (int r = 0; r < 8; ++r) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; ++t) {
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

        // V tile dequant.
        for (int t = 0; t < tile_len; ++t) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const uint8_t*       v_packed = value_cache_packed
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D;
            const __nv_fp8_e4m3* v_scale  = value_cache_scale
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D;
            dequant_nvfp4_row_to_smem(v_packed, v_scale, s_val + t * head_dim,
                                      tid, head_dim);
        }
        __syncthreads();

        // P · V accumulate.
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; ++t) {
                    val_acc += s_score[t] * __half2float(s_val[t * head_dim + d]);
                }
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    // If the partition had NO unmasked rows (e.g. sliding-window
    // inside the partition left everything masked), row_sum stays 0.
    // Emit the same sentinel as the early-out path so the reduce
    // kernel ignores us.
    if (row_sum <= 0.0f) {
        write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid);
        return;
    }

    // Per-partition normalize — divide the partial output by row_sum
    // BEFORE writing tmp_out. The reduce kernel's reweighting relies
    // on this invariant (vLLM does the same, see paged_attention_kernel.
    // line ~339).
    float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = tid + r * FA2_THREADS;
        if (r < dims_per_thread && d < head_dim) {
            tmp_out[tmp_out_row + d] = __float2half(acc[r] * inv_sum);
        }
    }
    if (tid == 0) {
        max_logits[scratch_idx] = row_max;
        exp_sums  [scratch_idx] = row_sum;
    }
}

// ---------------------------------------------------------------------
// BC=16 variant for head_dim=512 (doesn't fit 2*32*512*2 smem budget).
// Body identical; FA2_BC macro switched.
// ---------------------------------------------------------------------
#if FA2_BC == 32
#  undef FA2_BC
#  define FA2_BC 16

extern "C"
__global__ void flash_attention_2_decode_nvfp4kv_split_bc16_kernel(
    __half*              __restrict__ tmp_out,
    float*               __restrict__ max_logits,
    float*               __restrict__ exp_sums,
    const unsigned char* __restrict__ query,
    const uint8_t*       __restrict__ key_cache_packed,
    const uint8_t*       __restrict__ value_cache_packed,
    const __nv_fp8_e4m3* __restrict__ key_cache_scale,
    const __nv_fp8_e4m3* __restrict__ value_cache_scale,
    const int*           __restrict__ block_tables,
    const int*           __restrict__ context_lens,
    const float*         __restrict__ q_descale,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int window_size_left,
    int partition_size,
    int max_num_partitions
) {
    const int seq_idx      = blockIdx.x;
    const int head_idx     = blockIdx.y;
    const int partition_id = blockIdx.z;
    const int tid          = threadIdx.x;

    const int context_len = context_lens[seq_idx];

    const int scratch_idx = (seq_idx * num_heads + head_idx) * max_num_partitions
                          + partition_id;
    const int tmp_out_row = scratch_idx * head_dim;

    // (lambda removed — use write_split_empty_partition(...) instead.)

    if (context_len == 0) { write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid); return; }

    const int part_start = partition_id * partition_size;
    const int part_end   = min(part_start + partition_size, context_len);
    if (part_start >= context_len) { write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid); return; }

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const float q_scale = *q_descale;
    const int decode_q_abs_pos = context_len - 1;
    const int window_start = (window_size_left < 0)
        ? 0 : max(0, decode_q_abs_pos - window_size_left);
    if (part_end <= window_start) { write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid); return; }

    extern __shared__ unsigned char smem_u8[];
    __half* s_key   = reinterpret_cast<__half*>(smem_u8);
    __half* s_val   = s_key + FA2_BC * head_dim;
    float*  s_score  = reinterpret_cast<float*>(s_val + FA2_BC * head_dim);
    float*  s_reduce = s_score + FA2_BC;

    const int tile_lo = part_start / FA2_BC;
    const int tile_hi = (part_end + FA2_BC - 1) / FA2_BC;
    const int ws_tile_lo = window_start / FA2_BC;
    const int first_tile = max(tile_lo, ws_tile_lo);

    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
    const int half_D      = head_dim >> 1;
    const int scales_per_D = head_dim >> 4;

    float q_reg[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = tid + r * FA2_THREADS;
        if (r < dims_per_thread && d < head_dim) {
            unsigned char qb = query[(seq_idx * num_heads + head_idx) * head_dim + d];
            q_reg[r] = fp8kv_decode_byte(qb) * q_scale * scale;
        } else {
            q_reg[r] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = 0.0f;

    for (int tile = first_tile; tile < tile_hi; ++tile) {
        const int tile_start = tile * FA2_BC;
        const int tile_end   = min((tile + 1) * FA2_BC, part_end);
        const int tile_len   = tile_end - tile_start;
        if (tile_len <= 0) continue;

        for (int t = 0; t < tile_len; ++t) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const uint8_t*       k_packed = key_cache_packed
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D;
            const __nv_fp8_e4m3* k_scale  = key_cache_scale
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D;
            dequant_nvfp4_row_to_smem(k_packed, k_scale, s_key + t * head_dim,
                                      tid, head_dim);
        }
        __syncthreads();

        for (int t = 0; t < tile_len; ++t) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = tid + r * FA2_THREADS;
                if (r < dims_per_thread && d < head_dim) {
                    dot += q_reg[r] * __half2float(s_key[t * head_dim + d]);
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
            for (int t = 0; t < tile_len; ++t)
                tile_max = fmaxf(tile_max, s_score[t]);
            s_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_reduce[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max  = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            #pragma unroll
            for (int r = 0; r < 8; ++r) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; ++t) {
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

        for (int t = 0; t < tile_len; ++t) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const uint8_t*       v_packed = value_cache_packed
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D;
            const __nv_fp8_e4m3* v_scale  = value_cache_scale
                + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D;
            dequant_nvfp4_row_to_smem(v_packed, v_scale, s_val + t * head_dim,
                                      tid, head_dim);
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; ++t) {
                    val_acc += s_score[t] * __half2float(s_val[t * head_dim + d]);
                }
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    if (row_sum <= 0.0f) { write_split_empty_partition(tmp_out, max_logits, exp_sums, scratch_idx, tmp_out_row, head_dim, tid); return; }

    float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = tid + r * FA2_THREADS;
        if (r < dims_per_thread && d < head_dim) {
            tmp_out[tmp_out_row + d] = __float2half(acc[r] * inv_sum);
        }
    }
    if (tid == 0) {
        max_logits[scratch_idx] = row_max;
        exp_sums  [scratch_idx] = row_sum;
    }
}

#  undef FA2_BC
#  define FA2_BC 32
#endif

// ---------------------------------------------------------------------
// Reduce kernel — combine per-partition results into final output.
//   Grid : (num_seqs, num_heads)
//   Block: (REDUCE_THREADS, 1, 1) — one block handles one (seq, head).
//   Smem : 2 * max_num_partitions * 4   (max_logits + rescaled exp_sums)
//        + (REDUCE_THREADS/32) * 4      (reduce scratch)
//
// Head-dtype agnostic — takes f16 tmp_out, writes f16 output. Phase-1
// guarantees tmp_out is already divided by the partition's own
// exp_sum, so we only reweight with the online-softmax combine:
//   w[p]   = exp_sums[p] * exp(max_logits[p] - global_max)
//   inv    = 1 / (sum_p(w[p]) + 1e-6)
//   out[d] = sum_p(tmp_out[p, d] * w[p] * inv)
//
// Short-circuits when num_partitions == 1 (copy-only).
// ---------------------------------------------------------------------
#define REDUCE_THREADS 128

extern "C"
__global__ void paged_attention_reduce_f16_kernel(
    __half*       __restrict__ output,         // [S, H, D]
    const __half* __restrict__ tmp_out,        // [S, H, P, D]
    const float*  __restrict__ max_logits,     // [S, H, P]
    const float*  __restrict__ exp_sums,       // [S, H, P]
    const int*    __restrict__ context_lens,   // [S]
    int num_heads,
    int head_dim,
    int max_num_partitions,
    int partition_size
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) {
        for (int d = tid; d < head_dim; d += REDUCE_THREADS) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d]
                = __float2half(0.0f);
        }
        return;
    }

    const int num_partitions =
        (context_len + partition_size - 1) / partition_size;

    const int scratch_base =
        (seq_idx * num_heads + head_idx) * max_num_partitions;
    const int tmp_out_base = scratch_base * head_dim;
    const int out_base     =
        (seq_idx * num_heads + head_idx) * head_dim;

    // num_partitions == 1: pure copy, no combine needed.
    if (num_partitions == 1) {
        for (int d = tid; d < head_dim; d += REDUCE_THREADS) {
            output[out_base + d] = tmp_out[tmp_out_base + d];
        }
        return;
    }

    // Smem: shared max-logits table + rescaled-exp-sums table + reduce
    // scratch. Sizes in the launcher.
    extern __shared__ unsigned char reduce_smem_u8[];
    float* shared_max_logits   = reinterpret_cast<float*>(reduce_smem_u8);
    float* shared_rescaled_sum =
        shared_max_logits + max_num_partitions;
    float* red_smem =
        shared_rescaled_sum + max_num_partitions;

    // Load max_logits into smem + compute global_max.
    float global_max = -FLT_MAX;
    for (int p = tid; p < num_partitions; p += REDUCE_THREADS) {
        float l = max_logits[scratch_base + p];
        shared_max_logits[p] = l;
        // -FLT_MAX on empty / masked partitions — max stays correct.
        global_max = fmaxf(global_max, l);
    }
    // Block-reduce max across threads.
    {
        int warp_id = tid >> 5;
        int lane    = tid & 31;
        int nw      = REDUCE_THREADS / 32;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            global_max = fmaxf(global_max,
                __shfl_xor_sync(0xFFFFFFFFu, global_max, o));
        }
        if (lane == 0) red_smem[warp_id] = global_max;
        __syncthreads();
        global_max = (tid < nw) ? red_smem[tid] : -FLT_MAX;
        if (tid < 32) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                global_max = fmaxf(global_max,
                    __shfl_xor_sync(0xFFFFFFFFu, global_max, o));
            }
            if (tid == 0) red_smem[0] = global_max;
        }
        __syncthreads();
        global_max = red_smem[0];
    }

    // If ALL partitions were sentinels (-FLT_MAX), the numerator is
    // zero everywhere. Write zeros and bail.
    if (global_max == -FLT_MAX) {
        for (int d = tid; d < head_dim; d += REDUCE_THREADS) {
            output[out_base + d] = __float2half(0.0f);
        }
        return;
    }

    // Compute rescaled exp sums + global sum.
    float global_exp_sum = 0.0f;
    for (int p = tid; p < num_partitions; p += REDUCE_THREADS) {
        float ml = shared_max_logits[p];
        float es = exp_sums[scratch_base + p];
        float rescaled = (ml == -FLT_MAX) ? 0.0f : es * expf(ml - global_max);
        shared_rescaled_sum[p] = rescaled;
        global_exp_sum += rescaled;
    }
    // Block-reduce the sum.
    {
        int warp_id = tid >> 5;
        int lane    = tid & 31;
        int nw      = REDUCE_THREADS / 32;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            global_exp_sum += __shfl_xor_sync(0xFFFFFFFFu, global_exp_sum, o);
        }
        if (lane == 0) red_smem[warp_id] = global_exp_sum;
        __syncthreads();
        global_exp_sum = (tid < nw) ? red_smem[tid] : 0.0f;
        if (tid < 32) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                global_exp_sum += __shfl_xor_sync(0xFFFFFFFFu, global_exp_sum, o);
            }
            if (tid == 0) red_smem[0] = global_exp_sum;
        }
        __syncthreads();
        global_exp_sum = red_smem[0];
    }

    const float inv_global = 1.0f / (global_exp_sum + 1e-6f);

    // Combine. Each thread owns a slice of head_dim.
    for (int d = tid; d < head_dim; d += REDUCE_THREADS) {
        float acc = 0.0f;
        for (int p = 0; p < num_partitions; ++p) {
            float w = shared_rescaled_sum[p] * inv_global;
            float v = __half2float(tmp_out[tmp_out_base + p * head_dim + d]);
            acc += v * w;
        }
        output[out_base + d] = __float2half(acc);
    }
}
