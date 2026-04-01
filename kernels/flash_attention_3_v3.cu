// FlashAttention-3 v3: split-KV + cp.async double-buffered + warp-parallel.
//
// v3 improvements over v2:
//   1. Split-KV: distributes KV tiles across multiple thread blocks per (seq, kv_head).
//      Each block writes partial output to workspace; combine kernel reduces them.
//      Heuristic: 1 split (<=512 ctx), 2 (<=2048), 4 (<=8192), 8 (>8192).
//   2. cp.async: asynchronous global-to-shared copies (SM 8.0+). Double-buffered
//      KV tiles overlap next-tile K prefetch with current-tile PV computation.
//      128-bit (16-byte) bulk copies bypass registers and L1 cache.
//   3. Cooperative warp pipelining: all 8 warps participate in both cp.async loads
//      and compute. V load issued before softmax so HBM transfer overlaps with
//      softmax reduction. K[tile+1] prefetch overlaps with PV accumulation.
//      Full producer/consumer warp specialization would reduce occupancy on this
//      memory-bound kernel (2 blocks/SM -> 1), so cooperative is better.
//
// Shared memory: 2x KV tile buffers (one for K, one for V). Scores array and
// warp reduction scratch. Double-buffered: ~35KB per block at head_dim=128.
// Still fits 2 blocks/SM on H100 (228KB smem total).
//
// GQA kernel:   grid(num_seqs, num_kv_heads, num_splits), block(256)
// Non-GQA:      grid(num_seqs, num_heads, num_splits), block(256)
// Combine:      grid(num_seqs, num_heads, 1), block(head_dim)

#include <float.h>
#include <cuda_fp16.h>

#define V3_BC 64
#define V3_THREADS 256
#define V3_WARPS 8
#define V3_GQA_MAX_HPG 16
#define V3_SCORE_STRIDE (V3_BC + 1)
#define V3_CHUNK 8  // f16 elements per cp.async (16 bytes)

// ======================================================================
// cp.async intrinsics (SM 8.0+)
// ======================================================================

// Copy 16 bytes from global to shared, bypassing L1 (cache-global)
__device__ __forceinline__ void v3_cp_async_16(void* smem, const void* gmem) {
    unsigned smem_addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem)
    );
}

__device__ __forceinline__ void v3_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// Wait until ALL pending async groups complete
__device__ __forceinline__ void v3_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

// Wait until at most 1 group pending (oldest group completes)
__device__ __forceinline__ void v3_cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;\n");
}

// ======================================================================
// Warp/block reductions
// ======================================================================

__device__ __forceinline__ float v3_warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float v3_warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float v3_block_reduce_max(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = v3_warp_max(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float m = s_warp[0];
        for (int w = 1; w < V3_WARPS; w++) m = fmaxf(m, s_warp[w]);
        s_warp[0] = m;
    }
    __syncthreads();
    return s_warp[0];
}

__device__ __forceinline__ float v3_block_reduce_sum(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = v3_warp_sum(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float s = s_warp[0];
        for (int w = 1; w < V3_WARPS; w++) s += s_warp[w];
        s_warp[0] = s;
    }
    __syncthreads();
    return s_warp[0];
}

// ======================================================================
// Async KV tile loader via cp.async
// ======================================================================

__device__ __forceinline__ void v3_async_load_tile(
    __half* s_buf,
    const __half* cache,
    const int* block_tables, int seq_idx, int max_blocks,
    int tile_start, int tile_len, int num_kv_heads, int kv_head_idx,
    int head_dim, int block_size, int tid
) {
    const int chunks_per_row = head_dim / V3_CHUNK;
    const int total_chunks = tile_len * chunks_per_row;

    for (int c = tid; c < total_chunks; c += V3_THREADS) {
        int t = c / chunks_per_row;
        int ch = c % chunks_per_row;
        int kv_pos = tile_start + t;
        int page_idx = kv_pos / block_size;
        int page_off = kv_pos % block_size;
        int phys_block = __ldg(&block_tables[seq_idx * max_blocks + page_idx]);

        const __half* src = &cache[((phys_block * block_size + page_off)
                                    * num_kv_heads + kv_head_idx) * head_dim + ch * V3_CHUNK];
        __half* dst = &s_buf[t * head_dim + ch * V3_CHUNK];

        v3_cp_async_16(dst, src);
    }
}

// ======================================================================
// GQA decode kernel: split-KV + double-buffered cp.async + warp-parallel
//
// Double-buffered pipeline per tile:
//   1. QK^T on s_K (K already loaded)
//   2. Issue async V load -> s_V (overlaps with softmax)
//   3. Online softmax (no KV buffer access)
//   4. Wait for V
//   5. Issue async K[next] load -> s_K (overlaps with P@V)
//   6. P@V on s_V
//   7. Sync (K[next] ready for next iteration)
// ======================================================================

extern "C"
__global__ void __launch_bounds__(V3_THREADS, 2)
fa3_v3_decode_gqa_kernel(
    __half* __restrict__ output,          // [num_seqs, num_heads, head_dim] (single-split)
    float* __restrict__ partial_out,      // [num_splits, num_seqs, num_heads, head_dim]
    float* __restrict__ partial_max,      // [num_splits, num_seqs, num_heads]
    float* __restrict__ partial_sum,      // [num_splits, num_seqs, num_heads]
    const __half* __restrict__ query,     // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int num_splits
) {
    const int seq_idx     = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int split_idx   = blockIdx.z;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int heads_per_group = num_heads / num_kv_heads;
    const int total_tiles = (context_len + V3_BC - 1) / V3_BC;
    const int half2_iters = (head_dim + 63) / 64;
    const int acc_dims = (head_dim + V3_THREADS - 1) / V3_THREADS;

    // Split-KV: this block's tile range
    const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    const int start_tile = split_idx * tiles_per_split;
    const int end_tile = min(start_tile + tiles_per_split, total_tiles);

    if (start_tile >= total_tiles) {
        // Empty split -- write sentinels for combine
        if (num_splits > 1) {
            for (int g = 0; g < heads_per_group && g < V3_GQA_MAX_HPG; g++) {
                int g_head = kv_head_idx * heads_per_group + g;
                int ws = (split_idx * gridDim.x + seq_idx) * num_heads + g_head;
                if (tid == 0) {
                    partial_max[ws] = -FLT_MAX;
                    partial_sum[ws] = 0.0f;
                }
                for (int r = 0; r < acc_dims && r < 4; r++) {
                    int d = tid + r * V3_THREADS;
                    if (d < head_dim) partial_out[ws * head_dim + d] = 0.0f;
                }
            }
        }
        return;
    }

    // ---- Double-buffered shared memory: separate K and V buffers ----
    extern __shared__ char smem_raw[];
    __half* s_K     = (__half*)smem_raw;
    __half* s_V     = s_K + V3_BC * head_dim;
    float* s_scores = (float*)(s_V + V3_BC * head_dim);
    float* s_warp   = s_scores + V3_GQA_MAX_HPG * V3_SCORE_STRIDE;

    // ---- Pre-load Q vectors for all heads in group ----
    float q_regs[V3_GQA_MAX_HPG][4];
    float head_row_max[V3_GQA_MAX_HPG];
    float head_row_sum[V3_GQA_MAX_HPG];
    float head_acc[V3_GQA_MAX_HPG][4];

    for (int g = 0; g < heads_per_group && g < V3_GQA_MAX_HPG; g++) {
        int g_head = kv_head_idx * heads_per_group + g;
        int q_base = (seq_idx * num_heads + g_head) * head_dim;
        #pragma unroll
        for (int r = 0; r < half2_iters && r < 2; r++) {
            int d = lane_id * 2 + r * 64;
            if (d + 1 < head_dim) {
                q_regs[g][r*2]   = __half2float(query[q_base + d]) * scale;
                q_regs[g][r*2+1] = __half2float(query[q_base + d + 1]) * scale;
            } else if (d < head_dim) {
                q_regs[g][r*2]   = __half2float(query[q_base + d]) * scale;
                q_regs[g][r*2+1] = 0.0f;
            } else {
                q_regs[g][r*2]   = 0.0f;
                q_regs[g][r*2+1] = 0.0f;
            }
        }
        head_row_max[g] = -FLT_MAX;
        head_row_sum[g] = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; r++) head_acc[g][r] = 0.0f;
    }

    // ---- Prologue: prefetch K[start_tile] into s_K ----
    {
        const int tile_start = start_tile * V3_BC;
        const int tile_len = min(V3_BC, context_len - tile_start);
        v3_async_load_tile(s_K, key_cache, block_tables, seq_idx, max_blocks_per_seq,
                           tile_start, tile_len, num_kv_heads, kv_head_idx,
                           head_dim, block_size, tid);
        v3_cp_async_commit();
        v3_cp_async_wait_all();
        __syncthreads();
    }

    for (int tile = start_tile; tile < end_tile; tile++) {
        const int tile_start = tile * V3_BC;
        const int tile_len = min(V3_BC, context_len - tile_start);

        // ---- 1. QK^T on s_K (already loaded) ----
        for (int g = 0; g < heads_per_group && g < V3_GQA_MAX_HPG; g++) {
            float* g_scores = s_scores + g * V3_SCORE_STRIDE;

            for (int base_t = 0; base_t < tile_len; base_t += V3_WARPS) {
                int t = base_t + warp_id;
                if (t < tile_len) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < half2_iters && r < 2; r++) {
                        int d = lane_id * 2 + r * 64;
                        if (d + 1 < head_dim) {
                            __half2 kv = *reinterpret_cast<const __half2*>(&s_K[t * head_dim + d]);
                            dot += q_regs[g][r*2]   * __half2float(kv.x);
                            dot += q_regs[g][r*2+1] * __half2float(kv.y);
                        } else if (d < head_dim) {
                            dot += q_regs[g][r*2] * __half2float(s_K[t * head_dim + d]);
                        }
                    }
                    dot = v3_warp_sum(dot);
                    if (lane_id == 0) g_scores[t] = dot;
                }
            }
            __syncthreads();
        }

        // ---- 2. Issue async V load into s_V (overlaps with softmax) ----
        v3_async_load_tile(s_V, value_cache, block_tables, seq_idx, max_blocks_per_seq,
                           tile_start, tile_len, num_kv_heads, kv_head_idx,
                           head_dim, block_size, tid);
        v3_cp_async_commit();

        // ---- 3. Online softmax (no KV buffer access -- V loading in background) ----
        for (int g = 0; g < heads_per_group && g < V3_GQA_MAX_HPG; g++) {
            float* g_scores = s_scores + g * V3_SCORE_STRIDE;

            float tile_max = v3_block_reduce_max(
                (tid < tile_len) ? g_scores[tid] : -FLT_MAX,
                tid, lane_id, warp_id, s_warp);

            float prev_max = head_row_max[g];
            float new_max = fmaxf(prev_max, tile_max);
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                #pragma unroll
                for (int r = 0; r < acc_dims && r < 4; r++) head_acc[g][r] *= correction;
                head_row_sum[g] *= correction;
            }
            head_row_max[g] = new_max;

            float my_exp = (tid < tile_len) ? expf(g_scores[tid] - new_max) : 0.0f;
            if (tid < tile_len) g_scores[tid] = my_exp;
            head_row_sum[g] += v3_block_reduce_sum(my_exp, tid, lane_id, warp_id, s_warp);
        }

        // ---- 4. Wait for V load to complete ----
        v3_cp_async_wait_all();
        __syncthreads();

        // ---- 5. Issue async K[next] prefetch into s_K (overlaps with P@V) ----
        if (tile + 1 < end_tile) {
            const int next_start = (tile + 1) * V3_BC;
            const int next_len = min(V3_BC, context_len - next_start);
            v3_async_load_tile(s_K, key_cache, block_tables, seq_idx, max_blocks_per_seq,
                               next_start, next_len, num_kv_heads, kv_head_idx,
                               head_dim, block_size, tid);
            v3_cp_async_commit();
        }

        // ---- 6. P@V on s_V, with V reuse across heads ----
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * V3_THREADS;
            if (d < head_dim) {
                for (int t = 0; t < tile_len; t++) {
                    float v = __half2float(s_V[t * head_dim + d]);
                    for (int g = 0; g < heads_per_group && g < V3_GQA_MAX_HPG; g++)
                        head_acc[g][r] += s_scores[g * V3_SCORE_STRIDE + t] * v;
                }
            }
        }

        // ---- 7. Wait for K[next] prefetch before next iteration ----
        if (tile + 1 < end_tile) {
            v3_cp_async_wait_all();
        }
        __syncthreads();
    }

    // ---- Write output ----
    if (num_splits == 1) {
        // Single split: normalize and write f16 directly
        for (int g = 0; g < heads_per_group && g < V3_GQA_MAX_HPG; g++) {
            int g_head = kv_head_idx * heads_per_group + g;
            float inv = (head_row_sum[g] > 0.0f) ? (1.0f / head_row_sum[g]) : 0.0f;
            int out_base = (seq_idx * num_heads + g_head) * head_dim;
            #pragma unroll
            for (int r = 0; r < acc_dims && r < 4; r++) {
                int d = tid + r * V3_THREADS;
                if (d < head_dim)
                    output[out_base + d] = __float2half(head_acc[g][r] * inv);
            }
        }
    } else {
        // Multi-split: write unnormalized f32 partials to workspace
        int num_seqs = gridDim.x;
        for (int g = 0; g < heads_per_group && g < V3_GQA_MAX_HPG; g++) {
            int g_head = kv_head_idx * heads_per_group + g;
            int ws = (split_idx * num_seqs + seq_idx) * num_heads + g_head;
            #pragma unroll
            for (int r = 0; r < acc_dims && r < 4; r++) {
                int d = tid + r * V3_THREADS;
                if (d < head_dim)
                    partial_out[ws * head_dim + d] = head_acc[g][r];
            }
            if (tid == 0) {
                partial_max[ws] = head_row_max[g];
                partial_sum[ws] = head_row_sum[g];
            }
        }
    }
}

// ======================================================================
// Non-GQA decode kernel: split-KV + double-buffered cp.async
// ======================================================================

extern "C"
__global__ void __launch_bounds__(V3_THREADS, 2)
fa3_v3_decode_kernel(
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
    const int warp_id   = tid / 32;
    const int lane_id   = tid % 32;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const int total_tiles = (context_len + V3_BC - 1) / V3_BC;
    const int half2_iters = (head_dim + 63) / 64;
    const int acc_dims = (head_dim + V3_THREADS - 1) / V3_THREADS;

    const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    const int start_tile = split_idx * tiles_per_split;
    const int end_tile = min(start_tile + tiles_per_split, total_tiles);

    if (start_tile >= total_tiles) {
        if (num_splits > 1) {
            int ws = (split_idx * gridDim.x + seq_idx) * num_heads + head_idx;
            if (tid == 0) {
                partial_max[ws] = -FLT_MAX;
                partial_sum[ws] = 0.0f;
            }
            for (int r = 0; r < acc_dims && r < 4; r++) {
                int d = tid + r * V3_THREADS;
                if (d < head_dim) partial_out[ws * head_dim + d] = 0.0f;
            }
        }
        return;
    }

    // Double-buffered shared memory
    extern __shared__ char smem_raw[];
    __half* s_K    = (__half*)smem_raw;
    __half* s_V    = s_K + V3_BC * head_dim;
    float* s_score = (float*)(s_V + V3_BC * head_dim);
    float* s_warp  = s_score + V3_BC;

    // Load Q into registers
    const int q_base = (seq_idx * num_heads + head_idx) * head_dim;
    float q_reg[4];
    #pragma unroll
    for (int r = 0; r < half2_iters && r < 2; r++) {
        int d = lane_id * 2 + r * 64;
        if (d + 1 < head_dim) {
            q_reg[r*2]   = __half2float(query[q_base + d]) * scale;
            q_reg[r*2+1] = __half2float(query[q_base + d + 1]) * scale;
        } else if (d < head_dim) {
            q_reg[r*2]   = __half2float(query[q_base + d]) * scale;
            q_reg[r*2+1] = 0.0f;
        } else {
            q_reg[r*2]   = 0.0f;
            q_reg[r*2+1] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) acc[r] = 0.0f;

    // Prologue: prefetch K[start_tile]
    {
        const int tile_start = start_tile * V3_BC;
        const int tile_len = min(V3_BC, context_len - tile_start);
        v3_async_load_tile(s_K, key_cache, block_tables, seq_idx, max_blocks_per_seq,
                           tile_start, tile_len, num_kv_heads, kv_head_idx,
                           head_dim, block_size, tid);
        v3_cp_async_commit();
        v3_cp_async_wait_all();
        __syncthreads();
    }

    for (int tile = start_tile; tile < end_tile; tile++) {
        const int tile_start = tile * V3_BC;
        const int tile_len = min(V3_BC, context_len - tile_start);

        // 1. QK^T on s_K
        for (int base_t = 0; base_t < tile_len; base_t += V3_WARPS) {
            int t = base_t + warp_id;
            if (t < tile_len) {
                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < half2_iters && r < 2; r++) {
                    int d = lane_id * 2 + r * 64;
                    if (d + 1 < head_dim) {
                        __half2 kv = *reinterpret_cast<const __half2*>(&s_K[t * head_dim + d]);
                        dot += q_reg[r*2]   * __half2float(kv.x);
                        dot += q_reg[r*2+1] * __half2float(kv.y);
                    } else if (d < head_dim) {
                        dot += q_reg[r*2] * __half2float(s_K[t * head_dim + d]);
                    }
                }
                dot = v3_warp_sum(dot);
                if (lane_id == 0) s_score[t] = dot;
            }
        }
        __syncthreads();

        // 2. Issue async V load (overlaps with softmax)
        v3_async_load_tile(s_V, value_cache, block_tables, seq_idx, max_blocks_per_seq,
                           tile_start, tile_len, num_kv_heads, kv_head_idx,
                           head_dim, block_size, tid);
        v3_cp_async_commit();

        // 3. Online softmax (V loading in background)
        float tile_max = v3_block_reduce_max(
            (tid < tile_len) ? s_score[tid] : -FLT_MAX,
            tid, lane_id, warp_id, s_warp);

        float prev_max = row_max;
        float new_max = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            #pragma unroll
            for (int r = 0; r < acc_dims && r < 4; r++) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        float my_exp = (tid < tile_len) ? expf(s_score[tid] - new_max) : 0.0f;
        if (tid < tile_len) s_score[tid] = my_exp;
        row_sum += v3_block_reduce_sum(my_exp, tid, lane_id, warp_id, s_warp);

        // 4. Wait for V
        v3_cp_async_wait_all();
        __syncthreads();

        // 5. Issue async K[next] prefetch (overlaps with P@V)
        if (tile + 1 < end_tile) {
            const int next_start = (tile + 1) * V3_BC;
            const int next_len = min(V3_BC, context_len - next_start);
            v3_async_load_tile(s_K, key_cache, block_tables, seq_idx, max_blocks_per_seq,
                               next_start, next_len, num_kv_heads, kv_head_idx,
                               head_dim, block_size, tid);
            v3_cp_async_commit();
        }

        // 6. P@V on s_V
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * V3_THREADS;
            if (d < head_dim) {
                float v_acc = 0.0f;
                for (int t = 0; t < tile_len; t++)
                    v_acc += s_score[t] * __half2float(s_V[t * head_dim + d]);
                acc[r] += v_acc;
            }
        }

        // 7. Wait for K[next]
        if (tile + 1 < end_tile) {
            v3_cp_async_wait_all();
        }
        __syncthreads();
    }

    // Write output
    if (num_splits == 1) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        int out_base = (seq_idx * num_heads + head_idx) * head_dim;
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * V3_THREADS;
            if (d < head_dim)
                output[out_base + d] = __float2half(acc[r] * inv_sum);
        }
    } else {
        int ws = (split_idx * gridDim.x + seq_idx) * num_heads + head_idx;
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * V3_THREADS;
            if (d < head_dim)
                partial_out[ws * head_dim + d] = acc[r];
        }
        if (tid == 0) {
            partial_max[ws] = row_max;
            partial_sum[ws] = row_sum;
        }
    }
}

// ======================================================================
// Combine kernel: reduce partial outputs across splits -> f16 output
//
// Grid: (num_seqs, num_heads, 1)
// Block: (head_dim, 1, 1)
// ======================================================================

extern "C"
__global__ void fa3_v3_combine_f16_kernel(
    __half* __restrict__ output,           // [num_seqs, num_heads, head_dim]
    const float* __restrict__ partial_out, // [num_splits, num_seqs, num_heads, head_dim]
    const float* __restrict__ partial_max, // [num_splits, num_seqs, num_heads]
    const float* __restrict__ partial_sum, // [num_splits, num_seqs, num_heads]
    const int* __restrict__ context_lens,
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

    // Global max across all splits
    float global_max = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        int idx = (s * num_seqs + seq_idx) * num_heads + head_idx;
        global_max = fmaxf(global_max, partial_max[idx]);
    }

    if (global_max <= -FLT_MAX + 1.0f) {
        output[(seq_idx * num_heads + head_idx) * head_dim + dim_idx] = __float2half(0.0f);
        return;
    }

    // Combine with online-softmax correction
    float combined_out = 0.0f;
    float combined_sum = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int ws = (s * num_seqs + seq_idx) * num_heads + head_idx;
        float m = partial_max[ws];
        float sm = partial_sum[ws];

        if (sm <= 0.0f) continue;

        float correction = expf(m - global_max);
        combined_out += correction * partial_out[ws * head_dim + dim_idx];
        combined_sum += correction * sm;
    }

    float result = (combined_sum > 0.0f) ? (combined_out / combined_sum) : 0.0f;
    output[(seq_idx * num_heads + head_idx) * head_dim + dim_idx] = __float2half(result);
}
