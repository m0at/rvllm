// Unified FP8-KV multi-query prefill for sm_121.
//
// Port of vLLM's Triton `kernel_unified_attention_2d`
// (`vllm/v1/attention/ops/triton_unified_attention.py`). Replaces the
// per-token decode loop used by the current
// `gemma4_layer_exec::Gemma4Phase::Prefill` path — that loop issues
// `prompt_len` attention launches per layer (~110 k on a 1836-token
// prompt × 60 Gemma 4 layers) and dominates TTFT on sm_121.
//
// See `v3/UNIFIED_PREFILL_SPEC.md` for the algorithm, parameter
// choices, and smem budgeting.
//
// Each `.cu` in `kernels/` compiles to its own PTX module with no link
// step, so `extern __device__` against `flash_attention.cu` wouldn't
// resolve. The helpers we need (block reductions + FP8 decode) are
// small and inlined locally — the duplication is cheap compared to
// merging this kernel into the 2500-line flash_attention.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>

#define FA2_THREADS 128

__device__ __forceinline__ float fp8kv_decode_byte(unsigned char b) {
#if __CUDA_ARCH__ >= 1000
    __half_raw hr = __nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)b, __NV_E4M3);
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

// ROW REDUCE — butterfly within groups of 8 lanes. 128 threads laid out
// as 16 rows × 8 lanes_per_row: thread tid handles row = tid/8,
// lane_in_row = tid%8. Reduces across the 8 lanes of one row with
// offsets 1, 2, 4 — groups of 8 within a warp converge independently,
// so a single full-warp mask is safe.
__device__ __forceinline__ float row_reduce_max_8(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__device__ __forceinline__ float row_reduce_sum_8(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Linear-scan seq_idx finder. `cu_seqlens_q` is a prefix sum of query
// lengths; the grid-block index is `floor(cu_seqlens_q[i] / block_q) + i`
// for each seq i, and the kernel is launched with the upper-bound grid
// size per `unified_attention` in the vLLM host code. A short linear
// scan is faster than binary search for the sub-dozen seq counts we
// hit today; upgrade to binary search when continuous batching lands.
__device__ __forceinline__ int find_seq_idx_linear(
    const int* cu_seqlens_q, int q_block_global, int num_seqs, int block_q
) {
    int found = 0;
    for (int i = 0; i < num_seqs; i++) {
        int start_block = cu_seqlens_q[i] / block_q + i;
        if (start_block <= q_block_global) found = i;
        else break;
    }
    return found;
}

// ============================================================================
// flash_attention_2_prefill_fp8kv_unified_kernel
// ----------------------------------------------------------------------------
// Multi-query prefill. One CUDA block processes BLOCK_M query rows of a
// single KV head for one sequence. Grid dims:
//     gridDim.x = total_num_q_blocks  (host-computed upper bound;
//                                      programs past the per-seq limit
//                                      early-return)
//     gridDim.y = num_kv_heads
//
// Memory layouts (match existing Fa2PtxKernels paged decode):
//     query              [num_tokens, num_heads, head_dim]         FP8 E4M3
//     key_cache          [num_blocks, block_size, num_kv_heads, head_dim] FP8
//     value_cache        [num_blocks, block_size, num_kv_heads, head_dim] FP8
//     k_scale_cache      [num_blocks*block_size, num_kv_heads]     f32
//     v_scale_cache      [num_blocks*block_size, num_kv_heads]     f32
//     q_scale_cache      [num_tokens, num_heads]                   f32
//     block_tables       [num_seqs, max_blocks_per_seq]            i32
//     cu_seqlens_q       [num_seqs+1]                              i32 prefix sum
//     context_lens       [num_seqs]                                i32
//     output             [num_tokens, num_heads, head_dim]         f16
//
// Scope: causal, optional sliding window, no softcap / alibi / sinks /
// mm-prefix / qq-bias (Gemma 4 doesn't use them in attention — logit
// softcap lives at the LM head). num_seqs=1 for v1; the seq-search
// loop is in place so adding continuous batching later is a Rust-side
// change.
//
// smem layout (dynamic):
//     s_q_f32      [BLOCK_M * head_dim]      f32   (pre-scaled Q)
//     s_k_fp8      [head_dim * TILE_SIZE]    u8
//     s_v_fp8      [TILE_SIZE * head_dim]    u8
//     s_k_scale    [TILE_SIZE]               f32
//     s_v_scale    [TILE_SIZE]               f32
//     s_S          [BLOCK_M * TILE_SIZE]     f32
//     s_M          [BLOCK_M]                 f32   (online-softmax max)
//     s_L          [BLOCK_M]                 f32   (online-softmax denom)
//     s_acc        [BLOCK_M * head_dim]      f32
//     s_reduce     [FA2_THREADS / 32]        f32   (block reductions)
// ----------------------------------------------------------------------------

// BLOCK_M is a compile-time constant today: 16 covers every Gemma 4
// layer (num_queries_per_kv ∈ {2, 8} → BLOCK_M = max(16, npo2(q/kv))).
// Hard-code for now; templated variants follow when another model asks
// for them.
#ifndef UNIFIED_PREFILL_BLOCK_M
#define UNIFIED_PREFILL_BLOCK_M 16
#endif

extern "C"
__global__ void flash_attention_2_prefill_fp8kv_unified_kernel(
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
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ context_lens,
    const float* __restrict__ q_descale,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int tile_size,
    int num_queries_per_kv,
    int block_q,          // = BLOCK_M / num_queries_per_kv
    int num_seqs,
    int window_size_left  // -1 ⇒ no sliding window
) {
    // === Program setup ================================================
    const int q_block_global = blockIdx.x;
    const int kv_head        = blockIdx.y;
    const int tid            = threadIdx.x;
    constexpr int BLOCK_M    = UNIFIED_PREFILL_BLOCK_M;

    // Which sequence + local q_block does this program cover?
    const int seq_idx = find_seq_idx_linear(
        cu_seqlens_q, q_block_global, num_seqs, block_q);
    const int seq_q_start = cu_seqlens_q[seq_idx];
    const int seq_q_end   = cu_seqlens_q[seq_idx + 1];
    const int query_len   = seq_q_end - seq_q_start;
    const int q_block_start_global = seq_q_start / block_q + seq_idx;
    const int q_block_local = q_block_global - q_block_start_global;
    // Upper-bound grid — bail out of programs past this seq's end.
    if (q_block_local * block_q >= query_len) return;

    const int seq_len    = context_lens[seq_idx];
    const int prefix_len = seq_len - query_len;

    // Max prefix any query in this block needs (for tile count + bounds
    // check on block_tables). Matches the Triton 2D kernel:
    //     max_seq_prefix_len = prefix_len + q_block_local*block_q
    //                        + (BLOCK_M - 1) / num_queries_per_kv + 1
    //     clamp to seq_len.
    const int q_block_qpos_hi = q_block_local * block_q + (BLOCK_M - 1) / num_queries_per_kv;
    int max_seq_prefix_len = prefix_len + q_block_qpos_hi + 1;
    if (max_seq_prefix_len > seq_len) max_seq_prefix_len = seq_len;

    // Sliding-window tile pruning. When disabled (window_size_left < 0)
    // tile_start stays 0, tile_end is the full range.
    int tile_start = 0;
    int tile_end   = (max_seq_prefix_len + tile_size - 1) / tile_size;
    if (window_size_left > 0) {
        const int qpos_lo = q_block_local * block_q;
        const int qpos_hi = min(q_block_qpos_hi, query_len - 1);
        const int first_allowed = prefix_len + qpos_lo - window_size_left + 1;
        const int last_allowed  = prefix_len + qpos_hi;
        int ts = first_allowed / tile_size;
        if (first_allowed < 0) ts = 0;
        if (ts < 0) ts = 0;
        int te = last_allowed / tile_size + 1;
        if (te > tile_end) te = tile_end;
        tile_start = ts;
        tile_end   = te;
    }

    // === Shared memory layout =========================================
    // Runtime sizes — we can't use fixed constexpr because head_dim
    // varies per layer type (256 sliding vs 512 global). Host passes
    // a matching `dynamic_smem_bytes` via cuFuncSetAttribute.
    extern __shared__ unsigned char smem_raw[];
    float*         s_q       = reinterpret_cast<float*>(smem_raw);
    unsigned char* s_k_fp8   = reinterpret_cast<unsigned char*>(s_q + BLOCK_M * head_dim);
    unsigned char* s_v_fp8   = s_k_fp8 + head_dim * tile_size;
    float*         s_k_scale = reinterpret_cast<float*>(s_v_fp8 + tile_size * head_dim);
    float*         s_v_scale = s_k_scale + tile_size;
    float*         s_s       = s_v_scale + tile_size;
    float*         s_m       = s_s + BLOCK_M * tile_size;
    float*         s_l       = s_m + BLOCK_M;
    float*         s_alpha   = s_l + BLOCK_M;
    float*         s_acc     = s_alpha + BLOCK_M;

    const int row        = tid / 8;
    const int lane_in_row = tid & 7;

    // === Load Q into smem =============================================
    // Pre-scale by q_scale × softmax_scale so the inner dot product is
    // a pure sum of products. Matches the decode kernel's approach.
    for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
        const int m = idx / head_dim;
        const int d = idx - m * head_dim;
        const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
        const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
        if (q_pos_in_seq >= query_len || q_head >= num_heads) {
            s_q[idx] = 0.0f;
            continue;
        }
        const int tok = seq_q_start + q_pos_in_seq;
        const float q_scale = (q_scale_cache != nullptr)
            ? q_scale_cache[tok * num_heads + q_head]
            : *q_descale;
        const float qs = q_scale * scale;
        const unsigned char qb =
            query[(tok * num_heads + q_head) * head_dim + d];
        s_q[idx] = fp8kv_decode_byte(qb) * qs;
    }

    // === Init online-softmax state ===================================
    if (tid < BLOCK_M) {
        s_m[tid]     = -FLT_MAX;
        s_l[tid]     = 0.0f;
        s_alpha[tid] = 0.0f;
    }
    for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
        s_acc[idx] = 0.0f;
    }
    __syncthreads();

    // === Tile loop ====================================================
    for (int j = tile_start; j < tile_end; j++) {
        const int tile_base = j * tile_size;
        const int tile_len  = min(tile_size, max_seq_prefix_len - tile_base);
        if (tile_len <= 0) break;

        // -- Load K tile (FP8 in smem) + per-slot k_scale --
        // KV layout [num_blks, blk_size, num_kv_heads, head_dim] u8.
        // Vectorised 8-byte loads match the decode kernel pattern.
        {
            const int hdv = head_dim / 8;
            const int vec_total = tile_len * hdv;
            for (int vi = tid; vi < vec_total; vi += FA2_THREADS) {
                const int t = vi / hdv;
                const int d_base = (vi - t * hdv) * 8;
                const int kv_pos = tile_base + t;
                const int page_idx = kv_pos / block_size;
                const int page_off = kv_pos - page_idx * block_size;
                const int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                const int slot = phys_block * block_size + page_off;
                const unsigned char* k_row = key_cache
                    + (slot * num_kv_heads + kv_head) * head_dim + d_base;
                unsigned long long k8 = __ldg(
                    reinterpret_cast<const unsigned long long*>(k_row));
                unsigned char* dst = s_k_fp8 + t * head_dim + d_base;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    dst[b] = (unsigned char)(k8 >> (b * 8));
                }
            }
        }
        // Per-slot k_scale
        for (int t = tid; t < tile_len; t += FA2_THREADS) {
            const int kv_pos = tile_base + t;
            const int page_idx = kv_pos / block_size;
            const int page_off = kv_pos - page_idx * block_size;
            const int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const int slot = phys_block * block_size + page_off;
            s_k_scale[t] = (k_scale_cache != nullptr)
                ? __ldg(&k_scale_cache[slot * num_kv_heads + kv_head])
                : *k_descale_fallback;
        }
        __syncthreads();

        // -- Compute S[BLOCK_M, tile_len] = (scale·Q) · K^T · k_scale --
        // Thread layout: 128 threads round-robin over BLOCK_M×tile_len.
        // BLOCK_M=16, tile_len∈{16,32} gives 1-4 cells per thread.
        const int total_st = BLOCK_M * tile_len;
        for (int idx = tid; idx < total_st; idx += FA2_THREADS) {
            const int m = idx / tile_len;
            const int t = idx - m * tile_len;
            const float* qr = s_q + m * head_dim;
            const unsigned char* kr = s_k_fp8 + t * head_dim;
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < head_dim; d++) {
                dot += qr[d] * fp8kv_decode_byte(kr[d]);
            }
            dot *= s_k_scale[t];

            // Masks: causal + optional sliding window + valid query row.
            const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
            const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
            const int query_abs = prefix_len + q_pos_in_seq;
            const int kv_pos = tile_base + t;
            const bool valid_row = (q_pos_in_seq < query_len) && (q_head < num_heads);
            const bool causal = kv_pos <= query_abs;
            const bool sliding_ok =
                (window_size_left <= 0) || ((query_abs - kv_pos) < window_size_left);
            const bool valid = valid_row && causal && sliding_ok && (kv_pos < max_seq_prefix_len);
            s_s[idx] = valid ? dot : -FLT_MAX;
        }
        __syncthreads();

        // -- Online softmax: per-row max + denom + alpha ---------------
        // Row layout: 8 threads per row (row = tid/8, lane_in_row = tid%8).
        // Butterfly shfl within each row group converges independently.
        if (row < BLOCK_M) {
            float row_tile_max = -FLT_MAX;
            for (int t = lane_in_row; t < tile_len; t += 8) {
                row_tile_max = fmaxf(row_tile_max, s_s[row * tile_len + t]);
            }
            row_tile_max = row_reduce_max_8(row_tile_max);

            const float prev_M = s_m[row];
            float new_M = fmaxf(prev_M, row_tile_max);
            float alpha = (prev_M > -FLT_MAX && new_M > -FLT_MAX)
                ? expf(prev_M - new_M) : 0.0f;
            if (new_M == -FLT_MAX) {
                // All keys masked for this row this tile AND no prior
                // max — keep state empty but avoid NaN downstream.
                new_M = 0.0f;
            }

            float row_sum = 0.0f;
            for (int t = lane_in_row; t < tile_len; t += 8) {
                const float s = s_s[row * tile_len + t];
                const float p = (s > -FLT_MAX + 1.0f) ? expf(s - new_M) : 0.0f;
                s_s[row * tile_len + t] = p;  // overwrite S with P
                row_sum += p;
            }
            row_sum = row_reduce_sum_8(row_sum);

            if (lane_in_row == 0) {
                s_m[row]     = new_M;
                s_l[row]     = s_l[row] * alpha + row_sum;
                s_alpha[row] = alpha;
            }
        }
        __syncthreads();

        // -- Rescale running acc by alpha ------------------------------
        // Only needed if any row had a prior non-empty max; cheap
        // enough to always apply.
        for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
            const int m = idx / head_dim;
            s_acc[idx] *= s_alpha[m];
        }
        __syncthreads();

        // -- Load V tile (FP8 in smem) + per-slot v_scale --
        {
            const int hdv = head_dim / 8;
            const int vec_total = tile_len * hdv;
            for (int vi = tid; vi < vec_total; vi += FA2_THREADS) {
                const int t = vi / hdv;
                const int d_base = (vi - t * hdv) * 8;
                const int kv_pos = tile_base + t;
                const int page_idx = kv_pos / block_size;
                const int page_off = kv_pos - page_idx * block_size;
                const int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                const int slot = phys_block * block_size + page_off;
                const unsigned char* v_row = value_cache
                    + (slot * num_kv_heads + kv_head) * head_dim + d_base;
                unsigned long long v8 = __ldg(
                    reinterpret_cast<const unsigned long long*>(v_row));
                unsigned char* dst = s_v_fp8 + t * head_dim + d_base;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    dst[b] = (unsigned char)(v8 >> (b * 8));
                }
            }
        }
        for (int t = tid; t < tile_len; t += FA2_THREADS) {
            const int kv_pos = tile_base + t;
            const int page_idx = kv_pos / block_size;
            const int page_off = kv_pos - page_idx * block_size;
            const int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const int slot = phys_block * block_size + page_off;
            s_v_scale[t] = (v_scale_cache != nullptr)
                ? __ldg(&v_scale_cache[slot * num_kv_heads + kv_head])
                : *v_descale_fallback;
        }
        __syncthreads();

        // -- acc += P · V, dequant V on the fly, apply per-slot scale --
        // Each thread owns one (m, d) output cell; iterates across
        // tile_len reductions over t. BLOCK_M*head_dim / FA2_THREADS
        // gives 32 outputs/thread (head=256) or 64 (head=512).
        for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
            const int m = idx / head_dim;
            const int d = idx - m * head_dim;
            float sum = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                const float p = s_s[m * tile_len + t];
                const float v = fp8kv_decode_byte(s_v_fp8[t * head_dim + d]);
                sum += p * v * s_v_scale[t];
            }
            s_acc[idx] += sum;
        }
        __syncthreads();
    }

    // === Epilogue: acc / L → f16 output ==============================
    for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
        const int m = idx / head_dim;
        const int d = idx - m * head_dim;
        const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
        const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
        if (q_pos_in_seq >= query_len || q_head >= num_heads) continue;
        const int tok = seq_q_start + q_pos_in_seq;
        const float inv_l = (s_l[m] > 0.0f) ? (1.0f / s_l[m]) : 0.0f;
        output[(tok * num_heads + q_head) * head_dim + d] =
            __float2half(s_acc[idx] * inv_l);
    }
}
