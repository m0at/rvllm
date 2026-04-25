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

#include "fp8_mma_frag_pack.cuh"

#define FA2_THREADS 128

// Quantise an f32 to an FP8 E4M3 byte, saturating to ±448. Used by
// the P·V MMA branch to re-quantise the f32 softmax output P so it
// can enter the tensor core as an A operand.
__device__ __forceinline__ unsigned char f32_to_e4m3_byte(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return static_cast<unsigned char>(s);
#else
    if (v == 0.0f) return 0;
    unsigned sign = (v < 0.0f) ? 1u : 0u;
    float a = fabsf(v);
    if (a > 448.0f) a = 448.0f;
    int e = 0;
    while (a >= 2.0f) { a *= 0.5f; e++; }
    while (a < 1.0f && e > -6) { a *= 2.0f; e--; }
    int exp_bits = e + 7;
    if (exp_bits < 0) return (unsigned char)(sign << 7);
    if (exp_bits > 15) exp_bits = 15;
    int mant = (int)(rintf((a - 1.0f) * 8.0f));
    if (mant == 8) { mant = 0; exp_bits++; }
    if (exp_bits > 15) { exp_bits = 15; mant = 7; }
    return (unsigned char)((sign << 7) | ((exp_bits & 0xF) << 3) | (mant & 0x7));
#endif
}

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
// smem layout (dynamic, MMA_K = 32 fixed by the tensor-core inner dim):
//     s_q_fp8      [BLOCK_M * head_dim]      u8    (FP8 E4M3 bytes)
//     s_q_scale    [BLOCK_M]                 f32   (per-row scale ×
//                                                   softmax_scale; applied
//                                                   post-dot)
//     s_k_fp8      [head_dim * TILE_SIZE]    u8    (row-major [t][d])
//     s_v_fp8      [TILE_SIZE * head_dim]    u8
//     s_v_fp8_T    [MMA_K * head_dim]        u8    (P·V MMA B operand,
//                                                   zero-padded along k
//                                                   when tile_size < 32)
//     s_k_scale    [TILE_SIZE]               f32
//     s_v_scale    [TILE_SIZE]               f32
//     s_S          [BLOCK_M * MMA_K]         f32   (softmax writes with
//                                                   stride tile_len, but
//                                                   allocation is MMA_K
//                                                   so the P·V quant can
//                                                   feed zeros for k ≥
//                                                   tile_len)
//     s_M          [BLOCK_M]                 f32   (online-softmax max)
//     s_L          [BLOCK_M]                 f32   (online-softmax denom)
//     s_alpha      [BLOCK_M]                 f32
//     s_p_fp8      [BLOCK_M * MMA_K]         u8    (re-quantised P)
//     s_p_scale    [BLOCK_M]                 f32
//     s_acc        [BLOCK_M * head_dim]      f32
//     s_reduce     [FA2_THREADS / 32]        f32   (block reductions)
//
// Storing Q as FP8 rather than pre-decoded f32 drops the Q tile from
// BLOCK_M*head_dim*4 to BLOCK_M*head_dim bytes (12 KB savings at
// head_dim=256) and, more importantly, lets the tensor-core path
// (enabled by `use_mma=1`) pack the A fragment directly from smem
// with no extra f32→FP8 round trip.
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
    int window_size_left, // -1 ⇒ no sliding window
    int use_mma           // 0: scalar Q·Kᵀ, 1: tensor-core MMA path (Phase F3+)
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
    //
    // MMA_K = 32 is the tensor-core's fixed inner reduction for the
    // m16n8k32 FP8 MMA. For sliding (tile_size=32) this equals
    // tile_size; for global (tile_size=16) the MMA operand buffers
    // are zero-padded along the k axis. s_v_fp8_T, s_p_fp8, and s_s
    // (the P·V MMA operands + softmax scratch that feeds quantisation)
    // size themselves off MMA_K rather than tile_size so the same
    // kernel handles both layer types. s_v_fp8 / s_k_fp8 stay at
    // tile_size (the real K/V data count).
    constexpr int MMA_K = 32;
    extern __shared__ unsigned char smem_raw[];
    unsigned char* s_q_fp8   = smem_raw;
    float*         s_q_scale = reinterpret_cast<float*>(s_q_fp8 + BLOCK_M * head_dim);
    unsigned char* s_k_fp8   = reinterpret_cast<unsigned char*>(s_q_scale + BLOCK_M);
    unsigned char* s_v_fp8   = s_k_fp8 + head_dim * tile_size;
    unsigned char* s_v_fp8_T = s_v_fp8 + tile_size * head_dim;
    float*         s_k_scale = reinterpret_cast<float*>(s_v_fp8_T + MMA_K * head_dim);
    float*         s_v_scale = s_k_scale + tile_size;
    float*         s_s       = s_v_scale + tile_size;
    float*         s_m       = s_s + BLOCK_M * MMA_K;
    float*         s_l       = s_m + BLOCK_M;
    float*         s_alpha   = s_l + BLOCK_M;
    unsigned char* s_p_fp8   = reinterpret_cast<unsigned char*>(s_alpha + BLOCK_M);
    float*         s_p_scale = reinterpret_cast<float*>(s_p_fp8 + BLOCK_M * MMA_K);
    float*         s_acc     = s_p_scale + BLOCK_M;

    const int row        = tid / 8;
    const int lane_in_row = tid & 7;

    // === Load Q into smem =============================================
    // Store raw FP8 bytes + per-row (q_scale × softmax_scale). The
    // scalar Q·Kᵀ path decodes on the fly; the MMA path consumes the
    // bytes directly via `pack_a_frag_row_major_m16k32`. Scale is
    // applied post-dot (fp32 accumulator × s_q_scale[m] × s_k_scale[t]).
    for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
        const int m = idx / head_dim;
        const int d = idx - m * head_dim;
        const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
        const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
        if (q_pos_in_seq >= query_len || q_head >= num_heads) {
            s_q_fp8[idx] = 0;
            continue;
        }
        const int tok = seq_q_start + q_pos_in_seq;
        s_q_fp8[idx] = query[(tok * num_heads + q_head) * head_dim + d];
    }
    // Per-row q_scale × softmax_scale. Only BLOCK_M entries — written
    // by the first BLOCK_M threads.
    if (tid < BLOCK_M) {
        const int m = tid;
        const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
        const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
        if (q_pos_in_seq < query_len && q_head < num_heads) {
            const int tok = seq_q_start + q_pos_in_seq;
            const float qs = (q_scale_cache != nullptr)
                ? q_scale_cache[tok * num_heads + q_head]
                : *q_descale;
            s_q_scale[m] = qs * scale;
        } else {
            s_q_scale[m] = 0.0f;
        }
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
        //
        // Two paths, selected at launch via `use_mma`:
        //
        //   * SCALAR (use_mma==0). 128 threads round-robin over
        //     BLOCK_M × tile_len cells. Each thread does one dot
        //     product of length head_dim with FP8→f32 dequant in the
        //     inner loop. Correct but scalar-FMA bound.
        //
        //   * MMA    (use_mma==1). 4 warps × 1 n-tile each, each
        //     warp loops `head_dim / 32` k-steps calling
        //     `mma_m16n8k32_e4m3_e4m3_f32`. Accumulates 16×8 cells
        //     per warp in registers, unpacks into s_s with per-row /
        //     per-slot scale applied. Requires tile_len ∈ {8, 16,
        //     24, 32} so `tile_len / 8` warps ≤ 4 — fine for Gemma 4
        //     (tile_len=32 sliding, 16 global).
        const auto mask_pack = [&](int m, int t, float dot) -> float {
            const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
            const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
            const int query_abs = prefix_len + q_pos_in_seq;
            const int kv_pos = tile_base + t;
            const bool valid_row = (q_pos_in_seq < query_len) && (q_head < num_heads);
            const bool causal = kv_pos <= query_abs;
            const bool sliding_ok =
                (window_size_left <= 0) || ((query_abs - kv_pos) < window_size_left);
            const bool valid = valid_row && causal && sliding_ok
                && (kv_pos < max_seq_prefix_len);
            return valid ? dot : -FLT_MAX;
        };

        if (use_mma == 0) {
            const int total_st = BLOCK_M * tile_len;
            for (int idx = tid; idx < total_st; idx += FA2_THREADS) {
                const int m = idx / tile_len;
                const int t = idx - m * tile_len;
                const unsigned char* qr = s_q_fp8 + m * head_dim;
                const unsigned char* kr = s_k_fp8 + t * head_dim;
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < head_dim; d++) {
                    dot += fp8kv_decode_byte(qr[d]) * fp8kv_decode_byte(kr[d]);
                }
                dot *= s_q_scale[m] * s_k_scale[t];
                s_s[idx] = mask_pack(m, t, dot);
            }
        } else {
            // MMA path — one warp per n-tile of width 8.
            //
            // A fragment is the Q tile [16, k-step*32 .. k-step*32+32].
            // B fragment is the K slice [w*8 .. w*8+7, k-step*32 ..
            // k-step*32+32] — col-major [n][k] matches how we stored
            // s_k_fp8 (`t` outer, `d` inner), which means the "col
            // stride" for `pack_b_frag_col_major_n8k32` is head_dim.
            const int warp_id = tid >> 5;            // 0..3
            const int lane    = tid & 31;
            const int n_tiles = (tile_len + 7) >> 3; // ceil(tile_len/8)
            const int k_steps = head_dim >> 5;       // head_dim / 32

            if (warp_id < n_tiles) {
                const int n_base = warp_id * 8;
                const unsigned char* q_row0 = s_q_fp8;                // row stride = head_dim
                const unsigned char* k_base_ptr = s_k_fp8 + n_base * head_dim;
                float d_frag[4]; rvllm::zero_mma_d_frag(d_frag);
                uint32_t a[4];
                uint32_t b[2];
                #pragma unroll 1
                for (int ks = 0; ks < k_steps; ks++) {
                    const int d_off = ks * 32;
                    rvllm::pack_a_frag_row_major_m16k32(
                        q_row0 + d_off, head_dim, a, lane);
                    rvllm::pack_b_frag_col_major_n8k32(
                        k_base_ptr + d_off, head_dim, b, lane);
                    rvllm::mma_m16n8k32_e4m3_e4m3_f32(d_frag, a, b);
                }

                // Unpack d_frag into s_s, applying per-row / per-slot
                // scales + masks. Per PTX spec for m16n8k32 output
                // (same as `unpack_d_frag_to_smem_m16n8` but with scale
                // and mask fused — cheaper than writing raw then
                // rewriting).
                const int r_lo = lane >> 2;
                const int r_hi = r_lo + 8;
                const int c    = (lane & 3) << 1;
                const int t_lo = n_base + c;
                const int t_hi = n_base + c + 1;
                // Row r_lo
                if (t_lo < tile_len) {
                    float dot = d_frag[0] * s_q_scale[r_lo] * s_k_scale[t_lo];
                    s_s[r_lo * tile_len + t_lo] = mask_pack(r_lo, t_lo, dot);
                }
                if (t_hi < tile_len) {
                    float dot = d_frag[1] * s_q_scale[r_lo] * s_k_scale[t_hi];
                    s_s[r_lo * tile_len + t_hi] = mask_pack(r_lo, t_hi, dot);
                }
                // Row r_hi
                if (t_lo < tile_len) {
                    float dot = d_frag[2] * s_q_scale[r_hi] * s_k_scale[t_lo];
                    s_s[r_hi * tile_len + t_lo] = mask_pack(r_hi, t_lo, dot);
                }
                if (t_hi < tile_len) {
                    float dot = d_frag[3] * s_q_scale[r_hi] * s_k_scale[t_hi];
                    s_s[r_hi * tile_len + t_hi] = mask_pack(r_hi, t_hi, dot);
                }
            }
            // Warps past n_tiles are idle for Q·Kᵀ — they rejoin at
            // the __syncthreads below for the softmax phase.
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
        // -- (Phase F12) alpha-rescale of s_acc is FUSED into the
        //    P·V step below. Scalar path folds it into its s_acc
        //    write; MMA path loads `s_acc * alpha / p_scale` into
        //    the MMA C operand so that one tensor-core op replaces
        //    the former "rescale + accumulate" pair. No dedicated
        //    s_acc pass before V load anymore.

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

        // -- acc += P · V =================================================
        //
        // Two paths, selected at launch via `use_mma` + tile shape:
        //
        //   * SCALAR (always valid). Each thread owns one (m, d) output
        //     cell, loops `tile_len` reductions over t. BLOCK_M*head_dim /
        //     FA2_THREADS gives 32 outputs/thread (head=256) or 64 (512).
        //   * MMA    (`use_mma==1 && tile_size==32`, i.e. sliding layers
        //     only). Re-quantise P to FP8 per row (per-row dynamic max
        //     × 1/FP8_MAX; v_scale[t] is folded into P first so the MMA
        //     inputs carry it), transpose V into the [d][t] layout the
        //     MMA B operand needs, then 4 warps × head_dim/32 n-tiles
        //     per warp run one `mma.sync m16n8k32` each. Output is
        //     scaled by s_p_scale[row] and added into s_acc. Global
        //     layers (tile_size=16) fall through to scalar today — the
        //     k=32 MMA would need V-padding + extra smem that the 10
        //     global layers don't benefit enough from to justify.
        const bool pv_use_mma = (use_mma != 0);

        if (pv_use_mma) {
            // (1) Transpose V [t][d] → s_v_fp8_T [d][k] with k stride
            // = MMA_K (=32). For sliding layers tile_size=32=MMA_K.
            // For global layers (tile_size=16), positions k ∈
            // [tile_len, MMA_K) are zero-padded so they contribute
            // 0 to the MMA reduction. Phase F7.
            //
            // u32 writes (4 consecutive k values packed) sidestep a
            // sm_121 smem hazard where sibling lanes byte-writing
            // the same 4-byte word produce silently-corrupted
            // neighbouring regions (F6 tracked that down; bug 4 in
            // the F6 writeup).
            constexpr int MMA_K_GROUPS = MMA_K / 4;
            for (int idx = tid; idx < MMA_K_GROUPS * head_dim; idx += FA2_THREADS) {
                const int tg = idx / head_dim;     // 0..7
                const int d  = idx - tg * head_dim;
                const int t_base = tg * 4;
                uint32_t packed = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int t = t_base + k;
                    unsigned char b = (t < tile_len)
                        ? s_v_fp8[t * head_dim + d]
                        : 0;
                    packed |= (uint32_t)b << (k * 8);
                }
                *reinterpret_cast<uint32_t*>(s_v_fp8_T + d * MMA_K + t_base) = packed;
            }
            __syncthreads();
            // (2) Fold v_scale[t] into P, find per-row max, quantise to
            // FP8. 16 rows × 8 lanes-per-row layout re-used from softmax.
            const int pv_row = tid >> 3;
            const int pv_lr  = tid & 7;
            if (pv_row < BLOCK_M) {
                float local_max = 0.0f;
                // Softmax wrote s_s with stride `tile_len` (not tile_size).
                // For partial tiles tile_len < tile_size; `pv_row * tile_len`
                // is the correct address. Positions t ∈ [tile_len, tile_size)
                // have no softmax data — they feed 0 into the MMA via
                // s_p_fp8 padding below, without reading back from s_s.
                for (int t = pv_lr; t < tile_len; t += 8) {
                    const float p = s_s[pv_row * tile_len + t] * s_v_scale[t];
                    s_s[pv_row * tile_len + t] = p;
                    local_max = fmaxf(local_max, fabsf(p));
                }
                const float row_max = row_reduce_max_8(local_max);
                if (pv_lr == 0) {                    s_p_scale[pv_row] = (row_max > 1e-30f)
                        ? (row_max * (1.0f / 448.0f)) : 1.0f;
                }
            }
            __syncthreads();
            if (pv_row < BLOCK_M) {
                const float inv_scale = 1.0f / s_p_scale[pv_row];
                // Real P values cover t ∈ [0, tile_len). We fill
                // s_p_fp8 up to MMA_K=32 slots per row; positions
                // t ≥ tile_len feed zero into the MMA (valid for
                // global-layer tiles with tile_len ≤ 16 < MMA_K).
                for (int t = pv_lr; t < MMA_K; t += 8) {
                    const float p_scaled = (t < tile_len)
                        ? s_s[pv_row * tile_len + t] * inv_scale
                        : 0.0f;
                    s_p_fp8[pv_row * MMA_K + t] = f32_to_e4m3_byte(p_scaled);
                }
            }
            __syncthreads();

            // (3) MMA P·V_T. 4 warps × (head_dim/8)/4 n-tiles per warp.
            //     Each MMA output is 16×8; scale by s_p_scale[row] and
            //     ADD into s_acc.
            const int warp_id = tid >> 5;
            const int lane    = tid & 31;
            const int n_tiles_total    = head_dim >> 3;
            const int n_tiles_per_warp = n_tiles_total >> 2;
            uint32_t a[4];
            rvllm::pack_a_frag_row_major_m16k32(s_p_fp8, MMA_K, a, lane);
            // Phase F12 — fused alpha rescale via MMA C operand.
            //
            // Pre-F12 each tile did:
            //   s_acc *= alpha                  (dedicated pass, R+W)
            //   d = a · b  (zero C)             (MMA)
            //   s_acc += d * p_scale            (R+W)
            //
            // Now we exploit the MMA's `D = A·B + C` semantics: load
            //   C = s_acc_old * alpha / p_scale
            // into the accumulator, run the MMA, write back
            //   s_acc_new = d * p_scale
            //             = a·b * p_scale + C * p_scale
            //             = <this tile's contribution> + s_acc_old * alpha
            // in a single smem R + W per cell instead of two.
            const int r_lo = lane >> 2;
            const int r_hi = r_lo + 8;
            const int c    = (lane & 3) << 1;
            const float alpha_lo = s_alpha[r_lo];
            const float alpha_hi = s_alpha[r_hi];
            #pragma unroll 1
            for (int nt = 0; nt < n_tiles_per_warp; nt++) {
                const int n_base = (warp_id * n_tiles_per_warp + nt) * 8;
                const int d_lo = n_base + c;
                const int d_hi = d_lo + 1;
                const float pscale_lo = s_p_scale[r_lo];
                const float pscale_hi = s_p_scale[r_hi];
                const float inv_ps_lo = 1.0f / pscale_lo;
                const float inv_ps_hi = 1.0f / pscale_hi;
                uint32_t b[2];
                rvllm::pack_b_frag_col_major_n8k32(
                    s_v_fp8_T + n_base * MMA_K, MMA_K, b, lane);
                float d_frag[4];
                d_frag[0] = s_acc[r_lo * head_dim + d_lo] * alpha_lo * inv_ps_lo;
                d_frag[1] = s_acc[r_lo * head_dim + d_hi] * alpha_lo * inv_ps_lo;
                d_frag[2] = s_acc[r_hi * head_dim + d_lo] * alpha_hi * inv_ps_hi;
                d_frag[3] = s_acc[r_hi * head_dim + d_hi] * alpha_hi * inv_ps_hi;
                rvllm::mma_m16n8k32_e4m3_e4m3_f32(d_frag, a, b);
                s_acc[r_lo * head_dim + d_lo] = d_frag[0] * pscale_lo;
                s_acc[r_lo * head_dim + d_hi] = d_frag[1] * pscale_lo;
                s_acc[r_hi * head_dim + d_lo] = d_frag[2] * pscale_hi;
                s_acc[r_hi * head_dim + d_hi] = d_frag[3] * pscale_hi;
            }
        } else {
            // Phase F12 — fold alpha rescale into the scalar
            // accumulator write. One smem R+W per cell instead of
            // the former "rescale pass + += pass" two R+W.
            for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
                const int m = idx / head_dim;
                const int d = idx - m * head_dim;
                float sum = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    const float p = s_s[m * tile_len + t];
                    const float v = fp8kv_decode_byte(s_v_fp8[t * head_dim + d]);
                    sum += p * v * s_v_scale[t];
                }
                s_acc[idx] = s_acc[idx] * s_alpha[m] + sum;
            }
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
