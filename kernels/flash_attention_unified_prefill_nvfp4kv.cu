// Unified NVFP4-KV multi-query prefill for sm_121 — Phase 2b of task
// aa01001nvf4f16mma.
//
// Mirrors `flash_attention_unified_prefill.cu` (the FP8-KV variant)
// but reads the KV cache as NVFP4 (packed 4-bit + per-16-element
// E4M3 microscale) and drives the Q·Kᵀ and P·V tensor-core MMAs
// through f16 operands via the validated `f16_mma_frag_pack.cuh`.
// The architectural template and all bounds / causal / sliding-window
// logic match the FP8 sibling; diffs are confined to:
//
//   (a) K/V load path: dequant NVFP4 → f16 via `cvt.rn.f16x2.e2m1x2`
//       (nvfp4_utils.cuh `unpack16_nvfp4_to_f16_fast`). No per-slot
//       scalar K/V scale — the E4M3 microscale is baked into the f16
//       values at dequant time.
//   (b) Q load path: dequant FP8 → f16 once at entry (vs the FP8
//       kernel which stored raw FP8 and packed at MMA time).
//   (c) MMA shape: `mma_m16n8k16_f16_f16_f32` (m16n8k16, MMA_K=16)
//       instead of `mma_m16n8k32_e4m3_e4m3_f32`. Halves the per-step
//       k reduction; doubles the number of MMA calls per Q·Kᵀ column
//       compared to FP8. This is the industry-standard approach
//       (FlashInfer xqa / vLLM CuTe SM120) — see memory 22222222aa010020.
//   (d) P·V: P stays f16 (no FP8 re-quant pass); v_scale is already
//       folded into V_f16 at dequant, so the MMA output for a given
//       row scales only by `s_p_scale[row]` relative to the real
//       softmax P (i.e. after the per-row max reduction we store
//       P/s_p_scale so the f16 MMA operands stay inside the f16
//       dynamic range).
//
// Target perf: close the ~10× NVFP4-vs-FP8 batch-prefill gap
// (memory aa010020: NVFP4 62 s vs FP8 7 s at prompt_len=1082). Not
// a bit-identical port — f16 vs e4m3 accumulate differently — but
// within the fp64-reference bound that fa2_nvfp4_prefill_check.py
// already holds.
//
// Validation harness: to be extended from fa2_nvfp4_prefill_check.py
// in the next session. This file compiles standalone + emits a PTX
// module; Rust wiring + dispatch + end-to-end test are the Phase 2b
// follow-up.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>

#include "nvfp4_utils.cuh"
#include "f16_mma_frag_pack.cuh"

#define FA2_THREADS 128

// FP8 byte → f32 (only used for the per-row Q descale at entry).
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

// Direct fp8→half cast — skips the f32 round-trip when the consumer
// stores f16 anyway (Q load → s_q_f16). On sm_121 this is one PTX
// `cvt.rn.f16.e4m3` instruction; the older fallback path goes via
// the float decoder + __float2half.
__device__ __forceinline__ __half fp8kv_decode_byte_half(unsigned char b) {
#if __CUDA_ARCH__ >= 1000
    __half_raw hr = __nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)b, __NV_E4M3);
    return __half(hr);
#else
    return __float2half(fp8kv_decode_byte(b));
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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

// Dequant one NVFP4 row of `head_dim` elements into f16 smem. Same
// cooperative pattern as the decode kernel's f16 version:
// `head_dim/16` threads each own one 16-element block.
__device__ __forceinline__ void dequant_nvfp4_row_to_f16_smem(
    const uint8_t*       __restrict__ packed_row,   // head_dim / 2 bytes
    const __nv_fp8_e4m3* __restrict__ scale_row,    // head_dim / 16 E4M3
    __half*              __restrict__ s_dst,        // head_dim halves
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

// BLOCK_M: match the FP8 kernel's choice. 16 covers every Gemma 4
// layer (num_queries_per_kv ∈ {2, 8}).
#ifndef UNIFIED_PREFILL_BLOCK_M
#define UNIFIED_PREFILL_BLOCK_M 16
#endif

// ============================================================================
// flash_attention_2_prefill_nvfp4kv_unified_kernel
// ----------------------------------------------------------------------------
// One CUDA block processes BLOCK_M query rows of a single KV head for one
// sequence. Grid dims:
//     gridDim.x = total_num_q_blocks   (host-computed upper bound)
//     gridDim.y = num_kv_heads
//
// smem layout (MMA_K = 16 for f16 m16n8k16):
//     s_q_f16      [BLOCK_M * head_dim]      f16 — dequanted Q
//     s_q_scale    [BLOCK_M]                 f32 — post-MMA scale
//     s_k_f16      [tile_size * head_dim]    f16 — K tile row-major [t][d]
//     s_v_f16      [tile_size * head_dim]    f16 — V tile row-major [t][d]
//     s_v_f16_T    [MMA_K * head_dim]        f16 — V tile transposed for P·V
//     s_s          [BLOCK_M * MMA_K]         f32 — softmax output buffer
//     s_m          [BLOCK_M]                 f32 — online-softmax max
//     s_l          [BLOCK_M]                 f32 — online-softmax denom
//     s_alpha      [BLOCK_M]                 f32 — rescale factor
//     s_p_f16      [BLOCK_M * MMA_K]         f16 — P cast to f16 for MMA
//     s_p_scale    [BLOCK_M]                 f32 — P magnitude bound
//     s_acc        [BLOCK_M * head_dim]      f32 — accumulator
// ----------------------------------------------------------------------------

extern "C"
__global__ void flash_attention_2_prefill_nvfp4kv_unified_kernel(
    __half*              __restrict__ output,                 // [num_tokens, H, D]
    const unsigned char* __restrict__ query,                  // [num_tokens, H, D] fp8
    const uint8_t*       __restrict__ key_cache_packed,       // [B*bs*KH*D/2] u8
    const uint8_t*       __restrict__ value_cache_packed,     // [B*bs*KH*D/2] u8
    const __nv_fp8_e4m3* __restrict__ key_cache_scale,        // [B*bs*KH*D/16] e4m3
    const __nv_fp8_e4m3* __restrict__ value_cache_scale,      // [B*bs*KH*D/16] e4m3
    const float*         __restrict__ q_scale_cache,          // [num_tokens, H] f32 (optional)
    const int*           __restrict__ block_tables,           // [S, max_blocks]
    const int*           __restrict__ cu_seqlens_q,           // [S+1]
    const int*           __restrict__ context_lens,           // [S]
    const float*         __restrict__ q_descale,              // single f32 fallback
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int tile_size,
    int num_queries_per_kv,
    int block_q,
    int num_seqs,
    int window_size_left
) {
    constexpr int MMA_K   = 16;     // f16 m16n8k16
    constexpr int BLOCK_M = UNIFIED_PREFILL_BLOCK_M;

    const int q_block_global = blockIdx.x;
    const int kv_head        = blockIdx.y;
    const int tid            = threadIdx.x;

    // Locate this block's (seq, q_block_local).
    const int seq_idx = find_seq_idx_linear(
        cu_seqlens_q, q_block_global, num_seqs, block_q);
    const int seq_q_start = cu_seqlens_q[seq_idx];
    const int seq_q_end   = cu_seqlens_q[seq_idx + 1];
    const int query_len   = seq_q_end - seq_q_start;
    const int q_block_start_global = seq_q_start / block_q + seq_idx;
    const int q_block_local = q_block_global - q_block_start_global;
    if (q_block_local * block_q >= query_len) return;

    const int seq_len    = context_lens[seq_idx];
    const int prefix_len = seq_len - query_len;
    const int q_block_qpos_hi =
        q_block_local * block_q + (BLOCK_M - 1) / num_queries_per_kv;
    int max_seq_prefix_len = prefix_len + q_block_qpos_hi + 1;
    if (max_seq_prefix_len > seq_len) max_seq_prefix_len = seq_len;

    int tile_start = 0;
    int tile_end   = (max_seq_prefix_len + tile_size - 1) / tile_size;
    if (window_size_left > 0) {
        const int qpos_lo = q_block_local * block_q;
        const int qpos_hi = min(q_block_qpos_hi, query_len - 1);
        // window_size_left semantics MUST match the decode kernels
        // (flash_attention_split_decode_nvfp4kv.cu / decode.cu) which
        // use `first_allowed = max(0, q_abs - window_size_left)` and
        // mask `kv_pos < window_start` → window covers `W+1` tokens
        // [q_abs - W, q_abs]. The previous `q_abs - W + 1` here gave
        // only `W` tokens, so a single request that hit prefill +
        // decode saw two different window sizes — which slowly
        // shifted attention scores at the window edge across phases.
        const int first_allowed = prefix_len + qpos_lo - window_size_left;
        const int last_allowed  = prefix_len + qpos_hi;
        int ts = first_allowed / tile_size;
        if (first_allowed < 0) ts = 0;
        if (ts < 0) ts = 0;
        int te = last_allowed / tile_size + 1;
        if (te > tile_end) te = tile_end;
        tile_start = ts;
        tile_end   = te;
    }

    // === smem layout ==================================================
    // NB: MMA_K = 16 for f16 m16n8k16 whereas the FP8 unified variant
    // uses MMA_K = 32 (m16n8k32 e4m3). For tile_size > MMA_K, s_s must
    // be sized to BLOCK_M * tile_size (softmax writes with stride
    // tile_len up to tile_size), and the P·V path loops over
    // ceil(tile_len / MMA_K) MMA steps inside each tile iteration.
    const int s_s_stride = (tile_size > MMA_K) ? tile_size : MMA_K;
    extern __shared__ unsigned char smem_raw[];
    __half* s_q_f16    = reinterpret_cast<__half*>(smem_raw);
    float*  s_q_scale  = reinterpret_cast<float*>(s_q_f16 + BLOCK_M * head_dim);
    __half* s_k_f16    = reinterpret_cast<__half*>(s_q_scale + BLOCK_M);
    __half* s_v_f16    = s_k_f16 + tile_size * head_dim;
    __half* s_v_f16_T  = s_v_f16 + tile_size * head_dim;
    float*  s_s        = reinterpret_cast<float*>(s_v_f16_T + MMA_K * head_dim);
    float*  s_m        = s_s + BLOCK_M * s_s_stride;
    float*  s_l        = s_m + BLOCK_M;
    float*  s_alpha    = s_l + BLOCK_M;
    __half* s_p_f16    = reinterpret_cast<__half*>(s_alpha + BLOCK_M);
    // P safety scale (s_p_scale) removed: post-softmax P is in [0, 1] which
    // fits f16 trivially, so the per-row max-reduce + divide + multiply was
    // always identity (commit 2026-04-26). Saves BLOCK_M × 4 bytes of smem
    // + a row_reduce_max_8 + per-cell inv_ps multiply per P·V tile.
    float*  s_acc      = reinterpret_cast<float*>(s_p_f16 + BLOCK_M * MMA_K);

    const int row         = tid >> 3;
    const int lane_in_row = tid & 7;

    // === Load Q (FP8 → f16) ===========================================
    // Apply per-token Q descale AT LOAD, fold softmax `scale` into
    // s_q_scale (applied post-MMA to keep f16 dynamic range healthy).
    for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
        const int m = idx / head_dim;
        const int d = idx - m * head_dim;
        const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
        const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
        if (q_pos_in_seq >= query_len || q_head >= num_heads) {
            s_q_f16[idx] = __float2half(0.0f);
            continue;
        }
        const int tok = seq_q_start + q_pos_in_seq;
        unsigned char qb = query[(tok * num_heads + q_head) * head_dim + d];
        // Direct fp8→f16 (skips the f32 round-trip the older
        // fp8kv_decode_byte path goes through).
        s_q_f16[idx] = fp8kv_decode_byte_half(qb);
    }
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

    // === Init online-softmax state ====================================
    if (tid < BLOCK_M) {
        s_m[tid]     = -FLT_MAX;
        s_l[tid]     = 0.0f;
        s_alpha[tid] = 0.0f;
    }
    for (int idx = tid; idx < BLOCK_M * head_dim; idx += FA2_THREADS) {
        s_acc[idx] = 0.0f;
    }
    // Sync removed: K load below writes s_k_f16, doesn't read s_m /
    // s_l / s_alpha / s_acc / s_q_scale. The post-K-load sync gates
    // visibility for the Q·K^T MMA and the mask_pack that reads
    // s_q_scale. The first online-softmax read of s_l / s_m / s_alpha
    // happens after two more syncs (post-K-load + post-Q·K^T-MMA).

    const int half_D       = head_dim >> 1;   // NVFP4 packed bytes per row
    const int scales_per_D = head_dim >> 4;   // E4M3 scales per row
    const int blocks_per_row = scales_per_D;

    // === Tile loop ====================================================
    for (int j = tile_start; j < tile_end; j++) {
        const int tile_base = j * tile_size;
        const int tile_len  = min(tile_size, max_seq_prefix_len - tile_base);
        if (tile_len <= 0) break;

        // -- Load K tile (NVFP4 → f16 smem). Thread-per-(row, 16-block);
        //    keeps all 128 threads busy during the dequant phase
        //    instead of the `head_dim/16`-threads-per-row pattern the
        //    legacy decode kernel uses. At head_dim=256 that's 16 of
        //    128 threads → 128/128 after this change.
        {
            const int total_units = tile_len * blocks_per_row;
            for (int u = tid; u < total_units; u += FA2_THREADS) {
                const int t         = u / blocks_per_row;
                const int block_idx = u - t * blocks_per_row;
                const int kv_pos    = tile_base + t;
                const int page_idx  = kv_pos / block_size;
                const int page_off  = kv_pos - page_idx * block_size;
                const int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                const int slot      = phys_block * block_size + page_off;
                const uint8_t* k_packed = key_cache_packed
                    + (slot * num_kv_heads + kv_head) * half_D;
                const __nv_fp8_e4m3* k_scale = key_cache_scale
                    + (slot * num_kv_heads + kv_head) * scales_per_D;
                const int d_base = block_idx << 4;
                uint64_t packed8 = *reinterpret_cast<const uint64_t*>(
                    k_packed + (d_base >> 1));
                float    scale   = float(k_scale[block_idx]);
                rvllm_nvfp4::unpack16_nvfp4_to_f16_fast(
                    packed8, scale,
                    s_k_f16 + t * head_dim + d_base);
            }
        }
        __syncthreads();

        // -- Q·Kᵀ MMA: 4 warps × 1 n-tile of width 8 each. --
        // A fragment: Q tile [16, 16] slice, row-major, row stride = head_dim.
        // B fragment: K slice [8, 16] col-major, col stride = head_dim
        //   (s_k_f16 layout is [t][d] so t = n, d = k → "col-major [n][k]").
        const auto mask_pack = [&](int m, int t, float dot) -> float {
            const int q_pos_in_seq = q_block_local * block_q + m / num_queries_per_kv;
            const int q_head = kv_head * num_queries_per_kv + (m % num_queries_per_kv);
            const int query_abs = prefix_len + q_pos_in_seq;
            const int kv_pos = tile_base + t;
            const bool valid_row = (q_pos_in_seq < query_len) && (q_head < num_heads);
            const bool causal = kv_pos <= query_abs;
            // Match decode kernels: window covers [q_abs - W, q_abs]
            // inclusive (W+1 tokens). Was `<` (W tokens) — fixed for
            // phase consistency. See tile-bound comment above.
            const bool sliding_ok =
                (window_size_left <= 0) || ((query_abs - kv_pos) <= window_size_left);
            const bool valid = valid_row && causal && sliding_ok
                && (kv_pos < max_seq_prefix_len);
            return valid ? dot : -FLT_MAX;
        };

        {
            const int warp_id = tid >> 5;
            const int lane    = tid & 31;
            const int n_tiles = (tile_len + 7) >> 3;
            const int k_steps = head_dim >> 4;   // head_dim / 16

            if (warp_id < n_tiles) {
                const int n_base = warp_id * 8;
                const __half* q_row0 = s_q_f16;             // row stride = head_dim
                const __half* k_base_ptr = s_k_f16 + n_base * head_dim;
                float d_frag[4]; rvllm_f16mma::zero_mma_d_frag(d_frag);
                uint32_t a[4];
                uint32_t b[2];
                #pragma unroll 1
                for (int ks = 0; ks < k_steps; ks++) {
                    const int d_off = ks * 16;
                    rvllm_f16mma::pack_a_frag_row_major_m16k16_f16(
                        q_row0 + d_off,
                        /*stride_bytes=*/head_dim * (int)sizeof(__half),
                        a, lane);
                    rvllm_f16mma::pack_b_frag_col_major_n8k16_f16(
                        k_base_ptr + d_off,
                        /*stride_bytes=*/head_dim * (int)sizeof(__half),
                        b, lane);
                    rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d_frag, a, b);
                }
                // Unpack d_frag into s_s with per-row scale + causal mask.
                const int r_lo = lane >> 2;
                const int r_hi = r_lo + 8;
                const int c    = (lane & 3) << 1;
                const int t_lo = n_base + c;
                const int t_hi = n_base + c + 1;
                if (t_lo < tile_len) {
                    s_s[r_lo * tile_len + t_lo] =
                        mask_pack(r_lo, t_lo, d_frag[0] * s_q_scale[r_lo]);
                }
                if (t_hi < tile_len) {
                    s_s[r_lo * tile_len + t_hi] =
                        mask_pack(r_lo, t_hi, d_frag[1] * s_q_scale[r_lo]);
                }
                if (t_lo < tile_len) {
                    s_s[r_hi * tile_len + t_lo] =
                        mask_pack(r_hi, t_lo, d_frag[2] * s_q_scale[r_hi]);
                }
                if (t_hi < tile_len) {
                    s_s[r_hi * tile_len + t_hi] =
                        mask_pack(r_hi, t_hi, d_frag[3] * s_q_scale[r_hi]);
                }
            }
        }
        __syncthreads();

        // -- Online softmax (same as FP8 variant) --
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
            if (new_M == -FLT_MAX) new_M = 0.0f;

            float row_sum = 0.0f;
            for (int t = lane_in_row; t < tile_len; t += 8) {
                const float s = s_s[row * tile_len + t];
                const float p = (s > -FLT_MAX + 1.0f) ? expf(s - new_M) : 0.0f;
                s_s[row * tile_len + t] = p;
                row_sum += p;
            }
            row_sum = row_reduce_sum_8(row_sum);

            if (lane_in_row == 0) {
                s_m[row]     = new_M;
                s_l[row]     = s_l[row] * alpha + row_sum;
                s_alpha[row] = alpha;
            }
        }
        // Sync removed: V load below writes s_v_f16, doesn't read the
        // online-softmax state (s_m / s_l / s_alpha / s_s). Cross-tile
        // ordering is gated by the post-V-load sync further down.

        // -- Load V tile (NVFP4 → f16 smem). Same parallel pattern
        //    as the K load above.
        {
            const int total_units = tile_len * blocks_per_row;
            for (int u = tid; u < total_units; u += FA2_THREADS) {
                const int t         = u / blocks_per_row;
                const int block_idx = u - t * blocks_per_row;
                const int kv_pos    = tile_base + t;
                const int page_idx  = kv_pos / block_size;
                const int page_off  = kv_pos - page_idx * block_size;
                const int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                const int slot      = phys_block * block_size + page_off;
                const uint8_t* v_packed = value_cache_packed
                    + (slot * num_kv_heads + kv_head) * half_D;
                const __nv_fp8_e4m3* v_scale = value_cache_scale
                    + (slot * num_kv_heads + kv_head) * scales_per_D;
                const int d_base = block_idx << 4;
                uint64_t packed8 = *reinterpret_cast<const uint64_t*>(
                    v_packed + (d_base >> 1));
                float    scale   = float(v_scale[block_idx]);
                rvllm_nvfp4::unpack16_nvfp4_to_f16_fast(
                    packed8, scale,
                    s_v_f16 + t * head_dim + d_base);
            }
        }
        __syncthreads();

        // P safety scale removed (post-softmax P is in [0, 1], so the
        // clamp was always identity). Saves a row_reduce_max_8 + a
        // BLOCK_M-wide smem write per tile iteration.
        const int pv_row = tid >> 3;
        const int pv_lr  = tid & 7;

        // -- P·V loop over k sub-tiles of width MMA_K = 16. tile_size
        //    = 16 runs once; tile_size = 32 (sliding layers) runs twice.
        //    FIRST sub-tile folds the online-softmax alpha rescale into
        //    the MMA C operand; subsequent ones accumulate without
        //    re-scaling s_acc (alpha already baked in).
        const int k_sub_count = (tile_size + MMA_K - 1) / MMA_K;
        for (int ks_i = 0; ks_i < k_sub_count; ks_i++) {
            const int k_base_t = ks_i * MMA_K;
            if (k_base_t >= tile_len) break;

            // (1) Transpose V[t=k_base_t..k_base_t+MMA_K) into
            //     s_v_f16_T[d][0..MMA_K) with zero-pad past tile_len.
            for (int idx = tid; idx < MMA_K * head_dim; idx += FA2_THREADS) {
                const int k_off = idx / head_dim;
                const int d     = idx - k_off * head_dim;
                const int t_src = k_base_t + k_off;
                __half v = (t_src < tile_len)
                    ? s_v_f16[t_src * head_dim + d]
                    : __float2half(0.0f);
                s_v_f16_T[d * MMA_K + k_off] = v;
            }
            // Sync removed: P-pack below writes s_p_f16 and reads s_s,
            // doesn't read s_v_f16_T. The downstream sync after P-pack
            // gates both s_p_f16 + s_v_f16_T visibility for the MMA.

            // (2) Pack P[m, t=k_base_t..k_base_t+MMA_K) into s_p_f16
            //     [m, 0..MMA_K) directly. UN-normalized softmax
            //     numerator (P = exp(score - new_M)) — row sum
            //     accumulates into s_l[m] and is applied as 1/s_l in
            //     the epilogue, NOT before P*V. P ∈ [0, 1] so f16
            //     mantissa is sufficient (codex 52e review).
            //     The former s_p_scale was always 1.0; no inv-scale
            //     needed at this point.
            if (pv_row < BLOCK_M) {
                for (int t = pv_lr; t < MMA_K; t += 8) {
                    const int t_src = k_base_t + t;
                    const float p = (t_src < tile_len)
                        ? s_s[pv_row * tile_len + t_src]
                        : 0.0f;
                    s_p_f16[pv_row * MMA_K + t] = __float2half(p);
                }
            }
            __syncthreads();

            // (3) MMA. First ks_i applies alpha via C operand; later
            //     ks_i iterations pass alpha = 1 so s_acc only gets
            //     rescaled once per tile iteration.
            const int warp_id = tid >> 5;
            const int lane    = tid & 31;
            const int n_tiles_total    = head_dim >> 3;
            const int n_tiles_per_warp = n_tiles_total >> 2;
            uint32_t a[4];
            rvllm_f16mma::pack_a_frag_row_major_m16k16_f16(
                s_p_f16, /*stride_bytes=*/MMA_K * (int)sizeof(__half), a, lane);

            const int r_lo = lane >> 2;
            const int r_hi = r_lo + 8;
            const int c    = (lane & 3) << 1;
            const float alpha_lo = (ks_i == 0) ? s_alpha[r_lo] : 1.0f;
            const float alpha_hi = (ks_i == 0) ? s_alpha[r_hi] : 1.0f;

            #pragma unroll 1
            for (int nt = 0; nt < n_tiles_per_warp; nt++) {
                const int n_base = (warp_id * n_tiles_per_warp + nt) * 8;
                const int d_lo = n_base + c;
                const int d_hi = d_lo + 1;
                uint32_t b[2];
                rvllm_f16mma::pack_b_frag_col_major_n8k16_f16(
                    s_v_f16_T + n_base * MMA_K,
                    /*stride_bytes=*/MMA_K * (int)sizeof(__half), b, lane);
                // s_p_scale removed — load s_acc directly with alpha
                // applied (former pscale_lo / inv_ps_lo were identity).
                float d_frag[4];
                d_frag[0] = s_acc[r_lo * head_dim + d_lo] * alpha_lo;
                d_frag[1] = s_acc[r_lo * head_dim + d_hi] * alpha_lo;
                d_frag[2] = s_acc[r_hi * head_dim + d_lo] * alpha_hi;
                d_frag[3] = s_acc[r_hi * head_dim + d_hi] * alpha_hi;
                rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d_frag, a, b);
                s_acc[r_lo * head_dim + d_lo] = d_frag[0];
                s_acc[r_lo * head_dim + d_hi] = d_frag[1];
                s_acc[r_hi * head_dim + d_lo] = d_frag[2];
                s_acc[r_hi * head_dim + d_hi] = d_frag[3];
            }
            __syncthreads();
        }
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
