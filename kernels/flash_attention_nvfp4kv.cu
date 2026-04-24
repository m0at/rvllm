// FP8-Q / NVFP4-KV paged-attention decode for Gemma 4 on sm_121.
//
// Sibling of `flash_attention_2_decode_fp8kv_kernel` (flash_attention.cu)
// that reads K/V from a packed 4-bit cache + per-16-element E4M3
// microscale instead of dense FP8 E4M3. Q stays FP8 per-tensor; the
// kernel dequants K and V into the same f32-in-smem layout that the
// FP8 decode uses, then runs the identical online-softmax inner
// loop. Writing a separate .cu rather than extending the existing
// file keeps the FP8 code path untouched + lets `build.sh` emit a
// separate PTX module that the Rust side loads alongside
// `flash_attention.ptx`.
//
// Smem layout (per tile of `FA2_BC` KV rows):
//   s_key    [FA2_BC * head_dim]   f16   — K dequant scratch (Phase 2a)
//   s_val    [FA2_BC * head_dim]   f16   — V dequant scratch (Phase 2a)
//   s_score  [FA2_BC]              f32   — per-row Q·K^T scores
//   s_reduce [FA2_THREADS / 32]    f32   — block-reduce scratch
//
// Phase 2a of task aa01001nvf4f16mma: K/V dequant target switched
// from f32 to f16 to halve the smem footprint. Q·K^T and P·V inner
// loops read `__half2float(s_key[i])` / `__half2float(s_val[i])`
// per element — a single-cycle hw cvt. MMA integration (Phase 2b)
// is scoped to the prefill kernel where m > 1 makes the MMA fragment
// shape a real win; at decode m=1 the MMA forces 15/16 dummy rows
// so scalar FMA over f16 smem stays competitive.
//
// Build: same flow as flash_attention.cu (kernels/build.sh picks up
// `*.cu` automatically). On sm_121 use `-arch=sm_121a` as elsewhere
// in this branch.

#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "nvfp4_utils.cuh"

#ifndef FA2_BC
#  define FA2_BC 32
#endif
#define FA2_THREADS 128

// ---------------------------------------------------------------------
// Helpers — intentionally duplicated from flash_attention.cu rather than
// extracted to a shared header. The extract is a separate refactor that
// would touch the FP8 path, which this branch explicitly does not.
// ---------------------------------------------------------------------

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

// Dequant one NVFP4-packed KV row of `head_dim` elements into the
// shared-mem f16 buffer `s_dst` (row starts at `s_dst`, all head_dim
// entries). `packed_row` is the 4-bit byte stream, `scale_row` is the
// head_dim/16 E4M3 microscales.
//
// Fast path: the first `head_dim/16` threads in the block each claim
// one 16-element block (8 packed bytes → one u64 LDG → 4× native
// `cvt.rn.f16x2.e2m1x2` → 8× scale multiply + cast to f16 → 8 smem
// stores of f16x2). ~6× faster than the original thread-per-element
// switch-decode. Remaining threads (>= head_dim/16) are idle for
// this load — the Q·K^T dot-product downstream is thread-per-dim
// so they pick work back up there.
__device__ __forceinline__ void dequant_nvfp4_row_to_smem(
    const uint8_t*       __restrict__ packed_row,   // head_dim / 2 bytes
    const __nv_fp8_e4m3* __restrict__ scale_row,    // head_dim / 16 E4M3
    __half*              __restrict__ s_dst,        // head_dim halves
    int tid,
    int head_dim
) {
    const int num_blocks = head_dim >> 4;   // 16 elems per block
    if (tid < num_blocks) {
        // Each thread owns one 16-element block of this row.
        const int block_idx = tid;
        const int d_base    = block_idx << 4;
        uint64_t packed8 = *reinterpret_cast<const uint64_t*>(packed_row + (d_base >> 1));
        float    scale   = float(scale_row[block_idx]);
        rvllm_nvfp4::unpack16_nvfp4_to_f16_fast(packed8, scale, s_dst + d_base);
    }
}

// ---------------------------------------------------------------------
// Decode kernel — FP8 Q, NVFP4 K/V cache. Accumulator in f32, output
// in f16. Grid: (num_seqs, num_heads). Block: (FA2_THREADS, 1, 1).
// Dynamic smem: (2 * FA2_BC * head_dim + FA2_BC + FA2_THREADS/32) * 4.
// ---------------------------------------------------------------------
extern "C"
__global__ void flash_attention_2_decode_nvfp4kv_kernel(
    __half*              __restrict__ output,                 // [S, H, D] f16
    const unsigned char* __restrict__ query,                  // [S, H, D] fp8
    const uint8_t*       __restrict__ key_cache_packed,       // [B*bs*KH*D/2] u8
    const uint8_t*       __restrict__ value_cache_packed,     // [B*bs*KH*D/2] u8
    const __nv_fp8_e4m3* __restrict__ key_cache_scale,        // [B*bs*KH*D/16] e4m3
    const __nv_fp8_e4m3* __restrict__ value_cache_scale,      // [B*bs*KH*D/16] e4m3
    const int*           __restrict__ block_tables,           // [S, max_blocks]
    const int*           __restrict__ context_lens,           // [S]
    const float*         __restrict__ q_descale,              // single f32
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int /*window_size_left*/
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const float q_scale = *q_descale;

    // Smem layout: f16 K/V tiles + f32 score + f32 reduce scratch.
    // Allocate the f16 region first (alignment-friendly for f16x2
    // loads); stash pointer arithmetic on byte offsets so the f32
    // tail lands 4-byte aligned regardless of FA2_BC.
    extern __shared__ unsigned char smem_u8[];
    __half* s_key = reinterpret_cast<__half*>(smem_u8);
    __half* s_val = s_key + FA2_BC * head_dim;
    float*  s_score  = reinterpret_cast<float*>(s_val + FA2_BC * head_dim);
    float*  s_reduce = s_score + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
    const int half_D = head_dim >> 1;          // packed bytes per row
    const int scales_per_D = head_dim >> 4;    // E4M3 scales per row

    // Load Q row (FP8) with descale + attention scale folded in.
    // Q stays in f32 regs for the dot product — cheap (single cvt)
    // and keeps the softmax accumulator in f32 precision.
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

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        const int tile_start = tile * FA2_BC;
        const int tile_len   = min(FA2_BC, context_len - tile_start);

        // --- Load + dequant K tile (NVFP4 → f16 smem). -------------
        // First `head_dim/16` threads do the dequant (one 16-element
        // block each); the rest are idle here and pick work back up
        // in the thread-per-dim Q·K^T below.
        for (int t = 0; t < tile_len; ++t) {
            int kv_pos    = tile_start + t;
            int page_idx  = kv_pos / block_size;
            int page_off  = kv_pos % block_size;
            int phys_blk  = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const uint8_t*       k_packed = key_cache_packed
                + ((phys_blk * block_size + page_off) * num_kv_heads
                    + kv_head_idx) * half_D;
            const __nv_fp8_e4m3* k_scale  = key_cache_scale
                + ((phys_blk * block_size + page_off) * num_kv_heads
                    + kv_head_idx) * scales_per_D;
            dequant_nvfp4_row_to_smem(k_packed, k_scale,
                                      s_key + t * head_dim,
                                      tid, head_dim);
        }
        __syncthreads();

        // --- Q · K^T per column of the tile. -----------------------
        // f16 smem read + cvt-to-f32 per element keeps the dot in
        // f32 precision. The cvt is free in practice (single-cycle
        // hw op on sm_120+).
        for (int t = 0; t < tile_len; ++t) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = tid + r * FA2_THREADS;
                if (r < dims_per_thread && d < head_dim) {
                    dot += q_reg[r]
                         * __half2float(s_key[t * head_dim + d]);
                }
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) s_score[t] = dot;
            __syncthreads();
        }

        // --- Online-softmax update. --------------------------------
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
                float v = expf(s_score[t] - row_max);
                s_score[t] = v;
                tsum += v;
            }
            s_reduce[0] = tsum;
        }
        __syncthreads();
        row_sum += s_reduce[0];
        __syncthreads();

        // --- Load + dequant V tile (NVFP4 → f16 smem). -------------
        for (int t = 0; t < tile_len; ++t) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            const uint8_t*       v_packed = value_cache_packed
                + ((phys_blk * block_size + page_off) * num_kv_heads
                    + kv_head_idx) * half_D;
            const __nv_fp8_e4m3* v_scale  = value_cache_scale
                + ((phys_blk * block_size + page_off) * num_kv_heads
                    + kv_head_idx) * scales_per_D;
            dequant_nvfp4_row_to_smem(v_packed, v_scale,
                                      s_val + t * head_dim,
                                      tid, head_dim);
        }
        __syncthreads();

        // --- P · V accumulate. -------------------------------------
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; ++t) {
                    val_acc += s_score[t]
                             * __half2float(s_val[t * head_dim + d]);
                }
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    // Normalise + cast to f16 output.
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = tid + r * FA2_THREADS;
        if (r < dims_per_thread && d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] =
                __float2half(acc[r] * inv_sum);
        }
    }
}

// ---------------------------------------------------------------------
// BC=16 variant for head_dim=512 (Gemma 4 global attention) — same
// smem-budget reason as the FP8 kernel's _bc16 sibling. Built with
// `-DFA2_BC=16` by the kernels/build.sh dispatch (or manually on
// standalone compiles). A separate extern-C symbol keeps the
// launcher's decision about which kernel to use static.
// ---------------------------------------------------------------------

#if FA2_BC == 32
// When building with the default BC=32, also emit a BC=16 variant
// from the same TU. Recompiling the whole body with FA2_BC overridden
// locally — the include guard on the helpers above lets both coexist
// in the same PTX.
#  undef FA2_BC
#  define FA2_BC 16

extern "C"
__global__ void flash_attention_2_decode_nvfp4kv_bc16_kernel(
    __half*              __restrict__ output,
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
    int window_size_left
) {
    // Delegate to the main kernel body — but we can't literally call a
    // __global__ from another __global__. Duplicate the body instead;
    // the FA2_BC macro is what controls smem layout + loop bounds, so
    // copy-paste is the correct way to get a distinct entry point.
    //
    // NOTE: keep this body byte-identical to the BC=32 version above;
    // the macro drives the only real difference (smem-tile size).

    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;
    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));
    const float q_scale = *q_descale;

    extern __shared__ unsigned char smem_u8[];
    __half* s_key = reinterpret_cast<__half*>(smem_u8);
    __half* s_val = s_key + FA2_BC * head_dim;
    float*  s_score  = reinterpret_cast<float*>(s_val + FA2_BC * head_dim);
    float*  s_reduce = s_score + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
    const int half_D = head_dim >> 1;
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

    float row_max = -FLT_MAX, row_sum = 0.0f;
    float acc[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = 0.0f;

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        const int tile_start = tile * FA2_BC;
        const int tile_len   = min(FA2_BC, context_len - tile_start);

        for (int t = 0; t < tile_len; ++t) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
            dequant_nvfp4_row_to_smem(
                key_cache_packed + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D,
                key_cache_scale  + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D,
                s_key + t * head_dim, tid, head_dim);
        }
        __syncthreads();

        for (int t = 0; t < tile_len; ++t) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = tid + r * FA2_THREADS;
                if (r < dims_per_thread && d < head_dim)
                    dot += q_reg[r] * __half2float(s_key[t * head_dim + d]);
            }
            dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
            if (tid == 0) s_score[t] = dot;
            __syncthreads();
        }

        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; ++t) tile_max = fmaxf(tile_max, s_score[t]);
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
                float v = expf(s_score[t] - row_max);
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
            dequant_nvfp4_row_to_smem(
                value_cache_packed + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D,
                value_cache_scale  + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D,
                s_val + t * head_dim, tid, head_dim);
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                float val_acc = 0.0f;
                for (int t = 0; t < tile_len; ++t)
                    val_acc += s_score[t] * __half2float(s_val[t * head_dim + d]);
                acc[r] += val_acc;
            }
        }
        __syncthreads();
    }

    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = tid + r * FA2_THREADS;
        if (r < dims_per_thread && d < head_dim) {
            output[(seq_idx * num_heads + head_idx) * head_dim + d] =
                __float2half(acc[r] * inv_sum);
        }
    }
    (void)window_size_left;
}
#endif  // FA2_BC == 32 originally

// ---------------------------------------------------------------------
// FP8-Q / NVFP4-KV paged-prefill — multi-query self-attention with
// causal mask. Sibling of `flash_attention_2_prefill_fp8kv_kernel` in
// flash_attention.cu. Structural deltas vs the decode kernel above:
//   (1) wrap the whole softmax+accum in `for qi in 0..q_len`;
//   (2) query row indexed via `q_start = cu_seqlens_q[seq_idx] + qi`;
//   (3) causal mask: KV positions > `context_len - q_len + qi` score
//       -FLT_MAX before softmax;
//   (4) output row written at `q_pos_global`.
//
// Grid: (num_seqs, num_heads). Block: (FA2_THREADS, 1, 1). Smem:
// same layout as decode (2*FA2_BC*head_dim + FA2_BC + FA2_THREADS/32
// floats).
// ---------------------------------------------------------------------
#if FA2_BC == 16
// We're in the BC=16 redefine block above. Reset the macro back to
// the original default for the prefill entries so the BC=32 prefill
// + its BC=16 sibling below both get their own tile size.
#  undef FA2_BC
#  define FA2_BC 32
#endif

extern "C"
__global__ void flash_attention_2_prefill_nvfp4kv_kernel(
    __half*              __restrict__ output,                 // [num_q_tokens, H, D]
    const unsigned char* __restrict__ query,                  // [num_q_tokens, H, D] fp8
    const uint8_t*       __restrict__ key_cache_packed,
    const uint8_t*       __restrict__ value_cache_packed,
    const __nv_fp8_e4m3* __restrict__ key_cache_scale,
    const __nv_fp8_e4m3* __restrict__ value_cache_scale,
    const int*           __restrict__ block_tables,
    const int*           __restrict__ context_lens,
    const int*           __restrict__ cu_seqlens_q,
    const float*         __restrict__ q_descale,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int /*window_size_left*/
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int q_start = cu_seqlens_q[seq_idx];
    const int q_end   = cu_seqlens_q[seq_idx + 1];
    const int q_len   = q_end - q_start;
    if (q_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const float q_scale = *q_descale;

    extern __shared__ unsigned char smem_u8[];
    __half* s_key = reinterpret_cast<__half*>(smem_u8);
    __half* s_val = s_key + FA2_BC * head_dim;
    float*  s_score  = reinterpret_cast<float*>(s_val + FA2_BC * head_dim);
    float*  s_reduce = s_score + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
    const int half_D = head_dim >> 1;
    const int scales_per_D = head_dim >> 4;

    for (int qi = 0; qi < q_len; ++qi) {
        const int q_pos_global = q_start + qi;

        float q_reg[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                unsigned char qb = query[(q_pos_global * num_heads + head_idx) * head_dim + d];
                q_reg[r] = fp8kv_decode_byte(qb) * q_scale * scale;
            } else {
                q_reg[r] = 0.0f;
            }
        }

        float row_max = -FLT_MAX, row_sum = 0.0f;
        float acc[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) acc[r] = 0.0f;

        const int q_abs_kv_pos = context_len - q_len + qi;

        for (int tile = 0; tile < num_kv_tiles; ++tile) {
            const int tile_start = tile * FA2_BC;
            const int tile_len   = min(FA2_BC, context_len - tile_start);

            for (int t = 0; t < tile_len; ++t) {
                int kv_pos   = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                dequant_nvfp4_row_to_smem(
                    key_cache_packed + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D,
                    key_cache_scale  + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D,
                    s_key + t * head_dim, tid, head_dim);
            }
            __syncthreads();

            // Q·K^T + causal mask.
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
                    s_score[t] = (kv_pos > q_abs_kv_pos) ? -FLT_MAX : dot;
                }
                __syncthreads();
            }

            // Online softmax — skip masked entries on the exp path.
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
                dequant_nvfp4_row_to_smem(
                    value_cache_packed + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D,
                    value_cache_scale  + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D,
                    s_val + t * head_dim, tid, head_dim);
            }
            __syncthreads();

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = tid + r * FA2_THREADS;
                if (r < dims_per_thread && d < head_dim) {
                    float val_acc = 0.0f;
                    for (int t = 0; t < tile_len; ++t)
                        val_acc += s_score[t] * __half2float(s_val[t * head_dim + d]);
                    acc[r] += val_acc;
                }
            }
            __syncthreads();
        }

        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                output[(q_pos_global * num_heads + head_idx) * head_dim + d] =
                    __float2half(acc[r] * inv_sum);
            }
        }
    }
}

#undef FA2_BC
#define FA2_BC 16

extern "C"
__global__ void flash_attention_2_prefill_nvfp4kv_bc16_kernel(
    __half*              __restrict__ output,
    const unsigned char* __restrict__ query,
    const uint8_t*       __restrict__ key_cache_packed,
    const uint8_t*       __restrict__ value_cache_packed,
    const __nv_fp8_e4m3* __restrict__ key_cache_scale,
    const __nv_fp8_e4m3* __restrict__ value_cache_scale,
    const int*           __restrict__ block_tables,
    const int*           __restrict__ context_lens,
    const int*           __restrict__ cu_seqlens_q,
    const float*         __restrict__ q_descale,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int /*window_size_left*/
) {
    // Body identical to the BC=32 prefill above; FA2_BC macro drives
    // the only real difference (tile size → smem layout + loop bounds).
    // Keep the two bodies byte-identical; changes must land in both.

    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;
    const int q_start = cu_seqlens_q[seq_idx];
    const int q_end   = cu_seqlens_q[seq_idx + 1];
    const int q_len   = q_end - q_start;
    if (q_len == 0) return;
    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));
    const float q_scale = *q_descale;

    extern __shared__ unsigned char smem_u8[];
    __half* s_key = reinterpret_cast<__half*>(smem_u8);
    __half* s_val = s_key + FA2_BC * head_dim;
    float*  s_score  = reinterpret_cast<float*>(s_val + FA2_BC * head_dim);
    float*  s_reduce = s_score + FA2_BC;

    const int num_kv_tiles = (context_len + FA2_BC - 1) / FA2_BC;
    const int dims_per_thread = (head_dim + FA2_THREADS - 1) / FA2_THREADS;
    const int half_D = head_dim >> 1;
    const int scales_per_D = head_dim >> 4;

    for (int qi = 0; qi < q_len; ++qi) {
        const int q_pos_global = q_start + qi;
        float q_reg[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                unsigned char qb = query[(q_pos_global * num_heads + head_idx) * head_dim + d];
                q_reg[r] = fp8kv_decode_byte(qb) * q_scale * scale;
            } else q_reg[r] = 0.0f;
        }
        float row_max = -FLT_MAX, row_sum = 0.0f;
        float acc[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) acc[r] = 0.0f;
        const int q_abs_kv_pos = context_len - q_len + qi;

        for (int tile = 0; tile < num_kv_tiles; ++tile) {
            const int tile_start = tile * FA2_BC;
            const int tile_len   = min(FA2_BC, context_len - tile_start);

            for (int t = 0; t < tile_len; ++t) {
                int kv_pos   = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_blk = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                dequant_nvfp4_row_to_smem(
                    key_cache_packed + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D,
                    key_cache_scale  + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D,
                    s_key + t * head_dim, tid, head_dim);
            }
            __syncthreads();

            for (int t = 0; t < tile_len; ++t) {
                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int d = tid + r * FA2_THREADS;
                    if (r < dims_per_thread && d < head_dim)
                        dot += q_reg[r] * __half2float(s_key[t * head_dim + d]);
                }
                dot = block_reduce_sum(dot, s_reduce, tid, FA2_THREADS);
                if (tid == 0) {
                    int kv_pos = tile_start + t;
                    s_score[t] = (kv_pos > q_abs_kv_pos) ? -FLT_MAX : dot;
                }
                __syncthreads();
            }

            float tile_max = -FLT_MAX;
            if (tid == 0) {
                for (int t = 0; t < tile_len; ++t) tile_max = fmaxf(tile_max, s_score[t]);
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
                dequant_nvfp4_row_to_smem(
                    value_cache_packed + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * half_D,
                    value_cache_scale  + ((phys_blk * block_size + page_off) * num_kv_heads + kv_head_idx) * scales_per_D,
                    s_val + t * head_dim, tid, head_dim);
            }
            __syncthreads();

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = tid + r * FA2_THREADS;
                if (r < dims_per_thread && d < head_dim) {
                    float val_acc = 0.0f;
                    for (int t = 0; t < tile_len; ++t)
                        val_acc += s_score[t] * __half2float(s_val[t * head_dim + d]);
                    acc[r] += val_acc;
                }
            }
            __syncthreads();
        }

        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int d = tid + r * FA2_THREADS;
            if (r < dims_per_thread && d < head_dim) {
                output[(q_pos_global * num_heads + head_idx) * head_dim + d] =
                    __float2half(acc[r] * inv_sum);
            }
        }
    }
}
