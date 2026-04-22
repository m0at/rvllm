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

__device__ float block_reduce_sum(float val, float* smem_reduce, int tid, int num_threads) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (num_threads + 31) / 32;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem_reduce[warp_id] = val;
    __syncthreads();
    if (tid < num_warps) val = smem_reduce[tid]; else val = 0.0f;
    if (tid < 32) val = warp_reduce_sum(val);
    return val;
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
    // ------------------------------------------------------------------
    // Phase-A skeleton. The body lands in Phase B. Early-return so a
    // host-side launcher wiring PR can land + link + cover the compile
    // error surface before numerical work starts.
    // ------------------------------------------------------------------
    (void)output; (void)query; (void)key_cache; (void)value_cache;
    (void)k_scale_cache; (void)v_scale_cache; (void)q_scale_cache;
    (void)k_descale_fallback; (void)v_descale_fallback;
    (void)block_tables; (void)cu_seqlens_q; (void)context_lens;
    (void)q_descale; (void)scale; (void)num_heads; (void)num_kv_heads;
    (void)head_dim; (void)block_size; (void)max_blocks_per_seq;
    (void)tile_size; (void)num_queries_per_kv; (void)block_q;
    (void)num_seqs; (void)window_size_left;
    return;
}
