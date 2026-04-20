// Partial RoPE + NVFP4 paged-KV-cache write (Gemma 4).
//
// Sibling of `fused_rope_partial_fp8kv.cu` that stores KV as packed
// 4-bit values + per-16-element E4M3 microscales. Same RoPE math and
// split-half pairing as the FP8 kernel; only the quantise + write
// path changes.
//
// Differences vs the FP8 sibling:
//   * Thread-per-element (blockDim.x = head_dim) instead of
//     thread-per-pair — we need 16 cooperating threads to compute
//     one NVFP4 microscale via warp-shuffle reduction.
//   * Each thread handles both the "lo" dim (tid < half_head) and
//     the "hi" dim (tid >= half_head) of the split-half RoPE layout
//     by reading its own + its pair's input.
//   * Output fan-out:
//       key_cache_packed[cache_off + tid/2]   — half byte per thread
//       key_cache_scale [cache_off_scale + tid/16] — 1 E4M3 per 16
//     Same for V cache, plus Q FP8 output (unchanged; Q stays FP8
//     because the MMA input side of FA2 is still FP8).
//
// head_dim must be a multiple of 16 (NVFP4 block size). Gemma 4
// head_dim ∈ {256, 512} both satisfy.
//
// Q output (`q_fp8_out`) stays FP8 E4M3 with per-tensor scale.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "../../kernels/nvfp4_utils.cuh"

using rvllm_nvfp4::fp4_encode;

// Reduce the absolute-value peak across a 16-lane group within a warp.
// `lane_id` is `threadIdx.x & 31`; the group is determined by
// `lane_id >> 4` (upper 16 or lower 16 lanes of the warp).
__device__ __forceinline__ float block16_peak_abs(float v) {
    float a = fabsf(v);
    // Butterfly within 16-lane group: xor-shuffle with 8, 4, 2, 1.
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
    }
    return a;
}

extern "C"
__global__ void fused_rope_partial_nvfp4kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __nv_fp8_e4m3* __restrict__ q_fp8_out,
    uint8_t*       __restrict__ key_cache_packed,   // [..., head_dim/2]
    uint8_t*       __restrict__ value_cache_packed, // [..., head_dim/2]
    __nv_fp8_e4m3* __restrict__ key_cache_scale,    // [..., head_dim/16]
    __nv_fp8_e4m3* __restrict__ value_cache_scale,  // [..., head_dim/16]
    const __half*  __restrict__ cos_table,
    const __half*  __restrict__ sin_table,
    const int*     __restrict__ positions,
    const int*     __restrict__ slot_mapping,
    const float*   __restrict__ q_scale_ptr,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    if (tid >= head_dim) return;

    const int half_head    = head_dim >> 1;
    const int half_rotary  = rotary_dim >> 1;
    const int group_id     = tid >> 4;            // which 16-block inside head_dim
    const int groups_per_head = head_dim >> 4;
    const int lane_in_group = tid & 15;

    const int pos = positions[token_idx];

    // Compute RoPE'd output value for my dim.
    auto rope_one = [&] (const __half* in, int base) -> float {
        const bool is_lo = tid < half_head;
        const int pair   = is_lo ? tid + half_head : tid - half_head;
        const int freq   = is_lo ? tid : pair;

        float x_self = __half2float(in[base + tid]);
        if (freq < half_rotary) {
            float x_pair = __half2float(in[base + pair]);
            float cos_v  = __half2float(cos_table[pos * half_rotary + freq]);
            float sin_v  = __half2float(sin_table[pos * half_rotary + freq]);
            // Same matrix as the FP8 kernel:
            //   lo: x_lo*cos - x_hi*sin
            //   hi: x_lo*sin + x_hi*cos
            if (is_lo) return x_self * cos_v - x_pair * sin_v;
            else       return x_pair * sin_v + x_self * cos_v;
        }
        return x_self;  // pass-through (outside rotary range)
    };

    // ---- Q: stays FP8 per-tensor (MMA input side still fp8). ----
    if (head_idx < num_heads) {
        const int q_base = (token_idx * num_heads + head_idx) * head_dim;
        const float q_scale_inv = 1.0f / (*q_scale_ptr);
        float v = rope_one(q_in, q_base);
        q_fp8_out[q_base + tid] = __nv_fp8_e4m3(v * q_scale_inv);
    }

    // ---- K, V: NVFP4-packed cache write. ----
    if (head_idx < num_kv_heads) {
        const int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        const int slot   = slot_mapping[token_idx];
        if (slot < 0) return;

        const int cache_off_bytes  = (slot * num_kv_heads + head_idx) * (head_dim >> 1);
        const int cache_off_scales = (slot * num_kv_heads + head_idx) * groups_per_head;

        // Helper to pack one (K or V) head.
        auto quant_and_write = [&] (
            const __half* in,
            uint8_t*       out_packed,
            __nv_fp8_e4m3* out_scales,
            bool           apply_rope
        ) {
            float v = apply_rope ? rope_one(in, k_base)
                                 : __half2float(in[k_base + tid]);

            // Per-16-block peak → E4M3 scale.
            float peak = block16_peak_abs(v);
            float scale_f32 = (peak > 0.0f) ? (peak * (1.0f / 6.0f)) : 0.0f;
            __nv_fp8_e4m3 scale_e4m3 = __nv_fp8_e4m3(scale_f32);
            float scale = float(scale_e4m3);
            float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

            // First lane of each 16-group writes the scale.
            if (lane_in_group == 0) {
                out_scales[cache_off_scales + group_id] = scale_e4m3;
            }

            // Each thread encodes its own 4-bit nibble.
            uint32_t nibble = fp4_encode(v * inv_scale);

            // Pair lanes (even, odd) combine into one byte: low = even,
            // high = odd. Use shuffle to pull the odd nibble to the
            // even lane, then write one byte per even thread.
            uint32_t odd_nibble = __shfl_down_sync(0xFFFFFFFFu, nibble, 1);
            if ((tid & 1) == 0) {
                uint8_t byte = static_cast<uint8_t>((odd_nibble << 4) | (nibble & 0xFu));
                out_packed[cache_off_bytes + (tid >> 1)] = byte;
            }
        };

        quant_and_write(k_in, key_cache_packed,   key_cache_scale,   /*apply_rope=*/true);
        quant_and_write(v_in, value_cache_packed, value_cache_scale, /*apply_rope=*/false);
    }
}
