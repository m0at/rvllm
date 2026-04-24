// Compute dynamic per-token FP8 scales for Q, K, and V.
// Scans all heads for one token to find absmax, writes scale = absmax / 448.
// Runs BEFORE fused_rope_partial_fp8kv so the RoPE kernel can use dynamic scales.
//
// Grid:  (num_tokens, 1, 1)
// Block: (min(head_dim, 1024), 1, 1)
// Output: q_scales[num_tokens], k_scales[num_tokens], v_scales[num_tokens]

#include <cuda_fp16.h>

#define FP8_E4M3_MAX 448.0f
#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_max(val);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

extern "C" __global__ void __launch_bounds__(1024)
compute_qkv_scales_kernel(
    float* __restrict__ q_scales,
    float* __restrict__ k_scales,
    float* __restrict__ v_scales,
    const __half* __restrict__ q_normed,
    const __half* __restrict__ k_normed,
    const __half* __restrict__ v_normed,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const int* __restrict__ positions,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int half_head = head_dim / 2;
    const int half_rotary = rotary_dim / 2;
    const int pos = positions[token];

    __shared__ float smem[WARPS_MAX];

    // Q absmax: scan all heads, apply rotation, find max
    float q_max = 0.0f;
    for (int h = 0; h < num_heads; h++) {
        int base = (token * num_heads + h) * head_dim;
        for (int i = tid; i < half_head; i += blockDim.x) {
            float lo = __half2float(q_normed[base + i]);
            float hi = __half2float(q_normed[base + i + half_head]);
            if (i < half_rotary) {
                float c = __half2float(cos_table[pos * half_rotary + i]);
                float s = __half2float(sin_table[pos * half_rotary + i]);
                float r_lo = lo * c - hi * s;
                float r_hi = lo * s + hi * c;
                q_max = fmaxf(q_max, fmaxf(fabsf(r_lo), fabsf(r_hi)));
            } else {
                q_max = fmaxf(q_max, fmaxf(fabsf(lo), fabsf(hi)));
            }
        }
    }
    float q_absmax = block_reduce_max(q_max, smem);
    if (tid == 0) {
        float scale = fmaxf(q_absmax / FP8_E4M3_MAX, 1e-12f);
        q_scales[token] = scale;
    }

    // K absmax: scan all KV heads, apply rotation
    float k_max = 0.0f;
    for (int h = 0; h < num_kv_heads; h++) {
        int base = (token * num_kv_heads + h) * head_dim;
        for (int i = tid; i < half_head; i += blockDim.x) {
            float lo = __half2float(k_normed[base + i]);
            float hi = __half2float(k_normed[base + i + half_head]);
            if (i < half_rotary) {
                float c = __half2float(cos_table[pos * half_rotary + i]);
                float s = __half2float(sin_table[pos * half_rotary + i]);
                float r_lo = lo * c - hi * s;
                float r_hi = lo * s + hi * c;
                k_max = fmaxf(k_max, fmaxf(fabsf(r_lo), fabsf(r_hi)));
            } else {
                k_max = fmaxf(k_max, fmaxf(fabsf(lo), fabsf(hi)));
            }
        }
    }
    float k_absmax = block_reduce_max(k_max, smem);

    // V absmax: scan all KV heads, no rotation
    float v_max = 0.0f;
    for (int h = 0; h < num_kv_heads; h++) {
        int base = (token * num_kv_heads + h) * head_dim;
        for (int i = tid; i < half_head; i += blockDim.x) {
            float lo = __half2float(v_normed[base + i]);
            float hi = __half2float(v_normed[base + i + half_head]);
            v_max = fmaxf(v_max, fmaxf(fabsf(lo), fabsf(hi)));
        }
    }
    float v_absmax = block_reduce_max(v_max, smem);
    if (tid == 0) {
        if (k_scales == v_scales) {
            // Gemma 4 currently threads one shared KV descale into attention.
            // When K/V outputs alias, preserve the larger absmax so neither
            // cache clips before decode.
            float kv_absmax = fmaxf(k_absmax, v_absmax);
            float kv_scale = fmaxf(kv_absmax / FP8_E4M3_MAX, 1e-12f);
            k_scales[token] = kv_scale;
        } else {
            float k_scale = fmaxf(k_absmax / FP8_E4M3_MAX, 1e-12f);
            float v_scale = fmaxf(v_absmax / FP8_E4M3_MAX, 1e-12f);
            k_scales[token] = k_scale;
            v_scales[token] = v_scale;
        }
    }
}
