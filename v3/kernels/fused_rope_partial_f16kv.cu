// Partial RoPE + F16 paged-KV-cache write (Gemma 4, F16 mode).
// Same as fused_rope_partial_fp8kv but outputs F16 Q and F16 KV cache.
// No FP8 quantization -- full F16 precision throughout.

#include <cuda_fp16.h>

extern "C"
__global__ void fused_rope_partial_f16kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __half* __restrict__ q_out,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int half_rotary = rotary_dim / 2;
    const int half_head   = head_dim / 2;
    const int tid         = threadIdx.x;
    if (tid >= half_head) return;

    const int pos = positions[token_idx];

    // Q head: rotate only the first rotary_dim values, then pass through the tail.
    if (head_idx < num_heads) {
        int q_base = (token_idx * num_heads + head_idx) * head_dim;

        if (tid < half_rotary) {
            float cos_val = __half2float(cos_table[pos * half_rotary + tid]);
            float sin_val = __half2float(sin_table[pos * half_rotary + tid]);
            float q_lo = __half2float(q_in[q_base + tid]);
            float q_hi = __half2float(q_in[q_base + tid + half_rotary]);
            q_out[q_base + tid]             = __float2half(q_lo * cos_val - q_hi * sin_val);
            q_out[q_base + tid + half_rotary] = __float2half(q_lo * sin_val + q_hi * cos_val);
        } else {
            int tail_idx = rotary_dim + 2 * (tid - half_rotary);
            q_out[q_base + tail_idx] = q_in[q_base + tail_idx];
            q_out[q_base + tail_idx + 1] = q_in[q_base + tail_idx + 1];
        }
    }

    // K head: rotate only the first rotary_dim values, then pass through the tail.
    // V head: direct F16 cache write.
    if (head_idx < num_kv_heads) {
        int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        int slot = slot_mapping[token_idx];

        if (slot >= 0) {
            int cache_offset = (slot * num_kv_heads + head_idx) * head_dim;

            if (tid < half_rotary) {
                float cos_val = __half2float(cos_table[pos * half_rotary + tid]);
                float sin_val = __half2float(sin_table[pos * half_rotary + tid]);
                float k_lo = __half2float(k_in[k_base + tid]);
                float k_hi = __half2float(k_in[k_base + tid + half_rotary]);
                key_cache[cache_offset + tid]             = __float2half(k_lo * cos_val - k_hi * sin_val);
                key_cache[cache_offset + tid + half_rotary] = __float2half(k_lo * sin_val + k_hi * cos_val);
            } else {
                int tail_idx = rotary_dim + 2 * (tid - half_rotary);
                key_cache[cache_offset + tail_idx] = k_in[k_base + tail_idx];
                key_cache[cache_offset + tail_idx + 1] = k_in[k_base + tail_idx + 1];
            }

            // V: no rotation, direct F16 copy
            int v_base = (token_idx * num_kv_heads + head_idx) * head_dim;
            value_cache[cache_offset + tid]             = v_in[v_base + tid];
            value_cache[cache_offset + tid + half_head] = v_in[v_base + tid + half_head];
        }
    }
}
