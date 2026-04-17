// Fused RoPE + FP8 paged-KV-cache write.
//
// Differences from fused_rope_cache_f16tbl.cu:
//   * q / k / v are read as __half (from the cuBLASLt QKV output)
//   * outputs q_fp8 (scratch), key_cache_fp8, value_cache_fp8 are FP8 E4M3
//   * two per-tensor scales (f32, on device) carry the runtime scaling
//   * q is scaled once with q_scale before writing (so FA3's q_descale
//     recovers the original magnitude); kv cache entries use kv_scale.
//
// Scale convention: FP8 value v_fp8 = clamp(v_f16 / scale, -448, 448).
// FA3 reconstructs v_eff = v_fp8 * q_descale / k_descale during softmax;
// we store the scale directly (FA3 treats `descale` as a multiplier).
//
// Per-tensor scales are currently treated as *static inputs* computed
// once at engine init; the bench bringup seeds them from a calibration
// pass. If dynamic re-scaling is added later, this kernel stays the same.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C"
__global__ void fused_rope_cache_fp8kv_kernel(
    const __half* __restrict__ q_in,         // [num_tokens, num_heads * head_dim]
    const __half* __restrict__ k_in,         // [num_tokens, num_kv_heads * head_dim]
    const __half* __restrict__ v_in,         // [num_tokens, num_kv_heads * head_dim]
    __nv_fp8_e4m3* __restrict__ q_fp8_out,   // [num_tokens, num_heads * head_dim]
    __nv_fp8_e4m3* __restrict__ key_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_fp8_e4m3* __restrict__ value_cache, // same layout as key_cache
    const __half* __restrict__ cos_table,    // [max_pos, half_dim]
    const __half* __restrict__ sin_table,    // [max_pos, half_dim]
    const int* __restrict__ positions,       // [num_tokens]
    const int* __restrict__ slot_mapping,    // [num_tokens]
    const float* __restrict__ q_scale_ptr,   // per-tensor f32
    const float* __restrict__ kv_scale_ptr,  // per-tensor f32 (shared by K and V)
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int half_dim  = head_dim / 2;
    const int tid       = threadIdx.x;
    if (tid >= half_dim) return;

    const float q_scale_inv = 1.0f / (*q_scale_ptr);
    const float kv_scale_inv = 1.0f / (*kv_scale_ptr);

    const int pos = positions[token_idx];
    const float cos_val = __half2float(cos_table[pos * half_dim + tid]);
    const float sin_val = __half2float(sin_table[pos * half_dim + tid]);

    // Q: apply RoPE + FP8-quantize to q_fp8_out.
    if (head_idx < num_heads) {
        int q_base = (token_idx * num_heads + head_idx) * head_dim;
        float q0 = __half2float(q_in[q_base + 2 * tid]);
        float q1 = __half2float(q_in[q_base + 2 * tid + 1]);
        float q0r = q0 * cos_val - q1 * sin_val;
        float q1r = q0 * sin_val + q1 * cos_val;
        q_fp8_out[q_base + 2 * tid]     = __nv_fp8_e4m3(q0r * q_scale_inv);
        q_fp8_out[q_base + 2 * tid + 1] = __nv_fp8_e4m3(q1r * q_scale_inv);
    }

    // K: RoPE + FP8-quantize into paged cache. V: FP8-quantize only.
    if (head_idx < num_kv_heads) {
        int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        float k0 = __half2float(k_in[k_base + 2 * tid]);
        float k1 = __half2float(k_in[k_base + 2 * tid + 1]);
        float k0r = k0 * cos_val - k1 * sin_val;
        float k1r = k0 * sin_val + k1 * cos_val;

        int slot = slot_mapping[token_idx];
        if (slot >= 0) {
            int cache_offset = (slot * num_kv_heads + head_idx) * head_dim;
            key_cache[cache_offset + 2 * tid]     = __nv_fp8_e4m3(k0r * kv_scale_inv);
            key_cache[cache_offset + 2 * tid + 1] = __nv_fp8_e4m3(k1r * kv_scale_inv);

            int v_base = (token_idx * num_kv_heads + head_idx) * head_dim;
            float v0 = __half2float(v_in[v_base + 2 * tid]);
            float v1 = __half2float(v_in[v_base + 2 * tid + 1]);
            value_cache[cache_offset + 2 * tid]     = __nv_fp8_e4m3(v0 * kv_scale_inv);
            value_cache[cache_offset + 2 * tid + 1] = __nv_fp8_e4m3(v1 * kv_scale_inv);
        }
    }
}
