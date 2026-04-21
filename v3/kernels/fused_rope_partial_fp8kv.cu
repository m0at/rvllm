// Partial RoPE + FP8 paged-KV-cache write (Gemma 4).
//
// Per-slot-per-head KV scales: instead of a single global `kv_scale`
// applied to all K/V cache entries, each (slot, kv_head) pair gets
// its own f32 scale = amax_of_that_head / 448. This recovers the full
// ~7-bit FP8 E4M3 resolution on a per-entry basis, eliminating the
// per-tensor calibration guess the old kernel relied on.
//
// Q still uses a per-tensor scale: Q is never cached — it's consumed
// by attention in this same step — so a single step's Q scale is
// sufficient and matches what the per-tensor calibration was already
// correctly doing.
//
// Gemma 4 global attention layers use partial_rotary_factor=0.25
// (64/256 dims rotated). Sliding layers use 0.5 (128/256). The
// cos/sin tables are pre-sized to rotary_dim/2.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

// FP8 E4M3 max magnitude (7-bit grid, symmetric).
#define FP8_E4M3_MAX 448.0f

// Block-wide max-abs reduction across `half_head` threads. Uses
// `warp_max[]` in shared memory for the warp-to-warp step; returns
// the reduction in every thread's register.
__device__ __forceinline__
float block_max_abs(float x, int half_head, int tid, float* warp_max) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, off));
    }
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    if (lane == 0) warp_max[warp_id] = x;
    __syncthreads();
    int num_warps = (half_head + 31) >> 5;
    if (warp_id == 0) {
        float y = (lane < num_warps) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y = fmaxf(y, __shfl_xor_sync(0xffffffff, y, off));
        }
        if (lane == 0) warp_max[0] = y;
    }
    __syncthreads();
    float out = warp_max[0];
    __syncthreads(); // caller may reuse `warp_max` next
    return out;
}

extern "C"
__global__ void fused_rope_partial_fp8kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __nv_fp8_e4m3* __restrict__ q_fp8_out,
    __nv_fp8_e4m3* __restrict__ key_cache,
    __nv_fp8_e4m3* __restrict__ value_cache,
    float* __restrict__ k_scale_cache,    // [num_slots * num_kv_heads] f32 (per-slot-per-head K scale)
    float* __restrict__ v_scale_cache,    // [num_slots * num_kv_heads] f32 (per-slot-per-head V scale)
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    const float* __restrict__ q_scale_ptr,
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

    const float q_scale_inv = 1.0f / (*q_scale_ptr);
    const int pos = positions[token_idx];
    const bool q_perblock = (q_scale_cache != nullptr);

    __shared__ float warp_max[32];

    // =============== Q head ===============
    // Per-tensor scale is fine: Q is consumed by THIS step's attention,
    // never written to the KV cache.
    if (head_idx < num_heads) {
        int q_base = (token_idx * num_heads + head_idx) * head_dim;

        float q_lo_val, q_hi_val;
        if (tid < half_rotary) {
            float cos_val = __half2float(cos_table[pos * half_rotary + tid]);
            float sin_val = __half2float(sin_table[pos * half_rotary + tid]);
            float q_lo = __half2float(q_in[q_base + tid]);
            float q_hi = __half2float(q_in[q_base + tid + half_head]);
            q_lo_val = q_lo * cos_val - q_hi * sin_val;
            q_hi_val = q_lo * sin_val + q_hi * cos_val;
        } else {
            float q_lo = __half2float(q_in[q_base + tid]);
            float q_hi = __half2float(q_in[q_base + tid + half_head]);
            q_fp8_out[q_base + tid]             = __nv_fp8_e4m3(q_lo * q_scale_inv);
            q_fp8_out[q_base + tid + half_head] = __nv_fp8_e4m3(q_hi * q_scale_inv);
        }

        float q_scale;
        if (q_perblock) {
            // Block-reduce amax across this (token, head) pair to
            // derive a dynamic scale.
            float q_amax = block_max_abs(
                fmaxf(fabsf(q_lo_val), fabsf(q_hi_val)),
                half_head, tid, warp_max);
            q_scale = fmaxf(q_amax / FP8_E4M3_MAX, 1e-12f);
            if (tid == 0) {
                q_scale_cache[token_idx * num_heads + head_idx] = q_scale;
            }
        } else {
            q_scale = *q_scale_ptr;
        }
        float q_inv = 1.0f / q_scale;
        q_fp8_out[q_base + tid]             = __nv_fp8_e4m3(q_lo_val * q_inv);
        q_fp8_out[q_base + tid + half_head] = __nv_fp8_e4m3(q_hi_val * q_inv);
    }

    // =============== K / V head (per-slot-per-head scale) ===============
    if (head_idx < num_kv_heads) {
        int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        int v_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        int slot   = slot_mapping[token_idx];
        if (slot < 0) return;
        int cache_offset = (slot * num_kv_heads + head_idx) * head_dim;
        int scale_idx    = slot * num_kv_heads + head_idx;

        // --- Pass 1: compute post-RoPE K values, reduce to K amax. ---
        float k_lo_val, k_hi_val;
        if (tid < half_rotary) {
            float cos_val = __half2float(cos_table[pos * half_rotary + tid]);
            float sin_val = __half2float(sin_table[pos * half_rotary + tid]);
            float k_lo = __half2float(k_in[k_base + tid]);
            float k_hi = __half2float(k_in[k_base + tid + half_head]);
            k_lo_val = k_lo * cos_val - k_hi * sin_val;
            k_hi_val = k_lo * sin_val + k_hi * cos_val;
        } else {
            k_lo_val = __half2float(k_in[k_base + tid]);
            k_hi_val = __half2float(k_in[k_base + tid + half_head]);
        }
        float k_amax = block_max_abs(
            fmaxf(fabsf(k_lo_val), fabsf(k_hi_val)),
            half_head, tid, warp_max);

        // --- Pass 2: compute V values, reduce to V amax. ---
        float v_lo_val = __half2float(v_in[v_base + tid]);
        float v_hi_val = __half2float(v_in[v_base + tid + half_head]);
        float v_amax = block_max_abs(
            fmaxf(fabsf(v_lo_val), fabsf(v_hi_val)),
            half_head, tid, warp_max);

        // --- Compute per-slot scales + inverses. Clamp to avoid /0. ---
        float k_scale = fmaxf(k_amax / FP8_E4M3_MAX, 1e-12f);
        float v_scale = fmaxf(v_amax / FP8_E4M3_MAX, 1e-12f);
        float k_inv = 1.0f / k_scale;
        float v_inv = 1.0f / v_scale;

        if (tid == 0) {
            k_scale_cache[scale_idx] = k_scale;
            v_scale_cache[scale_idx] = v_scale;
        }

        // --- Quantize and write cache entries. ---
        key_cache[cache_offset + tid]             = __nv_fp8_e4m3(k_lo_val * k_inv);
        key_cache[cache_offset + tid + half_head] = __nv_fp8_e4m3(k_hi_val * k_inv);
        value_cache[cache_offset + tid]             = __nv_fp8_e4m3(v_lo_val * v_inv);
        value_cache[cache_offset + tid + half_head] = __nv_fp8_e4m3(v_hi_val * v_inv);
    }
}
