// FP8 KV Cache quantization/dequantization kernels.
//
// Provides E4M3 FP8 quantization with dynamic per-head scaling for KV cache
// compression, halving VRAM usage compared to FP16 storage.
//
// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
// Range: [-448, 448], resolution degrades gracefully.
//
// Kernel 1: quantize_kv_kernel
//   Quantizes f16/f32 KV data to u8 (FP8 E4M3) with per-head scale factors.
//   Grid:  (num_tokens, num_heads, 1)
//   Block: (head_dim, 1, 1) or (min(head_dim, 1024), 1, 1) with loop
//
// Kernel 2: dequantize_kv_kernel
//   Dequantizes u8 (FP8 E4M3) back to f32 using stored scale factors.
//   Same launch config as quantize.
//
// Kernel 3: dequantize_and_dot_kernel
//   Fused dequantize + Q*K dot product for attention, avoiding materialization
//   of the full dequantized KV cache.

#include <float.h>

// FP8 E4M3 max representable magnitude
#define FP8_E4M3_MAX 448.0f

// Convert a float to FP8 E4M3 (stored as unsigned char).
// Input is pre-scaled: val_scaled = val / scale, clamped to [-448, 448].
__device__ __forceinline__ unsigned char float_to_fp8_e4m3(float val) {
    // Clamp to FP8 E4M3 representable range
    val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);

    // Extract sign
    unsigned int bits = __float_as_uint(val);
    unsigned char sign = (bits >> 24) & 0x80;  // bit 7

    float abs_val = fabsf(val);
    if (abs_val < 1.9531e-3f) {
        // Subnormal or zero in E4M3 (exponent bias = 7, min normal = 2^-6)
        // Subnormals: mantissa * 2^-9 (smallest = 2^-9 = 1.953125e-3)
        // Round to nearest subnormal
        int mantissa = (int)(abs_val / 1.9531e-3f + 0.5f);
        mantissa = min(mantissa, 7);
        return sign | (unsigned char)mantissa;
    }

    // Normal range: extract biased exponent and mantissa
    // FP32: 1.mantissa * 2^(exp-127)
    // E4M3: 1.mantissa * 2^(exp-7), exp in [1..15]
    int fp32_exp = ((bits >> 23) & 0xFF) - 127;
    int fp8_exp = fp32_exp + 7;  // E4M3 bias = 7

    if (fp8_exp <= 0) {
        // Underflow to subnormal
        int shift = 1 - fp8_exp;
        float subnormal_val = abs_val / (1.9531e-3f);  // normalize to subnormal grid
        int mantissa = (int)(subnormal_val + 0.5f);
        mantissa = min(max(mantissa >> shift, 0), 7);
        return sign | (unsigned char)mantissa;
    }
    if (fp8_exp > 15) {
        // Overflow: clamp to max E4M3 = 448.0 = 1.75 * 2^8
        return sign | 0x7E;  // exp=15, mantissa=6 (max finite, not NaN)
    }

    // Extract 3-bit mantissa from fp32 mantissa (23 bits -> 3 bits, with rounding)
    unsigned int fp32_mantissa = bits & 0x7FFFFF;
    int mantissa = (fp32_mantissa + (1 << 19)) >> 20;  // round to nearest, 23-3=20 bit shift
    if (mantissa >= 8) {
        // Mantissa overflow, bump exponent
        mantissa = 0;
        fp8_exp++;
        if (fp8_exp > 15) {
            return sign | 0x7E;  // max finite
        }
    }

    // Avoid NaN encoding (exp=15, mantissa=7)
    if (fp8_exp == 15 && mantissa > 6) {
        mantissa = 6;
    }

    return sign | ((unsigned char)(fp8_exp & 0xF) << 3) | ((unsigned char)(mantissa & 0x7));
}

// Convert FP8 E4M3 (unsigned char) back to float.
__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char fp8) {
    float sign = (fp8 & 0x80) ? -1.0f : 1.0f;
    int exp = (fp8 >> 3) & 0xF;
    int mantissa = fp8 & 0x7;

    if (exp == 0) {
        if (mantissa == 0) return 0.0f;
        // Subnormal: 0.mantissa * 2^(-6)  = mantissa * 2^(-9)
        return sign * (float)mantissa * 1.9531e-3f;
    }
    if (exp == 15 && mantissa == 7) {
        // NaN in E4M3 -- treat as zero for safety
        return 0.0f;
    }

    // Normal: 1.mantissa * 2^(exp - 7)
    float fmantissa = 1.0f + (float)mantissa / 8.0f;
    return sign * ldexpf(fmantissa, exp - 7);
}

// Compute per-head absmax for dynamic scaling.
// Each thread block handles one (token, head) pair.
// head_dim threads cooperate via warp reduction.
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Quantize f32 KV values to FP8 E4M3 with per-head dynamic scaling.
//
// input:  [num_tokens, num_heads, head_dim] (f32)
// output: [num_tokens, num_heads, head_dim] (u8, FP8 E4M3)
// scales: [num_tokens, num_heads] (f32, per-head scale factor)
//
// scale = absmax(head_values) / FP8_E4M3_MAX
// quantized = float_to_fp8(value / scale)
extern "C"
__global__ void quantize_kv_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ scales,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    const int base = (token_idx * num_heads + head_idx) * head_dim;

    // Phase 1: find absmax across head_dim via warp reduction
    float local_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[base + d]));
    }
    local_max = warp_reduce_max(local_max);

    // Cross-warp reduction via shared memory
    __shared__ float smax[32];  // up to 32 warps
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) smax[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane_id < num_warps) ? smax[lane_id] : 0.0f;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            // Compute scale: absmax / FP8_MAX, with epsilon to avoid div-by-zero
            float scale = val / FP8_E4M3_MAX;
            scale = fmaxf(scale, 1e-12f);
            scales[token_idx * num_heads + head_idx] = scale;
            smax[0] = scale;
        }
    }
    __syncthreads();

    float scale = smax[0];
    float inv_scale = 1.0f / scale;

    // Phase 2: quantize
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = input[base + d] * inv_scale;
        output[base + d] = float_to_fp8_e4m3(val);
    }
}

// Dequantize FP8 E4M3 back to f32 using stored per-head scales.
//
// input:  [num_tokens, num_heads, head_dim] (u8, FP8 E4M3)
// output: [num_tokens, num_heads, head_dim] (f32)
// scales: [num_tokens, num_heads] (f32)
extern "C"
__global__ void dequantize_kv_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scales,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    const int base = (token_idx * num_heads + head_idx) * head_dim;
    float scale = scales[token_idx * num_heads + head_idx];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        output[base + d] = fp8_e4m3_to_float(input[base + d]) * scale;
    }
}

// Quantize directly from paged KV cache (f32) into FP8 paged cache (u8).
//
// Cache layout: [num_blocks, block_size, num_heads, head_dim]
// Slot-based addressing: slot -> block_idx * block_stride + offset * head_stride
//
// This kernel quantizes in-place from a source f32 cache into an FP8 cache,
// writing per-head scales alongside.
//
// Grid:  (num_slots, num_heads, 1)
// Block: (min(head_dim, 256), 1, 1)
extern "C"
__global__ void quantize_paged_kv_kernel(
    const float* __restrict__ src_cache,      // [num_blocks, block_size, num_heads, head_dim] f32
    unsigned char* __restrict__ dst_cache,     // [num_blocks, block_size, num_heads, head_dim] u8
    float* __restrict__ scales,               // [num_blocks, block_size, num_heads] f32
    const int* __restrict__ slot_mapping,     // [num_slots] slot indices
    int num_slots,
    int num_heads,
    int head_dim,
    int block_size
) {
    const int slot_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (slot_idx >= num_slots || head_idx >= num_heads) return;

    int slot = slot_mapping[slot_idx];
    if (slot < 0) return;  // padding

    const int block_idx = slot / block_size;
    const int block_off = slot % block_size;
    const int head_stride = head_dim;
    const int token_stride = num_heads * head_dim;
    const int block_stride = block_size * token_stride;

    const int base = block_idx * block_stride + block_off * token_stride + head_idx * head_stride;

    // Compute absmax
    float local_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(src_cache[base + d]));
    }
    local_max = warp_reduce_max(local_max);

    __shared__ float smax[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) smax[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane_id < num_warps) ? smax[lane_id] : 0.0f;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            float scale = fmaxf(val / FP8_E4M3_MAX, 1e-12f);
            int scale_idx = block_idx * block_size * num_heads + block_off * num_heads + head_idx;
            scales[scale_idx] = scale;
            smax[0] = scale;
        }
    }
    __syncthreads();

    float inv_scale = 1.0f / smax[0];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = src_cache[base + d] * inv_scale;
        dst_cache[base + d] = float_to_fp8_e4m3(val);
    }
}

// Dequantize paged FP8 KV cache entries back to f32.
// Inverse of quantize_paged_kv_kernel.
extern "C"
__global__ void dequantize_paged_kv_kernel(
    const unsigned char* __restrict__ src_cache,  // FP8 paged cache
    float* __restrict__ dst_cache,                // f32 output
    const float* __restrict__ scales,             // per-head scales
    const int* __restrict__ slot_mapping,
    int num_slots,
    int num_heads,
    int head_dim,
    int block_size
) {
    const int slot_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (slot_idx >= num_slots || head_idx >= num_heads) return;

    int slot = slot_mapping[slot_idx];
    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int block_off = slot % block_size;
    const int head_stride = head_dim;
    const int token_stride = num_heads * head_dim;
    const int block_stride = block_size * token_stride;

    const int base = block_idx * block_stride + block_off * token_stride + head_idx * head_stride;
    int scale_idx = block_idx * block_size * num_heads + block_off * num_heads + head_idx;
    float scale = scales[scale_idx];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst_cache[base + d] = fp8_e4m3_to_float(src_cache[base + d]) * scale;
    }
}
