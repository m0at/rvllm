// Dynamic per-tensor FP8 E4M3 quantization of activations on GPU.
//
// Single-block kernel: find global absmax, compute scale, quantize.
// Handles up to ~1M f16 elements (1024 threads * ~1000 elements/thread).
//
// Compile: nvcc -ptx -arch=sm_90 -O3 --use_fast_math

#include <cuda_fp16.h>

#define QFP8_THREADS 1024
#define QFP8_WARPS   (QFP8_THREADS / 32)

#define FP8_E4M3_MAX 448.0f

// Convert f32 to FP8 E4M3 byte (device-side, branchless where possible)
__device__ __forceinline__ unsigned char float_to_fp8_e4m3(float val) {
    unsigned char sign = (val < 0.0f) ? 0x80 : 0x00;
    float abs_val = fabsf(val);
    abs_val = fminf(abs_val, FP8_E4M3_MAX);

    if (abs_val < 1.9531e-3f) {
        unsigned char mantissa = (unsigned char)fminf(__float2int_rn(abs_val / 1.9531e-3f), 7);
        return sign | mantissa;
    }

    unsigned int bits = __float_as_uint(abs_val);
    int fp32_exp = (int)((bits >> 23) & 0xFF) - 127;
    int fp8_exp = fp32_exp + 7;

    if (fp8_exp <= 0) {
        float subnormal = abs_val / 1.9531e-3f;
        unsigned char m = (unsigned char)fminf(__float2int_rn(subnormal), 7);
        int shift = 1 - fp8_exp;
        m = (m >> shift);
        m = (m > 7) ? 7 : m;
        return sign | m;
    }
    if (fp8_exp > 15) return sign | 0x7E; // max finite

    unsigned int fp32_mantissa = bits & 0x7FFFFF;
    int mantissa = (int)((fp32_mantissa + (1 << 19)) >> 20);

    if (mantissa >= 8) {
        mantissa = 0;
        fp8_exp += 1;
        if (fp8_exp > 15) return sign | 0x7E;
    }
    // Avoid NaN (exp=15, mantissa=7)
    if (fp8_exp == 15 && mantissa > 6) mantissa = 6;

    return sign | ((unsigned char)(fp8_exp & 0xF) << 3) | (unsigned char)(mantissa & 0x7);
}

// Warp-level max reduction
__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, off);
        val = fmaxf(val, other);
    }
    return val;
}

// Single-block kernel: quantize f16 activation to FP8 E4M3 with per-tensor scale.
// Launch with grid=(1,1,1), block=(1024,1,1).
extern "C" __global__ void __launch_bounds__(QFP8_THREADS)
quantize_activation_fp8_kernel(
    unsigned char* __restrict__ output_fp8,  // [num_elements] FP8 output
    float*         __restrict__ output_scale, // [1] per-tensor scale
    const __half*  __restrict__ input_f16,    // [num_elements] f16 input
    int                         num_elements
) {
    const int tid = threadIdx.x;
    __shared__ float s_warp_max[QFP8_WARPS];

    // Phase 1: find global absmax
    float local_max = 0.0f;
    for (int i = tid; i < num_elements; i += QFP8_THREADS) {
        float v = fabsf(__half2float(input_f16[i]));
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce
    local_max = warp_max(local_max);
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) s_warp_max[warp_id] = local_max;
    __syncthreads();

    // Final reduce across warps (warp 0)
    if (warp_id == 0) {
        float val = (lane_id < QFP8_WARPS) ? s_warp_max[lane_id] : 0.0f;
        val = warp_max(val);
        if (lane_id == 0) {
            float scale = (val < 1e-12f) ? 1e-12f : (val / FP8_E4M3_MAX);
            s_warp_max[0] = scale;
            output_scale[0] = scale;
        }
    }
    __syncthreads();

    // Phase 2: quantize using computed scale
    float inv_scale = 1.0f / s_warp_max[0];
    for (int i = tid; i < num_elements; i += QFP8_THREADS) {
        float v = __half2float(input_f16[i]) * inv_scale;
        output_fp8[i] = float_to_fp8_e4m3(v);
    }
}

// Multi-block variant for large activations (>1M elements).
// Phase 1: each block computes partial absmax -> atomicMax to global.
// Phase 2: separate kernel quantizes using the global scale.
extern "C" __global__ void __launch_bounds__(QFP8_THREADS)
find_absmax_fp8_kernel(
    float*        __restrict__ global_absmax, // [1] -- must be zeroed before launch
    const __half* __restrict__ input_f16,
    int                        num_elements
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * QFP8_THREADS + tid;
    __shared__ float s_warp_max[QFP8_WARPS];

    float local_max = 0.0f;
    for (int i = gid; i < num_elements; i += gridDim.x * QFP8_THREADS) {
        local_max = fmaxf(local_max, fabsf(__half2float(input_f16[i])));
    }

    local_max = warp_max(local_max);
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) s_warp_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < QFP8_WARPS) ? s_warp_max[lane_id] : 0.0f;
        val = warp_max(val);
        if (lane_id == 0) {
            // atomicMax on f32 via int reinterpret (works for positive values)
            int ival = __float_as_int(val);
            atomicMax((int*)global_absmax, ival);
        }
    }
}

extern "C" __global__ void __launch_bounds__(QFP8_THREADS)
apply_fp8_quantize_kernel(
    unsigned char* __restrict__ output_fp8,
    float*         __restrict__ output_scale,
    const __half*  __restrict__ input_f16,
    const float*   __restrict__ global_absmax,
    int                         num_elements
) {
    const int gid = blockIdx.x * QFP8_THREADS + threadIdx.x;

    // Read global absmax, compute scale
    float absmax = global_absmax[0];
    float scale = (absmax < 1e-12f) ? 1e-12f : (absmax / FP8_E4M3_MAX);
    if (gid == 0) output_scale[0] = scale;

    float inv_scale = 1.0f / scale;
    for (int i = gid; i < num_elements; i += gridDim.x * QFP8_THREADS) {
        float v = __half2float(input_f16[i]) * inv_scale;
        output_fp8[i] = float_to_fp8_e4m3(v);
    }
}
