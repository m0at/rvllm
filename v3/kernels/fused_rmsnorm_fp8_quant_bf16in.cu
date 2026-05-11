// BF16-input variant of `fused_rmsnorm_fp8_quant_kernel` (cycle 53+
// Stage 1 BF16 residual chain). Reads bf16 residual, computes rmsnorm
// in f32, writes per-token-fp8-quantized output. The f32 internal math
// is unchanged from the f16 variant; only the input dtype changes.
//
// Used in place of `fused_rmsnorm_fp8_quant_kernel` whenever
// RVLLM_RESIDUAL_BF16=1 routes the residual buffer through bf16.
//
// Compile: nvcc -ptx -arch=sm_xx -O3 --use_fast_math

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define FP8_E4M3_MAX 448.0f
#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
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
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
fused_rmsnorm_fp8_quant_bf16in_kernel(
    __nv_fp8_storage_t*  __restrict__ output_fp8,
    float*               __restrict__ output_scales,
    const __nv_bfloat16* __restrict__ input,
    const __half*        __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(input[row_offset + i]);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    if (threadIdx.x == 0) smem[0] = sum_sq;
    __syncthreads();
    sum_sq = smem[0];

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    // Pass 2: compute normed values, find absmax
    float local_max = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(input[row_offset + i]) * rms * __half2float(weight[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float absmax = block_reduce_max(local_max, smem);
    if (threadIdx.x == 0) smem[0] = absmax;
    __syncthreads();
    absmax = smem[0];

    float scale = absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (tid == 0) output_scales[row] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 3: quantize to FP8
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(input[row_offset + i]) * rms * __half2float(weight[i]);
        output_fp8[row_offset + i] = __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}
