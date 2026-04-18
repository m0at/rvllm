// In-place f16 RMSNorm: reads f16 residual, normalizes, writes f16 back.
// Used by Gemma 4 post-attention and post-feedforward norms.
// The fused_rmsnorm_fp8_quant kernel writes FP8 output which cannot
// alias the f16 input -- this kernel does true in-place f16 normalization.

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
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

extern "C" __global__ void __launch_bounds__(1024)
rmsnorm_inplace_f16_kernel(
    __half* __restrict__ x,           // [num_tokens, hidden_size] in-place
    const __half* __restrict__ gamma,  // [hidden_size]
    float eps,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;

    __shared__ float smem[WARPS_MAX];

    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(x[row_offset + i]);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    __syncthreads();

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(x[row_offset + i]) * rms * __half2float(gamma[i]);
        x[row_offset + i] = __float2half(v);
    }
}
