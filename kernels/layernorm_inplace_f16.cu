// LayerNorm with bias (Qwen ViT vision blocks: norm1, norm2, merger.norm).
//
// out[t, d] = (x[t, d] - mean(x[t, :])) / sqrt(var(x[t, :]) + eps)
//             * gamma[d] + beta[d]
//
// One CUDA block per row, threads cooperatively reduce mean/var.
//
// Launch:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)

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
layernorm_inplace_f16_kernel(
    __half* __restrict__ x,           // [num_tokens, hidden] in-place
    const __half* __restrict__ gamma, // [hidden]
    const __half* __restrict__ beta,  // [hidden]
    float eps,
    int hidden
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_off = row * hidden;

    __shared__ float smem[WARPS_MAX * 2];
    float* smem_sum = smem;
    float* smem_var = smem + WARPS_MAX;

    // Pass 1: mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        local_sum += __half2float(x[row_off + i]);
    }
    float sum = block_reduce_sum(local_sum, smem_sum);
    if (tid == 0) smem_sum[0] = sum;
    __syncthreads();
    float mean = smem_sum[0] / (float)hidden;

    // Pass 2: var
    float local_var = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = __half2float(x[row_off + i]) - mean;
        local_var += v * v;
    }
    float var = block_reduce_sum(local_var, smem_var);
    if (tid == 0) smem_var[0] = var;
    __syncthreads();
    float inv_std = rsqrtf(smem_var[0] / (float)hidden + eps);

    // Pass 3: normalize + affine
    for (int i = tid; i < hidden; i += stride) {
        float v = (__half2float(x[row_off + i]) - mean) * inv_std;
        v = v * __half2float(gamma[i]) + __half2float(beta[i]);
        x[row_off + i] = __float2half(v);
    }
}
