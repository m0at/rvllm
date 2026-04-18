// In-place bf16 RMSNorm: reads bf16 input, normalizes with f16 gamma, writes bf16 back.
// For the delta path where GEMM outputs bf16 to avoid f16 overflow.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
rmsnorm_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,    // [num_tokens, hidden_size] bf16 in-place
    const __half* __restrict__ gamma,  // [hidden_size] f16
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
        float v = __bfloat162float(x[row_offset + i]);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    if (threadIdx.x == 0) smem[0] = sum_sq;
    __syncthreads();
    sum_sq = smem[0];

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(x[row_offset + i]) * rms * __half2float(gamma[i]);
        x[row_offset + i] = __float2bfloat16(v);
    }
}
