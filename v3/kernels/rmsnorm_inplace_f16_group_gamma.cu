// In-place f16 RMSNorm with per-group gamma rows.
//
// `x` is laid out as [num_rows, hidden_size]. Each row uses
// `gamma[(row % num_groups), :]`.

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_group(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum_group(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_sum_group(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum_group(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
rmsnorm_inplace_f16_group_gamma_kernel(
    __half* __restrict__ x,
    const __half* __restrict__ gamma,
    float eps,
    int hidden_size,
    int num_groups
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;
    const int gamma_offset = (row % num_groups) * hidden_size;

    __shared__ float smem[WARPS_MAX];

    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(x[row_offset + i]);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum_group(local_ss, smem);
    if (threadIdx.x == 0) smem[0] = sum_sq;
    __syncthreads();
    sum_sq = smem[0];

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(x[row_offset + i]) * rms * __half2float(gamma[gamma_offset + i]);
        x[row_offset + i] = __float2half(v);
    }
}
