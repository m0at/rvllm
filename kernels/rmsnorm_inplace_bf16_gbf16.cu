// RMSNorm in-place on bf16 with bf16 gamma. Used by the bf16 vision
// path where the loader keeps vision norm gammas in their native bf16
// (instead of the text-side convention where gamma is f16 even when
// activations are bf16). Otherwise identical to rmsnorm_inplace_bf16.

#include <cuda_bf16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_b(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum_b(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_b(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_b(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
rmsnorm_inplace_bf16_gbf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
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
    float sum_sq = block_reduce_sum_b(local_ss, smem);
    if (tid == 0) smem[0] = sum_sq;
    __syncthreads();
    float rms = rsqrtf(smem[0] / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(x[row_offset + i]) * rms
                * __bfloat162float(gamma[i]);
        x[row_offset + i] = __float2bfloat16(v);
    }
}
