// Row-wise softmax: f32 input → bf16 output. bf16 sibling of
// softmax_row_f32_to_f16. Same numerics (f32 max-subtract / exp /
// reciprocal-sum), only the output narrowing dtype differs.

#include <cuda_bf16.h>
#include <float.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v = __shfl_xor_sync(0xffffffff, val, offset);
        val = (v > val) ? v : val;
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_max_f32(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_max_f32(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max_f32(val);
    return val;
}

__device__ __forceinline__ float block_reduce_sum_f32(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_f32(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_f32(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
softmax_row_f32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ in,
    int seq_len
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const long long row_off = (long long)row * seq_len;

    __shared__ float smem[WARPS_MAX];

    float local_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += stride) {
        float v = in[row_off + i];
        if (v > local_max) local_max = v;
    }
    float row_max = block_reduce_max_f32(local_max, smem);
    if (tid == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];

    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += stride) {
        local_sum += expf(in[row_off + i] - row_max);
    }
    float row_sum = block_reduce_sum_f32(local_sum, smem);
    if (tid == 0) smem[0] = row_sum;
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    for (int i = tid; i < seq_len; i += stride) {
        float p = expf(in[row_off + i] - row_max) * inv_sum;
        out[row_off + i] = __float2bfloat16(p);
    }
}
