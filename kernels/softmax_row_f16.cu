// Row-wise softmax in-place on f16, used for ViT attention scores.
//
// scores: [num_rows, seq_len] f16  →  in-place softmax along seq_len.
// Each block handles one row; threads cooperate via shfl.
//
// Numerically-stable variant: subtract row max before exp.
//
// Launch:
//   Grid:  (num_rows, 1, 1)
//   Block: (min(seq_len, 1024), 1, 1)

#include <cuda_fp16.h>
#include <float.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v = __shfl_xor_sync(0xffffffff, val, offset);
        val = (v > val) ? v : val;
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
softmax_row_f16_kernel(
    __half* __restrict__ x,   // [num_rows, seq_len] in-place
    int seq_len
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const long long row_off = (long long)row * seq_len;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: find max for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += stride) {
        float v = __half2float(x[row_off + i]);
        if (v > local_max) local_max = v;
    }
    float row_max = block_reduce_max(local_max, smem);
    if (tid == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];

    // Pass 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += stride) {
        float v = __half2float(x[row_off + i]) - row_max;
        float e = expf(v);
        x[row_off + i] = __float2half(e);
        local_sum += e;
    }
    float row_sum = block_reduce_sum(local_sum, smem);
    if (tid == 0) smem[0] = row_sum;
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // Pass 3: divide
    for (int i = tid; i < seq_len; i += stride) {
        float v = __half2float(x[row_off + i]) * inv_sum;
        x[row_off + i] = __float2half(v);
    }
}
