// Row-wise softmax: f32 input → f16 output. Used for ViT attention
// scores in the Gemma 4 vision tower so the entire softmax (max-subtract
// → exp → reciprocal-sum → multiply) runs in f32 precision and only
// the final probabilities narrow to f16 — matching HF's
// `softmax(..., dtype=torch.float32).to(query.dtype)` semantics
// (modeling_gemma4.py:794–795). The earlier path cast f32 scores to f16
// BEFORE softmax which lost precision on large attention matrices
// (N ≈ 2400 for typical Gemma-vision inputs, where the max-subtract
// dynamic range really matters).
//
// in_f32:  [num_rows, seq_len] f32
// out_f16: [num_rows, seq_len] f16
//
// Numerically-stable: subtract row max in f32 before exp.
//
// Launch:
//   Grid:  (num_rows, 1, 1)
//   Block: (min(seq_len, 1024), 1, 1)

#include <cuda_fp16.h>
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
softmax_row_f32_to_f16_kernel(
    __half* __restrict__ out,
    const float* __restrict__ in,
    int seq_len
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const long long row_off = (long long)row * seq_len;

    __shared__ float smem[WARPS_MAX];

    // 1. row max (f32)
    float local_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += stride) {
        float v = in[row_off + i];
        if (v > local_max) local_max = v;
    }
    float row_max = block_reduce_max_f32(local_max, smem);
    if (tid == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];

    // 2. sum exp(v - row_max) (f32)
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += stride) {
        local_sum += expf(in[row_off + i] - row_max);
    }
    float row_sum = block_reduce_sum_f32(local_sum, smem);
    if (tid == 0) smem[0] = row_sum;
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // 3. write probabilities, narrow to f16 only at output.
    for (int i = tid; i < seq_len; i += stride) {
        float p = expf(in[row_off + i] - row_max) * inv_sum;
        out[row_off + i] = __float2half(p);
    }
}
