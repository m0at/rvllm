// Batched-prefill counterpart of shared_gate_dot_sigmoid_f16.
// One block per token; computes sigmoid(dot(weight, input[m, :]))
// → out_sigmoid[m] f32, for m in 0..num_tokens.
//
// Phase 6c / Round-27.

#include <cuda_fp16.h>
#include <math.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_sgb(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_sgb(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_sgb(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_sgb(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
shared_gate_dot_sigmoid_f16_batched_kernel(
    float*        __restrict__ out_sigmoid,  // [num_tokens] f32
    const __half* __restrict__ weight,       // [hidden] f16
    const __half* __restrict__ input,        // [num_tokens, hidden] f16
    int hidden,
    int num_tokens
) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (m >= num_tokens) return;

    const __half* in_row = input + (long long)m * hidden;

    __shared__ float smem[WARPS_MAX];

    float local = 0.0f;
    for (int k = tid; k < hidden; k += stride) {
        local += __half2float(weight[k]) * __half2float(in_row[k]);
    }
    float logit = block_reduce_sum_sgb(local, smem);
    if (tid == 0) {
        out_sigmoid[m] = 1.0f / (1.0f + expf(-logit));
    }
}
