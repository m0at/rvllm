// Fused shared-expert gate-logit + sigmoid for Qwen 3.6 MoE.
//
//   logit = sum_k (weight[k] * input[k])
//   out[0] = 1 / (1 + exp(-logit))
//
// One block, output is a single f32 scalar on device. Designed
// so the result can be consumed by `scaled_add_f16_to_f32_devw`
// on the same stream — no host round-trip needed (Phase 4b-prep
// iter21).
//
// Launch:
//   Grid:  (1, 1, 1)
//   Block: (BLOCK, 1, 1)   — power-of-two, ≤1024
//   Shared: WARPS_MAX * sizeof(float)

#include <cuda_fp16.h>
#include <math.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_sg(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_sg(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_sg(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_sg(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
shared_gate_dot_sigmoid_f16_kernel(
    float*        __restrict__ out_sigmoid,  // [1] f32
    const __half* __restrict__ weight,       // [hidden] f16
    const __half* __restrict__ input,        // [hidden] f16
    int hidden
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    __shared__ float smem[WARPS_MAX];

    float local = 0.0f;
    for (int k = tid; k < hidden; k += stride) {
        local += __half2float(weight[k]) * __half2float(input[k]);
    }
    float logit = block_reduce_sum_sg(local, smem);
    if (tid == 0) {
        out_sigmoid[0] = 1.0f / (1.0f + expf(-logit));
    }
}
