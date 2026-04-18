// Parameter-free RMS normalization for V (Gemma 4 v_norm).
// x = x / rms(x) — no learnable weight, just normalize magnitude.

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

// grid=(num_tokens * num_kv_heads), block=(min(head_dim, 1024))
// Each block normalizes one (token, head) vector of length head_dim.
extern "C" __global__ void __launch_bounds__(1024)
vnorm_f16_kernel(
    __half* __restrict__ v,  // [num_tokens, num_kv_heads, head_dim] in-place
    float eps,
    int head_dim
) {
    const int idx = blockIdx.x;  // token * num_kv_heads + head
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int base = idx * head_dim;

    __shared__ float smem[WARPS_MAX];

    float local_ss = 0.0f;
    for (int i = tid; i < head_dim; i += stride) {
        float val = __half2float(v[base + i]);
        local_ss += val * val;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    __syncthreads();

    float rms = rsqrtf(sum_sq / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += stride) {
        float val = __half2float(v[base + i]) * rms;
        v[base + i] = __float2half(val);
    }
}
