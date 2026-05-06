// Per-MoE-layer router GEMV for Qwen 3.6.
//
//   logits[e] = sum_k (router[e, k] * input[k])
//
// `router` is the layer's gate weight `[num_experts, hidden]` f16.
// `input` is the post-attention-rmsnormed hidden state `[hidden]`
// f16. Output is f32 since downstream softmax + top-k expects
// some headroom and the host loop already accumulated in f32.
//
// Replaces a host-cached f32 matvec (~524 k MAC × 30 MoE layers ×
// per token) with a GPU launch (Phase 4b-prep iter17).
//
// Launch:
//   Grid:  (num_experts, 1, 1)        — one block per logit
//   Block: (BLOCK, 1, 1)               — power-of-two dot-product width
//   Shared: WARPS_MAX * sizeof(float)
//
// No alignment requirement on `hidden`: trailing remainder handled
// via the strided loop. `hidden` is expected ≤ 2048 in practice.

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_rg(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_rg(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_rg(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_rg(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
router_gemv_f16_to_f32_kernel(
    float*        __restrict__ logits_out,   // [num_experts] f32
    const __half* __restrict__ router_w,     // [num_experts, hidden] f16, row-major
    const __half* __restrict__ input,        // [hidden] f16
    int num_experts,
    int hidden
) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (e >= num_experts) return;

    const __half* row = router_w + (long long)e * hidden;

    __shared__ float smem[WARPS_MAX];

    float local = 0.0f;
    for (int k = tid; k < hidden; k += stride) {
        local += __half2float(row[k]) * __half2float(input[k]);
    }
    float acc = block_reduce_sum_rg(local, smem);

    if (tid == 0) {
        logits_out[e] = acc;
    }
}
