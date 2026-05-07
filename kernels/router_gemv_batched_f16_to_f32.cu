// Batched-prefill counterpart of `router_gemv_f16_to_f32`. For each
// token t in [0, num_tokens) and each expert e in [0, num_experts),
// compute the dot product of `router_w[e, :]` and `input[t, :]` and
// store as `logits_out[t, e]`.
//
// Single-token kernel grid is (num_experts,); batched kernel grid is
// (num_experts, num_tokens, 1) — each block processes one (expert,
// token) pair. Block size stays 1024 threads (matches the single-
// token version). One launch per layer in batched-prefill instead of
// N launches in the per-token MoE path.
//
// Phase 6a / Round-27 (codex round-27 plan: "first 200 LOC cut =
// router + topk batched, routed FFN unchanged").

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_rgb(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_rgb(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_rgb(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_rgb(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
router_gemv_batched_f16_to_f32_kernel(
    float*        __restrict__ logits_out, // [num_tokens, num_experts] f32
    const __half* __restrict__ router_w,   // [num_experts, hidden] f16
    const __half* __restrict__ input,      // [num_tokens, hidden] f16
    int num_experts,
    int hidden,
    int num_tokens
) {
    const int e = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (e >= num_experts || t >= num_tokens) return;

    const __half* row = router_w + (long long)e * hidden;
    const __half* inp = input + (long long)t * hidden;

    __shared__ float smem[WARPS_MAX];

    float local = 0.0f;
    for (int k = tid; k < hidden; k += stride) {
        local += __half2float(row[k]) * __half2float(inp[k]);
    }
    float acc = block_reduce_sum_rgb(local, smem);

    if (tid == 0) {
        logits_out[(long long)t * num_experts + e] = acc;
    }
}
