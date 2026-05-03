// Per-head RMSNorm on Q, K, and V projections (Gemma 4 QKV-Norm), BF16.
//
// Cycle 55 step 11 (Phase B): bf16 sibling of
// fused_qkv_rmsnorm_kernel. Identical math (f32 accumulator,
// two-stage warp reduction, RMS-inverse, optional per-head-dim gamma);
// inputs/outputs/gamma flip __half → __nv_bfloat16. Used when the
// upstream QKV projection produces bf16 (Fp8GemvBf16In, cycle 55
// step 3).
//
// Q and K get learned gamma; V is parameter-free (magnitude-only).
//
// Grid:  (num_tokens, num_heads + 2 * num_kv_heads, 1)
//   blockIdx.y < num_heads                              -> Q head (gamma)
//   num_heads <= blockIdx.y < num_heads + num_kv_heads   -> K head (gamma)
//   blockIdx.y >= num_heads + num_kv_heads               -> V head (no gamma)
// Block: (min(head_dim, 1024), 1, 1)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C"
__global__ void fused_qkv_rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    const __half* __restrict__ q_gamma,
    const __half* __restrict__ k_gamma,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps,
    int src_row_stride
) {
    const int token = blockIdx.x;
    const int head_global = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];

    const __nv_bfloat16* src;
    __nv_bfloat16* dst;
    const __half* gamma;
    int n_heads_this;
    int head_local;
    bool use_gamma;

    if (head_global < num_heads) {
        // Q head
        head_local = head_global;
        n_heads_this = num_heads;
        src = q_in;
        dst = q_out;
        gamma = q_gamma;
        use_gamma = true;
    } else if (head_global < num_heads + num_kv_heads) {
        // K head
        head_local = head_global - num_heads;
        n_heads_this = num_kv_heads;
        src = k_in;
        dst = k_out;
        gamma = k_gamma;
        use_gamma = true;
    } else {
        // V head (parameter-free)
        head_local = head_global - num_heads - num_kv_heads;
        n_heads_this = num_kv_heads;
        src = v_in;
        dst = v_out;
        gamma = nullptr;
        use_gamma = false;
    }

    // Src strides by the full QKV row (interleaved GEMM output).
    const int src_offset = token * src_row_stride + head_local * head_dim;
    // Dst strides by the per-component span (compact scratch).
    const int dst_offset = (token * n_heads_this + head_local) * head_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(src[src_offset + i]);
        sum_sq += v * v;
    }

    // Warp reduce
    for (int off = warpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        float v = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) smem[0] = rsqrtf(v / (float)head_dim + eps);
    }
    __syncthreads();

    float rms_inv = smem[0];

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(src[src_offset + i]) * rms_inv;
        if (use_gamma) v *= __half2float(gamma[i]);
        dst[dst_offset + i] = __float2bfloat16(v);
    }
}
