// Per-head RMSNorm on Q and K projections (Gemma 4 QK-Norm), BF16 variant.
//
// Cycle 55 step 5 (Phase B): bf16-input/output sibling of
// fused_qk_rmsnorm_kernel. Identical math (f32 accumulator); only the
// I/O dtypes flip __half → __nv_bfloat16. Used under bf16-native
// dispatch (default since cycle 55 step 1) so the QKV chain doesn't
// have to narrow bf16→f16 between the projection GEMV and Q/K
// RMSNorm.
//
// Gemma 3/4 applies RMSNorm to each Q head and K head independently
// before RoPE. The learned gamma vectors are per-head-dim (not per-head),
// shared across all heads.
//
// Grid:  (num_tokens, num_heads + num_kv_heads, 1)
//   blockIdx.y < num_heads        -> Q head
//   blockIdx.y >= num_heads       -> K head (offset by num_heads)
// Block: (head_dim or 1024, 1, 1)
//
// Q input:  [num_tokens, num_heads, head_dim]
// K input:  [num_tokens, num_kv_heads, head_dim]
// Q output: [num_tokens, num_heads, head_dim]  (can alias input)
// K output: [num_tokens, num_kv_heads, head_dim]  (can alias input)
// q_gamma, k_gamma: [head_dim]

#include <cuda_bf16.h>

extern "C"
__global__ void fused_qk_rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    const __nv_bfloat16* __restrict__ q_gamma,
    const __nv_bfloat16* __restrict__ k_gamma,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps
) {
    const int token = blockIdx.x;
    const int head_global = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];

    const bool is_q = (head_global < num_heads);
    const int head_local = is_q ? head_global : (head_global - num_heads);
    const int n_heads_this = is_q ? num_heads : num_kv_heads;

    const __nv_bfloat16* src = is_q ? q_in : k_in;
    __nv_bfloat16* dst = is_q ? q_out : k_out;
    const __nv_bfloat16* gamma = is_q ? q_gamma : k_gamma;

    const int offset = (token * n_heads_this + head_local) * head_dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(src[offset + i]);
        sum_sq += v * v;
    }

    // Warp reduce
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    }
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        float v = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (lane == 0) {
            float rms_inv = rsqrtf(v / (float)head_dim + eps);
            smem[0] = rms_inv;
        }
    }
    __syncthreads();

    float rms_inv = smem[0];

    // Apply: out[i] = gamma[i] * x[i] * rms_inv
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(src[offset + i]);
        float g = __bfloat162float(gamma[i]);
        dst[offset + i] = __float2bfloat16(v * rms_inv * g);
    }
}
