// Per-head RMSNorm on Q, K, and V projections (Gemma 4 QKV-Norm).
//
// Q and K get learned gamma; V is parameter-free (magnitude-only).
//
// Grid:  (num_tokens, num_heads + 2 * num_kv_heads, 1)
//   blockIdx.y < num_heads                              -> Q head (gamma)
//   num_heads <= blockIdx.y < num_heads + num_kv_heads   -> K head (gamma)
//   blockIdx.y >= num_heads + num_kv_heads               -> V head (no gamma)
// Block: (min(head_dim, 1024), 1, 1)

#include <cuda_fp16.h>

extern "C"
__global__ void fused_qkv_rmsnorm_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    __half* __restrict__ v_inout,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    const __half* __restrict__ q_gamma,
    const __half* __restrict__ k_gamma,
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

    __half* dst;
    const __half* gamma;
    int n_heads_this;
    int head_local;
    bool use_gamma;
    int src_offset;

    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int qkv_rows = q_dim + 2 * kv_dim;
    const int row_base = token * qkv_rows;

    if (head_global < num_heads) {
        head_local = head_global;
        n_heads_this = num_heads;
        dst = q_out;
        gamma = q_gamma;
        use_gamma = true;
        src_offset = row_base + head_local * head_dim;
    } else if (head_global < num_heads + num_kv_heads) {
        head_local = head_global - num_heads;
        n_heads_this = num_kv_heads;
        dst = k_out;
        gamma = k_gamma;
        use_gamma = true;
        src_offset = row_base + q_dim + head_local * head_dim;
    } else {
        head_local = head_global - num_heads - num_kv_heads;
        n_heads_this = num_kv_heads;
        dst = v_inout;
        gamma = nullptr;
        use_gamma = false;
        src_offset = row_base + q_dim + kv_dim + head_local * head_dim;
    }

    const int dst_offset = (token * n_heads_this + head_local) * head_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __half2float(q_in[src_offset + i]);
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
        float v = __half2float(q_in[src_offset + i]) * rms_inv;
        if (use_gamma) v *= __half2float(gamma[i]);
        dst[dst_offset + i] = __float2half(v);
    }
}
