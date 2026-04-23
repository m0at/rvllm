// Per-head RMSNorm on Q, K, and V projections (Gemma 4 QKV-Norm).
//
// Q and K get learned gamma; V is parameter-free (magnitude-only).
//
// Grid:  (num_tokens, num_heads + 2 * num_kv_heads, 1)
//   blockIdx.y < num_heads                              -> Q head (gamma)
//   num_heads <= blockIdx.y < num_heads + num_kv_heads   -> K head (gamma)
//   blockIdx.y >= num_heads + num_kv_heads               -> V head (no gamma)
// Block: (min(head_dim, 1024), 1, 1)
//
// Source layout (`src_row_stride`):
//
// The upstream QKV GEMM writes a single row-major
// `[num_tokens, q_dim + 2*kv_dim]` buffer — token i's Q / K / V are
// interleaved within row i. Callers pass q_in / k_in / v_in pointing
// at the start of each component in row 0. To reach token `t`'s
// component base we must stride by the full row (`qkv_rows`), not
// by the per-component span (`n_heads_this * head_dim`).
//
// Q and K write to separate compact `[num_tokens, n_heads, head_dim]`
// scratch buffers (q_out, v_out is the V-only compact analogue). That
// way the downstream rope kernel's compact indexing is consistent
// across all three components.

#include <cuda_fp16.h>

extern "C"
__global__ void fused_qkv_rmsnorm_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
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

    const __half* src;
    __half* dst;
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
        float v = __half2float(src[src_offset + i]);
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
        float v = __half2float(src[src_offset + i]) * rms_inv;
        if (use_gamma) v *= __half2float(gamma[i]);
        dst[dst_offset + i] = __float2half(v);
    }
}
