// Compute alpha/beta for Qwen 3.6 Gated-DeltaNet linear-attention,
// fully on GPU. Each output element (v in 0..vus) is independent:
//
//   a_acc[v]  = sum_k(in_proj_a[v, k] * input[k])
//   b_acc[v]  = sum_k(in_proj_b[v, k] * input[k])
//   x[v]      = a_acc[v] + dt_bias[v]
//   sp[v]     = log(1 + exp(x))     (numerically-stable softplus:
//                                   if x>20, sp=x; else log1p(exp(x)))
//   g[v]      = -exp(a_log[v]) * sp
//   alpha[v]  = exp(g)
//   beta[v]   = 1 / (1 + exp(-b_acc[v]))
//
// Replaces a host pipeline of:
//   DtoH normed_input (~4 KB) →
//   f16 → f32 host conversion →
//   nested CPU GEMV over [vus, h_us] (~130k FLOPs / layer / token) →
//   HtoD alpha + beta (~256 B)
// with one launch (Phase 4b-prep iter5).
//
// All weights (a_w, b_w, a_log, dt_bias) and the input live on the
// device already; the kernel reads f16 directly and accumulates in
// f32. Output scales (alpha, beta) are written as f32 — same dtype
// the host pipeline produced.
//
// Launch:
//   Grid:  (vus, 1, 1)               — one block per output element
//   Block: (block_size, 1, 1)         — power-of-two dot-product width
//   Shared: block_size * sizeof(float)
//
// Constraint: `hidden % 1 == 0`; the kernel handles the trailing
// remainder via a strided loop (no alignment requirement).

#include <cuda_fp16.h>
#include <math.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_ab(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_ab(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_ab(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_ab(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
qwen_linear_alpha_beta_f16_kernel(
    float*        __restrict__ alpha_out,    // [vus] f32
    float*        __restrict__ beta_out,     // [vus] f32
    const __half* __restrict__ in_proj_a,    // [vus, hidden] f16, row-major
    const __half* __restrict__ in_proj_b,    // [vus, hidden] f16, row-major
    const __half* __restrict__ a_log,        // [vus] f16
    const __half* __restrict__ dt_bias,      // [vus] f16
    const __half* __restrict__ input,        // [hidden] f16
    int vus,
    int hidden
) {
    const int v   = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (v >= vus) return;

    const __half* a_row = in_proj_a + (long long)v * hidden;
    const __half* b_row = in_proj_b + (long long)v * hidden;

    __shared__ float smem_a[WARPS_MAX];
    __shared__ float smem_b[WARPS_MAX];

    // Pass 1: thread-local dot products.
    float local_a = 0.0f;
    float local_b = 0.0f;
    for (int k = tid; k < hidden; k += stride) {
        float xk = __half2float(input[k]);
        local_a += __half2float(a_row[k]) * xk;
        local_b += __half2float(b_row[k]) * xk;
    }
    float a_acc = block_reduce_sum_ab(local_a, smem_a);
    float b_acc = block_reduce_sum_ab(local_b, smem_b);

    // Thread 0: apply the post-GEMV ops + write outputs.
    if (tid == 0) {
        float dt_b = __half2float(dt_bias[v]);
        float al   = __half2float(a_log[v]);
        float x = a_acc + dt_b;
        // numerically-stable softplus
        float sp = (x > 20.0f) ? x : log1pf(expf(x));
        float g  = -expf(al) * sp;
        alpha_out[v] = expf(g);
        beta_out[v]  = 1.0f / (1.0f + expf(-b_acc));
    }
}
