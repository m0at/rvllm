// gated_delta_state_update_f16: per-head delta-rule state update for
// the Qwen 3.6 (qwen3-next) Gated-DeltaNet linear-attention block.
//
// Per timestep, per head:
//   S[h, v, k] = alpha[h] * S[h, v, k] + beta[h] * V[h, v] * K[h, k]
//
// alpha and beta are precomputed on the host from A_log + dt_bias +
// the per-head dt projection — keeping the decay+write-gate math out
// of the kernel lets the same kernel handle the parameter-rich
// variants (Mamba-2 style) without re-compiling.
//
// State layout: `[num_heads, d_v, d_k]` row-major. K/V are
// `[num_heads, d_k|d_v]` per the current-timestep slice.
//
// For Qwen 3.6 35B-A3B (qwen3-next): num_heads=32, d_k=d_v=128
// (consistent with `linear_attn.norm.weight = [128]`). One-block
// covers one head; 16×16 threads each touch one (v, k) pair, then
// each thread handles 8×8 elements via the inner loop.
//
// Caller is responsible for:
//   - allocating + persisting the state buffer across decode steps
//   - computing alpha/beta from A_log/dt_bias each step
//   - applying any per-head k/v scaling (RoPE-on-K, etc.) before launch

#include <cuda_fp16.h>

extern "C" __global__ void gated_delta_state_update_f16_kernel(
    __half* __restrict__ state,            // [num_heads, d_v, d_k] f16, in-place
    const __half* __restrict__ k,          // [num_heads, d_k] f16
    const __half* __restrict__ v,          // [num_heads, d_v] f16
    const float* __restrict__ alpha,       // [num_heads] f32 decay
    const float* __restrict__ beta,        // [num_heads] f32 write strength
    int num_heads,
    int d_k,
    int d_v
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int n_threads = blockDim.x * blockDim.y;
    int n_elems = d_v * d_k;
    float a = alpha[h];
    float b = beta[h];
    const __half* k_h = k + (long long)h * d_k;
    const __half* v_h = v + (long long)h * d_v;
    __half* s_h = state + (long long)h * d_v * d_k;
    for (int i = tid; i < n_elems; i += n_threads) {
        int v_idx = i / d_k;
        int k_idx = i % d_k;
        float s = __half2float(s_h[i]);
        float kv = __half2float(k_h[k_idx]) * __half2float(v_h[v_idx]);
        s_h[i] = __float2half(a * s + b * kv);
    }
}
