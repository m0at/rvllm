// gated_delta_rule_decode_f16: per-head Gated-DeltaNet decode-step kernel
// for the Qwen 3.6 (qwen3-next) linear-attention block.
//
// Reference: vLLM
//   layers/fla/ops/fused_recurrent.py::fused_recurrent_gated_delta_rule_packed_decode_kernel
//
// Per timestep, per v-head v (with k-head h = v / (HV/H)):
//   1. S[v, vd, kd] *= alpha[v]                                  // forget
//   2. v_corr[vd]    = V[v, vd] - sum_kd(S[v, vd, kd] * K[h, kd]) // delta
//   3. v_corr[vd]   *= beta[v]
//   4. S[v, vd, kd] += v_corr[vd] * K[h, kd]                     // update
//   5. O[v, vd]      = sum_kd(S[v, vd, kd] * Q[h, kd] * scale)   // readout
//
// Caller responsibilities:
//   - Q and K are passed already L2-normed AND already GQA-expanded
//     (one row per v-head, with row v copied from k-head v / (HV/H)).
//     This keeps the kernel agnostic to the GQA mapping.
//   - alpha, beta come from the host computation
//     g = -exp(A_log) * softplus(a + dt_bias),  alpha = exp(g)
//     beta = sigmoid(b)
//   - State is stored f16 [num_v_heads, head_v_dim, head_k_dim] and
//     persists across decode steps. cuMemcpyDtoH is no longer needed
//     between steps; this kernel is the only consumer.
//
// Launch: grid = (num_v_heads,), block = (head_v_dim,). Each thread
// owns one row vd of the state for its v-head, doing K=128 reads to
// compute v_corr and another K=128 to do the readout.

#include <cuda_fp16.h>

extern "C" __global__ void gated_delta_rule_decode_f16_kernel(
    __half*       __restrict__ state,    // [num_v_heads, head_v_dim, head_k_dim] f16, in/out
    const __half* __restrict__ q,        // [num_v_heads, head_k_dim] f16 (GQA-expanded, L2-normed)
    const __half* __restrict__ k,        // [num_v_heads, head_k_dim] f16 (GQA-expanded, L2-normed)
    const __half* __restrict__ v,        // [num_v_heads, head_v_dim] f16
    const float*  __restrict__ alpha,    // [num_v_heads] f32
    const float*  __restrict__ beta,     // [num_v_heads] f32
    __half*       __restrict__ out,      // [num_v_heads, head_v_dim] f16
    float scale,
    int head_v_dim,
    int head_k_dim
) {
    const int v_head = blockIdx.x;
    const int vd     = threadIdx.x;
    if (vd >= head_v_dim) return;

    extern __shared__ float smem[];
    float* k_smem    = smem;                                 // [head_k_dim]
    float* q_smem    = smem + head_k_dim;                    // [head_k_dim]
    float* corr_smem = smem + 2 * head_k_dim;                // [head_v_dim]

    // Stage K and Q for this v-head into shared memory.
    for (int kd = vd; kd < head_k_dim; kd += blockDim.x) {
        k_smem[kd] = __half2float(k[v_head * head_k_dim + kd]);
        q_smem[kd] = __half2float(q[v_head * head_k_dim + kd]);
    }
    __syncthreads();

    const float a = alpha[v_head];
    const float b = beta[v_head];

    // Phase 1: forget S by alpha + compute (v - S·K) for this row.
    float v_corr = __half2float(v[v_head * head_v_dim + vd]);
    long long row_off = ((long long)v_head * head_v_dim + vd) * head_k_dim;
    for (int kd = 0; kd < head_k_dim; ++kd) {
        float s_old = __half2float(state[row_off + kd]);
        float s_new = s_old * a;
        state[row_off + kd] = __float2half(s_new);
        v_corr -= s_new * k_smem[kd];
    }
    v_corr *= b;
    corr_smem[vd] = v_corr;
    __syncthreads();

    // Phase 2: S += v_corr · K, then O = S · (Q · scale) for this row.
    float bv = corr_smem[vd];
    float o_acc = 0.0f;
    for (int kd = 0; kd < head_k_dim; ++kd) {
        float s = __half2float(state[row_off + kd]) + bv * k_smem[kd];
        state[row_off + kd] = __float2half(s);
        o_acc += s * q_smem[kd];
    }
    out[v_head * head_v_dim + vd] = __float2half(o_acc * scale);
}
