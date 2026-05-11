// gated_delta_rule_prefill_f16: batched Gated-DeltaNet prefill kernel
// for the Qwen 3.6 (qwen3-next) linear-attention block.
//
// This is the prefill counterpart of `gated_delta_rule_decode_f16`. It
// processes `num_tokens` Q/K/V/alpha/beta inputs sequentially with the
// recurrent state carried inside the kernel — equivalent to looping
// the decode kernel `num_tokens` times from the host side, but with a
// single launch instead of N. State is read from device memory once at
// the start and written back once at the end (intermediate state lives
// in registers for vd-row, so we avoid the per-token DRAM round-trip
// the host loop suffered from).
//
// Mathematical equivalence to the per-token loop (must hold byte-for-
// byte modulo f16 RTNE differences from intra-loop reads vs. fresh
// per-iteration loads):
//
//   for t = 0..num_tokens:
//       1. S[v, vd, kd] *= alpha[t, v]                            // forget
//       2. v_corr[vd]    = V[t, v, vd] - sum_kd(S[..] * K[t, h, kd])
//       3. v_corr[vd]   *= beta[t, v]
//       4. S[..]        += v_corr[vd] * K[t, h, kd]               // update
//       5. O[t, v, vd]   = sum_kd(S[..] * Q[t, h, kd] * scale)    // readout
//
// Per-iteration we keep the slice S[v, vd, :] (head_k_dim=128 floats)
// in registers per thread, instead of writing it back to DRAM every
// step. That eliminates the dominant cost of the host-loop version.
//
// Caller layout:
//   state : [num_v_heads, head_v_dim, head_k_dim] f16, in/out
//   q     : [num_tokens, num_v_heads, head_k_dim] f16 (GQA-expanded, L2-normed)
//   k     : [num_tokens, num_v_heads, head_k_dim] f16 (GQA-expanded, L2-normed)
//   v     : [num_tokens, num_v_heads, head_v_dim] f16
//   alpha : [num_tokens, num_v_heads] f32
//   beta  : [num_tokens, num_v_heads] f32
//   out   : [num_tokens, num_v_heads, head_v_dim] f16
//
// Launch: grid = (num_v_heads,), block = (head_v_dim,). Each thread
// owns one row vd of the state for its v-head, exactly as in the
// decode kernel.

#include <cuda_fp16.h>

extern "C" __global__ void gated_delta_rule_prefill_f16_kernel(
    __half*       __restrict__ state,    // [num_v_heads, head_v_dim, head_k_dim] f16, in/out
    const __half* __restrict__ q,        // [num_tokens, num_v_heads, head_k_dim] f16
    const __half* __restrict__ k,        // [num_tokens, num_v_heads, head_k_dim] f16
    const __half* __restrict__ v,        // [num_tokens, num_v_heads, head_v_dim] f16
    const float*  __restrict__ alpha,    // [num_tokens, num_v_heads] f32
    const float*  __restrict__ beta,     // [num_tokens, num_v_heads] f32
    __half*       __restrict__ out,      // [num_tokens, num_v_heads, head_v_dim] f16
    float scale,
    int num_tokens,
    int num_v_heads,
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

    // Load the (v_head, vd, :) state row into registers. head_k_dim is
    // a small constant (128) for Qwen 3.6; keep as a stack array. If
    // the kernel ever runs on a head_k_dim > 128 architecture this
    // becomes a perf cliff.
    //
    // Capping at 128 keeps the per-thread register footprint bounded;
    // larger head_k_dim would spill. The host launcher must not call
    // this kernel for head_k_dim > 128.
    constexpr int MAX_HEAD_K_DIM = 128;
    float s_row[MAX_HEAD_K_DIM];
    long long row_off = ((long long)v_head * head_v_dim + vd) * head_k_dim;
    for (int kd = 0; kd < head_k_dim; ++kd) {
        s_row[kd] = __half2float(state[row_off + kd]);
    }

    for (int t = 0; t < num_tokens; ++t) {
        // Stage K[t, v_head, :] and Q[t, v_head, :] into shared memory.
        // Each thread loads strided across the head_k_dim axis.
        long long qk_t_off = ((long long)t * num_v_heads + v_head) * head_k_dim;
        for (int kd = vd; kd < head_k_dim; kd += blockDim.x) {
            k_smem[kd] = __half2float(k[qk_t_off + kd]);
            q_smem[kd] = __half2float(q[qk_t_off + kd]);
        }
        __syncthreads();

        const float a = alpha[(long long)t * num_v_heads + v_head];
        const float b = beta [(long long)t * num_v_heads + v_head];

        // Phase 1: forget S by alpha + compute (v - S·K) for this row.
        // The decode kernel does `state[..] = f16(s_old * a)` and then
        // in Phase 2 re-loads that stored value through f16 RTNE — so
        // s_new used in Phase 2 is the rounded version, not the raw
        // f32. To stay byte-equivalent, we ROUND s_row[kd] through
        // f16 immediately after the multiplication, before Phase 2
        // reads it back.
        long long v_off = ((long long)t * num_v_heads + v_head) * head_v_dim + vd;
        float v_corr = __half2float(v[v_off]);
        for (int kd = 0; kd < head_k_dim; ++kd) {
            float s_new = s_row[kd] * a;
            s_row[kd] = __half2float(__float2half(s_new));
            v_corr -= s_new * k_smem[kd];
        }
        v_corr *= b;
        corr_smem[vd] = v_corr;
        __syncthreads();

        // Phase 2: S += v_corr · K, then O = S · (Q · scale) for this
        // row. The decode kernel re-loads state[..] through f16 here
        // too — same RTNE round-trip — so we round again immediately
        // after the addition.
        float bv = corr_smem[vd];
        float o_acc = 0.0f;
        for (int kd = 0; kd < head_k_dim; ++kd) {
            float s = s_row[kd] + bv * k_smem[kd];
            s_row[kd] = __half2float(__float2half(s));
            o_acc += s * q_smem[kd];
        }
        long long o_off = ((long long)t * num_v_heads + v_head) * head_v_dim + vd;
        out[o_off] = __float2half(o_acc * scale);
        // Sync before next iteration's k_smem/q_smem reload.
        __syncthreads();
    }

    // Write the final state row back to DRAM exactly once per kernel
    // launch — the per-token decode kernel was forced to do this on
    // every step.
    for (int kd = 0; kd < head_k_dim; ++kd) {
        state[row_off + kd] = __float2half(s_row[kd]);
    }
}
