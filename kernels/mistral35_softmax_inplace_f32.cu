// Mistral 3.5 parallel in-place softmax over `scores[n_q_heads, past_len]`.
//
// Replaces the serial-in-thread-0 softmax stage of the legacy
// `mistral35_softmax_v_bf16` finisher. Parallelises max + sum
// reductions across the head_dim threads of one CTA per Q-head,
// then strides the normalisation across the same threads. The
// output overwrites the input row in place — matched by the
// follow-up `mistral35_v_dot_gqa_bf16` kernel which consumes the
// normalised probs.
//
// Numerically stable softmax: subtract max, exp, divide by sum.
//
// Launch:
//   Grid:  (n_q_heads, 1, 1)
//   Block: (TPB, 1, 1)         — typically head_dim (=128) but any
//                                power of two is fine
//   Smem:  TPB * sizeof(float) — reduction scratch

#include <float.h>

extern "C" __global__ void mistral35_softmax_inplace_f32_kernel(
    float* __restrict__ scores,
    int n_q_heads,
    int past_len
) {
    const int qh  = blockIdx.x;
    const int tid = threadIdx.x;
    const int tpb = blockDim.x;
    if (qh >= n_q_heads || past_len <= 0) return;

    float* row = scores + (long long)qh * past_len;

    extern __shared__ float smem[];

    // Pass 1: max-reduce.
    float local_max = -FLT_MAX;
    for (int t = tid; t < past_len; t += tpb) {
        const float v = row[t];
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = tpb >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = smem[tid], b = smem[tid + s];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    const float gmax = smem[0];
    __syncthreads();

    // Pass 2: write unnormalised exp(s - gmax) back, accumulate sum.
    float local_sum = 0.0f;
    for (int t = tid; t < past_len; t += tpb) {
        const float e = __expf(row[t] - gmax);
        row[t] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = tpb >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float gsum = smem[0];
    const float inv_sum = (gsum > 0.0f) ? 1.0f / gsum : 0.0f;
    __syncthreads();

    // Pass 3: normalise in place.
    for (int t = tid; t < past_len; t += tpb) {
        row[t] *= inv_sum;
    }
}
