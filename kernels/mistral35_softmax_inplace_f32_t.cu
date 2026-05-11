// =============================================================
//  mistral35_softmax_inplace_f32_t.cu
// =============================================================
//
// Codex #2 new review — batched in-place softmax over
// `scores_t [T, n_q_heads, max_past_len]`. Sibling to
// `mistral35_softmax_inplace_f32_kernel`; same numerically-stable
// pass structure (max → exp-and-sum → normalise) but operates on
// one (t, qh) row per CTA, with the third grid dim covering T.
//
// Causal mask is folded into the input by the qk_dot_t producer
// (positions > pos_start + t contain -INF); iterating the full
// row makes those entries exp() to 0 and they contribute neither
// to max nor to sum — byte-for-byte identical to a length-clamped
// softmax on the [0, pos_start + t] prefix.
//
// Launch:
//   Grid:  (n_q_heads, T, 1)
//   Block: (TPB, 1, 1)        — typically 128/256; smem = TPB*4
//   Smem:  TPB * sizeof(float)

#include <float.h>

extern "C" __global__ void mistral35_softmax_inplace_f32_t_kernel(
    float* __restrict__ scores_t,         // [T, n_q, max_past_len]
    int n_q_heads,
    int max_past_len,
    int t_count)
{
    const int qh  = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    const int tpb = blockDim.x;
    if (qh >= n_q_heads || t >= t_count) return;

    float* row = scores_t
        + (long long)t * (long long)n_q_heads * max_past_len
        + (long long)qh * max_past_len;

    extern __shared__ float smem[];

    // Pass 1: max-reduce.
    float local_max = -FLT_MAX;
    for (int p = tid; p < max_past_len; p += tpb) {
        const float v = row[p];
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

    // Pass 2: exp + sum.
    float local_sum = 0.0f;
    for (int p = tid; p < max_past_len; p += tpb) {
        const float e = (row[p] <= -FLT_MAX * 0.5f) ? 0.0f
                                                    : __expf(row[p] - gmax);
        row[p] = e;
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
    for (int p = tid; p < max_past_len; p += tpb) {
        row[p] *= inv_sum;
    }
}
