// GPU-side argmax kernel: finds the token ID with maximum logit per row.
// Eliminates full logits DtoH copy for greedy (temperature=0) decoding.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(vocab_size, 1024), 1, 1)
//   Shared memory: none (uses static shared arrays)
//
// Each block finds the argmax of one token's logits row via shared memory reduction,
// then writes the winning token ID to output_token[row].

#include <float.h>
#include <cuda_fp16.h>

extern "C"
__global__ void argmax_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;

    const __half* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < vocab_size; i += stride) {
        float v = __half2float(x[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        if (s * 2 < n && tid == 0) {
            if (s_val[s * 2] > s_val[0]) {
                s_val[0] = s_val[s * 2];
                s_idx[0] = s_idx[s * 2];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_token[row] = s_idx[0];
    }
}

extern "C"
__global__ void argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;

    const float* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    // Pass 1: thread-local max across strided elements
    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < vocab_size; i += stride) {
        float v = x[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Tree reduction for argmax. Handles any `n` (= blockDim.x), not
    // just powers-of-two — the orphan check below covers the slot
    // that the main pair-fold misses on each step.
    //
    // Correctness argument for non-power-of-two n:
    //   Each step folds pairs (i, i+s) for i in [0, s), so slots
    //   [2*s, n) are NOT touched by the main fold. The orphan check
    //   `if (s*2 < n)` folds the slot at index `2*s` (the leftmost
    //   uncovered slot) into slot 0. The DEPTH of slot `2*s` carries
    //   the data of every higher slot, because at the previous step
    //   `2s_prev = s` and the main fold of THAT step pulled higher
    //   data down into `[0, s_prev)`. Inductively, slot `2*s` after
    //   each step is a complete max over the upper region the main
    //   fold could not reach. Walking n=6 / n=11 / n=14 confirms it.
    //
    //   The reduction therefore terminates with `s_val[0]` holding
    //   the global max. Power-of-two n simply degenerates to the
    //   familiar branch-free version because the orphan check is
    //   never triggered (s*2 == n at every step).
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        // Orphan fold: catch slot `2*s` when n was not perfectly
        // halved by the previous step. See the proof above.
        if (s * 2 < n && tid == 0) {
            if (s_val[s * 2] > s_val[0]) {
                s_val[0] = s_val[s * 2];
                s_idx[0] = s_idx[s * 2];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the result
    if (tid == 0) {
        output_token[row] = s_idx[0];
    }
}
