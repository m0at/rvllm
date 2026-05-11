// Batched-prefill counterpart of `topk_softmax_f32`. Per-token GPU
// top-k + softmax over a batched [num_tokens, num_experts] logits
// tensor.
//
// Input:  logits   [num_tokens, num_experts] f32
// Output: top_idx  [num_tokens, k] i32
//         top_w    [num_tokens, k] f32
//
// Each block processes ONE token (block.x = num_experts threads,
// grid.x = num_tokens). Algorithm is the per-token kernel verbatim:
// k rounds of block-wide argmax, then softmax over the k chosen
// logits. Independence across tokens means the batched version is a
// pure grid-axis extension; per-token output is byte-identical to
// the single-token kernel called N times from the host.
//
// Phase 6a / Round-27.

#include <cuda_fp16.h>
#include <math.h>

#define MAX_BLOCK 1024
#define MAX_K     32

extern "C" __global__ void __launch_bounds__(MAX_BLOCK)
topk_softmax_batched_f32_kernel(
    const float* __restrict__ logits,    // [num_tokens, num_experts] f32
    int*         __restrict__ top_idx,   // [num_tokens, k] i32
    float*       __restrict__ top_w,     // [num_tokens, k] f32
    int num_experts,
    int k,
    int num_tokens
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= num_tokens) return;

    const float* my_logits  = logits  + (long long)t * num_experts;
    int*         my_top_idx = top_idx + (long long)t * k;
    float*       my_top_w   = top_w   + (long long)t * k;

    float my_val = (tid < num_experts) ? my_logits[tid] : -INFINITY;
    int   my_idx = tid;

    __shared__ float s_val[MAX_BLOCK];
    __shared__ int   s_idx[MAX_BLOCK];
    __shared__ int   sel_idx[MAX_K];
    __shared__ float sel_val[MAX_K];

    for (int kk = 0; kk < k; kk++) {
        s_val[tid] = my_val;
        s_idx[tid] = my_idx;
        __syncthreads();

        int n = num_experts;
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
            sel_idx[kk] = s_idx[0];
            sel_val[kk] = s_val[0];
        }
        __syncthreads();

        if (my_idx == sel_idx[kk]) {
            my_val = -INFINITY;
        }
        __syncthreads();
    }

    if (tid < 32) {
        float m = sel_val[0];
        float e = (tid < k) ? expf(sel_val[tid] - m) : 0.0f;
        float sum = e;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        }
        if (tid < k) {
            my_top_idx[tid] = sel_idx[tid];
            my_top_w[tid]   = e / sum;
        }
    }
}
