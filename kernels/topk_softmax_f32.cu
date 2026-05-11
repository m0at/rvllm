// GPU top-k + softmax for Qwen 3.6 MoE routing.
//
// Input:  logits  [num_experts] f32 (num_experts <= block size, ≤1024)
// Output: top_idx [k] i32, top_w [k] f32   (k <= 32 in practice)
//
// Algorithm: k rounds of block-wide argmax. Each round picks the
// remaining max via a 1024-wide warp/block reduction and masks it
// out for the next round. After k rounds, softmax is applied to
// the saved top-k logits (numerically stable: subtract running max).
//
// Designed to feed an indirect FP8 GEMV path so the per-expert
// kernels can read their expert index from `top_idx` on the device
// instead of having it baked into launch params — that's the
// prerequisite for CUDA Graph capture of the MoE forward
// (Phase 4b-prep iter33).
//
// Launch:
//   Grid:  (1, 1, 1)
//   Block: (num_experts, 1, 1)    — must be num_experts (≤ 1024)
//   Shared: ~12 KiB

#include <cuda_fp16.h>
#include <math.h>

#define MAX_BLOCK 1024
#define MAX_K     32

extern "C" __global__ void __launch_bounds__(MAX_BLOCK)
topk_softmax_f32_kernel(
    const float* __restrict__ logits,    // [num_experts] f32
    int*         __restrict__ top_idx,   // [k] i32
    float*       __restrict__ top_w,     // [k] f32
    int num_experts,
    int k
) {
    const int tid = threadIdx.x;

    // Per-thread local copy of the logit (modifiable for masking).
    float my_val = (tid < num_experts) ? logits[tid] : -INFINITY;
    int   my_idx = tid;

    __shared__ float s_val[MAX_BLOCK];
    __shared__ int   s_idx[MAX_BLOCK];

    __shared__ int   sel_idx[MAX_K];
    __shared__ float sel_val[MAX_K];

    // k rounds of block-wide argmax.
    for (int kk = 0; kk < k; kk++) {
        s_val[tid] = my_val;
        s_idx[tid] = my_idx;
        __syncthreads();

        // Tree reduction over all `num_experts` slots. Power-of-two
        // safe in the typical case (256, 1024); orphan-fold for
        // non-pow-2 sizes mirrors argmax.cu's pattern.
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

        // Mask out the chosen winner for next round.
        if (my_idx == sel_idx[kk]) {
            my_val = -INFINITY;
        }
        __syncthreads();
    }

    // Softmax over the k chosen logits. sel_val[0] is the global
    // max (from round 0) so subtraction gives stable exp(). Every
    // lane in warp 0 must participate in __shfl_xor_sync — lanes
    // >= k contribute zero so the sum is correct.
    if (tid < 32) {
        float m = sel_val[0];
        float e = (tid < k) ? expf(sel_val[tid] - m) : 0.0f;
        float sum = e;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        }
        if (tid < k) {
            top_idx[tid] = sel_idx[tid];
            top_w[tid]   = e / sum;
        }
    }
}
