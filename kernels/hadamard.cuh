// === HADAMARD ROTATION ===
// Signed Walsh-Hadamard rotation helpers for NVFP4 KV cache.
//
// R = H * diag(D) where H is the deterministic ±1 normalized Walsh-
// Hadamard matrix (1/sqrt(d) factor) and D is a fixed-seed-per-layer
// ±1 sign vector. R is orthogonal: R*R^T = I, so Q*K^T is invariant
// under simultaneous rotation of Q and K.
//
// Math contract: the rope kernel applies R to BOTH Q and K post-RoPE
// before quantize. Attention kernels are unchanged — they consume
// rotated Q (FP8) and rotated K (NVFP4) and compute the right answer
// because (Q*R)*(K*R)^T = Q * R * R^T * K^T = Q * K^T.
//
// V is NOT rotated in v0 (rotating V requires also rotating O-proj
// weights, which is a separate lift).
//
// The kernels these helpers are inlined into use blockDim.x = head_dim
// threads cooperating per (token, head). FWHT requires inter-thread
// cooperation in shared memory.

#ifndef RVLLM_HADAMARD_CUH_
#define RVLLM_HADAMARD_CUH_

#include <cuda_runtime.h>

namespace rvllm_hadamard {

// In-place FWHT on float32 vector of length D (must be a power of 2,
// D <= blockDim.x). Each thread holds at most one element in `s_buf`,
// indexed by threadIdx.x. Caller must `__syncthreads()` BEFORE calling
// (so writes to s_buf are visible) and AFTER (so callers read the
// post-FWHT values). D=256 and D=512 are both supported (Gemma 4
// head_dim ∈ {256, 512}, both pow2). Normalization by 1/sqrt(D) is
// applied at the end so R = H*diag(D) is exactly orthonormal.
//
// Stage count = log2(D); each stage is a butterfly between i and i^h
// for i with bit h cleared. We use the "lower half adds, upper half
// subtracts" formulation, which generalizes the d=2 case
// [a b] -> [a+b, a-b].
__device__ __forceinline__ void fwht_inplace_f32(
    float* __restrict__ s_buf,
    int D)
{
    const int tid = threadIdx.x;
    for (int h = 1; h < D; h <<= 1) {
        // Each pair (i, i^h) is touched by both threads i and i^h.
        // Read both BEFORE writing so we don't race with our pair.
        float a, b;
        if (tid < D) {
            int j = tid ^ h;
            a = s_buf[tid];
            b = s_buf[j];
        }
        __syncthreads();
        if (tid < D) {
            // Lower-bit-cleared lane adds, lane with bit set subtracts.
            // (i & h)==0 -> tid is "left" of pair, write a+b at tid.
            // (i & h)!=0 -> tid is "right", write a-b (which is
            // s_buf[tid_old] - s_buf[j_old] = a - b since for the
            // right side we read self into a and pair into b).
            if ((tid & h) == 0) {
                s_buf[tid] = a + b;
            } else {
                s_buf[tid] = b - a;  // pair_value - self_value
            }
        }
        __syncthreads();
    }
    // Apply 1/sqrt(D) normalization so R is orthonormal.
    if (tid < D) {
        s_buf[tid] *= rsqrtf((float)D);
    }
    __syncthreads();
}

// Apply per-channel sign flip. `signs[i]` ∈ {+1, -1} stored as int8_t
// (-1 = 0xFF in two's complement). Element-wise so commutes with FWHT
// — caller chooses to apply before or after FWHT (we apply before, so
// rotation order is: x_rot = (D .* x) projected through H).
__device__ __forceinline__ void apply_signs_f32(
    float* __restrict__ s_buf,
    const signed char* __restrict__ signs,
    int D)
{
    const int tid = threadIdx.x;
    if (tid < D) {
        // signs[i] is stored as ±1 in i8.
        int s = (int)signs[tid];
        s_buf[tid] *= (float)s;
    }
    __syncthreads();
}

}  // namespace rvllm_hadamard

#endif  // RVLLM_HADAMARD_CUH_
// === END HADAMARD ROTATION ===
