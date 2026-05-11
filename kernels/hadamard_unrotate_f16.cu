// Apply R^T = diag(D) · H to a per-(token, head) f16 vector in place.
//
// Companion to the Hadamard V-rotation in fused_rope_partial_nvfp4kv.cu:
// when V is stored rotated (V_cache = V · R), the attention output
// becomes attn_out = P · V · R. To recover the original P·V before the
// O-projection, multiply attn_out on the right by R^T per (token, head).
//
// R = H · diag(D)  =>  R^T = diag(D)^T · H^T = diag(D) · H  (H is symmetric and orthonormal).
//
// So R^T applied to a row vector x is: x_new = x · diag(D) · H = (x · diag(D)) · H.
// In per-element form for a length-D vector x:
//   x_new[k] = sum_j x[j] * D[j] * H[j, k]
// Equivalent (treating x as a column for clarity):
//   1) y = D ⊙ x          (element-wise)
//   2) z = H · y           (FWHT)
// Reverse of the rotation order in the rope kernel (which is H first, then D).
//
// Wait — rope applies R = H · diag(D) to a row vector x as
//   x_rot = x · R = x · H · diag(D) = (x · H) · diag(D)
// Step order: H multiply, then D ⊙. (See `apply_signs_f32 + fwht_inplace_f32`
// in the rope kernel: signs first, then fwht. Treating the FWHT helper
// as `H · y` left-multiplication on a column vector, this is x_rot[k] =
// sum_j H[k,j] * D[j] * x[j] which equals (x · H · diag(D))[k] only if H
// is symmetric — which it is.)
//
// For R^T = diag(D) · H applied to row vector x:
//   x_unrot = x · R^T = x · diag(D) · H = (x · diag(D)) · H
// Step order: D ⊙ first, then H. SAME order as rope ("apply_signs then
// fwht") because of H's symmetry. So we can reuse the same helpers.
//
// Equivalently, since R · R^T = I, applying R then R^T (in that order
// of left-mults on column-vector formulation) round-trips to identity.
// We exploit this: per-(token, head), run apply_signs + fwht — the
// EXACT same sequence the rope kernel runs for the K/V rotation.
// That's because (H · diag(D)) · (H · diag(D))^T = H · diag(D) · diag(D) · H
// = H · I · H = H^2 / D  ... wait that's not identity.
//
// Let me re-derive properly. With R = H · diag(D):
//   R · R^T = H · diag(D) · (H · diag(D))^T
//           = H · diag(D) · diag(D) · H^T   (transpose reverses order)
//           = H · diag(D)^2 · H              (D^2 = I element-wise since D ∈ {±1})
//           = H · I · H
//           = H · H
//           = I                              (H · H = I when H is normalized Hadamard)
// OK so R · R^T = I.
//
// For row-vector application with the FWHT helper acting as left-mult by H:
//   x_rot = x · R = (x_T)^T · (H · diag(D))^T (transpose to column form)
//                 = (diag(D) · H · x_T)^T
//   Step 1: y = H · x_T   (FWHT applied to column)
//   Step 2: x_rot_T = D ⊙ y   (element-wise)
//   Final: x_rot = (D ⊙ (H · x_T))^T
//
// So the FWHT-then-signs order recovers x_rot. Looking at rope kernel:
//   apply_signs_f32(s_hadamard, signs, head_dim);   // x_T = D ⊙ x
//   fwht_inplace_f32(s_hadamard, head_dim);          // x_T = H · (D ⊙ x)
// That gives x_rot = H · D ⊙ x = (D · H) · x_T (treating diag(D) as left-mult).
// Hmm, this corresponds to R = H · diag(D)?  Yes: R · x_T = H · diag(D) · x_T
// = H · (D ⊙ x). Same as the rope kernel sequence. Good.
//
// For un-rotation (R^T = diag(D) · H, apply_signs SECOND):
//   y = H · x_T          (FWHT)
//   x_unrot_T = D ⊙ y    (signs)
// Sequence: fwht_inplace_f32 first, then apply_signs_f32. REVERSED from rope.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "hadamard.cuh"

extern "C"
__global__ void hadamard_unrotate_f16_kernel(
    __half*       __restrict__ x,            // [num_tokens, num_heads, head_dim] f16
    const signed char* __restrict__ signs,   // [head_dim] ±1 per channel (per-layer fixed)
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    if (tid >= head_dim) return;
    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    __shared__ float s_buf[512];

    const int base = (token_idx * num_heads + head_idx) * head_dim;

    // Stage to f32 smem
    s_buf[tid] = __half2float(x[base + tid]);
    __syncthreads();

    // Apply R^T = diag(D) · H. Reverse order from rope (which does
    // signs then fwht for R = H · diag(D)).
    rvllm_hadamard::fwht_inplace_f32(s_buf, head_dim);
    rvllm_hadamard::apply_signs_f32(s_buf, signs, head_dim);

    // Write back as f16
    x[base + tid] = __float2half(s_buf[tid]);
}
