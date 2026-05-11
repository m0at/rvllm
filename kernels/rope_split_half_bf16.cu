// Mistral-style RoPE on a [n_heads, head_dim] BF16 buffer in-place.
//
// HuggingFace `rotate_half` convention: pair (i, i+half) for i in 0..half:
//   out[i]      = x[i]      * cos[i] - x[i+half] * sin[i]
//   out[i+half] = x[i+half] * cos[i] + x[i]      * sin[i]
//
// cos / sin tables are F32 [max_pos, head_dim/2] row-major, built host-side
// from the YaRN config in `mistral35_yarn::build_yarn_rope_tables` and
// uploaded once at bring-up. `position` is the absolute position
// 0..max_pos-1 the kernel reads from.
//
// Launch:
//   Grid:  (n_heads, 1, 1)
//   Block: (head_dim / 2, 1, 1)   — head_dim must be even, ≤ 2048
//
// One block per attention head; each thread handles exactly one (i, i+half)
// pair so register pressure stays small.

#include <cuda_bf16.h>

extern "C" __global__ void rope_split_half_bf16_kernel(
    __nv_bfloat16* __restrict__ qk,           // [n_heads, head_dim] in-place
    const float* __restrict__ cos_table,      // [max_pos, head_dim/2]
    const float* __restrict__ sin_table,      // [max_pos, head_dim/2]
    int head_dim,
    int position
) {
    const int h    = blockIdx.x;
    const int i    = threadIdx.x;
    const int half = head_dim >> 1;
    if (i >= half) return;

    const int row_off = position * half + i;
    const float c = cos_table[row_off];
    const float s = sin_table[row_off];

    const int idx_lo = h * head_dim + i;
    const int idx_hi = idx_lo + half;

    const float a = __bfloat162float(qk[idx_lo]);
    const float b = __bfloat162float(qk[idx_hi]);

    qk[idx_lo] = __float2bfloat16(a * c - b * s);
    qk[idx_hi] = __float2bfloat16(b * c + a * s);
}
