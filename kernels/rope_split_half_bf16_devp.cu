// Device-pointer position variant of rope_split_half_bf16 — codex
// review #2 graph-capture follow-up. Identical math to
// `rope_split_half_bf16_kernel`; the only change is `position` is
// passed as `const int*` so the launch-args can stay constant
// across the captured graph's replay loop and the host only has
// to update the underlying device int between decode steps.
//
// Launch:
//   Grid:  (n_heads, 1, 1)
//   Block: (head_dim / 2, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void rope_split_half_bf16_devp_kernel(
    __nv_bfloat16* __restrict__ qk,           // [n_heads, head_dim] in-place
    const float* __restrict__ cos_table,      // [max_pos, head_dim/2]
    const float* __restrict__ sin_table,      // [max_pos, head_dim/2]
    int head_dim,
    const int* __restrict__ position_ptr
) {
    const int h    = blockIdx.x;
    const int i    = threadIdx.x;
    const int half = head_dim >> 1;
    if (i >= half) return;
    const int position = *position_ptr;

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
