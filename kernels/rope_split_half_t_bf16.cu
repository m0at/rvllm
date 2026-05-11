// Batched RoPE for chunk-path prefill: applies HuggingFace
// `rotate_half` to a `[T, n_heads, head_dim]` BF16 buffer in place
// at positions `[pos_start .. pos_start + T)`. Drop-in batched
// replacement for the per-position `rope_split_half_bf16_kernel`
// which the chunked-prefill path was calling T times per layer.
//
// At T=383 / 88 layers, the per-position dispatch fires 67K kernel
// launches per request just for RoPE; this batched form collapses
// it to 88 launches.
//
// Maths is identical:
//   out[t, h, i]      = x[t, h, i]      * cos[(pos_start+t), i]
//                       - x[t, h, i+half] * sin[(pos_start+t), i]
//   out[t, h, i+half] = x[t, h, i+half] * cos[(pos_start+t), i]
//                       + x[t, h, i]      * sin[(pos_start+t), i]
//
// Launch:
//   Grid:  (n_heads, T, 1)
//   Block: (head_dim / 2, 1, 1)
//
// One CTA per (head, t) pair; threads within a CTA handle the
// `head_dim/2` (i, i+half) element pairs. The CTA's `t` axis
// selects the row, position, and cos/sin entry.

#include <cuda_bf16.h>

extern "C" __global__ void rope_split_half_t_bf16_kernel(
    __nv_bfloat16* __restrict__ qk,           // [T, n_heads, head_dim] in-place
    const float*  __restrict__ cos_table,     // [max_pos, head_dim/2]
    const float*  __restrict__ sin_table,     // [max_pos, head_dim/2]
    int n_heads,
    int head_dim,
    int pos_start,
    int t_count
) {
    const int h    = blockIdx.x;
    const int t    = blockIdx.y;
    const int i    = threadIdx.x;
    const int half = head_dim >> 1;
    if (h >= n_heads || t >= t_count || i >= half) return;

    const int position = pos_start + t;
    const int cos_row  = position * half + i;
    const float c = cos_table[cos_row];
    const float s = sin_table[cos_row];

    const long long row_stride = (long long)n_heads * head_dim;
    const long long row_off    = (long long)t * row_stride;
    const int idx_lo = (int)(row_off + (long long)h * head_dim + i);
    const int idx_hi = idx_lo + half;

    const float a = __bfloat162float(qk[idx_lo]);
    const float b = __bfloat162float(qk[idx_hi]);

    qk[idx_lo] = __float2bfloat16(a * c - b * s);
    qk[idx_hi] = __float2bfloat16(b * c + a * s);
}
