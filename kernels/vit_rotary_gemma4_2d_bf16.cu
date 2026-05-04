// Gemma 4 multidimensional 2D rotary (bf16 sibling).
//
// Same per-chunk semantics as vit_rotary_gemma4_2d_f16: head split
// into two 36-channel chunks, rotate_half within each chunk
// independently. Cos/sin tables are bf16 [seq_len, head_dim].
//
// Layout:
//   qk:        [seq_len, num_heads, head_dim] bf16, in-place
//   cos/sin:   [seq_len, head_dim]            bf16
//
// Launch:
//   Grid:  (seq_len, num_heads, 1)
//   Block: (rotary_dim/2, 1, 1)        // = head_dim/4 = 18 for Gemma 4

#include <cuda_bf16.h>

extern "C" __global__ void vit_rotary_gemma4_2d_bf16_kernel(
    __nv_bfloat16* __restrict__ qk,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    int num_heads,
    int head_dim
) {
    const int tok = blockIdx.x;
    const int head = blockIdx.y;
    const int chunk_size = head_dim / 2;
    const int half = chunk_size / 2;
    const int t = threadIdx.x;
    if (t >= half) return;

    const long long base = ((long long)tok * num_heads + head) * head_dim;
    const long long table_off = (long long)tok * head_dim;

    {
        const int i = t;
        const int j = t + half;
        float qi = __bfloat162float(qk[base + i]);
        float qj = __bfloat162float(qk[base + j]);
        float ci = __bfloat162float(cos_table[table_off + i]);
        float si = __bfloat162float(sin_table[table_off + i]);
        float cj = __bfloat162float(cos_table[table_off + j]);
        float sj = __bfloat162float(sin_table[table_off + j]);
        qk[base + i] = __float2bfloat16(qi * ci - qj * si);
        qk[base + j] = __float2bfloat16(qj * cj + qi * sj);
    }
    {
        const int i = t + chunk_size;
        const int j = t + chunk_size + half;
        float qi = __bfloat162float(qk[base + i]);
        float qj = __bfloat162float(qk[base + j]);
        float ci = __bfloat162float(cos_table[table_off + i]);
        float si = __bfloat162float(sin_table[table_off + i]);
        float cj = __bfloat162float(cos_table[table_off + j]);
        float sj = __bfloat162float(sin_table[table_off + j]);
        qk[base + i] = __float2bfloat16(qi * ci - qj * si);
        qk[base + j] = __float2bfloat16(qj * cj + qi * sj);
    }
}
