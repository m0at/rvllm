// 2D rotary position embedding for Pixtral ViT (partial rotary).
// BF16 sibling of vit_rotary_2d_f16, used by the Mistral 3.5 vision
// tower. Rotates only the first `rotary_dim = head_dim/2` channels;
// remaining channels pass through unchanged.
//
// NeoX-style rotate_half:
//   q_new[i] = q[i]*cos[i] + (-q[i+R/2])*sin[i]    for i ∈ [0, R/2)
//   q_new[j] = q[j]*cos[j] +  q[j-R/2] *sin[j]     for j ∈ [R/2, R)
//
// 2D-ness baked into the cos/sin tables (host-built):
//   cos[t, 0..R/2]   = cos(row_freq · row[t])    // row dimension
//   cos[t, R/2..R]   = cos(col_freq · col[t])    // col dimension
//
// Layout:
//   qk:        [seq_len, num_heads, head_dim] BF16, in-place
//   cos/sin:   [seq_len, rotary_dim]          F32
//
// Launch:
//   Grid:  (seq_len, num_heads, 1)
//   Block: (rotary_dim/2, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void pixtral_rotary_2d_bf16_kernel(
    __nv_bfloat16* __restrict__ qk,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int num_heads,
    int head_dim,
    int rotary_dim
) {
    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int half = rotary_dim / 2;
    const int i    = threadIdx.x;
    if (i >= half) return;
    const int j    = i + half;

    const long long base       = ((long long)tok * num_heads + head) * head_dim;
    const long long table_off  = (long long)tok * rotary_dim;

    const float c_i = cos_table[table_off + i];
    const float s_i = sin_table[table_off + i];
    const float c_j = cos_table[table_off + j];
    const float s_j = sin_table[table_off + j];

    const float qi = __bfloat162float(qk[base + i]);
    const float qj = __bfloat162float(qk[base + j]);

    qk[base + i] = __float2bfloat16(qi * c_i - qj * s_i);
    qk[base + j] = __float2bfloat16(qj * c_j + qi * s_j);
}
