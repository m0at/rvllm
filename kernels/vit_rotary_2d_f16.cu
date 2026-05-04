// 2D rotary position embedding for Qwen 3.6 ViT.
//
// Matches HF apply_rotary_pos_emb_vision (transformers/qwen3_vl):
//   q_new[i] = q[i]*cos[i] + (-q[i+H/2])*sin[i]    for i ∈ [0, H/2)
//   q_new[j] = q[j]*cos[j] + q[j-H/2]*sin[j]       for j ∈ [H/2, H)
// where H = head_dim, and cos/sin are per-token tables of shape
// [seq_len, head_dim]. The "2D-ness" is encoded in the tables: the
// host computes
//   cos_table[t, 0..H/2]    = cos(row_freq[h_pos[t]])
//   cos_table[t, H/2..H]    = cos(col_freq[w_pos[t]])
// so the lower half of the head is rotated with the row's H-frequency
// and the upper half with the column's W-frequency, but inside each
// (i, i+H/2) pair the rotate_half mixing crosses both axes — exactly
// what HF's apply_rotary_pos_emb_vision does.
//
// Layout:
//   qk:        [seq_len, num_heads, head_dim] f16, in-place
//   cos/sin:   [seq_len, head_dim]            f16
//
// Launch:
//   Grid:  (seq_len, num_heads, 1)
//   Block: (head_dim/2, 1, 1)              // each thread handles one (i, i+H/2) pair

#include <cuda_fp16.h>

extern "C" __global__ void vit_rotary_2d_f16_kernel(
    __half* __restrict__ qk,                 // [seq_len, num_heads, head_dim] in-place
    const __half* __restrict__ cos_table,    // [seq_len, head_dim]
    const __half* __restrict__ sin_table,    // [seq_len, head_dim]
    int num_heads,
    int head_dim
) {
    const int tok = blockIdx.x;
    const int head = blockIdx.y;
    const int half = head_dim / 2;
    const int i = threadIdx.x;
    if (i >= half) return;
    const int j = i + half;

    long long base = ((long long)tok * num_heads + head) * head_dim;
    long long table_off = (long long)tok * head_dim;

    float c_i = __half2float(cos_table[table_off + i]);
    float s_i = __half2float(sin_table[table_off + i]);
    float c_j = __half2float(cos_table[table_off + j]);
    float s_j = __half2float(sin_table[table_off + j]);

    float qi = __half2float(qk[base + i]);
    float qj = __half2float(qk[base + j]);
    qk[base + i] = __float2half(qi * c_i - qj * s_i);
    qk[base + j] = __float2half(qj * c_j + qi * s_j);
}
