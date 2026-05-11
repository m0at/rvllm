// 2D rotary position embedding for Qwen 3.6 ViT (partial rotary).
//
// Matches vLLM Qwen3_VisionTransformer (qwen3_vl.py:565):
//   rotary_pos_emb = get_rope(head_size=head_dim, ..., is_neox_style=True,
//                             rope_parameters={"partial_rotary_factor": 0.5})
//
// Only the first `rotary_dim = head_dim/2` channels are rotated; the
// remaining `head_dim - rotary_dim` channels pass through unchanged.
// Within the rotated region, NeoX-style rotate_half is applied:
//   q_new[i] = q[i]*cos[i] + (-q[i+R/2])*sin[i]    for i ∈ [0, R/2)
//   q_new[j] = q[j]*cos[j] +  q[j-R/2] *sin[j]     for j ∈ [R/2, R)
// where R = rotary_dim. The 2D-ness is encoded in the cos/sin tables:
//   cos[t, 0..R/2]   = cos(row_freq[h_pos[t]])    // rotates with row
//   cos[t, R/2..R]   = cos(col_freq[w_pos[t]])    // rotates with col
// so the (i, i+R/2) pair mixes one row-rotated and one col-rotated
// channel — exactly HF apply_rotary_pos_emb_vision.
//
// Layout:
//   qk:        [seq_len, num_heads, head_dim] f16, in-place
//   cos/sin:   [seq_len, rotary_dim]          f16
//
// Launch:
//   Grid:  (seq_len, num_heads, 1)
//   Block: (rotary_dim/2, 1, 1)            // each thread handles one (i, i+R/2) pair

#include <cuda_fp16.h>

extern "C" __global__ void vit_rotary_2d_f16_kernel(
    __half* __restrict__ qk,                 // [seq_len, num_heads, head_dim] in-place
    const __half* __restrict__ cos_table,    // [seq_len, rotary_dim]
    const __half* __restrict__ sin_table,    // [seq_len, rotary_dim]
    int num_heads,
    int head_dim,
    int rotary_dim
) {
    const int tok = blockIdx.x;
    const int head = blockIdx.y;
    const int half = rotary_dim / 2;
    const int i = threadIdx.x;
    if (i >= half) return;
    const int j = i + half;

    long long base = ((long long)tok * num_heads + head) * head_dim;
    long long table_off = (long long)tok * rotary_dim;

    float c_i = __half2float(cos_table[table_off + i]);
    float s_i = __half2float(sin_table[table_off + i]);
    float c_j = __half2float(cos_table[table_off + j]);
    float s_j = __half2float(sin_table[table_off + j]);

    float qi = __half2float(qk[base + i]);
    float qj = __half2float(qk[base + j]);
    qk[base + i] = __float2half(qi * c_i - qj * s_i);
    qk[base + j] = __float2half(qj * c_j + qi * s_j);
    // Channels [rotary_dim, head_dim) untouched by design (partial rotary).
}
