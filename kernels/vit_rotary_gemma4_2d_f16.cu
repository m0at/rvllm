// Gemma 4 multidimensional 2D rotary for vision tower.
//
// HF apply_multidimensional_rope (modeling_gemma4.py) splits each
// head into TWO 36-channel chunks and applies rotate_half WITHIN
// each chunk independently:
//
//   chunk 0 (channels [0, 36)):  axis-0 rotation
//     pair (i, i+18) for i ∈ [0, 18)
//     q_new[i]    = q[i]   *cos[i]   - q[i+18]*sin[i]
//     q_new[i+18] = q[i+18]*cos[i+18]+ q[i]   *sin[i+18]
//
//   chunk 1 (channels [36, 72)): axis-1 rotation
//     pair (j, j+18) for j ∈ [36, 54)
//     same formula on (q[j], q[j+18])
//
// The cos/sin tables are [seq_len, head_dim=72] f16 with layout:
//   [0..18]   cos(axis_0_pos * inv_freq[k])
//   [18..36]  same as [0..18]                (cat-of-itself, k=0..18)
//   [36..54]  cos(axis_1_pos * inv_freq[k])
//   [54..72]  same as [36..54]
// (HF builds this via two `cat([freqs, freqs], dim=-1)` and a final
// `cat([cos_axis0, cos_axis1], dim=-1)`.)
//
// This kernel differs from vit_rotary_2d_f16 (Qwen 3.6) in that Qwen
// applies rotate_half over the FULL head_dim (pairing 0..36 with
// 36..72 globally), whereas Gemma 4 keeps the two axis-chunks
// independent.
//
// Layout:
//   qk:        [seq_len, num_heads, head_dim] f16, in-place
//   cos/sin:   [seq_len, head_dim]            f16
//
// Launch:
//   Grid:  (seq_len, num_heads, 1)
//   Block: (rotary_dim/2, 1, 1)        // = head_dim/4 = 18 for Gemma 4

#include <cuda_fp16.h>

extern "C" __global__ void vit_rotary_gemma4_2d_f16_kernel(
    __half* __restrict__ qk,                 // [seq_len, num_heads, head_dim]
    const __half* __restrict__ cos_table,    // [seq_len, head_dim]
    const __half* __restrict__ sin_table,    // [seq_len, head_dim]
    int num_heads,
    int head_dim                             // 72 (= 2 chunks of 36)
) {
    const int tok = blockIdx.x;
    const int head = blockIdx.y;
    const int chunk_size = head_dim / 2;     // 36
    const int half = chunk_size / 2;         // 18
    const int t = threadIdx.x;
    if (t >= half) return;

    const long long base = ((long long)tok * num_heads + head) * head_dim;
    const long long table_off = (long long)tok * head_dim;

    // chunk 0: channels [0, 36)
    {
        const int i = t;          // [0, 18)
        const int j = t + half;   // [18, 36)
        float qi = __half2float(qk[base + i]);
        float qj = __half2float(qk[base + j]);
        float ci = __half2float(cos_table[table_off + i]);
        float si = __half2float(sin_table[table_off + i]);
        float cj = __half2float(cos_table[table_off + j]);
        float sj = __half2float(sin_table[table_off + j]);
        qk[base + i] = __float2half(qi * ci - qj * si);
        qk[base + j] = __float2half(qj * cj + qi * sj);
    }

    // chunk 1: channels [36, 72)
    {
        const int i = t + chunk_size;          // [36, 54)
        const int j = t + chunk_size + half;   // [54, 72)
        float qi = __half2float(qk[base + i]);
        float qj = __half2float(qk[base + j]);
        float ci = __half2float(cos_table[table_off + i]);
        float si = __half2float(sin_table[table_off + i]);
        float cj = __half2float(cos_table[table_off + j]);
        float sj = __half2float(sin_table[table_off + j]);
        qk[base + i] = __float2half(qi * ci - qj * si);
        qk[base + j] = __float2half(qj * cj + qi * sj);
    }
}
