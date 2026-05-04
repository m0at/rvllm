// 2D rotary position embedding for Qwen 3.6 ViT.
//
// The vision tower splits each head's head_dim into two halves: the
// first half (head_dim/2 channels) is rotated using row coordinates
// (position-h), the second half by column coordinates (position-w).
// Within each half, NeoX-style pairing is used: dims (i, i+rot/4) get
// rotated together.
//
// Reference: vllm-git/vllm/model_executor/models/qwen3_vl.py:
//   rot_pos_emb(grid_thw) generates a [seq_len, head_dim/2] table
//   that is duplicated to [seq_len, head_dim] via cat([cos, cos]).
// Here we apply it in-place to Q and K tensors of shape
// [seq_len, num_heads, head_dim] f16.
//
// Layout convention:
//   - tokens are flattened in (t, gh, gw) order matching grid_thw
//   - cos_h_table[i] = cos(angle for height pos i_h)
//     sin_h_table[i] = sin(...)
//     cos_w_table[j] = cos(angle for width  pos j_w)
//     sin_w_table[j] = sin(...)
//     each of shape [grid_max, head_dim/4]
//   - per token at (i_h, j_w):
//       lower half (dims 0..head_dim/2)  rotated by (cos_h, sin_h)
//       upper half (dims head_dim/2..)   rotated by (cos_w, sin_w)
//
// Launch:
//   Grid:  (seq_len, num_heads, 1)
//   Block: (head_dim/4, 1, 1)         // each thread handles 2 dims per half
//
// Each thread rotates a (lo, lo+head_dim/4) pair AND a
// (head_dim/2 + lo, head_dim/2 + lo + head_dim/4) pair using the
// per-token h/w cos/sin lookup.

#include <cuda_fp16.h>

extern "C" __global__ void vit_rotary_2d_f16_kernel(
    __half* __restrict__ qk,                 // [seq_len, num_heads, head_dim] in-place
    const __half* __restrict__ cos_h_table,  // [grid_max_h, head_dim/4]
    const __half* __restrict__ sin_h_table,  // [grid_max_h, head_dim/4]
    const __half* __restrict__ cos_w_table,  // [grid_max_w, head_dim/4]
    const __half* __restrict__ sin_w_table,  // [grid_max_w, head_dim/4]
    const int* __restrict__ pos_h,           // [seq_len] row index per token
    const int* __restrict__ pos_w,           // [seq_len] col index per token
    int num_heads,
    int head_dim
) {
    const int tok = blockIdx.x;
    const int head = blockIdx.y;
    const int half = head_dim / 2;
    const int quarter = head_dim / 4;
    const int lo = threadIdx.x;
    if (lo >= quarter) return;

    const int ih = pos_h[tok];
    const int iw = pos_w[tok];

    float cos_h = __half2float(cos_h_table[ih * quarter + lo]);
    float sin_h = __half2float(sin_h_table[ih * quarter + lo]);
    float cos_w = __half2float(cos_w_table[iw * quarter + lo]);
    float sin_w = __half2float(sin_w_table[iw * quarter + lo]);

    long long base = ((long long)tok * num_heads + head) * head_dim;
    // Lower half: rotate by H pos. Pair (lo, lo+quarter) inside the half.
    {
        long long a_idx = base + lo;
        long long b_idx = base + lo + quarter;
        float a = __half2float(qk[a_idx]);
        float b = __half2float(qk[b_idx]);
        qk[a_idx] = __float2half(a * cos_h - b * sin_h);
        qk[b_idx] = __float2half(a * sin_h + b * cos_h);
    }
    // Upper half: rotate by W pos. Pair (half+lo, half+lo+quarter).
    {
        long long a_idx = base + half + lo;
        long long b_idx = base + half + lo + quarter;
        float a = __half2float(qk[a_idx]);
        float b = __half2float(qk[b_idx]);
        qk[a_idx] = __float2half(a * cos_w - b * sin_w);
        qk[b_idx] = __float2half(a * sin_w + b * cos_w);
    }
}
