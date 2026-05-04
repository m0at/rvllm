// 2D position-embedding lookup-and-add for Gemma 4 vision.
//
// pos_table: [2, num_pos, hidden] f16 (axis 0 = row, axis 1 = col, etc.)
// row_pos:   [N] i32 row index per token (in [0, num_pos))
// col_pos:   [N] i32 col index per token
// hidden:    [N, hidden] f16, in-place
//
// hidden[t, c] += pos_table[0, row_pos[t], c] + pos_table[1, col_pos[t], c]
//
// Launch:
//   Grid:  (N, 1, 1)
//   Block: (BLOCK, 1, 1)        each thread covers multiple channels

#include <cuda_fp16.h>

extern "C" __global__ void vit_pos_emb_lookup_2d_f16_kernel(
    __half* __restrict__ hidden,           // [N, hidden] in-place
    const __half* __restrict__ pos_table,  // [2, num_pos, hidden]
    const int* __restrict__ row_pos,       // [N]
    const int* __restrict__ col_pos,       // [N]
    int num_pos,
    int hidden_dim
) {
    const int t = blockIdx.x;
    const long long row = row_pos[t];
    const long long col = col_pos[t];

    const __half* row_p = pos_table + 0LL * num_pos * hidden_dim + row * hidden_dim;
    const __half* col_p = pos_table + 1LL * num_pos * hidden_dim + col * hidden_dim;
    __half* dst = hidden + (long long)t * hidden_dim;

    for (int c = threadIdx.x; c < hidden_dim; c += blockDim.x) {
        float v = __half2float(dst[c])
                + __half2float(row_p[c])
                + __half2float(col_p[c]);
        dst[c] = __float2half(v);
    }
}
