// 2D position-embedding lookup-and-add for Gemma 4 vision (bf16 sibling).
//
// pos_table is the bf16 raw upload from the checkpoint; hidden_states is
// the bf16 vision residual buffer in the bf16 vision path.
//
//   hidden[t, c] += pos_table[0, axis_0_pos[t], c]
//                + pos_table[1, axis_1_pos[t], c]

#include <cuda_bf16.h>

extern "C" __global__ void vit_pos_emb_lookup_2d_bf16_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pos_table,
    const int* __restrict__ axis_0_pos,
    const int* __restrict__ axis_1_pos,
    int num_pos,
    int hidden_dim
) {
    const int t = blockIdx.x;
    const long long a0 = axis_0_pos[t];
    const long long a1 = axis_1_pos[t];

    const __nv_bfloat16* p0 =
        pos_table + 0LL * num_pos * hidden_dim + a0 * hidden_dim;
    const __nv_bfloat16* p1 =
        pos_table + 1LL * num_pos * hidden_dim + a1 * hidden_dim;
    __nv_bfloat16* dst = hidden + (long long)t * hidden_dim;

    for (int c = threadIdx.x; c < hidden_dim; c += blockDim.x) {
        float v = __bfloat162float(dst[c])
                + __bfloat162float(p0[c])
                + __bfloat162float(p1[c]);
        dst[c] = __float2bfloat16(v);
    }
}
