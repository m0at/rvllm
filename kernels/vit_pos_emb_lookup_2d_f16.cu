// 2D position-embedding lookup-and-add for Gemma 4 vision.
//
// HF Gemma4VisionPatchEmbedder._position_embeddings (modeling_gemma4.py:550)
// stores `pixel_position_ids` as (x, y) = (col, row); the position table
// has shape [2, num_pos, hidden] and table[0] is indexed by axis-0 of
// pixel_position_ids (= col), table[1] by axis-1 (= row).
//
// Param names use the generic axis_0_pos / axis_1_pos so the
// caller-side semantics are explicit: whichever Rust buffer holds
// "the first axis of pixel_position_ids" goes into axis_0_pos.
//
// hidden[t, c] += pos_table[0, axis_0_pos[t], c]
//              + pos_table[1, axis_1_pos[t], c]
//
// Launch:
//   Grid:  (N, 1, 1)
//   Block: (BLOCK, 1, 1)        each thread covers multiple channels

#include <cuda_fp16.h>

extern "C" __global__ void vit_pos_emb_lookup_2d_f16_kernel(
    __half* __restrict__ hidden,             // [N, hidden] in-place
    const __half* __restrict__ pos_table,    // [2, num_pos, hidden]
    const int* __restrict__ axis_0_pos,      // [N]
    const int* __restrict__ axis_1_pos,      // [N]
    int num_pos,
    int hidden_dim
) {
    const int t = blockIdx.x;
    const long long a0 = axis_0_pos[t];
    const long long a1 = axis_1_pos[t];

    const __half* p0 = pos_table + 0LL * num_pos * hidden_dim + a0 * hidden_dim;
    const __half* p1 = pos_table + 1LL * num_pos * hidden_dim + a1 * hidden_dim;
    __half* dst = hidden + (long long)t * hidden_dim;

    for (int c = threadIdx.x; c < hidden_dim; c += blockDim.x) {
        float v = __half2float(dst[c])
                + __half2float(p0[c])
                + __half2float(p1[c]);
        dst[c] = __float2half(v);
    }
}
