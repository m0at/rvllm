// Bilinear interpolation of Qwen3-VL learned absolute position embedding.
//
// Matches vllm-git/vllm/model_executor/models/qwen3_vl.py:276
// (pos_embed_interpolate_native): given a [num_grid² , hidden] learned
// table, sample at sub-grid points (h_idx, w_idx) with bilinear weights,
// and emit one row per output position in spatial-merge order:
//   permute(h_blk, w_blk, intra_h, intra_w, hidden) → flat (h*w, hidden).
//
// Adds the result IN PLACE to `hidden_states` (already populated by
// patch_embed), matching qwen3_vl.py:801 `hidden_states += pos_embeds`.
//
// Layout:
//   pos_table: [num_grid * num_grid, hidden] f16  (num_grid=48 for Qwen3-VL)
//   hidden_states: [grid_h * grid_w, hidden] f16, in-place
//
// Per output token (idx in [0, grid_h * grid_w)):
//   merged_h = grid_h / m_size, merged_w = grid_w / m_size
//   bh = idx / (merged_w * m_size * m_size)        (block-row)
//   rem = idx mod (merged_w * m_size * m_size)
//   bw = rem / (m_size * m_size)                    (block-col)
//   intra = rem mod (m_size * m_size)
//   ih = intra / m_size, iw = intra mod m_size
//   h = bh * m_size + ih, w = bw * m_size + iw
//   h_idx = h * (num_grid - 1) / (grid_h - 1)
//   w_idx = w * (num_grid - 1) / (grid_w - 1)
//   bilinear over the 4 nbr cells of pos_table.
//
// Launch:
//   Grid:  (grid_h * grid_w, 1, 1)
//   Block: (hidden, 1, 1)        — but hidden=1152 > 1024, so we use
//                                  block=256 with each thread covering
//                                  multiple channels.

#include <cuda_fp16.h>

extern "C" __global__ void vit_pos_embed_interp_f16_kernel(
    __half* __restrict__ hidden_states,        // [seq, hidden] in-place
    const __half* __restrict__ pos_table,      // [num_grid*num_grid, hidden]
    int grid_h,
    int grid_w,
    int num_grid,
    int m_size,
    int hidden
) {
    const int idx = blockIdx.x;
    if (idx >= grid_h * grid_w) return;

    const int merged_w = grid_w / m_size;
    const int per_block = m_size * m_size;
    const int bh = idx / (merged_w * per_block);
    const int rem = idx - bh * (merged_w * per_block);
    const int bw = rem / per_block;
    const int intra = rem - bw * per_block;
    const int ih = intra / m_size;
    const int iw = intra - ih * m_size;
    const int h = bh * m_size + ih;
    const int w = bw * m_size + iw;

    // linspace(0, num_grid-1, grid_h):  h_idx = h * (num_grid-1) / (grid_h-1)
    // (when grid_h == 1, torch.linspace returns just 0; guard.)
    float h_idx = (grid_h > 1) ? ((float)h * (float)(num_grid - 1) / (float)(grid_h - 1)) : 0.0f;
    float w_idx = (grid_w > 1) ? ((float)w * (float)(num_grid - 1) / (float)(grid_w - 1)) : 0.0f;
    int h_floor = (int)h_idx;
    int w_floor = (int)w_idx;
    int h_ceil = h_floor + 1; if (h_ceil > num_grid - 1) h_ceil = num_grid - 1;
    int w_ceil = w_floor + 1; if (w_ceil > num_grid - 1) w_ceil = num_grid - 1;
    float dh = h_idx - (float)h_floor;
    float dw = w_idx - (float)w_floor;

    float w11 = dh * dw;
    float w10 = dh - w11;            // dh * (1-dw)
    float w01 = dw - w11;            // (1-dh) * dw
    float w00 = 1.0f - dh - w01;     // (1-dh) * (1-dw)

    const int i00 = h_floor * num_grid + w_floor;
    const int i01 = h_floor * num_grid + w_ceil;
    const int i10 = h_ceil  * num_grid + w_floor;
    const int i11 = h_ceil  * num_grid + w_ceil;

    __half* dst = hidden_states + (long long)idx * hidden;
    const __half* p00 = pos_table + (long long)i00 * hidden;
    const __half* p01 = pos_table + (long long)i01 * hidden;
    const __half* p10 = pos_table + (long long)i10 * hidden;
    const __half* p11 = pos_table + (long long)i11 * hidden;

    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = __half2float(p00[c]) * w00
                + __half2float(p01[c]) * w01
                + __half2float(p10[c]) * w10
                + __half2float(p11[c]) * w11;
        dst[c] = __float2half(__half2float(dst[c]) + v);
    }
}
