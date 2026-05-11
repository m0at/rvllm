// Pixtral 2x2 spatial patch merger (BF16, channel-outer / HF-compatible).
//
// Mirrors HF Mistral3PatchMerger which uses
// `torch.nn.functional.unfold` with kernel_size=2, stride=2 on
// NCHW input [1, hidden, h, w], then `.view(d*4, -1).t()` to
// produce a [merged_n, 4*hidden] tensor with the inner axis in
// the unfold's natural order:
//
//   out[mh, mw, c*4 + 0] = in[2*mh,   2*mw,   c]   // top-left
//   out[mh, mw, c*4 + 1] = in[2*mh,   2*mw+1, c]   // top-right
//   out[mh, mw, c*4 + 2] = in[2*mh+1, 2*mw,   c]   // bottom-left
//   out[mh, mw, c*4 + 3] = in[2*mh+1, 2*mw+1, c]   // bottom-right
//
// (channel-OUTER, position-INNER per group of 4). The pre-Round-12
// version of this kernel was HWC token-major
// `[TL_c0..H, TR_c0..H, BL_c0..H, BR_c0..H]` which corresponded to
// our preprocess-side layout but did NOT match HF's merging_layer
// weight layout — the Linear after this kernel would multiply the
// wrong columns, producing cos≈0 against the HF reference.
//
// Round-12 phase 3-test (c) fix: switch to channel-outer to match
// HF's unfold layout. Now agrees byte-for-byte with HF's patch
// merger output (verified via cosine gate).
//
// Layout:
//   in:  [grid_h, grid_w, hidden]            BF16  (kept HWC token-major)
//   out: [grid_h/2, grid_w/2, 4 * hidden]    BF16  (channel-outer inner)
//
// Launch:
//   Grid:  (merged_tokens = (grid_h/2) * (grid_w/2), 1, 1)
//   Block: (256, 1, 1) — strided over the 4*H output channels.

#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__ void patch_merger_pixtral_2x2_kernel(
    __nv_bfloat16* __restrict__ out,         // [merged_h, merged_w, 4*H]
    const __nv_bfloat16* __restrict__ in,    // [grid_h,   grid_w,   H]
    int grid_h,
    int grid_w,
    int hidden
) {
    const int merged_w = grid_w >> 1;
    const int merged_idx = blockIdx.x;
    const int mh = merged_idx / merged_w;
    const int mw = merged_idx - mh * merged_w;

    // Source (HWC) token offsets for the 4 neighbours.
    const long long src_stride_row = (long long)grid_w * hidden;
    const long long base_in_tl = ((long long)(2*mh)   * grid_w + (2*mw))   * hidden;
    const long long base_in_tr = base_in_tl + hidden;
    const long long base_in_bl = base_in_tl + src_stride_row;
    const long long base_in_br = base_in_bl + hidden;

    const long long base_out = (long long)merged_idx * (4 * (long long)hidden);
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;
    const int H = hidden;

    // For each input-channel c, write the 4 neighbour samples
    // contiguously at out[base + c*4 .. c*4 + 4].
    for (int c = tid; c < H; c += stride) {
        out[base_out + c*4 + 0] = in[base_in_tl + c];
        out[base_out + c*4 + 1] = in[base_in_tr + c];
        out[base_out + c*4 + 2] = in[base_in_bl + c];
        out[base_out + c*4 + 3] = in[base_in_br + c];
    }
}
