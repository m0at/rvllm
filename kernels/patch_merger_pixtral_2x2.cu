// Pixtral 2x2 spatial patch merger (BF16, contiguous in/out).
//
// Pixtral's projector reduces the (grid_h, grid_w) ViT output by
// 2x by concatenating each 2x2 spatial neighbourhood into one
// merged token of width `4 * hidden`. Mirrors HF Mistral's
// `Mistral3PatchMerger`:
//
//   in:  [grid_h, grid_w, hidden]            BF16
//   out: [grid_h/2, grid_w/2, 4 * hidden]   BF16
//
// Concat order along the merged hidden axis is row-major in the 2x2
// neighbourhood (top-left, top-right, bottom-left, bottom-right):
//
//   out[mh, mw, 0       .. 1H] = in[2*mh,   2*mw,   :]
//   out[mh, mw, 1H      .. 2H] = in[2*mh,   2*mw+1, :]
//   out[mh, mw, 2H      .. 3H] = in[2*mh+1, 2*mw,   :]
//   out[mh, mw, 3H      .. 4H] = in[2*mh+1, 2*mw+1, :]
//
// where H = hidden. This is the layout HF emits via:
//   x.unfold(1, 2, 2).unfold(2, 2, 2).flatten(-3, -1)
// with axis order (h, w) outer, (sub_h, sub_w, c) inner.
//
// Launch:
//   Grid:  (merged_tokens = (grid_h/2) * (grid_w/2), 1, 1)
//   Block: (256, 1, 1) — strided over the 4*H output channels.
//
// Pure permutation + concat, no math. Each thread copies a packed
// uint4 (8 bf16) per stride to keep memory bandwidth saturated.
//
// Round-12 (Pixtral vision phase 1).

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

    // Source row-major offsets of the four 2x2 neighbours (in tokens).
    const long long in_stride_row = (long long)grid_w * hidden;
    const long long base_in_tl = ((long long)(2 * mh)     * grid_w + (2 * mw))     * hidden;
    const long long base_in_tr = base_in_tl + hidden;
    const long long base_in_bl = base_in_tl + in_stride_row;
    const long long base_in_br = base_in_bl + hidden;

    const long long base_out = (long long)merged_idx * (4 * (long long)hidden);
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    // 8 bf16 = 16 bytes per uint4 lane. Vectorise when hidden is a
    // multiple of 8 (true for Pixtral's hidden=1664 and any sane
    // divisor); fall through to scalar otherwise.
    const int H = hidden;
    if ((H & 7) == 0) {
        const int4* in_tl = reinterpret_cast<const int4*>(in + base_in_tl);
        const int4* in_tr = reinterpret_cast<const int4*>(in + base_in_tr);
        const int4* in_bl = reinterpret_cast<const int4*>(in + base_in_bl);
        const int4* in_br = reinterpret_cast<const int4*>(in + base_in_br);
        int4* out_tl = reinterpret_cast<int4*>(out + base_out);
        int4* out_tr = reinterpret_cast<int4*>(out + base_out + H);
        int4* out_bl = reinterpret_cast<int4*>(out + base_out + 2 * H);
        int4* out_br = reinterpret_cast<int4*>(out + base_out + 3 * H);
        const int H8 = H >> 3;
        for (int j = tid; j < H8; j += stride) {
            out_tl[j] = in_tl[j];
            out_tr[j] = in_tr[j];
            out_bl[j] = in_bl[j];
            out_br[j] = in_br[j];
        }
    } else {
        for (int j = tid; j < H; j += stride) {
            out[base_out + 0*H + j] = in[base_in_tl + j];
            out[base_out + 1*H + j] = in[base_in_tr + j];
            out[base_out + 2*H + j] = in[base_in_bl + j];
            out[base_out + 3*H + j] = in[base_in_br + j];
        }
    }
}
