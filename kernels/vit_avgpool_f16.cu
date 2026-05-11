// 2D average pooling for Gemma 4 vision tower.
//
// Input:  [num_patches_h, num_patches_w, hidden] f16, contiguous in
//         row-major (h, w, d).
// Pool:   stride = kernel_size, no padding (drops the trailing rows/
//         cols that don't fit a full kernel — Gemma's resize already
//         guarantees divisibility by `pooling_kernel_size`).
// Output: [out_h, out_w, hidden] f16 where out_h = num_h / kernel,
//         out_w = num_w / kernel.
//
// Each block handles one output (oh, ow) position. Threads cooperate
// over the hidden dim.
//
// Launch:
//   Grid:  (out_h * out_w, 1, 1)
//   Block: (min(hidden, 1024), 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void vit_avgpool_f16_kernel(
    __half* __restrict__ output,        // [out_h, out_w, hidden]
    const __half* __restrict__ input,   // [num_h, num_w, hidden]
    int num_h,
    int num_w,
    int hidden,
    int kernel_size
) {
    const int out_w = num_w / kernel_size;
    const int oh = blockIdx.x / out_w;
    const int ow = blockIdx.x % out_w;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const float inv = 1.0f / (float)(kernel_size * kernel_size);

    for (int d = tid; d < hidden; d += stride) {
        float acc = 0.0f;
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = oh * kernel_size + kh;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = ow * kernel_size + kw;
                long long src = ((long long)ih * num_w + iw) * hidden + d;
                acc += __half2float(input[src]);
            }
        }
        long long dst = ((long long)oh * out_w + ow) * hidden + d;
        output[dst] = __float2half(acc * inv);
    }
}
