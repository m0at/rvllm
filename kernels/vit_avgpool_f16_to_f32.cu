// f16-input, f32-output sibling of vit_avgpool_f16. Used by Gemma 4
// vision pooler so the subsequent `*= sqrt(hidden_size)` and the
// `(x - std_bias) * std_scale` standardize step can run in f32 — the
// f16 path overflowed (31/256 rows hit f16 ±65504 inf) on
// 640×488-class inputs because peak post-encoder activations × 33.94
// (sqrt(1152)) exceed f16 range. HF stays bf16; we keep f32 only
// across the pooler→standardize span and narrow back to f16 at the
// end.
//
// Input:  [num_h, num_w, hidden] f16
// Output: [out_h, out_w, hidden] f32  where out_* = num_* / kernel
// Same average over a kernel×kernel tile as the f16 variant.
//
// Launch:
//   Grid:  (out_h * out_w, 1, 1)
//   Block: (min(hidden, 1024), 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void vit_avgpool_f16_to_f32_kernel(
    float* __restrict__ output,            // [out_h, out_w, hidden] f32
    const __half* __restrict__ input,      // [num_h, num_w, hidden] f16
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
        output[dst] = acc * inv;
    }
}
