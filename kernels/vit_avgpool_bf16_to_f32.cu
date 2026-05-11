// bf16-input, f32-output average pool — Gemma 4 vision pooler in the
// bf16 vision path. Output f32 so the subsequent
// `*= sqrt(hidden)` and `(x - bias) * scale` don't lose precision.

#include <cuda_bf16.h>

extern "C" __global__ void vit_avgpool_bf16_to_f32_kernel(
    float* __restrict__ output,                 // [out_h, out_w, hidden] f32
    const __nv_bfloat16* __restrict__ input,    // [num_h, num_w, hidden] bf16
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
                acc += __bfloat162float(input[src]);
            }
        }
        long long dst = ((long long)oh * out_w + ow) * hidden + d;
        output[dst] = acc * inv;
    }
}
