// dst[i] += src[i] for bf16 vectors of length n.
// bf16 sibling of vector_add_f16; used by the bf16 vision residual path
// to keep the encoder accumulator out of f16's ±65504 range and
// preserve more of HF's bf16 dynamic range.

#include <cuda_bf16.h>

extern "C" __global__ void __launch_bounds__(1024)
vector_add_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = __bfloat162float(dst[idx]);
    float b = __bfloat162float(src[idx]);
    dst[idx] = __float2bfloat16(a + b);
}
