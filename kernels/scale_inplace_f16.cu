// Pointwise scalar scale: x[i] = x[i] * scale
//
// Used for ViT attention: applying 1/sqrt(head_dim) scale to QK^T scores
// before softmax. Operates in place on an f16 array.

#include <cuda_fp16.h>

extern "C" __global__ void scale_inplace_f16_kernel(
    __half* __restrict__ x,
    float scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__half2float(x[i]) * scale);
}
