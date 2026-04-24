// In-place row correction for f16 data: data[m,n] *= scale[m] / scale[0].

#include <cuda_fp16.h>

extern "C" __global__ void scale_rows_f16_kernel(
    __half* __restrict__ data,
    const float* __restrict__ scale,
    int m,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;
    float base = scale[0];
    if (base == 0.0f) return;
    int row = idx / n;
    float v = __half2float(data[idx]) * (scale[row] / base);
    data[idx] = __float2half(v);
}
