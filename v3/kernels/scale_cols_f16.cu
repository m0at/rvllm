// Per-column scaling on F16 data: data[m,n] *= scale[n]
// In-place. scale is f32.
// Grid: (ceil(n/256), m), Block: (256)

#include <cuda_fp16.h>

extern "C" __global__ void scale_cols_f16_kernel(
    __half* __restrict__ data,
    const float* __restrict__ scale,
    int m,
    int n
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= n) return;
    int idx = row * n + col;
    float v = __half2float(data[idx]) * scale[col];
    data[idx] = __float2half(v);
}
