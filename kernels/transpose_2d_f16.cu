// Out-of-place 2D transpose for f16: out[r, c] = in[c, r].
//   in:  [rows, cols] f16
//   out: [cols, rows] f16
//
// Launch:
//   Grid: (ceil(cols/16), ceil(rows/16), 1)
//   Block: (16, 16, 1)

#include <cuda_fp16.h>

extern "C" __global__ void transpose_2d_f16_kernel(
    __half* __restrict__ out,           // [cols, rows]
    const __half* __restrict__ in,      // [rows, cols]
    int rows,
    int cols
) {
    int c = blockIdx.x * 16 + threadIdx.x;
    int r = blockIdx.y * 16 + threadIdx.y;
    if (r >= rows || c >= cols) return;
    out[(long long)c * rows + r] = in[(long long)r * cols + c];
}
