// Copy a contiguous column slice from a row-major f16 matrix:
// out[row, col] = in[row, col_offset + col]
#include <cuda_fp16.h>

extern "C" __global__ void slice_cols_f16_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int in_cols,
    int col_offset,
    int out_cols
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_cols) return;

    int out_idx = row * out_cols + col;
    int in_idx = row * in_cols + col_offset + col;
    out[out_idx] = in[in_idx];
}
