// In-place row correction: data[m,n] *= scale[m] / scale[0].
// Used after cuBLASLt FP8 GEMM when activation scales are per row but
// CUBLASLT_MATMUL_DESC_B_SCALE_MODE is unavailable.

extern "C" __global__ void scale_rows_f32_kernel(
    float* __restrict__ data,
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
    data[idx] *= scale[row] / base;
}
