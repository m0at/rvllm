// Post-GEMM per-row scale RATIO correction for FP8 GEMM where the
// activation had per-token scales but cuBLASLt was invoked in
// SCALAR B_SCALE mode (the only mode that produces a matmul
// heuristic on Blackwell-consumer / sm_121). In scalar mode cuBLASLt
// reads `scale[0]` and applies it uniformly to all output rows, so
// rows 1..M-1 come out scaled with token 0's scale instead of their
// own. This kernel corrects that by multiplying row m by
// `scale[m] / scale[0]`, which collapses to a no-op for m == 0 and
// restores the intended value `sum_k fp8_a[m,k] * a_scale[m] *
// fp8_b[n,k] * b_scale` for m > 0.

extern "C" __launch_bounds__(256) __global__ void scale_rows_f32_ratio_kernel(
    float* __restrict__ data,        // [M, N] row-major, in-place
    const float* __restrict__ scale, // [M]
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int m = idx / N;
        // scale[0] is token 0's already-applied scale; multiplying by
        // (scale[m] / scale[0]) converts it to token m's scale. At
        // m=0 the ratio is 1.0, so row 0 is untouched.
        float s0 = scale[0];
        data[idx] *= scale[m] / s0;
    }
}
