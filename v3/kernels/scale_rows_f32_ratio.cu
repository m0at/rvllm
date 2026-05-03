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

#include <cuda_runtime.h>
#include <math.h>

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
        // Codex44-2: a degenerate / padded row 0 with zero or non-
        // finite scale would propagate NaN/Inf across the whole
        // matrix. Treat that as "scalar correction unavailable" and
        // leave the row as cuBLASLt produced it — the per-row math
        // upstream already absorbed scale[0]==1.0 in the all-zeros
        // path, so a no-op here is the closest correct fallback. A
        // bad scale on m>0 only zeros that row, which is the FP8
        // path's own contract.
        float s0 = scale[0];
        if (s0 > 0.0f && isfinite(s0)) {
            data[idx] *= scale[m] / s0;
        }
    }
}
