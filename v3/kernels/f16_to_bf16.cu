// f16 → bf16 conversion. Used at the Stage 1 BF16 residual chain
// entry: embedding gather writes f16, then we widen the exponent to
// bf16 storage so subsequent layer adds preserve dynamic range.
//
// Mantissa narrows (f16=10b → bf16=7b) but exponent widens
// (f16=5b → bf16=8b), so values within f16 range round-trip through
// bf16 cleanly except for sub-mantissa precision. The reverse
// (bf16→f16) is the saturating cast in `bf16_to_f16_sat.cu`.
//
// Grid: (ceil(n / blockDim.x), ), Block: 256 typical.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void __launch_bounds__(1024)
f16_to_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __half*  __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(__half2float(src[idx]));
    }
}
