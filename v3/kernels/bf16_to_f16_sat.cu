// bf16 -> f16 conversion with saturation clamp to f16 range.
//
// NB: no __restrict__ on dst/src so the kernel is safe to call with
// dst == src (in-place dtype reinterpretation, e.g. at LM-head input
// after a bf16 residual chain). Each thread reads its own index then
// writes the same index, no cross-thread aliasing hazard.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __launch_bounds__(1024) __global__ void bf16_to_f16_sat_kernel(
    __half* dst,
    const __nv_bfloat16* src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(src[idx]);
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    dst[idx] = __float2half(v);
}
