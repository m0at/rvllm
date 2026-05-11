// Per-channel standardize, f32-in / bf16-out, with bf16 bias+scale
// weights. Used by the bf16 vision path: pooler stays in f32 across
// avg-pool → sqrt-scale → standardize, then we narrow into bf16 for
// the MultimodalEmbedder's parameter-free RMSNorm + projection. The
// bf16 (not f16) narrowing preserves another ~3 stops of dynamic
// range vs the f16 path, matching HF's bf16-throughout behaviour.

#include <cuda_bf16.h>

extern "C" __global__ void vit_standardize_f32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ scale,
    int hidden
) {
    const int t = blockIdx.x;
    __nv_bfloat16* dst = out + (long long)t * hidden;
    const float* row = x + (long long)t * hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = (row[c] - __bfloat162float(bias[c])) * __bfloat162float(scale[c]);
        dst[c] = __float2bfloat16(v);
    }
}
