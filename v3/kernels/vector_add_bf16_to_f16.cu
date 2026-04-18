// dst_f16[i] += bf16_to_f32(src_bf16[i]) for vectors of length n.
// Reads bf16 source, converts through f32, adds to f16 destination.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void __launch_bounds__(1024)
vector_add_bf16_to_f16_kernel(
    __half* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float d = __half2float(dst[idx]);
        float s = __bfloat162float(src[idx]);
        dst[idx] = __float2half(d + s);
    }
}
