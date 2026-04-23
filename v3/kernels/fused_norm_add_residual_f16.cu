// Fused: channelscale + rmsnorm + add-to-residual(f16) + optional layer_scalar
//
// Grid: (num_tokens), Block: (min(hidden, 1024))
// Shared memory: hidden * sizeof(float)

#include <cuda_fp16.h>

extern "C" __global__ void fused_norm_add_residual_f16_kernel(
    const float* __restrict__ gemm_out,
    const float* __restrict__ channelscale,
    const half*  __restrict__ gamma,
    half*        __restrict__ residual,
    const half*  __restrict__ layer_scalar,
    int hidden,
    float eps
) {
    extern __shared__ float svals[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row = gemm_out + (size_t)token * hidden;
    half* res = residual + (size_t)token * hidden;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = row[i] * channelscale[i];
        svals[i] = v;
        local_ss += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float warp_ss[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_ss[warp_id] = local_ss;
    __syncthreads();

    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += warp_ss[w];
        warp_ss[0] = total;
    }
    __syncthreads();
    float rms_inv = rsqrtf(warp_ss[0] / (float)hidden + eps);

    float ls = layer_scalar ? __half2float(*layer_scalar) : 1.0f;
    for (int i = tid; i < hidden; i += stride) {
        float normed = svals[i] * rms_inv * __half2float(gamma[i]);
        float r = __half2float(res[i]) + normed;
        res[i] = __float2half(r * ls);
    }
}

// Variant: reads f16 input (not f32), skips channelscale. Used when the
// preceding GEMM already baked the per-channel scale into its output —
// e.g. the Sm121 `fp8_gemv_blockwise_wpr_native_f16in_kernel` path.
// Identical rmsnorm + residual-add math as the base kernel above.
//
// Grid: (num_tokens), Block: (min(hidden, 1024))
// Shared memory: hidden * sizeof(float)
extern "C" __global__ void fused_norm_add_residual_f16in_kernel(
    const half*  __restrict__ gemm_out,
    const half*  __restrict__ gamma,
    half*        __restrict__ residual,
    const half*  __restrict__ layer_scalar,
    int hidden,
    float eps
) {
    extern __shared__ float svals[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const half* row = gemm_out + (size_t)token * hidden;
    half* res = residual + (size_t)token * hidden;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = __half2float(row[i]);
        svals[i] = v;
        local_ss += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float warp_ss[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_ss[warp_id] = local_ss;
    __syncthreads();

    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += warp_ss[w];
        warp_ss[0] = total;
    }
    __syncthreads();
    float rms_inv = rsqrtf(warp_ss[0] / (float)hidden + eps);

    float ls = layer_scalar ? __half2float(*layer_scalar) : 1.0f;
    for (int i = tid; i < hidden; i += stride) {
        float normed = svals[i] * rms_inv * __half2float(gamma[i]);
        float r = __half2float(res[i]) + normed;
        res[i] = __float2half(r * ls);
    }
}
