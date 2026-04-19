// Fused: f32->bf16 + rmsnorm + add-to-residual(f16)
//
// Replaces 3 kernels per residual update (2x per layer = 120x total):
//   f32_to_bf16(gemm_out -> delta)
//   rmsnorm_inplace_bf16(delta, gamma)
//   vector_add_bf16_to_f16(residual += delta)
//
// Grid: (num_tokens), Block: (min(hidden, 1024))

#include <cuda_fp16.h>

static __device__ __forceinline__ float bf16_round(float v) {
    v = fminf(fmaxf(v, -3.3895314e+38f), 3.3895314e+38f);
    unsigned int bits;
    memcpy(&bits, &v, 4);
    // Round to nearest even (IEEE 754)
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    bits &= 0xFFFF0000u;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

extern "C" __global__ void fused_norm_add_residual_kernel(
    const float* __restrict__ gemm_out,
    const half*  __restrict__ gamma,
    half*        __restrict__ residual,
    int hidden,
    float eps
) {
    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row = gemm_out + (size_t)token * hidden;
    half* res = residual + (size_t)token * hidden;

    // Pass 1: sum of squares (bf16-truncated values)
    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = bf16_round(row[i]);
        local_ss += v * v;
    }

    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float smem[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) smem[warp_id] = local_ss;
    __syncthreads();

    float total_ss;
    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        total_ss = 0.0f;
        for (int w = 0; w < nw; w++) total_ss += smem[w];
        smem[0] = total_ss;
    }
    __syncthreads();
    float rms_inv = rsqrtf(smem[0] / (float)hidden + eps);

    // Pass 2: normalize, scale by gamma, add to residual
    for (int i = tid; i < hidden; i += stride) {
        float v = bf16_round(row[i]);
        float normed = v * rms_inv * __half2float(gamma[i]);
        float r = __half2float(res[i]) + normed;
        res[i] = __float2half(r);
    }
}
