// Per-token-amax FP8 E4M3 quantisation of an f16 [M, K] input.
// Sibling of `fp8_quantize_per_token_f16.cu` — that one emits
// per-(token, K-block) scales for cuBLASLt's VEC128_32F mode;
// this one emits one scale per row for CUTLASS SM120's
// `prep_sfa` (which max-reduces per-128-row chunks and
// replicates across K/128 tiles internally).
//
// Output:
//   * `output_fp8`    [M, K]  e4m3 storage, row-major
//   * `output_scales` [M]      f32 per-token amax/448
//
// Used by `Qwen36Bringup::fp8_proj_dispatch` on the m≥128 branch
// (the CUTLASS SM120 fast path on sm_121).
//
// Launch:
//   Grid:  (M, 1, 1)
//   Block: (min(K, 1024), 1, 1)
//   Shared: WARPS_MAX * sizeof(float)

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_max_qa(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max_qa(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_max_qa(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_max_qa(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
fp8_quantize_per_token_amax_f16_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ input_f16,
    int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __half*       in_row  = input_f16  + (long long)row * K;
    __nv_fp8_storage_t* out_row = output_fp8 + (long long)row * K;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: per-row amax.
    float local_max = 0.0f;
    for (int k = tid; k < K; k += stride) {
        float v = fabsf(__half2float(in_row[k]));
        if (v > local_max) local_max = v;
    }
    float amax = block_reduce_max_qa(local_max, smem);
    if (tid == 0) smem[0] = amax;
    __syncthreads();
    amax = smem[0];

    const float E4M3_MAX = 448.0f;
    float scale     = (amax > 0.0f) ? (amax / E4M3_MAX) : 1.0f;
    float inv_scale = (amax > 0.0f) ? (E4M3_MAX / amax) : 0.0f;

    if (tid == 0) output_scales[row] = scale;

    // Pass 2: quantise.
    for (int k = tid; k < K; k += stride) {
        float v = __half2float(in_row[k]) * inv_scale;
        out_row[k] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}
