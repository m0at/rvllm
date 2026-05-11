// Per-(token, 128-K-block) FP8 E4M3 quantisation of an f16 [M, K]
// input. Produces:
//   * `output_fp8`    [M, K]              e4m3 storage, row-major
//   * `output_scales` [M, ceil(K/128)]    f32, row-major
//
// Layout matches cuBLASLt's
// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F` (mode 4) so the
// activation scales pair correctly with weight scales in mode
// `BLK128x128_32F` (mode 5) when the weight ships its
// `[N/128, K/128]` row-major blockscale untouched.
//
// Used by `Qwen36Bringup::fp8_proj_dispatch` on the m≥2 (batched-
// prefill) path. The m=1 path keeps using the existing
// `Fp8GemvF16InLaunch` kernel which consumes f16 input directly —
// for one row the GEMV is faster than quantize + tensor-core GEMM.
//
// Launch:
//   Grid:  (ceil(K/128), M, 1)   — one block per (row, K-block)
//   Block: (128, 1, 1)            — one thread per K element in block
//   Shared: 32 * sizeof(float)    — warp-amax intermediates
//
// Each block reads 128 f16 elements, finds amax via warp-reduction,
// writes 128 fp8 + 1 f32 scale. K need NOT be a multiple of 128 —
// the last K-block clamps to `K - kb*128` valid elements; out-of-
// range threads write zero fp8 (consistent with cuBLASLt's "pad
// with quantised zero").

#include <cuda_fp16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float warp_reduce_max_q(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

extern "C" __global__ void __launch_bounds__(128)
fp8_quantize_per_token_f16_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ input_f16,
    int K
) {
    const int kb  = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int k_blocks = (K + 127) / 128;

    const int k_start = kb * 128;
    const int k = k_start + tid;
    const bool valid = k < K;

    const __half*       in_row  = input_f16  + (long long)row * K;
    __nv_fp8_storage_t* out_row = output_fp8 + (long long)row * K;

    // Pass 1: find amax over this 128-block. Threads outside the
    // valid range contribute 0.
    float v = valid ? fabsf(__half2float(in_row[k])) : 0.0f;
    // 128 threads = 4 warps; reduce in two stages.
    float warp_amax = warp_reduce_max_q(v);
    __shared__ float warp_max[4];
    int wid = tid >> 5;
    int lid = tid & 31;
    if (lid == 0) warp_max[wid] = warp_amax;
    __syncthreads();
    float amax = (lid < 4) ? warp_max[lid] : 0.0f;
    if (wid == 0) amax = warp_reduce_max_q(amax);
    __shared__ float blk_amax;
    if (tid == 0) blk_amax = amax;
    __syncthreads();
    amax = blk_amax;

    const float E4M3_MAX = 448.0f;
    float scale     = (amax > 0.0f) ? (amax / E4M3_MAX) : 1.0f;
    float inv_scale = (amax > 0.0f) ? (E4M3_MAX / amax) : 0.0f;

    if (tid == 0) {
        // VEC128_32F layout: scales[row, kb] in row-major.
        output_scales[(long long)row * k_blocks + kb] = scale;
    }

    // Pass 2: quantise this 128-block. Out-of-range threads write
    // a quantised zero (the un-padded weight column for this K-tail
    // is also zero-padded in cuBLASLt, so the partial-product is
    // 0 — no leakage into the matmul result).
    if (valid) {
        float vq = __half2float(in_row[k]) * inv_scale;
        out_row[k] = __nv_cvt_float_to_fp8(vq, __NV_SATFINITE, __NV_E4M3);
    } else {
        out_row[k] = __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
    }
}
