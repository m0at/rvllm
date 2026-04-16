// Fused RMSNorm + per-token FP8 E4M3 quantization kernels for SM90.
// All kernels: 1 block per token, warp-shuffle reductions.
// Compile: nvcc -ptx -arch=sm_90 -O3 --use_fast_math

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define FP8_E4M3_MAX 448.0f
#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

// Kernel A: RMSNorm + per-token FP8 E4M3 quantization.
// grid=(num_tokens), block=(min(hidden_size, 1024))
// shared mem: WARPS_MAX * sizeof(float)
extern "C" __global__ void __launch_bounds__(1024)
fused_rmsnorm_fp8_quant_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ input,
    const __half*       __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    __syncthreads();

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    // Pass 2: compute normed values, find absmax
    float local_max = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]) * rms * __half2float(weight[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float absmax = block_reduce_max(local_max, smem);
    __syncthreads();

    // Compute scale
    float scale = absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (tid == 0) output_scales[row] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 3: quantize to FP8
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]) * rms * __half2float(weight[i]);
        output_fp8[row_offset + i] = __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

// Kernel B: Residual add + RMSNorm + per-token FP8 E4M3 quantization.
// grid=(num_tokens), block=(min(hidden_size, 1024))
// shared mem: WARPS_MAX * sizeof(float)
extern "C" __global__ void __launch_bounds__(1024)
fused_add_rmsnorm_fp8_quant_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    __half*             __restrict__ residual_out,
    const __half*       __restrict__ input,
    const __half*       __restrict__ residual,
    const __half*       __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: residual add + sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]) + __half2float(residual[row_offset + i]);
        residual_out[row_offset + i] = __float2half(v);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    __syncthreads();

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    // Pass 2: compute normed values, find absmax
    float local_max = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(residual_out[row_offset + i]) * rms * __half2float(weight[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float absmax = block_reduce_max(local_max, smem);
    __syncthreads();

    float scale = absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (tid == 0) output_scales[row] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 3: quantize to FP8
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(residual_out[row_offset + i]) * rms * __half2float(weight[i]);
        output_fp8[row_offset + i] = __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

// Kernel C: Plain per-token FP8 E4M3 quantization (no norm).
// grid=(num_tokens), block=(min(dim/8, 1024))
// shared mem: WARPS_MAX * sizeof(float)
//
// Vectorized: uint4 128-bit loads (8 halves), uint2 64-bit FP8 stores (8 FP8),
// register-cached between absmax and quantize passes so each HBM element is
// read exactly once (HBM traffic halved vs the scalar 2-pass version).
// Requires dim % 8 == 0 (all LLM hidden/head dimensions satisfy this).
//
// Up to MAX_VEC_PER_THREAD uint4s per thread held in registers. For block=1024,
// this supports dim up to 8*8*1024 = 65536 halves.
#define MAX_VEC_PER_THREAD 8

extern "C" __global__ void __launch_bounds__(1024)
quantize_fp8_per_token_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ input,
    int dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * dim;
    const int vec_per_row = dim >> 3; // dim / 8

    __shared__ float smem[WARPS_MAX];

    // Register cache: one uint4 (=4 __half2) per iteration of the strided loop.
    __half2 cache[MAX_VEC_PER_THREAD * 4];

    // Pass 1: vectorized load + absmax, cache halves in registers.
    float local_max = 0.0f;
    int slot = 0;
    const uint4* vin = reinterpret_cast<const uint4*>(input + row_offset);
    for (int i = tid; i < vec_per_row; i += stride) {
        uint4 raw = vin[i];
        __half2 h01 = reinterpret_cast<__half2&>(raw.x);
        __half2 h23 = reinterpret_cast<__half2&>(raw.y);
        __half2 h45 = reinterpret_cast<__half2&>(raw.z);
        __half2 h67 = reinterpret_cast<__half2&>(raw.w);
        cache[slot * 4 + 0] = h01;
        cache[slot * 4 + 1] = h23;
        cache[slot * 4 + 2] = h45;
        cache[slot * 4 + 3] = h67;
        slot++;

        float2 f01 = __half22float2(h01);
        float2 f23 = __half22float2(h23);
        float2 f45 = __half22float2(h45);
        float2 f67 = __half22float2(h67);
        local_max = fmaxf(local_max, fmaxf(fabsf(f01.x), fabsf(f01.y)));
        local_max = fmaxf(local_max, fmaxf(fabsf(f23.x), fabsf(f23.y)));
        local_max = fmaxf(local_max, fmaxf(fabsf(f45.x), fabsf(f45.y)));
        local_max = fmaxf(local_max, fmaxf(fabsf(f67.x), fabsf(f67.y)));
    }
    float absmax = block_reduce_max(local_max, smem);
    __syncthreads();

    float scale = absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (tid == 0) output_scales[row] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: quantize from register cache, vectorized 64-bit FP8 stores.
    uint2* vout = reinterpret_cast<uint2*>(output_fp8 + row_offset);
    slot = 0;
    for (int i = tid; i < vec_per_row; i += stride) {
        float2 f01 = __half22float2(cache[slot * 4 + 0]);
        float2 f23 = __half22float2(cache[slot * 4 + 1]);
        float2 f45 = __half22float2(cache[slot * 4 + 2]);
        float2 f67 = __half22float2(cache[slot * 4 + 3]);
        slot++;

        __nv_fp8_storage_t q0 = __nv_cvt_float_to_fp8(f01.x * inv_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t q1 = __nv_cvt_float_to_fp8(f01.y * inv_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t q2 = __nv_cvt_float_to_fp8(f23.x * inv_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t q3 = __nv_cvt_float_to_fp8(f23.y * inv_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t q4 = __nv_cvt_float_to_fp8(f45.x * inv_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t q5 = __nv_cvt_float_to_fp8(f45.y * inv_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t q6 = __nv_cvt_float_to_fp8(f67.x * inv_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t q7 = __nv_cvt_float_to_fp8(f67.y * inv_scale, __NV_SATFINITE, __NV_E4M3);

        uint32_t lo = uint32_t(q0) | (uint32_t(q1) << 8) | (uint32_t(q2) << 16) | (uint32_t(q3) << 24);
        uint32_t hi = uint32_t(q4) | (uint32_t(q5) << 8) | (uint32_t(q6) << 16) | (uint32_t(q7) << 24);
        vout[i] = make_uint2(lo, hi);
    }
}
