// Half-precision RMSNorm kernel with vectorized float4 loads/stores.
// Loads 8 half elements at once via float4 (16 bytes), accumulates in f32.
// Uses warp shuffle reduction for minimal shared memory usage.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size/8, 1024), 1, 1)
//   Shared memory: (blockDim.x / 32 + 1) * sizeof(float)
//
// Requires hidden_size to be a multiple of 8.

#include <cuda_fp16.h>

extern "C"
__global__ void rms_norm_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    // Each float4 covers 8 halves (16 bytes)
    const int vec_size = hidden_size / 8;

    const float4* x = reinterpret_cast<const float4*>(input + token_idx * hidden_size);
    float4* y = reinterpret_cast<float4*>(output + token_idx * hidden_size);
    const float4* w = reinterpret_cast<const float4*>(weight);

    // Pass 1: sum of squares in f32 with vectorized loads
    float local_ss = 0.0f;
    for (int i = tid; i < vec_size; i += stride) {
        float4 raw = x[i];
        // Reinterpret as 4 half2 pairs
        __half2* h2 = reinterpret_cast<__half2*>(&raw);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(h2[j]);
            local_ss += f2.x * f2.x + f2.y * f2.y;
        }
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
    }

    extern __shared__ float sdata[];
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (lane == 0) {
        sdata[warp_id] = local_ss;
    }
    __syncthreads();

    const int num_warps = (stride + 31) >> 5;
    float sum = (tid < num_warps) ? sdata[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        sdata[0] = rsqrtf(sum / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms_scale = sdata[0];

    // Pass 2: normalize and scale with vectorized loads/stores
    for (int i = tid; i < vec_size; i += stride) {
        float4 raw_x = x[i];
        float4 raw_w = w[i];
        __half2* hx = reinterpret_cast<__half2*>(&raw_x);
        __half2* hw = reinterpret_cast<__half2*>(&raw_w);
        float4 result;
        __half2* hr = reinterpret_cast<__half2*>(&result);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 fx = __half22float2(hx[j]);
            float2 fw = __half22float2(hw[j]);
            float2 fo;
            fo.x = fx.x * fw.x * rms_scale;
            fo.y = fx.y * fw.y * rms_scale;
            hr[j] = __float22half2_rn(fo);
        }
        y[i] = result;
    }
}

// Fused residual add + RMS norm, f16 variant with vectorized loads/stores.
extern "C"
__global__ void fused_residual_rmsnorm_f16_kernel(
    __half* __restrict__ output,
    __half* __restrict__ residual,
    const __half* __restrict__ input,
    const __half* __restrict__ add,
    const __half* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = token_idx * hidden_size;
    const int vec_size = hidden_size / 8;

    const float4* x_in = reinterpret_cast<const float4*>(input + row_offset);
    const float4* x_add = reinterpret_cast<const float4*>(add + row_offset);
    float4* res = reinterpret_cast<float4*>(residual + row_offset);
    float4* y = reinterpret_cast<float4*>(output + row_offset);
    const float4* w = reinterpret_cast<const float4*>(weight);

    extern __shared__ float sdata[];

    // Pass 1: residual add + sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < vec_size; i += stride) {
        float4 raw_in = x_in[i];
        float4 raw_add = x_add[i];
        __half2* h_in = reinterpret_cast<__half2*>(&raw_in);
        __half2* h_add = reinterpret_cast<__half2*>(&raw_add);
        float4 result;
        __half2* h_res = reinterpret_cast<__half2*>(&result);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 fi = __half22float2(h_in[j]);
            float2 fa = __half22float2(h_add[j]);
            float2 fr;
            fr.x = fi.x + fa.x;
            fr.y = fi.y + fa.y;
            local_ss += fr.x * fr.x + fr.y * fr.y;
            h_res[j] = __float22half2_rn(fr);
        }
        res[i] = result;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
    }

    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (lane == 0) {
        sdata[warp_id] = local_ss;
    }
    __syncthreads();

    const int num_warps = (stride + 31) >> 5;
    float sum = (tid < num_warps) ? sdata[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        sdata[0] = rsqrtf(sum / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms_scale = sdata[0];

    // Pass 2: normalize with vectorized loads/stores
    for (int i = tid; i < vec_size; i += stride) {
        float4 raw_r = res[i];
        float4 raw_w = w[i];
        __half2* hr = reinterpret_cast<__half2*>(&raw_r);
        __half2* hw = reinterpret_cast<__half2*>(&raw_w);
        float4 result;
        __half2* ho = reinterpret_cast<__half2*>(&result);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 fr = __half22float2(hr[j]);
            float2 fw = __half22float2(hw[j]);
            float2 fo;
            fo.x = fr.x * fw.x * rms_scale;
            fo.y = fr.y * fw.y * rms_scale;
            ho[j] = __float22half2_rn(fo);
        }
        y[i] = result;
    }
}
