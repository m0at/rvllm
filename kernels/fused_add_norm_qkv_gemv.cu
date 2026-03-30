// Fused residual-add + RMSNorm + QKV GEMV kernel for M=1 decode.
// f16 I/O, f32 accumulation. Each block handles RPB=8 output rows.
// Phase 2 uses warp-per-row: 8 warps compute 8 rows in parallel.
//
// 4 variants: {add+norm, norm-only} x {no-bias, with-bias}
//
// Launch config:
//   Grid:  ((qkv_dim + RPB - 1) / RPB, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: hidden_size * sizeof(float) + RPB * sizeof(float)

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define THREADS 256
#define RPB 8

__device__ __forceinline__ float warp_reduce_sum_anqg(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// --------------------------------------------------------------------------
// Fused add + RMSNorm + QKV GEMV (layers 1..N where residual add is needed)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h8 = hidden_size / 8;
    const int4* in4 = (const int4*)input;
    const int4* add4 = (const int4*)add_vec;
    #pragma unroll 2
    for (int i = tid; i < h8; i += THREADS) {
        int4 a4 = in4[i]; int4 b4 = add4[i];
        half2 a01=*reinterpret_cast<half2*>(&a4.x), a23=*reinterpret_cast<half2*>(&a4.y);
        half2 a45=*reinterpret_cast<half2*>(&a4.z), a67=*reinterpret_cast<half2*>(&a4.w);
        half2 b01=*reinterpret_cast<half2*>(&b4.x), b23=*reinterpret_cast<half2*>(&b4.y);
        half2 b45=*reinterpret_cast<half2*>(&b4.z), b67=*reinterpret_cast<half2*>(&b4.w);
        int base = i * 8;
        float v0=__half2float(a01.x)+__half2float(b01.x), v1=__half2float(a01.y)+__half2float(b01.y);
        float v2=__half2float(a23.x)+__half2float(b23.x), v3=__half2float(a23.y)+__half2float(b23.y);
        float v4=__half2float(a45.x)+__half2float(b45.x), v5=__half2float(a45.y)+__half2float(b45.y);
        float v6=__half2float(a67.x)+__half2float(b67.x), v7=__half2float(a67.y)+__half2float(b67.y);
        s_normed[base]=v0; s_normed[base+1]=v1; s_normed[base+2]=v2; s_normed[base+3]=v3;
        s_normed[base+4]=v4; s_normed[base+5]=v5; s_normed[base+6]=v6; s_normed[base+7]=v7;
        local_ss += v0*v0+v1*v1+v2*v2+v3*v3+v4*v4+v5*v5+v6*v6+v7*v7;
    }
    for (int i = h8*8+tid; i < hidden_size; i += THREADS) {
        float v = __half2float(input[i]) + __half2float(add_vec[i]);
        s_normed[i] = v; local_ss += v*v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    // Phase 2: GEMV -- warp-per-row, 128-bit vectorized loads (matches Triton v4.b32)
    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            // 128-bit loads: 4 x int32 = 8 x f16 per load (same as Triton's ld.global.v4.b32)
            const int4* w4 = (const int4*)(proj_weight + (long long)row * hidden_size);
            const int h8 = hidden_size / 8;  // number of 128-bit chunks
            float acc = 0.0f;

            #pragma unroll 4
            for (int i = lane_id; i < h8; i += 32) {
                int4 packed = w4[i];
                half2 w01 = *reinterpret_cast<half2*>(&packed.x);
                half2 w23 = *reinterpret_cast<half2*>(&packed.y);
                half2 w45 = *reinterpret_cast<half2*>(&packed.z);
                half2 w67 = *reinterpret_cast<half2*>(&packed.w);
                int base = i * 8;
                acc += __half2float(w01.x) * s_normed[base]
                     + __half2float(w01.y) * s_normed[base + 1]
                     + __half2float(w23.x) * s_normed[base + 2]
                     + __half2float(w23.y) * s_normed[base + 3]
                     + __half2float(w45.x) * s_normed[base + 4]
                     + __half2float(w45.y) * s_normed[base + 5]
                     + __half2float(w67.x) * s_normed[base + 6]
                     + __half2float(w67.y) * s_normed[base + 7];
            }
            // Handle remainder (hidden_size not multiple of 8)
            for (int i = h8 * 8 + lane_id; i < hidden_size; i += 32)
                acc += __half2float(proj_weight[(long long)row * hidden_size + i]) * s_normed[i];

            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// Fused add + RMSNorm + QKV GEMV + bias (models with QKV bias like Qwen2.5)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_bias_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    const __half* __restrict__ bias,        // [qkv_dim]
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h8 = hidden_size / 8;
    const int4* in4 = (const int4*)input;
    const int4* add4 = (const int4*)add_vec;
    #pragma unroll 2
    for (int i = tid; i < h8; i += THREADS) {
        int4 a4 = in4[i]; int4 b4 = add4[i];
        half2 a01=*reinterpret_cast<half2*>(&a4.x), a23=*reinterpret_cast<half2*>(&a4.y);
        half2 a45=*reinterpret_cast<half2*>(&a4.z), a67=*reinterpret_cast<half2*>(&a4.w);
        half2 b01=*reinterpret_cast<half2*>(&b4.x), b23=*reinterpret_cast<half2*>(&b4.y);
        half2 b45=*reinterpret_cast<half2*>(&b4.z), b67=*reinterpret_cast<half2*>(&b4.w);
        int base = i * 8;
        float v0=__half2float(a01.x)+__half2float(b01.x), v1=__half2float(a01.y)+__half2float(b01.y);
        float v2=__half2float(a23.x)+__half2float(b23.x), v3=__half2float(a23.y)+__half2float(b23.y);
        float v4=__half2float(a45.x)+__half2float(b45.x), v5=__half2float(a45.y)+__half2float(b45.y);
        float v6=__half2float(a67.x)+__half2float(b67.x), v7=__half2float(a67.y)+__half2float(b67.y);
        s_normed[base]=v0; s_normed[base+1]=v1; s_normed[base+2]=v2; s_normed[base+3]=v3;
        s_normed[base+4]=v4; s_normed[base+5]=v5; s_normed[base+6]=v6; s_normed[base+7]=v7;
        local_ss += v0*v0+v1*v1+v2*v2+v3*v3+v4*v4+v5*v5+v6*v6+v7*v7;
    }
    for (int i = h8*8+tid; i < hidden_size; i += THREADS) {
        float v = __half2float(input[i]) + __half2float(add_vec[i]);
        s_normed[i] = v; local_ss += v*v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    // Phase 2: GEMV + bias -- warp-per-row
    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const int4* w4 = (const int4*)(proj_weight + (long long)row * hidden_size);
            const int wh8 = hidden_size / 8;
            float acc = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < wh8; i += 32) {
                int4 packed = w4[i];
                half2 w01=*reinterpret_cast<half2*>(&packed.x), w23=*reinterpret_cast<half2*>(&packed.y);
                half2 w45=*reinterpret_cast<half2*>(&packed.z), w67=*reinterpret_cast<half2*>(&packed.w);
                int base = i * 8;
                acc += __half2float(w01.x)*s_normed[base] + __half2float(w01.y)*s_normed[base+1]
                     + __half2float(w23.x)*s_normed[base+2] + __half2float(w23.y)*s_normed[base+3]
                     + __half2float(w45.x)*s_normed[base+4] + __half2float(w45.y)*s_normed[base+5]
                     + __half2float(w67.x)*s_normed[base+6] + __half2float(w67.y)*s_normed[base+7];
            }
            for (int i = wh8*8+lane_id; i < hidden_size; i += 32)
                acc += __half2float(proj_weight[(long long)row*hidden_size+i]) * s_normed[i];
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}

// --------------------------------------------------------------------------
// First-layer variant: RMSNorm + QKV GEMV (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < (hidden_size/2); i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const int4* w4 = (const int4*)(proj_weight + (long long)row * hidden_size);
            const int wh8 = hidden_size / 8;
            float acc = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < wh8; i += 32) {
                int4 packed = w4[i];
                half2 w01 = *reinterpret_cast<half2*>(&packed.x);
                half2 w23 = *reinterpret_cast<half2*>(&packed.y);
                half2 w45 = *reinterpret_cast<half2*>(&packed.z);
                half2 w67 = *reinterpret_cast<half2*>(&packed.w);
                int base = i * 8;
                acc += __half2float(w01.x)*s_normed[base] + __half2float(w01.y)*s_normed[base+1]
                     + __half2float(w23.x)*s_normed[base+2] + __half2float(w23.y)*s_normed[base+3]
                     + __half2float(w45.x)*s_normed[base+4] + __half2float(w45.y)*s_normed[base+5]
                     + __half2float(w67.x)*s_normed[base+6] + __half2float(w67.y)*s_normed[base+7];
            }
            for (int i = wh8*8+lane_id; i < hidden_size; i += 32)
                acc += __half2float(proj_weight[(long long)row*hidden_size+i]) * s_normed[i];
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// First-layer variant: RMSNorm + QKV GEMV + bias (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_bias_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    const __half* __restrict__ bias,        // [qkv_dim]
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < (hidden_size/2); i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const int4* w4 = (const int4*)(proj_weight + (long long)row * hidden_size);
            const int wh8 = hidden_size / 8;
            float acc = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < wh8; i += 32) {
                int4 packed = w4[i];
                half2 w01=*reinterpret_cast<half2*>(&packed.x), w23=*reinterpret_cast<half2*>(&packed.y);
                half2 w45=*reinterpret_cast<half2*>(&packed.z), w67=*reinterpret_cast<half2*>(&packed.w);
                int base = i * 8;
                acc += __half2float(w01.x)*s_normed[base] + __half2float(w01.y)*s_normed[base+1]
                     + __half2float(w23.x)*s_normed[base+2] + __half2float(w23.y)*s_normed[base+3]
                     + __half2float(w45.x)*s_normed[base+4] + __half2float(w45.y)*s_normed[base+5]
                     + __half2float(w67.x)*s_normed[base+6] + __half2float(w67.y)*s_normed[base+7];
            }
            for (int i = wh8*8+lane_id; i < hidden_size; i += 32)
                acc += __half2float(proj_weight[(long long)row*hidden_size+i]) * s_normed[i];
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}

// ==========================================================================
// FP8 variants: weights stored as E4M3 bytes with per-row scale
// ==========================================================================

// --------------------------------------------------------------------------
// FP8: Fused add + RMSNorm + QKV GEMV (layers 1..N)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_fp8_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h8 = hidden_size / 8;
    const int4* in4 = (const int4*)input;
    const int4* add4 = (const int4*)add_vec;
    #pragma unroll 2
    for (int i = tid; i < h8; i += THREADS) {
        int4 a4 = in4[i]; int4 b4 = add4[i];
        half2 a01=*reinterpret_cast<half2*>(&a4.x), a23=*reinterpret_cast<half2*>(&a4.y);
        half2 a45=*reinterpret_cast<half2*>(&a4.z), a67=*reinterpret_cast<half2*>(&a4.w);
        half2 b01=*reinterpret_cast<half2*>(&b4.x), b23=*reinterpret_cast<half2*>(&b4.y);
        half2 b45=*reinterpret_cast<half2*>(&b4.z), b67=*reinterpret_cast<half2*>(&b4.w);
        int base = i * 8;
        float v0=__half2float(a01.x)+__half2float(b01.x), v1=__half2float(a01.y)+__half2float(b01.y);
        float v2=__half2float(a23.x)+__half2float(b23.x), v3=__half2float(a23.y)+__half2float(b23.y);
        float v4=__half2float(a45.x)+__half2float(b45.x), v5=__half2float(a45.y)+__half2float(b45.y);
        float v6=__half2float(a67.x)+__half2float(b67.x), v7=__half2float(a67.y)+__half2float(b67.y);
        s_normed[base]=v0; s_normed[base+1]=v1; s_normed[base+2]=v2; s_normed[base+3]=v3;
        s_normed[base+4]=v4; s_normed[base+5]=v5; s_normed[base+6]=v6; s_normed[base+7]=v7;
        local_ss += v0*v0+v1*v1+v2*v2+v3*v3+v4*v4+v5*v5+v6*v6+v7*v7;
    }
    for (int i = h8*8+tid; i < hidden_size; i += THREADS) {
        float v = __half2float(input[i]) + __half2float(add_vec[i]);
        s_normed[i] = v; local_ss += v*v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// FP8: First-layer variant -- RMSNorm + QKV GEMV (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_fp8_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < (hidden_size/2); i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// FP8: Fused add + RMSNorm + QKV GEMV + bias
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_fp8_bias_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    const __half* __restrict__ bias,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h8 = hidden_size / 8;
    const int4* in4 = (const int4*)input;
    const int4* add4 = (const int4*)add_vec;
    #pragma unroll 2
    for (int i = tid; i < h8; i += THREADS) {
        int4 a4 = in4[i]; int4 b4 = add4[i];
        half2 a01=*reinterpret_cast<half2*>(&a4.x), a23=*reinterpret_cast<half2*>(&a4.y);
        half2 a45=*reinterpret_cast<half2*>(&a4.z), a67=*reinterpret_cast<half2*>(&a4.w);
        half2 b01=*reinterpret_cast<half2*>(&b4.x), b23=*reinterpret_cast<half2*>(&b4.y);
        half2 b45=*reinterpret_cast<half2*>(&b4.z), b67=*reinterpret_cast<half2*>(&b4.w);
        int base = i * 8;
        float v0=__half2float(a01.x)+__half2float(b01.x), v1=__half2float(a01.y)+__half2float(b01.y);
        float v2=__half2float(a23.x)+__half2float(b23.x), v3=__half2float(a23.y)+__half2float(b23.y);
        float v4=__half2float(a45.x)+__half2float(b45.x), v5=__half2float(a45.y)+__half2float(b45.y);
        float v6=__half2float(a67.x)+__half2float(b67.x), v7=__half2float(a67.y)+__half2float(b67.y);
        s_normed[base]=v0; s_normed[base+1]=v1; s_normed[base+2]=v2; s_normed[base+3]=v3;
        s_normed[base+4]=v4; s_normed[base+5]=v5; s_normed[base+6]=v6; s_normed[base+7]=v7;
        local_ss += v0*v0+v1*v1+v2*v2+v3*v3+v4*v4+v5*v5+v6*v6+v7*v7;
    }
    for (int i = h8*8+tid; i < hidden_size; i += THREADS) {
        float v = __half2float(input[i]) + __half2float(add_vec[i]);
        s_normed[i] = v; local_ss += v*v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}

// --------------------------------------------------------------------------
// FP8: First-layer variant -- RMSNorm + QKV GEMV + bias (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_fp8_bias_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    const __half* __restrict__ bias,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < (hidden_size/2); i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}
