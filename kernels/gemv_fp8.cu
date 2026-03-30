// FP8 E4M3 weight GEMV kernels for M=1 decode.
//
// Halves weight memory bandwidth vs f16 GEMV by storing weights as 1-byte
// FP8 E4M3 with per-row scale factors. Input/output remain f16, accumulation
// in f32 for stability.
//
// Two kernels:
//   1. gemv_fp8_kernel -- standalone GEMV with FP8 weights
//   2. fused_add_norm_fp8_gemv_kernel -- fused add+norm+GEMV with FP8 weights
//
// FP8 E4M3: 1 sign, 4 exponent, 3 mantissa. Range [-448, 448].
// Uses the software fp8<->float routines from fp8_kv.cu pattern
// (no cuda_fp8.h dependency -- works on sm_80+).

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define FP8_THREADS 256
#define FP8_RPB 8

// Hardware FP8 E4M3 -> float via __nv_fp8_e4m3 (sm_89+, single instruction)
__device__ __forceinline__ float fp8e4m3_to_float(unsigned char fp8) {
    return float(*reinterpret_cast<const __nv_fp8_e4m3*>(&fp8));
}

__device__ __forceinline__ float warp_reduce_sum_fp8(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Standalone FP8 weight GEMV
// y[out_dim] = weight_fp8[out_dim, in_dim] @ x[in_dim] * scale[out_dim]
//
// Grid:  ((out_dim + RPB-1) / RPB, 1, 1)
// Block: (256, 1, 1)
// Shared: RPB * sizeof(float)
// ---------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(FP8_THREADS)
gemv_fp8_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const __half* __restrict__ scale,
    int out_dim,
    int in_dim
) {
    const int block_base = blockIdx.x * FP8_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp-per-row: 8 warps handle 8 output rows
    const int row = block_base + warp_id;
    if (row >= out_dim) return;

    const unsigned char* w_row = weight + (long long)row * in_dim;
    float row_scale = __half2float(scale[row]);
    float acc = 0.0f;

    // Load 4 FP8 values at once for coalescing (4 bytes per iteration)
    int in_dim_aligned = in_dim & ~3;
    for (int i = lane_id * 4; i < in_dim_aligned; i += 32 * 4) {
        // Load 4 consecutive FP8 bytes
        // Use uint32 load for coalescing when aligned
        unsigned char w0 = w_row[i];
        unsigned char w1 = w_row[i + 1];
        unsigned char w2 = w_row[i + 2];
        unsigned char w3 = w_row[i + 3];

        float f0 = fp8e4m3_to_float(w0) * row_scale;
        float f1 = fp8e4m3_to_float(w1) * row_scale;
        float f2 = fp8e4m3_to_float(w2) * row_scale;
        float f3 = fp8e4m3_to_float(w3) * row_scale;

        acc += f0 * __half2float(input[i])
             + f1 * __half2float(input[i + 1])
             + f2 * __half2float(input[i + 2])
             + f3 * __half2float(input[i + 3]);
    }

    // Handle tail elements
    for (int i = in_dim_aligned + lane_id; i < in_dim; i += 32) {
        acc += fp8e4m3_to_float(w_row[i]) * row_scale * __half2float(input[i]);
    }

    acc = warp_reduce_sum_fp8(acc);
    if (lane_id == 0) {
        output[row] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Fused residual-add + RMSNorm + FP8 weight GEMV
//
// Phase 1: add + RMSNorm into shared memory (all f16 I/O, f32 compute)
// Phase 2: GEMV with FP8 weight loads (1 byte per element vs 2 for f16)
//
// Grid:  ((out_dim + RPB-1) / RPB, 1, 1)
// Block: (256, 1, 1)
// Shared: hidden_size * sizeof(float) + RPB * sizeof(float)
// ---------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(FP8_THREADS)
fused_add_norm_fp8_gemv_kernel(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    float eps,
    int hidden_size,
    int out_dim
) {
    const int block_base = blockIdx.x * FP8_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = FP8_THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    // Phase 1: residual add + RMSNorm
    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    for (int i = tid; i < h2; i += FP8_THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]) + __half2float(add_vec[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_fp8(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_fp8(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    // Block 0 writes the residual output
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += FP8_THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    // Apply norm weights
    for (int i = tid; i < hidden_size; i += FP8_THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    // Phase 2: GEMV with FP8 weights -- warp-per-row
    {
        const int row = block_base + warp_id;
        if (row < out_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;

            // 4-wide FP8 loads
            int hs_aligned = hidden_size & ~3;
            for (int i = lane_id * 4; i < hs_aligned; i += 32 * 4) {
                unsigned char w0 = w_row[i];
                unsigned char w1 = w_row[i + 1];
                unsigned char w2 = w_row[i + 2];
                unsigned char w3 = w_row[i + 3];

                acc += fp8e4m3_to_float(w0) * row_sc * s_normed[i]
                     + fp8e4m3_to_float(w1) * row_sc * s_normed[i + 1]
                     + fp8e4m3_to_float(w2) * row_sc * s_normed[i + 2]
                     + fp8e4m3_to_float(w3) * row_sc * s_normed[i + 3];
            }

            for (int i = hs_aligned + lane_id; i < hidden_size; i += 32) {
                acc += fp8e4m3_to_float(w_row[i]) * row_sc * s_normed[i];
            }

            acc = warp_reduce_sum_fp8(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Norm-only variant (first layer, no residual add)
// ---------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(FP8_THREADS)
fused_norm_fp8_gemv_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    float eps,
    int hidden_size,
    int out_dim
) {
    const int block_base = blockIdx.x * FP8_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = FP8_THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < h2; i += FP8_THREADS) {
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

    local_ss = warp_reduce_sum_fp8(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_fp8(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += FP8_THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < out_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;

            int hs_aligned = hidden_size & ~3;
            for (int i = lane_id * 4; i < hs_aligned; i += 32 * 4) {
                unsigned char w0 = w_row[i];
                unsigned char w1 = w_row[i + 1];
                unsigned char w2 = w_row[i + 2];
                unsigned char w3 = w_row[i + 3];

                acc += fp8e4m3_to_float(w0) * row_sc * s_normed[i]
                     + fp8e4m3_to_float(w1) * row_sc * s_normed[i + 1]
                     + fp8e4m3_to_float(w2) * row_sc * s_normed[i + 2]
                     + fp8e4m3_to_float(w3) * row_sc * s_normed[i + 3];
            }

            for (int i = hs_aligned + lane_id; i < hidden_size; i += 32) {
                acc += fp8e4m3_to_float(w_row[i]) * row_sc * s_normed[i];
            }

            acc = warp_reduce_sum_fp8(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused SiLU(gate)*up + FP8 down projection GEMV
//
// Grid:  ((hidden_size + RPB-1) / RPB, 1, 1)
// Block: (256, 1, 1)
// Shared: RPB * sizeof(float)
// ---------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(FP8_THREADS)
fused_silu_down_fp8_gemv_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const unsigned char* __restrict__ weight,
    const __half* __restrict__ weight_scale,
    int hidden_size,
    int intermediate_size
) {
    const int block_base = blockIdx.x * FP8_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int row = block_base + warp_id;
    if (row >= hidden_size) return;

    const unsigned char* w_row = weight + (long long)row * intermediate_size;
    float row_sc = __half2float(weight_scale[row]);
    float acc = 0.0f;

    // Each lane processes 4 elements at a time
    int is_aligned = intermediate_size & ~3;
    for (int i = lane_id * 4; i < is_aligned; i += 32 * 4) {
        // Compute silu(gate) * up inline
        float g0 = __half2float(gate[i]);
        float g1 = __half2float(gate[i + 1]);
        float g2 = __half2float(gate[i + 2]);
        float g3 = __half2float(gate[i + 3]);

        float s0 = g0 / (1.0f + __expf(-g0)) * __half2float(up[i]);
        float s1 = g1 / (1.0f + __expf(-g1)) * __half2float(up[i + 1]);
        float s2 = g2 / (1.0f + __expf(-g2)) * __half2float(up[i + 2]);
        float s3 = g3 / (1.0f + __expf(-g3)) * __half2float(up[i + 3]);

        acc += fp8e4m3_to_float(w_row[i])     * row_sc * s0
             + fp8e4m3_to_float(w_row[i + 1]) * row_sc * s1
             + fp8e4m3_to_float(w_row[i + 2]) * row_sc * s2
             + fp8e4m3_to_float(w_row[i + 3]) * row_sc * s3;
    }

    for (int i = is_aligned + lane_id; i < intermediate_size; i += 32) {
        float g = __half2float(gate[i]);
        float s = g / (1.0f + __expf(-g)) * __half2float(up[i]);
        acc += fp8e4m3_to_float(w_row[i]) * row_sc * s;
    }

    acc = warp_reduce_sum_fp8(acc);
    if (lane_id == 0) {
        output[row] = __float2half(acc);
    }
}
