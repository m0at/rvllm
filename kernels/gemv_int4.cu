// INT4 (W4A16) weight GEMV kernel for M=1 decode.
//
// Quarters weight bandwidth vs f16 GEMV by storing weights as packed 4-bit
// integers with per-group scale+zero_point. Input/output remain f16,
// accumulation in f32 for stability.
//
// Weight layout: GPTQ/AWQ-compatible column-packed uint32
//   Each uint32 packs 8 x int4 values (little-endian nibble order).
//   weight[out_dim, in_dim/8] as uint32, each containing 8 consecutive K elements.
//
// Quantization: asymmetric per-group
//   group_size: typically 128
//   scales[out_dim, num_groups] as f16
//   zeros[out_dim, num_groups] as f16 (or packed uint32 for AWQ)
//   dequant: w_f16 = scale * (w_int4 - zero)
//
// Three kernels:
//   1. gemv_int4_kernel       -- standalone W4A16 GEMV
//   2. fused_add_norm_int4_qkv_gemv -- fused add+norm+QKV GEMV with int4 weights
//   3. fused_add_norm_int4_gateup_gemv -- fused add+norm+gate+up GEMV with int4
//
// Launch config (standalone):
//   Grid:  ((out_dim + RPB - 1) / RPB, 1, 1)
//   Block: (256, 1, 1)
//   Shared: RPB * sizeof(float) (warp reduction scratch)

#include <cuda_fp16.h>

#define INT4_THREADS 256
#define INT4_RPB 8
#define INT4_GROUP_SIZE 128

__device__ __forceinline__ float warp_reduce_sum_i4(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Extract int4 value from packed uint32 at nibble position [0..7]
__device__ __forceinline__ int extract_int4(unsigned int packed, int nibble) {
    return (int)((packed >> (nibble * 4)) & 0xF);
}

// ============================================================================
// Standalone INT4 GEMV
// y[out_dim] = dequant(weight_int4[out_dim, in_dim/8]) @ x[in_dim]
//
// weight:  [out_dim, in_dim/8]  packed uint32 (8 int4 per uint32)
// scales:  [out_dim, num_groups] f16, where num_groups = in_dim / group_size
// zeros:   [out_dim, num_groups] f16 (zero point per group)
// ============================================================================
extern "C"
__global__ void __launch_bounds__(INT4_THREADS)
gemv_int4_kernel(
    __half* __restrict__ output,       // [out_dim]
    const __half* __restrict__ input,  // [in_dim]
    const unsigned int* __restrict__ weight,  // [out_dim, in_dim/8] packed
    const __half* __restrict__ scales, // [out_dim, num_groups]
    const __half* __restrict__ zeros,  // [out_dim, num_groups]
    int out_dim,
    int in_dim,
    int group_size
) {
    const int block_base = blockIdx.x * INT4_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_groups = (in_dim + group_size - 1) / group_size;

    const int row = block_base + warp_id;
    if (row >= out_dim) return;

    const int packed_k = in_dim / 8;  // number of uint32 per row
    const unsigned int* row_w = weight + (long long)row * packed_k;
    const __half* row_scales = scales + (long long)row * num_groups;
    const __half* row_zeros = zeros + (long long)row * num_groups;

    float acc = 0.0f;

    // Each lane processes packed_k/32 uint32s = in_dim/256 groups of 8 weights
    for (int i = lane_id; i < packed_k; i += 32) {
        unsigned int packed = row_w[i];
        int k_base = i * 8;
        int group_id = k_base / group_size;

        float s = __half2float(row_scales[group_id]);
        float z = __half2float(row_zeros[group_id]);

        // Unpack 8 int4 values and accumulate
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int w_int = extract_int4(packed, n);
            float w_f = s * ((float)w_int - z);
            float x_f = __half2float(input[k_base + n]);
            acc += w_f * x_f;
        }
    }

    // Warp reduction
    acc = warp_reduce_sum_i4(acc);

    // Cross-warp reduction
    __shared__ float s_scratch[INT4_RPB];
    if (lane_id == 0) s_scratch[warp_id] = acc;
    // Only 1 warp per row (warp_id = row within block), no cross-warp needed
    // for RPB=8 with 8 warps: each warp owns one row
    if (lane_id == 0) output[row] = __float2half(acc);
}

// ============================================================================
// Fused add + RMSNorm + INT4 QKV GEMV
// residual_out = input + add_vec
// normed = RMSNorm(residual_out) * norm_weight
// output = dequant(proj_weight_int4) @ normed
// ============================================================================
extern "C"
__global__ void __launch_bounds__(INT4_THREADS)
fused_add_norm_int4_qkv_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned int* __restrict__ proj_weight,  // [qkv_dim, hidden/8] packed int4
    const __half* __restrict__ scales,             // [qkv_dim, num_groups]
    const __half* __restrict__ zeros,              // [qkv_dim, num_groups]
    float eps,
    int hidden_size,
    int qkv_dim,
    int group_size
) {
    const int block_base = blockIdx.x * INT4_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = INT4_THREADS / 32;
    const int num_groups = (hidden_size + group_size - 1) / group_size;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    // Phase 1: residual add + RMSNorm into shared memory
    float local_ss = 0.0f;
    const int h8 = hidden_size / 8;
    const int4* in4 = (const int4*)input;
    const int4* add4 = (const int4*)add_vec;

    #pragma unroll 2
    for (int i = tid; i < h8; i += INT4_THREADS) {
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
    for (int i = h8*8+tid; i < hidden_size; i += INT4_THREADS) {
        float v = __half2float(input[i]) + __half2float(add_vec[i]);
        s_normed[i] = v; local_ss += v*v;
    }

    local_ss = warp_reduce_sum_i4(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_i4(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    // Write pre-norm residual (block 0 only)
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += INT4_THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    // Apply norm weights
    for (int i = tid; i < hidden_size; i += INT4_THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    // Phase 2: INT4 GEMV -- warp-per-row
    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const int packed_k = hidden_size / 8;
            const unsigned int* row_w = proj_weight + (long long)row * packed_k;
            const __half* row_s = scales + (long long)row * num_groups;
            const __half* row_z = zeros + (long long)row * num_groups;

            float acc = 0.0f;

            for (int i = lane_id; i < packed_k; i += 32) {
                unsigned int packed = row_w[i];
                int k_base = i * 8;
                int group_id = k_base / group_size;
                float s = __half2float(row_s[group_id]);
                float z = __half2float(row_z[group_id]);

                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    int w_int = extract_int4(packed, n);
                    float w_f = s * ((float)w_int - z);
                    acc += w_f * s_normed[k_base + n];
                }
            }

            acc = warp_reduce_sum_i4(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// ============================================================================
// Fused add + RMSNorm + INT4 gate+up GEMV (SwiGLU MLP first stage)
// Same structure as QKV but for the larger gate+up projection
// ============================================================================
extern "C"
__global__ void __launch_bounds__(INT4_THREADS)
fused_add_norm_int4_gateup_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned int* __restrict__ proj_weight,  // [gate_up_dim, hidden/8]
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    float eps,
    int hidden_size,
    int gate_up_dim,
    int group_size
) {
    const int block_base = blockIdx.x * INT4_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = INT4_THREADS / 32;
    const int num_groups = (hidden_size + group_size - 1) / group_size;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    // Phase 1: residual add + RMSNorm (identical to QKV variant)
    float local_ss = 0.0f;
    const int h8 = hidden_size / 8;
    const int4* in4 = (const int4*)input;
    const int4* add4 = (const int4*)add_vec;

    #pragma unroll 2
    for (int i = tid; i < h8; i += INT4_THREADS) {
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
    for (int i = h8*8+tid; i < hidden_size; i += INT4_THREADS) {
        float v = __half2float(input[i]) + __half2float(add_vec[i]);
        s_normed[i] = v; local_ss += v*v;
    }

    local_ss = warp_reduce_sum_i4(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_i4(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += INT4_THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += INT4_THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    // Phase 2: INT4 GEMV for gate+up (larger output dimension)
    {
        const int row = block_base + warp_id;
        if (row < gate_up_dim) {
            const int packed_k = hidden_size / 8;
            const unsigned int* row_w = proj_weight + (long long)row * packed_k;
            const __half* row_s = scales + (long long)row * num_groups;
            const __half* row_z = zeros + (long long)row * num_groups;

            float acc = 0.0f;

            for (int i = lane_id; i < packed_k; i += 32) {
                unsigned int packed = row_w[i];
                int k_base = i * 8;
                int group_id = k_base / group_size;
                float s = __half2float(row_s[group_id]);
                float z = __half2float(row_z[group_id]);

                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    int w_int = extract_int4(packed, n);
                    float w_f = s * ((float)w_int - z);
                    acc += w_f * s_normed[k_base + n];
                }
            }

            acc = warp_reduce_sum_i4(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// ============================================================================
// Fused SiLU + INT4 down-projection GEMV
// silu_out = SiLU(gate_up[0:intermediate]) * gate_up[intermediate:2*intermediate]
// output = dequant(down_weight_int4) @ silu_out
// ============================================================================
extern "C"
__global__ void __launch_bounds__(INT4_THREADS)
fused_silu_int4_down_gemv(
    __half* __restrict__ output,          // [out_dim]
    const __half* __restrict__ gate_up,   // [intermediate * 2]
    const unsigned int* __restrict__ down_weight, // [out_dim, intermediate/8] packed
    const __half* __restrict__ scales,    // [out_dim, num_groups]
    const __half* __restrict__ zeros,     // [out_dim, num_groups]
    int out_dim,
    int intermediate,
    int group_size
) {
    const int block_base = blockIdx.x * INT4_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_groups = (intermediate + group_size - 1) / group_size;

    const int row = block_base + warp_id;
    if (row >= out_dim) return;

    const int packed_k = intermediate / 8;
    const unsigned int* row_w = down_weight + (long long)row * packed_k;
    const __half* row_s = scales + (long long)row * num_groups;
    const __half* row_z = zeros + (long long)row * num_groups;

    float acc = 0.0f;

    for (int i = lane_id; i < packed_k; i += 32) {
        unsigned int packed = row_w[i];
        int k_base = i * 8;
        int group_id = k_base / group_size;
        float s = __half2float(row_s[group_id]);
        float z = __half2float(row_z[group_id]);

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int k = k_base + n;
            // Fused SiLU: gate = SiLU(gate_up[k]), up = gate_up[intermediate + k]
            float gate_val = __half2float(gate_up[k]);
            float up_val = __half2float(gate_up[intermediate + k]);
            float silu = gate_val / (1.0f + expf(-gate_val));
            float act = silu * up_val;

            int w_int = extract_int4(packed, n);
            float w_f = s * ((float)w_int - z);
            acc += w_f * act;
        }
    }

    acc = warp_reduce_sum_i4(acc);
    if (lane_id == 0) output[row] = __float2half(acc);
}
