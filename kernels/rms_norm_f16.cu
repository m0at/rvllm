// Half-precision RMSNorm kernel using __half and __half2 for throughput.
// Accumulation is done in f32 for numerical stability, but reads/writes
// are f16, halving memory bandwidth requirements.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared memory: blockDim.x * sizeof(float)

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

    const __half* x = input + token_idx * hidden_size;
    __half* y = output + token_idx * hidden_size;

    extern __shared__ float sdata[];

    // Pass 1: sum of squares in f32 for precision
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(x[i]);
        local_ss += val * val;
    }
    sdata[tid] = local_ss;
    __syncthreads();

    // Parallel reduction
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_scale = rsqrtf(sdata[0] / (float)hidden_size + eps);

    // Pass 2: normalize and scale, write f16
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(x[i]) * __half2float(weight[i]) * rms_scale;
        y[i] = __float2half(val);
    }
}

// Fused residual add + RMS norm, f16 variant.
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

    extern __shared__ float sdata[];

    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(input[row_offset + i]) + __half2float(add[row_offset + i]);
        residual[row_offset + i] = __float2half(val);
        local_ss += val * val;
    }
    sdata[tid] = local_ss;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_scale = rsqrtf(sdata[0] / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(residual[row_offset + i]) * __half2float(weight[i]) * rms_scale;
        output[row_offset + i] = __float2half(val);
    }
}
