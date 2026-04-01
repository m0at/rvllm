// FP8 E4M3 -> FP16 dequantization kernel for weight loading.
//
// Converts FP8 E4M3 format (1 sign, 4 exponent bias=7, 3 mantissa) to FP16
// with optional per-tensor scale factor. Used to dequantize FP8 model weights
// on-the-fly before cuBLAS HGEMM.
//
// Launch config:
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none

#include <cuda_fp16.h>

// Convert a single FP8 E4M3 byte to FP16 via bit manipulation.
// FP8 E4M3: 1 sign | 4 exponent (bias=7) | 3 mantissa
// FP16:     1 sign | 5 exponent (bias=15) | 10 mantissa
__device__ __forceinline__ __half fp8e4m3_to_half(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;

    if (e == 0u) {
        if (m == 0u) {
            // +/- zero
            unsigned short h = (unsigned short)(s << 15);
            return __ushort_as_half(h);
        }
        // Subnormal: value = (-1)^s * 2^(1-7) * (0 + m/8) = (-1)^s * m * 2^(-9)
        // Use float intermediate for subnormal normalization
        float fval = (float)m * 1.953125e-3f; // 1/512 = 2^(-9)
        if (s) fval = -fval;
        return __float2half(fval);
    }

    if (e == 0xFu && m == 0x7u) {
        // NaN (E4M3 only has one NaN: 0x7F / 0xFF)
        return __ushort_as_half((unsigned short)0x7FFFu);
    }

    // Normal: rebase exponent from E4M3 bias (7) to FP16 bias (15)
    // fp16_exp = e + 8, fp16_mant = m << 7 (3 mantissa bits -> 10)
    unsigned short h = (unsigned short)((s << 15) | ((e + 8u) << 10) | (m << 7));
    return __ushort_as_half(h);
}

// Dequantize FP8 E4M3 to FP16 without scaling.
extern "C"
__global__ void dequant_fp8_to_f16_kernel(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fp8e4m3_to_half(input[idx]);
    }
}

// Dequantize FP8 E4M3 to FP16 with per-tensor f32 scale.
// output[i] = half(fp8_to_float(input[i]) * scale)
// Uses f32 intermediate to avoid precision loss from converting scale to f16 first.
extern "C"
__global__ void dequant_fp8_scaled_to_f16_kernel(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    const float* __restrict__ scale_ptr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = scale_ptr[0];
        float val = __half2float(fp8e4m3_to_half(input[idx])) * s;
        output[idx] = __float2half(val);
    }
}

// Dequantize FP8 E4M3 to FP16 with block-wise f32 scale (128x128 blocks).
// Weight is [rows, cols] row-major. Scale is [ceil(rows/block_size), ceil(cols/block_size)] flattened.
// output[idx] = half(fp8_to_float(input[idx]) * scale[block_row * num_col_blocks + block_col])
// Uses f32 intermediate to avoid precision loss from converting scale to f16 first.
extern "C"
__global__ void dequant_fp8_blockwise_to_f16_kernel(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    const float* __restrict__ scale_ptr,
    int rows,
    int cols,
    int block_size,
    int num_col_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx < n) {
        int row = idx / cols;
        int col = idx % cols;
        int scale_row = row / block_size;
        int scale_col = col / block_size;
        int scale_idx = scale_row * num_col_blocks + scale_col;
        float s = scale_ptr[scale_idx];
        float val = __half2float(fp8e4m3_to_half(input[idx])) * s;
        output[idx] = __float2half(val);
    }
}

// ---- BF16 dequantization kernels ----
// These produce BF16 outputs to match HuggingFace/PyTorch BF16 inference numerics.
// BF16 has the same exponent range as f32 (8-bit exponent) but only 7 mantissa bits,
// avoiding the exponent overflow/underflow issues of f16 (5-bit exponent).

// Helper: convert f32 to BF16 stored as unsigned short, with round-to-nearest-even.
__device__ __forceinline__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_uint(val);
    unsigned int exp = (bits >> 23) & 0xFFu;
    if (exp == 0xFFu) {
        // NaN or Inf: preserve sign + exponent, truncate mantissa
        return (unsigned short)(bits >> 16);
    }
    // Round to nearest even
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}

// Dequantize FP8 E4M3 to BF16 without scaling.
extern "C"
__global__ void dequant_fp8_to_bf16_kernel(
    unsigned short* __restrict__ output,
    const unsigned char* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(fp8e4m3_to_half(input[idx]));
        output[idx] = float_to_bf16(val);
    }
}

// Dequantize FP8 E4M3 to BF16 with per-tensor f32 scale.
extern "C"
__global__ void dequant_fp8_scaled_to_bf16_kernel(
    unsigned short* __restrict__ output,
    const unsigned char* __restrict__ input,
    const float* __restrict__ scale_ptr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = scale_ptr[0];
        float val = __half2float(fp8e4m3_to_half(input[idx])) * s;
        output[idx] = float_to_bf16(val);
    }
}

// Dequantize FP8 E4M3 to BF16 with block-wise f32 scale (128x128 blocks).
extern "C"
__global__ void dequant_fp8_blockwise_to_bf16_kernel(
    unsigned short* __restrict__ output,
    const unsigned char* __restrict__ input,
    const float* __restrict__ scale_ptr,
    int rows,
    int cols,
    int block_size,
    int num_col_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx < n) {
        int row = idx / cols;
        int col = idx % cols;
        int scale_row = row / block_size;
        int scale_col = col / block_size;
        int scale_idx = scale_row * num_col_blocks + scale_col;
        float s = scale_ptr[scale_idx];
        float val = __half2float(fp8e4m3_to_half(input[idx])) * s;
        output[idx] = float_to_bf16(val);
    }
}
