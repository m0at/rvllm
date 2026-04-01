// f32 <-> f16 / bf16 cast kernels for mixed-precision inference.
//
// Launch config:
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void cast_f32_to_f16_kernel(
    __half* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

extern "C"
__global__ void cast_f16_to_f32_kernel(
    float* __restrict__ output,
    const __half* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// Cast f32 to BF16 (stored as unsigned short / half::bf16).
// BF16: 1 sign + 8 exponent + 7 mantissa. Uses round-to-nearest-even.
extern "C"
__global__ void cast_f32_to_bf16_kernel(
    unsigned short* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int bits = __float_as_uint(input[idx]);
        unsigned int exp = (bits >> 23) & 0xFFu;
        if (exp == 0xFFu) {
            // NaN or Inf: preserve sign + exponent, truncate mantissa
            output[idx] = (unsigned short)(bits >> 16);
            return;
        }
        // Round to nearest even
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        output[idx] = (unsigned short)(bits >> 16);
    }
}

// Cast BF16 (stored as unsigned short / half::bf16) to f32.
// BF16 -> f32 is exact: just shift left by 16 bits.
extern "C"
__global__ void cast_bf16_to_f32_kernel(
    float* __restrict__ output,
    const unsigned short* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int bits = ((unsigned int)input[idx]) << 16;
        output[idx] = __uint_as_float(bits);
    }
}

// Round f32 values to BF16 precision in-place.
// BF16: 1 sign + 8 exponent + 7 mantissa (same exponent as f32, truncated mantissa).
// Uses round-to-nearest-even: adds 0x7FFF + bit16 for tie-breaking.
// This keeps f32 computation on the same numerical trajectory as BF16 inference
// (matching HuggingFace / PyTorch BF16 behavior).
extern "C"
__global__ void round_f32_to_bf16_kernel(
    float* __restrict__ data,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int bits = __float_as_uint(data[idx]);
        // Handle special cases: NaN, Inf pass through
        unsigned int exp = (bits >> 23) & 0xFFu;
        if (exp == 0xFFu) return;  // NaN or Inf - don't modify
        // Round to nearest even (BF16 rounding)
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        bits &= 0xFFFF0000u;
        data[idx] = __uint_as_float(bits);
    }
}
