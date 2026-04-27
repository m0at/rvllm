// Cycle 39 step 2: AWQ INT4×FP16 GEMV for decode (M=1).
//
// W4A16 attack on the bs=1 decode bandwidth ceiling:
//   FP8-block weights = 31 GB/token at 273 GB/s = 8.8 tok/s ceiling
//   INT4 weights      = ~16 GB/token + scales + zeros = ~17.5 GB/token
//                                         ≈ 15.6 tok/s ceiling
// Plausible 1.5-2× decode improvement from this kernel alone, before
// kernel-launch / prefill optimizations.
//
// Layout (compressed-tensors AWQ, [N, K] weight matrix with per-group
// scale/zero-point on the K dimension):
//
//   weight_packed:  uint8 [N, K/2]
//                   byte at [n, k/2] holds elements (n, 2*(k/2))
//                   in the LOW nibble and (n, 2*(k/2)+1) in the HIGH
//                   nibble. (Same convention as our NVFP4 packer.)
//   scale_f16:      __half [N, K/group_size]
//   zero_packed:    uint8 [N, K/(2*group_size)] — 2 INT4 zeros per byte,
//                   one per group. zero_point per (n, g) lives in
//                   nibble (g & 1) of byte zero_packed[n, g/2].
//   activation:     __half [K]   (one row, M=1 decode)
//   output:         __half [N]
//
// Symmetric AWQ variants set zero_point = 8 (== 2^(bits-1)) so the
// dequant becomes `value = (w - 8) * scale`. Asymmetric variants
// store a real zero per group. Both are decoded by the same formula:
//
//   value(n, k) = (nibble(weight[n, k/2], k & 1) - zero(n, k/group_size))
//                  * scale(n, k/group_size)
//
// One block per output row (gridDim.x = N). Each block has 32 threads
// (one warp), each thread loops over K/32 in stride. Shared-memory
// reduction at the end.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" __global__
void awq_int4_gemv_f16_kernel(
    const __half*  __restrict__ activation,    // [K] f16
    const uint8_t* __restrict__ weight_packed, // [N, K/2] u8 (2x INT4)
    const __half*  __restrict__ scale,         // [N, K/group_size] f16
    const uint8_t* __restrict__ zero_packed,   // [N, K/(2*group_size)] u8
    __half*        __restrict__ output,        // [N] f16
    int            N,
    int            K,
    int            group_size                  // typically 128
) {
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;
    const int LANES = blockDim.x;  // expected 32

    if (n >= N) return;

    // Walk K in stride-LANES chunks. We grab two INT4 elements per byte
    // load; bumping by `2 * LANES` keeps each thread on its own
    // (k%2)==0 boundary so the unpack stays branch-free.
    float partial = 0.0f;
    const int row_byte_off  = n * (K >> 1);                    // K/2
    const int row_scale_off = n * (K / group_size);
    const int row_zero_off  = n * (K / (2 * group_size));

    for (int k_base = tid * 2; k_base < K; k_base += LANES * 2) {
        const int byte_idx = row_byte_off + (k_base >> 1);
        const uint8_t b    = weight_packed[byte_idx];
        const int w_lo     = b & 0x0F;          // element k_base
        const int w_hi     = (b >> 4) & 0x0F;   // element k_base+1

        const int g = k_base / group_size;      // both lanes share group
        const __half s_h = scale[row_scale_off + g];
        const float  s_f = __half2float(s_h);

        // Per-(n, g) zero point: byte at zero_packed[n, g/2], nibble (g&1).
        const uint8_t z_byte = zero_packed[row_zero_off + (g >> 1)];
        const int z_int = ((g & 1) == 0) ? (z_byte & 0x0F) : ((z_byte >> 4) & 0x0F);
        const float z_f = static_cast<float>(z_int);

        const float dq_lo = (static_cast<float>(w_lo) - z_f) * s_f;
        const float dq_hi = (static_cast<float>(w_hi) - z_f) * s_f;

        // Gate by K bound — last partial group may be short.
        const float a_lo = (k_base     < K)
            ? __half2float(activation[k_base    ])
            : 0.0f;
        const float a_hi = (k_base + 1 < K)
            ? __half2float(activation[k_base + 1])
            : 0.0f;

        partial += dq_lo * a_lo + dq_hi * a_hi;
    }

    // Warp reduction (assume LANES == 32, single warp per block).
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        partial += __shfl_xor_sync(0xFFFFFFFFu, partial, o);
    }
    if (tid == 0) {
        output[n] = __float2half(partial);
    }
}
