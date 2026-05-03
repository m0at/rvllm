// Cycle 40 step 3: AWQ INT4×FP16 GEMV for decode (M=1).
//
// Updated cycle 40 to match the actual compressed-tensors AWQ format
// shipped by HuggingFace AWQ Gemma 4 31B (ebircak/gemma-4-31B-it-4bit-
// W4A16-AWQ): weights and zero-points are I32 packed (8 INT4 per int32),
// not U8 packed (2 per byte) as the cycle-39 prototype assumed.
//
// W4A16 attack on the bs=1 decode bandwidth ceiling:
//   FP8-block weights = ~31 GB/token at 273 GB/s = 8.8 tok/s ceiling
//   INT4 weights      = ~16 GB + scales/zeros ≈ 17.5 GB/token
//                                              ≈ 15.6 tok/s ceiling
//
// Compressed-tensors AWQ layout (Gemma 4 q_proj example: out=8192, in=5376):
//   weight_packed:    int32_t [N, K/8]
//                     each int32 holds 8 INT4 elements along K.
//                     For output row n, K-position k:
//                       lane = k % 8
//                       value = (weight_packed[n, k/8] >> (4*lane)) & 0xF
//   weight_scale:     bfloat16 [N, K/group_size]
//   weight_zero_point: int32_t [N/8, K/group_size]
//                     packed along N — each int32 holds 8 INT4 zeros for
//                     8 consecutive output rows in the same K-group:
//                       value = (weight_zero_point[n/8, g] >> (4*(n%8))) & 0xF
//   activation:       __half [K]
//   output:           __half [N]
//
// Asymmetric AWQ (the common case in Gemma 4 W4A16): symmetric=false in
// quant_config.json. zero_point is a real per-group, per-row INT4 value.
// Symmetric variants set zero_point = 8 (== 2^(bits-1)).
//
//   value(n, k) = (nibble - zero_point) * scale
//
// Both use the same dequant.
//
// One block per output row (gridDim.x = N). 32 threads per block (one
// warp). Each thread loops over K in stride 32*8 = 256 elements (one
// int32 = 8 INT4 lanes; 32 threads cover 256 K-positions per iteration).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__
void awq_int4_gemv_f16_kernel(
    const __half*           __restrict__ activation,         // [K] f16
    const int32_t*          __restrict__ weight_packed,      // [N, K/8] i32
    const __nv_bfloat16*    __restrict__ weight_scale,       // [N, K/g] bf16
    const int32_t*          __restrict__ weight_zero_point,  // [N/8, K/g] i32
    __half*                 __restrict__ output,             // [N] f16
    int                     N,
    int                     K,
    int                     group_size                       // typically 128
) {
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;
    const int LANES = blockDim.x;  // expected 32

    if (n >= N) return;

    const int K_div_8 = K >> 3;
    const int K_div_g = K / group_size;
    const int row_w_off = n * K_div_8;
    const int row_s_off = n * K_div_g;
    // Zero point: indexed by [n/8, g], unpack nibble (n & 7).
    const int n_group       = n >> 3;
    const int n_lane        = n & 7;
    const int row_z_off     = n_group * K_div_g;
    const int z_shift_bits  = 4 * n_lane;

    // Walk K in stride-LANES int32 chunks. Each int32 carries 8 INT4
    // elements along K, so per-iteration work is (LANES * 8) K-positions.
    float partial = 0.0f;

    for (int k_int_idx = tid; k_int_idx < K_div_8; k_int_idx += LANES) {
        const int k_base = k_int_idx << 3;        // first K-position in this i32
        const int g      = k_base / group_size;   // K-group all 8 lanes share

        // Load the packed weight i32 and the per-(n, g) scale + per-(n, g) zero.
        const int32_t w_pack = weight_packed[row_w_off + k_int_idx];
        const float   s_f    = __bfloat162float(weight_scale[row_s_off + g]);
        const int32_t z_pack = weight_zero_point[row_z_off + g];
        const int     z_int  = (z_pack >> z_shift_bits) & 0xF;
        const float   z_f    = static_cast<float>(z_int);

        // Decode 8 INT4 lanes and dot-product against activation.
        // Lane order matches Python `(packed >> (4*i)) & 0xF` for i in 0..7
        // — the convention compressed-tensors uses.
        #pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const int k = k_base + lane;
            if (k >= K) break;
            const int   w_int = (w_pack >> (4 * lane)) & 0xF;
            const float dq    = (static_cast<float>(w_int) - z_f) * s_f;
            const float a     = __half2float(activation[k]);
            partial += dq * a;
        }
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
