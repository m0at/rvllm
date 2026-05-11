// =============================================================
//  mistral35_w4a16_gemv_bf16.cu
// =============================================================
//
// Fused m=1 W4A16 GEMV for Mistral 3.5 LLMCompressor "nvfp4-pack-
// quantized" weights. Streams the dequant inside the GEMV tile —
// no 705 MB BF16 weight scratch, no per-projection 88×7 dequant
// launches per token.
//
// Math (Round-8 fix — no `/FP4_MAX`):
//   out[n] = Σ_k act[k] * w[n,k]
//   w[n,k] = fp4_decode(nibble_k) * scale_e4m3[n, k/16] * alpha
//   alpha  = 1 / weight_global_scale (uploaded by mistral35_load.rs)
//          = amax_tensor / 2688
// `fp4_decode()` returns the TRUE e2m1 value in [-6..+6], so no
// further normalization is needed. Matches the upstream Marlin /
// ModelOpt W4A16 NVFP4 contract.
//
// Layout:
//   act:        [K]                BF16
//   w_packed:   [N, K/2]           u8     low nibble = elem 2i, high = 2i+1
//   w_scale:    [N, K/16]          E4M3 (u8)
//   alpha_ptr:  scalar             f32   (already inverted: 1/gs_real)
//   out:        [N]                BF16
//
// Launch: grid = (N, 1, 1), block = (256, 1, 1).
// Each block computes one output row n. K must be multiple of 16
// (asserted by host). Each thread handles K/256 K-blocks of 16
// elements; tree-reduces the partial dot products via shared mem.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

extern "C" __global__ void __launch_bounds__(256)
mistral35_w4a16_gemv_bf16_kernel(
    __nv_bfloat16* __restrict__ out,             // [N]
    const uint8_t* __restrict__ w_packed,        // [N, K/2]
    const uint8_t* __restrict__ w_scale,         // [N, K/16] E4M3 bytes
    const float* __restrict__ w_global_scale,    // scalar (= 1/gs_real)
    const __nv_bfloat16* __restrict__ act,       // [K]
    int N,
    int K)
{
    const int n   = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    const int K16 = K >> 4;  // number of K-blocks of 16

    // Round-8 fix: removed the bogus `/FP4_MAX = /6` factor. The
    // CT W4A16 NVFP4 dequant contract (verified vs the upstream
    // ModelOpt W4A16 path landed in vllm 0.20.2 PR #41769 plus the
    // on-disk weight_global_scale = 12416 ≈ 2688/amax for layer-0
    // q_proj of /home/r00t/mistral-3.5) is:
    //    w = e2m1[nibble] * scale_block * (1.0 / gs_disk)
    //      = e2m1[nibble] * scale_block * (amax / 2688)
    // `fp4_decode()` already returns the TRUE e2m1 value in [-6..+6],
    // so the prior `* (1/6)` factor was 6× too small. Without this
    // fix every NVFP4 weight collapses 6×, q/k both scale 1/6, qk^T
    // scores scale 1/36 → softmax goes flat → mode-locked attractor
    // (`\n\n\n` / `'aimerais`) regardless of prompt. See
    // v3/tools/mistral35_marlin_oracle_check.py.
    const float alpha = *w_global_scale;
    const float combined_a = alpha;

    // Accumulate dot product: each thread strides over K-blocks.
    float acc = 0.0f;
    const uint8_t* w_packed_row = w_packed + (size_t)n * (K / 2);
    const uint8_t* w_scale_row  = w_scale  + (size_t)n * K16;

    for (int kb = tid; kb < K16; kb += 256) {
        // Fetch this K-block's E4M3 scale byte.
        const __nv_fp8_e4m3 e4m3 =
            *reinterpret_cast<const __nv_fp8_e4m3*>(w_scale_row + kb);
        const float combined = float(e4m3) * combined_a;

        // 16 elements per K-block = 8 packed bytes.
        const int k_start = kb * 16;
        const uint8_t* bytes = w_packed_row + (k_start >> 1);

        // Unrolled 16-element dot.
        float local = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t byte = bytes[i];
            const float w_lo =
                rvllm_nvfp4::fp4_decode(byte & 0xFu) * combined;
            const float w_hi =
                rvllm_nvfp4::fp4_decode((byte >> 4) & 0xFu) * combined;
            const float a_lo =
                __bfloat162float(act[k_start + 2 * i]);
            const float a_hi =
                __bfloat162float(act[k_start + 2 * i + 1]);
            local += w_lo * a_lo + w_hi * a_hi;
        }
        acc += local;
    }

    // Block-wide reduction over 256 threads.
    __shared__ float smem[256];
    smem[tid] = acc;
    __syncthreads();

    // Tree reduction: 256 → 128 → 64 → 32 → warp reduction.
    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = smem[tid];
        // Warp reduction via shfl.
        for (int s = 16; s > 0; s >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, s);
        }
        if (tid == 0) {
            out[n] = __float2bfloat16_rn(v);
        }
    }
}
