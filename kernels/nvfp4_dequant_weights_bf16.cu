// =============================================================
//  nvfp4_dequant_weights_bf16.cu
// =============================================================
//
// Dequantize a Mistral 3.5 / LLMCompressor "nvfp4-pack-quantized"
// weight tensor into BF16. The checkpoint format is W4A16: only
// weights are quantized to NVFP4 (E2M1 nibble + per-16-element
// E4M3 block scale + per-tensor F32 global scale); activations
// stay BF16. Intended GEMM path: plain `bf16_gemm_f32` after
// dequant, not a tensor-core NVFP4 GEMM (which would require an
// activation quant pass and loses ~12% in the E4M3 denormal
// range that hidden-state amaxes land in).
//
// Layout:
//   weight_packed     [N, K/2]   u8     low nibble = elem 2i, high = 2i+1
//   weight_scale      [N, K/16]  e4m3   row-major
//   weight_global_scale            f32  scalar (encode scale, 448/amax)
//
// Output:
//   w_bf16            [N, K]     bf16   row-major
//
// Per-element decode:
//   w[n, k] = fp4_decode(nibble) * e4m3_scale[n, k/16] / global_scale
//
// Grid (k/256, n), block 256. Each thread handles one element;
// reads its own 1-byte SFA per K-block (16 threads in the same
// K-block alias-read the same byte — L1 caches it). global_scale
// inverted once per thread, redundantly, but the read is also L1-
// cached. Avoiding shfl keeps launch overhead at o(N*K/256)
// blocks, which is ~3 orders of magnitude fewer than the per-
// 16-element variant for large weights (gate/up/down).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

extern "C" __global__ void __launch_bounds__(256)
nvfp4_dequant_weights_bf16_kernel(
    const uint8_t* __restrict__ w_packed,    // [N, K/2]
    const uint8_t* __restrict__ w_scale,     // [N, K/16]  E4M3 bytes
    const float* __restrict__ w_global_scale,  // scalar
    __nv_bfloat16* __restrict__ out_bf16,    // [N, K]
    int N,
    int K)
{
    const int row = blockIdx.y;
    if (row >= N) return;
    const int kpos = blockIdx.x * 256 + threadIdx.x;
    if (kpos >= K) return;

    const int kb = kpos >> 4;  // /16

    // Per-thread reads — L1 cache absorbs the redundant fetches
    // within the 16-thread K-block group.
    const __nv_fp8_e4m3 e4m3 =
        *reinterpret_cast<const __nv_fp8_e4m3*>(
            w_scale + row * (K / 16) + kb);
    const float scale_f = float(e4m3);

    // Round-8 fix: removed the bogus `/FP4_MAX = /6` factor.
    //
    // The CT (compressed-tensors) "nvfp4-pack-quantized" on-disk
    // convention stores `weight_global_scale = 2688/amax_tensor`
    // (verified for /home/r00t/mistral-3.5 layer-0 q_proj: 12416.0,
    // implying amax = 0.21649). The loader reciprocates once
    // (`alpha = 1.0 / gs_disk = amax/2688`). Dequant is therefore:
    //   w = e2m1[nibble] * scale_e4m3 * alpha
    //     = e2m1[nibble] * scale_e4m3 * (amax/2688)
    //
    // matching the upstream Marlin / ModelOpt W4A16 contract
    // (vllm 0.20.2 PR #41769). The historical comment here claimed
    // an extra `/6` was needed; that was a self-consistent
    // hallucination — the fused GEMV had the same `/6` so the two
    // paths agreed with each other while both were 6× too small
    // vs every external W4A16 reference.
    //
    // See v3/tools/mistral35_marlin_oracle_check.py:
    //   HYP A (no /6):  cos=0.999999  rms_ratio=6.003 vs old rvllm dump
    //   HYP B (with /6) cos=0.999999  rms_ratio=1.000 (= old buggy path)
    const float alpha = *w_global_scale;
    const float combined = scale_f * alpha;

    const uint8_t byte = w_packed[row * (K / 2) + (kpos >> 1)];
    const uint32_t nibble = (kpos & 1) ? (byte >> 4) : (byte & 0xFu);
    const float v = rvllm_nvfp4::fp4_decode(nibble) * combined;

    out_bf16[row * K + kpos] = __float2bfloat16_rn(v);
}
