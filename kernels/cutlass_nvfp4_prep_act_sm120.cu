// =============================================================
//  cutlass_nvfp4_prep_act_sm120.cu — Mistral 3.5 activation prep
// =============================================================
//
// Quantises BF16 / F16 activation tensors `[m, k]` to NVFP4 packed
// bytes `[m, k/2]` plus per-(token, K-block-of-16) E4M3 scales.
// Output format matches the contract that
// `cutlass_nvfp4_gemm_sm120` consumes:
//
//   a_packed[m, k/2]     U8       low-nibble = elem 2i, high = 2i+1
//   sfa_natural[m, k/16] E4M3     row-major (NOT yet CUTLASS-interleaved
//                                 — see "Layout caveat" below)
//
// Each (row, k_block) pair runs in one CUDA block of 16 threads.
// Lane-wide absmax via `__shfl_xor` over the 16-lane subwarp;
// scale is `peak / 6.0` rounded to E4M3, inv-scale is then
// applied to each lane's input value before nibble encoding via
// `fp4_encode` (`kernels/nvfp4_utils.cuh`). Bytes are formed by
// pairing lanes (2t, 2t+1); lanes 0..7 each write one packed byte
// out of the eight per block.
//
// Layout caveat:
//   The natural row-major SFA layout this kernel writes is NOT the
//   CUTLASS Sm120 NVFP4 interleaved SFA layout that the GEMM
//   wrapper requests via `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA`.
//   The interleaved layout encodes a tensor-core-friendly
//   bank-swizzle. Production wiring needs a second kernel that
//   reformats `sfa_natural` -> `sfa_cutlass`, OR (preferable) this
//   kernel learns to write directly into the interleaved layout
//   using `cute::Layout` machinery. Both options are mechanical
//   given the cute layout type from
//   `Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig`;
//   neither blocks the rest of the bring-up, so the natural-layout
//   intermediate lands first to enable a numpy-comparable reference
//   round-trip.
//
// Once the layout transform lands, the entry point name stays the
// same and the Rust wrapper (`launch_nvfp4_prep_sfa`) flips from
// rc=-100 (the GEMM-side stub) to delegating here.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"  // fp4_encode, block_scale_e4m3

using rvllm_nvfp4::fp4_encode;

namespace {

// One CUDA block per (row, k_block-of-16). Block dim = 16 lanes.
// Caller must pass a kernel grid of (k/16, m, 1) — k must be a
// multiple of 16 (asserted at the host).
//
// dtype: 0 = F16, 1 = BF16. Other values rejected at the host.
template <int DTYPE>
__global__ void nvfp4_prep_act_kernel(
    const void* __restrict__ a_input,    // [m, k] F16 or BF16
    uint8_t*    __restrict__ a_packed,   // [m, k/2] U8
    __nv_fp8_e4m3* __restrict__ sfa_natural, // [m, k/16] E4M3 row-major
    int m,
    int k)
{
    const int kb  = blockIdx.x;          // K-block index in this row
    const int row = blockIdx.y;
    const int t   = threadIdx.x;         // 0..15

    if (row >= m) return;
    const int kpos = kb * 16 + t;
    // k is asserted divisible by 16 host-side; the bound here
    // protects ragged trailing partial blocks if a future caller
    // relaxes the divisibility.
    if (kpos >= k) return;

    // Load one input element as f32.
    float v;
    if constexpr (DTYPE == 0) {
        const __half* p = reinterpret_cast<const __half*>(a_input);
        v = __half2float(p[row * k + kpos]);
    } else {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(a_input);
        v = __bfloat162float(p[row * k + kpos]);
    }

    // Block-wide absmax via butterfly shuffle on the 16-lane subwarp.
    // The active mask is the low 16 bits of the warp; we operate
    // entirely within that range so the higher 16 lanes (if any)
    // don't participate. With blockDim.x == 16 this is exactly the
    // first half of one warp.
    const unsigned mask = 0x0000FFFFu;
    float am = fabsf(v);
    am = fmaxf(am, __shfl_xor_sync(mask, am, 8));
    am = fmaxf(am, __shfl_xor_sync(mask, am, 4));
    am = fmaxf(am, __shfl_xor_sync(mask, am, 2));
    am = fmaxf(am, __shfl_xor_sync(mask, am, 1));

    // All 16 lanes now hold the block absmax. Form the E4M3 scale +
    // its inverse for the per-lane nibble encode.
    const __nv_fp8_e4m3 scale = __nv_fp8_e4m3((am > 0.0f) ? am / 6.0f : 0.0f);
    const float scale_f       = float(scale);
    const float inv_scale     = (scale_f > 0.0f) ? 1.0f / scale_f : 0.0f;

    const uint32_t bits = fp4_encode(v * inv_scale);

    // Pair lanes: byte t in [0..7] holds (lane 2t+1) << 4 | lane 2t.
    if (t < 8) {
        const uint32_t lo = __shfl_sync(mask, bits, 2 * t);
        const uint32_t hi = __shfl_sync(mask, bits, 2 * t + 1);
        const uint8_t byte = static_cast<uint8_t>(((hi & 0xFu) << 4) | (lo & 0xFu));
        // a_packed has k/2 bytes per row. Each k_block produces 8 bytes.
        a_packed[row * (k / 2) + kb * 8 + t] = byte;
    }
    if (t == 0) {
        sfa_natural[row * (k / 16) + kb] = scale;
    }
}

} // anonymous namespace

extern "C" {

/// Prep kernel entry. See header above for layout semantics.
///
/// Returns 0 on success; -1 on bad shape; -2 on bad dtype; -3 on
/// kernel launch failure. Negative values map cleanly to the Rust
/// `launch_nvfp4_prep_sfa` error path.
///
/// `a_input_dtype`: 0 = F16, 1 = BF16. Anything else returns -2
/// (rejected at the host).
int cutlass_nvfp4_gemm_sm120_prep_act(
    const void* a_input,
    void*       a_packed,
    void*       sfa_natural,
    int m,
    int k,
    int a_input_dtype,
    cudaStream_t stream)
{
    if (m <= 0 || k <= 0)         return -1;
    if (k % 16 != 0)              return -1;  // per spec: k mod 16 == 0
    if (a_input == nullptr)       return -1;
    if (a_packed == nullptr)      return -1;
    if (sfa_natural == nullptr)   return -1;

    const dim3 grid(k / 16, m, 1);
    const dim3 block(16, 1, 1);

    if (a_input_dtype == 0) {
        nvfp4_prep_act_kernel<0><<<grid, block, 0, stream>>>(
            a_input,
            reinterpret_cast<uint8_t*>(a_packed),
            reinterpret_cast<__nv_fp8_e4m3*>(sfa_natural),
            m, k);
    } else if (a_input_dtype == 1) {
        nvfp4_prep_act_kernel<1><<<grid, block, 0, stream>>>(
            a_input,
            reinterpret_cast<uint8_t*>(a_packed),
            reinterpret_cast<__nv_fp8_e4m3*>(sfa_natural),
            m, k);
    } else {
        return -2;
    }

    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}

/// Natural-layout SFA bytes — one E4M3 byte per (row, k_block).
/// The CUTLASS-interleaved variant lives behind
/// `cutlass_nvfp4_gemm_sm120_sfa_bytes` (in cutlass_nvfp4_gemm_sm120.cu)
/// and may be larger because of the layout's swizzle padding.
size_t cutlass_nvfp4_gemm_sm120_prep_act_sfa_bytes(int m, int k) {
    if (m <= 0 || k <= 0 || (k % 16) != 0) return 0;
    return static_cast<size_t>(m) * static_cast<size_t>(k / 16);
}

} // extern "C"
