// =============================================================
//  cutlass_nvfp4_gemm_sm120.cu — Mistral 3.5 NVFP4 GEMM (sm_121a)
// =============================================================
//
// Ported from CUTLASS 4.4.2 example
//   examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
// The example demonstrates a blockscaled NVFP4 × NVFP4 → BF16 GEMM
// using the new Block Scaled Tensor Core MMA instructions on
// SM120/SM121 (sm_120a / sm_121a). We keep the same template config
// and re-expose it as five extern-C entry points that match the
// Rust ABI declared in
// `v3/crates/rvllm-cutlass/src/lib_so.rs`:
//
//   cutlass_nvfp4_gemm_sm120              — batched prefill (any M)
//   cutlass_nvfp4_gemm_sm120_decode_m1    — alias today (no m=1
//                                            specialisation yet)
//   cutlass_nvfp4_gemm_sm120_workspace    — workspace bytes query
//   cutlass_nvfp4_gemm_sm120_sfa_bytes    — activation scale staging size
//   cutlass_nvfp4_gemm_sm120_prep_sfa     — BF16/F16 → NVFP4 + SFA prep
//                                            (NOT YET IMPLEMENTED — stub
//                                            returns -100 so callers see
//                                            a clean "missing" rather than
//                                            silent NaN)
//
// The Mistral checkpoint ships weights pre-quantized:
//   weight_packed       U8     [N, K/2]
//   weight_scale        E4M3   [N, K/16]
//   weight_global_scale F32    [1]
// which maps directly to CUTLASS's `nv_float4_t<float_e2m1_t>` B
// operand + the `Sm1xxBlkScaledConfig` SFB layout. The global F32
// scale is folded into the epilogue's `alpha`.
//
// Activations enter as `nv_float4_t<float_e2m1_t>` A + an `[m, K/16]`
// E4M3 SFA tensor. The runtime is responsible for pre-quantizing
// BF16/F16 activations into this format via prep_sfa (the kernel
// itself is the next milestone — a per-token absmax → E4M3 SFA
// + nibble-pack pass mirroring the existing
// `fused_rope_partial_nvfp4kv` quantization step on the KV side).
//
// IMPORTANT: this source is NOT yet listed in
// `kernels/build_cutlass_sm120_so.sh::SOURCES` (kept off the build
// path until a manual rebuild + cosine validation lands). Adding
// it there is the gate that flips `CutlassBackend::nvfp4_active()`
// to true at startup.

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// --- Operand types (mirror CUTLASS example 79a) ---------------------
//
// nv_float4_t<float_e2m1_t> is CUTLASS's NVFP4 operand abstraction:
// the underlying packed representation is two e2m1 values per byte,
// with a per-16-K-block E4M3 microscale. The `ScaleFactorType` is
// resolved by the type itself (`ElementA::ScaleFactorType` ==
// E4M3), so the SFA / SFB tensors line up with the Mistral
// checkpoint's `weight_scale` byte stream without further
// re-staging.
using ElementA           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD           = cutlass::bfloat16_t;     // BF16 output (matches Mistral runtime)
using ElementC           = cutlass::bfloat16_t;     // unused (no bias)
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;                                              // packed
constexpr int AlignmentB = 32;                                              // packed
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;     // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;     // 8

using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

#ifndef RVLLM_NVFP4_TILE_M
#  define RVLLM_NVFP4_TILE_M 128
#endif
#ifndef RVLLM_NVFP4_TILE_N
#  define RVLLM_NVFP4_TILE_N 128
#endif
#ifndef RVLLM_NVFP4_TILE_K
#  define RVLLM_NVFP4_TILE_K 128
#endif
using ThreadBlockShape =
    Shape<cute::Int<RVLLM_NVFP4_TILE_M>,
          cute::Int<RVLLM_NVFP4_TILE_N>,
          cute::Int<RVLLM_NVFP4_TILE_K>>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
  >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// CUTLASS-derived block-scaled config — drives the SFA/SFB physical
// layouts that match the [N, K/16] / [M, K/16] E4M3 tensors the
// kernel consumes. No hand-rolled layout arithmetic; we read it
// from the type itself so future CUTLASS bumps stay correct.
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA =
    typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB =
    typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

extern "C" {

// -------------------------------------------------------------------
// Main entry: NVFP4 × NVFP4 → BF16 GEMM. The global F32 scale is
// folded into the epilogue alpha (multiplied with whatever per-row
// alpha the caller wants on top, default 1.0).
// -------------------------------------------------------------------
int cutlass_nvfp4_gemm_sm120(
    void* output,                          // BF16 [m, n], row-major
    const void* a_fp8,                     // NVFP4-packed [m, k/2]   (renamed for ABI compat)
    const void* b_packed,                  // NVFP4-packed [n, k/2]
    const void* sfa,                       // E4M3 [m, k/16]          (CUTLASS interleaved)
    const void* b_scale_e4m3,              // E4M3 [n, k/16]          (CUTLASS interleaved)
    const void* b_global_scale_f32,        // F32 [1]
    int m, int n, int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream)
{
    // Per the integration spec, the global scale folds into alpha.
    // We host-copy 4 bytes once per call — negligible vs the GEMM.
    float alpha_host = 1.0f;
    if (b_global_scale_f32 != nullptr) {
        cudaError_t e = cudaMemcpyAsync(
            &alpha_host, b_global_scale_f32, sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) return -10;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return -11;
    }

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    auto layout_SFA =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        { // mainloop
            reinterpret_cast<const typename ElementA::DataType*>(a_fp8), stride_A,
            reinterpret_cast<const typename ElementB::DataType*>(b_packed), stride_B,
            reinterpret_cast<const typename ElementA::ScaleFactorType*>(sfa), layout_SFA,
            reinterpret_cast<const typename ElementB::ScaleFactorType*>(b_scale_e4m3), layout_SFB,
        },
        { // epilogue
            {alpha_host, 0.0f},
            nullptr, stride_C,                                       // no C
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    return 0;
}

// m=1 decode specialisation. Today this is a thin alias on the
// general kernel — CUTLASS's tile scheduler degrades to a single
// tile for m=1 anyway. A purpose-built m=1 kernel is a follow-up
// (smaller register footprint + skipping the reduction across
// MmaTileM); the alias keeps the Rust ABI honest until then.
int cutlass_nvfp4_gemm_sm120_decode_m1(
    void* output,
    const void* a_fp8,
    const void* b_packed,
    const void* sfa,
    const void* b_scale_e4m3,
    const void* b_global_scale_f32,
    int m, int n, int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream)
{
    return cutlass_nvfp4_gemm_sm120(
        output, a_fp8, b_packed, sfa, b_scale_e4m3, b_global_scale_f32,
        m, n, k, workspace, workspace_size, stream);
}

// -------------------------------------------------------------------
// Workspace size query.
// -------------------------------------------------------------------
size_t cutlass_nvfp4_gemm_sm120_workspace(int m, int n, int k) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
    auto layout_SFA =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        { nullptr, stride_A, nullptr, stride_B,
          nullptr, layout_SFA, nullptr, layout_SFB },
        { {1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D }
    };

    return Gemm::get_workspace_size(args);
}

// SFA scratch bytes for an `[m, K/16]` E4M3 tensor in the CUTLASS
// interleaved layout. We compute the size from the layout helper
// (which accounts for the swizzling) rather than naively
// `m * (k/16) * 1 byte` — the interleaved layout has alignment
// padding the naïve formula doesn't.
size_t cutlass_nvfp4_gemm_sm120_sfa_bytes(int m, int k) {
    // A reasonable n is needed for the layout helper but doesn't
    // affect the SFA size — pass m (any value works, helper only
    // reads the (m, k) dims that drive SFA).
    int n_dummy = (m > 0) ? m : 1;
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n_dummy, k, 1));
    // E4M3 = 1 byte per element. `size(filter_zeros(layout))` gives
    // the live-element count in the interleaved layout (matches
    // what the example's `block_SFA.reset(make_Coord(size(...)))`
    // allocates).
    return static_cast<size_t>(size(filter_zeros(layout_SFA)));
}

// Activation prep (BF16/F16 → NVFP4 packed + per-block E4M3 SFA).
// Not yet implemented — returns -100 so the runtime sees a clean
// "missing entry" path until the per-token absmax + nibble pack
// kernel lands. The Rust wrapper already maps -100 to a typed
// `Mistral35Error::Nvfp4SymbolsMissing`-equivalent.
int cutlass_nvfp4_gemm_sm120_prep_sfa(
    const void*, void*, void*,
    int, int, int,
    cudaStream_t)
{
    return -100;
}

} // extern "C"

#else  // sm < 120

extern "C" {

int cutlass_nvfp4_gemm_sm120(
    void*, const void*, const void*, const void*, const void*, const void*,
    int, int, int, void*, size_t, cudaStream_t)
{
    return -100;
}

int cutlass_nvfp4_gemm_sm120_decode_m1(
    void*, const void*, const void*, const void*, const void*, const void*,
    int, int, int, void*, size_t, cudaStream_t)
{
    return -100;
}

size_t cutlass_nvfp4_gemm_sm120_workspace(int, int, int) { return 0; }
size_t cutlass_nvfp4_gemm_sm120_sfa_bytes(int, int)      { return 0; }

int cutlass_nvfp4_gemm_sm120_prep_sfa(
    const void*, void*, void*, int, int, int, cudaStream_t)
{
    return -100;
}

} // extern "C"

#endif  // SM120/SM121
