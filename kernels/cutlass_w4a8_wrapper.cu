// Thin C-linkage wrapper around CUTLASS 4.x example 55 (hopper int4 x fp8 gemm).
//
// Exposes one entry point: rvllm_w4a8_gemm_run(...) that runs
//   D (f16)  =  alpha * A_fp8 * B_int4_quant  +  beta * C (f16)
// where B is int4 with per-group FP8 scales (group_size = 128) packed as
// cutlass::Array<e4m3, 8> (LUT trick from example 55).
//
// A is RowMajor [M, K] FP8 E4M3.
// B is ColMajor [K, N] int4 two's complement, AWQ-reordered offline.
// Scales are MN-major [N, K/group_size] as packed e4m3x8 LUT blocks.
// D is RowMajor [M, N] fp16.
//
// We swap A <-> B and transpose inside CUTLASS to keep the narrow type in
// registers AND use TMA epilogues (example 55's approach). Caller still
// sees the logical A*B^T semantics.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#error "SM90 MMA required"
#endif

// =========================================================================
// Types (match example 55 int4_fp8 config)
// =========================================================================
using MmaType    = cutlass::float_e4m3_t;
using QuantType  = cutlass::int4b_t;

constexpr int kGroupSize = 128;

// Tile: 128 M x 128 N x 128 K (one WGMMA-friendly tile)
constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value; // = 128 for FP8

using ElementA       = MmaType;
using LayoutA        = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB       = QuantType;
using LayoutB        = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

using LayoutAtomQuant    = decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered  = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,int>, StrideB>{}));

using ElementScale = MmaType; // scales are FP8 E4M3 (example 55 convention)
using LayoutScale  = cutlass::layout::RowMajor;

using ElementC       = cutlass::half_t;
using LayoutC        = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD       = ElementC;
using LayoutD        = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

using TileShape      = Shape<_128, _128, cute::Int<TileShapeK>>;
using ClusterShape   = Shape<_1, _1, _1>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSched  = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileT  = cutlass::epilogue::collective::EpilogueTileAuto;

// Epilogue: linear-combo D = alpha * accum + beta * C (transposed layout
// for the explicit-swap convention).
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileT,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSched
>::CollectiveOp;

// Mainloop: INT4 (B, swapped to operand A internally) with packed e4m3 LUT
// scales (Array<e4m3, 8>). B is shuffle-reordered for contiguous thread reads.
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>, LayoutB_Reordered, AlignmentB,
    ElementA, LayoutA_T, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// =========================================================================
// C ABI entry point
// =========================================================================
extern "C" int rvllm_w4a8_gemm_run(
    const void* a_fp8,              // [M, K] RowMajor E4M3
    const void* b_int4_reordered,   // [K, N] INT4 ColMajor, AWQ-shuffled offline
    const void* b_scales_packed,    // [N, K/group_size] as Array<e4m3, 8> LUT blocks
    const void* c_f16,              // [M, N] RowMajor (may be nullptr; used only if beta != 0)
    void*       d_f16,              // [M, N] RowMajor output
    int         m,
    int         n,
    int         k,
    int         group_size,         // must equal kGroupSize (128)
    float       alpha,
    float       beta,
    void*       workspace,
    size_t      workspace_bytes,
    cudaStream_t stream
) {
    if (group_size != kGroupSize) {
        fprintf(stderr, "rvllm_w4a8: group_size must be %d, got %d\n", kGroupSize, group_size);
        return -1;
    }
    if (k % kGroupSize != 0) {
        fprintf(stderr, "rvllm_w4a8: K (%d) must be divisible by group_size (%d)\n", k, kGroupSize);
        return -2;
    }

    const int scale_k = k / kGroupSize;

    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;

    // Explicit swap+transpose: CUTLASS call is (B, A) -> D^T.
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, 1));

    // Reordered B layout: N-major tiles with the 8-elem-per-thread atom.
    Layout layout_B_reordered = tile_to_shape(
        LayoutAtomQuant{},
        cute::make_shape(n, k, 1),
        cute::conditional_t<::cutlass::gemm::detail::is_k_major<LayoutB>(),
            Step<_2,_1,_3>, Step<_1,_2,_3>>{}
    );

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        { n, m, k, /*batch*/ 1 },
        {
            reinterpret_cast<const ElementB*>(b_int4_reordered), layout_B_reordered,
            reinterpret_cast<const ElementA*>(a_fp8), stride_A,
            reinterpret_cast<const cutlass::Array<ElementScale, 8>*>(b_scales_packed), stride_S,
            kGroupSize,
        },
        {
            { alpha, beta },
            reinterpret_cast<const ElementC*>(c_f16), stride_C,
            reinterpret_cast<ElementD*>(d_f16), stride_D,
        }
    };

    Gemm gemm;
    cutlass::Status s = gemm.can_implement(arguments);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "rvllm_w4a8: can_implement failed: %d\n", (int)s);
        return -10;
    }
    size_t need = Gemm::get_workspace_size(arguments);
    if (need > workspace_bytes) {
        fprintf(stderr, "rvllm_w4a8: workspace too small: need %zu, got %zu\n", need, workspace_bytes);
        return -11;
    }
    s = gemm.initialize(arguments, workspace, stream);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "rvllm_w4a8: initialize failed: %d\n", (int)s);
        return -12;
    }
    s = gemm.run(stream);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "rvllm_w4a8: run failed: %d\n", (int)s);
        return -13;
    }
    return 0;
}

// Workspace size probe.
extern "C" size_t rvllm_w4a8_gemm_workspace_size(int m, int n, int k) {
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;
    const int scale_k = k / kGroupSize;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, 1));

    Layout layout_B_reordered = tile_to_shape(
        LayoutAtomQuant{},
        cute::make_shape(n, k, 1),
        cute::conditional_t<::cutlass::gemm::detail::is_k_major<LayoutB>(),
            Step<_2,_1,_3>, Step<_1,_2,_3>>{}
    );

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        { n, m, k, 1 },
        {
            nullptr, layout_B_reordered,
            nullptr, stride_A,
            nullptr, stride_S,
            kGroupSize,
        },
        {
            { 1.0f, 0.0f },
            nullptr, stride_C,
            nullptr, stride_D,
        }
    };
    return Gemm::get_workspace_size(arguments);
}
