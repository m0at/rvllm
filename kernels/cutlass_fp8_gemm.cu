// CUTLASS 3.x SM90 FP8 GEMM kernel for rvLLM.
//
// Computes: D[m,n] = cast_to_f16(A_scale[m] * B_scale[0] * sum_k(A_fp8[m,k] * B_fp8[k,n]))
//
// Uses CUTLASS 3.x EVT (Epilogue Visitor Tree) to fuse per-row A scaling
// and per-tensor B scaling into the GEMM epilogue. No post-kernel needed.
//
// Build: compiled as part of libcutlass_kernels.so via build_cutlass_so.sh

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>

using namespace cute;

// ============================================================================
// SM90 Hopper-optimized FP8 GEMM with fused per-row/per-tensor scaling
// ============================================================================

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = cutlass::half_t;
using ElementAccum = float;
using ElementCompute = float;
using ElementScalar = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

// Tile shape: 128x128x128 for FP8 on SM90
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

// ============================================================================
// EVT epilogue: D[m,n] = cast<f16>(row_scale[m] * col_scale * Acc[m,n])
//
// Tree:  multiply(RowBroadcast(A_scale), multiply(ScalarBroadcast(B_scale), AccFetch))
// ============================================================================

// Inner node: col_scale * Acc
using ScaledAcc = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,
    cutlass::epilogue::fusion::Sm90AccFetch
>;

// Outer node: row_scale[m] * (col_scale * Acc)
using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, ElementScalar>,
    ScaledAcc
>;

// Build epilogue with EVT -- no C input (void)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    void, LayoutD, 8,
    ElementD, LayoutD, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueEVT
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,
    ElementB, LayoutB, 16,
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

extern "C" {

int cutlass_fp8_gemm(
    void* output,           // [M, N] f16
    const void* a,          // [M, K] fp8_e4m3
    const void* b,          // [N, K] fp8_e4m3
    const void* a_scales,   // [M] f32 per-row scales (device ptr)
    const void* b_scale,    // [1] f32 per-tensor scale (device ptr)
    int M, int N, int K,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
        },
        {   // epilogue args
            {},  // thread args -- filled below
            nullptr, {}, // no C
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };

    // EVT thread args: {first_child, ..., last_child, op}
    // Tree: multiply(RowBroadcast(A_scale), multiply(ScalarBroadcast(B_scale), AccFetch))
    //
    // RowBroadcast Args: {ptr_aux, null_default, dAux}
    // ScalarBroadcast Args: {scalars[1], scalar_ptrs[1], dScalar[1]}
    // Compute Args: {} (no extra params)
    // AccFetch Args: {} (no extra params)
    args.epilogue.thread = {
        // RowBroadcast(A_scale) -- per-row device pointer
        {reinterpret_cast<const ElementScalar*>(a_scales), ElementScalar(0), {}},
        // multiply(ScalarBroadcast(B_scale), AccFetch)
        {
            // ScalarBroadcast: use device pointer (no DtoH sync)
            // Args: {scalars[1], scalar_ptrs[1], dScalar[1]}
            {{ElementScalar(0)}, {reinterpret_cast<const ElementScalar*>(b_scale)}, {}},
            // AccFetch
            {},
            // multiply op
            {}
        },
        // outer multiply op
        {}
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

size_t cutlass_fp8_gemm_workspace_size(int M, int N, int K) {
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{}, nullptr, {}, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

} // extern "C"

// ============================================================================
// SM90 FP8 GEMM with small tile for decode (M <= 64)
//
// For small batch sizes, tile_M=128 wastes 50%+ utilization. This variant
// uses tile 64x128x256 (matching vLLM) for full tile utilization at M<=64.
// Same NT layout, same EVT epilogue, just smaller M-tile and deeper K-tile.
// ============================================================================

using SmallTileShape = Shape<_64, _128, _128>;
using SmallClusterShape = Shape<_1, _1, _1>;

// EVT: same tree as the main kernel
using SmallScaledAcc = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,
    cutlass::epilogue::fusion::Sm90AccFetch
>;

using SmallEpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, SmallTileShape, ElementScalar>,
    SmallScaledAcc
>;

using SmallCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    SmallTileShape, SmallClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    void, LayoutD, 8,
    ElementD, LayoutD, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    SmallEpilogueEVT
>::CollectiveOp;

using SmallCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,
    ElementB, LayoutB, 16,
    ElementAccum,
    SmallTileShape,
    SmallClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename SmallCollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

using SmallGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    SmallCollectiveMainloop,
    SmallCollectiveEpilogue
>;

using SmallGemm = cutlass::gemm::device::GemmUniversalAdapter<SmallGemmKernel>;

extern "C" {

int cutlass_fp8_gemm_small(
    void* output,
    const void* a,
    const void* b,
    const void* a_scales,
    const void* b_scale,
    int M, int N, int K,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename SmallGemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename SmallGemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename SmallGemm::GemmKernel::StrideD{}, {M, N, 1});

    typename SmallGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
        },
        {
            {},
            nullptr, {},
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };

    // EVT: RowBroadcast(a_scales) * (ScalarBroadcast(b_scale) * Acc)
    args.epilogue.thread = {
        {reinterpret_cast<const ElementScalar*>(a_scales), ElementScalar(0), {}},
        {
            {{ElementScalar(0)}, {reinterpret_cast<const ElementScalar*>(b_scale)}, {}},
            {},
            {}
        },
        {}
    };

    SmallGemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    return 0;
}

size_t cutlass_fp8_gemm_small_workspace_size(int M, int N, int K) {
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename SmallGemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename SmallGemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename SmallGemm::GemmKernel::StrideD{}, {M, N, 1});

    typename SmallGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{}, nullptr, {}, nullptr, stride_D}
    };

    SmallGemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

} // extern "C"
