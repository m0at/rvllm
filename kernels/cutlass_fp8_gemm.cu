// CUTLASS 3.x SM90 FP8 GEMM kernel for rvLLM.
//
// Computes: D[m,n] = cast_to_f16(A_scale[m] * B_scale[0] * sum_k(A_fp8[m,k] * B_fp8[k,n]))
//
// Uses CUTLASS 3.x collective builder with Hopper-specific WGMMA + TMA.
// FP8 GEMM runs with alpha=1, beta=0, then a small post-kernel applies
// per-row A scales and per-tensor B scale.
//
// Build: compiled as part of libcutlass_kernels.so via build_cutlass_so.sh

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>

using namespace cute;

// ============================================================================
// SM90 Hopper-optimized FP8 GEMM using CUTLASS 3.x collective builder
// ============================================================================

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccum = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Tile shape: 128x128x128 for FP8 on SM90 (K=128 because FP8 elements are half the size of F16)
using TileShape = Shape<_128, _128, _128>;

// Cluster shape: 1x1x1 (single SM)
using ClusterShape = Shape<_1, _1, _1>;

// Build the collective GEMM for SM90
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,   // alignment A: 16 bytes = 16 fp8 elements
    ElementB, LayoutB, 16,   // alignment B: 16 bytes = 16 fp8 elements
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccum, ElementCompute,
            ElementC, LayoutC, 8,
            ElementD, LayoutD, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementC, LayoutC, 8,
    ElementD, LayoutD, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ============================================================================
// Post-GEMM kernel: apply per-row A scales and per-tensor B scale
// ============================================================================

__global__ void apply_fp8_scales_kernel(
    __half* output, const float* row_scales, const float* col_scale,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx / N;
        float val = __half2float(output[idx]) * row_scales[row] * col_scale[0];
        val = fmaxf(-65504.0f, fminf(65504.0f, val));
        output[idx] = __float2half_rn(val);
    }
}

extern "C" {

int cutlass_fp8_gemm(
    void* output,           // [M, N] f16
    const void* a,          // [M, K] fp8_e4m3
    const void* b,          // [N, K] fp8_e4m3 (row-major, treated as col-major B^T)
    const void* a_scales,   // [M] f32 per-row scales
    const void* b_scale,    // [1] f32 per-tensor scale
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
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(output), stride_C,
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

    // Apply per-row A scales and per-tensor B scale
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_fp8_scales_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(output),
        reinterpret_cast<const float*>(a_scales),
        reinterpret_cast<const float*>(b_scale),
        M, N
    );

    return 0;
}

size_t cutlass_fp8_gemm_workspace_size(int M, int N, int K) {
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

} // extern "C"
