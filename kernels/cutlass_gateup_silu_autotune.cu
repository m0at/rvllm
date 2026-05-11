// CUTLASS 3.x SM90 autotuned GateUp GEMM + fused SiLU*Mul for rvLLM.
//
// 32 tile/cluster/schedule variants of the 2-kernel approach:
//   1. CUTLASS GEMM: temp[M, 2*I] = input[M, K] @ weight[2*I, K]^T
//   2. Fused SiLU*Mul: output[M, I] = SiLU(temp[:, 0:I]) * temp[:, I:2*I]

#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
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

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// ============================================================================
// Fused SiLU*Mul kernel: reads [M, 2*I], writes [M, I]
// Vectorized __half2 loads for 2x bandwidth efficiency.
// ============================================================================

static __global__ void silu_mul_fused_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ temp,
    int M,
    int intermediate_size
) {
    const int total = M * intermediate_size;
    const int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 >= total) return;

    const int token = idx2 / intermediate_size;
    const int elem  = idx2 % intermediate_size;
    const int row_width = intermediate_size * 2;
    const int row_base = token * row_width;

    if (elem + 1 < intermediate_size) {
        __half2 gate_h2 = *reinterpret_cast<const __half2*>(&temp[row_base + elem]);
        __half2 up_h2   = *reinterpret_cast<const __half2*>(&temp[row_base + intermediate_size + elem]);

        float g0 = __half2float(gate_h2.x);
        float g1 = __half2float(gate_h2.y);
        float u0 = __half2float(up_h2.x);
        float u1 = __half2float(up_h2.y);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));

        __half2 result;
        result.x = __float2half(s0 * u0);
        result.y = __float2half(s1 * u1);
        *reinterpret_cast<__half2*>(&output[token * intermediate_size + elem]) = result;
    } else {
        float gate = __half2float(temp[row_base + elem]);
        float up   = __half2float(temp[row_base + intermediate_size + elem]);
        float silu = gate / (1.0f + expf(-gate));
        output[token * intermediate_size + elem] = __float2half(silu * up);
    }
}

// ============================================================================
// Templated GEMM type + dispatch
// ============================================================================

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct GateUpGemmTypes {
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, 8,
        ElementC, LayoutC, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 8,
        ElementB, LayoutB, 8,
        ElementAccum,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
size_t cutlass_gateup_silu_ws_dispatch(int M, int N, int K) {
    using Types = GateUpGemmTypes<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename Types::Gemm;

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
    size_t gemm_ws = gemm_op.get_workspace_size(args);
    gemm_ws = (gemm_ws + 255) & ~255;

    size_t temp_bytes = (size_t)M * N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;

    return gemm_ws + temp_bytes;
}

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
int cutlass_gateup_silu_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K,
    void* workspace, size_t workspace_size, cudaStream_t stream
) {
    using Types = GateUpGemmTypes<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    // Find temp buffer offset
    typename Gemm::Arguments args_probe{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D}
    };
    Gemm gemm_probe;
    size_t gemm_ws = gemm_probe.get_workspace_size(args_probe);
    gemm_ws = (gemm_ws + 255) & ~255;

    // Bounds-check the caller's workspace BEFORE we start writing
    // through the `temp` pointer. Previously `workspace_size` was
    // accepted as a parameter and ignored, so a too-small buffer
    // produced silent OOB writes past the scratch region with no
    // error escape. The required size matches the formula the
    // sister `gateup_silu_workspace_size_*` function exports for
    // callers (gemm_ws + 256-aligned M*N*sizeof(half)). We refuse
    // to launch when the caller's buffer doesn't fit.
    size_t temp_bytes = (size_t)M * (size_t)N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;
    size_t needed = gemm_ws + temp_bytes;
    if (workspace_size < needed) {
        return -22 /* -EINVAL */;
    }

    char* ws_ptr = reinterpret_cast<char*>(workspace);
    void* gemm_workspace = ws_ptr;
    __half* temp = reinterpret_cast<__half*>(ws_ptr + gemm_ws);

    // Step 1: CUTLASS GEMM -> temp[M, N]
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(temp), stride_C,
            reinterpret_cast<ElementC*>(temp), stride_D,
        }
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, gemm_workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    // Step 2: Fused SiLU*Mul: temp[M, N] -> output[M, N/2]
    int intermediate_size = N / 2;
    int total = M * intermediate_size;
    int threads_needed = (total + 1) / 2;
    int block = 256;
    int grid = (threads_needed + block - 1) / block;

    silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output),
        temp,
        M,
        intermediate_size
    );

    return 0;
}

// ============================================================================
// SM count helper (cached)
// ============================================================================

static int get_sm_count() {
    static int sm_count = 0;
    if (sm_count == 0) {
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    }
    return sm_count;
}

// ============================================================================
// Staged types: explicit pipeline stage count instead of auto-carveout
// ============================================================================

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
struct GateUpStagedTypes {
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, 8,
        ElementC, LayoutC, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 8,
        ElementB, LayoutB, 8,
        ElementAccum,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCount<NumStages>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ============================================================================
// Staged dispatch + workspace
// ============================================================================

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
size_t cutlass_gateup_silu_staged_ws(int M, int N, int K) {
    using Types = GateUpStagedTypes<TileShape_, ClusterShape_, KernelSchedule_, NumStages>;
    using Gemm = typename Types::Gemm;

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
    size_t gemm_ws = gemm_op.get_workspace_size(args);
    gemm_ws = (gemm_ws + 255) & ~255;

    size_t temp_bytes = (size_t)M * N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;

    return gemm_ws + temp_bytes;
}

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
int cutlass_gateup_silu_staged_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K,
    void* workspace, size_t workspace_size, cudaStream_t stream
) {
    using Types = GateUpStagedTypes<TileShape_, ClusterShape_, KernelSchedule_, NumStages>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args_probe{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D}
    };
    Gemm gemm_probe;
    size_t gemm_ws = gemm_probe.get_workspace_size(args_probe);
    gemm_ws = (gemm_ws + 255) & ~255;

    // Bounds-check the caller's workspace BEFORE we start writing
    // through the `temp` pointer. Previously `workspace_size` was
    // accepted as a parameter and ignored, so a too-small buffer
    // produced silent OOB writes past the scratch region with no
    // error escape. The required size matches the formula the
    // sister `gateup_silu_workspace_size_*` function exports for
    // callers (gemm_ws + 256-aligned M*N*sizeof(half)). We refuse
    // to launch when the caller's buffer doesn't fit.
    size_t temp_bytes = (size_t)M * (size_t)N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;
    size_t needed = gemm_ws + temp_bytes;
    if (workspace_size < needed) {
        return -22 /* -EINVAL */;
    }

    char* ws_ptr = reinterpret_cast<char*>(workspace);
    void* gemm_workspace = ws_ptr;
    __half* temp = reinterpret_cast<__half*>(ws_ptr + gemm_ws);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(temp), stride_C,
            reinterpret_cast<ElementC*>(temp), stride_D,
        }
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, gemm_workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    int intermediate_size = N / 2;
    int total = M * intermediate_size;
    int threads_needed = (total + 1) / 2;
    int block = 256;
    int grid = (threads_needed + block - 1) / block;

    silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output),
        temp,
        M,
        intermediate_size
    );

    return 0;
}

// ============================================================================
// Swizzle dispatch + workspace
// ============================================================================

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int MaxSwizzle>
size_t cutlass_gateup_silu_swizzle_ws(int M, int N, int K) {
    using Types = GateUpGemmTypes<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info,
        {MaxSwizzle}
    };

    Gemm gemm_op;
    size_t gemm_ws = gemm_op.get_workspace_size(args);
    gemm_ws = (gemm_ws + 255) & ~255;

    size_t temp_bytes = (size_t)M * N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;

    return gemm_ws + temp_bytes;
}

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int MaxSwizzle>
int cutlass_gateup_silu_swizzle_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K,
    void* workspace, size_t workspace_size, cudaStream_t stream
) {
    using Types = GateUpGemmTypes<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::Arguments args_probe{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info,
        {MaxSwizzle}
    };
    Gemm gemm_probe;
    size_t gemm_ws = gemm_probe.get_workspace_size(args_probe);
    gemm_ws = (gemm_ws + 255) & ~255;

    // Bounds-check the caller's workspace BEFORE we start writing
    // through the `temp` pointer. Previously `workspace_size` was
    // accepted as a parameter and ignored, so a too-small buffer
    // produced silent OOB writes past the scratch region with no
    // error escape. The required size matches the formula the
    // sister `gateup_silu_workspace_size_*` function exports for
    // callers (gemm_ws + 256-aligned M*N*sizeof(half)). We refuse
    // to launch when the caller's buffer doesn't fit.
    size_t temp_bytes = (size_t)M * (size_t)N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;
    size_t needed = gemm_ws + temp_bytes;
    if (workspace_size < needed) {
        return -22 /* -EINVAL */;
    }

    char* ws_ptr = reinterpret_cast<char*>(workspace);
    void* gemm_workspace = ws_ptr;
    __half* temp = reinterpret_cast<__half*>(ws_ptr + gemm_ws);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(temp), stride_C,
            reinterpret_cast<ElementC*>(temp), stride_D,
        },
        hw_info,
        {MaxSwizzle}
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, gemm_workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    int intermediate_size = N / 2;
    int total = M * intermediate_size;
    int threads_needed = (total + 1) / 2;
    int block = 256;
    int grid = (threads_needed + block - 1) / block;

    silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output),
        temp,
        M,
        intermediate_size
    );

    return 0;
}

// ============================================================================
// Stream-K types: cooperative kernel with StreamKScheduler
// ============================================================================

template <typename TileShape_, typename ClusterShape_>
struct GateUpStreamKTypes {
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, 8,
        ElementC, LayoutC, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 8,
        ElementB, LayoutB, 8,
        ElementAccum,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::StreamKScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ============================================================================
// Split-K dispatch + workspace
// ============================================================================

template <typename TileShape_, typename ClusterShape_, int SplitK>
size_t cutlass_gateup_silu_splitk_ws(int M, int N, int K) {
    using Types = GateUpStreamKTypes<TileShape_, ClusterShape_>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::GemmKernel::TileSchedulerArguments sched_args;
    sched_args.splits = SplitK;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info,
        sched_args
    };

    Gemm gemm_op;
    size_t gemm_ws = gemm_op.get_workspace_size(args);
    gemm_ws = (gemm_ws + 255) & ~255;

    size_t temp_bytes = (size_t)M * N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;

    return gemm_ws + temp_bytes;
}

template <typename TileShape_, typename ClusterShape_, int SplitK>
int cutlass_gateup_silu_splitk_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K,
    void* workspace, size_t workspace_size, cudaStream_t stream
) {
    using Types = GateUpStreamKTypes<TileShape_, ClusterShape_>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::GemmKernel::TileSchedulerArguments sched_args;
    sched_args.splits = SplitK;

    typename Gemm::Arguments args_probe{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info,
        sched_args
    };
    Gemm gemm_probe;
    size_t gemm_ws = gemm_probe.get_workspace_size(args_probe);
    gemm_ws = (gemm_ws + 255) & ~255;

    // Bounds-check the caller's workspace BEFORE we start writing
    // through the `temp` pointer. Previously `workspace_size` was
    // accepted as a parameter and ignored, so a too-small buffer
    // produced silent OOB writes past the scratch region with no
    // error escape. The required size matches the formula the
    // sister `gateup_silu_workspace_size_*` function exports for
    // callers (gemm_ws + 256-aligned M*N*sizeof(half)). We refuse
    // to launch when the caller's buffer doesn't fit.
    size_t temp_bytes = (size_t)M * (size_t)N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;
    size_t needed = gemm_ws + temp_bytes;
    if (workspace_size < needed) {
        return -22 /* -EINVAL */;
    }

    char* ws_ptr = reinterpret_cast<char*>(workspace);
    void* gemm_workspace = ws_ptr;
    __half* temp = reinterpret_cast<__half*>(ws_ptr + gemm_ws);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(temp), stride_C,
            reinterpret_cast<ElementC*>(temp), stride_D,
        },
        hw_info,
        sched_args
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, gemm_workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    int intermediate_size = N / 2;
    int total = M * intermediate_size;
    int threads_needed = (total + 1) / 2;
    int block = 256;
    int grid = (threads_needed + block - 1) / block;

    silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output),
        temp,
        M,
        intermediate_size
    );

    return 0;
}

// ============================================================================
// Stream-K dispatch + workspace (default scheduler args)
// ============================================================================

template <typename TileShape_, typename ClusterShape_>
size_t cutlass_gateup_silu_streamk_ws(int M, int N, int K) {
    using Types = GateUpStreamKTypes<TileShape_, ClusterShape_>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info
    };

    Gemm gemm_op;
    size_t gemm_ws = gemm_op.get_workspace_size(args);
    gemm_ws = (gemm_ws + 255) & ~255;

    size_t temp_bytes = (size_t)M * N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;

    return gemm_ws + temp_bytes;
}

template <typename TileShape_, typename ClusterShape_>
int cutlass_gateup_silu_streamk_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K,
    void* workspace, size_t workspace_size, cudaStream_t stream
) {
    using Types = GateUpStreamKTypes<TileShape_, ClusterShape_>;
    using Gemm = typename Types::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::Arguments args_probe{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info
    };
    Gemm gemm_probe;
    size_t gemm_ws = gemm_probe.get_workspace_size(args_probe);
    gemm_ws = (gemm_ws + 255) & ~255;

    // Bounds-check the caller's workspace BEFORE we start writing
    // through the `temp` pointer. Previously `workspace_size` was
    // accepted as a parameter and ignored, so a too-small buffer
    // produced silent OOB writes past the scratch region with no
    // error escape. The required size matches the formula the
    // sister `gateup_silu_workspace_size_*` function exports for
    // callers (gemm_ws + 256-aligned M*N*sizeof(half)). We refuse
    // to launch when the caller's buffer doesn't fit.
    size_t temp_bytes = (size_t)M * (size_t)N * sizeof(__half);
    temp_bytes = (temp_bytes + 255) & ~255;
    size_t needed = gemm_ws + temp_bytes;
    if (workspace_size < needed) {
        return -22 /* -EINVAL */;
    }

    char* ws_ptr = reinterpret_cast<char*>(workspace);
    void* gemm_workspace = ws_ptr;
    __half* temp = reinterpret_cast<__half*>(ws_ptr + gemm_ws);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(temp), stride_C,
            reinterpret_cast<ElementC*>(temp), stride_D,
        },
        hw_info
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, gemm_workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    int intermediate_size = N / 2;
    int total = M * intermediate_size;
    int threads_needed = (total + 1) / 2;
    int block = 256;
    int grid = (threads_needed + block - 1) / block;

    silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output),
        temp,
        M,
        intermediate_size
    );

    return 0;
}

// ============================================================================
// Schedule aliases + macro
// ============================================================================

using WS   = cutlass::gemm::KernelTmaWarpSpecialized;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using PP   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

#define GATEUP_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED)   \
extern "C" size_t cutlass_gateup_silu_v##ID##_workspace_size(int M, int N, int K) { \
    return cutlass_gateup_silu_ws_dispatch<                                     \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(M, N, K);                    \
}                                                                               \
extern "C" int cutlass_gateup_silu_v##ID(                                       \
    void* o, const void* i, const void* w,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return cutlass_gateup_silu_dispatch<                                        \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(o, i, w, M, N, K, ws, ws_sz, s); \
}

#define GATEUP_STAGED_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, STAGES) \
extern "C" size_t cutlass_gateup_silu_v##ID##_workspace_size(int M, int N, int K) { \
    return cutlass_gateup_silu_staged_ws<                                       \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, STAGES>(M, N, K);            \
}                                                                               \
extern "C" int cutlass_gateup_silu_v##ID(                                       \
    void* o, const void* i, const void* w,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return cutlass_gateup_silu_staged_dispatch<                                 \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, STAGES>(o, i, w, M, N, K, ws, ws_sz, s); \
}

#define GATEUP_SWIZZLE_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, SWIZZLE) \
extern "C" size_t cutlass_gateup_silu_v##ID##_workspace_size(int M, int N, int K) { \
    return cutlass_gateup_silu_swizzle_ws<                                      \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SWIZZLE>(M, N, K);           \
}                                                                               \
extern "C" int cutlass_gateup_silu_v##ID(                                       \
    void* o, const void* i, const void* w,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return cutlass_gateup_silu_swizzle_dispatch<                                \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SWIZZLE>(o, i, w, M, N, K, ws, ws_sz, s); \
}

#define GATEUP_SPLITK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SPLITS) \
extern "C" size_t cutlass_gateup_silu_v##ID##_workspace_size(int M, int N, int K) { \
    return cutlass_gateup_silu_splitk_ws<                                       \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SPLITS>(M, N, K);                   \
}                                                                               \
extern "C" int cutlass_gateup_silu_v##ID(                                       \
    void* o, const void* i, const void* w,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return cutlass_gateup_silu_splitk_dispatch<                                 \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SPLITS>(o, i, w, M, N, K, ws, ws_sz, s); \
}

#define GATEUP_STREAMK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K) \
extern "C" size_t cutlass_gateup_silu_v##ID##_workspace_size(int M, int N, int K) { \
    return cutlass_gateup_silu_streamk_ws<                                      \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>>(M, N, K);                           \
}                                                                               \
extern "C" int cutlass_gateup_silu_v##ID(                                       \
    void* o, const void* i, const void* w,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return cutlass_gateup_silu_streamk_dispatch<                                \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>>(o, i, w, M, N, K, ws, ws_sz, s);    \
}

// ============================================================================
// 10 variants -- gate_up shape is (M, 37888, 3584), very wide N
// ============================================================================
//
// ID  Tile MxNxK     Cluster  Schedule   Notes
//  0  64x256x64      1x1x1   WS         small M, wide N
//  1  64x256x64      1x2x1   WS         small M, wide N, N-clustered
//  2  64x256x64      1x2x1   Coop       small M, wide N, N-clustered, cooperative
//  3  128x256x64     1x1x1   WS         wide N baseline
//  4  128x256x64     1x1x1   Coop       wide N cooperative
//  5  128x256x64     1x2x1   WS         wide N, N-clustered
//  6  128x256x64     1x2x1   Coop       wide N, N-clustered, cooperative
//  7  128x256x64     1x2x1   PP         wide N, N-clustered, pingpong
//  8  128x256x128    1x2x1   WS         K=128, N-clustered
//  9  256x128x64     2x1x1   WS         tall M, M-clustered

GATEUP_VARIANT(0,  64, 256, 64, 1,1,1, WS)
GATEUP_VARIANT(1,  64, 256, 64, 1,2,1, WS)
GATEUP_VARIANT(2, 128, 256,128, 1,2,1, Coop)  // K=128 N-clustered Coop (pair of v8 WS)
GATEUP_VARIANT(3, 128, 256, 64, 1,1,1, WS)
GATEUP_VARIANT(4, 128, 256, 64, 1,1,1, Coop)
GATEUP_VARIANT(5, 128, 256, 64, 1,2,1, WS)
GATEUP_VARIANT(6, 128, 256, 64, 1,2,1, Coop)
GATEUP_VARIANT(7, 128, 256, 64, 1,2,1, PP)
GATEUP_VARIANT(8, 128, 256,128, 1,2,1, WS)
GATEUP_VARIANT(9, 256, 128, 64, 2,1,1, WS)
GATEUP_VARIANT(10, 128, 256, 64, 2,2,1, WS)   // 4-SM cluster
GATEUP_VARIANT(11, 128, 256, 64, 1,4,1, WS)   // 4-SM along N for huge N=37888
// PingPong mirrors
GATEUP_VARIANT(12,  64, 256, 64, 1,2,1, PP)   // PP mirror of v2
GATEUP_VARIANT(13, 128, 256, 64, 1,1,1, PP)   // PP mirror of v4

// Explicit stages (v14-v17)
GATEUP_STAGED_VARIANT(14, 128, 256, 64, 1,2,1, WS, 2)
GATEUP_STAGED_VARIANT(15, 128, 256, 64, 1,2,1, WS, 4)
GATEUP_STAGED_VARIANT(16, 128, 256, 64, 1,1,1, WS, 2)
GATEUP_STAGED_VARIANT(17, 128, 256, 64, 1,1,1, WS, 4)

// Swizzle (v18-v23)
GATEUP_SWIZZLE_VARIANT(18, 128, 256, 64, 1,2,1, WS, 2)
GATEUP_SWIZZLE_VARIANT(19, 128, 256, 64, 1,2,1, WS, 4)
GATEUP_SWIZZLE_VARIANT(20, 128, 256, 64, 1,1,1, WS, 2)
GATEUP_SWIZZLE_VARIANT(21, 128, 256, 64, 1,1,1, WS, 4)
GATEUP_SWIZZLE_VARIANT(22,  64, 256, 64, 1,2,1, WS, 2)
GATEUP_SWIZZLE_VARIANT(23,  64, 256, 64, 1,2,1, WS, 4)

// Split-K (v24-v29) -- Cooperative only
GATEUP_SPLITK_VARIANT(24, 128, 256, 64, 1,2,1, 2)
GATEUP_SPLITK_VARIANT(25, 128, 256, 64, 1,2,1, 4)
GATEUP_SPLITK_VARIANT(26, 128, 256, 64, 1,2,1, 8)
GATEUP_SPLITK_VARIANT(27, 128, 256, 64, 1,1,1, 2)
GATEUP_SPLITK_VARIANT(28, 128, 256, 64, 1,1,1, 4)
GATEUP_SPLITK_VARIANT(29, 128, 256, 64, 1,1,1, 8)

// Stream-K (v30-v31) -- Cooperative only
GATEUP_STREAMK_VARIANT(30, 256, 256, 64, 1,1,1)  // big tile stream-K
GATEUP_STREAMK_VARIANT(31, 128, 256, 64, 1,1,1)
