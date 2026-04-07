// CUTLASS 3.x SM90 GateUp GEMM + fused SiLU*Mul for rvLLM.
//
// Fuses the gate+up projection with SiLU activation:
//   temp[M, 2*I] = input[M, K] @ weight[2*I, K]^T   (CUTLASS GEMM)
//   output[M, I] = SiLU(temp[:, 0:I]) * temp[:, I:2*I]
//
// The CUTLASS GEMM writes full [M, 2*I] to a workspace, then a fused
// SiLU*Mul kernel reads both halves and writes [M, I].
// This eliminates one full [M, 2*I] round-trip vs cuBLAS + separate kernels.
//
// Shapes for Qwen2.5-7B: M=128, K=3584, N=37888 (2*18944), output [128, 18944].
// Saves 128*37888*2 = 9.7MB intermediate tensor read+write.
//
// Build: nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
//        -I/root/cutlass/include -I/root/cutlass/tools/util/include \
//        -O3 -o libcutlass_gateup_silu.so --shared -Xcompiler -fPIC cutlass_gateup_silu.cu

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>

using namespace cute;

// ============================================================================
// Type aliases
// ============================================================================

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccum = float;

// A: [M, K] row-major, B: [N, K] row-major (presented as col-major to CUTLASS)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// ============================================================================
// CUTLASS 3.x SM90 GEMM: optimized tile for wide N (37888)
// 128x256x64 tile gives better N-coverage for this shape
// ============================================================================

using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _2, _1>;  // 2-SM cluster along N for wide output

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccum, ElementAccum,
            ElementC, LayoutC, 8,
            ElementC, LayoutC, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementAccum,
    ElementC, LayoutC, 8,
    ElementC, LayoutC, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

template <class T>
struct SiLuMulAux {
    CUTLASS_HOST_DEVICE
    T operator()(T const& z, T const& aux) const {
        cutlass::epilogue::thread::SiLu<T> silu;
        cutlass::multiplies<T> mul;
        return mul(silu(z), aux);
    }
};

using GateElementAux = cutlass::half_t;
using GateLayoutAux = cutlass::layout::RowMajor;
using GateFusionOp = cutlass::epilogue::fusion::LinCombDeEltAct<
    GateLayoutAux,
    SiLuMulAux,
    ElementC,
    ElementAccum,
    GateElementAux,
    void,
    ElementAccum
>;

using GateKernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using GateEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using GateCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementAccum,
    void, LayoutC, 8,
    ElementC, LayoutC, 8,
    GateEpilogueSchedule,
    GateFusionOp
>::CollectiveOp;

using GateCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename GateCollectiveEpilogue::SharedStorage))>,
    GateKernelSchedule
>::CollectiveOp;

using GateGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    GateCollectiveMainloop,
    GateCollectiveEpilogue
>;

using GateGemm = cutlass::gemm::device::GemmUniversalAdapter<GateGemmKernel>;

// ============================================================================
// Fused SiLU*Mul kernel: reads [M, 2*I], writes [M, I]
// Uses vectorized __half2 loads for 2x bandwidth efficiency
// ============================================================================

__global__ void silu_mul_fused_kernel(
    __half* __restrict__ output,       // [M, intermediate_size]
    const __half* __restrict__ temp,   // [M, 2 * intermediate_size]
    int M,
    int intermediate_size
) {
    // Process 2 elements at a time via half2
    const int total = M * intermediate_size;
    const int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 >= total) return;

    const int token = idx2 / intermediate_size;
    const int elem  = idx2 % intermediate_size;
    const int row_width = intermediate_size * 2;
    const int row_base = token * row_width;

    // Bounds: if elem+1 would cross row, handle scalar
    if (elem + 1 < intermediate_size) {
        // Vectorized path: load 2 gate + 2 up values
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
        // Scalar tail for last element in row (when intermediate_size is odd)
        float gate = __half2float(temp[row_base + elem]);
        float up   = __half2float(temp[row_base + intermediate_size + elem]);
        float silu = gate / (1.0f + expf(-gate));
        output[token * intermediate_size + elem] = __float2half(silu * up);
    }
}

// ============================================================================
// Public API
// ============================================================================

extern "C" {

// Returns workspace size needed: GEMM workspace + temp buffer for [M, 2*I]
size_t cutlass_gateup_silu_workspace_size(int M, int N, int K) {
    // N = 2 * intermediate_size (full gate+up width)
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

    // Temp buffer for [M, N] half-precision GEMM output
    size_t temp_bytes = (size_t)M * N * sizeof(__half);
    // Align to 256 bytes
    temp_bytes = (temp_bytes + 255) & ~255;
    gemm_ws = (gemm_ws + 255) & ~255;

    return gemm_ws + temp_bytes;
}

size_t cutlass_gate_silu_mul_workspace_size(int M, int N, int K) {
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename GateGemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GateGemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GateGemm::GemmKernel::StrideD{}, {M, N, 1});

    typename GateGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {}
    };
    args.epilogue.ptr_C = nullptr;
    args.epilogue.dC = {};
    args.epilogue.ptr_D = nullptr;
    args.epilogue.dD = stride_D;

    GateGemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// Fused GateUp GEMM + SiLU*Mul
//   output:     [M, N/2] -- final activated output (half width)
//   input:      [M, K]   -- hidden states
//   weight:     [N, K]   -- gate_up_proj weights, N = 2 * intermediate_size
//   workspace:  allocated by caller, size from cutlass_gateup_silu_workspace_size
//   M, K, N:    N is the FULL gate+up width (2 * intermediate_size)
int cutlass_gateup_silu(
    void* output,
    const void* input,
    const void* weight,
    int M, int N, int K,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    // Split workspace: [gemm_workspace | temp_buffer]
    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    // Compute GEMM workspace size to find temp buffer offset
    typename Gemm::Arguments args_probe{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D}
    };
    Gemm gemm_probe;
    size_t gemm_ws = gemm_probe.get_workspace_size(args_probe);
    gemm_ws = (gemm_ws + 255) & ~255;

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
    // Each thread handles 2 elements
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

int cutlass_gate_silu_mul(
    void* output,
    const void* input,
    const void* gate_weight,
    const void* aux_up,
    int M, int N, int K,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GateGemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GateGemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GateGemm::GemmKernel::StrideD{}, {M, N, 1});

    typename GateGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(gate_weight), stride_B,
        },
        {}
    };
    args.epilogue.ptr_C = nullptr;
    args.epilogue.dC = {};
    args.epilogue.ptr_D = reinterpret_cast<ElementC*>(output);
    args.epilogue.dD = stride_D;
    args.epilogue.thread.alpha = ElementAccum(1.0f);
    args.epilogue.thread.beta = ElementAccum(0.0f);
    args.epilogue.thread.aux_ptr = reinterpret_cast<const ElementC*>(aux_up);
    args.epilogue.thread.dAux = stride_D;

    GateGemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    return 0;
}

} // extern "C"
