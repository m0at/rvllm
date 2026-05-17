// Thin C-linkage wrapper around CUTLASS 4.x example 55 (hopper int4 x fp8 gemm).
//
// Exposes one entry point: rvllm_w4a8_gemm_run(...) that runs
//   D (f16)  =  alpha * A_fp8 * B_int4_quant  +  beta * C (f16)
// where B is int4 with per-group FP8 scales (group_size = 128) packed as
// cutlass::Array<e4m3, 8> (LUT trick from example 55).
//
// A is RowMajor [M, K] FP8 E4M3.
// B is logical [N, K] int4 two's complement, CUTLASS-reordered offline.
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
#include "cutlass/util/mixed_dtype_utils.hpp"  // compute_memory_reordering_atom

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
constexpr size_t kInt4ReorderPaddingBytes = 16 * 1024;

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

    // Reordered B layout — follow example 55 line 399: shape-only overload.
    auto shape_B = cute::make_shape(n, k, 1);
    LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

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

static __global__ void rowscale_f16_kernel(
    __half* __restrict__ data,
    const float* __restrict__ row_scales,
    int m,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;
    int row = idx / n;
    float v = __half2float(data[idx]) * row_scales[row];
    data[idx] = __float2half_rn(v);
}

extern "C" int rvllm_w4a8_gemm_run_rowscale(
    const void* a_fp8,
    const float* a_scales,
    const void* b_int4_reordered,
    const void* b_scales_packed,
    void*       d_f16,
    int         m,
    int         n,
    int         k,
    int         group_size,
    void*       workspace,
    size_t      workspace_bytes,
    cudaStream_t stream
) {
    if (a_scales == nullptr) return -30;
    int rc = rvllm_w4a8_gemm_run(
        a_fp8,
        b_int4_reordered,
        b_scales_packed,
        nullptr,
        d_f16,
        m,
        n,
        k,
        group_size,
        1.0f,
        0.0f,
        workspace,
        workspace_bytes,
        stream
    );
    if (rc != 0) return rc;

    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    rowscale_f16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(d_f16),
        a_scales,
        m,
        n
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -31;
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

    auto shape_B = cute::make_shape(n, k, 1);
    LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

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

static size_t w4a8_int4_reordered_bytes(int n, int k) {
    auto shape_B = cute::make_shape(n, k, 1);
    LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
    size_t elems = static_cast<size_t>(cute::cosize(layout_B_reordered));
    return (elems + 1) / 2 + kInt4ReorderPaddingBytes;
}

extern "C" size_t rvllm_w4a8_int4_reordered_bytes(int n, int k) {
    return w4a8_int4_reordered_bytes(n, k);
}

// =========================================================================
// Weight encoder: FP16 weights -> (reordered INT4 + LUT-packed FP8 scales).
//
// Simple symmetric per-group (g=128) quantization. For each [N, group] block
// of K-contiguous weights:
//    scale_fp32 = max(|w|) / 7
//    w_int4     = round(w / scale_fp32)  clamped to [-8, 7]
// Then the INT4 positive encoding is "unified" with the negative encoding
// (example 55 convention) and the scale is packed as Array<e4m3, 8> holding
//    {scale * -8, scale * -7, ..., scale * -1}
// Finally the INT4 tensor is memory-reordered via the LayoutAtomQuant atom
// so each thread reads 8 contiguous elements in one load.
//
// No AWQ activation protection in v1 — symmetric weight-only quant. Upgrade
// path if quality suffers: compute per-channel activation max from a
// calibration pass, multiply weights by s^alpha, divide activations by
// s^alpha at runtime. Separate function.
// =========================================================================

#include "cutlass/util/device_memory.h"
#include <cuda_fp16.h>

// Quantize FP16 weights to INT4 with per-group FP32 scales, both on device.
// Writes two's-complement INT4 into CUTLASS LayoutB order and per-group f32
// scales into scales_f32. Host code applies CUTLASS's unified INT4 encoding
// and shuffle reorder after this kernel.
static __global__ void quantize_sym_group_kernel(
    const __half* __restrict__ w_fp16,  // [N, K] row-major (N rows)
    int* __restrict__ w_int4_raw,        // packed int4 in LayoutB storage
    float* __restrict__ scales_f32,       // [N, K/group]
    int n, int k, int group
) {
    int row = blockIdx.y;
    int grp = blockIdx.x;
    int tid = threadIdx.x;  // 0..31
    if (row >= n || grp * group >= k) return;

    const __half* w_row = w_fp16 + (size_t)row * k + grp * group;

    // Pass 1: max |w| in group, reduce across the 32-thread warp.
    float local_max = 0.0f;
    for (int i = tid; i < group; i += 32) {
        float v = __half2float(w_row[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

    float scale = local_max / 7.0f;
    if (scale == 0.0f) scale = 1e-9f;  // avoid div0
    if (tid == 0) scales_f32[row * (k / group) + grp] = scale;

    float inv_scale = 1.0f / scale;

    // Pass 2: quantize + pack 8 int4 elements into one int32. The swapped
    // CUTLASS B stride is K-contiguous for shape [N,K].
    for (int i = tid; i < group; i += 32) {
        float v = __half2float(w_row[i]) * inv_scale;
        int q = __float2int_rn(v);
        if (q > 7) q = 7;
        if (q < -8) q = -8;
        unsigned int nib = (unsigned int)(q & 0xF);
        size_t elem_idx = (size_t)row * k + (size_t)(grp * group + i);
        size_t word_idx = elem_idx / 8;
        int slot_idx = (int)(elem_idx % 8);
        atomicOr(&w_int4_raw[word_idx], (int)(nib << (slot_idx * 4)));
    }
}

static __global__ void convert_scales_kernel(
    const float* __restrict__ scales_f32,
    ElementScale* __restrict__ scales_e4m3,
    int n, int scale_k
) {
    int row = blockIdx.y;
    int grp = blockIdx.x;
    if (row >= n || grp >= scale_k) return;

    float s = scales_f32[row * scale_k + grp];
    scales_e4m3[row * scale_k + grp] = ElementScale(s);
}

// Host entry: quantize + pack weights + reorder into the kernel-expected
// layout. Expects:
//   w_fp16        [N, K] row-major device ptr (input; will be read)
//   w_int4_out    device ptr sized by rvllm_w4a8_int4_reordered_bytes(N, K)
//   scales_out    [N, K/group, 8] bytes device ptr (output; e4m3 LUT)
//   workspace     temporary f32 scales buffer, >= N*K/group*4 bytes
//   shuffle       reserved; output is always CUTLASS-reordered because
//                 rvllm_w4a8_gemm_run always consumes LayoutB_Reordered.
extern "C" int rvllm_w4a8_encode_weight_fp16(
    const void* w_fp16,
    int         n,
    int         k,
    int         group_size,
    void*       w_int4_out,
    void*       scales_packed_out,
    void*       scales_f32_workspace,
    int         shuffle,
    cudaStream_t stream
) {
    if (group_size != kGroupSize) return -1;
    if (k % group_size != 0)      return -2;
    if (k % 8 != 0)               return -3;
    const int scale_k = k / group_size;
    const size_t int4_elems = (size_t)n * k;
    const size_t int4_output_bytes = w4a8_int4_reordered_bytes(n, k);

    // 1) Quantize + build f32 scales.
    cudaMemsetAsync(w_int4_out, 0, int4_output_bytes, stream);
    dim3 grid_q(scale_k, n, 1);
    dim3 block_q(32, 1, 1);
    quantize_sym_group_kernel<<<grid_q, block_q, 0, stream>>>(
        (const __half*)w_fp16,
        (int*)w_int4_out,
        (float*)scales_f32_workspace,
        n, k, group_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -20;

    // 2) Convert scales to e4m3, then use CUTLASS's host-side LUT packer.
    cutlass::DeviceAllocation<ElementScale> scales_e4m3((size_t)n * scale_k);
    dim3 grid_s(scale_k, n, 1);
    dim3 block_s(1, 1, 1);
    convert_scales_kernel<<<grid_s, block_s, 0, stream>>>(
        (const float*)scales_f32_workspace,
        scales_e4m3.get(),
        n, scale_k
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) return -21;

    // 3) Match example 55 formatting: unify INT4 encodings, then reorder.
    // These CUTLASS utility helpers are host-side wrappers and synchronize.
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -22;
    bool scales_ok = cutlass::pack_scale_fp8<ElementScale, QuantType>(
        scales_e4m3.get(),
        reinterpret_cast<cutlass::Array<ElementScale, 8>*>(scales_packed_out),
        (size_t)n * scale_k
    );
    if (!scales_ok) return -23;

    cutlass::DeviceAllocation<ElementB> unified(
        int4_elems + kInt4ReorderPaddingBytes * 8 / cutlass::sizeof_bits<ElementB>::value
    );
    cudaMemset(unified.get(), 0, unified.bytes());
    bool ok = cutlass::unified_encode_int4b(
        reinterpret_cast<const ElementB*>(w_int4_out),
        unified.get(),
        int4_elems
    );
    if (!ok) return -24;

    auto shape_B = cute::make_shape(n, k, 1);
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    auto layout_B = cute::make_layout(shape_B, stride_B);
    LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
    cutlass::reorder_tensor(
        unified.get(),
        layout_B,
        reinterpret_cast<ElementB*>(w_int4_out),
        layout_B_reordered
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) return -25;

    (void)shuffle;

    return 0;
}
