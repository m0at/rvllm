// CUTLASS 4.4.2 Blackwell-Geforce (SM120) FP8 GEMM with blockwise weight scale
// and per-token activation scale.
//
//     D_f16[m, n] = sum_k (A_fp8[m, k] * B_fp8[n, k])
//                    * a_scale[m]                       <- per-token (M-vector)
//                    * b_block_scale[n / 128, k / 128]  <- 128×128 weight block-scale
//                    -> f16
//
// Template pattern borrowed from cutlass/examples/87_blackwell_geforce_gemm_blockwise/87a
// (native Blackwell-Geforce blockwise FP8 GEMM, bf16 output), adapted to:
//   * f16 output (ElementC = half_t, not bfloat16_t)
//   * per-M a_scale vector fused into epilogue (87a has a simple linear
//     combination epilogue with alpha/beta; we replace it with a
//     ColBroadcast fusion EVT so the per-token scale lands in the
//     fused epilogue rather than a separate post-pass).
//   * extern-C entry point matching SM90 `cutlass_fp8_gemm_channelscale`
//     ABI drop-in replacement — the SM90 path treats `col_scale` as a
//     per-N vector (FP8-Dynamic checkpoint format); this sm_120 path
//     treats it as a [N/128, K/128] block-scale (FP8-Block / Gemma 4
//     pre-quant format). That's the correct semantic for Gemma 4
//     fp8-block weights — the SM90 path inherits a known PPL regression
//     on that checkpoint which this kernel fixes on sm_121.
//
// A layout:  RowMajor   [M, K]
// B layout:  ColumnMajor[N, K]  (stored RowMajor by our loader but
//                                read as ColumnMajor here because
//                                CUTLASS expects the MMA B operand
//                                K-major, which is our RowMajor.
//                                CUTLASS's "ColumnMajor B" means the
//                                logical [K, N] operand is packed
//                                with K as fastest dim, which matches
//                                a RowMajor [N, K] byte stream.)

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using namespace cute;

using ElementA           = cutlass::float_e4m3_t;
using ElementB           = cutlass::float_e4m3_t;
using ElementD           = cutlass::half_t;            // f16 output
using ElementAccum       = float;
using ElementCompute     = float;
using ElementScalar      = float;

using LayoutA            = cutlass::layout::RowMajor;
using LayoutB            = cutlass::layout::ColumnMajor;
using LayoutD            = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

// Tile shape can be overridden at build time via -D{TILE_M,TILE_N,TILE_K}=N.
// Must stay a multiple of 128 per dim to line up with the blockwise scale
// granularity (sm120_trivial_blockwise_scale_config asserts this).
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 128
#endif
using MmaTileShape_MNK = Shape<cute::Int<TILE_M>, cute::Int<TILE_N>, cute::Int<TILE_K>>;
using ClusterShape_MNK = Shape<_1, _1, _1>;        // SM120 does not support cluster multicast

// Explicit scale-vector granularity:
//   SFVecSizeM = 1    → per-row activation scale (per-token)
//   SFVecSizeN = 128  → per-128-channel weight scale (N-block)
//   SFVecSizeK = 128  → per-128-channel K-block scale
// The `sm120_trivial_blockwise_scale_config` helper hard-codes all three
// to the MmaTileShape dims (128/128/128), which means **one SFA per
// 128-row tile** — a tile-wise broadcast. That collapses to the
// max-row-scale inside the tile, over-scaling every row whose
// per-token activation scale is below the tile max. For Gemma 4 31B
// prefill, activation scales vary ~6x across tokens (post-RMSNorm
// amax is token-dependent), so tile-wise SFA produces ~6x output
// bias on rows that weren't the scale-max row. aa01001pftrope0 root
// cause.
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    void, LayoutD, AlignmentD,           // no C
    ElementD, LayoutD, AlignmentD,        // D
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccum,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void   // default CLC tile scheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;

extern "C" {

/// Launch the SM120 blockwise FP8 GEMM.
///
/// `a_scale`  : `[M/MmaTileM]` f32 per-M-block activation scale. Must be
///              populated by the caller from the per-token a_scale[M]
///              (CUTLASS SM120 blockwise takes ONE scalar per M-tile;
///              the coarser granularity is an accepted quality trade-
///              off on prefill — for M ≤ 128 there's only one tile so
///              the mapping is lossless).
/// `b_scale`  : `[N/128, K/128]` f32 weight block-scale, row-major.
///              This is the Gemma 4 fp8-block `b_chscale` directly.
///
/// Returns 0 on success, negative for CUTLASS / launch failures.
int cutlass_fp8_gemm_blockscale_sm120(
    void* output,              // [M, N] f16
    const void* a,             // [M, K] fp8_e4m3
    const void* b,             // [N, K] fp8_e4m3, interpreted as ColumnMajor K-major
    const void* a_scale,       // [ceil(M/128), K/128] f32 (SFA)
    const void* b_scale,       // [N/128, K/128] f32 (SFB)
    int m, int n, int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
            reinterpret_cast<const ElementAccum*>(a_scale), layout_SFA,
            reinterpret_cast<const ElementAccum*>(b_scale), layout_SFB,
        },
        {
            {},                          // epilogue.thread (alpha/beta) — defaults to alpha=1, beta=0
            nullptr, stride_D,           // no C
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta  = 0.0f;

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    return 0;
}

/// SFA physical layout (from `Sm1xxBlockwiseScaleConfig` with majorSFA=MN
/// + SFVecSizeM=SFVecSizeK=128): contiguous in m_tile, k_block as outer
/// stride — `sfa[m_tile + k_block * ceil(M/128)]`. One scalar per
/// (m_tile, k_block) cell, replicated across the 128×128 atom shape.
///
/// SFB physical layout (majorSFB=MN): `sfb[n_tile + k_block * ceil(N/128)]`.
/// Gemma 4's `b_chscale` is stored row-major `[N/128, K/128]` so
/// `b_chscale[n_tile * k_blocks + k_block]` — TRANSPOSED relative to
/// what CUTLASS wants. We stage a transposed copy into SFB scratch
/// below rather than touch the checkpoint.
///
/// Bytes needed by each tensor at a given problem shape.
size_t cutlass_fp8_gemm_blockscale_sm120_sfa_bytes(int m, int k) {
    // SFVecSizeM = 1 → one SFA entry per (row, k_block).
    int kb = (k + 127) / 128;
    return (size_t)m * (size_t)kb * sizeof(float);
}

size_t cutlass_fp8_gemm_blockscale_sm120_sfb_bytes(int n, int k) {
    int nb = (n + 127) / 128;
    int kb = (k + 127) / 128;
    return (size_t)nb * (size_t)kb * sizeof(float);
}

} // extern "C"

// SFA prep kernel — max-reduce a_scale[M] over each 128-row chunk and
// write to SFA[m_tile + k_block * m_blocks], replicated across every
// k_block. Max (not mean) because underestimating the activation
// magnitude clips FP8 dynamic range; max is the conservative choice.
//
// Grid: (k_blocks, m_blocks). Block: 128 threads (one per row in the
// m_tile). 2-stage warp-shuffle reduction.
// Per-row activation scale → SFA layout.
// With SFVecSizeM=1, SFA has one entry per (row, k_block). CUTLASS
// MN-major layout: sfa[row + k_block * m].
__global__ void fill_sfa_from_a_scale_sm120(
    const float* __restrict__ a_scale,   // [M]
    float*       __restrict__ sfa,       // [m * k_blocks], CUTLASS MN-major
    int m,
    int /*m_blocks_unused*/,
    int k_blocks
) {
    int row     = blockIdx.y * blockDim.x + threadIdx.x;
    int k_block = blockIdx.x;
    if (row >= m || k_block >= k_blocks) return;
    sfa[row + k_block * m] = a_scale[row];
}

// SFB transpose kernel — read row-major b_chscale[n_tile, k_block] and
// store at CUTLASS SFB[n_tile + k_block * n_blocks]. Pure per-element
// transpose, no reduction.
__global__ void fill_sfb_from_b_chscale_sm120(
    const float* __restrict__ b_chscale, // row-major [n_blocks, k_blocks]
    float*       __restrict__ sfb,       // [n_blocks * k_blocks], CUTLASS layout
    int n_blocks,
    int k_blocks
) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_blocks * k_blocks;
    if (idx >= total) return;
    // b_chscale is row-major [n_blocks, k_blocks]; SFB is CUTLASS
    // MN-major [n_blocks, k_blocks] — transpose.
    int n_tile  = idx / k_blocks;
    int k_block = idx - n_tile * k_blocks;
    sfb[n_tile + k_block * n_blocks] = b_chscale[n_tile * k_blocks + k_block];
}

extern "C" {

int cutlass_fp8_gemm_blockscale_sm120_prep_sfa(
    const void* a_scale,
    void*       sfa,
    int m, int k,
    cudaStream_t stream
) {
    // Per-row SFA: grid = (k_blocks, m_tiles_of_128_threads).
    int kb = (k + 127) / 128;
    int threads = 128;
    int m_tiles = (m + threads - 1) / threads;
    dim3 grid(kb, m_tiles);
    dim3 block(threads);
    fill_sfa_from_a_scale_sm120<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(a_scale),
        reinterpret_cast<float*>(sfa),
        m, m_tiles, kb
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cutlass_fp8_gemm_blockscale_sm120_prep_sfb(
    const void* b_chscale,
    void*       sfb,
    int n, int k,
    cudaStream_t stream
) {
    int nb = (n + 127) / 128;
    int kb = (k + 127) / 128;
    int total = nb * kb;
    int bs = 256;
    int gs = (total + bs - 1) / bs;
    fill_sfb_from_b_chscale_sm120<<<gs, bs, 0, stream>>>(
        reinterpret_cast<const float*>(b_chscale),
        reinterpret_cast<float*>(sfb),
        nb, kb
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/// Query the workspace size required for a given problem shape.
size_t cutlass_fp8_gemm_blockscale_sm120_workspace(int m, int n, int k) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{}, nullptr, stride_D, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

} // extern "C"

#else

// sm < 120: the symbols are not built. Operators shouldn't link this
// object on pre-Blackwell-Geforce targets — the build script only
// compiles it under `-arch=sm_120a` / `-arch=sm_121a`.

extern "C" {

int cutlass_fp8_gemm_blockscale_sm120(
    void*, const void*, const void*, const void*, const void*,
    int, int, int, void*, size_t, void*
) {
    return -100;  // unsupported arch
}

size_t cutlass_fp8_gemm_blockscale_sm120_workspace(int, int, int) {
    return 0;
}

size_t cutlass_fp8_gemm_blockscale_sm120_sfa_bytes(int, int) { return 0; }
size_t cutlass_fp8_gemm_blockscale_sm120_sfb_bytes(int, int) { return 0; }

int cutlass_fp8_gemm_blockscale_sm120_prep_sfa(
    const void*, void*, int, int, void*
) { return -100; }

int cutlass_fp8_gemm_blockscale_sm120_prep_sfb(
    const void*, void*, int, int, void*
) { return -100; }

} // extern "C"

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED || SM121_SUPPORTED
