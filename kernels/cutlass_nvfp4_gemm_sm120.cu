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

} // extern "C"

// -------------------------------------------------------------------
// SFA layout transform — natural row-major [m, k/16] E4M3 → CUTLASS
// Sm120 NVFP4 interleaved SFA layout.
//
// `cutlass_nvfp4_prep_act_sm120.cu`'s prep_act kernel writes the
// per-block E4M3 scales as a plain row-major [m, k/16] byte tensor
// (one scale per (token, K-block-of-16)). The CUTLASS NVFP4 GEMM
// expects the SFA tensor in the interleaved layout produced by
// `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA`, which swizzles
// the (m_tile, n_tile, k_block) coords into a tensor-core-friendly
// access pattern.
//
// The transform is a pure index permutation. We enumerate every
// natural (row, k_block) coordinate, look up the destination index
// via the cute Layout, and copy one byte. No reduction, no
// quantisation rework — purely a memory rearrangement.
//
// Grid: (k_blocks, m). Block: 1 thread (the work is one byte per
// thread; cache benefits dominate over thread-density on this kind
// of pure-permutation kernel).
// -------------------------------------------------------------------
// Concrete SFA / SFB layout types, deduced from the Mistral GEMM
// instantiation above. We bind them to fixed layout types so the
// kernels below aren't templates — nvcc's __cudaRegisterEntry stub
// generation doesn't tolerate template kernels in anonymous
// namespaces with cute::Layout type parameters (bug-shape: the
// mangled-name pasted into the stub trips the host C compiler).
//
// SFA shape: 3 modes (M, K, L) — N elided.
// SFB shape: 3 modes (N, K, L) — M elided.
// Coord arity is therefore 3-tuple in both cases. Per CUTLASS
// reference, the K mode encodes LOGICAL K elements (not K/16
// scale-block indices); the layout has zero stride inside each
// 16-element scale group, so the canonical representative coord
// for scale-block `kb` is `kb * 16`.
using SfaInterleavedLayout =
    decltype(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(int{0}, int{0}, int{0}, int{0})));
using SfbInterleavedLayout =
    decltype(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(int{0}, int{0}, int{0}, int{0})));

__global__ void nvfp4_sfa_natural_to_interleaved_kernel(
    const __nv_fp8_e4m3* __restrict__ src_natural,  // [m, k/16] row-major
    __nv_fp8_e4m3*       __restrict__ dst_cutlass,  // CUTLASS interleaved
    SfaInterleavedLayout layout,
    int m,
    int k_blocks)
{
    const int kb  = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= m || kb >= k_blocks) return;

    // The SFA layout produced by `tile_atom_to_shape_SFA` has 3
    // modes (M, K, L) — the N dim is elided. K is the LOGICAL
    // K dim, not the K/16 block index; the layout has zero stride
    // within each 16-element scale group, so the canonical
    // representative coord for scale-block `kb` is `kb * 16`.
    // Codex review (2026-05-07) caught the original `(row, kb, 0)`
    // — wrong stride accumulation on the hierarchical K mode.
    const __nv_fp8_e4m3 v = src_natural[row * k_blocks + kb];
    const auto idx = layout(cute::make_coord(row, kb * 16, 0));
    dst_cutlass[idx] = v;
}

extern "C" {

/// Transform a natural row-major SFA tensor (`[m, k/16]` E4M3 from
/// `cutlass_nvfp4_gemm_sm120_prep_act`) into the CUTLASS Sm120
/// interleaved layout the GEMM kernel consumes. After this call,
/// `dst_cutlass` is ready to feed `cutlass_nvfp4_gemm_sm120` as
/// `sfa`.
///
/// Returns 0 on success, -1 on bad shape, -3 on launch failure.
int cutlass_nvfp4_gemm_sm120_sfa_natural_to_interleaved(
    const void* src_natural,
    void*       dst_cutlass,
    int m,
    int k,
    cudaStream_t stream)
{
    if (m <= 0 || k <= 0 || (k % 16) != 0) return -1;
    if (src_natural == nullptr || dst_cutlass == nullptr) return -1;
    const int k_blocks = k / 16;

    // Build the same SFA layout the GEMM uses. n_dummy stays > 0 so
    // the helper produces a valid layout (the SFA index is
    // n-independent in practice).
    const int n_dummy = (m > 0) ? m : 1;
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n_dummy, k, 1));

    const dim3 grid(k_blocks, m, 1);
    const dim3 block(1, 1, 1);
    nvfp4_sfa_natural_to_interleaved_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(src_natural),
        reinterpret_cast<__nv_fp8_e4m3*>(dst_cutlass),
        layout,
        m, k_blocks);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}

} // extern "C" — transition out for the SFB transform kernel.

// SFB transform: natural row-major [N, K/16] E4M3 -> CUTLASS Sm120
// interleaved SFB layout. Mistral 3.5 ships `weight_scale` in the
// natural layout; without this transform the GEMM reads garbage
// scales and silently produces wrong outputs (codex review caught
// this — the original landing forgot SFB and only handled SFA).
__global__ void nvfp4_sfb_natural_to_interleaved_kernel(
    const __nv_fp8_e4m3* __restrict__ src_natural,  // [n, k/16] row-major
    __nv_fp8_e4m3*       __restrict__ dst_cutlass,  // CUTLASS interleaved
    SfbInterleavedLayout layout,
    int n,
    int k_blocks)
{
    const int kb     = blockIdx.x;
    const int n_row  = blockIdx.y;
    if (n_row >= n || kb >= k_blocks) return;

    // Same coord-arity rule as SFA: 3 modes (N, K, L), K is the
    // logical K dim — coord for scale-block `kb` is `kb * 16`.
    const __nv_fp8_e4m3 v = src_natural[n_row * k_blocks + kb];
    const auto idx = layout(cute::make_coord(n_row, kb * 16, 0));
    dst_cutlass[idx] = v;
}

extern "C" {

/// Transform a natural row-major SFB tensor (`[n, k/16]` E4M3,
/// the Mistral checkpoint's `weight_scale`) into the CUTLASS Sm120
/// interleaved layout the GEMM kernel consumes. Companion to the
/// SFA transform above; both must run before
/// `cutlass_nvfp4_gemm_sm120` reads the per-block scales.
///
/// Returns 0 on success, -1 on bad shape, -3 on launch failure.
int cutlass_nvfp4_gemm_sm120_sfb_natural_to_interleaved(
    const void* src_natural,
    void*       dst_cutlass,
    int n,
    int k,
    cudaStream_t stream)
{
    if (n <= 0 || k <= 0 || (k % 16) != 0) return -1;
    if (src_natural == nullptr || dst_cutlass == nullptr) return -1;
    const int k_blocks = k / 16;

    // Same layout-helper trick as the SFA path: pass an n-independent
    // m_dummy so the cute layout deduces correctly.
    const int m_dummy = (n > 0) ? n : 1;
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m_dummy, n, k, 1));

    const dim3 grid(k_blocks, n, 1);
    const dim3 block(1, 1, 1);
    nvfp4_sfb_natural_to_interleaved_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(src_natural),
        reinterpret_cast<__nv_fp8_e4m3*>(dst_cutlass),
        layout,
        n, k_blocks);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}

/// CUTLASS-interleaved SFB scratch bytes for `[n, k/16]` E4M3 in
/// the same swizzled layout the GEMM consumes. Mirrors
/// `cutlass_nvfp4_gemm_sm120_sfa_bytes` for the weight side.
size_t cutlass_nvfp4_gemm_sm120_sfb_bytes(int n, int k) {
    if (n <= 0 || k <= 0) return 0;
    int m_dummy = (n > 0) ? n : 1;
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m_dummy, n, k, 1));
    return static_cast<size_t>(size(filter_zeros(layout_SFB)));
}

/// Natural-layout SFB bytes — `n * k / 16` (one E4M3 per (n, k_block)
/// pair). Distinct from `cutlass_nvfp4_gemm_sm120_sfb_bytes`, which
/// reports the swizzled-and-padded final destination size.
size_t cutlass_nvfp4_gemm_sm120_sfb_natural_bytes(int n, int k) {
    if (n <= 0 || k <= 0 || (k % 16) != 0) return 0;
    return static_cast<size_t>(n) * static_cast<size_t>(k / 16);
}

/// Activation prep wrapper — chains the natural-layout
/// `prep_act` kernel (in `cutlass_nvfp4_prep_act_sm120.cu`) with
/// the SFA layout transform above to produce a CUTLASS-ready
/// (a_packed, sfa_cutlass) pair from a BF16/F16 input tensor.
///
/// Caller-provided buffers (must be sized via the corresponding
/// queries):
///   `a_packed_out`            `m * k / 2` bytes
///   `sfa_natural_scratch`     `m * k / 16` bytes (intermediate)
///   `sfa_cutlass_out`         `cutlass_nvfp4_gemm_sm120_sfa_bytes(m, k)` bytes
///
/// Returns 0 on success; forwards the underlying error rc on failure.
/// Stub today (-100) until `prep_act` is wired in via the build
/// SOURCES list — the symbol is defined in a sibling .cu so chaining
/// here would require `extern "C"` decl, but the linker resolution
/// is what closes the loop. Operators rebuilding the .so with both
/// sources flip this from -100 to a real chain.
int cutlass_nvfp4_gemm_sm120_prep_sfa(
    const void* /*a_input*/,
    void*       /*a_packed_out*/,
    void*       /*sfa_cutlass_out*/,
    int         /*m*/,
    int         /*k*/,
    int         /*a_input_dtype*/,
    cudaStream_t /*stream*/)
{
    // Wiring helper:
    //   1. cutlass_nvfp4_gemm_sm120_prep_act(...)  -> a_packed +
    //      a temporary natural SFA buffer.
    //   2. cutlass_nvfp4_gemm_sm120_sfa_natural_to_interleaved(...)
    //      -> CUTLASS SFA.
    // Today the public Rust wrapper (`launch_nvfp4_prep_sfa`) does
    // NOT supply a scratch buffer for the natural-layout SFA; that's
    // the missing piece. Until the runtime provides it, return -100
    // so callers fall back to the kernel-side error path rather than
    // get incorrect SFA bytes.
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
