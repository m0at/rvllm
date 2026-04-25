// Smoke + numerical-correctness probe for the Blackwell native E2M1
// tensor-core MMA
//     mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32.
//
// **WIP** — the packed kernel launches and the MMA runs, but the
// host-side fp64 comparison currently FAILS. See the header comment
// in `nvfp4_mma_frag_pack.cuh` for the layout-isolation work that's
// still needed before this probe turns green.
//
// Two entry points:
//
//  (1) `nvfp4_mma_probe_kernel` — original minimal smoke test. Takes
//      per-lane pre-packed operand fragments, runs one MMA, writes
//      the 4 × f32 accumulator back per lane. Kept for back-compat
//      with `v3/tools/nvfp4_mma_probe_check.py`'s existing zero-input
//      assembly check.
//
//  (2) `nvfp4_mma_packed_probe_kernel` — new. Takes the A and B tiles
//      as packed NVFP4 bytes in device memory (A: 16×32 bytes for
//      m=16, k=64 e2m1 values; B: 8×32 bytes), packs the per-lane
//      fragments via the header helpers, runs the MMA, writes the
//      full 16×8 f32 output tile back out. The host test drives this
//      with non-trivial inputs and compares to an fp64 reference.
//
// The packing helpers live in `nvfp4_mma_frag_pack.cuh`; see that
// file for the lane layout. The smem tiles are just a device-mem
// staging area here — loaded once per kernel launch from the
// packed-byte inputs, then read by the frag packers. The FA2
// integration will instead populate these tiles directly from
// a page-table walk over the NVFP4 K/V cache.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "nvfp4_mma_frag_pack.cuh"

// V2 probe — production NVFP4 blockscaled MMA
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//       .row.col.f32.e2m1.e2m1.f32.ue4m3
// Takes per-lane pre-packed A (4 u32), B (2 u32), and per-lane packed
// scale-factor registers sfa (u32 = 4 × ue4m3) and sfb (u32). The FA2
// integration will produce sfa/sfb from the cache's per-16-elem E4M3
// microscales directly. Used by `v3/tools/nvfp4_mma_layout_probe.py`
// to derive the correct lane-and-nibble mapping.
// V3 probe — the actual FA2-integration contract. Takes A and B data
// + per-16-element E4M3 microscales in the same layout the paged KV
// cache produces, i.e. A[16 rows × 32 bytes] + A_scales[16 × 4 E4M3]
// and B[8 rows × 32 bytes] + B_scales[8 × 4 E4M3]. Loads each lane's
// sfa/sfb from the scale arrays using the MMA's scale-broadcast
// contract:
//
//   - 4 lanes share each m-row (A) or n-col (B) (confirmed by
//     `nvfp4_mma_layout_probe.py`).
//   - Each MMA consumes 4 scales per operand per row (scale_vec::4X,
//     one per K=16 sub-block).
//   - With bidA=tidA=bidB=tidB=0, hardware indexes the sfa/sfb u32 by
//     the K-block index; all 4 lanes sharing a row therefore pass the
//     SAME u32 = all 4 E4M3 scales for their row packed LSB→MSB.
//
// FA2 integration will use this exact scale-loading pattern — the
// only further change is that the A/B bytes come from the paged-cache
// `[num_blocks, block_size, num_kv_heads, head_dim/2]` layout rather
// than flat `a_bytes_in` / `b_bytes_in`.
extern "C"
__global__ void nvfp4_mma_scaled_probe_kernel(
    const unsigned char* __restrict__ a_bytes_in,   // 16 × 32 bytes (row-major, K across cols)
    const unsigned char* __restrict__ b_bytes_in,   //  8 × 32 bytes
    const unsigned char* __restrict__ a_scales_in,  // 16 × 4 E4M3 (row-major)
    const unsigned char* __restrict__ b_scales_in,  //  8 × 4 E4M3
    float*               __restrict__ d_tile_out    // 16 × 8 f32
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    extern __shared__ unsigned char smem[];
    constexpr int A_BYTES = 16 * 32;
    constexpr int B_BYTES =  8 * 32;
    unsigned char* s_a = smem;
    unsigned char* s_b = smem + A_BYTES;
    float*         s_d = reinterpret_cast<float*>(s_b + B_BYTES);

    const int tid = threadIdx.x;

    // Load A/B bytes into smem.
    constexpr int A_U32 = A_BYTES / 4;
    constexpr int B_U32 = B_BYTES / 4;
    uint32_t* s_a_u32 = reinterpret_cast<uint32_t*>(s_a);
    uint32_t* s_b_u32 = reinterpret_cast<uint32_t*>(s_b);
    const uint32_t* a_u32_in = reinterpret_cast<const uint32_t*>(a_bytes_in);
    const uint32_t* b_u32_in = reinterpret_cast<const uint32_t*>(b_bytes_in);
    for (int i = tid; i < A_U32; i += 32) s_a_u32[i] = a_u32_in[i];
    for (int i = tid; i < B_U32; i += 32) s_b_u32[i] = b_u32_in[i];
    __syncthreads();

    // Pack A/B fragments (byte layout inherited from fp8 — validated by
    // the packed probe).
    uint32_t a_frag[4];
    uint32_t b_frag[2];
    rvllm_nvfp4::pack_a_frag_e2m1_m16k64(s_a, 32, a_frag, tid);
    rvllm_nvfp4::pack_b_frag_e2m1_n8k64 (s_b, 32, b_frag, tid);

    // **Scale-byte layout is not yet fully understood for varying
    // scales across rows / K-blocks.** Tests so far:
    //   - Uniform scales per operand: correct.
    //   - Row-0 scale gradient with all-ones data: under the formula
    //     below, gave D[0,:]=64 (expected 104 under "byte k = K-block
    //     k"). Multiple candidate byte→(reg or kblock) mappings
    //     produce near-correct but not exact results.
    //   - Random fp16 → NVFP4 quant + MMA: ~30% error vs fp64 ref.
    //
    // Retaining the formula where each 4-lane group serves a row-pair
    // (m, m+8) split by the lane LSB and byte-order = K-block order.
    // This is a known partial-fix state; the FA2 integration will
    // need to revisit either the sfa packing OR switch to issuing
    // separate MMAs per m-row pair with exact scales.
    //
    // See `/tmp/nvfp4_kbdet.py`, `/tmp/nvfp4_varying_scales.py`,
    // and `v3/tools/nvfp4_mma_scaled_probe_check.py` for reproducers.
    const int a_row = ((tid & 1) << 3) | (tid >> 2);
    const int n_col = tid >> 2;
    uint32_t sfa;
    {
        uint32_t s0 = a_scales_in[a_row * 4 + 0];
        uint32_t s1 = a_scales_in[a_row * 4 + 1];
        uint32_t s2 = a_scales_in[a_row * 4 + 2];
        uint32_t s3 = a_scales_in[a_row * 4 + 3];
        sfa = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
    }
    uint32_t sfb;
    {
        uint32_t s0 = b_scales_in[n_col * 4 + 0];
        uint32_t s1 = b_scales_in[n_col * 4 + 1];
        uint32_t s2 = b_scales_in[n_col * 4 + 2];
        uint32_t s3 = b_scales_in[n_col * 4 + 3];
        sfb = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
    }

    float d[4];
    rvllm_nvfp4::zero_mma_d_frag(d);
    rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d, a_frag, b_frag, sfa, sfb);

    rvllm_nvfp4::unpack_d_frag_to_smem_m16n8(
        s_d, 8 * (int)sizeof(float), d, tid);
    __syncthreads();
    constexpr int D_F32 = 16 * 8;
    for (int i = tid; i < D_F32; i += 32) d_tile_out[i] = s_d[i];
#else
    (void)a_bytes_in; (void)b_bytes_in; (void)a_scales_in; (void)b_scales_in;
    if (threadIdx.x < 32) {
        for (int i = 0; i < 16 * 8 / 32; ++i) {
            d_tile_out[threadIdx.x + i * 32] = 0.0f;
        }
    }
#endif
}

// Path A probe — mixed FP8 E4M3 (A/Q) × NVFP4 E2M1 (B/K) via the
// unscaled `kind::f8f6f4.m16n8k32.e4m3.e2m1.f32` MMA, with per-K=16
// NVFP4 block scales applied post-MMA. Runs two MMAs per K=32 tile
// (one per K=16 sub-block), multiplies each accumulator by its
// scale, sums. Output D is [16 × 8] f32 (not halved — accumulator
// level). Validates the full FA2 integration contract.
//
// Inputs:
//   a_bytes_fp8   — Q tile, 16 rows × 32 bytes (fp8 E4M3, already descaled).
//   b_bytes_nvfp4 — K tile, 8 rows × 16 bytes (packed 2 e2m1 per byte).
//   b_scales      — K tile scales, 8 rows × 2 E4M3 bytes (one per K=16).
//
// Smem layout: a_smem (512 bytes) + b_smem_nvfp4 (128 bytes) + d_smem
// (512 bytes) = 1152 bytes dynamic smem.
extern "C"
__global__ void nvfp4_mma_path_a_probe_kernel(
    const unsigned char* __restrict__ a_bytes_fp8,    // 16 × 32 bytes
    const unsigned char* __restrict__ b_bytes_nvfp4,  //  8 × 16 bytes
    const unsigned char* __restrict__ b_scales,       //  8 ×  2 E4M3 bytes
    float*               __restrict__ d_tile_out      // 16 × 8 f32
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    extern __shared__ unsigned char smem[];
    constexpr int A_BYTES   = 16 * 32;
    constexpr int B_BYTES   =  8 * 16;  // packed NVFP4
    unsigned char* s_a  = smem;
    unsigned char* s_b  = smem + A_BYTES;
    float*         s_d  = reinterpret_cast<float*>(s_b + B_BYTES);
    const int tid = threadIdx.x;

    // Cooperative load A (512 B) + B (128 B) = 640 bytes via 32 lanes.
    constexpr int A_U32 = A_BYTES / 4;  // 128
    constexpr int B_U32 = B_BYTES / 4;  //  32
    uint32_t* s_a_u32 = reinterpret_cast<uint32_t*>(s_a);
    uint32_t* s_b_u32 = reinterpret_cast<uint32_t*>(s_b);
    const uint32_t* a_u32_in = reinterpret_cast<const uint32_t*>(a_bytes_fp8);
    const uint32_t* b_u32_in = reinterpret_cast<const uint32_t*>(b_bytes_nvfp4);
    for (int i = tid; i < A_U32; i += 32) s_a_u32[i] = a_u32_in[i];
    for (int i = tid; i < B_U32; i += 32) s_b_u32[i] = b_u32_in[i];
    __syncthreads();

    // Pack A fragment once — shared across the 2 MMAs.
    uint32_t a_frag[4];
    rvllm::pack_a_frag_row_major_m16k32(s_a, /*stride_bytes=*/32, a_frag, tid);

    // Scale loading — per D-output layout (not per B-input layout).
    //
    // Standard m16n8k32 D-output mapping for this lane:
    //   d[0] → D[tid/4    , (tid%4)*2    ]   n_a = (tid%4)*2
    //   d[1] → D[tid/4    , (tid%4)*2 + 1]   n_b = (tid%4)*2 + 1
    //   d[2] → D[tid/4 + 8, (tid%4)*2    ]   n_a (same as d[0])
    //   d[3] → D[tid/4 + 8, (tid%4)*2 + 1]   n_b (same as d[1])
    //
    // Each d[r] output for MMA 0 should be multiplied by
    // b_scales[its_n_col, K-block 0], and for MMA 1 by
    // b_scales[..., K-block 1]. So each lane loads 4 scales: 2 n-cols ×
    // 2 K-blocks.
    const int n_a = (tid & 3) * 2;
    const int n_b = n_a + 1;
    float sA_k0, sA_k1, sB_k0, sB_k1;
    {
        __half_raw h;
        h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b_scales[n_a*2 + 0], __NV_E4M3);
        sA_k0 = __half2float(__half(h));
        h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b_scales[n_a*2 + 1], __NV_E4M3);
        sA_k1 = __half2float(__half(h));
        h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b_scales[n_b*2 + 0], __NV_E4M3);
        sB_k0 = __half2float(__half(h));
        h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b_scales[n_b*2 + 1], __NV_E4M3);
        sB_k1 = __half2float(__half(h));
    }

    // --- MMA 0: K-block 0 active ---
    uint32_t b_frag0[2];
    rvllm_nvfp4::pack_b_frag_e4m3_x_e2m1_m16k32_path_a(
        s_b, /*stride_bytes_nvfp4=*/16, /*kblk_idx=*/0, b_frag0, tid);
    float d0[4] = {0, 0, 0, 0};
    rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d0, a_frag, b_frag0);
    d0[0] *= sA_k0;
    d0[1] *= sB_k0;
    d0[2] *= sA_k0;
    d0[3] *= sB_k0;

    // --- MMA 1: K-block 1 active ---
    uint32_t b_frag1[2];
    rvllm_nvfp4::pack_b_frag_e4m3_x_e2m1_m16k32_path_a(
        s_b, 16, /*kblk_idx=*/1, b_frag1, tid);
    float d1[4] = {0, 0, 0, 0};
    rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d1, a_frag, b_frag1);
    d1[0] *= sA_k1;
    d1[1] *= sB_k1;
    d1[2] *= sA_k1;
    d1[3] *= sB_k1;

    // Sum: each d[r] is the scaled K-block 0 and K-block 1 contribution
    // to the SAME D cell. Their sum is D[m, n].
    float d[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) d[r] = d0[r] + d1[r];

    rvllm_nvfp4::unpack_d_frag_to_smem_m16n8(
        s_d, /*stride_bytes=*/8 * (int)sizeof(float), d, tid);
    __syncthreads();
    constexpr int D_F32 = 16 * 8;
    for (int i = tid; i < D_F32; i += 32) d_tile_out[i] = s_d[i];
#else
    (void)a_bytes_fp8; (void)b_bytes_nvfp4; (void)b_scales;
    if (threadIdx.x < 32) {
        for (int i = 0; i < 16 * 8 / 32; ++i)
            d_tile_out[threadIdx.x + i * 32] = 0.0f;
    }
#endif
}

extern "C"
__global__ void nvfp4_mma_v2_probe_kernel(
    const uint32_t* __restrict__ a_frag,   // [32 × 4 u32]
    const uint32_t* __restrict__ b_frag,   // [32 × 2 u32]
    const uint32_t* __restrict__ sfa_frag, // [32]  per-lane u32 (4 × ue4m3)
    const uint32_t* __restrict__ sfb_frag, // [32]  per-lane u32
    float*          __restrict__ d_out     // [32 × 4 f32]
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x;
    uint32_t a[4] = {
        a_frag[lane * 4 + 0], a_frag[lane * 4 + 1],
        a_frag[lane * 4 + 2], a_frag[lane * 4 + 3],
    };
    uint32_t b[2] = { b_frag[lane * 2 + 0], b_frag[lane * 2 + 1] };
    uint32_t sfa  = sfa_frag[lane];
    uint32_t sfb  = sfb_frag[lane];
    float d[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d, a, b, sfa, sfb);
    d_out[lane * 4 + 0] = d[0];
    d_out[lane * 4 + 1] = d[1];
    d_out[lane * 4 + 2] = d[2];
    d_out[lane * 4 + 3] = d[3];
#else
    (void)a_frag; (void)b_frag; (void)sfa_frag; (void)sfb_frag;
    if (threadIdx.x < 32) {
        d_out[threadIdx.x * 4 + 0] = 0.0f;
        d_out[threadIdx.x * 4 + 1] = 0.0f;
        d_out[threadIdx.x * 4 + 2] = 0.0f;
        d_out[threadIdx.x * 4 + 3] = 0.0f;
    }
#endif
}

extern "C"
__global__ void nvfp4_mma_probe_kernel(
    const uint32_t* __restrict__ a_frag,  // [32 lanes × 4 u32] — per-lane A
    const uint32_t* __restrict__ b_frag,  // [32 lanes × 2 u32] — per-lane B
    float*          __restrict__ d_out    // [32 lanes × 4] — per-lane D
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x;
    uint32_t a[4] = {
        a_frag[lane * 4 + 0], a_frag[lane * 4 + 1],
        a_frag[lane * 4 + 2], a_frag[lane * 4 + 3],
    };
    uint32_t b[2] = { b_frag[lane * 2 + 0], b_frag[lane * 2 + 1] };
    float d[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    rvllm_nvfp4::zero_mma_d_frag(d);
    rvllm_nvfp4::mma_m16n8k32_e2m1_e2m1_f32(d, a, b);
    d_out[lane * 4 + 0] = d[0];
    d_out[lane * 4 + 1] = d[1];
    d_out[lane * 4 + 2] = d[2];
    d_out[lane * 4 + 3] = d[3];
#else
    (void)a_frag; (void)b_frag;
    if (threadIdx.x < 32) {
        d_out[threadIdx.x * 4 + 0] = 0.0f;
        d_out[threadIdx.x * 4 + 1] = 0.0f;
        d_out[threadIdx.x * 4 + 2] = 0.0f;
        d_out[threadIdx.x * 4 + 3] = 0.0f;
    }
#endif
}

// Packed-input variant: stage A and B tiles into shared memory in
// the standard NVFP4 byte layout (2 e2m1 per byte), pack the per-lane
// fragments via the header helpers, run the MMA, unpack D to a
// [16 × 8] f32 tile back in device memory.
//
// Block: 32 threads (single warp). Grid: (1, 1, 1). Dynamic smem
// sized from host = A_BYTES + B_BYTES + D_BYTES, tightly packed.
extern "C"
__global__ void nvfp4_mma_packed_probe_kernel(
    const unsigned char* __restrict__ a_bytes_in,  // 16 * 32 bytes, row-major
    const unsigned char* __restrict__ b_bytes_in,  //  8 * 32 bytes, col-major along n
    float*               __restrict__ d_tile_out   // 16 * 8 f32 = 512 bytes
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    extern __shared__ unsigned char smem[];
    constexpr int A_BYTES = 16 * 32;  // m=16, k=64 e2m1 → 32 bytes/row
    constexpr int B_BYTES =  8 * 32;  // n=8,  k=64 e2m1 → 32 bytes/col
    unsigned char* s_a = smem;
    unsigned char* s_b = smem + A_BYTES;
    float*         s_d = reinterpret_cast<float*>(s_b + B_BYTES);

    const int tid = threadIdx.x;

    constexpr int A_U32 = A_BYTES / 4;   // 128
    constexpr int B_U32 = B_BYTES / 4;   //  64
    uint32_t* s_a_u32 = reinterpret_cast<uint32_t*>(s_a);
    uint32_t* s_b_u32 = reinterpret_cast<uint32_t*>(s_b);
    const uint32_t* a_u32_in = reinterpret_cast<const uint32_t*>(a_bytes_in);
    const uint32_t* b_u32_in = reinterpret_cast<const uint32_t*>(b_bytes_in);
    for (int i = tid; i < A_U32; i += 32) s_a_u32[i] = a_u32_in[i];
    for (int i = tid; i < B_U32; i += 32) s_b_u32[i] = b_u32_in[i];
    __syncthreads();

    uint32_t a_frag[4];
    uint32_t b_frag[2];
    rvllm_nvfp4::pack_a_frag_e2m1_m16k64(s_a, /*stride_bytes=*/32, a_frag, tid);
    rvllm_nvfp4::pack_b_frag_e2m1_n8k64 (s_b, /*stride_bytes=*/32, b_frag, tid);

    // Scales = 1.0 everywhere (ue4m3 encoding 0x38 packed 4×).
    const uint32_t sfa_one = 0x38383838u;
    const uint32_t sfb_one = 0x38383838u;

    float d[4];
    rvllm_nvfp4::zero_mma_d_frag(d);
    rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d, a_frag, b_frag, sfa_one, sfb_one);

    rvllm_nvfp4::unpack_d_frag_to_smem_m16n8(
        s_d, /*stride_bytes=*/8 * (int)sizeof(float), d, tid);
    __syncthreads();

    constexpr int D_F32 = 16 * 8;
    for (int i = tid; i < D_F32; i += 32) d_tile_out[i] = s_d[i];
#else
    (void)a_bytes_in; (void)b_bytes_in;
    if (threadIdx.x < 32) {
        for (int i = 0; i < 16 * 8 / 32; ++i) {
            d_tile_out[threadIdx.x + i * 32] = 0.0f;
        }
    }
#endif
}
