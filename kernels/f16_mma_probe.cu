// Numerical-correctness probe for the SM80-era f16 tensor-core MMA
//     mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
// De-risks Phase 1 of task aa01001nvf4f16mma: validate the per-lane
// packer layout against an fp64 reference BEFORE touching the FA2
// kernel inner loop. The aa01001nvf4mma0 attempt with the e2m1
// tensor-core MMA got stuck here precisely because the layout was
// opaque; the m16n8k16 f16 MMA is a well-documented SM80 instruction
// with stable, vendor-published per-lane mapping. If this probe is
// green, porting the FA2 inner loop is a straight-line edit.
//
// Kernel staging: takes A ([16 × 16] f16, row-major) and B
// ([8 × 16] f16, col-major along N) from device memory, copies
// into smem via cooperative loads, packs the per-lane fragments
// through `f16_mma_frag_pack.cuh`, runs one MMA, unpacks the
// 16×8 f32 output tile back to device memory. Warp-sized block
// (32 threads), single-block launch.

#include <cuda_fp16.h>
#include <cstdint>
#include "f16_mma_frag_pack.cuh"

extern "C"
__global__ void f16_mma_probe_kernel(
    const __half* __restrict__ a_in,      // [16 × 16] f16, row-major
    const __half* __restrict__ b_in,      // [ 8 × 16] f16, col-major along n (storage [N][K])
    float*        __restrict__ d_out      // [16 × 8]  f32, row-major
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ unsigned char smem[];
    constexpr int A_BYTES = 16 * 16 * (int)sizeof(__half);  // 512 B
    constexpr int B_BYTES =  8 * 16 * (int)sizeof(__half);  // 256 B
    __half* s_a = reinterpret_cast<__half*>(smem);
    __half* s_b = reinterpret_cast<__half*>(smem + A_BYTES);
    float*  s_d = reinterpret_cast<float* >(smem + A_BYTES + B_BYTES);

    const int tid = threadIdx.x;

    // Cooperative H2S load via u32 writes (768 bytes → 192 u32 → 6 per
    // lane). Keep wide stores so the smem stores stay coalesced-ish.
    constexpr int A_U32 = A_BYTES / 4;   // 128
    constexpr int B_U32 = B_BYTES / 4;   //  64
    uint32_t*       s_a_u32 = reinterpret_cast<uint32_t*>(s_a);
    uint32_t*       s_b_u32 = reinterpret_cast<uint32_t*>(s_b);
    const uint32_t* a_u32   = reinterpret_cast<const uint32_t*>(a_in);
    const uint32_t* b_u32   = reinterpret_cast<const uint32_t*>(b_in);
    for (int i = tid; i < A_U32; i += 32) s_a_u32[i] = a_u32[i];
    for (int i = tid; i < B_U32; i += 32) s_b_u32[i] = b_u32[i];
    __syncthreads();

    uint32_t a_frag[4];
    uint32_t b_frag[2];
    rvllm_f16mma::pack_a_frag_row_major_m16k16_f16(
        s_a, /*stride_bytes=*/16 * (int)sizeof(__half), a_frag, tid);
    rvllm_f16mma::pack_b_frag_col_major_n8k16_f16(
        s_b, /*stride_bytes=*/16 * (int)sizeof(__half), b_frag, tid);

    float d[4];
    rvllm_f16mma::zero_mma_d_frag(d);
    rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d, a_frag, b_frag);

    rvllm_f16mma::unpack_d_frag_to_smem_m16n8(
        s_d, /*stride_bytes=*/8 * (int)sizeof(float), d, tid);
    __syncthreads();

    // Stream the 16 × 8 f32 tile to device memory.
    constexpr int D_F32 = 16 * 8;
    for (int i = tid; i < D_F32; i += 32) d_out[i] = s_d[i];
#else
    (void)a_in; (void)b_in;
    if (threadIdx.x < 32) {
        for (int i = 0; i < 16 * 8 / 32; ++i) {
            d_out[threadIdx.x + i * 32] = 0.0f;
        }
    }
#endif
}
