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
#include <cstdint>
#include "nvfp4_mma_frag_pack.cuh"

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

    // Cooperative load of A (512 bytes) + B (256 bytes) = 768 bytes
    // via 32 threads. 768 / 32 = 24 bytes per lane; we load 4-byte
    // aligned u32s to keep the smem writes wide.
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

    float d[4];
    rvllm_nvfp4::zero_mma_d_frag(d);
    rvllm_nvfp4::mma_m16n8k32_e2m1_e2m1_f32(d, a_frag, b_frag);

    rvllm_nvfp4::unpack_d_frag_to_smem_m16n8(
        s_d, /*stride_bytes=*/8 * (int)sizeof(float), d, tid);
    __syncthreads();

    // Stream smem D tile to device memory, 4 f32 per lane.
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
