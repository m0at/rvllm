// Phase F2 probe — exercises the smem → fragment packers from
// `kernels/fp8_mma_frag_pack.cuh` end-to-end. The host loads A / B
// as flat row-major / col-major FP8 tiles into smem, the kernel
// calls the packing helpers on each lane, issues one MMA, and writes
// the D tile back to gmem via the smem unpacker.
//
// If this passes bit-for-bit against an fp64 reference, the packers
// can be lifted into the real FA2 prefill kernel without further
// lane-layout debugging.

#include <cstdint>

#include "fp8_mma_frag_pack.cuh"

extern "C"
__global__ void fp8_mma_smem_probe_kernel(
    const unsigned char* __restrict__ a_tile,  // [16][32] row-major
    const unsigned char* __restrict__ b_tile,  // [8][32]  col-major ([N][K])
    float*               __restrict__ d_out    // [16][8]  row-major
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr int M = 16, N = 8, K = 32;

    extern __shared__ unsigned char smem_raw[];
    unsigned char* s_a = smem_raw;                     // [M][K] = 512 B
    unsigned char* s_b = s_a + M * K;                  // [N][K] = 256 B
    float*         s_d = reinterpret_cast<float*>(s_b + N * K); // [M][N]

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    constexpr int THREADS = 32; // one warp

    // Cooperative load A / B into smem. Using u32 copies to match the
    // alignment the packers will reuse.
    for (int i = tid; i < (M * K) / 4; i += THREADS) {
        reinterpret_cast<uint32_t*>(s_a)[i] =
            reinterpret_cast<const uint32_t*>(a_tile)[i];
    }
    for (int i = tid; i < (N * K) / 4; i += THREADS) {
        reinterpret_cast<uint32_t*>(s_b)[i] =
            reinterpret_cast<const uint32_t*>(b_tile)[i];
    }
    __syncthreads();

    // Pack per-lane fragments straight out of smem.
    uint32_t a_frag[4];
    uint32_t b_frag[2];
    float    d_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    rvllm::pack_a_frag_row_major_m16k32(s_a, /*stride=*/K, a_frag, lane);
    rvllm::pack_b_frag_col_major_n8k32 (s_b, /*stride=*/K, b_frag, lane);

    // One MMA, accumulate into d_frag.
    rvllm::mma_m16n8k32_e4m3_e4m3_f32(d_frag, a_frag, b_frag, d_frag);

    // Unpack D back into a row-major [M][N] tile in smem, then dump.
    rvllm::unpack_d_frag_to_smem_m16n8(
        s_d, /*stride_bytes=*/N * sizeof(float), d_frag, lane);
    __syncthreads();

    for (int i = tid; i < M * N; i += THREADS) {
        d_out[i] = s_d[i];
    }
#else
    (void)a_tile; (void)b_tile;
    if (threadIdx.x < 16 * 8) {
        d_out[threadIdx.x] = 0.0f;
    }
#endif
}
