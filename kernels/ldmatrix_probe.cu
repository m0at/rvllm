// F10 probe — compare `ldmatrix.sync.aligned.m8n8.x4.shared.b16` output
// against our current manual `pack_a_frag_row_major_m16k32` byte
// layout.
//
// We fill a 16×32-byte shared tile (the MMA A-fragment shape) with a
// deterministic pattern: s[row, col] = (row << 4) | (col & 0x0F).
// Then both paths read it per-lane and write their 4 u32 outputs to
// global memory. A Python harness dumps both and computes the lane
// permutation (if any) between ldmatrix output and the MMA spec.

#include <cstdint>
#include "fp8_mma_frag_pack.cuh"

extern "C" __global__ void ldmatrix_a_probe_kernel(
    uint32_t* __restrict__ d_manual,  // [32 lanes × 4 u32]
    uint32_t* __restrict__ d_ldmat)   // [32 lanes × 4 u32]
{
    __shared__ unsigned char s_tile[16 * 32];  // 512 bytes
    const int tid = threadIdx.x;
    if (tid < 128) {
        // Deterministic fill. 128 threads × 4 bytes = 512 bytes.
        const int byte_idx = tid * 4;
        const int row = byte_idx / 32;
        const int col = byte_idx & 31;
        uint32_t v = 0;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint32_t byte = ((row & 0xF) << 4) | ((col + b) & 0xF);
            v |= byte << (b * 8);
        }
        *reinterpret_cast<uint32_t*>(s_tile + byte_idx) = v;
    }
    __syncthreads();
    if (tid >= 32) return;

    // Path 1 — existing manual pack.
    uint32_t a_man[4];
    rvllm::pack_a_frag_row_major_m16k32(s_tile, /*stride=*/32, a_man, tid);
    d_manual[tid * 4 + 0] = a_man[0];
    d_manual[tid * 4 + 1] = a_man[1];
    d_manual[tid * 4 + 2] = a_man[2];
    d_manual[tid * 4 + 3] = a_man[3];

    // Path 2 — single ldmatrix load.
    // `.m8n8.x4.b16` loads 4 matrices of 8×8 b16 elements each,
    // collectively 512 bytes. Source addressing uses a per-lane
    // address; conventional layout is lanes 0-7 supply matrix 0's
    // row start pointers, lanes 8-15 → matrix 1, lanes 16-23 → 2,
    // lanes 24-31 → 3. For a row-major 16×32-byte tile arranged as
    // M0 / M1 stacked vertically per (rows 0..7 / 8..15) and then
    // M2 / M3 by the upper 16 cols vs lower 16 cols, we feed four
    // row starts:
    //   M0: s_tile + (tid % 8) * 32               + 0
    //   M1: s_tile + ((tid % 8) + 8) * 32          + 0
    //   M2: s_tile + (tid % 8) * 32               + 16
    //   M3: s_tile + ((tid % 8) + 8) * 32          + 16
    // Lane `tid` picks one of these based on which submatrix group
    // its row-index falls into. Empirically verified by the Python
    // harness.
    const int lane_mod8 = tid & 7;
    const int group = tid >> 3;   // 0..3 — picks M0..M3
    const unsigned char* base;
    switch (group) {
        case 0:  base = s_tile +  lane_mod8       * 32 +  0; break;
        case 1:  base = s_tile + (lane_mod8 + 8)  * 32 +  0; break;
        case 2:  base = s_tile +  lane_mod8       * 32 + 16; break;
        default: base = s_tile + (lane_mod8 + 8)  * 32 + 16; break;
    }
    const uint32_t smem_addr =
        __cvta_generic_to_shared(const_cast<unsigned char*>(base));

    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
    d_ldmat[tid * 4 + 0] = r0;
    d_ldmat[tid * 4 + 1] = r1;
    d_ldmat[tid * 4 + 2] = r2;
    d_ldmat[tid * 4 + 3] = r3;
}
