// Standalone "hello world" for the Blackwell FP8 E4M3 tensor-core MMA:
//     mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
//
// Motivation: see `v3/UNIFIED_PREFILL_MMA_PLAN.md`. Phase F1 proves
// the PTX assembles on sm_121, the per-lane fragment layout matches
// what we'll pack in the full FA2 rewrite, and the output is within
// FP8-quant noise of an fp64 reference.
//
// Tile shape per warp: M=16, N=8, K=32.
//
// Per-lane fragment layout (lane ∈ [0, 31]):
//
// A (m16 × k32) row-major — 16 elements (16 FP8 bytes, 4 × u32):
//     lane i covers rows {i/4, i/4+8}, k = (i%4)*8 + [0..7]
//     a[0]: row (i/4),    k = (i%4)*8 + [0..3]
//     a[1]: row (i/4+8),  k = (i%4)*8 + [0..3]
//     a[2]: row (i/4),    k = (i%4)*8 + [4..7]
//     a[3]: row (i/4+8),  k = (i%4)*8 + [4..7]
//
// B (n8 × k32) col-major — 8 elements (8 FP8 bytes, 2 × u32):
//     lane i covers col (i/4), k = (i%4)*8 + [0..7]
//     b[0]: col (i/4), k = (i%4)*8 + [0..3]
//     b[1]: col (i/4), k = (i%4)*8 + [4..7]
//
// D (m16 × n8) f32 — 4 elements per lane:
//     lane i: rows {i/4, i/4+8}, cols (i%4)*2 + [0..1]
//     d[0]: row (i/4),    col (i%4)*2 + 0
//     d[1]: row (i/4),    col (i%4)*2 + 1
//     d[2]: row (i/4+8),  col (i%4)*2 + 0
//     d[3]: row (i/4+8),  col (i%4)*2 + 1
//
// This file is one warp doing one MMA. The host-side harness packs
// known inputs into the per-lane fragments, launches the kernel,
// reads back the fragments, and compares against an fp64 reference.
// No smem, no ldmatrix — those are layered on in later phases.

#include <cstdint>

extern "C"
__global__ void fp8_e4m3_mma_probe_kernel(
    const uint32_t* __restrict__ a_frag,  // [32 lanes × 4 u32]
    const uint32_t* __restrict__ b_frag,  // [32 lanes × 2 u32]
    float*          __restrict__ d_out    // [32 lanes × 4 f32]
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x;
    uint32_t a0 = a_frag[lane * 4 + 0];
    uint32_t a1 = a_frag[lane * 4 + 1];
    uint32_t a2 = a_frag[lane * 4 + 2];
    uint32_t a3 = a_frag[lane * 4 + 3];
    uint32_t b0 = b_frag[lane * 2 + 0];
    uint32_t b1 = b_frag[lane * 2 + 1];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3)
    );

    d_out[lane * 4 + 0] = d0;
    d_out[lane * 4 + 1] = d1;
    d_out[lane * 4 + 2] = d2;
    d_out[lane * 4 + 3] = d3;
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
