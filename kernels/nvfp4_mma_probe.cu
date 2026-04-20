// Standalone "hello world" for the Blackwell native E2M1 tensor-core
// MMA: `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32`.
// Issues one warp-cooperative MMA with fixed inputs and writes the
// 32 accumulator lanes back out for a host-side fp64 compare.
//
// Purpose: validate that the PTX path assembles on sm_121a, the
// operand packing matches the CUTLASS spec we'll reuse in the full
// FA2 rewrite, and we see plausible fp32 values (not NaN / zeros).
// The full FA2 integration of native e2m1 MMA is tracked as a
// follow-up in GB10_SPEC.md — this probe lets us develop the
// register layout + PTX string in isolation.
//
// Tile: m=16, n=8, k=32. Warp size 32. Per-lane fragments:
//   A: 4 × uint32_t → 128 bits = 32 E2M1 values covering a 16×32
//      portion of A, distributed across the warp.
//   B: 2 × uint32_t → 64 bits  = 16 E2M1 values covering an 8×32
//      portion of B.
//   D: 4 × float    → 128 bits  = 4 f32 acc slots.
//
// The operand-lane layout is fixed by the PTX: for A (row-major
// m16k32), lane `i` holds rows {i/4, i/4+8} cols {(i%4)*8..(i%4)*8+7}
// and the pack order per u32 is {k0..k7 of row r0, k0..k7 of row r1,
// k8..k15 of r0, k8..k15 of r1}. For B (col-major n8k32), similar
// distribution along the n×k tile. Full fragment spec is the
// reference `mma.sync` PTX ISA (8.8+).

#include <cuda_fp16.h>
#include <cstdint>

extern "C"
__global__ void nvfp4_mma_probe_kernel(
    const uint32_t* __restrict__ a_frag,  // [32 lanes × 4 u32] — per-lane A
    const uint32_t* __restrict__ b_frag,  // [32 lanes × 2 u32] — per-lane B
    float*          __restrict__ d_out    // [32 lanes × 4] — per-lane D
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
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
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
