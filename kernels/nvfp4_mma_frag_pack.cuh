// **WIP** — layout not yet validated against fp64 reference. Do NOT
// use this header from the FA2 integration until the packer matches
// the PTX thread mapping for e2m1 operands (see TODO below).
//
// Fragment-packing helpers for the Blackwell native NVFP4 (e2m1)
// tensor-core MMA
//     mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32
//
// Current status: assembling PTX + launching the MMA works (zero-
// input smoke test in nvfp4_mma_probe.cu passes, output finite). But
// the byte-level packer inherited from `fp8_mma_frag_pack.cuh` does
// NOT produce correct MMA output for e2m1 — an all-ones A × all-ones
// B with K=64 returns 2.0 everywhere (not 64.0), and a K=32
// interpretation returns the same 2.0. The f8f6f4 family keeps the
// register shape (A: 4×u32, B: 2×u32) across types, but the byte
// position → matrix element mapping for e2m1 is not simply "2 e2m1
// per byte, same byte positions as FP8". Next step is a one-hot
// probe sweep (set exactly one byte-bit at a time, measure which
// output slot moves) to derive the real lane→(row, col) mapping.
//
// Until that lands, the FA2 decode/prefill kernels remain on the
// dequant-to-fp32-smem + scalar-FMA path (`flash_attention_nvfp4kv.cu`).
//
// Layout note — the PTX `f8f6f4` family takes the SAME per-lane
// register shape (A: 4×u32, B: 2×u32, D: 4×f32) for every supported
// element type; the hardware picks 8 / 6 / 4 bits per element based
// on the type tag. The byte-level storage in shared memory is
// therefore identical to the e4m3 variant, and we can REUSE the
// byte-granularity packers from `fp8_mma_frag_pack.cuh` verbatim.
// The semantic change is that each byte now holds TWO e2m1 values
// (low nibble = even col, high nibble = odd col), so the effective
// K dimension per MMA is 64 e2m1 elements (vs 32 for e4m3). Callers
// must reason about tile bounds in *bytes* (one byte = two e2m1),
// and in the K direction advance in strides of 32 bytes per MMA.
//
// In other words:
//  * smem A tile:  16 rows × (k_elems / 2) bytes — row stride is
//                  whatever the producer wrote (typically k_elems/2).
//  * smem B tile:   8 rows × (k_elems / 2) bytes — col-major along N.
//  * One MMA call  consumes 16×64 e2m1 of A and 8×64 e2m1 of B,
//                  i.e. 32 bytes of K stride.
//
// The accumulator unpack layout is also identical to the FP8 path
// (the MMA output is f32 regardless of operand type), so
// `unpack_d_frag_to_smem_m16n8` from `fp8_mma_frag_pack.cuh` drops
// in unchanged.
//
// Unit-tested by the upgraded `nvfp4_mma_probe.cu` +
// `v3/tools/nvfp4_mma_probe_check.py` which now drive the kernel
// with non-trivial inputs and compare against an fp64 reference.

#pragma once

#include <cstdint>
#include "fp8_mma_frag_pack.cuh"

namespace rvllm_nvfp4 {

// Re-export the byte-level packers under the NVFP4 namespace. The
// byte-level semantics are identical; the only difference is the
// element interpretation at the MMA site.
__device__ __forceinline__ void pack_a_frag_e2m1_m16k64(
    const unsigned char* smem_a,
    int                  stride_bytes,   // row stride in BYTES
    uint32_t             a[4],
    int                  lane)
{
    // Byte layout matches e4m3 — reuse the packer.
    rvllm::pack_a_frag_row_major_m16k32(smem_a, stride_bytes, a, lane);
}

__device__ __forceinline__ void pack_b_frag_e2m1_n8k64(
    const unsigned char* smem_b,
    int                  stride_bytes,   // col stride in BYTES
    uint32_t             b[2],
    int                  lane)
{
    rvllm::pack_b_frag_col_major_n8k32(smem_b, stride_bytes, b, lane);
}

__device__ __forceinline__ void unpack_d_frag_to_smem_m16n8(
    float*       smem_d,
    int          stride_bytes,
    const float  d[4],
    int          lane)
{
    rvllm::unpack_d_frag_to_smem_m16n8(smem_d, stride_bytes, d, lane);
}

__device__ __forceinline__ void zero_mma_d_frag(float d[4]) {
    rvllm::zero_mma_d_frag(d);
}

// Thin wrapper around the PTX. Same `"+f"` (read-write) constraint
// pattern as the e4m3 variant — see the comment on
// `mma_m16n8k32_e4m3_e4m3_f32` for the F6 dedup bug history.
__device__ __forceinline__ void mma_m16n8k32_e2m1_e2m1_f32(
    float           d[4],
    const uint32_t  a[4],
    const uint32_t  b[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
#else
    (void)a; (void)b;
    d[0] = d[1] = d[2] = d[3] = 0.0f;
#endif
}

} // namespace rvllm_nvfp4
