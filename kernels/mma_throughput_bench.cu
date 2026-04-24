// Warp-level MMA throughput microbench for sm_121 (GB10).
//
// Compares per-warp wall-clock MMA issue rate for three variants:
//   (a) m16n8k16 f16.f16.f32      — current dequant-to-f16 path
//   (b) m16n8k32 f8f6f4.e4m3.e2m1 — Path A (no scales)
//   (c) m16n8k64 mxf4nvf4.block_scale.scale_vec::4X — Path B
//
// All operands held in registers (no memory traffic). Zero-initialized
// so compiler can't dead-code the MMAs if we write the accumulator out
// after the loop (forces the dependency chain).
//
// Metric: MMA issues per second per warp. Scale by (flops per MMA) to
// get per-MMA FLOPS: (a) = 16*8*16*2 = 4096 flops, (b) = 16*8*32*2 =
// 8192 flops, (c) = 16*8*64*2 = 16384 flops.
//
// Measured with clock64() on-device. Single warp, 1 CUDA block.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "nvfp4_mma_frag_pack.cuh"
#include "f16_mma_frag_pack.cuh"

#ifndef BENCH_ITERS
#define BENCH_ITERS 8192
#endif

// (a) f16 m16n8k16
extern "C"
__global__ void mma_bench_f16_k16_kernel(unsigned long long* out_cycles,
                                         float* out_sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x;
    // Per-lane operand registers.
    uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};  // f16 all-ones
    uint32_t b[2] = {0x3c003c00u, 0x3c003c00u};
    float d[4] = {0, 0, 0, 0};

    __syncthreads();
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < BENCH_ITERS; ++i) {
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d, a, b);
    }
    unsigned long long t1 = clock64();

    if (lane == 0) {
        *out_cycles = t1 - t0;
    }
    // Prevent dead-code elimination.
    if (lane == 0) *out_sink = d[0] + d[1] + d[2] + d[3];
#else
    (void)out_cycles; (void)out_sink;
#endif
}

// (b) f8f6f4 m16n8k32 e4m3×e2m1 (Path A MMA, unscaled)
extern "C"
__global__ void mma_bench_f8f6f4_e4m3_e2m1_k32_kernel(
    unsigned long long* out_cycles, float* out_sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x;
    // e4m3 +1.0 = 0x38. Fill all 4 bytes with 0x38.
    uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    // e2m1 +1.0 shifted by 2 = 0x08 per byte. All bytes 0x08.
    uint32_t b[2] = {0x08080808u, 0x08080808u};
    float d[4] = {0, 0, 0, 0};

    __syncthreads();
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < BENCH_ITERS; ++i) {
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d, a, b);
    }
    unsigned long long t1 = clock64();

    if (lane == 0) {
        *out_cycles = t1 - t0;
    }
    if (lane == 0) *out_sink = d[0] + d[1] + d[2] + d[3];
#else
    (void)out_cycles; (void)out_sink;
#endif
}

// (c) mxf4nvf4 m16n8k64 (Path B MMA, block_scale scale_vec::4X)
extern "C"
__global__ void mma_bench_mxf4nvf4_k64_kernel(
    unsigned long long* out_cycles, float* out_sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x;
    // e2m1 +1.0 = 0x2 per nibble, 2 per byte → 0x22.
    uint32_t a[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
    uint32_t b[2] = {0x22222222u, 0x22222222u};
    // ue4m3 +1.0 = 0x38 per byte, 4 per u32.
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;
    float d[4] = {0, 0, 0, 0};

    __syncthreads();
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < BENCH_ITERS; ++i) {
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d, a, b, sfa, sfb);
    }
    unsigned long long t1 = clock64();

    if (lane == 0) {
        *out_cycles = t1 - t0;
    }
    if (lane == 0) *out_sink = d[0] + d[1] + d[2] + d[3];
#else
    (void)out_cycles; (void)out_sink;
#endif
}
