// Warp-level MMA THROUGHPUT microbench (not LATENCY) for sm_121.
//
// The sibling `mma_throughput_bench.cu` has a dependent-accumulator
// loop (`+f(d)` → next MMA waits for prev) — it measures issue
// LATENCY, not throughput. This file fixes that: 8 independent
// per-warp accumulators unrolled in one loop so the tensor-core
// pipeline can issue without stalls, exposing real steady-state
// throughput.
//
// Methodology:
//   - 8 per-warp `d[4]` accumulators (d0..d7), zero-initialized.
//   - Inner loop: `for i in N/8: mma(d0,...); mma(d1,...); ... mma(d7,...)`
//     so 8 independent MMAs are in flight.
//   - Use clock64() delta for wall-cycles.
//   - Multiple resident warps (4 warps × 128 threads, 1 block) so the
//     warp scheduler sees contention — matches how FA2 kernels issue.
//   - "Best of 10 runs" reporting; kills cache effects.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "nvfp4_mma_frag_pack.cuh"
#include "f16_mma_frag_pack.cuh"

#ifndef BENCH_ITERS_INDEP
#define BENCH_ITERS_INDEP 4096
#endif

#define MMA_UNROLL 8

extern "C"
__global__ void mma_bench_indep_f16_k16_kernel(
    unsigned long long* out_cycles_per_warp, float* out_sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
    uint32_t b[2] = {0x3c003c00u, 0x3c003c00u};
    float d0[4] = {0}, d1[4] = {0}, d2[4] = {0}, d3[4] = {0};
    float d4[4] = {0}, d5[4] = {0}, d6[4] = {0}, d7[4] = {0};

    __syncthreads();
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < BENCH_ITERS_INDEP / MMA_UNROLL; ++i) {
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d0, a, b);
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d1, a, b);
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d2, a, b);
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d3, a, b);
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d4, a, b);
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d5, a, b);
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d6, a, b);
        rvllm_f16mma::mma_m16n8k16_f16_f16_f32(d7, a, b);
    }
    unsigned long long t1 = clock64();

    if (lane == 0) out_cycles_per_warp[warp] = t1 - t0;
    // Prevent DCE.
    float sink = d0[0] + d1[0] + d2[0] + d3[0]
               + d4[0] + d5[0] + d6[0] + d7[0];
    if (lane == 0 && warp == 0) *out_sink = sink;
#else
    (void)out_cycles_per_warp; (void)out_sink;
#endif
}

extern "C"
__global__ void mma_bench_indep_f8f6f4_e4m3_e2m1_k32_kernel(
    unsigned long long* out_cycles_per_warp, float* out_sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    uint32_t b[2] = {0x08080808u, 0x08080808u};
    float d0[4] = {0}, d1[4] = {0}, d2[4] = {0}, d3[4] = {0};
    float d4[4] = {0}, d5[4] = {0}, d6[4] = {0}, d7[4] = {0};

    __syncthreads();
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < BENCH_ITERS_INDEP / MMA_UNROLL; ++i) {
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d0, a, b);
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d1, a, b);
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d2, a, b);
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d3, a, b);
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d4, a, b);
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d5, a, b);
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d6, a, b);
        rvllm_nvfp4::mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(d7, a, b);
    }
    unsigned long long t1 = clock64();

    if (lane == 0) out_cycles_per_warp[warp] = t1 - t0;
    float sink = d0[0] + d1[0] + d2[0] + d3[0]
               + d4[0] + d5[0] + d6[0] + d7[0];
    if (lane == 0 && warp == 0) *out_sink = sink;
#else
    (void)out_cycles_per_warp; (void)out_sink;
#endif
}

extern "C"
__global__ void mma_bench_indep_mxf4nvf4_k64_kernel(
    unsigned long long* out_cycles_per_warp, float* out_sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint32_t a[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
    uint32_t b[2] = {0x22222222u, 0x22222222u};
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;
    float d0[4] = {0}, d1[4] = {0}, d2[4] = {0}, d3[4] = {0};
    float d4[4] = {0}, d5[4] = {0}, d6[4] = {0}, d7[4] = {0};

    __syncthreads();
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < BENCH_ITERS_INDEP / MMA_UNROLL; ++i) {
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d0, a, b, sfa, sfb);
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d1, a, b, sfa, sfb);
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d2, a, b, sfa, sfb);
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d3, a, b, sfa, sfb);
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d4, a, b, sfa, sfb);
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d5, a, b, sfa, sfb);
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d6, a, b, sfa, sfb);
        rvllm_nvfp4::mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(d7, a, b, sfa, sfb);
    }
    unsigned long long t1 = clock64();

    if (lane == 0) out_cycles_per_warp[warp] = t1 - t0;
    float sink = d0[0] + d1[0] + d2[0] + d3[0]
               + d4[0] + d5[0] + d6[0] + d7[0];
    if (lane == 0 && warp == 0) *out_sink = sink;
#else
    (void)out_cycles_per_warp; (void)out_sink;
#endif
}
