// Standalone probe kernel for `nvfp4_utils.cuh`: takes a buffer of
// FP32 values in 16-element blocks, packs each block into NVFP4
// (4-bit + one E4M3 scale per block), then unpacks it back. Caller
// compares the round-tripped FP32 against the original to gate the
// packing / dequant math before it lands in the FA2 KV kernels.
//
// Build (together with the full kernels/build.sh per-arch suite, or
// standalone with the one-liner below):
//   nvcc -O3 -arch=sm_121a -std=c++17 -ptx \
//        -o kernels/sm_121/nvfp4_roundtrip_probe.ptx \
//        kernels/nvfp4_roundtrip_probe.cu
//
// Exposed kernel:
//   nvfp4_roundtrip_kernel(
//       float*       out,        // [N] round-tripped FP32
//       const float* in,         // [N] source FP32
//       int          n_blocks    // N / 16
//   )
//
// Grid: (ceil(n_blocks / 32), 1, 1)   Block: (32, 1, 1)   One thread
// per 16-element block. Perf is not the point — this TU exists to
// validate the helpers, not to be on the hot path.

#include "nvfp4_utils.cuh"

extern "C" __global__ void nvfp4_roundtrip_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int n_blocks
) {
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;

    const float* src = in + blk * 16;
    float*       dst = out + blk * 16;

    // Stage source into registers (the helpers take raw FP32 arrays —
    // passing through registers skips a redundant memory bounce).
    float block[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) block[i] = src[i];

    __nv_fp8_e4m3 scale = rvllm_nvfp4::block_scale_e4m3(block);

    uint8_t packed[8];
    rvllm_nvfp4::pack16_fp32_to_nvfp4(block, scale, packed);

    float recovered[16];
    rvllm_nvfp4::unpack16_nvfp4_to_fp32(packed, scale, recovered);

    #pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = recovered[i];
}
