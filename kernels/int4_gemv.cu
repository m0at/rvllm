#include <cuda_fp16.h>

// Fused INT4 (GPTQ) GEMV with per-group asymmetric dequantization.
//
// Computes output[M,N] = input[M,K] @ dequant(qweight[N,K])^T
// where dequant(q) = (q - zero_point) * scale
//
// Weight layout (repacked for coalesced reads during loading):
//   qweight: [N, K/8] as int32, row-major (8 unsigned INT4 per int32)
//   scales:  [N, num_groups] as f32, row-major
//   zeros:   [N, num_groups] as f32, row-major (pre-unpacked)
//   input:   [M, K] as f32
//   output:  [M, N] as f32
//
// INT4 packing in each int32:
//   bits[0:4]   = element k+0
//   bits[4:8]   = element k+1
//   bits[8:12]  = element k+2
//   bits[12:16] = element k+3
//   bits[16:20] = element k+4
//   bits[20:24] = element k+5
//   bits[24:28] = element k+6
//   bits[28:32] = element k+7
//
// Values are unsigned [0, 15]. Dequant: (val - zero_point) * scale
//
// SM_121 (GB10 Blackwell consumer) constraints:
//   - Scalar 32-bit loads only (no float4/uint4 -- causes hang)
//   - Static shared memory <= 32 bytes
//   - Single __syncthreads() at end for warp reduction
//
// Bandwidth: reads K/2 bytes of weight per output element (vs K for FP8).
// Expected ~2x throughput improvement over FP8 GEMV.
//
// Launch config:
//   Grid:  (N, M, 1)   -- one block per output element
//   Block: (256, 1, 1)  -- 256 threads cooperatively reduce over K
//   Shared: 8 * sizeof(float) = 32 bytes (warp reduction)

extern "C"
__global__ void int4_gemv_kernel(
    float* __restrict__ output,
    const int* __restrict__ qweight,      // [N, K/8] repacked int32
    const float* __restrict__ scales,     // [N, num_groups]
    const float* __restrict__ zeros,      // [N, num_groups] (pre-unpacked)
    const float* __restrict__ input,      // [M, K]
    int M, int N, int K,
    int group_size,                       // typically 128
    int num_groups                        // K / group_size
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;
    const int K_PACKED = K / 8;           // number of int32s per row

    const int* w_row = qweight + (long long)n * K_PACKED;
    const float* x_row = input + (long long)m * K;
    const float* s_row = scales + (long long)n * num_groups;
    const float* z_row = zeros + (long long)n * num_groups;

    // Number of packed int32s per group: group_size / 8
    const int packs_per_group = group_size >> 3;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Main loop: each thread processes one int32 (8 INT4 values) per iteration.
    // Two iterations unrolled for ILP.
    for (int pk = threadIdx.x; pk < K_PACKED; pk += BLOCK_DIM) {
        int packed = __ldg(w_row + pk);
        int k_base = pk * 8;

        // Group index: k_base / group_size = pk / packs_per_group
        int group = pk / packs_per_group;
        float s = __ldg(s_row + group);
        float z = __ldg(z_row + group);

        // Unpack 8 nibbles and dequantize: (nibble - zero) * scale * input
        float q0 = (float)((packed >>  0) & 0xF);
        float q1 = (float)((packed >>  4) & 0xF);
        float q2 = (float)((packed >>  8) & 0xF);
        float q3 = (float)((packed >> 12) & 0xF);
        float q4 = (float)((packed >> 16) & 0xF);
        float q5 = (float)((packed >> 20) & 0xF);
        float q6 = (float)((packed >> 24) & 0xF);
        float q7 = (float)((packed >> 28) & 0xF);

        acc0 += (q0 - z) * s * __ldg(x_row + k_base);
        acc0 += (q1 - z) * s * __ldg(x_row + k_base + 1);
        acc0 += (q2 - z) * s * __ldg(x_row + k_base + 2);
        acc0 += (q3 - z) * s * __ldg(x_row + k_base + 3);
        acc1 += (q4 - z) * s * __ldg(x_row + k_base + 4);
        acc1 += (q5 - z) * s * __ldg(x_row + k_base + 5);
        acc1 += (q6 - z) * s * __ldg(x_row + k_base + 6);
        acc1 += (q7 - z) * s * __ldg(x_row + k_base + 7);
    }

    float acc = acc0 + acc1;

    // Remainder: handle K not divisible by 8 (rare with standard GPTQ)
    {
        int aligned_k = K_PACKED * 8;
        for (int kr = aligned_k + threadIdx.x; kr < K; kr += BLOCK_DIM) {
            int group = kr / group_size;
            float s = __ldg(s_row + group);
            float z = __ldg(z_row + group);
            // Read single nibble from packed data
            int pk_idx = kr / 8;
            int nibble_idx = kr % 8;
            int packed = __ldg(w_row + pk_idx);
            float q = (float)((packed >> (nibble_idx * 4)) & 0xF);
            acc += (q - z) * s * __ldg(x_row + kr);
        }
    }

    // --- Warp-level reduction ---
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // --- Inter-warp reduction via shared memory ---
    __shared__ float warp_sums[BLOCK_DIM / 32];   // 8 floats = 32 bytes
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        acc = (lane < (BLOCK_DIM / 32)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_DIM / 64); offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            output[(long long)m * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// INT4 dequantization kernel: unpack GPTQ int4 to f16 for HGEMM (prefill).
//
// Used when M > 32 (prefill): dequant once, then cuBLAS HGEMM.
// ---------------------------------------------------------------------------
extern "C"
__global__ void dequant_int4_to_f16_kernel(
    __half* __restrict__ output,           // [N, K] as f16
    const int* __restrict__ qweight,       // [N, K/8] repacked int32
    const float* __restrict__ scales,      // [N, num_groups]
    const float* __restrict__ zeros,       // [N, num_groups]
    int N, int K,
    int group_size,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int n = idx / K;
    int k = idx % K;

    // Read packed value
    int pk_idx = k / 8;
    int nibble_idx = k % 8;
    int packed = __ldg(qweight + (long long)n * (K / 8) + pk_idx);
    float q = (float)((packed >> (nibble_idx * 4)) & 0xF);

    // Dequantize
    int group = k / group_size;
    float s = __ldg(scales + (long long)n * num_groups + group);
    float z = __ldg(zeros + (long long)n * num_groups + group);
    float val = (q - z) * s;

    output[idx] = __float2half(val);
}
