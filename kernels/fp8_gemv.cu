// Fused FP8 E4M3 GEMV with block-wise scales.
//
// Computes output[M,N] = input[M,K] @ weight_fp8[N,K]^T * block_scale
// without materializing a temporary FP16 weight buffer.
//
// For decode (small M), this is ~5x faster than dequant+HGEMM because it
// reads FP8 weights once from global memory instead of: read FP8 → write FP16
// → read FP16 for HGEMM. Memory traffic: N*K bytes vs 5*N*K bytes.
//
// FP8 E4M3 → f32 conversion uses direct bit manipulation for speed.
// Block-wise scale: scale[ceil(N/128), ceil(K/128)], one f32 per 128×128 block.
//
// Launch config:
//   Grid:  (N, M, 1)     — one block per (output_row, batch_element)
//   Block: (256, 1, 1)   — threads cooperatively reduce over K
//   Shared: 8 * sizeof(float)  (warp reduction)

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

// Convert FP8 E4M3 byte to float via branchless bit manipulation.
// FP8 E4M3: 1 sign | 4 exponent (bias=7) | 3 mantissa
// FP32:     1 sign | 8 exponent (bias=127) | 23 mantissa
//
// Branchless: eliminates divergent branches for subnormal/NaN cases.
// Subnormals (e=0, m!=0) flushed to zero — acceptable for inference
// since FP8 E4M3 subnormals represent values < 2^-9 ≈ 0.002.
__device__ __forceinline__ float fp8e4m3_to_float(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;

    // Normal: rebase exponent from bias=7 to bias=127 (add 120)
    // Mantissa: 3 bits → top 3 of 23 bits (shift left 20)
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);

    // Branchless zero for: e==0 (zero/subnormal) or NaN (e=15,m=7)
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    // -1u = 0xFFFFFFFF when is_normal==1, 0 when is_normal==0
    f32_bits &= (unsigned int)(-(int)is_normal);

    return __uint_as_float(f32_bits);
}

// ---------------------------------------------------------------------------
// Fused FP8 GEMV with block-wise f32 scale.
//
// output[m, n] = Σ_k fp8_to_f32(weight[n, k]) * scale[n/128, k/128] * input[m, k]
//
// Each block computes one output element output[m, n].
// 256 threads cooperatively reduce over the K dimension.
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks               // ceil(K / 128)
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;
    int scale_row = n >> 7;           // n / 128
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    // Dual accumulators break the dependency chain for better ILP.
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Main loop: process 8 FP8 bytes at a time via two coalesced uint32 reads.
    // __ldg uses the read-only cache for better bandwidth.
    for (int k = threadIdx.x * 8; k + 7 < K; k += BLOCK_DIM * 8) {
        // Two coalesced 4-byte reads = 8 FP8 values
        unsigned int lo4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k));
        unsigned int hi4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 4));

        // Scale: one lookup per 4 elements (128-wide blocks, 4-byte reads)
        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        acc0 += fp8e4m3_to_float(lo4 & 0xFFu)         * s0 * __ldg(x_row + k);
        acc0 += fp8e4m3_to_float((lo4 >> 8) & 0xFFu)  * s0 * __ldg(x_row + k + 1);
        acc0 += fp8e4m3_to_float((lo4 >> 16) & 0xFFu) * s0 * __ldg(x_row + k + 2);
        acc0 += fp8e4m3_to_float((lo4 >> 24) & 0xFFu) * s0 * __ldg(x_row + k + 3);
        acc1 += fp8e4m3_to_float(hi4 & 0xFFu)         * s4 * __ldg(x_row + k + 4);
        acc1 += fp8e4m3_to_float((hi4 >> 8) & 0xFFu)  * s4 * __ldg(x_row + k + 5);
        acc1 += fp8e4m3_to_float((hi4 >> 16) & 0xFFu) * s4 * __ldg(x_row + k + 6);
        acc1 += fp8e4m3_to_float((hi4 >> 24) & 0xFFu) * s4 * __ldg(x_row + k + 7);
    }

    float acc = acc0 + acc1;

    // Remainder: handle K not divisible by 8
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + threadIdx.x; kr < K; kr += BLOCK_DIM) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s * __ldg(x_row + kr);
        }
    }

    // --- Warp-level reduction ---
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // --- Inter-warp reduction via shared memory ---
    __shared__ float warp_sums[BLOCK_DIM / 32];   // 8 warps for 256 threads
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
// Vectorized FP8 GEMV: float4 input reads reduce instruction count.
//
// Same algorithm as fp8_gemv_blockwise_kernel but replaces 8 individual
// __ldg(x_row + k + i) with 2 float4 vector reads. This reduces input
// load instructions from 8 to 2 per iteration, freeing issue slots for
// the FP8→f32 ALU conversions.
//
// Launch config: same as blockwise kernel.
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_vec_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks               // ceil(K / 128)
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;
    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc = 0.0f;

    // Main loop: 8 FP8 values per iteration, vectorized input reads.
    // k = threadIdx.x * 8 ensures k*sizeof(float) is 32-byte aligned → float4 safe.
    for (int k = threadIdx.x * 8; k + 7 < K; k += BLOCK_DIM * 8) {
        // Two coalesced 4-byte weight reads = 8 FP8 values
        unsigned int lo4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k));
        unsigned int hi4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 4));

        // Two float4 input reads instead of 8 scalar reads
        float4 x_lo = __ldg(reinterpret_cast<const float4*>(x_row + k));
        float4 x_hi = __ldg(reinterpret_cast<const float4*>(x_row + k + 4));

        // Scale lookups
        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        acc += fp8e4m3_to_float(lo4 & 0xFFu)         * s0 * x_lo.x;
        acc += fp8e4m3_to_float((lo4 >> 8) & 0xFFu)  * s0 * x_lo.y;
        acc += fp8e4m3_to_float((lo4 >> 16) & 0xFFu) * s0 * x_lo.z;
        acc += fp8e4m3_to_float((lo4 >> 24) & 0xFFu) * s0 * x_lo.w;
        acc += fp8e4m3_to_float(hi4 & 0xFFu)         * s4 * x_hi.x;
        acc += fp8e4m3_to_float((hi4 >> 8) & 0xFFu)  * s4 * x_hi.y;
        acc += fp8e4m3_to_float((hi4 >> 16) & 0xFFu) * s4 * x_hi.z;
        acc += fp8e4m3_to_float((hi4 >> 24) & 0xFFu) * s4 * x_hi.w;
    }
    // Remainder: handle K not divisible by 8
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + threadIdx.x; kr < K; kr += BLOCK_DIM) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s * __ldg(x_row + kr);
        }
    }

    // --- Warp-level reduction ---
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // --- Inter-warp reduction via shared memory ---
    __shared__ float warp_sums[BLOCK_DIM / 32];
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
// Optimized FP8 GEMV with shared-memory LUT for FP8→f32 conversion.
//
// Key optimizations over the basic blockwise kernel:
//   1. Shared-memory LUT: 256-entry lookup table avoids 7-8 ALU ops per byte
//   2. Vectorized 16-byte loads: reads 16 FP8 values per iteration via uint4
//   3. Cached scale: reuses scale value across 128 consecutive K elements
//
// Launch: Grid (N, M), Block (256), Shared: 256*4 + 8*4 = 1056 bytes
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_lut_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks               // ceil(K / 128)
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;

    // Build FP8 E4M3 → f32 lookup table in shared memory (256 entries)
    __shared__ float fp8_lut[256];
    if (threadIdx.x < 256) {
        fp8_lut[threadIdx.x] = fp8e4m3_to_float((unsigned char)threadIdx.x);
    }
    __syncthreads();

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc = 0.0f;

    // Main loop: 16 FP8 values per iteration via uint4 load
    int k_base = threadIdx.x * 16;
    int k_stride = BLOCK_DIM * 16;  // 4096

    for (int k = k_base; k + 15 < K; k += k_stride) {
        // One coalesced 16-byte read = 16 FP8 values
        uint4 w16 = __ldg(reinterpret_cast<const uint4*>(w_row + k));
        unsigned char* wb = reinterpret_cast<unsigned char*>(&w16);

        // Scale: 128-wide blocks. Check if we cross a scale boundary.
        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);

        // Unroll 16 elements with LUT lookup
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int kk = k + j;
            // Check for scale boundary crossing (every 128 elements)
            float s = ((kk >> 7) != sc0) ? __ldg(&scale[scale_row * num_col_blocks + (kk >> 7)]) : s0;
            acc += fp8_lut[wb[j]] * s * __ldg(x_row + kk);
        }
    }

    // Remainder
    {
        int aligned_k = (K / 16) * 16;
        for (int kr = aligned_k + threadIdx.x; kr < K; kr += BLOCK_DIM) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8_lut[__ldg(w_row + kr)] * s * __ldg(x_row + kr);
        }
    }

    // --- Warp-level reduction ---
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // --- Inter-warp reduction via shared memory ---
    __shared__ float warp_sums[BLOCK_DIM / 32];
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
// Optimized FP8 GEMV v2: shared-memory LUT, scalar loads only.
//
// Key optimization: replaces fp8e4m3_to_float ALU bit manipulation (~10 insns
// per byte, with branches) with a 256-entry shared-memory LUT (~5 cycle read).
// All loads are scalar (32-bit) to avoid sm_121 hang with 128-bit v4 loads.
//
// Processes 8 FP8 per iteration (same as blockwise_kernel) with dual
// accumulators, but with ~80 fewer ALU instructions per iteration.
//
// Launch config:
//   Grid:  (N, M, 1)
//   Block: (256, 1, 1)
//   Shared: static 256*4 + 8*4 = 1056 bytes (LUT + warp_sums)
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_v2_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks               // ceil(K / 128)
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;

    // Build FP8 E4M3 -> f32 lookup table in shared memory
    __shared__ float fp8_lut[256];
    fp8_lut[threadIdx.x] = fp8e4m3_to_float((unsigned char)threadIdx.x);
    __syncthreads();

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Main loop: 8 FP8 per iteration, scalar 32-bit loads only.
    for (int k = threadIdx.x * 8; k + 7 < K; k += BLOCK_DIM * 8) {
        // Two coalesced 4-byte reads = 8 FP8 values (scalar, no v4)
        unsigned int lo4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k));
        unsigned int hi4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 4));

        // Scale: one lookup per 4 bytes (safe since 8 divides 128)
        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        // LUT-based FP8->f32 replaces ~10 ALU instructions per conversion
        acc0 += fp8_lut[lo4 & 0xFFu]         * s0 * __ldg(x_row + k);
        acc0 += fp8_lut[(lo4 >> 8) & 0xFFu]  * s0 * __ldg(x_row + k + 1);
        acc0 += fp8_lut[(lo4 >> 16) & 0xFFu] * s0 * __ldg(x_row + k + 2);
        acc0 += fp8_lut[(lo4 >> 24) & 0xFFu] * s0 * __ldg(x_row + k + 3);
        acc1 += fp8_lut[hi4 & 0xFFu]         * s4 * __ldg(x_row + k + 4);
        acc1 += fp8_lut[(hi4 >> 8) & 0xFFu]  * s4 * __ldg(x_row + k + 5);
        acc1 += fp8_lut[(hi4 >> 16) & 0xFFu] * s4 * __ldg(x_row + k + 6);
        acc1 += fp8_lut[(hi4 >> 24) & 0xFFu] * s4 * __ldg(x_row + k + 7);
    }

    float acc = acc0 + acc1;

    // Remainder: handle K not divisible by 8
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + threadIdx.x; kr < K; kr += BLOCK_DIM) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8_lut[__ldg(w_row + kr)] * s * __ldg(x_row + kr);
        }
    }

    // --- Warp-level reduction ---
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // --- Inter-warp reduction via shared memory ---
    __shared__ float warp_sums[BLOCK_DIM / 32];
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
// Fused FP8 GEMV with per-tensor f32 scale (single scale value).
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_scaled_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ input,
    int M, int N, int K
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;
    float s = scale_ptr[0];
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += BLOCK_DIM) {
        acc += fp8e4m3_to_float(w_row[k]) * s * x_row[k];
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM / 32];
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
// Fused FP8 GEMV without scale (raw FP8 → f32).
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ input,
    int M, int N, int K
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += BLOCK_DIM) {
        acc += fp8e4m3_to_float(w_row[k]) * x_row[k];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM / 32];
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
// Warp-per-row FP8 GEMV: each warp independently computes one output element.
//
// Key insight: The standard kernel assigns 256 threads to ONE output row,
// giving each thread only ~2.5 iterations for K=5120. With LPDDR5X's 150ns
// latency, this means the warp finishes before enough loads are in-flight.
//
// This kernel assigns 1 warp (32 threads) per output row. Each thread does
// K/(32*8) = 20 iterations for K=5120, keeping the memory pipeline much
// busier. 8 warps per block = 8 rows per block. No __syncthreads() needed.
//
// All 8 warps read the same input vector → guaranteed L1 cache reuse.
// Each warp reads a different weight row → independent memory streams.
//
// Launch config:
//   Grid:  (ceil(N/8), M, 1)
//   Block: (256, 1, 1)
//   Shared: 0
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_wpr_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int warp = threadIdx.x >> 5;       // 0..7
    int lane = threadIdx.x & 31;       // 0..31
    int n = blockIdx.x * 8 + warp;     // output row
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    // 32 threads per warp, 8 FP8 per iteration via u64 loads.
    // Uses 64-bit scalar loads (NOT v4/128-bit which hangs on sm_121).
    // Halves load instruction count vs 2x u32, which at throttled 507 MHz
    // means more outstanding bytes per cycle → better LPDDR5X saturation.
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = lane * 8; k + 7 < K; k += 256) {
        // One 64-bit weight load = 8 FP8 bytes (vs two 32-bit loads)
        unsigned long long w8 = __ldg(reinterpret_cast<const unsigned long long*>(w_row + k));
        unsigned int lo4 = (unsigned int)(w8);
        unsigned int hi4 = (unsigned int)(w8 >> 32);

        // Four 64-bit input loads = 8 f32 values (vs eight 32-bit loads)
        unsigned long long x01 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
        unsigned long long x23 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 2));
        unsigned long long x45 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));
        unsigned long long x67 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 6));

        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        acc0 += fp8e4m3_to_float(lo4 & 0xFFu)         * s0 * __uint_as_float((unsigned int)(x01));
        acc0 += fp8e4m3_to_float((lo4 >> 8) & 0xFFu)  * s0 * __uint_as_float((unsigned int)(x01 >> 32));
        acc0 += fp8e4m3_to_float((lo4 >> 16) & 0xFFu) * s0 * __uint_as_float((unsigned int)(x23));
        acc0 += fp8e4m3_to_float((lo4 >> 24) & 0xFFu) * s0 * __uint_as_float((unsigned int)(x23 >> 32));
        acc1 += fp8e4m3_to_float(hi4 & 0xFFu)         * s4 * __uint_as_float((unsigned int)(x45));
        acc1 += fp8e4m3_to_float((hi4 >> 8) & 0xFFu)  * s4 * __uint_as_float((unsigned int)(x45 >> 32));
        acc1 += fp8e4m3_to_float((hi4 >> 16) & 0xFFu) * s4 * __uint_as_float((unsigned int)(x67));
        acc1 += fp8e4m3_to_float((hi4 >> 24) & 0xFFu) * s4 * __uint_as_float((unsigned int)(x67 >> 32));
    }

    float acc = acc0 + acc1;

    // Remainder
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s * __ldg(x_row + kr);
        }
    }

    // Warp reduction only — no shared memory, no __syncthreads()!
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = acc;
    }
}

// ---------------------------------------------------------------------------
// WPR + LUT: maximum instruction efficiency for sustained clocks.
//
// At throttled 507 MHz, instruction issue rate limits memory bandwidth utilization.
// This kernel minimizes instructions per FP8 byte:
//   - LUT: 1 shared-mem read vs 6 ALU ops for fp8→f32 conversion
//   - WPR: warp-per-row avoids shared-mem reduction and __syncthreads overhead
//   - Scalar loads only (no float4/v4) to avoid sm_121 hang with shared memory
//
// The LUT costs one __syncthreads() at block start (all 256 threads populate it
// in parallel), but saves ~5 instructions × 8 values × 20 iterations = 800
// instructions per thread over the loop lifetime.
//
// Launch config:
//   Grid:  (ceil(N/8), M, 1)
//   Block: (256, 1, 1)
//   Shared: static 256 * 4 = 1024 bytes (LUT only, no warp_sums needed)
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_wpr_lut_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    // Build FP8 E4M3 → f32 LUT in shared memory (256 entries, 1KB)
    __shared__ float fp8_lut[256];
    fp8_lut[threadIdx.x] = fp8e4m3_to_float((unsigned char)threadIdx.x);
    __syncthreads();

    int warp = threadIdx.x >> 5;       // 0..7
    int lane = threadIdx.x & 31;       // 0..31
    int n = blockIdx.x * 8 + warp;     // output row
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // 32 threads per warp, 8 FP8 per iteration, stride = 256 per iteration
    // Uses scalar input reads (no float4/v4) to avoid sm_121 hang with shared memory.
    for (int k = lane * 8; k + 7 < K; k += 256) {
        // Two coalesced 4-byte weight reads
        unsigned int lo4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k));
        unsigned int hi4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 4));

        // Scale lookup (rare boundary crossing)
        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        // LUT conversion: 1 shared-mem read replaces 6 ALU ops per byte
        acc0 += fp8_lut[lo4 & 0xFFu]         * s0 * __ldg(x_row + k);
        acc0 += fp8_lut[(lo4 >> 8) & 0xFFu]  * s0 * __ldg(x_row + k + 1);
        acc0 += fp8_lut[(lo4 >> 16) & 0xFFu] * s0 * __ldg(x_row + k + 2);
        acc0 += fp8_lut[(lo4 >> 24) & 0xFFu] * s0 * __ldg(x_row + k + 3);
        acc1 += fp8_lut[hi4 & 0xFFu]         * s4 * __ldg(x_row + k + 4);
        acc1 += fp8_lut[(hi4 >> 8) & 0xFFu]  * s4 * __ldg(x_row + k + 5);
        acc1 += fp8_lut[(hi4 >> 16) & 0xFFu] * s4 * __ldg(x_row + k + 6);
        acc1 += fp8_lut[(hi4 >> 24) & 0xFFu] * s4 * __ldg(x_row + k + 7);
    }

    float acc = acc0 + acc1;

    // Remainder
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8_lut[__ldg(w_row + kr)] * s * __ldg(x_row + kr);
        }
    }

    // Warp reduction only — no shared memory reduction needed
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = acc;
    }
}

// sm_121 / Blackwell ISA only: the native cvt.rn.f16x2.e4m3x2 instruction
// was introduced with the Blackwell PTX ISA (compute capability ≥ 10.0).
// Earlier targets (sm_90 Hopper, sm_89 Ada, …) still compile the file but
// skip the native kernel via __CUDA_ARCH__; the branchless fp8e4m3_to_float
// path above covers them.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000

// ---------------------------------------------------------------------------
// Hardware-accelerated FP8 → f32 conversion using native CVT instruction.
//
// cvt.rn.f16x2.e4m3x2 converts 2 packed FP8 E4M3 bytes to 2 packed FP16
// in a SINGLE instruction on sm_121 (Blackwell). Then cvt.f32.f16 converts
// each half to float.
//
// Total: 3 instructions per FP8 element vs 24 instructions with branchless
// bit manipulation. This 8x reduction in conversion ALU directly reduces
// power consumption, helping the GB10 sustain higher clocks under its
// firmware power cap.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fp8x2_to_f32(unsigned short packed_fp8x2,
                                              float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed_fp8x2));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

// ---------------------------------------------------------------------------
// WPR + Native FP8 CVT: minimum-instruction GEMV for power-limited hardware.
//
// Combines the warp-per-row strategy with hardware FP8 conversion:
//   - Native cvt.rn.f16x2.e4m3x2: 3 instructions per FP8 element (vs 24)
//   - 64-bit loads: halves load instruction count
//   - WPR: no shared memory, no __syncthreads()
//
// Inner loop instruction budget per 8 FP8 elements:
//   OLD: 5 loads + 192 conversion ALU + 16 scale/FMA = ~213 instructions
//   NEW: 5 loads + 12 conversion + 16 scale/FMA = ~33 instructions (6.5x less)
//
// Fewer instructions → less power → firmware allows higher sustained clocks
// → more memory bandwidth → faster decode.
//
// Launch config:
//   Grid:  (ceil(N/8), M, 1)
//   Block: (256, 1, 1)
//   Shared: 0
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int warp = threadIdx.x >> 5;       // 0..7
    int lane = threadIdx.x & 31;       // 0..31
    int n = blockIdx.x * 8 + warp;     // output row
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = lane * 8; k + 7 < K; k += 256) {
        // One 64-bit weight load = 8 FP8 bytes
        unsigned long long w8 = __ldg(reinterpret_cast<const unsigned long long*>(w_row + k));

        // Four 64-bit input loads = 8 f32 values
        unsigned long long x01 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
        unsigned long long x23 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 2));
        unsigned long long x45 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));
        unsigned long long x67 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 6));

        // Scale
        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        // Native hardware FP8→f32 conversion: 4 pairs of 2 FP8 values
        float w0, w1, w2, w3, w4, w5, w6, w7;
        fp8x2_to_f32((unsigned short)(w8),       w0, w1);
        fp8x2_to_f32((unsigned short)(w8 >> 16), w2, w3);
        fp8x2_to_f32((unsigned short)(w8 >> 32), w4, w5);
        fp8x2_to_f32((unsigned short)(w8 >> 48), w6, w7);

        // Multiply-accumulate with scale and input
        acc0 += w0 * s0 * __uint_as_float((unsigned int)(x01));
        acc0 += w1 * s0 * __uint_as_float((unsigned int)(x01 >> 32));
        acc0 += w2 * s0 * __uint_as_float((unsigned int)(x23));
        acc0 += w3 * s0 * __uint_as_float((unsigned int)(x23 >> 32));
        acc1 += w4 * s4 * __uint_as_float((unsigned int)(x45));
        acc1 += w5 * s4 * __uint_as_float((unsigned int)(x45 >> 32));
        acc1 += w6 * s4 * __uint_as_float((unsigned int)(x67));
        acc1 += w7 * s4 * __uint_as_float((unsigned int)(x67 >> 32));
    }

    float acc = acc0 + acc1;

    // Remainder
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            // Remainder uses branchless conversion (few elements, not hot)
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s * __ldg(x_row + kr);
        }
    }

    // Warp reduction only
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = acc;
    }
}

// ---------------------------------------------------------------------------
// WPR + Native FP8 CVT, F16-input variant.
//
// Identical algorithm to `fp8_gemv_blockwise_wpr_native_kernel` but takes
// f16 activations instead of f32. Purpose: skip the activation FP8-quant
// step in Gemma 4 decode — at M=1 the pre-quant is pure overhead (one
// activation row feeds 4 GEMMs) and going f16→f32→fma is cheaper than
// f16→fp8→dequant→fma.
//
// Native `cvt.f32.f16` is a single-cycle insn on every arch from sm_70,
// so the input-type switch is free; only the load pattern changes (8
// f16 = 16 B = 2 × u64 instead of 4 × u64 for 8 f32).
//
// Output is f16 to match the f16 accumulator of the calling layer (avoids
// an extra f32→f16 cast kernel at the Rust call site).
//
// Launch config: identical to wpr_native — grid (ceil(N/8), M), block 256.
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_f16in_kernel(
    __half* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const __half* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const __half* x_row = input + (long long)m * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = lane * 8; k + 7 < K; k += 256) {
        unsigned long long w8 = __ldg(reinterpret_cast<const unsigned long long*>(w_row + k));

        // 8 f16 values = 16 bytes = 2 × u64
        unsigned long long x_lo = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
        unsigned long long x_hi = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));

        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        float w0, w1, w2, w3, w4, w5, w6, w7;
        fp8x2_to_f32((unsigned short)(w8),       w0, w1);
        fp8x2_to_f32((unsigned short)(w8 >> 16), w2, w3);
        fp8x2_to_f32((unsigned short)(w8 >> 32), w4, w5);
        fp8x2_to_f32((unsigned short)(w8 >> 48), w6, w7);

        float x0, x1, x2, x3, x4, x5, x6, x7;
        asm("cvt.f32.f16 %0, %1;" : "=f"(x0) : "h"((unsigned short)(x_lo)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x1) : "h"((unsigned short)(x_lo >> 16)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x2) : "h"((unsigned short)(x_lo >> 32)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x3) : "h"((unsigned short)(x_lo >> 48)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x4) : "h"((unsigned short)(x_hi)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x5) : "h"((unsigned short)(x_hi >> 16)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x6) : "h"((unsigned short)(x_hi >> 32)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x7) : "h"((unsigned short)(x_hi >> 48)));

        acc0 += w0 * s0 * x0;
        acc0 += w1 * s0 * x1;
        acc0 += w2 * s0 * x2;
        acc0 += w3 * s0 * x3;
        acc1 += w4 * s4 * x4;
        acc1 += w5 * s4 * x5;
        acc1 += w6 * s4 * x6;
        acc1 += w7 * s4 * x7;
    }

    float acc = acc0 + acc1;

    // Remainder
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s
                   * __half2float(__ldg(x_row + kr));
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Fused FP8 GEMV with block-wise f32 scale, BF16-input fast path.
//
// Cycle 55 step 3 (Phase B): bf16-input sibling of
// `fp8_gemv_blockwise_wpr_native_f16in_kernel`. Identical math (f32
// internal accumulator, FP8 weight dequant, blockwise scale) — only
// the input + output dtypes flip f16 → bf16. Eliminates the cycle-54
// runtime bf16→f16-sat narrow at QKV/gate_up entry on the M=1 decode
// hot path.
//
// Launch config: identical to wpr_native_f16in — grid (ceil(N/8), M),
// block 256.
// ---------------------------------------------------------------------------
extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_bf16in_kernel(
    __nv_bfloat16* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const __nv_bfloat16* x_row = input + (long long)m * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = lane * 8; k + 7 < K; k += 256) {
        unsigned long long w8 = __ldg(reinterpret_cast<const unsigned long long*>(w_row + k));

        // 8 bf16 values = 16 bytes = 2 × u64
        unsigned long long x_lo = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
        unsigned long long x_hi = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));

        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        float w0, w1, w2, w3, w4, w5, w6, w7;
        fp8x2_to_f32((unsigned short)(w8),       w0, w1);
        fp8x2_to_f32((unsigned short)(w8 >> 16), w2, w3);
        fp8x2_to_f32((unsigned short)(w8 >> 32), w4, w5);
        fp8x2_to_f32((unsigned short)(w8 >> 48), w6, w7);

        // bf16 → f32 is a simple left-shift-16 since bf16 is the
        // upper half of f32. Compiler emits a single mov.b32 in PTX.
        float x0, x1, x2, x3, x4, x5, x6, x7;
        // Cycle 55 step 16: bf16 → f32 via explicit shift, no pointer
        // reinterpret_cast (avoids any alignment/aliasing UB the
        // earlier `*reinterpret_cast<const __nv_bfloat16*>(&x_lo)`
        // pattern might have hit). bf16 IS the upper 16 bits of f32,
        // so the conversion is a single `mov.b32` in PTX.
        // x_lo / x_hi each hold 4 bf16 values packed little-endian
        // (bytes 0-1 = bf16[0], 2-3 = bf16[1], etc).
        auto bf16_bits_to_f32 = [] (unsigned int bits16) -> float {
            return __uint_as_float(bits16 << 16);
        };
        x0 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_lo)));
        x1 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_lo >> 16)));
        x2 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_lo >> 32)));
        x3 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_lo >> 48)));
        x4 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_hi)));
        x5 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_hi >> 16)));
        x6 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_hi >> 32)));
        x7 = bf16_bits_to_f32((unsigned int)((unsigned short)(x_hi >> 48)));

        acc0 += w0 * s0 * x0;
        acc0 += w1 * s0 * x1;
        acc0 += w2 * s0 * x2;
        acc0 += w3 * s0 * x3;
        acc1 += w4 * s4 * x4;
        acc1 += w5 * s4 * x5;
        acc1 += w6 * s4 * x6;
        acc1 += w7 * s4 * x7;
    }

    float acc = acc0 + acc1;

    // Remainder
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s
                   * __bfloat162float(__ldg(x_row + kr));
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        output[(long long)m * N + n] = __float2bfloat16(acc);
    }
}

#endif  // __CUDA_ARCH__ >= 1000
