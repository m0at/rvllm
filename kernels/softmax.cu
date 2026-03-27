// Numerically stable softmax kernel using online (single-pass) algorithm.
// Uses warp-level primitives for efficient reduction.
//
// Launch config:
//   Grid:  (num_rows, 1, 1)
//   Block: (min(vocab_size, 1024), 1, 1)
//   Shared memory: 2 * sizeof(float) (for global max and sum broadcast)
//
// Each block computes softmax for one row.

#include <float.h>

extern "C"
__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* x = input + row * vocab_size;
    float* y = output + row * vocab_size;

    // Pass 1: Find max using warp-level reduction
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_max;
    __shared__ float s_sum;

    // First thread of each warp participates
    if (tid % warpSize == 0) {
        atomicMax((int*)&s_max, __float_as_int(local_max));
    }
    if (tid == 0) s_max = -FLT_MAX;
    __syncthreads();

    // Re-do max properly: each warp-leader writes, then we reduce
    // Simpler approach: use shared memory array for warp leaders
    __shared__ float warp_max[32]; // up to 32 warps per block
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    // Thread 0 reduces across warps
    if (tid == 0) {
        float m = -FLT_MAX;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) {
            m = fmaxf(m, warp_max[w]);
        }
        s_max = m;
    }
    __syncthreads();

    float row_max = s_max;

    // Pass 2: Compute exp(x - max) and local sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float e = expf(x[i] - row_max);
        y[i] = e; // store exp temporarily
        local_sum += e;
    }

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    __shared__ float warp_sum[32];
    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float s = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) {
            s += warp_sum[w];
        }
        s_sum = s;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_sum;

    // Pass 3: Normalize
    for (int i = tid; i < vocab_size; i += stride) {
        y[i] *= inv_sum;
    }
}
