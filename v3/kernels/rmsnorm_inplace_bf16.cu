// In-place bf16 RMSNorm: reads bf16 input, normalizes with f16 gamma, writes bf16 back.
// For the delta path where GEMM outputs bf16 to avoid f16 overflow.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(1024)
rmsnorm_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,    // [num_tokens, hidden_size] bf16 in-place
    const __half* __restrict__ gamma,  // [hidden_size] f16
    float eps,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;

    __shared__ float smem[WARPS_MAX];

    // Cycle 55 step 17: optional f16-narrowing inside bf16 rmsnorm.
    // bf16 has 7-bit mantissa vs f16's 10-bit. The cycle-54 stage-2.1
    // narrow at QKV input was BOUNDING precision drift by f16-narrowing
    // residual before SS; FULL_CHAIN removes that bound. To test
    // whether bf16's lower mantissa precision is the FULL_CHAIN
    // breakage mechanism, the `RVLLM_BF16_RMSNORM_F16_PRECISION=1`
    // env causes the SS accumulator to read f16-narrowed values
    // (matches f16 rmsnorm semantics exactly while output stays bf16).
    // Hypothesis: if this fixes FULL_CHAIN, precision compounding is
    // confirmed; the right architectural shape is hybrid bf16-storage
    // + f16-mantissa SS.
    // Note: the narrow is on the COMPUTE path only — output dtype
    // stays bf16 so storage benefits remain.
    // Cycle 55 step 17 EXPERIMENT: f16-narrow inside SS to test
    // precision-compounding hypothesis (bf16's 7-bit mantissa vs
    // f16's 10-bit at sum-of-squares stage). If this single change
    // restores FULL_CHAIN at short context, the compounding mechanism
    // is confirmed and the architectural answer is hybrid bf16-storage
    // + f16-precision compute at SS bottlenecks. Output stays bf16.
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(x[row_offset + i]);
        v = __half2float(__float2half(v));  // f16-narrow for SS
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    if (threadIdx.x == 0) smem[0] = sum_sq;
    __syncthreads();
    sum_sq = smem[0];

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(x[row_offset + i]) * rms * __half2float(gamma[i]);
        x[row_offset + i] = __float2bfloat16(v);
    }
}
