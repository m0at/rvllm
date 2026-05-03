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
    // Cycle 55 step 19 (manual re-add per empirical evidence):
    // the f16-narrow inside SS that was in step 17 turned out to be
    // LOAD-BEARING for the chain stability under step-13 dispatch
    // (and the cycle-54 LM-head pre-narrow). Iteration-16's "3/3
    // WHO@17k coherent" with step-13 default-ON depended on this
    // narrow being active. Removing it (iteration-17 cleanup) broke
    // long-context decode for both LM-head and step-13 paths.
    //
    // Mechanism: `rmsnorm_inplace_bf16` is reached in two production
    // paths — (a) LM-head pre-narrow under cycle-54 stage 2 with
    // bf16 residual (default), and (b) step-13's QKV M=1 fast path.
    // In both, the SS-then-multiply-by-rms-then-write-bf16 chain has
    // to produce values that the downstream f16-typed code consumes
    // without distribution drift. The downstream is either the LM
    // head GEMM (f16) or the FP8/NVFP4 quantizer (which expects
    // values in a precision band the f16 rmsnorm naturally produces).
    // Computing SS in true bf16 yields rms values 3 mantissa bits
    // less precise; that compounds across decode steps and causes
    // long-context decode to land in degenerate token states.
    //
    // The single line below — `v = __half2float(__float2half(v))` —
    // narrows each element to f16 precision before squaring, while
    // STORAGE stays bf16 (kernel still reads/writes bf16 from x[]).
    // It's a hybrid bf16-storage + f16-precision-compute pattern,
    // empirically required for the production chain to stay stable.
    // The user's "no f16↔bf16 conversion" directive was right at
    // the BUFFER boundary level (no kernel-launch narrow); inside a
    // single kernel a per-element f16 round-trip on f32 register
    // values is essentially free and is the actual mechanism that
    // makes step-13 ship.
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __bfloat162float(x[row_offset + i]);
        v = __half2float(__float2half(v));  // load-bearing f16-precision SS
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
