// Stage 1 of BF16 residual chain (cycle 53+).
//
// Sibling of `fused_norm_add_residual{,_f16}.cu` — same fused
// (norm + add-to-residual + optional layer_scalar) op but the residual
// buffer is bf16 instead of f16. All math stays f32 internally; only
// the storage at the residual ↔ input boundary changes.
//
// vLLM's Gemma 4 production runs bf16 activations + fp8 KV; rvllm's
// f16 residual loses ~3 mantissa bits of dynamic range per layer
// against the bf16 reference, accumulating direction drift across 60
// layers on long-context prompts. This kernel set lets us swap the
// residual-chain boundary to bf16 behind `RVLLM_RESIDUAL_BF16=1`.
//
// Two kernels mirror the original f16 set:
//   * fused_norm_add_residual_bf16_kernel      — f32 input (post-GEMM,
//     no channelscale path), bf16 residual.
//   * fused_norm_add_residual_bf16_f16in_kernel — f16 input (post-
//     channelscale-baked-into-GEMM path), bf16 residual.
//
// Grid: (num_tokens), Block: (min(hidden, 1024))
// Shared memory: hidden * sizeof(float)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void fused_norm_add_residual_bf16_kernel(
    const float*         __restrict__ gemm_out,
    const __half*        __restrict__ gamma,
    __nv_bfloat16*       __restrict__ residual,
    const __half*        __restrict__ layer_scalar,
    int hidden,
    float eps
) {
    extern __shared__ float svals[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float*   row = gemm_out + (size_t)token * hidden;
    __nv_bfloat16* res = residual + (size_t)token * hidden;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = row[i];
        svals[i] = v;
        local_ss += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float warp_ss[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_ss[warp_id] = local_ss;
    __syncthreads();

    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += warp_ss[w];
        warp_ss[0] = total;
    }
    __syncthreads();
    float rms_inv = rsqrtf(warp_ss[0] / (float)hidden + eps);

    float ls = layer_scalar ? __half2float(*layer_scalar) : 1.0f;
    for (int i = tid; i < hidden; i += stride) {
        float normed = svals[i] * rms_inv * __half2float(gamma[i]);
        float r = __bfloat162float(res[i]) + normed;
        res[i] = __float2bfloat16(r * ls);
    }
}

// Cycle 55 step 19: bf16-input variant. Used under FULL_CHAIN dispatch
// when the upstream Fp8GemvBf16In produces bf16 GEMV output directly,
// so the residual epilogue stays end-to-end bf16 with no f16 narrow.
// Same body as the _f16in variant; only the gemm_out dtype + reader
// flip __half/__half2float → __nv_bfloat16/__bfloat162float.
extern "C" __global__ void fused_norm_add_residual_bf16_bf16in_kernel(
    const __nv_bfloat16* __restrict__ gemm_out,
    const __half*        __restrict__ gamma,
    __nv_bfloat16*       __restrict__ residual,
    const __half*        __restrict__ layer_scalar,
    int hidden,
    float eps
) {
    extern __shared__ float svals[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const __nv_bfloat16* row = gemm_out + (size_t)token * hidden;
    __nv_bfloat16*       res = residual + (size_t)token * hidden;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = __bfloat162float(row[i]);
        svals[i] = v;
        local_ss += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float warp_ss[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_ss[warp_id] = local_ss;
    __syncthreads();

    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += warp_ss[w];
        warp_ss[0] = total;
    }
    __syncthreads();
    float rms_inv = rsqrtf(warp_ss[0] / (float)hidden + eps);

    float ls = layer_scalar ? __half2float(*layer_scalar) : 1.0f;
    for (int i = tid; i < hidden; i += stride) {
        float normed = svals[i] * rms_inv * __half2float(gamma[i]);
        float r = __bfloat162float(res[i]) + normed;
        res[i] = __float2bfloat16(r * ls);
    }
}

// f16-input variant (post-channelscale-baked-into-GEMM path). Identical
// body to the kernel above except the source row is f16, not f32.
extern "C" __global__ void fused_norm_add_residual_bf16_f16in_kernel(
    const __half*        __restrict__ gemm_out,
    const __half*        __restrict__ gamma,
    __nv_bfloat16*       __restrict__ residual,
    const __half*        __restrict__ layer_scalar,
    int hidden,
    float eps
) {
    extern __shared__ float svals[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const __half*  row = gemm_out + (size_t)token * hidden;
    __nv_bfloat16* res = residual + (size_t)token * hidden;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = __half2float(row[i]);
        svals[i] = v;
        local_ss += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float warp_ss[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_ss[warp_id] = local_ss;
    __syncthreads();

    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += warp_ss[w];
        warp_ss[0] = total;
    }
    __syncthreads();
    float rms_inv = rsqrtf(warp_ss[0] / (float)hidden + eps);

    float ls = layer_scalar ? __half2float(*layer_scalar) : 1.0f;
    for (int i = tid; i < hidden; i += stride) {
        float normed = svals[i] * rms_inv * __half2float(gamma[i]);
        float r = __bfloat162float(res[i]) + normed;
        res[i] = __float2bfloat16(r * ls);
    }
}
