// Per-v-head RMSNormGated + silu(z) gate for Qwen 3.6 Gated-DeltaNet
// linear-attention. Replaces the post-delta-rule host pipeline:
//
//   DtoH readout (vus*hvd f16)        →
//   f16→f32 host conversion           →
//   DtoH z (vus*hvd f16)              →
//   DtoH norm gamma (hvd f16)         →
//   for v in 0..vus:                  ← CPU loop
//       sumsq over hvd, rms
//       for d in 0..hvd:
//         out = (readout/rms * gamma) * silu(z)
//   HtoD gated (vus*hvd f16)
//
// Inputs (all already on device):
//   readout : [vus, hvd] f16
//   z       : [vus, hvd] f16  (= in_proj_z output, value_dim = vus*hvd)
//   gamma   : [hvd]      f16  (la.norm.weight, layer constant)
// Output:
//   gated   : [vus, hvd] f16
//
// The math, vLLM-equivalent for `RMSNormGated` with
// `norm_before_gate=True` + activation=swish:
//   rms[v]    = sqrt(sum_d(readout[v,d]^2) / hvd + eps)
//   out[v,d]  = (readout[v,d]/rms[v] * gamma[d]) * silu(z[v,d])
//   silu(z)   = z / (1 + exp(-z))
//
// Launch:
//   Grid:  (vus, 1, 1)             — one block per v-head
//   Block: (hvd, 1, 1)              — one thread per element
//   Shared: WARPS_MAX * sizeof(float)

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_rm(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_rm(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_rm(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_rm(val);
    return val;
}

extern "C" __global__ void __launch_bounds__(256)
qwen_linear_rmsnorm_gated_f16_kernel(
    __half*       __restrict__ gated_out,
    const __half* __restrict__ readout,
    const __half* __restrict__ z_logits,
    const __half* __restrict__ gamma,
    int vus,
    int hvd,
    float eps
) {
    const int v   = blockIdx.x;
    const int tid = threadIdx.x;
    if (v >= vus || tid >= hvd) return;

    __shared__ float smem[WARPS_MAX];
    __shared__ float rms_b;

    const long long base = (long long)v * hvd;

    // Pass 1: sumsq.
    float r = __half2float(readout[base + tid]);
    float r_sq = r * r;
    float sumsq = block_reduce_sum_rm(r_sq, smem);
    if (tid == 0) {
        rms_b = sqrtf(sumsq / (float)hvd + eps);
    }
    __syncthreads();
    float rms = rms_b;

    // Pass 2: per-element RMSNormGated.
    float g = __half2float(gamma[tid]);
    float zv = __half2float(z_logits[base + tid]);
    float sig = 1.0f / (1.0f + expf(-zv));
    float silu_z = zv * sig;
    float out = (r / rms) * g * silu_z;
    gated_out[base + tid] = __float2half(out);
}
