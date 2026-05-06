// Fused silu + (Q,K) L2-norm + GQA-expand + (V) silu-pack for Qwen
// 3.6 Gated-DeltaNet linear-attention.
//
// Inputs:
//   conv_out : [conv_dim = 2*key_dim + value_dim] f16 on device
//                Q lives in [0..key_dim),
//                K in [key_dim..2*key_dim),
//                V in [2*key_dim..conv_dim).
// Outputs:
//   q_exp  : [vus, hkd] f16  — silu(Q) per-k-head L2-normalised,
//                              repeated v_per_k times along the
//                              v-head axis (GQA expansion).
//   k_exp  : [vus, hkd] f16  — same as q_exp for K.
//   v_pack : [vus, hvd] f16  — silu(V) packed per-v-head.
//
// Replaces the entire host pipeline:
//   DtoH conv_out (~16 KB) →
//   CPU silu loop over qkv_n elements →
//   per-k-head L2-norm over hkd elements (16 heads × 128 = 2048 FLOPs) →
//   GQA expand by per-v-head selection of the parent k-head's Q/K →
//   per-v-head V slice copy →
//   HtoD q_exp + k_exp + v_pack
// with one launch (Phase 4b-prep iter6).
//
// Launch:
//   Grid:  (vus, 1, 1)              — one block per v-head.
//   Block: (max(hkd, hvd), 1, 1)    — Qwen 3.6 has hkd == hvd == 128.
//   Shared: WARPS_MAX * sizeof(float) (for warp-amax reductions)

#include <cuda_fp16.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum_silu(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_silu(float val, float* smem) {
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    val = warp_reduce_sum_silu(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    val = (lid < nw) ? smem[lid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_silu(val);
    return val;
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

extern "C" __global__ void __launch_bounds__(256)
qwen_linear_silu_l2_gqa_f16_kernel(
    __half* __restrict__       q_exp_out,    // [vus, hkd]
    __half* __restrict__       k_exp_out,    // [vus, hkd]
    __half* __restrict__       v_pack_out,   // [vus, hvd]
    const __half* __restrict__ conv_out,     // [conv_dim]
    int vus,
    int hkd,
    int hvd,
    int key_dim,        // = num_k_heads * hkd
    int num_v_heads,
    int v_per_k         // = num_v_heads / num_k_heads
) {
    const int v   = blockIdx.x;
    const int tid = threadIdx.x;
    if (v >= vus) return;

    // Map this v-head to its parent k-head.
    const int kh = v / v_per_k;
    const int q_base = kh * hkd;
    const int k_base = key_dim + kh * hkd;
    const int v_base = 2 * key_dim + v * hvd;

    __shared__ float smem_q[WARPS_MAX];
    __shared__ float smem_k[WARPS_MAX];
    __shared__ float qsq_b;
    __shared__ float ksq_b;

    // --- Q & K silu + L2-norm (over hkd elements of the k-head) ---
    float q_val = 0.0f;
    float k_val = 0.0f;
    if (tid < hkd) {
        q_val = silu_f(__half2float(conv_out[q_base + tid]));
        k_val = silu_f(__half2float(conv_out[k_base + tid]));
    }
    float q_sq_local = q_val * q_val;
    float k_sq_local = k_val * k_val;
    float q_sq = block_reduce_sum_silu(q_sq_local, smem_q);
    float k_sq = block_reduce_sum_silu(k_sq_local, smem_k);
    if (tid == 0) {
        qsq_b = q_sq;
        ksq_b = k_sq;
    }
    __syncthreads();
    float q_norm = sqrtf(qsq_b + 1e-6f);
    float k_norm = sqrtf(ksq_b + 1e-6f);

    if (tid < hkd) {
        // Write Q and K (L2-normalised) into the v-head's row of
        // q_exp / k_exp. GQA expansion happens by virtue of every
        // v-head in the same k-group reading the SAME q_base / k_base
        // and writing into ITS OWN v-head row.
        q_exp_out[(long long)v * hkd + tid] = __float2half(q_val / q_norm);
        k_exp_out[(long long)v * hkd + tid] = __float2half(k_val / k_norm);
    }

    // --- V silu (no norm, just per-v-head pack) ---
    if (tid < hvd) {
        float vv = silu_f(__half2float(conv_out[v_base + tid]));
        v_pack_out[(long long)v * hvd + tid] = __float2half(vv);
    }
}
