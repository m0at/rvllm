// Mistral 3.5 NVFP4 KV-cache write kernel. Converts the newly
// computed BF16 K and V row for one decode/prefill step into
// 4-bit packed nibbles plus an E4M3 per-16-element scale, then
// stores at slot `pos` of the cache.
//
// Storage layout (per side, K and V identical):
//   packed: u8     [max_pos, n_kv_heads, head_dim / 2]
//                  low nibble  = elem 2i
//                  high nibble = elem 2i+1
//   scale:  E4M3   [max_pos, n_kv_heads, head_dim / 16]
//                  per 16-element block
//
// Compression: BF16 = 2 bytes/elem → NVFP4 = 0.5 byte + 1/16 byte
// = 0.5625 byte/elem. 3.55× smaller, 3.55× less decode-attention
// bandwidth — the bigger the past_len gets, the bigger the win
// (KV reads dominate decode at past_len > ~2000 for Mistral 3.5).
//
// Per 16-element block: amax → scale = amax / 6 (= NVFP4_MAX);
// encoded nibble = fp4_encode(elem / scale). The scale is stored
// as E4M3 (8-bit float, 1.5.2 layout, max ≈ 448) which loses some
// resolution at very large activations but matches the rvllm
// NVFP4 weight contract.
//
// Launch:
//   Grid:  (n_kv_heads, 2, 1)        — y=0 → K, y=1 → V
//   Block: (head_dim / 16, 1, 1)     — one thread per 16-elem block
// For Mistral 3.5 (head_dim=128, n_kv_heads=8) that's 8 blocks of
// 8 threads = 64 threads per kernel launch. Latency-minimal.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

extern "C" __global__ void mistral35_kv_cache_write_nvfp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ k_in,      // [n_kv, head_dim]
    const __nv_bfloat16* __restrict__ v_in,      // [n_kv, head_dim]
    uint8_t* __restrict__ k_packed,              // [max_pos, n_kv, head_dim/2]
    uint8_t* __restrict__ k_scale,               // [max_pos, n_kv, head_dim/16]
    uint8_t* __restrict__ v_packed,              // [max_pos, n_kv, head_dim/2]
    uint8_t* __restrict__ v_scale,               // [max_pos, n_kv, head_dim/16]
    int n_kv_heads,
    int head_dim,
    int slot_pos
) {
    const int kv = blockIdx.x;
    const int kv_or_v = blockIdx.y;  // 0=K, 1=V
    const int block_idx = threadIdx.x;
    const int blocks_per_head = head_dim / 16;
    if (kv >= n_kv_heads || block_idx >= blocks_per_head) return;

    const __nv_bfloat16* in =
        (kv_or_v == 0 ? k_in : v_in) + (long long)kv * head_dim;
    uint8_t* out_packed =
        (kv_or_v == 0 ? k_packed : v_packed)
        + (long long)slot_pos * n_kv_heads * (head_dim / 2)
        + (long long)kv * (head_dim / 2);
    uint8_t* out_scale =
        (kv_or_v == 0 ? k_scale : v_scale)
        + (long long)slot_pos * n_kv_heads * (head_dim / 16)
        + (long long)kv * (head_dim / 16);

    // This thread handles one 16-element block.
    const int elem_start = block_idx * 16;

    // (1) Load 16 BF16 elements + amax.
    float vals[16];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float v = __bfloat162float(in[elem_start + i]);
        vals[i] = v;
        const float a = fabsf(v);
        if (a > amax) amax = a;
    }

    // (2) Scale = amax / 6 (NVFP4_MAX = 6). If amax==0, scale=0 →
    //     every encoded nibble is zero, decode reproduces zero
    //     exactly.
    const float NVFP4_MAX = 6.0f;
    float scale = (amax > 0.0f) ? (amax / NVFP4_MAX) : 0.0f;
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // (3) Store scale as E4M3.
    const __nv_fp8_e4m3 scale_e4m3 = __nv_fp8_e4m3(scale);
    out_scale[block_idx] = *reinterpret_cast<const uint8_t*>(&scale_e4m3);

    // (4) Encode each of 16 elements → nibble, pack 2 per byte.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t lo = rvllm_nvfp4::fp4_encode(vals[2 * i]     * inv_scale);
        const uint32_t hi = rvllm_nvfp4::fp4_encode(vals[2 * i + 1] * inv_scale);
        out_packed[block_idx * 8 + i] = (uint8_t)((hi << 4) | (lo & 0xFu));
    }
}
