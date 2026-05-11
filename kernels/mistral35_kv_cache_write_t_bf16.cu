// Mistral 3.5 batched KV-cache write for chunk-path prefill —
// stores `[T, n_kv_heads, head_dim]` BF16 K and V tensors into
// the per-layer cache slots `[pos_start .. pos_start + T)` in
// one launch. Drop-in batched replacement for
// `mistral35_kv_cache_write_bf16_kernel` which the chunked
// prefill path called T times per layer.
//
// Cache layout (BF16):  [max_pos, n_kv_heads, head_dim]
// Source layout (BF16): [T,      n_kv_heads, head_dim]
//
// At T=383 / 88 layers the per-position dispatch fires
// `T * 88 = 33K` kernel launches per request for KV-write
// alone. The batched form collapses this to 88 launches.
//
// Launch:
//   Grid:  (n_kv_heads, T, 1)
//   Block: (head_dim, 1, 1)   — head_dim <= 1024

#include <cuda_bf16.h>

extern "C" __global__ void mistral35_kv_cache_write_t_bf16_kernel(
    const __nv_bfloat16* __restrict__ k_in_t,   // [T, n_kv, head_dim]
    const __nv_bfloat16* __restrict__ v_in_t,
    __nv_bfloat16*       __restrict__ k_cache,   // [max_pos, n_kv, head_dim]
    __nv_bfloat16*       __restrict__ v_cache,
    int n_kv_heads,
    int head_dim,
    int pos_start,
    int t_count
) {
    const int kvh = blockIdx.x;
    const int t   = blockIdx.y;
    const int d   = threadIdx.x;
    if (kvh >= n_kv_heads || t >= t_count || d >= head_dim) return;

    const long long row_stride = (long long)n_kv_heads * head_dim;
    const long long src_off    =
        (long long)t * row_stride + (long long)kvh * head_dim + d;
    const long long dst_off    =
        (long long)(pos_start + t) * row_stride
        + (long long)kvh * head_dim + d;
    k_cache[dst_off] = k_in_t[src_off];
    v_cache[dst_off] = v_in_t[src_off];
}
