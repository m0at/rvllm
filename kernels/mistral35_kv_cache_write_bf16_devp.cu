// Device-pointer position variant of mistral35_kv_cache_write_bf16
// — codex review #2 graph-capture follow-up. Identical math; only
// `position` is now `const int*` so a captured graph can be
// replayed across decode steps with the host updating the
// underlying device int between replays.

#include <cuda_bf16.h>

extern "C" __global__ void mistral35_kv_cache_write_bf16_devp_kernel(
    const __nv_bfloat16* __restrict__ k_in,    // [n_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ k_cache,        // [max_pos, n_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ v_cache,
    int n_kv_heads,
    int head_dim,
    const int* __restrict__ position_ptr
) {
    const int kvh = blockIdx.x;
    const int d   = threadIdx.x;
    if (d >= head_dim) return;
    const int position = *position_ptr;
    const int src_off = kvh * head_dim + d;
    const int dst_off = position * n_kv_heads * head_dim
                      + kvh * head_dim + d;
    k_cache[dst_off] = k_in[src_off];
    v_cache[dst_off] = v_in[src_off];
}
