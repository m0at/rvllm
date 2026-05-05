// Qwen 3.6 partial-NeoX RoPE + F16 paged-KV-cache write.
//
// Differs from `fused_rope_partial_f16kv.cu` in pair convention:
// Qwen pairs `(i, i + rotary_dim/2)` within the first `rotary_dim`
// elements of each head (rotary_dim=64, head_dim=256, partial=0.25);
// indices [rotary_dim, head_dim) pass through untouched. The Gemma
// kernel pairs `(i, i + head_dim/2)` which is correct only for the
// full-rotary case (rotary_dim == head_dim).
//
// Replaces this host-side sequence in `apply_layer_full_attn`:
//   DtoH cos/sin → host f16→f32 → DtoH q/k/v →
//   CPU NeoX rotation loop on Q + K → HtoD rotated Q →
//   HtoD K, V to KV-cache slots
// with one GPU launch.
//
// Launch:
//   Grid:  (num_tokens, max(num_heads, num_kv_heads), 1)
//   Block: (head_dim/2, 1, 1)  (rotated half + passthrough half)
//
// Inputs:
//   q_in  : [num_tokens, num_heads,    head_dim] f16
//   k_in  : [num_tokens, num_kv_heads, head_dim] f16
//   v_in  : [num_tokens, num_kv_heads, head_dim] f16
//   cos_table, sin_table : [max_pos, rotary_dim/2] f16
//   positions    : [num_tokens] i32
//   slot_mapping : [num_tokens] i32 (KV-cache slot index, -1 = skip)
// Outputs:
//   q_out                : [num_tokens, num_heads, head_dim] f16 (rotated)
//   key_cache, value_cache : K/V written into per-slot positions
//                           [slot, num_kv_heads, head_dim]

#include <cuda_fp16.h>

extern "C"
__global__ void fused_rope_qwen_partial_f16kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __half* __restrict__ q_out,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int half_rot  = rotary_dim / 2;
    const int half_head = head_dim / 2;
    if (tid >= half_head) return;

    const int pos = positions[token_idx];

    // ── Q ──────────────────────────────────────────────────────────
    if (head_idx < num_heads) {
        const int q_base = (token_idx * num_heads + head_idx) * head_dim;
        if (tid < half_rot) {
            float c = __half2float(cos_table[pos * half_rot + tid]);
            float s = __half2float(sin_table[pos * half_rot + tid]);
            float lo = __half2float(q_in[q_base + tid]);
            float hi = __half2float(q_in[q_base + tid + half_rot]);
            q_out[q_base + tid]            = __float2half(lo * c - hi * s);
            q_out[q_base + tid + half_rot] = __float2half(lo * s + hi * c);
        } else {
            // Pass-through both halves of the un-rotated tail in one
            // thread (pair (tid, tid + half_head)).
            q_out[q_base + tid]             = q_in[q_base + tid];
            q_out[q_base + tid + half_head] = q_in[q_base + tid + half_head];
        }
    }

    // ── K + V ─────────────────────────────────────────────────────
    if (head_idx < num_kv_heads) {
        const int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        const int slot   = slot_mapping[token_idx];
        if (slot >= 0) {
            const int cache_off = (slot * num_kv_heads + head_idx) * head_dim;
            if (tid < half_rot) {
                float c = __half2float(cos_table[pos * half_rot + tid]);
                float s = __half2float(sin_table[pos * half_rot + tid]);
                float lo = __half2float(k_in[k_base + tid]);
                float hi = __half2float(k_in[k_base + tid + half_rot]);
                key_cache[cache_off + tid]            = __float2half(lo * c - hi * s);
                key_cache[cache_off + tid + half_rot] = __float2half(lo * s + hi * c);
            } else {
                key_cache[cache_off + tid]             = k_in[k_base + tid];
                key_cache[cache_off + tid + half_head] = k_in[k_base + tid + half_head];
            }
            // V: copy-only, no rotation. Cover both halves per thread
            // so we touch every element regardless of half_rot vs
            // half_head asymmetry.
            value_cache[cache_off + tid]             = v_in[k_base + tid];
            value_cache[cache_off + tid + half_head] = v_in[k_base + tid + half_head];
        }
    }
}
