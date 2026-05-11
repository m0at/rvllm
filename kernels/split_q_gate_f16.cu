// Per-head split of q_proj output `[num_heads, 2*head_dim]` into
// q `[num_heads, head_dim]` and gate `[num_heads, head_dim]`.
//
// Replaces the host pipeline in `apply_layer_full_attn`:
//   DtoH q_proj output ([num_heads × 2 × head_dim] f16) →
//   CPU copy_from_slice loop over each head splitting (q | gate) →
//   HtoD q_split + HtoD gate_split
// with one GPU launch.
//
// Layout: q_proj writes head h to bytes
// `[h * 2 * head_dim, (h+1) * 2 * head_dim)`. The first half is Q;
// the second half is the gate (per Qwen 3.6 chunk(-1, 2)).
//
// Launch:
//   Grid:  (num_heads, num_tokens, 1)
//   Block: (head_dim, 1, 1)            (one thread per element)

#include <cuda_fp16.h>

extern "C" __global__ void split_q_gate_f16_kernel(
    __half* __restrict__ q_out,
    __half* __restrict__ gate_out,
    const __half* __restrict__ qg_in,
    int num_heads,
    int head_dim
) {
    const int h     = blockIdx.x;
    const int token = blockIdx.y;
    const int d     = threadIdx.x;
    if (d >= head_dim) return;

    const long long head_stride_in  = (long long)2 * head_dim;
    const long long token_stride_in = (long long)num_heads * head_stride_in;
    const long long token_stride_out = (long long)num_heads * head_dim;

    const long long src_base   = token * token_stride_in + h * head_stride_in;
    const long long dst_base   = token * token_stride_out + h * head_dim;

    q_out[dst_base + d]    = qg_in[src_base + d];
    gate_out[dst_base + d] = qg_in[src_base + head_dim + d];
}
