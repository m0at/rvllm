// BF16-precision embedding gather kernel: gathers embedding rows on GPU.
//
// Cycle 55 step 11 (Phase C): bf16 sibling of embedding_gather_f16.
// Identical pure-copy semantics; embed table + output flip
// __half → __nv_bfloat16. Eliminates the f16→bf16 widen previously
// done in cycle 54 right after the f16 embedding_gather.
//
// Phase D (bf16-native model loading) will switch the embedding
// table itself to bf16 storage at load time so this kernel reads
// directly from the trained-distribution bytes; until then, callers
// can use this kernel only after the embedding table is materialized
// as bf16 in device memory.
//
// Launch config (matches f16 sibling):
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared memory: none

#include <cuda_bf16.h>

extern "C"
__global__ void embedding_gather_bf16_kernel(
    __nv_bfloat16* __restrict__ output,            // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ embed_table, // [vocab_size, hidden_size]
    const int* __restrict__ token_ids,             // [num_tokens]
    int hidden_size,
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int token_id = token_ids[token_idx];
    const int out_offset = token_idx * hidden_size;

    // Bounds check: out-of-range tokens get zeros
    if (token_id < 0 || token_id >= vocab_size) {
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
        for (int i = tid; i < hidden_size; i += stride) {
            output[out_offset + i] = zero;
        }
        return;
    }

    const int embed_offset = token_id * hidden_size;
    for (int i = tid; i < hidden_size; i += stride) {
        output[out_offset + i] = embed_table[embed_offset + i];
    }
}
