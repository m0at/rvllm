// Half-precision embedding gather kernel: gathers embedding rows on GPU.
// Embedding table is f16, output is f16. No f32 conversion needed -- pure copy.
//
// Uses 128-bit vectorized loads/stores (8 x __half per transaction) for max
// memory throughput. Reinterprets __half pointers as uint4 (128-bit).
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size/8, 1024), 1, 1)  -- or (min(hidden_size, 1024), 1, 1) for non-aligned
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void embedding_gather_f16_kernel(
    __half* __restrict__ output,            // [num_tokens, hidden_size]
    const __half* __restrict__ embed_table, // [vocab_size, hidden_size]
    const int* __restrict__ token_ids,      // [num_tokens]
    int hidden_size,
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int token_id = token_ids[token_idx];
    const int out_offset = token_idx * hidden_size;

    // 128-bit = 16 bytes = 8 __half elements
    const int vec_elems = 8;
    const int vec_size = hidden_size / vec_elems;

    if (token_id < 0 || token_id >= vocab_size) {
        // Zero fill for out-of-range tokens
        if ((hidden_size & 7) == 0) {
            uint4* out_v = reinterpret_cast<uint4*>(output + out_offset);
            const uint4 zero_v = make_uint4(0u, 0u, 0u, 0u);
            for (int i = tid; i < vec_size; i += stride) {
                out_v[i] = zero_v;
            }
        } else {
            const __half zero = __float2half(0.0f);
            for (int i = tid; i < hidden_size; i += stride) {
                output[out_offset + i] = zero;
            }
        }
        return;
    }

    const int embed_offset = token_id * hidden_size;

    if ((hidden_size & 7) == 0) {
        // Vectorized: 128-bit loads/stores (8 halfs per transaction)
        const uint4* src_v = reinterpret_cast<const uint4*>(embed_table + embed_offset);
        uint4* dst_v = reinterpret_cast<uint4*>(output + out_offset);
        for (int i = tid; i < vec_size; i += stride) {
            dst_v[i] = src_v[i];
        }
    } else {
        // Scalar fallback for non-aligned hidden sizes
        for (int i = tid; i < hidden_size; i += stride) {
            output[out_offset + i] = embed_table[embed_offset + i];
        }
    }
}
