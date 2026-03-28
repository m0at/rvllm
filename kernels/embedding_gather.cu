// Embedding gather kernel: directly gathers embedding rows on GPU,
// avoiding the GPU->CPU->GPU round-trip for embedding lookup.
//
// Uses float4 vectorized loads/stores (128-bit) for 4x memory throughput.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size/4, 1024), 1, 1)  -- or (min(hidden_size, 1024), 1, 1) for non-aligned
//   Shared memory: none

extern "C"
__global__ void embedding_gather_kernel(
    float* __restrict__ output,            // [num_tokens, hidden_size]
    const float* __restrict__ embed_table, // [vocab_size, hidden_size]
    const int* __restrict__ token_ids,     // [num_tokens]
    int hidden_size,
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int token_id = token_ids[token_idx];
    const int out_offset = token_idx * hidden_size;

    // Vectorized path: hidden_size divisible by 4 (common: 4096, 5120, etc.)
    const int vec_size = hidden_size >> 2;  // hidden_size / 4

    if (token_id < 0 || token_id >= vocab_size) {
        // Zero fill for out-of-range tokens
        if ((hidden_size & 3) == 0) {
            float4* out4 = reinterpret_cast<float4*>(output + out_offset);
            const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = tid; i < vec_size; i += stride) {
                out4[i] = zero4;
            }
        } else {
            for (int i = tid; i < hidden_size; i += stride) {
                output[out_offset + i] = 0.0f;
            }
        }
        return;
    }

    const int embed_offset = token_id * hidden_size;

    if ((hidden_size & 3) == 0) {
        // Vectorized: 128-bit loads/stores
        const float4* src4 = reinterpret_cast<const float4*>(embed_table + embed_offset);
        float4* dst4 = reinterpret_cast<float4*>(output + out_offset);
        for (int i = tid; i < vec_size; i += stride) {
            dst4[i] = src4[i];
        }
    } else {
        // Scalar fallback for non-aligned hidden sizes
        for (int i = tid; i < hidden_size; i += stride) {
            output[out_offset + i] = embed_table[embed_offset + i];
        }
    }
}
