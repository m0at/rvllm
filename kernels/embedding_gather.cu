// Embedding gather kernel: directly gathers embedding rows on GPU,
// avoiding the GPU->CPU->GPU round-trip for embedding lookup.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
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

    // Bounds check: out-of-range tokens get zeros
    if (token_id < 0 || token_id >= vocab_size) {
        for (int i = tid; i < hidden_size; i += stride) {
            output[out_offset + i] = 0.0f;
        }
        return;
    }

    const int embed_offset = token_id * hidden_size;
    for (int i = tid; i < hidden_size; i += stride) {
        output[out_offset + i] = embed_table[embed_offset + i];
    }
}
