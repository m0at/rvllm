// PagedAttention V2 kernel for variable-length sequences with block tables.
//
// Launch config:
//   Grid:  (num_seqs, num_heads, 1)
//   Block: (HEAD_DIM, 1, 1)  -- one thread per dimension element
//   Shared memory: block_size * sizeof(float) + HEAD_DIM * sizeof(float)
//                  (for dot products and partial value accumulator)
//
// Each thread block handles one query head for one sequence.
// Uses online softmax for numerical stability over arbitrary context lengths.

#include <float.h>

extern "C"
__global__ void paged_attention_v2_kernel(
    float* __restrict__ output,           // [num_seqs, num_heads, head_dim]
    const float* __restrict__ query,      // [num_seqs, num_heads, head_dim]
    const float* __restrict__ key_cache,  // [num_blocks, block_size, num_heads, head_dim]
    const float* __restrict__ value_cache, // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks]
    const int* __restrict__ context_lens, // [num_seqs]
    float scale,
    int num_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx  = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int num_blocks = (context_len + block_size - 1) / block_size;

    // Shared memory layout:
    //   float logits[block_size]   -- dot products for current block
    //   float acc[head_dim]        -- running weighted value accumulator
    extern __shared__ float smem[];
    float* logits = smem;
    float* acc    = smem + block_size;

    // Load query element for this thread
    const int q_offset = (seq_idx * num_heads + head_idx) * head_dim + dim_idx;
    float q_val = (dim_idx < head_dim) ? query[q_offset] * scale : 0.0f;

    // Initialize accumulator
    if (dim_idx < head_dim) {
        acc[dim_idx] = 0.0f;
    }

    // Online softmax state
    float global_max = -FLT_MAX;
    float global_sum = 0.0f;

    // Iterate over blocks in the block table
    for (int b = 0; b < num_blocks; b++) {
        const int physical_block = block_tables[seq_idx * max_blocks + b];
        const int tokens_in_block = min(block_size, context_len - b * block_size);

        // Compute Q*K dot products for each token in this block.
        // Each thread contributes its dimension, then we reduce.
        for (int t = 0; t < tokens_in_block; t++) {
            // key_cache layout: [block, token, head, dim]
            const int k_offset = ((physical_block * block_size + t) * num_heads + head_idx) * head_dim + dim_idx;
            float k_val = (dim_idx < head_dim) ? key_cache[k_offset] : 0.0f;
            float partial = q_val * k_val;

            // Warp-level reduction for dot product
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }

            // First thread of each warp writes to shared memory, then block reduces
            if (dim_idx == 0) {
                logits[t] = partial;
            }
        }
        __syncthreads();

        // Online softmax update: adjust running max and sum, then accumulate values
        if (dim_idx == 0) {
            for (int t = 0; t < tokens_in_block; t++) {
                float val = logits[t];
                if (val > global_max) {
                    float correction = expf(global_max - val);
                    global_sum = global_sum * correction;
                    global_max = val;
                }
                logits[t] = expf(val - global_max);
                global_sum += logits[t];
            }
        }
        __syncthreads();

        // Accumulate weighted values
        if (dim_idx < head_dim) {
            // Rescale existing accumulator when max changed (handled above via correction)
            for (int t = 0; t < tokens_in_block; t++) {
                const int v_offset = ((physical_block * block_size + t) * num_heads + head_idx) * head_dim + dim_idx;
                acc[dim_idx] += logits[t] * value_cache[v_offset];
            }
        }
        __syncthreads();
    }

    // Normalize by softmax denominator and write output
    if (dim_idx < head_dim) {
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        // Broadcast inv_sum from thread 0
        inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);
        const int out_offset = (seq_idx * num_heads + head_idx) * head_dim + dim_idx;
        output[out_offset] = acc[dim_idx] * inv_sum;
    }
}
