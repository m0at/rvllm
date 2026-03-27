// Fused residual add + RMS normalization kernel.
//
// Combines the two most common back-to-back operations in a transformer layer:
//   1. residual = input + attn_output  (or mlp_output)
//   2. output = rmsnorm(residual, weight, eps)
//
// Fusing these saves one full GPU memory round-trip for the residual tensor.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared memory: blockDim.x * sizeof(float)

extern "C"
__global__ void fused_residual_rmsnorm_kernel(
    float* __restrict__ output,        // [num_tokens, hidden_size] normalized output
    float* __restrict__ residual,      // [num_tokens, hidden_size] write-back residual (input + add)
    const float* __restrict__ input,   // [num_tokens, hidden_size]
    const float* __restrict__ add,     // [num_tokens, hidden_size] tensor to add
    const float* __restrict__ weight,  // [hidden_size]
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int row_offset = token_idx * hidden_size;

    extern __shared__ float sdata[];

    // Step 1: Fused residual add + sum-of-squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = input[row_offset + i] + add[row_offset + i];
        residual[row_offset + i] = val;
        local_ss += val * val;
    }
    sdata[tid] = local_ss;
    __syncthreads();

    // Parallel reduction for sum of squares
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Step 2: Compute RMS scale factor
    float rms_scale = rsqrtf(sdata[0] / (float)hidden_size + eps);

    // Step 3: Normalize and scale
    for (int i = tid; i < hidden_size; i += stride) {
        output[row_offset + i] = residual[row_offset + i] * weight[i] * rms_scale;
    }
}
