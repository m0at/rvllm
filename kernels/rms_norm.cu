// RMSNorm kernel: output[i] = input[i] * weight[i] / sqrt(mean(input^2) + eps)
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared memory: blockDim.x * sizeof(float) (for reduction)
//
// Each block processes one token (one row of the input matrix).
// Uses shared memory parallel reduction for sum-of-squares.

extern "C"
__global__ void rms_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* x = input + token_idx * hidden_size;
    float* y = output + token_idx * hidden_size;

    extern __shared__ float sdata[];

    // Step 1: Compute partial sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = x[i];
        local_ss += val * val;
    }
    sdata[tid] = local_ss;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Step 2: Compute RMS scale factor
    float rms_scale = rsqrtf(sdata[0] / (float)hidden_size + eps);

    // Step 3: Apply normalization with weight
    for (int i = tid; i < hidden_size; i += stride) {
        y[i] = x[i] * weight[i] * rms_scale;
    }
}
