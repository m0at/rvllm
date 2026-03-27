// Element-wise bias addition kernel.
// Adds a per-feature bias vector to each row (token) of a 2D tensor.
// Replaces the CPU round-trip in gpu_worker.rs add_bias_gpu().
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(dim, 1024), 1, 1)
//   Shared memory: none

extern "C"
__global__ void add_bias_kernel(
    float* __restrict__ tensor,       // [num_tokens, dim] -- modified in-place
    const float* __restrict__ bias,   // [dim]
    int dim
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = token_idx * dim;

    for (int i = tid; i < dim; i += stride) {
        tensor[offset + i] += bias[i];
    }
}

// Element-wise tensor addition: output = a + b
// Replaces the CPU round-trip in gpu_worker.rs add_tensors_gpu().
//
// Launch config:
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none

extern "C"
__global__ void add_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

// In-place tensor addition: a += b
//
// Launch config:
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none

extern "C"
__global__ void add_inplace_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}
