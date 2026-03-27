// Activation function kernels: SiLU, fused SiLU*mul, GELU.
//
// Launch config (all kernels):
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none
//
// Simple element-wise operations, one thread per element.

extern "C"
__global__ void silu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// Fused SiLU(gate) * up for MLP gate projection.
// Avoids an extra kernel launch and memory round-trip.
extern "C"
__global__ void fused_silu_mul_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = gate[idx];
        output[idx] = (x / (1.0f + expf(-x))) * up[idx];
    }
}

// GELU with tanh approximation (matches PyTorch's gelu(approximate='tanh')).
extern "C"
__global__ void gelu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // 0.7978845608 = sqrt(2/pi)
        // 0.044715 = empirical constant from the GELU paper
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
