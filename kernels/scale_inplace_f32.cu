// Pointwise scalar scale for f32: x[i] *= scale.
//
// f32 sibling of scale_inplace_f16. Used by the f32-pooler path to
// apply Gemma's `*= sqrt(hidden_size)` factor (≈ 33.94) without
// risking f16 overflow on peak post-encoder activations.
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)

extern "C" __global__ void scale_inplace_f32_kernel(
    float* __restrict__ x,
    float scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= scale;
}
