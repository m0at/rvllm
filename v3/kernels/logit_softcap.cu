// Logit softcapping: logits = cap * tanh(logits / cap)
//
// Applied in-place on f16 logits before argmax sampling.
// Gemma 3/4 uses cap=30.0.
//
// Grid:  (num_tokens, 1, 1)
// Block: (min(vocab, 1024), 1, 1)

#include <cuda_fp16.h>

extern "C"
__global__ void logit_softcap_kernel(
    __half* __restrict__ logits,    // [num_tokens, vocab]
    int vocab,
    float cap
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float inv_cap = 1.0f / cap;

    __half* row_ptr = logits + row * vocab;

    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = __half2float(row_ptr[i]);
        float capped = cap * tanhf(v * inv_cap);
        row_ptr[i] = __float2half(capped);
    }
}

// Codex40-2: f32 variant for the generate path, which samples directly
// from f32 logits (no intermediate f16 conversion). PPL/bench convert
// f32→f16 and use the f16 kernel above; generate previously skipped
// the softcap entirely, drifting from the trained-token-distribution.
extern "C"
__global__ void logit_softcap_f32_kernel(
    float* __restrict__ logits,     // [num_tokens, vocab]
    int vocab,
    float cap
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float inv_cap = 1.0f / cap;

    float* row_ptr = logits + row * vocab;

    for (int i = tid; i < vocab; i += blockDim.x) {
        row_ptr[i] = cap * tanhf(row_ptr[i] * inv_cap);
    }
}
