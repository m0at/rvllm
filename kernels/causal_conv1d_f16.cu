// causal_conv1d_f16: 1D depthwise causal convolution for the Qwen 3.6
// Gated-DeltaNet (qwen3-next) linear-attention block. Each channel
// has its own kernel-size-`ks` filter; output[t, c] is the dot
// product of the previous `ks` input timesteps for channel `c`.
//
// Layout convention: caller arranges `input` so positions
// `[t, t+1, ..., t+ks-1]` are the historical window feeding output
// position `t` (i.e. the conv1d state cache prepends ks-1 history
// timesteps before the new tokens). Weight layout matches HF:
// `weight[c, 0, k]` where k=0 is the oldest position in the window
// and k=ks-1 is the most-recent.
//
//   out[t, c] = Σ_{k=0..ks-1} input[t+k, c] * weight[c, 0, k]
//
// Grid: (ceil(channels/BLOCK), seq_len). Block: (BLOCK, 1).

#include <cuda_fp16.h>

extern "C" __global__ void causal_conv1d_f16_kernel(
    __half* __restrict__ output,            // [seq_len, channels] f16
    const __half* __restrict__ input,       // [seq_len + ks - 1, channels] f16
    const __half* __restrict__ weight,      // [channels, 1, ks] f16
    int seq_len,
    int channels,
    int ks
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= channels || t >= seq_len) return;
    float acc = 0.0f;
    const __half* w_row = weight + (long long)c * ks;
    for (int k = 0; k < ks; ++k) {
        float i_v = __half2float(input[(long long)(t + k) * channels + c]);
        float w_v = __half2float(w_row[k]);
        acc += i_v * w_v;
    }
    output[(long long)t * channels + c] = __float2half(acc);
}
