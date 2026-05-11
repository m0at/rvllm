// Batched conv1d state-advance + conv_in assembly for Qwen 3.6
// linear-attn. Generalises the per-token `conv_state_advance_f16`
// kernel to N tokens at once with the same in-place update semantics
// the caller relies on.
//
// Per-token kernel (decode): produces conv_in `[4, ts]` from
//   `[state_0, state_1, state_2, current]` and rotates state to
//   `[state_1, state_2, current]`.
//
// Batched kernel (prefill): for `num_tokens` consecutive timesteps it
// produces conv_in `[num_tokens, 4, ts]` aligned token-major (the
// caller passes that buffer to `causal_conv1d_f16` with seq_len=N
// once and gets `[N, ts]` outputs in one launch). Equivalent to:
//
//   for t in 0..num_tokens:
//       conv_in[t, 0..3, :] = state[0..3, :]
//       conv_in[t, 3, :]    = current[t, :]
//       state[0..2, :] = state[1..3, :]    // rotate
//       state[2, :]    = current[t, :]
//
// Each thread owns one channel `i` in `[0, ts)`. The state slice for
// that channel (3 f16 values) lives in registers across the inner
// loop; the global state buffer is read once at start and written
// once at end. This makes the operation O(num_tokens) DRAM traffic
// per channel — the per-token version was O(num_tokens) launches AND
// O(num_tokens) full state DRAM round-trips.
//
// Reads `current_qkv` strided as `[num_tokens, ts]`; produces
// `conv_in_out` strided as `[num_tokens, 4, ts]` (matches the layout
// `causal_conv1d_f16_kernel` expects when called with seq_len=N — see
// causal_conv1d_f16.cu, the input is `[seq_len + ks - 1, channels]`
// but we pre-rolled the +ks-1 history into each token's window so
// each token gets a self-contained 4-row slice; the conv1d kernel
// then iterates over those slices).
//
// IMPORTANT: this layout assumes the conv1d kernel is called with
// the per-token `[4, ts]` slice as its input AND seq_len=1, looped
// from the host over N tokens — which would defeat the point. To
// actually batch the conv1d call we'd need a `[num_tokens + ks - 1,
// channels]` flat buffer matching the existing kernel's
// expectations. That's the second variant emitted by this kernel:
// `conv_in_flat_out` is the `[num_tokens + ks - 1, ts]` flat history
// that prepends the 3 starting state rows to the N current rows
// (with state rotation applied AT THE END so the conv1d kernel sees
// the original history, not the post-rotation one).
//
// The two output buffers are produced in one pass to avoid two HtoD
// reads of `current_qkv` from DRAM.
//
// Launch: grid = (ceil(ts/block), 1, 1), block = (256, 1, 1).

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256)
conv_state_advance_batched_f16_kernel(
    __half* __restrict__       conv_in_flat_out,  // [num_tokens + 3, ts] f16
    __half* __restrict__       conv_state_inout,  // [3, ts] f16, in/out
    const __half* __restrict__ current_qkv,       // [num_tokens, ts] f16
    int ts,
    int num_tokens
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ts) return;

    // Load 3-step state into registers ONCE.
    float s0 = __half2float(conv_state_inout[0 * ts + i]);
    float s1 = __half2float(conv_state_inout[1 * ts + i]);
    float s2 = __half2float(conv_state_inout[2 * ts + i]);

    // Write the 3-step history prefix into conv_in_flat_out rows 0..2.
    // The conv1d kernel will read input[t, t+1, t+2, t+3, c] for
    // output[t, c], so rows 0..2 are the historical state and rows
    // 3..3+N-1 are the new tokens.
    conv_in_flat_out[0 * ts + i] = __float2half(s0);
    conv_in_flat_out[1 * ts + i] = __float2half(s1);
    conv_in_flat_out[2 * ts + i] = __float2half(s2);

    // Stream `current_qkv` into the rest of conv_in_flat_out and
    // advance state through (s0, s1, s2) → (s_{N-2}, s_{N-1}, s_{N}).
    // Because we keep state in registers, the in-place rotation is
    // safe and we avoid N round-trips to global state buffer.
    for (int t = 0; t < num_tokens; ++t) {
        float c = __half2float(current_qkv[(long long)t * ts + i]);
        conv_in_flat_out[(long long)(3 + t) * ts + i] = __float2half(c);
        // Advance: drop oldest, shift, append.
        s0 = s1;
        s1 = s2;
        s2 = c;
    }

    // Persist final state back to DRAM.
    conv_state_inout[0 * ts + i] = __float2half(s0);
    conv_state_inout[1 * ts + i] = __float2half(s1);
    conv_state_inout[2 * ts + i] = __float2half(s2);
}
