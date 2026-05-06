// Conv1d state-advance + conv_in assembly for Qwen 3.6 linear-attn.
//
// Inputs:
//   conv_state_inout : `[3, ts]` f16 — three previous timesteps,
//                      mutated in-place to the new state
//                      `[state_t-2, state_t-1, current]`.
//   current_qkv      : `[ts]`     f16 — this step's q/k/v projection.
// Outputs:
//   conv_in_out      : `[4, ts]`  f16 — `[state_t-3, state_t-2,
//                      state_t-1, current]` for the conv1d kernel.
//   conv_state_inout — updated as above.
//
// Replaces a host pipeline of:
//   DtoH qkv (~32 KB) +
//   DtoH conv_state (~96 KB) +
//   CPU vec slicing into conv_in_host +
//   HtoD conv_in (~128 KB) +
//   CPU vec slicing into conv_state_new +
//   HtoD conv_state (~96 KB)
// with one launch (Phase 4b-prep iter4).
//
// The kernel reads all three old state slots into registers BEFORE
// writing anything, so the in-place state update is safe (no aliasing
// hazard between the read of `conv_state[1*ts+i]` and the write of
// `conv_state[0*ts+i]`).
//
// Launch:
//   Grid:  ((ts + block - 1) / block, 1, 1)
//   Block: (256, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256)
conv_state_advance_f16_kernel(
    __half* __restrict__       conv_in_out,
    __half* __restrict__       conv_state_inout,
    const __half* __restrict__ current_qkv,
    int ts
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ts) return;

    // Snapshot old state to registers first.
    const __half s0 = conv_state_inout[0 * ts + i];
    const __half s1 = conv_state_inout[1 * ts + i];
    const __half s2 = conv_state_inout[2 * ts + i];
    const __half c  = current_qkv[i];

    // Assemble conv_in.
    conv_in_out[0 * ts + i] = s0;
    conv_in_out[1 * ts + i] = s1;
    conv_in_out[2 * ts + i] = s2;
    conv_in_out[3 * ts + i] = c;

    // Advance state to [s1, s2, current].
    conv_state_inout[0 * ts + i] = s1;
    conv_state_inout[1 * ts + i] = s2;
    conv_state_inout[2 * ts + i] = c;
}
