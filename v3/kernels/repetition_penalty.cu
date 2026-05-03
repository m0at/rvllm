// Codex41-3: GPU-side Hugging-Face repetition-penalty.
//
// Replaces the host-side path that did:
//   1) cuStreamSynchronize
//   2) cuMemcpyDtoH (vocab*4 bytes)
//   3) edit `recent_ids.len()` entries
//   4) cuMemcpyHtoD (vocab*4 bytes)
// per decode step (~7ms on a 262k-vocab Gemma 4).
//
// Math (HF convention):
//   logits[id] = logits[id] >= 0 ? logits[id] / penalty
//                                : logits[id] * penalty
// for every `id` in `recent_ids` (no duplicates required; idempotent
// per id since we run the kernel once with all recent ids batched).
//
// Grid:  (1, 1, 1)
// Block: (min(num_ids, 1024), 1, 1)
//   Each thread handles one id; for num_ids > 1024 we loop.
//
// Inputs:
//   logits:   [vocab] f32 (mutated in place)
//   ids:      [num_ids] i32 (token IDs to penalize)
//   num_ids:  count
//   vocab:    bound check
//   penalty:  > 1.0 (typically 1.05 for Gemma 4)

#include <cstdint>

extern "C"
__global__ void apply_repetition_penalty_f32_kernel(
    float*         __restrict__ logits,
    const int32_t* __restrict__ ids,
    int            num_ids,
    int            vocab,
    float          penalty
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    for (int i = tid; i < num_ids; i += stride) {
        int32_t id = ids[i];
        if (id < 0 || id >= vocab) continue;
        float v = logits[id];
        // HF convention: scale toward zero.
        v = (v >= 0.0f) ? v / penalty : v * penalty;
        logits[id] = v;
    }
}
