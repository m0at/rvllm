// qwen_fill_pos_slots_i32: device-side fill of the per-token
// `positions` and `context_lens` arrays for Qwen 3.6 prefill.
//
// Replaces the legacy per-token host->device upload of an 8-byte
// `pos_cl_region` (position + context_len). The legacy upload used
// `cuMemcpyHtoD_v2` on the legacy default stream; the kernels that
// read those values run on `CU_STREAM_NON_BLOCKING` and the two
// streams do NOT synchronise. That race produced non-deterministic
// "Schule"-vs-"Supermarkt"-canary output (Round-25/26 codex review).
//
// This kernel runs on the same `self.stream` as everything else, so
// the fill is stream-ordered with the kernels that read the arrays
// (RoPE + KV-write, FA2 paged decode, FA2 prefill in Phase Full).
//
// Math is trivial:
//   positions[t]    = start_pos + t        (absolute token position)
//   context_lens[t] = start_pos + t + 1    (causal: attend over
//                                            slots [0, position+1))
//
// In Qwen 3.6 the KV-cache slot equals the absolute position, so
// callers reuse `positions` as `slot_mapping` (same buffer). Phase
// Full (batched FA2 prefill) reads `positions[0..N]` and
// `context_lens[0..N]` directly with grid.x=num_tokens; the per-
// token full-attn calls pass `positions + t*4` / `context_lens + t*4`
// as scalar-looking pointers (the existing fused_rope kernel has
// array semantics but is launched with grid.x=1 in decode mode).
//
// Launch: grid = (ceil(num_tokens / block), 1, 1), block = (256,
// 1, 1). Cheap, single launch per prefill or per request.

extern "C" __global__ void __launch_bounds__(256)
qwen_fill_pos_slots_i32_kernel(
    int* __restrict__ positions,       // [num_tokens] i32
    int* __restrict__ context_lens,    // [num_tokens] i32
    int start_pos,
    int num_tokens
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    int abs = start_pos + t;
    positions[t]    = abs;
    context_lens[t] = abs + 1;
}
