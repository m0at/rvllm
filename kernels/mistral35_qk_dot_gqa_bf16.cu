// Mistral 3.5 GQA-aware Q·Kᵀ scoring for the m=1 decode attention path.
//
// Drop-in replacement for `mistral35_qk_dot_bf16` that exploits the
// gqa_ratio (= 12 for Mistral 3.5: 96 Q-heads / 8 KV-heads) to load
// `K[t, kv_head, :]` once per CTA and reuse it across all `gqa_ratio`
// Q-heads sharing that kv_head. The original kernel had grid
// (n_q_heads, past_len) = (96, N) where each (q_head, t) block did
// its own K read — total K bandwidth ~12× wasted vs a GQA-aware
// scheduler.
//
// Memory layouts (BF16):
//   Q:        [n_q_heads,  head_dim]
//   K_cache:  [max_pos,    n_kv_heads, head_dim]
//
// Output layout (F32, same as the legacy kernel):
//   scores:   [n_q_heads,  past_len]
//
// Launch:
//   Grid:  (n_kv_heads, past_len, 1)
//   Block: (head_dim, 1, 1)        — head_dim ≤ 1024 (Mistral: 128)
//   Shared mem layout:
//     k_smem [head_dim] f32  — one BF16-loaded-as-f32 K row per CTA
//     red    [head_dim] f32  — tree-reduce scratch shared with k_smem
//   Total smem: head_dim * sizeof(float).
//
// K bandwidth is reduced by `gqa_ratio` (12× for Mistral 3.5).
// Q bandwidth stays the same (each q_head's row read once per
// (kv_head, t) by all threads via global memory broadcast).
//
// Round-12 phase 5c codex review #2 — partial fix (qk_dot side).
// `mistral35_softmax_v_bf16` still has the redundant V load; the
// matching GQA-aware finisher lands in a follow-up.

#include <cuda_bf16.h>

extern "C" __global__ void mistral35_qk_dot_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    float* __restrict__ scores,
    int head_dim,
    int n_q_heads,
    int n_kv_heads,
    int gqa_ratio,
    int past_len,
    float inv_sqrt_d
) {
    const int kv_head = blockIdx.x;
    const int t       = blockIdx.y;
    const int d       = threadIdx.x;
    if (kv_head >= n_kv_heads || t >= past_len || d >= head_dim) return;

    extern __shared__ float smem[];

    // Stage 1: every thread loads ONE element of K[t, kv_head, :] into
    // shared memory. This is the load-once-per-(kv_head, t) part — no
    // matter how many Q-heads share this kv_head, K is fetched from
    // global only here.
    const float k_val = __bfloat162float(
        k_cache[(long long)t * n_kv_heads * head_dim
                + (long long)kv_head * head_dim + d]);
    smem[d] = k_val;
    __syncthreads();

    // Stage 2: per-Q-head dot product. We loop over the
    // `gqa_ratio` Q-heads that share this kv_head. Per iteration:
    // each thread reads its Q[qh, d] from global and multiplies with
    // the cached k_val (held in `kv` register from stage 1, OR
    // re-read from shared — register is faster but consumes one f32
    // per thread which is fine here). Tree reduction across head_dim
    // gives the dot product; thread 0 scales by inv_sqrt_d and
    // writes scores[qh, t].
    //
    // K stays in registers (`k_val`) — we re-use the same register
    // value across iterations. The smem region is repurposed each
    // iteration as the tree-reduce scratch.
    const int qh_base = kv_head * gqa_ratio;
    #pragma unroll 1
    for (int gi = 0; gi < gqa_ratio; ++gi) {
        const int qh = qh_base + gi;
        // Hoist Q[qh, d] read.
        const float qv = __bfloat162float(q[(long long)qh * head_dim + d]);

        // Per-element product into smem (overwriting K cache, which
        // we already consumed into `k_val`).
        smem[d] = qv * k_val;
        __syncthreads();

        // Tree reduction over head_dim threads. head_dim is even
        // (128 for Mistral 3.5).
        for (int s = head_dim >> 1; s > 0; s >>= 1) {
            if (d < s) smem[d] += smem[d + s];
            __syncthreads();
        }
        if (d == 0) {
            scores[(long long)qh * past_len + t] = smem[0] * inv_sqrt_d;
        }
        // Repopulate smem[d] = k_val before next iter so the next
        // iteration's `smem[d] = qv * k_val` overwrites it cleanly
        // — but we don't actually read smem[d] before that overwrite,
        // so we just resync.
        __syncthreads();
    }
}
