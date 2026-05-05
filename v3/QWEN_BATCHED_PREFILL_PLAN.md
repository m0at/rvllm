# Qwen 3.6 batched-prefill plan

**Status:** in progress, Phase 1 landed (this commit).
**Goal:** flatten the per-token prefill loop in
`Qwen36Bringup::forward_qwen36_decode` so the layer-chain runs
on the full `[N, D]` prompt-hidden region in one pass, matching
Gemma's `unified_prefill` mode. End state: O(layers + log N)
launches per prefill instead of O(prompt_tokens × layers).

## Current state (post Round 16 #2 fence-drop + #3 GPU-argmax)

```rust
for tok_local in 0..num_tokens {
    DtoD-extract  hidden_region[off] → last_hidden_region   // 1 launch, no fence
    for layer L in 0..40 {
        apply_layer_linear_attn(last_hidden_region, …)      // ~470 LOC, 11+ fences, m=1
        // OR apply_layer_full_attn(...)                    // ~similar
        per-layer MoE block(last_hidden_region, …)          // m=1 grouped FP8 GEMV
    }
    DtoD-writeback last_hidden_region → hidden_region[off]   // 1 launch
}
final fence
forward_qwen36_outside_closer(hidden_region, …)
```

`last_hidden_region` is a single-token (`hidden=2048` × f16 = 4 kB)
buffer. Every projection call inside the layer functions runs
the **`fp8_gemv_wpr_native_f16in` GEMV kernel hard-coded at
`m: u32 = 1`**.

## Cost model (rough)

For a 1024-token prompt at 40 layers:
* **Launch overhead**: ≈40 × 1024 × ~10 launches/layer = 410 k
  launches × ~5 µs ≈ **2 s** of host-side launch chatter alone.
* **Fence overhead**: ≈11 × 40 × 1024 ≈ 450 k fences × ~2 µs ≈
  **0.9 s** (Round 16 partial-fix only dropped the 2 outer
  per-token fences; the inner 11/layer remain).
* **Kernel work** (GEMV at m=1): bandwidth-bound, ~hundreds of
  GB/s on GB10. Real GEMM (m=N) wins back the
  arithmetic-intensity headroom.

Switching to a real `[N, D]` batched prefill should:
* turn the 11/layer fences inside `apply_layer_linear_attn`
  into ~3/layer (only at recurrence boundaries),
* turn `m=1` GEMV into `m=N` GEMM (≈100× higher AI for N=1024),
* eliminate ≥2 DtoD copies per token.

## Risk profile

* **Silent garbage**: any kernel-ABI miswire produces tokens
  that look fine on the wire but are semantically wrong (Codex
  Round 16 #1 hardened against the embed→norm→lm_head
  fallback for exactly this reason). Per-phase byte-equivalence
  vs the per-token reference is mandatory.
* **Linear-attn recurrent state**: the only kernel that genuinely
  cannot be batched in the naïve sense — `state_t = f(state_{t-1},
  x_t)`. Choices:
  1. *Loop INSIDE the kernel* over N tokens with state held in
     shared/registers — eliminates host-loop launch overhead,
     no parallelism. Cheap, mechanical.
  2. *Chunked-recurrent* parallel form (RWKV / RetNet / Mamba2
     style) — true parallelism; significantly more kernel work.
  Phase 4 starts with (1); (2) is a future optimisation.

## Phase break-down

| Phase | Scope | LOC est. | Deliverable | Verify |
|---|---|---|---|---|
| **0** (done) | Round-16 partial fixes: GPU argmax, drop redundant per-token fences | ~80 | `qwen36_bring_up.rs` | Byte-equivalent canary |
| **1** (this commit) | Refactor `apply_layer_*` to take a raw token-slot pointer instead of a `Region`; eliminate the two per-token DtoD copies; caller passes `hidden_region.device_ptr() + tok_local × hidden_bytes` directly | ~150 | `qwen36_bring_up.rs` | Byte-equivalent canary on Qwen joke + vision |
| **2** | `bench-qwen-prefill` standalone binary: time prefill at N ∈ {32, 256, 1024, 4096}, log layer-by-layer breakdown | ~250 | new `crates/rvllm-runtime/src/bin/bench_qwen_prefill.rs` | Self-validating |
| **3a** (done) | Add the dispatcher entry point `Qwen36Bringup::fp8_proj_dispatch(out, w, scale, x_f16, m, n, k, …)`. m=1 path delegates to existing `Fp8GemvF16InLaunch` (byte-identical). m≥2 returns a typed `Phase 3b` deferral error instead of silently producing zeros. Verified by `probe-qwen-fp8-proj`: m=1 dispatcher output matches the direct GEMV launch byte-for-byte (16 384 bytes compared at q_proj of layer 3). | ~150 | `qwen36_bring_up.rs`, new `bin/probe_qwen_fp8_proj.rs` | `probe-qwen-fp8-proj` passes |
| **3b** (done) | Wire blockwise FP8 GEMM (per-token VEC128_32F activation scale + per-128×128 weight scale) into the dispatcher's m≥2 branch. New `fp8_quantize_per_token_f16.cu` kernel produces `[M, K] fp8` + `[M, K/128] f32` activation scales (cuBLASLt mode `VEC128_32F`). New `cublaslt.fp8_gemm_blockwise(...)` entry point sets both `A_SCALE_MODE = BLK128x128_32F` (mode 5) and `B_SCALE_MODE = VEC128_32F` (mode 4) on the matmul descriptor. **sm_121 caveat**: cuBLASLt 13.x does NOT ship a blockwise FP8 kernel for sm_121 — `AlgoGetHeuristic` returns no-algo. The dispatcher catches the heuristic error and falls back to looped-m=1 `Fp8GemvF16InLaunch` (the existing GEMV) so behaviour stays correct on this arch. On sm_100 / sm_120 (B100/200, RTX 5090, RTX 6000 Blackwell) the cuBLASLt blockwise tensor-core path dispatches directly. Verified by `probe-qwen-fp8-proj`: cosine = 1.000000 across m∈{2,4,16,64} on the q_proj@layer 3 shape (8192×2048) — the fallback IS the looped GEMV so output is byte-identical. | ~350 | `kernels/fp8_quantize_per_token_f16.cu`, `cublaslt.rs::fp8_gemm_blockwise`, `qwen36_bring_up.rs` | `probe-qwen-fp8-proj` cosine ≥ 0.9999 across m∈{2,4,16,64} |
| **3c** (done) | sm_121-specific perf: dispatcher's m≥128 branch now goes through CUTLASS SM120 blockwise FP8 GEMM (the same `libcutlass_sm120.so` Gemma loads at lm_head). New `fp8_quantize_per_token_amax_f16.cu` kernel (per-token amax sibling of the per-K-block kernel from 3b) feeds CUTLASS's `prep_sfa`; the existing `[N/128, K/128]` weight blockscale feeds `prep_sfb`. `Qwen36Bringup` gained a `cutlass: CutlassBackend` field loaded at bring-up. Dispatch: m=1 → GEMV (bit-identical), m∈[2,127] → cuBLASLt blockwise (sm_100/120 fast path) or looped-GEMV fallback (sm_121), m≥128 AND `SoSm120` available → CUTLASS SM120. Verified: probe at q_proj@layer 3 (n=8192, k=2048): m={128, 256} cosine ≥0.9997 vs looped-GEMV reference (relaxed threshold from 0.9999 because CUTLASS uses per-token-amax replicated vs the reference's per-K-block scaling — different but mathematically very close). | ~250 | `kernels/fp8_quantize_per_token_amax_f16.cu`, `qwen36_bring_up.rs` | `probe-qwen-fp8-proj` extended with m∈{128, 256}; cosine ≥ 0.999 (CUTLASS branch) and ≥ 0.9999 (cuBLASLt / GEMV branches) |
| **4a** (done) | Route every per-layer projection (`apply_layer_linear_attn`, `_full_attn`, `_moe` shared + routed experts) through `Qwen36Bringup::fp8_proj_dispatch` instead of direct `Fp8GemvF16InLaunch`. 13 call sites. At m=1 (today's caller) this dispatches byte-identically to the same GEMV kernel; the value is that any future caller flip from m=1 to m=N (Phase 4b/5/7) immediately picks up the cuBLASLt-blockwise (sm_100/120) or CUTLASS SM120 (sm_121) fast paths without further per-call-site edits. Greedy determinism canary (German joke) + Qwen vision smoke (test_224.png) both byte-identical post-routing. | ~150 | `qwen36_bring_up.rs` | cargo test 117 lib + 29 integration; Qwen text + vision E2E unchanged |
| **4b** | Make `apply_layer_full_attn` batched-causal: input `[N, D]`, output `[N, D]`, KV-cache slot writes vectorised. Reuse Gemma's batched FA2 prefill path. The body still has substantial host-side processing (DtoH q/k/v → CPU RoPE → HtoD; host-side residual add) that must be ported to GPU first — call this Phase 4b-prep. Then the actual `[N, D]` conversion is mostly tile-shape arithmetic since the projections already accept m=N via the dispatcher (Phase 4a). | ~400 | `qwen36_bring_up.rs`, possibly new RoPE batched kernel | Per-layer cosine vs per-token reference (≥0.999), greedy canary byte-identical at m=1, then m=N=prompt_len works |
| **5** | Linear-attn: loop INSIDE the kernel over N tokens with state in shared. Same recurrent semantics, just batched at the launch level. | ~300 + new kernel | new `kernels/qwen_linear_attn_batched.cu` | Per-step state-equivalence vs per-token reference |
| **6** | MoE block batched: top-k routing over `[N, D]` produces `[N, K]` expert assignments; per-expert grouped GEMM. The existing `forward_layer3_full_moe_probe` path is per-token — needs the same `m: u32` extension as Phase 3. | ~500 | `qwen36_bring_up.rs`, MoE helpers | Cosine vs per-token reference |
| **7** | `forward_qwen36_decode` outer loop deleted: the layer-chain runs once over `hidden_region[0..N]`. Optional: CUDA Graph capture of the prefill, replayed for re-runs of the same shape. | ~150 | `qwen36_bring_up.rs` | TTFT measurement (Phase 2 harness) |

## Phase 1 (this commit)

### Change

`apply_layer_linear_attn` and `apply_layer_full_attn` previously
took `last_hidden_region: &rvllm_mem::Region<'_>` and called
`.device_ptr()` 28 places, plus one `copy_from_host` for the
host-residual-add path inside `apply_layer_linear_attn`. The
caller therefore had to set up a separate `last_hidden_region`
scratch and shuttle each token's slice in and out via DtoD
copies.

The refactor:

* Both functions now take **`last_hidden_ptr: u64`** and
  `last_hidden_bytes: usize`. The 28 `.device_ptr()` calls
  become direct uses of `last_hidden_ptr`. The single
  `copy_from_host` becomes a raw `cuMemcpyHtoDAsync_v2`.
* Caller (line ~4500 in `forward_qwen36_decode`) computes
  `let tok_ptr = hidden_region.device_ptr() + (tok_local as
  u64) * (last_hidden_bytes as u64);` and passes it.
* The two `cuMemcpyDtoDAsync_v2` calls (extract + writeback)
  in the per-token loop are deleted. Layer kernels now read
  and write directly into `hidden_region` at the per-token
  offset.

### Why this is a no-op for correctness

Each layer function's reads and writes through
`last_hidden_region.device_ptr()` are equivalent to reads and
writes through `hidden_region.device_ptr() + tok_local ×
hidden_bytes` provided the offset is constant for the duration
of one layer-chain pass. Since the per-token outer loop pins
`tok_local`, the offset is constant — same byte addresses, just
no intermediate scratch.

### What this enables

Phase 3+ can change the layer functions' `m: u32 = 1` to
`m: u32 = num_tokens` and pass `hidden_region.device_ptr()`
(no offset) plus `num_tokens` — same code path, just operating
on `[N, D]` instead of `[1, D]`. The Phase 1 refactor is the
mechanical groundwork that makes that swap one signature change
away from working.

### Performance gain in this phase alone

Tiny: ~2 DtoDAsync launches × ~5 µs/launch saved per token =
~1 ms on a 100-token prompt. The gain is in *enabling* Phase 3,
not the saved copies themselves.

## Open questions for later phases

* Does the existing `fused_rmsnorm_fp8_quant` kernel handle
  `m > 1` correctly? Its caller chain assumes per-token output;
  output layout for `[N, D]` may need a wrapper.
* Linear-attn conv state: 1-D conv across the recurrent
  history — needs to be batched across token positions while
  staying causal.
* CUDA Graph capture (Phase 7) interacts badly with linear-attn
  state if the graph is replayed across requests; must
  capture-per-request or expose param updates for the state
  region pointer.

These are solvable, just non-trivial. They get chosen during
Phase 4 / Phase 5 based on bench-harness numbers from Phase 2.
