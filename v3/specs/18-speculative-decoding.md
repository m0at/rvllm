# Spec 18 — Speculative Decoding (draft + verify)

**Status:** Planned. Not implemented.
**Expected win:** 2–3× decode throughput at low–medium batch sizes.
**Effort:** Real project — week of work, multiple subsystems.

## Problem

Decode throughput is bandwidth-bound on the target model's weights.
At N=128 we're already close to peak; at smaller batches we're much
lower. Target-model decode is wasted bandwidth for "obvious" tokens.

Speculative decoding amortizes target weight reads: a small draft
model proposes K tokens; the target model verifies K+1 tokens in ONE
forward pass (still one weight read), accepts a prefix, discards the
rest. Effective throughput = (accepted tokens) / (target step time).

For well-behaved chat workloads, acceptance rates of 0.5–0.75 are
routine (K=4 draft tokens, ~2–3 accepted per target step), giving
2–3× speedup.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Draft model (Qwen2.5-0.5B or 1.5B)              │
│    - Same tokenizer as target                    │
│    - Autoregressive decode K tokens from x_t     │
│    - Runs on same H100, smaller KV share         │
├──────────────────────────────────────────────────┤
│  Verify                                          │
│    - Target forward on K+1 query positions in ONE│
│      step (K draft + current)                    │
│    - Argmax each target position                 │
│    - Accept longest common prefix with draft     │
│    - Commit accepted + one "free" target sample  │
└──────────────────────────────────────────────────┘
```

## Subsystems affected

- **Runtime.** `Engine::step_launch` becomes "draft K then verify K+1"
  instead of "decode 1". Graph capture captures TWO graphs per bucket:
  draft-step (batch × 1) and verify-step (batch × K+1).
- **Scheduler.** Each request has two live positions (draft current +
  target current). When draft proposes tokens, they go into a "proposed"
  queue; verify pops the front K, target accepts some prefix, scheduler
  commits accepted and re-drafts from the last accepted.
- **KV cache.** Target KV pages are written AT VERIFY TIME, not at draft
  time. Rejected draft tokens never hit target KV. Draft KV pages are a
  separate cache (draft model's own blocks).
- **Attention.** Verify is a multi-query prefill-like step: K+1 queries
  attending to the full context. FA3's paged prefill path (already in
  `prefill.rs`, currently stubbed) is the right kernel — implement its
  FFI wiring.
- **Loader.** Two safetensors loads: target + draft. Draft gets its own
  FP8 quantization pass.

## Correctness

Speculative decoding is mathematically equivalent to the target model's
autoregressive greedy decode when:
- Draft and target share a tokenizer.
- Acceptance uses exact token match (for greedy) or proper rejection
  sampling (for sampled decode).

For v3's current greedy path, acceptance is: compare draft token at
position i to target argmax at position i; accept iff equal; stop at
first mismatch; the target argmax at the mismatch position is an
additional "free" sample.

## Key numbers (target)

- Draft: Qwen2.5-0.5B-Instruct, FP8, batch-shared with target on same H100.
  Memory footprint ~500 MB + draft KV.
- K = 4 proposed tokens per step (tunable).
- Acceptance rate: 0.5–0.75 on chat workloads (needs measurement).
- Target step at N=128 currently: 6.15 ms → verify-step slightly longer
  (K+1 queries). Call it 7 ms. Draft step at batch=128 small model:
  ~2 ms. Total per draft-verify iteration: ~9 ms, producing (1 + K×accept)
  ≈ 3–4 accepted tokens. Effective tok/s: ~50,000+ at N=128.

## Files to touch (rough scope)

- `v3/crates/rvllm-runtime/src/engine.rs` — two-stage step, draft + verify.
- `v3/crates/rvllm-runtime/src/scheduler.rs` — proposed-token queue per request.
- `v3/crates/rvllm-runtime/src/bring_up.rs` — load draft alongside target;
  two graph-captures per bucket.
- `v3/crates/rvllm-attention/src/prefill.rs` — implement paged_prefill FFI
  (currently stub). Verify step uses it.
- `v3/crates/rvllm-loader/src/load.rs` — accept two model dirs.
- `kernels/` — likely no new kernels; verify reuses target's layer_exec
  with num_tokens = (K+1) × batch.
- `v3/crates/rvllm-bench/src/main.rs` — bench harness for spec-decode
  with acceptance-rate reporting.

## Dependencies

- Spec 17 (CUTLASS EVT) is orthogonal. Can land before or after.
- Spec 15 (FA3 paged_prefill FFI) must be implemented — the verify step
  is a multi-query attention, which is what paged_prefill solves.
- Prefill in general needs to exist in the engine (not just the bench's
  faux-prefill). Target's prefill can reuse the same `layer_exec`
  with `num_tokens = input_len × batch`.

## Verification

- Correctness: speculative greedy output must be byte-identical to
  non-speculative greedy output on a fixed prompt (first 64 tokens).
- Acceptance-rate report: print mean K_accept / K_proposed across the
  bench.
- Bench: tok/s at N=32, 64, 128 with K=4 draft tokens vs baseline.
- Sweep K ∈ {2, 4, 8} to find the optimum.

## Risks

1. **Draft KV memory.** Draft's KV cache eats into target's. Probably
   fine at 7B target + 0.5B draft on H100 80GB, but tight. Monitor
   arena used vs available.
2. **Acceptance rate variability.** Draft quality matters more than
   any other parameter. 0.5B may be too weak for 7B target on some
   workloads; may need 1.5B draft → larger memory footprint.
3. **Graph capture complexity.** Two bucket graphs per batch size
   (draft × 1, verify × K+1) doubles the capture surface. Need to
   extend `GraphPool` to key on (bucket, phase).
4. **Prefill path in engine.** Currently the engine has no prefill;
   v3 has only been exercised on decode. Prefill must land first
   (it's on the roadmap regardless).

## Out of scope (future)

- Sampled decode (temperature > 0) with proper rejection sampling
  — harder math; start with greedy.
- Tree-structured speculative decoding (multiple draft branches).
- Medusa-style speculative (multiple extra heads instead of a draft
  model). Different architecture choice.
