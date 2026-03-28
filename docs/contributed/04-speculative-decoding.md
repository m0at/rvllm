# Spec 4: Speculative Decoding End-to-End

## Summary

Speculative decoding generates multiple draft tokens cheaply, then verifies them in a single target model forward pass. Accepted tokens skip individual decode steps, improving throughput for latency-bound workloads. vLLM supports 6 proposer types (EAGLE, Medusa, n-gram, draft model, etc.). rvLLM has a speculative decoding crate with correct verification logic and basic engine scaffolding, but no working proposer and no integration with the main engine.

This spec defines two implementation phases: first a working n-gram proposer (no model needed, validates the end-to-end pipeline), then a draft-model proposer (higher quality, requires model loading).

## vLLM Reference Behavior

### Propose-Verify Loop

Each step of speculative decoding:
1. **Propose**: The proposer generates K draft tokens per sequence using a cheap method (n-gram lookup, small draft model, or EAGLE head).
2. **Score**: The target model runs a single forward pass on the original token + K draft tokens (K+1 positions total), producing logits for all K+1 positions.
3. **Verify**: Rejection sampling compares draft token probabilities against target probabilities. Tokens are accepted left-to-right; the first rejection stops the chain. A bonus token is sampled from the corrected distribution at the rejection point.
4. **Update**: Accepted tokens are appended to the sequence. KV cache is updated accordingly. Rejected draft tokens' KV cache entries are discarded.

### N-gram Proposer

The simplest proposer. For each sequence, looks at the last N tokens and searches the sequence's own history for a matching N-gram. If found, the tokens following that N-gram in history become the draft.

vLLM's `NGramProposer` (`vllm/v1/spec_decode/ngram_proposer.py`):
- Configurable `ngram_size` (default 3) and `num_speculative_tokens` (typically 3-5)
- Searches request token history for longest matching suffix
- Falls back to random/no proposal if no match
- GPU-accelerated variant exists (`ngram_proposer_gpu.py`) using Triton kernels

### Rejection Sampling

vLLM's rejection sampler (`vllm/v1/spec_decode/rejection_sampler.py`):
- For each draft token at position i: accept with probability `min(1, target_prob[i] / draft_prob[i])`
- On rejection: sample from the corrected distribution `max(0, target_prob - draft_prob)` normalized
- The token after the last accepted draft token is always sampled fresh from the target distribution (bonus token)
- Supports both greedy (argmax comparison) and stochastic (probabilistic acceptance) modes

### Scheduler Integration

- Scheduler allocates extra KV cache slots for draft tokens (K extra slots per sequence)
- If a sequence can't fit K extra slots, it falls back to normal decode
- After verification, `num_computed_tokens` is advanced by (accepted_count + 1)
- Rejected draft tokens' KV cache slots are freed

### Key vLLM Files

- `vllm/v1/spec_decode/ngram_proposer.py` -- N-gram draft token proposal
- `vllm/v1/spec_decode/draft_model.py` -- Draft model proposal
- `vllm/v1/spec_decode/eagle.py` -- EAGLE proposer (most complex, highest quality)
- `vllm/v1/sample/rejection_sampler.py` -- Rejection sampling verification
- `vllm/v1/spec_decode/utils.py` -- KV cache management for draft tokens

## Current rvLLM State

### What Exists

**`crates/rvllm-speculative/`** contains:

1. **`config.rs`**: `SpeculativeConfig` with `draft_model_path`, `num_speculative_tokens` (default 3), `acceptance_threshold`, `enabled` flag.

2. **`verification.rs`**: `verify_tokens()` -- correct implementation of modified rejection sampling. Takes draft tokens + draft logprobs + target logprobs, returns `VerificationResult` with `accepted_tokens`, `bonus_token`, `num_accepted`. Well-tested with 6 unit tests: `self_speculation_accepts_all`, `zero_target_rejects_all`, `partial_acceptance`, `bonus_token_on_full_accept`, `probabilistic_acceptance_rate`, `empty_input`.

3. **`draft.rs`**: `DraftModelRunner` -- placeholder with no actual model inference. The interface is `fn generate_draft_tokens(&self, input_tokens: &[TokenId], num_tokens: usize) -> Result<Vec<DraftToken>>` where `DraftToken` has `token_id`, `logprob`, and `draft_probs: Vec<f32>`. The stub uses a deterministic formula based on the last context token: `selected = ((last as usize + i + 1) % self.vocab_size)` with prob 1.0 on the selected token.

4. **`engine.rs`**: `SpeculativeEngine` -- wraps an async engine with a propose-verify loop. Has metrics tracking (total_draft_tokens, accepted, bonus, steps). The loop structure is correct but calls the stub draft runner.

5. **`scheduler.rs`**: `SpeculativeScheduler` -- a standalone coordinator (not a wrapper around the base scheduler). Has `prepare_draft_and_target()` which runs the draft model and builds combined input for the target model. Does NOT have `reserve_draft_slots()`.

### What's Missing

1. **No working proposer**: `DraftModelRunner::generate_draft_tokens()` is a placeholder returning deterministic tokens.
2. **No integration with main engine**: `SpeculativeEngine` is standalone, not wired into `GpuLLMEngine` or `AsyncGpuLLMEngine`. Zero references to speculative in `rvllm-engine/` or `rvllm-config/`.
3. **No KV cache management for drafts**: Draft tokens need KV cache slots allocated, and rejected tokens need their slots freed.
4. **No n-gram proposer**: The simplest proposer type doesn't exist.
5. **Target model forward pass doesn't handle K+1 positions**: The worker needs to forward K+1 tokens per sequence (1 original + K draft) and return logits for all positions.
6. **No config wiring**: `rvllm-config` has zero references to speculative decoding. `EngineConfig` would need a new optional `SpeculativeConfig` field.

## Implementation Plan

### Phase 1: N-gram Proposer

**File**: `crates/rvllm-speculative/src/ngram.rs` (new)

Implement a CPU-based n-gram proposer:

```rust
pub struct NGramProposer {
    ngram_size: usize,
    num_speculative_tokens: usize,
}

impl NGramProposer {
    pub fn propose(&self, token_history: &[TokenId]) -> Vec<TokenId> {
        if token_history.len() < self.ngram_size {
            return vec![];
        }
        // Extract the last ngram_size tokens as the search pattern
        let pattern = &token_history[token_history.len() - self.ngram_size..];
        
        // Search backwards through history for a matching n-gram
        // (skip the last ngram_size tokens to avoid matching itself)
        let search_end = token_history.len() - self.ngram_size;
        for start in (0..search_end).rev() {
            if token_history[start..start + self.ngram_size] == *pattern {
                // Found a match -- tokens after this match are the draft
                let draft_start = start + self.ngram_size;
                let draft_end = (draft_start + self.num_speculative_tokens).min(search_end);
                if draft_start < draft_end {
                    return token_history[draft_start..draft_end].to_vec();
                }
            }
        }
        vec![] // No match found
    }
}
```

The n-gram proposer doesn't need draft logprobs for greedy verification (compare argmax of target vs draft token ID). For stochastic sampling, we approximate draft_prob as 1.0 for the proposed token (conservative -- always accepts if target agrees).

### Phase 2: Wire Speculative Engine into Main Engine

**Files**:
- `crates/rvllm-engine/src/gpu_engine.rs` (gated behind `#[cfg(feature = "cuda")]`)
- `crates/rvllm-speculative/src/engine.rs`
- `crates/rvllm-config/src/lib.rs` (add `SpeculativeConfig` to `EngineConfig`)

**Note**: `GpuLLMEngine` has tightly-coupled private methods (`step()` → `prepare_step()` → `build_metadata()` → `worker.execute()` → `process_worker_outputs()`). Integrating speculative decoding requires restructuring this pipeline to support K+1 token forward passes per sequence — this is significant effort.

The existing `SpeculativeScheduler.prepare_draft_and_target()` already does half of this work (running draft model and building combined input). Reference it during integration.

The speculative engine wraps the target engine. Each step becomes:

1. For each running decode sequence, call `ngram_proposer.propose(token_history)` to get K draft tokens. **If no n-gram match is found, fall back to normal single-token decode** — this decision point directly affects engine integration complexity.
2. Build a forward pass input with K+1 tokens per speculating sequence: the last accepted token + K draft tokens.
3. Call the target model forward to get logits for all K+1 positions.
4. Run `verify_tokens()` on each sequence using target logits.
5. Append accepted tokens + bonus token to the sequence.
6. Update `num_computed_tokens` by `num_accepted + 1`.

**Bonus token subtlety**: When all K drafts are accepted, the verification code uses `target_probs[k-1]` for the bonus token. The code comment says "In practice the engine provides K+1 target distributions." Ensure the target forward pass returns K+1 distributions (not just K) so the bonus token is sampled from the correct position.

### Phase 3: KV Cache Management for Draft Tokens

**Files**:
- `crates/rvllm-engine/src/gpu_engine.rs` (inline block allocator)
- `crates/rvllm-block-manager/src/manager.rs` (standalone path)
- `crates/rvllm-speculative/src/scheduler.rs`

**Note**: The CUDA engine uses its own inline block allocator (`self.seq_block_tables`, `self.free_blocks`), NOT `BlockManager`. Changes must target both paths. The `BlockManager` already has `allocate()`, `free()`, `can_allocate()`, CoW via `cow_if_needed()`, and swap-in/swap-out — the CoW mechanism could be relevant for draft token branch management.

Draft tokens need KV cache slots. The approach:

1. Before speculative forward, allocate K extra slots per speculating sequence.
2. After verification, free slots for rejected draft tokens. If the sequence accepted 2 of 5 draft tokens, free 3 slots.
3. Both the engine's inline allocator and `BlockManager` need a `free_partial(seq_id, num_slots_to_free_from_end)` operation.

### Phase 4: Draft Model Proposer (Future)

**File**: `crates/rvllm-speculative/src/draft.rs`

Replace the stub with actual model inference:
1. Load a smaller model (e.g., Qwen2.5-0.5B as draft for Qwen2.5-7B)
2. Run K autoregressive decode steps on the draft model
3. Return draft tokens + draft logprobs (needed for stochastic verification)

This requires a second model runner instance. It's a significant effort and should follow after the n-gram pipeline proves the end-to-end integration works.

## Testing Strategy

1. **N-gram proposer unit tests**: Test with known token histories. Pattern `[1, 2, 3, 4, 1, 2, 3]` with ngram_size=3 should propose `[4]` (the tokens after the first occurrence of `[1, 2, 3]`).
2. **Verification tests**: Already exist (6 tests). Extend for the n-gram case where draft_prob is approximated.
3. **End-to-end**: Run speculative decoding with n-gram proposer on a model. Verify output is identical to non-speculative decoding with greedy sampling.
4. **Acceptance rate**: Log the acceptance rate. For repetitive text (code, JSON), n-gram should achieve 50-70% acceptance. For novel text, expect 10-30%.
5. **Throughput**: Verify speculative decoding improves tokens/second vs non-speculative for repetitive workloads.

## Files Changed

| File | Change |
|------|--------|
| `crates/rvllm-speculative/src/ngram.rs` | New: N-gram proposer |
| `crates/rvllm-speculative/src/lib.rs` | Export ngram module |
| `crates/rvllm-speculative/src/engine.rs` | Wire n-gram proposer, fix target forward for K+1 tokens |
| `crates/rvllm-engine/src/gpu_engine.rs` | Option to wrap with SpeculativeEngine |
| `crates/rvllm-engine/src/async_gpu_engine.rs` | Speculative mode flag |
| `crates/rvllm-block-manager/src/manager.rs` | Add `free_partial()` for rejected draft slots |
| `crates/rvllm-worker/src/input.rs` | Build input for K+1 token sequences |
| `crates/rvllm-config/src/lib.rs` | Wire SpeculativeConfig into EngineConfig |

## Open Questions

- Should the n-gram proposer search across ALL requests' histories (cross-request n-gram cache) or only within each request? vLLM searches only within each request. Recommendation: per-request only, matching vLLM.
- For greedy decoding, should we skip probabilistic rejection sampling and just compare token IDs? vLLM does this. Recommendation: yes, greedy verification is simpler and correct.
- Should we implement tree-structured speculative decoding (multiple draft branches)? vLLM's EAGLE uses this. Recommendation: not in this phase, add later.
