# Spec 2: Chunked Prefill

## Summary

Long prompts (thousands of tokens) block the entire GPU for a single prefill step, starving decode requests of forward passes and spiking latency. Chunked prefill splits long prompts into fixed-size chunks and interleaves them with decode steps, keeping latency bounded while maintaining throughput.

rvLLM's **standalone** scheduler crate (`rvllm-scheduler`) has chunked prefill scaffolding (`max_prefill_chunk`, `num_prompt_tokens_processed`, `token_chunk_size`). It correctly computes chunk sizes and advances progress across steps. However, the CUDA inference path (`GpuLLMEngine`) uses its own inline `FifoScheduler` that has **none** of this functionality. What's missing is both the engine-level scheduling and the downstream wiring: the worker/model runner doesn't use `token_chunk_size` to build partial prefill inputs, and attention doesn't handle mixed prefill+decode batches correctly when a request is mid-prefill.

### Architecture Note: Dual Scheduler Paths

The codebase has TWO independent schedulers (same pattern as prefix caching in Spec 01):

1. **`rvllm-scheduler::Scheduler`** — full-featured with chunked prefill, preemption, swap. Uses its own local `SequenceGroup` type with `num_prompt_tokens_processed`.
2. **`GpuLLMEngine::FifoScheduler`** — simple inline FIFO (~150 lines including struct + impl in `gpu_engine.rs`). No chunked prefill, no preemption. Uses `rvllm_sequence::SequenceGroup` which does NOT have `num_prompt_tokens_processed`.

The CUDA engine uses path (2). Implementation must either extend `FifoScheduler` with chunked prefill or wire the engine to use the standalone scheduler (which has API mismatches — see `crates/rvllm-engine/src/engine.rs` comment: "rvllm_scheduler uses a different SequenceGroup type").

## vLLM Reference Behavior

### Unified Token Budget

vLLM's v1 scheduler doesn't distinguish "prefill phase" from "decode phase." Each step has a token budget (`max_num_batched_tokens`). Every request contributes `num_new_tokens = num_tokens - num_computed_tokens`, clamped by the remaining budget. This naturally handles:
- Full prefills (new request, all tokens are new)
- Chunked prefills (long request, only chunk_size tokens fit in budget)
- Decode (1 new token per request)
- Mixed batches (some requests prefilling, others decoding, in the same step)

### Chunk Size Control

- `long_prefill_token_threshold`: max tokens per prefill chunk (default: ~4% of `max_model_len`)
- `max_num_partial_prefills`: limit on concurrent partially-prefilled requests (default: 1)
- If chunked prefill is disabled and a prefill exceeds budget, the scheduler stops (doesn't schedule it partially)

### Mixed Batch Attention

The attention backend splits the batch into decode tokens (query_len=1) and prefill tokens (query_len>1). Decode tokens come first. FlashAttention uses different kernel paths for each group. The key insight: `seq_lens` for a chunked prefill request equals `num_computed_tokens + chunk_size` (the full context seen so far), while `query_lens` equals only the new chunk size.

### Output Suppression

Partially-prefilled requests don't produce output tokens. vLLM tracks `is_prefill_chunk = (num_computed_tokens < num_tokens)` and discards sampled tokens for incomplete prefills.

### Key vLLM Files

- `vllm/v1/core/sched/scheduler.py` -- token budget logic, `_update_after_schedule()`
- `vllm/v1/worker/gpu_model_runner.py` -- position computation, `query_start_loc`, `seq_lens`
- `vllm/v1/attention/backends/utils.py` -- `split_decodes_and_prefills()`

## Current rvLLM State

### What Works

The scheduler in `rvllm-scheduler` correctly handles chunked prefill:

```rust
// SchedulerConfig
pub max_prefill_chunk: usize,      // analogous to long_prefill_token_threshold

// SequenceGroup
pub num_prompt_tokens_processed: usize,  // tracks chunk progress

// tokens_for_group() correctly computes chunk sizes:
fn tokens_for_group(&self, group: &SequenceGroup) -> usize {
    let remaining = group.remaining_prefill();
    if remaining > 0 {
        if self.config.max_prefill_chunk > 0 {
            remaining.min(self.config.max_prefill_chunk)
        } else {
            remaining
        }
    } else {
        group.num_active()
    }
}
```

`SchedulerOutputs` includes `token_chunk_size` per scheduled group. Tests exist: `chunked_prefill_splits_long_prompt`, `chunked_prefill_progresses_over_steps`.

### What's Missing

1. **`GpuLLMEngine`'s `FifoScheduler` has no chunked prefill**: No `max_prefill_chunk`, no `num_prompt_tokens_processed` tracking, no `token_chunk_size`. This is the prerequisite for all other work.
2. **Worker input preparation doesn't use `token_chunk_size`**: The worker always builds input for the full prompt or 1 decode token. It doesn't handle "tokens 512..1024 of a 4096-token prompt."
3. **Positions are wrong for chunks**: A second chunk should have positions starting at `num_computed_tokens`, not 0.
4. **Attention metadata for mid-prefill**: `seq_lens` should be `num_computed_tokens + chunk_size`, not just `chunk_size`. The KV cache already has the earlier chunks' data.
5. **Slot mapping for chunks**: New tokens in a chunk need slots in the KV cache starting at position `num_computed_tokens`, not position 0.
6. **Output suppression**: No mechanism to discard sampled tokens for incomplete prefills.
7. **Mixed batch attention kernel path**: The FlashAttention CUDA implementation uses `is_decode = (num_tokens == num_seqs)` to choose kernel paths. In a mixed batch with chunked prefill, this heuristic fails — some sequences have `query_len > 1` (prefill chunk) while others have `query_len = 1` (decode).
8. **`SequenceGroupMetadata` lacks chunk fields**: The struct in `crates/rvllm-sequence/src/metadata.rs` needs `num_computed_tokens` and `token_chunk_size` fields.

**Partial existing infrastructure**: `crates/rvllm-worker/src/input.rs` already has `prepare_prefill_refs()` / `prepare_decode_refs()` / `merge_inputs()` for splitting mixed batches by the `is_prompt` flag. Note: `merge_inputs()` currently puts **prefill tokens first, then decode tokens** — this is the opposite of vLLM's convention (decode first). The ordering may need to change depending on kernel requirements. This handles the basic split but not partially-prefilled sequences. Also, `seq_start_pos` is already computed in `crates/rvllm-model-runner/src/gpu_runner.rs` from `query_lens` and uploaded to GPU.

## Implementation Plan

### Phase 0: Engine Scheduler Integration

**Files**: `crates/rvllm-engine/src/gpu_engine.rs`

Before any downstream work, `GpuLLMEngine`'s `FifoScheduler` needs chunked prefill support. Either:
- **Option A**: Extend `FifoScheduler` inline — add `max_prefill_chunk`, track `num_prompt_tokens_processed` per sequence, compute `token_chunk_size` when scheduling.
- **Option B**: Wire the engine to use `rvllm-scheduler::Scheduler` — requires resolving the `SequenceGroup` type mismatch between `rvllm_sequence::SequenceGroup` and the scheduler's local type.

Recommendation: Option A is simpler and lower-risk. The standalone scheduler can serve as a reference implementation.

### Phase 1: Worker Input Preparation for Chunks

**Files**: `crates/rvllm-worker/src/input.rs`, `crates/rvllm-engine/src/gpu_engine.rs`

When `SequenceGroupMetadata` indicates a chunked prefill (`token_chunk_size < prompt_len`):

1. **Token IDs**: Use `prompt_tokens[num_computed..num_computed+chunk_size]` instead of the full prompt.
2. **Positions**: Generate `[num_computed, num_computed+1, ..., num_computed+chunk_size-1]`.
3. **Context length**: Set to `num_computed + chunk_size` (total context, including earlier chunks in KV cache).
4. **Slot mapping**: Compute slots for positions `num_computed..num_computed+chunk_size`.

The engine must pass `num_computed_tokens` and `token_chunk_size` through `SequenceGroupMetadata` to the worker.

### Phase 2: Attention Metadata for Partial Prefills

**Files**: `crates/rvllm-model-runner/src/bridge.rs`, `crates/rvllm-worker/src/input.rs`, `crates/rvllm-model-runner/src/gpu_runner.rs`

**Note on `AttentionMetadata`**: There are two `AttentionMetadata` structs. The one used by the worker/runner is in `crates/rvllm-model-runner/src/bridge.rs` (uses `Vec<u32>`, already has `query_lens`). The one in `crates/rvllm-attention/src/metadata.rs` uses `GpuBuffer<i32>` and is the GPU-side representation. Changes must flow through both.

The bridge `AttentionMetadata` already has `query_lens: Vec<u32>`. It needs proper population for chunked prefill:

```rust
// In bridge.rs AttentionMetadata (already exists, needs correct population):
pub struct AttentionMetadata {
    pub query_lens: Vec<u32>,      // tokens being computed THIS step (chunk_size for prefill, 1 for decode)
    pub seq_lens: Vec<u32>,        // total context length (num_computed + chunk_size)
    // ... existing fields
}
```

`seq_start_pos` is already computed in `crates/rvllm-model-runner/src/gpu_runner.rs` from `query_lens` and uploaded to GPU as part of the packed metadata buffer.

For a mixed batch with 3 decode requests and 1 chunked prefill (chunk_size=256, total context=768):
- `query_lens = [1, 1, 1, 256]`
- `seq_lens = [100, 50, 75, 768]`
- `seq_start_pos = [0, 1, 2, 3, 259]`

### Phase 3: Output Suppression for Incomplete Prefills

**Files**: `crates/rvllm-engine/src/gpu_engine.rs`, `crates/rvllm-engine/src/output.rs`

After the worker returns sampled tokens, the engine must check whether each request's prefill is complete:

```rust
// In step(), after sampling:
for output in &worker_outputs {
    let group = &scheduled_groups[&output.request_id];
    if group.is_prefilling() {
        // Don't emit this output to the client -- prefill is incomplete
        // Advance num_prompt_tokens_processed instead
        continue;
    }
    // Normal output handling
}
```

### Phase 4: Block Allocation for Chunks

**Files**: `crates/rvllm-engine/src/gpu_engine.rs`

**Note**: The CUDA engine uses its own inline block allocator (`self.seq_block_tables`, `self.free_blocks`, `self.next_block_id`), NOT `BlockManager` from `rvllm-block-manager`. Changes must target `GpuLLMEngine::build_metadata()`. The standalone `BlockManager` can be updated separately for the non-CUDA path.

Blocks must be allocated incrementally:
- On the first chunk, allocate blocks for `ceil(chunk_size / block_size)` blocks.
- On subsequent chunks, allocate additional blocks only if needed (new tokens cross a block boundary).
- The block table grows incrementally across chunks.

The engine tracks the block table per sequence and extends it as chunks are processed.

## Testing Strategy

1. **Basic chunked prefill**: Submit a 2048-token prompt with `max_prefill_chunk=512`. Verify it takes 4 steps to complete prefill, then decode starts.
2. **Position correctness**: Verify that chunk 2 gets positions [512, 513, ..., 1023], not [0, 1, ..., 511].
3. **Mixed batch**: While a long prompt is chunked-prefilling, submit short requests. Verify the short requests decode normally alongside the chunks.
4. **Output suppression**: Verify no output tokens are emitted during prefill chunks, only after the last chunk.
5. **Coherency**: Output from chunked prefill must be identical to non-chunked prefill for the same prompt.
6. **KV cache correctness**: After chunked prefill, the KV cache must contain the same data as a full prefill. Test by comparing attention outputs.

## Files Changed

| File | Change |
|------|--------|
| `crates/rvllm-engine/src/gpu_engine.rs` | Add chunked prefill to `FifoScheduler`, pass chunk metadata, suppress partial prefill outputs, incremental block allocation |
| `crates/rvllm-sequence/src/metadata.rs` | Add `num_computed_tokens` and `token_chunk_size` to `SequenceGroupMetadata` |
| `crates/rvllm-worker/src/input.rs` | Handle `num_computed_tokens` and `token_chunk_size` in input preparation |
| `crates/rvllm-model-runner/src/bridge.rs` | Ensure `AttentionMetadata` fields populated correctly for chunks |
| `crates/rvllm-model-runner/src/gpu_runner.rs` | Update `seq_start_pos` and metadata upload for mixed batches |
| `crates/rvllm-engine/src/output.rs` | Filter outputs for incomplete prefills |
| `crates/rvllm-sequence/src/sequence.rs` | Ensure `num_computed_tokens` is advanced per chunk |
| `crates/rvllm-attention/src/flash_attention_impl.rs` | Fix `is_decode` heuristic for mixed batches |

## Dependencies

- None (can be implemented independently of prefix caching)
- Pairs well with prefix caching: cached prefix blocks reduce the first chunk

## Open Questions

- Should rvLLM default to chunked prefill enabled or disabled? vLLM defaults to enabled in v1. Recommendation: disabled by default for now (`max_prefill_chunk = 0` means no chunking).
- Should we limit concurrent partial prefills (`max_num_partial_prefills`)? Recommendation: yes, default to 1, matching vLLM.
