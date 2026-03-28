# Spec 1: Automatic Prefix Caching (APC)

## Summary

When multiple requests share a prompt prefix (e.g., system prompt, few-shot examples), the KV cache blocks for that prefix can be reused instead of recomputed. This is called Automatic Prefix Caching (APC). vLLM implements this via chain hashing of block-aligned token sequences, an LRU eviction policy on freed blocks, and scheduler integration that skips already-computed tokens.

rvLLM has a `PrefixCache` struct and `BlockManager` integration, but the implementation has correctness issues and is not wired end-to-end. This spec brings it to parity with vLLM 0.18.0.

## vLLM Reference Behavior

### Chain Hashing

vLLM hashes each block independently using a chain: each block's hash depends on its parent block's hash plus the current block's token IDs. This ensures two sequences share a block only if their ENTIRE prefix up to that block boundary is identical, while keeping hash computation O(block_size) per block rather than O(prefix_length).

```
block_hash[0] = hash(SEED, tokens[0..block_size])
block_hash[1] = hash(block_hash[0], tokens[block_size..2*block_size])
block_hash[N] = hash(block_hash[N-1], tokens[N*block_size..(N+1)*block_size])
```

Only full blocks are hashed and cached. Partial blocks (the last block of a prompt that isn't full) are never cached.

### Block Lookup

On a new request, the scheduler computes block hashes incrementally, then walks the hash chain looking for cache hits. The walk stops at the first miss -- you can't skip a block in the middle because later blocks depend on earlier ones via the chain hash.

The hit count determines `num_computed_tokens`, which tells the model runner to skip those tokens during prefill.

### Eviction

Freed blocks go onto an LRU free list (doubly-linked list for O(1) operations). When a block is freed, it stays in the hash table -- it's only removed when its slot is reallocated for a new block. Cache hits call `touch()` to move the block out of the free list and increment its reference count.

### Key vLLM Files

- `vllm/v1/core/kv_cache_utils.py` -- `hash_block_tokens()`, `FreeKVCacheBlockQueue`, `KVCacheBlock`
- `vllm/v1/core/block_pool.py` -- `BlockPool`, `cache_full_blocks()`, `get_cached_block()`
- `vllm/v1/core/single_type_kv_cache_manager.py` -- `find_longest_cache_hit()`
- `vllm/v1/core/sched/scheduler.py` -- `get_computed_blocks()` call during scheduling

## Current rvLLM State

### What Exists

- `PrefixCache` in `crates/rvllm-block-manager/src/prefix_cache.rs` with `lookup()`, `insert()`, `release()`, `evict_one()`
- `BlockManager` integration: `allocate()` checks prefix cache, `register_prefix()` caches blocks after prefill. Note: `BlockManager` correctly passes actual block IDs from the sequence's block table in `register_prefix()`.
- Config flag: `CacheConfigImpl.enable_prefix_caching` in `crates/rvllm-config/src/cache.rs` (the struct is `CacheConfigImpl`, not `CacheConfig`)
- Engine creates `PrefixCache` if enabled, calls `count_hits()` during step

### Architecture Note: Dual Block Management Paths

**This is critical for implementers.** The codebase has TWO independent block management systems:

1. **`BlockManager`** in `rvllm-block-manager/src/manager.rs` — full-featured with prefix cache integration, CoW, swap-in/swap-out. Used by `rvllm-scheduler/src/scheduler.rs`.
2. **Inline allocator** in `GpuLLMEngine::build_metadata()` at `gpu_engine.rs:~802-895` — simple allocator using `self.seq_block_tables`, `self.free_blocks`, `self.next_block_id`. Used by the CUDA inference path.

The CUDA engine path (`GpuLLMEngine`) uses path (2) exclusively. It also has its own inline `FifoScheduler` (defined in `gpu_engine.rs`, ~150 lines including struct + impl) that does NOT use `rvllm-scheduler`. Changes to `BlockManager` or `rvllm-scheduler::Scheduler` have **no effect** on the CUDA inference path.

**All implementation work must target `GpuLLMEngine` and its inline allocator/scheduler.** Changes to `BlockManager` are useful for correctness but won't be exercised until the engine is refactored to use it.

### What's Broken

1. **Wrong hash algorithm**: `hash_prefix()` hashes `tokens[0..(block_idx+1)*block_size]` (entire prefix up to each block boundary). For block 0 it hashes `bs` tokens, for block 1 it hashes `2*bs`, etc. Computing all N block hashes costs `bs + 2*bs + ... + N*bs = O(N^2 * bs)`. Should chain-hash each block independently in O(block_size) per block.
2. **Cache hits are decoration**: `count_hits()` result at `gpu_engine.rs:621` is only logged via `debug!()` and never used to set `num_computed_tokens` or skip computation. The model runner always recomputes the full prefill.
3. **Wrong block IDs on registration (engine path only)**: `GpuLLMEngine` at `gpu_engine.rs:543-545` builds placeholder `BlockId(i as u32)` sequential IDs instead of using `self.seq_block_tables`. Note: `BlockManager.register_prefix()` correctly passes actual block IDs — this bug is only in the engine's inline path.
4. **O(n) eviction**: `evict_one()` at `prefix_cache.rs:~154` scans `self.cache.iter().filter(...).min_by_key(...)` — O(n) scan of the HashMap instead of O(1) LRU list.
5. **No scheduler integration**: Neither the `FifoScheduler` in `gpu_engine.rs` nor `rvllm-scheduler::Scheduler` has prefix cache awareness.

## Implementation Plan

### Phase 1: Fix the Hash Algorithm

**File**: `crates/rvllm-block-manager/src/prefix_cache.rs`

Replace the current `hash_prefix()` with chain hashing:

```rust
pub fn hash_block(parent_hash: Option<PrefixHash>, block_tokens: &[TokenId]) -> PrefixHash {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if let Some(ph) = parent_hash {
        ph.0.hash(&mut hasher);
    }
    block_tokens.hash(&mut hasher);
    PrefixHash(hasher.finish())
}
```

Add a method to compute all block hashes for a token sequence:

```rust
pub fn compute_block_hashes(&self, tokens: &[TokenId]) -> Vec<PrefixHash> {
    let mut hashes = Vec::new();
    let mut parent = None;
    for chunk in tokens.chunks_exact(self.block_size) {
        let h = Self::hash_block(parent, chunk);
        hashes.push(h);
        parent = Some(h);
    }
    hashes
}
```

Update `lookup()` to walk the hash chain and stop at the first miss, returning the count of consecutive cache hits.

### Phase 2: O(1) LRU Eviction

**File**: `crates/rvllm-block-manager/src/prefix_cache.rs`

Replace the `HashMap`-based eviction with a doubly-linked list. Use an intrusive list or a `VecDeque<BlockId>` with a position map for O(1) removal:

```rust
struct LruEntry {
    block_id: BlockId,
    hash: PrefixHash,
    prev: Option<usize>,
    next: Option<usize>,
}

struct LruList {
    entries: Vec<LruEntry>,      // arena-allocated nodes
    head: Option<usize>,         // LRU end (evict from here)
    tail: Option<usize>,         // MRU end (insert here)
    block_to_slot: HashMap<BlockId, usize>,
}
```

Operations:
- `push_back(block_id, hash)` -- add to MRU end (O(1))
- `pop_front()` -- evict from LRU end (O(1))
- `remove(block_id)` -- remove on cache hit / touch (O(1) via block_to_slot)

### Phase 3: Wire Cache Hits to Skip Computation

**Files**:
- `crates/rvllm-engine/src/gpu_engine.rs` -- modify `FifoScheduler` and `build_metadata()` for prefix cache awareness
- `crates/rvllm-sequence/src/sequence.rs` -- `num_computed_tokens` field already exists (initialized to 0)
- `crates/rvllm-sequence/src/metadata.rs` -- add `num_computed_tokens` to `SequenceGroupMetadata`
- `crates/rvllm-worker/src/input.rs` -- adjust input preparation for partial prefill

This is the critical integration:

1. **Engine scheduler**: In `GpuLLMEngine`'s `FifoScheduler`, before scheduling a new request, call `prefix_cache.lookup(block_hashes)` to get the number of cached blocks. Set `sequence.num_computed_tokens = cached_blocks * block_size`.

2. **SequenceGroupMetadata**: Add `num_computed_tokens: usize` to `SequenceGroupMetadata` in `crates/rvllm-sequence/src/metadata.rs`. This struct currently has `{ request_id, is_prompt, seq_data, sampling_params, block_tables }` — the new field is how the engine communicates cache hits to the worker.

3. **Engine block allocation**: In `GpuLLMEngine::build_metadata()`, the inline block allocator must "adopt" cached blocks: add them to `self.seq_block_tables` for the sequence, increment ref count, and only allocate new physical blocks for the remaining (uncached) tokens. Pass the actual block IDs from `self.seq_block_tables` (not placeholders) when registering prefix blocks.

4. **Worker input preparation**: When `num_computed_tokens > 0`, `prepare_prefill()` in `input.rs` must change:
   - **Tokens**: Send only `prompt_token_ids[num_computed_tokens..]` (skip cached tokens)
   - **Positions**: Start at `num_computed_tokens`, not 0 — i.e., `num_computed_tokens..seq_len`
   - **Slot mapping**: Only generate mappings for the new tokens (`num_computed_tokens` onward)
   - **Block table**: Include ALL blocks (cached + new) since the attention kernel needs the full KV cache for attending to the cached prefix
   - Currently `prepare_prefill()` always sends all `prompt_token_ids` with positions `0..seq_len`, so this is a significant change

5. **Standalone scheduler** (optional, lower priority): `rvllm-scheduler/src/scheduler.rs` could also gain prefix cache awareness. This is separate from the CUDA engine path and can be done independently.

### Concurrency Note

`GpuLLMEngine` is marked `unsafe impl Send`. If the engine is ever shared across threads (e.g., via the async API server), the `PrefixCache` needs locking. The `SharedBlockManager` wraps `BlockManager` in a `Mutex`, but `GpuLLMEngine` has no similar wrapper for `PrefixCache`. Consider wrapping in `Mutex` from the start.

### Interaction with Chunked Prefill

If prefix caching is combined with chunked prefill (Spec 02), `num_computed_tokens` from the prefix cache interacts with `num_prompt_tokens_processed` from the chunking system. The semantics: `num_computed_tokens` (from cache) sets the starting point, and chunked prefill processes the remaining tokens in chunks. Ensure these don't double-count.

### Phase 4: Correct Block Registration

**File**: `crates/rvllm-engine/src/gpu_engine.rs`

After a prefill completes, register the actual allocated block IDs:

```rust
// After worker returns, register newly-filled blocks in prefix cache
if let Some(ref mut pc) = self.prefix_cache {
    if let Some(block_table) = self.seq_block_tables.get(&seq_id) {
        let block_hashes = pc.compute_block_hashes(&prompt_tokens);
        for (hash, &block_id) in block_hashes.iter().zip(block_table.iter()) {
            pc.insert(*hash, block_id);
        }
    }
}
```

## Testing Strategy

1. **Hash correctness**: Two sequences with identical prefixes produce identical block hashes. Different prefixes produce different hashes. Chain property: changing token 0 changes ALL subsequent block hashes.
2. **Cache hit**: Submit request A, then request B with the same system prompt. Verify B's `num_computed_tokens` equals the shared prefix length (rounded down to block boundary).
3. **Eviction**: Fill the cache beyond `max_cached_blocks`, verify LRU blocks are evicted first, MRU blocks survive.
4. **Coherency**: Output from a cache-hit prefill must be identical to a full prefill. Test with diverse prompts.
5. **Ref counting**: Verify blocks in use by active sequences are never evicted.

## Files Changed

| File | Change |
|------|--------|
| `crates/rvllm-block-manager/src/prefix_cache.rs` | Rewrite hash algorithm, add LRU list, fix lookup/insert |
| `crates/rvllm-engine/src/gpu_engine.rs` | Fix block ID registration, wire cache hit count, add prefix cache awareness to `FifoScheduler` and `build_metadata()` |
| `crates/rvllm-sequence/src/metadata.rs` | Add `num_computed_tokens` to `SequenceGroupMetadata` |
| `crates/rvllm-worker/src/input.rs` | Adjust input prep for partial prefill (skip computed tokens, offset positions) |
| `crates/rvllm-block-manager/src/manager.rs` | Update allocate() to adopt cached blocks (for standalone scheduler path) |
| `crates/rvllm-scheduler/src/scheduler.rs` | Add prefix cache lookup (for standalone scheduler path, lower priority) |

## Open Questions

- Should we support extra hash keys (LoRA adapter name, multimodal inputs) from the start, or add them later? Recommendation: add an `extra_keys: Option<&[u8]>` parameter to `hash_block()` now but don't use it yet.
- The prefix cache is already per-engine (GpuLLMEngine has its own `prefix_cache` field). This is consistent with vLLM's per-scheduler design.
