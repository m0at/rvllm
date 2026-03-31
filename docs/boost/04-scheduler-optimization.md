# 04: Scheduler Optimization

Chunked prefill, prefill/decode interleaving, priority preemption, and continuous batching improvements.

---

## 1. Current Scheduler Architecture

### 1.1 Dual-Scheduler Problem

rvLLM has **two independent scheduler implementations** that coexist without integration:

**Scheduler A -- `rvllm-scheduler` crate** (`crates/rvllm-scheduler/src/scheduler.rs`):
A well-structured continuous batching scheduler with three queues (waiting, running, swapped), chunked prefill support, multiple scheduling policies (FCFS, Priority, SJF), and swap/recompute preemption modes. It integrates with the `rvllm-block-manager` crate for block allocation. However, it uses its own `SequenceGroup` struct (lines 22-79) that differs from `rvllm_sequence::SequenceGroup`, and the engine does not currently wire through to it.

**Scheduler B -- `FifoScheduler` in `gpu_engine.rs`** (lines 158-308):
The scheduler that actually runs in production. It is an inline FIFO scheduler embedded inside `GpuLLMEngine`. It has two queues (waiting, running) with no swapped queue, no chunked prefill, no priority scheduling, and no preemption. Its admission logic is purely: promote from waiting to running until `max_num_seqs` or `max_num_batched_tokens` is hit, gated by a block watermark check.

**This dual-scheduler situation is the single largest architectural debt in the scheduling subsystem.** The production path (`FifoScheduler`) lacks every feature the standalone `rvllm-scheduler` crate has.

### 1.2 Production Scheduler (FifoScheduler) Batch Formation

The `FifoScheduler::schedule()` method at `gpu_engine.rs:239-308` works as follows:

1. **Purge finished groups** from both `running` and `waiting` (line 241-242).
2. **Re-count tokens for running groups**: each running group contributes either its full prompt length (if `output_len == 0`, i.e., still in prefill) or 1 token (decode phase) per active sequence (lines 247-261).
3. **Admit from waiting**: while `running.len() < max_num_seqs` and `free_block_count >= watermark_blocks`, pop from waiting, check token budget and block availability, admit (lines 267-303).
4. Return all running groups plus total token count.

Key observations:
- **No separation between prefill and decode**: a new request's entire prompt is processed in a single forward pass. A 2048-token prompt blocks the entire batch for that iteration.
- **No token budget control**: a single prefill can consume the entire `max_num_batched_tokens` budget, starving all decode-phase sequences of their 1-token generation for that iteration.
- **No preemption**: if blocks run out, sequences are aborted (`FinishedAborted` at `gpu_engine.rs:1097`) rather than swapped or requeued.
- **No priority**: strictly FIFO.

### 1.3 Block Allocation in GpuLLMEngine

Block allocation happens in `build_metadata()` (`gpu_engine.rs:1041-1126`), not in the scheduler. It uses a simple bump allocator with a free list:

```
next_block_id (monotonic) + free_blocks (recycled from finished sequences)
```

There is no integration with `rvllm-block-manager`'s `BlockManager` (which supports reference counting, CoW for beam search, prefix caching, swap in/out). The `GpuLLMEngine` manages its own `HashMap<SequenceId, Vec<BlockId>>` of per-sequence block tables (`seq_block_tables` at line 347).

### 1.4 CUDA Graph Interaction

The `GraphRunner` (`crates/rvllm-worker/src/graph_runner.rs`) captures CUDA graphs **only for decode steps** (line 79: `if input.is_prefill { return false; }`). Supported batch sizes are defined in `GRAPH_BATCH_SIZES` at `crates/rvllm-gpu/src/cuda_graph.rs:21-24`:

```
1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
200, 208, 216, 224, 232, 240, 248, 256
```

Input is padded to the nearest cached batch size via `padded_batch_size()` (`cuda_graph.rs:29-31`). Padding uses dummy tokens with `slot_mapping = u32::MAX` so the cache-write kernel skips them.

### 1.5 Input Preparation

`crates/rvllm-worker/src/input.rs` already supports **mixed prefill+decode batches** via `prepare_input()` (line 15). When both prefill and decode groups exist, prefill tokens come first, then decode tokens, merged into a single `ModelInput`. The `query_lens` field distinguishes them (prefill sequences have `query_lens == context_lens`, decode sequences have `query_lens == 1`).

This means the **attention kernel already supports mixed batches**. The bottleneck is purely in the scheduler.

---

## 2. What vLLM's Scheduler Does

vLLM (v0.6+) implements an iteration-level scheduler that makes per-step decisions using a token budget. Key features:

### 2.1 Chunked Prefill
Long prompts are split into fixed-size chunks (default 512 tokens). Each iteration processes one chunk, interleaved with decode tokens from other requests. This bounds the latency impact of any single prefill on in-flight decode requests.

### 2.2 Prefix Caching (Automatic Prefix Caching / APC)
KV cache blocks are hashed by their token content. When two requests share a prefix, the second request reuses the existing KV blocks without recomputation. vLLM uses a radix-tree structure for efficient prefix matching.

### 2.3 Preemption Policies
When GPU memory is exhausted, vLLM can:
- **Swap**: move KV blocks to CPU RAM via cudaMemcpyAsync, resume later.
- **Recompute**: discard KV blocks and re-prefill from scratch.

vLLM selects the lowest-priority (or most recently arrived) request for preemption.

### 2.4 Continuous Batching with Token Budget
Each iteration has a token budget (e.g., 8192). The scheduler fills the budget with a mix of:
- Decode tokens (1 per running sequence)
- Prefill chunks (up to `max_prefill_chunk` per new/chunked request)

Running decode sequences always get priority over new prefill admissions, ensuring TTFT for in-flight requests is not degraded.

---

## 3. Chunked Prefill: Why It Matters

### 3.1 The Problem with Monolithic Prefill

Consider the current rvLLM behavior with `max_num_seqs=256` and a steady-state of 200 decode sequences. A new request arrives with a 4096-token prompt:

- **Without chunked prefill**: The scheduler admits the new request. The batch now has 200 decode tokens + 4096 prefill tokens = 4296 tokens. The forward pass takes ~8x longer than a normal decode step. All 200 decode sequences are delayed by ~7 decode iterations' worth of time, spiking their inter-token latency (ITL) from ~4ms to ~32ms.

- **With chunked prefill (chunk=512)**: The scheduler admits only 512 prefill tokens per iteration. The batch is 200 + 512 = 712 tokens. The forward pass takes ~3.5x a normal decode step. After 8 iterations, the prefill completes. The worst-case ITL spike is ~14ms instead of ~32ms.

### 3.2 GPU Utilization Argument

A100 80GB has 312 TFLOPS of FP16 tensor core throughput. A pure decode batch of 200 sequences produces 200 tokens -- the GEMMs are [200, hidden] x [hidden, hidden], achieving only ~8% of peak FLOPS. By interleaving 512 prefill tokens, the effective batch matrix becomes [712, hidden] x [hidden, hidden], which is 3.5x more compute-dense and achieves ~28% of peak FLOPS. The GPU is doing more useful work per iteration.

### 3.3 Optimal Chunk Size

The chunk size balances:
- **Too small (64-128)**: Many iterations to complete one prefill. CPU scheduling overhead dominates. CUDA graph invalidation at each batch size change.
- **Too large (2048+)**: Defeats the purpose -- decode latency spikes return.
- **Sweet spot (256-512)**: Empirically validated by vLLM and SGLang. 512 tokens is the default in vLLM v0.6.

---

## 4. Prefill/Decode Disaggregation

### 4.1 Concept

Rather than running prefill and decode in the same forward pass, split them into separate execution paths:

- **Prefill path**: Run exclusively when a new request needs prefilling. Uses FlashAttention-2 (dense attention, no paged KV cache lookup). Compute-bound on GEMMs.
- **Decode path**: Run for all in-flight decode sequences. Uses paged attention with block tables. Memory-bandwidth-bound on KV cache reads.

### 4.2 Why Disaggregation Helps

Prefill and decode have fundamentally different GPU resource profiles:

| Characteristic | Prefill | Decode |
|---|---|---|
| Tokens per seq | Hundreds-thousands | 1 |
| GEMM shape | Fat matrices [N_tokens, hidden] | Thin matrices [batch, hidden] |
| Attention | Dense, no paging needed | Paged, random block access |
| Bottleneck | Compute (tensor cores) | Memory bandwidth (HBM) |
| CUDA graphs | Not applicable (variable size) | Critical for launch overhead |

Mixing them in one forward pass forces the attention kernel to handle both dense-query and single-query patterns. The current `input.rs:merge_inputs()` function sets `is_prefill = true` for mixed batches (line 316), which means the attention kernel takes the prefill path for the entire batch, losing paged-attention optimizations for the decode sequences.

### 4.3 Implementation Approach

Two options:
1. **Split forward passes**: Run prefill and decode as separate `GpuModelRunner::forward()` calls within the same step. This doubles kernel launch overhead but allows optimal attention kernels for each.
2. **Fused kernel with ragged queries**: A single attention kernel that processes variable query lengths per sequence (FlashDecoding style). This is what vLLM's `flash_attn_with_kvcache` does via the `query_lens` metadata.

Option 2 is preferred. The `query_lens` field already exists in `AttentionMetadata` (see `bridge.rs` and `input.rs`). The attention kernel needs modification to use per-sequence query lengths rather than a global `is_prefill` flag.

---

## 5. Dynamic Batch Sizing

### 5.1 Current Static Limits

The current scheduler has two hard limits:
- `max_num_seqs = 256` (from `SchedulerConfigImpl::default()` at `crates/rvllm-config/src/scheduler.rs:57`)
- `max_num_batched_tokens = 2048` (same file, line 58, though the `SchedulerConfig` in `rvllm-scheduler` defaults to 8192)

These are static and do not adapt to workload characteristics.

### 5.2 Why Dynamic Sizing Matters

Consider two scenarios on A100 80GB with Qwen2.5-1.5B:
- **Short prompts (32 tokens), short outputs (32 tokens)**: Optimal batch size is 256+ sequences. KV cache per sequence is tiny (~2 blocks). GPU can handle 512+ concurrent sequences.
- **Long prompts (4096 tokens), long outputs (4096 tokens)**: Optimal batch size is 16-32 sequences. KV cache per sequence is huge (~512 blocks). Block exhaustion is the constraint.

A static `max_num_seqs=256` wastes GPU capacity in the first scenario and causes block exhaustion (and aborted sequences) in the second.

### 5.3 Algorithm: Token Budget as the Primary Constraint

Replace the hard sequence limit with a dynamic calculation:

```
available_blocks = free_gpu_blocks - watermark_reserve
estimated_blocks_per_seq = avg_output_length / block_size + prompt_blocks
max_concurrent = available_blocks / estimated_blocks_per_seq
effective_max_seqs = min(max_concurrent, hardware_max_seqs)
```

The scheduler should also track the running average of `output_length / input_length` ratio and adjust `estimated_blocks_per_seq` dynamically based on actual usage patterns.

---

## 6. Priority Scheduling

### 6.1 Current State

The `rvllm-scheduler` crate has a `Priority` policy (`policy.rs:46`) that sorts by `SequenceGroup::priority` (higher value first). But the production `FifoScheduler` in `gpu_engine.rs` has no priority support.

### 6.2 Requirements for Production Priority Scheduling

For a 10,000 QPS cloud inference service, priority scheduling needs:
- **Priority classes**: At minimum, `real-time` (SLA-bound), `interactive` (human-facing), `batch` (offline analytics).
- **Preemptive priority**: A real-time request arriving while batch sequences are running should preempt the lowest-priority batch sequences, swapping their KV blocks to CPU.
- **Starvation prevention**: Aging mechanism that gradually increases a request's effective priority as it waits, preventing indefinite starvation.
- **SLA-aware scheduling**: A request approaching its TTFT SLA deadline should be promoted.

### 6.3 Preemption for Priority

The `rvllm-scheduler` crate already has `preempt_if_needed()` (lines 323-355) that can swap or recompute. The integration path is:

1. Sort running sequences by priority (ascending) so lowest priority is at the end.
2. When a high-priority request arrives and cannot be admitted (blocks or sequence limit), pop the lowest-priority running sequence.
3. Swap its blocks to CPU (`BlockManager::swap_out`).
4. Admit the high-priority request.
5. When blocks free up, swap the preempted sequence back in (`BlockManager::swap_in`).

---

## 7. Prefix Caching

### 7.1 Current Implementation

rvLLM has prefix caching infrastructure in two places:

**`rvllm-block-manager/src/prefix_cache.rs`**: A hash-based prefix cache with LRU eviction. It hashes token prefixes at block boundaries, supporting lookup, insert, release, and eviction. This is integrated into `BlockManager::allocate()` (lines 204-265) and `BlockManager::can_allocate()` (lines 184-201).

**`gpu_engine.rs`**: The `GpuLLMEngine` has an `Option<PrefixCache>` field (line 338) and registers prefix blocks after prefill (lines 602-618 and 696-711). However, the registration uses synthetic block IDs (`(0..num_full_blocks).map(|i| BlockId(i as u32))`) rather than the actual allocated block IDs, making the caching ineffective.

### 7.2 What Needs to Change

The prefix cache needs to be integrated with the actual block allocation pipeline:

1. **Before prefill**: Call `prefix_cache.lookup()` to find cached prefix blocks. Use those block IDs directly in the block table for the new sequence.
2. **After prefill**: Register the actual physical block IDs from the sequence's block table.
3. **Block lifetime**: Cached blocks must not be freed when the owning sequence finishes. The `BlockManager` already handles this via reference counting -- cached blocks have an extra ref from the prefix cache.

The `BlockManager::allocate()` method already does this correctly (lines 221-257). The problem is that `GpuLLMEngine` does not use `BlockManager` at all -- it uses its own `seq_block_tables` with a bump allocator.

---

## 8. Impact on CUDA Graph Strategy

### 8.1 Current Graph Coverage

CUDA graphs are captured for decode steps only, at 35 pre-defined batch sizes from 1 to 256 (in steps of 8 after 16). Actual batch sizes are padded up to the nearest cached size.

### 8.2 New Requirements with Chunked Prefill

With chunked prefill, batches will contain a mix of prefill chunks and decode tokens. The total token count varies more dynamically:

```
decode_tokens = num_running_decode_seqs (1 per seq)
prefill_tokens = min(remaining_prompt, max_prefill_chunk) per chunked request
total_tokens = decode_tokens + prefill_tokens
```

This creates three categories of forward passes:

1. **Pure decode** (most common): Total tokens = num_decode_seqs. CUDA graphs apply. Current infrastructure handles this.
2. **Pure prefill** (rare, only when no decode seqs exist): Total tokens = prompt chunk. No graphs needed -- prefill is naturally compute-dense.
3. **Mixed prefill+decode** (common with chunked prefill): Total tokens = decode + prefill_chunk. CUDA graphs **cannot** be used because:
   - The attention metadata changes shape (variable query_lens per sequence).
   - The `is_prefill` flag changes the kernel path.
   - Graph capture requires fixed kernel arguments.

### 8.3 Strategies

**Option A: No graphs for mixed batches.** Accept the kernel launch overhead (~2ms per step for 28-layer model). This is simple and correct. The prefill chunk adds enough compute per iteration to amortize the launch overhead.

**Option B: Separate forward passes per step.** Run prefill tokens as one forward pass (no graph), then run decode tokens as a second forward pass (with graph). Two forward passes per step, but graphs cover the decode path. Requires double the per-step kernel launches but the decode graph eliminates overhead for the high-frequency path.

**Option C: Extended graph pool for mixed batches.** Capture graphs for common (total_tokens, num_decode_seqs, num_prefill_chunks) triples. This is combinatorially explosive and not practical.

**Recommendation: Option B.** It preserves CUDA graph benefits for decode while allowing flexible prefill chunking.

---

## 9. Iteration-Level Scheduling: Packing More Work

### 9.1 Core Principle

Every GPU iteration should process the maximum useful tokens within the token budget. "Useful" means: tokens that advance real requests toward completion.

Currently, the `FifoScheduler` can waste iterations:
- A prefill-heavy iteration delays all decode sequences.
- A decode-only iteration with few sequences underutilizes the GPU.

### 9.2 Two-Phase Scheduling Algorithm

Each `schedule()` call should operate in two phases:

**Phase 1: Reserve decode tokens.** All running decode sequences get their 1-token reservation first. This is non-negotiable -- they are already occupying KV cache blocks.

```
decode_budget = num_running_decode_seqs * 1
remaining_budget = max_num_batched_tokens - decode_budget
```

**Phase 2: Allocate prefill tokens from remaining budget.** Fill the remaining token budget with prefill chunks from waiting or partially-prefilled requests.

```
for each eligible prefill request (sorted by policy):
    chunk = min(remaining_prefill, max_prefill_chunk, remaining_budget)
    if chunk > 0 and can_allocate_blocks:
        schedule chunk
        remaining_budget -= chunk
```

This ensures decode sequences are never starved, while new requests make progress using spare GPU capacity.

---

## 10. Token Budget Allocation

### 10.1 The Budget Equation

For a 10,000 QPS service with Qwen2.5-1.5B on A100:

```
target_batch_latency = 5ms (200 iterations/sec)
max_tokens_per_iteration = 8192 (limited by KV cache and GEMM throughput)
peak_throughput = 8192 * 200 = 1,638,400 tokens/sec
```

With an average input of 256 tokens and output of 256 tokens:
```
tokens_per_request = 256 (prefill) + 256 (decode) = 512
effective_throughput = 1,638,400 / 512 = 3,200 requests/sec per GPU
```

To reach 10,000 QPS: 4 GPUs, with headroom for burst.

### 10.2 Budget Split Heuristic

The optimal prefill/decode split depends on the current state:

```
if num_waiting > 0 and prefill_frac < 0.3:
    # Too few prefills, admit more aggressively
    max_prefill_budget = 0.5 * max_num_batched_tokens
elif decode_sequences_near_SLA:
    # Prioritize decode
    max_prefill_budget = 0.1 * max_num_batched_tokens
else:
    # Default: 30% prefill, 70% decode
    max_prefill_budget = 0.3 * max_num_batched_tokens
```

### 10.3 Fairness: Prevent Prefill Starvation

Without bounds, a system under heavy decode load would never admit new requests. The scheduler should guarantee a minimum prefill budget per iteration (e.g., 256 tokens = half a chunk) unless there are genuinely no waiting requests.

---

## 11. Expected Throughput Improvement

Based on the optimization roadmap's benchmarks (`docs/optimization-roadmap.md`), rvLLM currently achieves:

| N (concurrency) | Current tok/s | vLLM tok/s | Gap |
|---|---|---|---|
| 64 | 4,063 | 3,828 | +6% |
| 128 | 6,360 | 6,400 | -1% |
| 256 | 8,316 | 9,437 | -12% |
| 512 | 8,528 | 10,771 | -21% |
| 1024 | 8,578 | 12,740 | -33% |

The vLLM gap at high concurrency (N >= 256) is primarily due to better scheduling and kernel efficiency. Here are projected improvements from scheduler optimizations alone:

### 11.1 Chunked Prefill (5-15% at N >= 128)

By interleaving prefill chunks with decode, the GPU processes more tokens per iteration. At N=256 with 50% of requests in prefill at any time, interleaving adds ~256 useful decode tokens to iterations that would otherwise be blocked by large prefills. **Estimated: +8% at N=256, +12% at N=512.**

### 11.2 Iteration-Level Token Budget (+5-10%)

Better packing of the token budget means fewer wasted iterations. Currently, prefill-heavy iterations waste capacity, and decode-light iterations underutilize the GPU. **Estimated: +5% average across all N.**

### 11.3 Prefix Caching (+10-30% for workloads with shared prefixes)

System prompts, few-shot examples, and chat history create shared prefixes. For a typical chatbot workload where 80% of requests share a 500-token system prompt: skipping 500-token prefills saves 500 * 0.8 = 400 tokens of compute per request. **Estimated: +15% for chat workloads, +0% for diverse prompts.**

### 11.4 Priority Preemption (+3-5% for mixed-priority workloads)

Prevents low-priority batch requests from blocking high-priority interactive requests. Does not directly improve peak throughput but improves effective throughput by reducing wasted capacity on requests that will be preempted anyway.

### 11.5 Combined Estimate

| N | Current | Projected with Scheduler Opt | Improvement |
|---|---|---|---|
| 64 | 4,063 | 4,470 | +10% |
| 128 | 6,360 | 7,300 | +15% |
| 256 | 8,316 | 9,980 | +20% |
| 512 | 8,528 | 10,660 | +25% |
| 1024 | 8,578 | 11,180 | +30% |

This would close most of the gap with vLLM at N=256 and narrow it significantly at N=512+. The remaining gap would be pure kernel efficiency (attention, GEMM, fused ops).

---

## 12. Implementation Plan

### Phase 1: Unify the Scheduler (2-3 days)

**Goal**: Replace `FifoScheduler` in `gpu_engine.rs` with the `rvllm-scheduler` crate's `Scheduler`.

**Changes**:
1. **`crates/rvllm-scheduler/src/scheduler.rs`**: Adapt `SequenceGroup` to use `rvllm_sequence::SequenceGroup` directly. Add methods `update_seq_token()`, `finish_seq()`, `live_seq_ids()`, `set_block_budget()` to match the interface `GpuLLMEngine` currently expects from `FifoScheduler`.
2. **`crates/rvllm-engine/src/gpu_engine.rs`**: Replace `FifoScheduler` with `rvllm_scheduler::Scheduler`. Wire `BlockManager` from `rvllm-block-manager` instead of the inline bump allocator. Remove `seq_block_tables`, `next_block_id`, `free_blocks` fields.
3. **`crates/rvllm-block-manager/src/manager.rs`**: Add `set_watermark()` and `usable_gpu_blocks()` as public methods (already exist but verify API compatibility).

### Phase 2: Enable Chunked Prefill (1-2 days)

**Goal**: Split long prompts into chunks that interleave with decode.

**Changes**:
1. **`crates/rvllm-scheduler/src/scheduler.rs`**: The `tokens_for_group()` method (lines 292-305) already implements chunked prefill logic. Wire the `max_prefill_chunk` config from `SchedulerConfig`.
2. **`crates/rvllm-config/src/scheduler.rs`**: Add `max_prefill_chunk: usize` field to `SchedulerConfigImpl`. Default to 512.
3. **`crates/rvllm-engine/src/gpu_engine.rs`**: In `build_metadata()`, handle partially-prefilled sequences. When `seq.num_computed_tokens > 0 && seq.num_computed_tokens < prompt_len`, include only the next chunk of tokens in `seq_data.prompt_token_ids`, and set `is_prompt = true` with the correct position offset.
4. **`crates/rvllm-worker/src/input.rs`**: Extend `prepare_prefill()` to handle partial prefills. The `position_ids` should start at `num_computed_tokens` rather than 0. The `slot_mapping` should cover only the new chunk's positions.

### Phase 3: Two-Phase Token Budget (1 day)

**Goal**: Guarantee decode tokens are always scheduled, prefill fills remaining budget.

**Changes**:
1. **`crates/rvllm-scheduler/src/scheduler.rs`**: Modify the `schedule()` method's Step 5 (lines 220-255) to process running decode groups first (1 token each, always admitted), then process prefill groups with the remaining budget.
2. Add a new field `prefill_budget_fraction: f32` to `SchedulerConfig` (default 0.3) to cap the fraction of tokens allocated to prefill.

### Phase 4: Priority Preemption (2 days)

**Goal**: Preempt low-priority sequences when high-priority requests arrive.

**Changes**:
1. **`crates/rvllm-scheduler/src/scheduler.rs`**: The `preempt_if_needed()` method already exists. Extend it to also trigger when a high-priority request in the waiting queue cannot be admitted due to sequence/token/block limits, and there exist lower-priority running sequences.
2. **`crates/rvllm-scheduler/src/policy.rs`**: Add aging logic: `effective_priority = base_priority + (now - arrival_time).as_secs() * aging_rate`.
3. **`crates/rvllm-core/`**: Add a `Priority` field to the request API so users can specify priority classes.

### Phase 5: Fix Prefix Caching Integration (1 day)

**Goal**: Make prefix caching actually reuse computed KV blocks.

**Changes**:
1. **`crates/rvllm-engine/src/gpu_engine.rs`**: After `build_metadata()`, pass the actual allocated block IDs (from `seq_block_tables`) to `prefix_cache.register_prefix_blocks()` instead of synthetic sequential IDs.
2. **`crates/rvllm-engine/src/gpu_engine.rs`**: Before block allocation in `build_metadata()`, check the prefix cache for hits and pre-populate the sequence's block table with cached block IDs.
3. **Wire through to `BlockManager`**: Once Phase 1 is complete, prefix caching comes for free since `BlockManager::allocate()` already does prefix cache lookup.

### Phase 6: CUDA Graph Strategy for Mixed Batches (1-2 days)

**Goal**: Maintain graph benefits for decode while supporting mixed prefill+decode iterations.

**Changes**:
1. **`crates/rvllm-worker/src/gpu_worker.rs`**: In the `execute()` path, split mixed batches into two forward passes: one prefill (no graph), one decode (with graph). The `SchedulerOutputs` already separates `num_prefill_groups` from decode groups.
2. **`crates/rvllm-worker/src/graph_runner.rs`**: No changes needed -- `can_use_graph()` already rejects prefill inputs.
3. **`crates/rvllm-worker/src/input.rs`**: Add a `split_mixed_batch()` function that separates a `[SequenceGroupMetadata]` into prefill-only and decode-only slices, returning two `ModelInput` structs.

---

## 13. Interaction with Paged KV Cache Block Allocation

### 13.1 Current Block Lifecycle

```
request arrives
  -> scheduler admits (FifoScheduler::schedule)
  -> build_metadata allocates blocks (bump allocator + free list)
  -> worker executes, writes KV to blocks
  -> sequence generates tokens, may need more blocks
  -> sequence finishes, blocks recycled to free list
```

### 13.2 Required Block Lifecycle with New Scheduler

```
request arrives
  -> scheduler checks BlockManager::can_allocate()
  -> if blocks available: admit, BlockManager::allocate()
  -> if blocks unavailable: check for preemption candidates
     -> if lower-priority running: BlockManager::swap_out(), preempt, allocate for new request
     -> if prefix cache has evictable blocks: BlockManager::evict_prefix_block(), then allocate
     -> if nothing evictable: keep in waiting queue
  -> worker executes
  -> after prefill: BlockManager::register_prefix() to cache prefix blocks
  -> sequence finishes: BlockManager::free() (respects ref counts for shared prefix blocks)
```

### 13.3 Block Watermark Tuning

The watermark prevents block exhaustion during decode. Currently set to `num_gpu_blocks / 25` (~4%) in `prepare_step()` at `gpu_engine.rs:828`. With chunked prefill, the watermark needs adjustment:

```
watermark = max(
    num_gpu_blocks * 0.04,                    // minimum 4% reserve
    num_running_decode_seqs * avg_decode_remaining / block_size  // projected decode block needs
)
```

This ensures running decode sequences have room to grow their KV caches without triggering preemption.

### 13.4 Block Fragmentation

The current bump allocator produces monotonically increasing block IDs, which leads to fragmentation when sequences of different lengths finish in different orders. The `BlockManager` from `rvllm-block-manager` uses a free-list (`MemoryPool` trait) that recycles in FIFO order, which has the same fragmentation issue.

For high-concurrency production use, consider a buddy allocator or slab allocator for blocks, grouping blocks by sequence length class to reduce fragmentation. This is a longer-term optimization (Phase 7+).

---

## Summary of Critical Path

The single highest-impact change is **Phase 1: Unify the Scheduler**. Every subsequent optimization depends on having a proper scheduler with three queues, block manager integration, and preemption support. The `rvllm-scheduler` crate is 90% complete -- it needs API unification with `rvllm_sequence::SequenceGroup` and wiring into `GpuLLMEngine`.

The second highest-impact change is **Phase 2: Chunked Prefill**, which directly addresses the throughput gap at N >= 256 by preventing large prefills from blocking decode sequences.

Together, Phases 1-3 should close approximately 60-70% of the vLLM throughput gap at high concurrency, bringing rvLLM to ~10,000-11,000 tok/s at N=1024 (vs vLLM's 12,740).

### Critical Files for Implementation
- `crates/rvllm-engine/src/gpu_engine.rs`
- `crates/rvllm-scheduler/src/scheduler.rs`
- `crates/rvllm-worker/src/input.rs`
- `crates/rvllm-block-manager/src/manager.rs`
- `crates/rvllm-worker/src/graph_runner.rs`
