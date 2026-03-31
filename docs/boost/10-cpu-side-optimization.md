# 10: CPU-Side Optimization

## The Core Problem: HTTP Gap vs Direct Engine Gap

The benchmark data makes the case clearly:

**HTTP (512 tok/req):**
- N=16: 0.88x vLLM
- N=32: 0.85x vLLM
- N=64: 0.77x vLLM
- N=128: 0.67x vLLM

**Direct engine (128 tok/req):**
- N=16: 0.96x vLLM
- N=32: 0.86x vLLM
- N=64: 0.94x vLLM
- N=128: 0.85x vLLM

The gap widens with N because at N=128, each GPU step produces 128 tokens. If CPU overhead adds 1ms per step, that 1ms is amortized over only 128 tokens, but at high throughput the absolute throughput loss becomes significant. At N=128 via HTTP: rvLLM achieves 8,161 tok/s vs 12,312 tok/s via direct engine. That is a **34% throughput loss** attributable to the HTTP/CPU path.

## Request Lifecycle Analysis

Full request path traced through the codebase:

### Stage 1: HTTP Receive and JSON Deserialization

Files: `crates/rvllm-api/src/routes/chat.rs`, `crates/rvllm-api/src/types/request.rs`

- axum receives TCP bytes, parses HTTP headers
- `Json<ChatCompletionRequest>` triggers serde_json deserialization of the full request body
- Request validation (`req.validate()`)
- Cost estimate: ~5-20us per request depending on message count. At N=128 concurrent requests arriving in a burst, this is serialized on the tokio worker thread pool.

### Stage 2: Chat Template Application

File: `crates/rvllm-api/src/routes/chat.rs`, line 69-79

- `ChatMessage` objects are constructed by cloning strings: `ChatMessage::new(&m.role, &m.content)` -- this allocates N new strings per message
- `state.tokenizer.read().await` acquires an `RwLock<Tokenizer>` -- this is a **critical contention point** since the tokenizer is behind a `tokio::sync::RwLock` (file: `crates/rvllm-api/src/server.rs`, line 77)
- `apply_chat_template` concatenates strings to form the prompt
- Cost estimate: 10-50us per request, but the RwLock serializes concurrent callers

### Stage 3: Tokenization

File: `crates/rvllm-tokenizer/src/tokenizer.rs`, line 151-157

- `self.inner.encode(text, false)` calls into the HuggingFace `tokenizers` Rust crate
- `encoding.get_ids().to_vec()` copies the token IDs into a new Vec
- The HuggingFace tokenizer is **not thread-safe** (it requires `&self` but internally may not be safe for concurrent use), which is why it is behind the RwLock
- Cost estimate: 50-200us per request for typical prompts. The BPE merge algorithm is O(n log n) in prompt length.
- At N=128 concurrent requests, tokenization is **serialized** behind the RwLock. 128 x 100us = 12.8ms of serial tokenization before any GPU work begins.

### Stage 4: Channel Send to Engine Background Task

File: `crates/rvllm-engine/src/async_gpu_engine.rs`, lines 131-153

- `mpsc::channel(64)` for output, `oneshot::channel()` for request ID
- `self.gen_tx.send(...)` enqueues the request
- `.await` on `id_rx` waits for the background loop to assign an ID
- Cost: ~1us per send, but the oneshot receive blocks until the background loop drains the channel

### Stage 5: Background Loop Drain

File: `crates/rvllm-engine/src/async_gpu_engine.rs`, lines 233-255

- `drain_commands_to_queue` and `drain_generate_requests_to_queue` use `try_recv()` in a loop
- Each request is pushed to a `std::sync::Mutex<Vec<PendingRequest>>` shared queue
- The Mutex is held briefly per push, but at 128 concurrent requests, this is 128 mutex lock/unlock cycles
- Cost: ~0.5us per request for the mutex, negligible

### Stage 6: GPU Thread Drain

File: `crates/rvllm-engine/src/gpu_engine.rs`, lines 636-655

- `drain_request_queue()` locks the shared Mutex, takes all pending requests
- For each request: `self.add_request(...)` which includes:
  - **Tokenization again** on the GPU thread: `self.tokenizer.encode(&prompt)?` (line 486) -- this is a second tokenization of the same prompt text
  - `Sequence::new(seq_id, prompt_token_ids.clone())` -- clones the token IDs
  - `SequenceGroup::new(...)` -- clones params, prompt string
  - `self.scheduler.add_seq_group(seq_group)` -- pushes to waiting queue

**CRITICAL FINDING**: Tokenization happens TWICE per request. Once in the API layer (Stage 2-3 for chat template) and once in the GPU engine (Stage 6). The second tokenization is entirely redundant because the `PendingRequest` carries the raw prompt string, not pre-tokenized IDs.

### Stage 7: Scheduling

File: `crates/rvllm-engine/src/gpu_engine.rs`, lines 239-307

- `FifoScheduler::schedule()` iterates waiting/running queues
- Clones SequenceGroup objects: `self.running.iter().cloned().collect()` (line 306) -- this clones all running SequenceGroups every step
- Cost: O(N) in batch size per step, with significant allocation from cloning

### Stage 8: Metadata Build

File: `crates/rvllm-engine/src/gpu_engine.rs`, lines 1041-1120

- `build_metadata` iterates all groups and sequences
- For each sequence: `seq.prompt_token_ids.clone()`, `seq.output_token_ids.clone()`, `group.sampling_params.clone()`, `existing.clone()` (block tables)
- Many HashMap lookups and insertions
- Cost: proportional to total sequences and their token counts

### Stage 9: GPU Forward

Worker, separate thread.

- This is the actual GPU compute -- ~4ms at N=128
- Already well-optimized with CUDA graph replay

### Stage 10: Output Processing

File: `crates/rvllm-engine/src/gpu_engine.rs`, lines 874-989

- `process_worker_outputs` iterates all groups and sequences
- Per-token detokenization: `self.tokenizer.decode(&[*token_id])` (line 915) -- only when stop_strings are configured
- Deferred text reconstruction: `self.tokenizer.decode(&state.token_ids)` when all sequences finish (lines 957-964) -- batch decode of all accumulated tokens
- `OutputProcessor::build_request_output` clones prompt string, prompt_token_ids, and all sequence state text (line 129 of output.rs: `prompt.to_string()`, `prompt_token_ids.to_vec()`)
- Cost: O(N) with significant allocation

### Stage 11: Output Fan-Out

File: `crates/rvllm-engine/src/async_gpu_engine.rs`, lines 465-495

- `send_outputs` iterates all outputs, sends each through `mpsc::Sender`
- `tx.send(output).await` -- this is an async send that may context-switch

### Stage 12: SSE Serialization

Files: `crates/rvllm-api/src/routes/chat.rs`, lines 104-135; `crates/rvllm-api/src/types/streaming.rs`

- For each output: construct `ChatCompletionStreamChunk` (heap allocation for id, model, content strings)
- `serde_json::to_string(chunk)` -- JSON serialization per chunk
- `format!("data: {}\n\n", json)` -- string formatting per chunk
- `SystemTime::now()` called per chunk construction (syscall)
- `uuid::Uuid::new_v4()` called per request (RNG + formatting)
- Cost: ~5-20us per chunk, but at 128 concurrent streams each producing tokens every ~4ms, this is significant aggregate load

## Specific Bottleneck Quantification

### 1. Double Tokenization (Critical, ~10-20% of CPU time at high N)

In the GPU engine path (`GpuLLMEngine`), `add_request` at line 486 calls `self.tokenizer.encode(&prompt)`. But the API layer already called `apply_chat_template` which implicitly encodes the messages. The `PendingRequest` struct (line 324-328) carries only the raw `prompt: String`, not pre-tokenized IDs. This forces re-tokenization on the GPU thread, which is the critical path.

**Fix:** Add `prompt_token_ids: Option<Vec<TokenId>>` to `PendingRequest`, tokenize in the API layer, pass pre-tokenized IDs to the engine.

### 2. RwLock Contention on Tokenizer (Critical at burst arrivals)

`crates/rvllm-api/src/server.rs` line 77: `tokenizer: Arc<RwLock<Tokenizer>>`. The `RwLock` is `tokio::sync::RwLock`, which means acquiring a read lock requires an async context switch even when uncontested. At N=128 concurrent requests arriving simultaneously, all 128 will attempt to acquire the read lock to call `apply_chat_template` and perform tokenization. While `tokio::sync::RwLock` allows concurrent readers, the tokenizer's `encode` method takes `&self` -- but the actual HuggingFace tokenizer internally uses shared mutable state for the encoding buffer.

**Fix:** The HuggingFace `tokenizers` crate's `Tokenizer::encode` takes `&self` and is thread-safe (it uses internal immutable state for the model). Replace `RwLock<Tokenizer>` with just `Arc<Tokenizer>` for read-only operations. Only `decode_incremental` requires `&mut self`.

### 3. Excessive Cloning in Scheduler (Medium, ~2-5% of step time)

`FifoScheduler::schedule()` at line 306: `self.running.iter().cloned().collect()` clones every running SequenceGroup every step. Each SequenceGroup contains a Vec of Sequences, each containing `prompt_token_ids: Vec<TokenId>` and `output_token_ids: Vec<TokenId>`. At N=128 with 512-token prompts, this is cloning 128 * 512 * 4 bytes = 256KB of token IDs per step, plus all the sampling params and string data.

**Fix:** Use indices or references instead of cloning. The scheduler can return indices into its internal storage, and the engine can access groups by reference.

### 4. Per-Step Allocation in Output Processing (Medium, ~1-3%)

`OutputProcessor::build_request_output` (file `crates/rvllm-engine/src/output.rs`, lines 96-134) allocates:
- `prompt.to_string()` -- copies the entire prompt string
- `prompt_token_ids.to_vec()` -- copies all prompt token IDs
- `state.text.clone()` -- copies accumulated output text
- `state.token_ids.clone()` -- copies output token IDs

This happens for EVERY sequence in EVERY step, even for streaming where only the delta matters.

**Fix:** For streaming, send only the delta (new token and text). For non-streaming, only build the full output when finished.

### 5. No Global Allocator Override (Low-hanging fruit)

Grep confirms no `#[global_allocator]` or jemalloc/mimalloc anywhere in the codebase. The system allocator (glibc malloc on Linux) has known contention issues under multi-threaded workloads due to global mutex on the heap. jemalloc or mimalloc provide per-thread arenas that eliminate this.

**Fix:** Add `tikv-jemallocator` to `rvllm-server/Cargo.toml` and set `#[global_allocator]`.

### 6. Tokio Runtime Configuration (Low-medium)

`#[tokio::main]` at `crates/rvllm-server/src/main.rs` line 113 uses default tokio configuration, which creates worker_threads = num_cpus. On a multi-socket server or a machine with many cores, this can lead to:
- Work stealing overhead between workers
- Cache line bouncing
- Unnecessary thread context switches

**Fix:** Explicitly configure `#[tokio::main(flavor = "multi_thread", worker_threads = 4)]` since the actual async work is minimal (HTTP handling + channel management). The GPU thread is already a dedicated OS thread.

### 7. SSE Chunk Construction Overhead (Low-medium at high N)

Every streaming chunk in `crates/rvllm-api/src/types/streaming.rs` calls `SystemTime::now()` (a syscall: `clock_gettime`), constructs multiple heap-allocated Strings (`id.to_string()`, `model.to_string()`), and runs `serde_json::to_string` (which allocates a buffer, serializes, returns a String).

At N=128 streams, each producing a token every ~4ms step, that is 128 JSON serializations per step = ~32,000/second.

**Fix:** Pre-compute the `id` and `model` strings once per request. Use a reusable buffer for JSON serialization. Cache the timestamp (it does not need nanosecond precision for SSE).

### 8. IncrementalDecoder Re-decoding (Medium at long sequences)

`crates/rvllm-tokenizer/src/incremental.rs` line 28: `tokenizer.decode(&self.buffer, true)` decodes ALL accumulated tokens every time a new token is added. This is O(n^2) in sequence length for the incremental decoder. At 512 tokens output length, the last decode call processes all 512 tokens.

The GPU engine avoids this by deferring text reconstruction (line 914: only decoding when stop_strings are present), but the incremental decoder is available and would be used in streaming mode if wired differently.

### 9. HashMap Allocations in build_metadata (Low-medium)

`build_metadata` creates a `HashMap<SequenceId, SequenceData>` and `HashMap<SequenceId, Vec<BlockId>>` per group per step. At N=128, that is 128 HashMap allocations per step. Each HashMap allocation involves a heap allocation for the bucket array.

**Fix:** Pre-allocate and reuse these HashMaps. Or switch to a flat `Vec<(SequenceId, SequenceData)>` since the number of sequences per group is typically 1 (best_of=1).

### 10. async_trait Dynamic Dispatch (Low)

`crates/rvllm-api/src/server.rs` line 32-42: `InferenceEngine` trait uses `#[async_trait]` which boxes the returned Future on every call to `generate()`. This is one heap allocation per request.

**Fix:** Use RPITIT (return position impl trait in trait) available in Rust 1.75+ to avoid boxing.

## Estimated Impact Budget

Assuming a ~4ms GPU step at N=128 producing 128 tokens:

| Bottleneck | Current Cost (est.) | Fix | Savings |
|---|---|---|---|
| Double tokenization | 100-200us/request burst | Pre-tokenize once | 100-200us total |
| Tokenizer RwLock contention | 50-500us wait under burst | Arc\<Tokenizer\> | 50-500us |
| Scheduler clone | 200-500us/step | Index-based scheduling | 200-500us/step |
| Output clone per step | 100-300us/step | Delta-only streaming | 100-300us/step |
| System allocator contention | 50-200us/step | jemalloc | 50-200us/step |
| SSE serialization | 50-150us/step | Buffer reuse | 50-150us/step |
| Tokio overhead | 20-100us/step | Tune worker count | 20-100us/step |
| **Total per-step overhead** | **~600-2000us/step** | | **~400-1500us saved** |

On a 4ms GPU step, saving 0.5-1.5ms of CPU overhead would reduce step time by 12-37%. This translates directly to throughput improvement.

- **Current HTTP throughput at N=128:** 8,161 tok/s
- **Current direct engine at N=128:** 12,312 tok/s
- **Implied CPU overhead:** (12312 - 8161) / 12312 = 33.7% of throughput lost to CPU
- **Target after optimization:** If we halve CPU overhead, HTTP should reach ~10,200 tok/s (0.83x direct engine, or 0.70x vLLM vs current 0.67x)
- **Optimistic target:** If we eliminate 75% of CPU overhead, HTTP reaches ~11,200 tok/s (0.77x vLLM)

## Implementation Priority (Ranked by throughput-per-engineering-hour)

### P0 -- Immediate (hours each, large impact)

1. **Eliminate double tokenization**: Add `prompt_token_ids: Option<Vec<TokenId>>` to `PendingRequest`, tokenize in the API layer, skip re-tokenization in GPU engine. Touches 3 files.

2. **Replace RwLock\<Tokenizer\> with Arc\<Tokenizer\>**: The encode/decode methods on HuggingFace tokenizers are thread-safe. Split the incremental decoder out. Touches 2 files.

3. **Add jemalloc**: Add `tikv-jemallocator` dependency, 3 lines in `main.rs`. Measured to give 5-15% improvement on allocation-heavy Rust servers.

### P1 -- Short term (day each, medium impact)

4. **Delta-only streaming output**: Instead of cloning the full `RequestOutput` every step, send only new token IDs and text deltas. Requires refactoring `OutputProcessor::build_request_output`.

5. **Eliminate scheduler cloning**: Replace `self.running.iter().cloned().collect()` with an index-based return. The engine accesses groups by index into the scheduler's internal storage.

6. **Configure tokio worker threads**: Set `worker_threads = 4` explicitly. The HTTP server does minimal CPU work; most computation is on the dedicated GPU thread.

### P2 -- Medium term (days, lower per-item impact but good aggregate)

7. **Pre-allocate metadata HashMaps**: Reuse per-step allocations. Switch from HashMap to Vec for single-sequence groups.

8. **SSE buffer reuse**: Use `serde_json::to_writer` with a reusable buffer instead of `to_string`. Cache timestamp and ID strings.

9. **Busy-polling on GPU thread**: Replace `std::sync::mpsc::Receiver::recv()` (which uses a condvar/futex) with a spin-loop when the GPU has active work. Eliminates ~5-10us wake-up latency between step completion and result delivery.

10. **NUMA-aware pinning**: On multi-socket servers, pin the GPU thread to the NUMA node closest to the GPU's PCIe root complex. Use `libnuma` or `core_affinity` crate.

### P3 -- Long term (week+, advanced)

11. **io_uring for networking** (Linux only): Replace epoll-based tokio networking with io_uring via `tokio-uring` or `monoio`. Reduces syscall count by 50-70% for HTTP serving.

12. **Zero-copy token ID passing**: Use a shared arena allocator for token IDs. Instead of `Vec<TokenId>`, use a slice into a pre-allocated buffer. Eliminates per-request heap allocation for token IDs.

13. **Batch tokenization**: When multiple requests arrive simultaneously, use `tokenizer.encode_batch()` (already implemented at line 160 of tokenizer.rs) to tokenize them all at once. The HuggingFace tokenizer uses Rayon internally for batch encoding.

## Profiling Methodology

To measure each component's contribution:

1. **RVLLM_PROFILE=1**: Already implemented in `gpu_engine.rs` (lines 719-734). Reports drain/overlap/total times per step.

2. **Per-stage timing**: Add `Instant::now()` measurements around each stage in the HTTP handler. Log at info level behind a `RVLLM_HTTP_PROFILE` env var.

3. **Allocation profiling**: Use `dhat` (Rust heap profiler) or `jemalloc`'s `MALLOC_CONF=prof:true` to count allocations per step.

4. **Syscall counting**: Use `strace -c -p <pid>` on Linux to count syscalls during steady-state serving. Key metrics: `clock_gettime`, `futex`, `epoll_wait`, `read`, `write`.

5. **Lock contention**: Use `tokio-console` to visualize task wake-up latency and RwLock contention.

6. **perf stat**: Measure cache misses, context switches, and IPC during serving. `perf stat -e cache-misses,context-switches,instructions,cycles -p <pid>`.

7. **flamegraph**: Use `cargo-flamegraph` or `perf record` + `flamegraph` to identify which functions consume the most CPU time during HTTP serving. Compare against direct engine flamegraph to isolate HTTP-path overhead.

## Critical Files for Implementation

- `crates/rvllm-engine/src/async_gpu_engine.rs`
- `crates/rvllm-engine/src/gpu_engine.rs`
- `crates/rvllm-api/src/server.rs`
- `crates/rvllm-api/src/routes/chat.rs`
- `crates/rvllm-server/src/main.rs`
