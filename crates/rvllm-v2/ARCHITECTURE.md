# rvLLM Architecture Comparison: v1 vs v2 vs vLLM 0.19

## Core Differences

### 1. Scheduling Model

| | v1 | v2 | vLLM 0.19 |
|---|---|---|---|
| **Granularity** | SequenceGroup (beam search groups) | Individual requests | SequenceGroup |
| **State protocol** | Full metadata rebuild each step | StepDiff (added/removed/continued) | Full metadata rebuild |
| **Queues** | waiting/running/swapped | waiting/running/swapped | waiting/running/swapped |
| **Prefill** | Chunked, configurable | Chunked, configurable | Chunked prefill |
| **Preemption** | Swap or Recompute | Swap or Recompute | Swap or Recompute |

v2's diff protocol is the key architectural difference. Instead of rebuilding the full batch metadata each step, the scheduler emits a delta: which requests were added, removed, or continued. This reduces per-step CPU work and enables the stateful worker pattern.

### 2. Worker State Model

| | v1 | v2 | vLLM 0.19 |
|---|---|---|---|
| **Pattern** | Stateless executor adapter | Stateful HashMap\<RequestId, WorkerRequest\> | Stateless model runner |
| **State lives in** | Scheduler + tokenizer | Worker (persistent) | Scheduler |
| **Per-step update** | Full metadata passed | apply_diff() mutates state | Full metadata passed |

v2's worker owns a persistent map of per-request state (prompt tokens, output tokens, block tables, sampling params). The diff protocol lets the worker incrementally update this state instead of rebuilding it each step.

### 3. Forward Dispatch

| | v1 | v2 | vLLM 0.19 |
|---|---|---|---|
| **Paths** | 9+ (FusedDecode, PersistentDecode, Megakernel, BatchedV2, ...) | 1 (forward_batched_v2) | Multiple (decode/prefill) |
| **Selection** | FSM-based (auto-detect T=1 vs T>1, FP8, etc.) | Always BatchedV2 | Path selection per batch type |

v1 has specialized decode paths (fused GEMV, persistent cooperative kernels, megakernel interpreter) that are faster at T=1 but add FSM complexity. v2 uses one unified path for all batch sizes, matching the simplicity goal.

### 4. GEMM Strategy

| | v1 | v2 | vLLM 0.19 |
|---|---|---|---|
| **Default** | Hybrid (cuBLAS + CUTLASS) | Hybrid (cuBLAS + CUTLASS) | cuBLAS + custom kernels |
| **QKV** | cuBLAS hgemm (fused Q+K+V) | cuBLAS hgemm (fused Q+K+V) | Separate Q,K,V or fused |
| **GateUp+SiLU** | CUTLASS fused | CUTLASS fused | Custom fused |
| **O-proj** | cuBLAS or CUTLASS | cuBLAS or CUTLASS | cuBLAS |
| **Down-proj** | cuBLAS | cuBLAS | cuBLAS |

Both v1 and v2 use the same Hybrid strategy: cuBLAS for memory-bound GEMMs (QKV, O-proj, down-proj), CUTLASS for compute-bound fusions (GateUp+SiLU, O-proj+residual).

### 5. KV Cache

| | v1 | v2 | vLLM 0.19 |
|---|---|---|---|
| **Layout** | [num_blocks, block_size, num_heads, head_dim] | Same | Same |
| **Block copies (CoW)** | Standard copies | D2D via cuMemcpyDtoDAsync | Standard copies |
| **Swap in/out** | HtoD/DtoH async | HtoD/DtoH async | HtoD/DtoH async |

v2's key optimization: copy-on-write block copies go directly GPU-to-GPU via cuMemcpyDtoDAsync_v2, avoiding the GPU->CPU->GPU roundtrip.

### 6. Metadata Upload

| | v1 | v2 | vLLM 0.19 |
|---|---|---|---|
| **Strategy** | Single packed HtoD (6 fields in 1 memcpy) | Single packed HtoD (6 fields in 1 memcpy) | Multiple separate transfers |
| **Fields** | token_ids, positions, context_lens, block_tables, slot_mapping, seq_start_pos | Same | Similar |

Both v1 and v2 pack all metadata into one contiguous buffer and upload with a single memcpy_htod call, reducing PCIe transfer overhead.

### 7. Graph Capture

| | v1 | v2 | vLLM 0.19 |
|---|---|---|---|
| **Status** | Active (persistent/megakernel paths) | Infrastructure ready, disabled | Active |
| **Approach** | Selective per-path | Padded batch sizes (1,2,4,...,256) | Batch-level |

v2 has the graph capture infrastructure in place but it's not yet enabled (dead code). v1 uses it selectively for specialized decode paths.

## What v2 Gains Over v1

1. **Simpler codebase**: 1 forward path instead of 9+. No FSM dispatch logic.
2. **Diff protocol**: Scheduler emits deltas, not full state. Less CPU work per step.
3. **Stateful worker**: Persistent per-request state eliminates metadata rebuild overhead.
4. **D2D KV copies**: Direct GPU-to-GPU for copy-on-write blocks.
5. **Cleaner separation**: Engine -> Scheduler -> Worker -> Runner -> Layer is a clean pipeline.

## What v2 Loses vs v1

1. **Specialized decode paths**: v1's fused GEMV, persistent, and megakernel paths are faster at T=1 (single-token decode). v2's BatchedV2 path is general but not as optimized for this case.
2. **Graph capture**: v1 actively uses CUDA graphs for specialized paths. v2's graph capture is not yet enabled.
3. **FP8 support**: v1 has FP8 decode paths with cublasLt. v2 is f16 only.

## Layer-by-Layer Forward Flow (v2)

```
Embedding gather (custom kernel)
  |
  v
For each layer:
  1. Fused residual + RMSNorm (or plain RMSNorm for layer 0)
  2. QKV projection (cuBLAS hgemm)
  3. RoPE rotation (custom kernel)
  4. KV cache write (scatter kernel)
  5. Flash Attention 3 (GQA split-K decode / standard prefill)
  6. O-projection + residual + post-attention RMSNorm
  7. GateUp + SiLU (CUTLASS fused, or cuBLAS + SiLU kernel)
  8. Down projection (cuBLAS hgemm)
  |
  v
Final fused residual + RMSNorm
  |
  v
LM head GEMM (cuBLAS hgemm_f32_output: f16 hidden x f16 weight -> f32 logits)
  |
  v
DtoH logits transfer
```

## File Map

```
crates/rvllm-v2/src/
  types.rs          -- StepDiff, WorkerRequest, GpuBatchInput, ForwardOutput
  block_manager.rs  -- Dense slot allocator with prefix cache
  scheduler.rs      -- Diff-emitting scheduler (3 queues, chunked prefill)
  kv_cache.rs       -- CUDA KV cache with D2D block copies
  input.rs          -- InputBuilder: HashMap<RequestId,WorkerRequest> -> flat GPU arrays
  layer.rs          -- GpuTransformerLayer::forward_batched_v2() (single path)
  runner.rs         -- GpuModelRunner::forward() (layer loop, embed, LM head)
  worker.rs         -- Stateful Worker with apply_diff()
  engine.rs         -- Engine step loop (schedule -> worker.step -> process)
  integration.rs    -- init_engine(), model loading, serving loop
  bin/bench.rs      -- Standalone benchmark (direct engine, no HTTP)
```
