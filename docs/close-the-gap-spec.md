# Close the 10% gap -- pure Rust implementation spec

> Superseded by the April 4, 2026 benchmark set and architecture fixes.
> Current public direct-engine comparison on H100 / Qwen2.5-7B / `output-len=128`:
> `N=1 127.9 vs 165.5`, `N=32 4407.5 vs 4467.7`, `N=64 7964.0 vs 7972.1`, `N=128 13148.3 vs 13903.5` (rvLLM vs vLLM 0.19.0).
> The biggest fixes since this spec were the batch-1 dispatch correction and the explicit batched `Hybrid` GEMM policy.

Three agents, all working in existing crates. Zero new dependencies.

## Agent 1: Metadata pre-packing (5% gap)

**Files**: `crates/rvllm-model-runner/src/gpu_runner.rs`, `crates/rvllm-worker/src/gpu_worker.rs`

**Problem**: nsys shows 435 HtoD memcpy calls totaling 1.47 seconds in a 4-second benchmark. We call `copy_from_host` separately for token_ids, positions, block_tables, context_lens, slot_mapping, and seq_lens every single step. Each is a separate cuMemcpyHtoDAsync with driver overhead.

**Fix**:
1. In `GpuModelRunner`, add a `PackedMetadata` struct that holds all per-step metadata in a single contiguous pinned host buffer:
   ```
   struct PackedMetadata {
       // Pinned host buffer, pre-allocated at max_batch_size
       host_buf: Vec<u8>,  // pinned via cuMemAllocHost
       // GPU buffer, pre-allocated at max_batch_size
       gpu_buf: CudaSlice<u8>,
       // Offsets into the buffer for each field
       token_ids_offset: usize,
       positions_offset: usize,
       block_tables_offset: usize,
       context_lens_offset: usize,
       slot_mapping_offset: usize,
       seq_lens_offset: usize,
       // Current packed size
       packed_size: usize,
   }
   ```
2. Add `PackedMetadata::new(max_batch_size, max_blocks_per_seq)` that allocates pinned host + GPU buffers sized for worst case.
3. Add `PackedMetadata::pack(&mut self, token_ids, positions, block_tables, context_lens, slot_mapping, seq_lens)` that writes all fields contiguously into `host_buf` and records offsets.
4. Add `PackedMetadata::upload(&self, stream)` that does ONE `cuMemcpyHtoDAsync` of `packed_size` bytes.
5. Add `PackedMetadata::gpu_token_ids(&self) -> DevicePtr` etc. accessors that return pointers into the GPU buffer at the recorded offsets.
6. Replace all the individual `copy_from_host` calls in `upload_metadata()` with a single `packed_meta.pack(...); packed_meta.upload(stream);`
7. For CUDA graph replay: the GPU buffer address is stable across replays, so the graph can reference these pointers directly.

**Key constraint**: The pinned host buffer must be allocated with `cuMemAllocHost` (not regular malloc) for async transfer. Use cudarc's `CudaContext::alloc_host` or raw `cuMemAllocHost` FFI.

**Verification**: Build on H100 with `cargo build --release --features cuda -p rvllm`. Run benchmark, check nsys shows 1 large HtoD instead of 435 small ones. Then `N=32 --output-len 256` must produce same tok/s or better.

---

## Agent 2: JIT attention decode kernel (3% gap)

**Files**: `crates/rvllm-fusion/src/ptx_emit.rs`, `crates/rvllm-fusion/src/ir.rs`, `crates/rvllm-fusion/src/matcher.rs`, `crates/rvllm-fusion/src/jit.rs`

**Problem**: Our `fa3_v3_decode_gqa_kernel` (hand-written CUDA) runs at 28.5 us per call. FlashInfer's equivalent runs at 22 us. That's 23% slower, costing ~3% of total throughput. 980 calls in the benchmark.

**Fix**: Generate an optimized decode attention kernel from our Rust PTX emitter, the same way we already generate fused norm+GEMV kernels. The key optimizations that FlashInfer uses:

1. **Vectorized KV cache loads**: Load 8 f16 values per thread using `ld.global.v4.b32` (128-bit). Our current kernel loads f16 pairs.

2. **Pipelined loads**: Use `cp.async` to prefetch the next KV cache page while computing attention on the current page. Double-buffer in shared memory.

3. **Better softmax reduction**: Use warp-level `shfl.sync` for the exp-sum reduction across the sequence dimension, avoiding shared memory round-trips for partial sums.

4. **GQA head grouping**: Process all Q heads that share a KV head in the same thread block, reusing the loaded K/V values. Our current kernel reloads K/V for each Q head group.

Implementation:
1. Add `AttentionDecodePattern` to `ir.rs` with fields: num_qo_heads, num_kv_heads, head_dim, page_size, max_seq_len.
2. Add `emit_attention_decode_ptx()` to `ptx_emit.rs` that generates a paged decode attention kernel with:
   - One thread block per (batch, kv_head_group)
   - Vectorized 128-bit KV loads from paged cache
   - cp.async double-buffered shared memory for KV pages
   - Warp shuffle softmax reduction
   - All Q heads sharing a KV head processed together (GQA reuse)
3. Add detection in `matcher.rs` for the attention decode pattern.
4. Wire into `jit.rs` so it's compiled at model load time alongside the existing fused kernels.
5. Add a dispatch path in `gpu_layer/decode.rs` that uses the JIT attention kernel instead of `fa3_v3_decode_gqa_kernel`.

**The PTX template**: The kernel structure is:
```
// Per thread block: handles one (batch_idx, kv_head_group)
// Threads: 128 (4 warps)
// Shared memory: 2 pages of KV cache (double buffer)

for each page in sequence:
    cp.async load next_page KV into smem[next_buf]
    compute QK^T for current_page from smem[cur_buf]  // dot products
    track running max and exp-sum for online softmax
    accumulate O += softmax_weight * V
    swap buffers
// Final: normalize O by total exp-sum, write output
```

**Key constraint**: The PTX emitter must handle the paged KV cache layout (block_table indirection). Generate the block_table lookup as PTX integer arithmetic, not as a separate kernel.

---

## Agent 3: Fuse remaining ops (2% gap)

**Files**: `crates/rvllm-fusion/src/ptx_emit.rs`, `crates/rvllm-fusion/src/matcher.rs`, `crates/rvllm-model-runner/src/gpu_layer/decode.rs`, `crates/rvllm-model-runner/src/gpu_layer/batched.rs`

**Problem**: nsys shows 4,788 `add_bias_f16_kernel` calls (2.6%) and 1,596 `deinterleave_qkv_f16_kernel` calls (1.0%). These are separate kernel launches after the GEMM that should be fused.

**Fix 1 -- Fuse bias-add into GEMV output**:
The current flow is: GEMV -> write to global memory -> launch add_bias_f16 -> read + add + write.
Fused flow: GEMV -> add bias in registers -> write to global memory.

Our existing fused GEMV kernels (`fused_cute_add_norm_qkv_bias_gemv` etc.) already do some bias fusion. Extend the PTX emitter patterns to always include bias-add when bias weights are present:
1. In `ptx_emit.rs`, modify `emit_gemv_kernel()` to accept an optional `bias_ptr` parameter.
2. After the GEMV accumulation and before the final store, add: `ld.global.f16 bias; cvt.f32.f16; add.f32 acc, acc, bias; cvt.f16.f32; st.global.f16`.
3. Update the matcher to detect GEMV+bias sequences and emit the fused version.

**Fix 2 -- Fuse QKV deinterleave into projection output**:
The current flow is: QKV GEMV -> write interleaved [Q0,K0,V0,Q1,K1,V1,...] -> launch deinterleave -> read + scatter to separate Q,K,V buffers.
Fused flow: QKV GEMV -> write directly to separate Q, K, V output pointers using computed offsets.

1. In `ptx_emit.rs`, add a `emit_gemv_deinterleaved_output()` variant that takes 3 output pointers (q_out, k_out, v_out) and the head dimensions.
2. After GEMV accumulation, instead of storing to a single output at `row_idx * out_features + col_idx`, compute which of Q/K/V the column belongs to and store to the correct output pointer.
3. Wire into the QKV projection path in `decode.rs`.

**Fix 3 -- Fuse reshape_and_cache into RoPE**:
The current flow is: RoPE kernel -> write rotated Q,K -> launch reshape_and_cache -> read K,V + write to paged cache.
Fused flow: RoPE kernel -> write rotated Q to output + write rotated K directly to paged cache slot.

1. Extend `fused_rope_cache_f16_kernel` (already partially fused) to also handle the V cache write, eliminating the separate `reshape_and_cache_f16io_kernel` call.
2. This is a modification to the existing hand-written CUDA kernel, not the PTX emitter.

**Verification**: After all 3 fusions, nsys should show zero `add_bias_f16_kernel` calls, zero `deinterleave_qkv_f16_kernel` calls, and fewer `reshape_and_cache_f16io_kernel` calls. Total kernel count per step should drop significantly.

---

## Shared context

- H100 instance is running at ssh6.vast.ai:12718
- Model is at /root/models/Qwen2.5-7B
- Binary builds with: `cargo build --release --features cuda -p rvllm`
- Benchmark: `./target/release/rvllm benchmark --model /root/models/Qwen2.5-7B --dtype half --n 32 --output-len 256`
- Current baseline: 4,385 tok/s at N=32 (0.92x vLLM 0.19.0)
- Target: >4,773 tok/s (match or beat vLLM 0.19.0)
- Do NOT add Co-Authored-By lines
- All GPU code behind `#[cfg(feature = "cuda")]`
- No fallbacks. If a kernel is missing, fail.
