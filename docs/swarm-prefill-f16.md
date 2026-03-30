# Swarm: Prefill F16 Attention + Pre-capture Graphs

## Goal
Remove f32 cast round-trip from prefill attention path. Currently prefill does:
1. Cast Q f16 -> f32 (allocates N*q_dim f32 buffer, launches cast kernel)
2. Run FA2 prefill kernel with f32 Q, f16 KV cache, f32 output
3. Cast output f32 -> f16 (allocates, launches cast kernel)

This is 2 extra allocations + 2 extra kernels + 2x memory traffic on Q and output.
Fix: write an f16-native prefill attention kernel and wire it in.

Measured impact: prefill takes ~130ms for 1280 tokens at N=256 (9.8k tok/s).
Target: 3-5x improvement -> 30-50k tok/s prefill.

## Context

- Model: Qwen2.5-1.5B (num_heads=12, num_kv_heads=2, head_dim=128, 28 layers)
- H100 SXM 80GB, CUDA 12.x, sm_90
- All data path is f16. Only RoPE tables (f32), epsilon scalar (f32), and final logits (f32) are not f16.
- KV cache is f16 paged (block_size=16)
- Decode attention already has an f16io kernel (`flash_attention_3_decode_f16io_kernel` in `kernels/flash_attention_3.cu`)
- Prefill attention is the old FA2 kernel that takes f32 Q (`flash_attention_2_f16kv_kernel` in `kernels/flash_attention.cu`)

## Agent Tasks

### Agent 1: Write f16-native prefill attention kernel

**Worktree branch: `prefill-f16-kernel`**

File: `kernels/flash_attention_3_prefill.cu`

Write a new kernel: `flash_attention_3_prefill_f16io_kernel`

Signature (must match exactly for the Rust launcher):
```c
extern "C"
__global__ void flash_attention_3_prefill_f16io_kernel(
    __half* __restrict__ output,       // [total_tokens, num_heads, head_dim]
    const __half* __restrict__ q,      // [total_tokens, num_heads, head_dim]  (contiguous [all_Q])
    const __half* __restrict__ key_cache,   // paged: [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ value_cache, // paged: [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    const int* __restrict__ seq_start_pos,  // [num_seqs + 1] -- cumulative query token starts
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int num_tokens,
    int causal                         // 1 for causal masking
)
```

Key differences from decode kernel:
- Prefill has MULTIPLE query tokens per sequence (not just 1)
- `seq_start_pos[s]` gives the first query token index for sequence s
- Query tokens for seq s are at positions `seq_start_pos[s]` through `seq_start_pos[s+1]-1`
- For causal masking: query token at position `seq_start_pos[s] + qi` can attend to KV positions 0..context_lens[s]-num_query_tokens+qi (inclusive)
- GQA: num_heads=12, num_kv_heads=2, so head_group = num_heads/num_kv_heads = 6

Algorithm:
- Grid: (num_seqs, num_heads, 1) -- one block per (seq, head)
- Block: 256 threads
- Each block processes ALL query tokens for its sequence
- For each query token qi in the sequence:
  - Stream K/V tiles from paged cache (same as decode)
  - Compute QK^T dot products with causal mask
  - Online softmax
  - Accumulate PV
  - Write output[seq_start_pos[s] + qi, head, :] = result
- All computation in f32 internally, inputs/outputs f16
- Use half2 vectorized loads from KV cache (like the decode kernel)

Reference the existing decode kernel at `kernels/flash_attention_3.cu` for:
- half2 vectorized cache loading pattern
- Warp-parallel dot product reduction
- Shared memory layout (BC=64, single K/V buffer reuse)

Compile test: `nvcc -ptx -arch=sm_90 -O3 kernels/flash_attention_3_prefill.cu` must succeed.

### Agent 2: Wire f16 prefill attention into gpu_layer.rs

**Worktree branch: `prefill-f16-wire`**

File: `crates/rvllm-model-runner/src/gpu_layer.rs`

Replace the prefill attention block (lines ~474-502) that currently does:
```rust
if input.is_prefill {
    // cast f16->f32, run f32 prefill, cast f32->f16
    ...
}
```

With:
```rust
if input.is_prefill {
    // Try f16-native prefill kernel first
    if let Ok(f16_prefill) = self.loader.get_func("flash_attention_3_prefill", "flash_attention_3_prefill_f16io_kernel") {
        // Launch directly with f16 Q, get f16 output. No casts.
        // ... (same args as existing prefill but f16 types)
    } else {
        // Fallback: old f32 cast path (keep existing code)
    }
}
```

The new `prefill_attention_f16io` method should mirror `prefill_attention` but:
- Take `&CudaView<'_, f16>` for Q (not f32)
- Return `CudaSlice<f16>` (not f32)
- No cast kernels needed
- Same launch config: grid(num_seqs, num_heads), block(256)
- Same shared memory calculation as decode FA3: `(BC * head_dim + BC + 8) * 4`

Keep the old f32 path as fallback (in case kernel isn't loaded).

### Agent 3: Pre-capture CUDA graphs at startup

**Worktree branch: `precapture-graphs`**

File: `crates/rvllm-worker/src/gpu_worker.rs`

Currently, graphs are captured lazily on first encounter for each batch size. This causes 12-35ms stalls mid-generation.

Add a method `precapture_decode_graphs()` called from `init_cache()` (after model is loaded and scratch is allocated). It should:

1. Define bucket sizes: `[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]`
   (or up to `max_batch_size` from config)

2. For each bucket size:
   - Build dummy metadata: token_ids=[0]*N, positions=[0]*N, context_lens=[1]*N, block_tables=[[0]]*N, slot_mapping=[0]*N, query_lens=[1]*N
   - Upload dummy metadata
   - Run warmup forward (forward_gpu_only)
   - Sync stream
   - Re-upload dummy metadata
   - Capture graph (begin_capture_on, forward_gpu_only, end_capture_on)
   - Store in graph pool

3. Log total pre-capture time

This eliminates ALL mid-generation graph capture stalls. The dummy sequences use block 0 which is always allocated.

Also: update `gpu_forward_ex()` to skip the `forward_count > GRAPH_WARMUP_CALLS` check when graphs are pre-captured. If a graph exists for the padded batch, just replay it immediately.

## Merge Order

1. Merge Agent 1 (kernel) first -- it's a new file, no conflicts
2. Merge Agent 2 (wiring) second -- depends on Agent 1's kernel name
3. Merge Agent 3 (pre-capture) third -- independent, no conflicts

## Verification

After merge, on H100:
```bash
cd /root/rvllm
make kernels CUDA_ARCH=sm_90
cargo build --release --features cuda

# Coherence
curl localhost:8080/v1/completions -d '{"model":"/root/models/Qwen2.5-1.5B","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
# Expect: "Paris"

# Bench
RVLLM_PTX_DIR=/root/rvllm/kernels/sm_90 ./target/release/rvllm benchmark --model /root/models/Qwen2.5-1.5B --n "1,4,16,64,128,256,512"
```

Expected improvement: N=256 should go from 28.5k to 35-45k tok/s (prefill overhead reduced 3-5x).
