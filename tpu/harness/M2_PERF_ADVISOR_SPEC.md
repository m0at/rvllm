# MiniMax-M2.7-NVFP4 on TPU v6e-8 — perf optimization advisor swarm (16 agents)

## Current baseline (measured today)

- Slice: TPU v6e-8 (8 chips, 32 GB HBM each, 709 GB /dev/shm tmpfs host)
- Model: MiniMax-M2.7-NVFP4 from lukealonso (230B total / 10B active, 62 layers,
  256 experts top-8, NVFP4 packed uint8 + FP8 E4M3 scales + FP32 global scale,
  3072 hidden, 48 Q / 8 KV heads, HEAD_DIM 128, partial RoPE rotary_dim=64, ctx 196K).
- Load: 77s (parallel ThreadPool over 61 safetensors shards).
- **B=1 shard_map dispatch**: **726 ms/step = 1.38 tok/s** (5 iters, stdev 0.5 ms).
- **B=8 shard_map dispatch**: **871 ms/step = 9.18 tok/s** (6.7x amortization).
- B>=16 OOMs because KV cache is replicated (4.2 GB for B=16 / chip).
- dense-B1 variant (compute all 32 local experts + mask + psum): compiling now.
- Pallas NVFP4 matmul kernel (Agent 14 style): written, not yet tested.

Target: 100+ tok/s at B=1, 3000+ tok/s at B=128.

## Files to READ

- `/Users/andy/rvllm/tpu/harness/m2_synth_bench.py` (main forward, scan-based)
- `/Users/andy/rvllm/tpu/harness/m2_moe.py` (shard_map + all_to_all, the 1.4 tok/s path)
- `/Users/andy/rvllm/tpu/harness/m2_moe_dense.py` (dense-B1 variant, in-flight)
- `/Users/andy/rvllm/tpu/harness/nvfp4_jax_ops.py` (pure-JAX NVFP4 matmul, K-tiled)
- `/Users/andy/rvllm/tpu/harness/nvfp4_matmul_pallas.py` (Pallas kernel sketch, untested)
- `/Users/andy/rvllm/tpu/harness/m2_real_bench.py` (real-model loader)
- `/Users/andy/rvllm/tpu/harness/gemma4_tpu_infer.py` (reference: working v6e-4 at 44 tok/s B=1 dense)
- `/Users/andy/rvllm/tpu/harness/m2_checkpoint_schema/REAL_SCHEMA.md` (ground truth)

## 16 Agents — answer in <=15 lines each, no code edits

### Agent 1 — Batched-expert MXU call
Currently MoE runs per-expert matmuls via Python loop / vmap inside shard_map. Can
we express all 32 local experts as ONE batched MXU call `(EPS=32, B, H) x (EPS, MI, H)`?
Sketch the einsum + sharding. Expected speedup at B=1.

### Agent 2 — Skip unselected experts at B=1
Dense-B1 computes all 32 local experts and masks. But at B=1 only ~1 of 32 per shard
is selected (top-8 total / 8 shards). Could we use `jnp.where` on expert axis BEFORE
the matmul to skip, effectively doing 1 expert per shard? How does XLA treat this?

### Agent 3 — Load NVFP4 scales into VMEM
For a Pallas kernel, where should FP8 scales live? They're 1/16 the size of packed
weights (uint8 each 16 weight values). Can we pin them in VMEM for all layers?

### Agent 4 — Per-token attention cost vs per-token MoE cost at B=1
Decompose our 726 ms/step into attention vs MoE. At B=1 ctx=2048, attention is
~O(BHKD) per layer = tiny. MoE is ~O(E*MI*H) = huge. What fraction is MoE?
Recommend instrumenting with jax.block_until_ready + timers.

### Agent 5 — Reduce NL via MTP speculative decode
Config has `use_mtp: True, num_mtp_modules: 3`. If lookahead is 3 and acceptance rate
70%, effective ~2x speedup. But the checkpoint has no MTP weights (per REAL_SCHEMA).
Is this salvageable? Train MTP heads from base? Cost estimate.

### Agent 6 — int8 pre-dequant vs on-the-fly NVFP4
Path A keeps NVFP4 packed in HBM, dequants every matmul. Path B pre-dequants to int8
at load (~225 GB vs 130 GB NVFP4). int8 matmul on TPU v6e has native support. Would
a B=1 forward at int8 be faster despite 1.7x memory? Assume bandwidth-bound.

### Agent 7 — Per-chip full-expert replication + broadcast input
Alternative: replicate ALL 256 experts on each chip (at NVFP4: 16 GB). Every chip
computes the FULL MoE locally with no cross-chip comms. Fits in 32 GB HBM minus
14 GB attention weights minus activations. Would this actually work? Latency estimate.

### Agent 8 — Fused RMSNorm + matmul via Pallas
Current code has rmsnorm then matmul as two separate ops. Pallas could fuse: load
x, compute norm, compute matmul in one kernel. For our 62 layers × 4 matmuls / layer,
how much does this save?

### Agent 9 — Tile shape tuning for NVFP4 pallas
Our Pallas sketch uses BM=128, BN=256, BK=512. For MiniMax shapes (H=3072, MI=1536),
what's optimal tile? Does v6e MXU prefer different tile? Check SRAM budget.

### Agent 10 — async collective overlap
`lax.psum` in our dense MoE block blocks the forward until all shards agree. Can we
overlap psum with the next layer's compute? Reference: gemma4_tpu_infer uses
LIBTPU_INIT_ARGS flags for async collective fusion.

### Agent 11 — Compile cache per-shape reuse
JAX compiles fresh for each new batch size. Could we compile once for B=32 and then
use it for B=1 by padding? Memory cost, benefit at B=1.

### Agent 12 — Host-side prefill with large batch
Prefill (processing the prompt tokens) is different from decode. For 2048-token
generation, we'd prefill ~20 tokens then decode 2028. Can we batch the prefill to
B=16 or B=32 while decode stays at B=1? What's the prefill tok/s?

### Agent 13 — K/V cache int8 quant
Currently KV is bf16, 254 KB/token/layer × 62 = 16 MB/token. At ctx=196K × B=1 = 3 TB
impossible. int8 KV halves this. Gemma4 uses int8 KV successfully. Code change scope.

### Agent 14 — Blocksparse attention for long ctx
For the 196K context path, attention becomes O(N^2) on each decode step. Blocksparse
/ flash-style attention would help. Does MiniMax-M2 use any explicit sparsity pattern?
(Config: `sliding_window: null` — implies dense attention.)

### Agent 15 — v6e pipeline parallelism alternative
Instead of expert-parallel, split 62 layers across 8 chips as 7-8 layers/chip (pipeline
parallel). Each chip holds its layer's full weights (no expert sharding). Cross-chip
activation transfer is small (B*H bf16 = 6 KB). What's the latency?

### Agent 16 — Synthesis
Given all above, produce a prioritized 5-step plan. Which change gives the biggest
B=1 speedup? Which at B=128? Target: 100+ tok/s at B=1, 3000+ at B=128.

## Output format per agent

```
## Agent N — <title>

**Root claim:** <one sentence>

**Evidence:** <code reference or reasoning>

**Estimated speedup:** <x.Yx at B=N>

**Implementation cost:** <hours>

**Depends on:** <other agents or none>

**Confidence:** high/medium/low
```

No code edits. Advisory only. Coordinator (me) will synthesize + apply.
