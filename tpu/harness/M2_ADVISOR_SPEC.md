# MiniMax-M2.7-NVFP4 — 16-agent advisor swarm

Mission: produce the MiniMax-M2.7-NVFP4 throughput bench on TPU v6e-8.

## Concrete error to solve

When the JIT-compiled `forward_step` runs with `moe_block_nvfp4` using the
shard_map path:

```
FAILED: TypeError: mul got incompatible shapes for broadcasting:
  (8, 2, 3072), (8, 1, 2).
```

This happens inside `_moe_shard_map_nvfp4` in
`/Users/andy/rvllm/tpu/harness/m2_moe.py`.

Shape shorthand for context (B=1, TOP_K=8, H=3072, NUM_EXPERTS=256, n_shards=8,
experts_per_shard=32, capacity = ceil(B*TOP_K/NUM_EXPERTS)*2 = ceil(8/256)*2 = 2).
So capacity=2, n_shards=8. The shapes `(8, 2, 3072)` is `(n_shards, capacity, H)`
(the per-shard token buffer after all_to_all); `(8, 1, 2)` has n_shards=8 and
two other dims — likely a per-expert / top-k weight tensor with an incorrect
broadcast axis.

## Architecture summary

- Model: MiniMax-M2.7-NVFP4 (230B total / 10B active, MoE top-8).
- Shapes: hidden=3072, 48 Q heads, 8 KV heads (GQA), head_dim=128, rotary_dim=64,
  MOE_INTER=1536, 256 experts top-8, 62 layers, VOCAB=200064.
- Quant: NVFP4 modelopt two-level scale. Per tensor:
  - `.weight`: uint8 packed, (rows, cols/2), two FP4 values per byte (E2M1 signed)
  - `.weight_scale`: uint8 (FP8 E4M3), (rows, cols/16), per-16-element block scale
  - `.weight_scale_2`: FP32 scalar, per-tensor global scale
  - `.input_scale`: FP32 scalar (unused in path A)
- Path A (target): on-the-fly dequant inside the GEMM, no materialized bf16 weight.
- Sharding: 8-way expert parallel on `('expert',)` mesh axis. 32 experts/chip.

## Full list of repo files you may READ

- `/Users/andy/rvllm/tpu/harness/m2_moe.py` — contains `_moe_shard_map_nvfp4`,
  `moe_block_nvfp4`, `router_sigmoid_topk`. This is where the shape bug lives.
- `/Users/andy/rvllm/tpu/harness/m2_mesh.py` — mesh + `expert_all_to_all_*` helpers.
- `/Users/andy/rvllm/tpu/harness/nvfp4_jax_ops.py` — `nvfp4_to_bf16_jax`,
  `nvfp4_matmul` (uses jax.lax.dot_general).
- `/Users/andy/rvllm/tpu/harness/m2_synth_bench.py` — the synthetic bench that
  consumes `moe_block_nvfp4`.
- `/Users/andy/rvllm/tpu/harness/m2_tpu_infer.py` — the real-model integrator
  (also consumes `moe_block_nvfp4`).
- `/Users/andy/rvllm/tpu/harness/m2_checkpoint_schema/REAL_SCHEMA.md` — ground
  truth checkpoint layout.

Reference (already working on TPU v6e-4 in production):
- `/Users/andy/rvllm/tpu/harness/gemma4_tpu_infer.py` — 1826 lines, includes a
  functioning MoE FFN (gather-based) for Gemma 4 26B-A4B. Its `moe_ffn`
  function at ~line 258 works in production; you may use it as a style template.

## What each advisor agent does

Each agent reads a subset of the above files and answers ONE focused question.
Agents do NOT edit code; they produce a concrete proposal in <=15 lines that
the coordinator (me) will apply or combine.

Every agent must:
- State the root cause in one sentence.
- Give a precise code change (file + function + the exact replacement).
- Flag any dependency on other agents' answers.

## Agent assignments (16)

### Agent 1 — Pinpoint the shape bug
Read `_moe_shard_map_nvfp4` in `m2_moe.py` and trace every array shape from
function entry to exit for B=1, n_shards=8, capacity=2. Identify the exact
line where the `(8, 2, 3072) * (8, 1, 2)` mismatch happens. Hypothesis: a
routing-weight tensor (shape `(B, TOP_K)` or `(n_shards, capacity)`) ended up
with an extra singleton axis somewhere in the pack/all_to_all pipeline.
Report the offending line + the fix (what shape SHOULD it have, and how to
enforce that).

### Agent 2 — Router sort-and-pack correctness
Read the "sort tokens by destination shard" block in `_moe_shard_map_nvfp4`.
For B=1 with TOP_K=8 and n_shards=8, we have 8 dispatch slots total that must
bucket-sort by `dest_shard = expert_id // experts_per_shard`. Verify the
cumsum-position trick produces correct (shard, capacity_slot) indices. Flag
any overflow handling that creates extra axes.

### Agent 3 — all_to_all tiled semantics
For `jax.lax.all_to_all(x, axis_name='expert', split_axis=0, concat_axis=0,
tiled=True)` with x shape `(n_shards, capacity, H)` inside shard_map where
each shard has local x of that shape, what's the post-a2a shape? Confirm
whether `tiled=True` preserves or collapses leading axes. Propose the exact
reshape that should happen after each a2a.

### Agent 4 — in_specs / out_specs correctness
Read the `in_specs=` and `out_specs=` arguments to the shard_map call in
`_moe_shard_map_nvfp4`. Given mesh axis 'expert' of size 8, and inputs
`x: (B, H)`, `topk_w: (B, TOP_K)`, `topk_idx: (B, TOP_K) int32`, and the three
weight tuples each of shape `(NUM_EXPERTS=256, ..., ...)` sharded as
`P('expert', None, None)`, confirm each spec is correct. Propose fixes for
any wrong ones.

### Agent 5 — Weight index locality
Inside `_moe_shard_map_nvfp4`'s `_local` closure, each shard receives its slice
of `w1_packed, w1_scale, w1_scale2` (shape `(experts_per_shard=32, ..., ...)`).
When the per-expert loop does `w1p_l[e]` where `e` is the LOCAL expert index
0..31, confirm the indexing is against the LOCAL axis (after sharding strips
the first dim). Flag any places where global expert id is used instead of
local id.

### Agent 6 — Routing-weight shape tracking
`recv_w` is referenced on the line:
```
local_out = local_out * recv_w[:, None].astype(local_out.dtype)
```
Trace `send_w` -> `recv_w`: what shape does `send_w` have before the a2a, and
what shape does `recv_w` have after? Should `recv_w[:, None]` be `[..., None]`
instead? Propose the corrected broadcast.

### Agent 7 — Scatter-add combine correctness
Read the "send_back" / "recv_back" / "flat_back" / scatter-add `out.at[flat_tok]
.add(contrib)` block. Confirm that tokens going through TOP_K=8 expert contributions
get properly summed back to the original token positions, and the final
`out_specs` returns a replicated-over-expert view. Flag any missing weighted
multiply.

### Agent 8 — Why compile OOMs at 36.84GB when HBM is 31.25GB
XLA reported 36.84G allocation in HBM during compile for the full-62-layer
forward with NVFP4-packed experts. Per-chip weight memory at NL=62 is
~14 GB. Where do the other ~22 GB come from during compile? Likely
candidates: materializing dequantized bf16 weights (would be 62 layers x
3 x 2.4 GB = 450 GB total, 56 GB per chip at bf16 = obvious). Propose
compile-time flags or shard_map pattern changes that force XLA to NOT
materialize the full bf16 weight per expert.

### Agent 9 — `nvfp4_matmul` fusion with dot_general
Read `nvfp4_matmul` in `nvfp4_jax_ops.py`. Does it currently materialize the
full dequantized `w_bf16` as an intermediate before calling `dot_general`,
or does it use a bit-level trick to avoid that? If the intermediate is
materialized, propose either (a) a tiled rewrite that streams K-dim blocks
through dequant+dot, or (b) `jax.checkpoint` / `scan` to avoid peak memory.

### Agent 10 — Gemma 4 MoE style transfer
Read `moe_ffn` in `gemma4_tpu_infer.py` (~line 258) — that's the production
MoE FFN for Gemma 4 26B-A4B 128-experts-top-8 on TPU. It's simpler (gather-
based). Propose an alternative `moe_block_nvfp4_v2` that mimics Gemma 4's
layout but with NVFP4 dequant and 8-way expert shard, dropping the shard_map
+ all_to_all entirely in favor of jax.jit + pjit auto-sharding. Estimate the
perf impact at B=1 vs B=64.

### Agent 11 — Capacity = 2 and TOP_K = 8 with n_shards = 8
At B=1 TOP_K=8 n_shards=8: capacity = ceil(B*TOP_K/n_shards)*2 = ceil(8/8)*2 = 2.
So each shard has 2 slots for 1 token's contribution going to that shard. But
with TOP_K=8 and NUM_EXPERTS=256 and n_shards=8 = 32 experts/shard, for a single
token 8 experts might all be on DIFFERENT shards (one per shard), or multiple on
same shard. Verify capacity=2 is sufficient. Propose a test case.

### Agent 12 — B=1 decode vs B=batch behavior
The perf path (all_to_all + dispatch) has overhead that dominates at B=1.
A real production MoE at B=1 might use an alternative 'dense dispatch'
path: compute all 256 experts' outputs locally on each shard and broadcast
the top-8 weighted sum. Propose a `moe_block_nvfp4_dense_b1` variant for
B<=8 that avoids all_to_all entirely.

### Agent 13 — Real-model load parallelization
The real-model load takes 40+ min single-thread because `read_bf16` and
`read_nvfp4` iterate 190K tensors with Python dict lookup + safetensors read
+ numpy conversion + jax.device_put. Propose a parallelized loader that
reads tensors per-SHARD (61 files, each ~2 GB), processes all tensors from a
shard in one worker, and batches device_put operations. Target: <3 min load.

### Agent 14 — Fused dequant + matmul kernel in Pallas/Triton
The JAX-only `nvfp4_matmul` is unlikely to hit peak TPU MMU throughput
because XLA can't always fuse the LUT gather + broadcast-scale into the
matmul. Propose a Pallas kernel for NVFP4 matmul targeting TPU v6e MXU.
Estimate: Pallas supports custom loops; you'd load packed + scales in
SRAM tiles, dequant to bf16 in registers, run MXU matmul, accumulate in
f32, store. Provide a ~40-line Pallas sketch.

### Agent 15 — FP8 E4M3 decode fast path
Current `fp8_e4m3_decode` uses `jnp.power(2, exp-7)`. That's a floating-
point pow; XLA may not lower to a fast op. Propose a bit-level decode:
build a uint32 by placing exp+120 (re-bias from 7 to 127) into the f32
exponent field, and mantissa into bits 20..22 of f32 mantissa. Then
`jax.lax.bitcast_convert_type(..., jnp.float32)`. Give the exact code.

### Agent 16 — End-to-end synthesis plan
Read the outputs of agents 1-15 (or assume what they'll say based on the
code state). Produce a prioritized 5-step action plan: which fixes apply
first, which agents' proposals conflict, and the minimum set of changes
to land a real bench number at B=1, NL=62, ctx=2048. Target: 100+ tok/s
if MoE dispatch is well-sharded; report the confidence interval.

## Output format per agent

```
## Agent N — <title>

**Root cause:** <one sentence>

**Fix:** <file:function, exact replacement code or change>

**Depends on:** <agents whose output affects this, or "none">

**Confidence:** high | medium | low
```

## Coordinator (me) synthesis

After all 16 report:
1. Apply the must-have fixes (agents 1, 6 especially).
2. Consider the Pallas / Gemma-style rewrites (agents 10, 14) as v2 perf path.
3. Run the synth bench; if that works, re-run real-model load.
4. Only escalate to agent 11-15 proposals if v1 fix doesn't yield >10 tok/s.
