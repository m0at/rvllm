# Phase F9 — attempted: eliminate `s_v_fp8_T`. Reverted.

**Branch:** `rusty_sm121_unified_prefill_mma`
**Date:** 2026-04-23

## Hypothesis

F8's leading candidate: store V directly in the MMA's `[d][k]` layout
at stride MMA_K during the load, drop the separate `s_v_fp8_T`
transpose region, alias `s_k_fp8` with `s_v_fp8` (they're live in
disjoint phases). Expected:

* Smem: sliding 48 → 32 KB → 3 blocks/SM (was 2), global 77 → 61 KB.
* Register count: 56 → 48 (fewer live smem pointers).
* TTFT: ~1.3× on attention → 3.8 s → ~3 s.

## Implementation

Replaced the F7 coalesced V load + u32-packed transpose pass with a
fused load that writes to `s_v_fp8[d * MMA_K + t_base]` directly:

```cuda
for (int idx = tid; idx < (MMA_K/4) * head_dim; idx += FA2_THREADS) {
    const int tg = idx / head_dim;
    const int d  = idx - tg * head_dim;
    const int t_base = tg * 4;
    uint32_t packed = 0;
    for (int k = 0; k < 4; k++) {
        const int t = t_base + k;
        unsigned char b = (t < tile_len)
            ? value_cache[(slot_for(t) * num_kv_heads + kv_head) * head_dim + d]
            : 0;
        packed |= (uint32_t)b << (k * 8);
    }
    *reinterpret_cast<uint32_t*>(s_v_fp8 + d * MMA_K + t_base) = packed;
}
```

Also aliased `s_k_fp8` with `s_v_fp8` (same smem base), dropped the
transpose block from the P·V MMA, pointed the scalar P·V at the new
`[d][k]` layout.

Correctness passes identically to F7 (diff tool sees max 6.3e-2
abs, same as F7 — the bytes are the same, only the layout changed).
ptxas confirms registers down from 56 → 48 and no spills.

## Result — TTFT regressed

Live on the 1836-token chat prompt:

| config              | TTFT (steady)  |
|---------------------|----------------|
| F7 (baseline)       | 3.8 s          |
| F9 (fused `[d][k]` load) | **4.8 s** (+26%) |

26 % **slower** despite the smem savings. Output is still correct
(`"Computing's"`), so the regression is pure performance.

## Root cause

The tradeoff: F7's transpose pass reads the paged cache with
coalesced 8-byte `__ldg` loads and scatters bytes inside smem.
F9 flips it — reads are scattered across slots (one u8 per lane per
iteration × 4 slots per u32), writes to smem are 4-byte-aligned.

- **F7 cache reads**: 32 threads × 8 bytes = 256 contiguous bytes per
  warp per iteration. One 128-byte coalesced line fetch pattern.
- **F9 cache reads**: 4 separate u8 loads per iteration (per thread).
  Each is coalesced across the 32-lane warp (same slot, same kv_head,
  consecutive d) as 32 bytes. Four scattered 32-byte fetches vs one
  256-byte fetch → half the effective cache throughput.

Plus:

- Per-iteration block-table lookups went from 1 per u64 (8 bytes) to
  4 per u32 (4 bytes) → 8× more block-table reads. Compiler hoisting
  helps but doesn't fully eliminate the overhead.
- The smem-writes-per-iter dropped, but smem writes were never the
  bottleneck.

The occupancy improvement (sliding 2 → 3 blocks/SM) would help if
the attention kernel were latency-bound. It isn't obviously — the
F7 measurements suggest it's more throughput-bound on the global-
memory side.

## Revert

F9 changes rolled back to F7 state (commit `aa927b9`). rvllm-serve
rebuilt + restarted; live TTFT confirmed back at 3.8 s. No commit
on the branch for the F9 attempt — kept as this report only.

## What the next attempt should try instead

Looking at the actual bottleneck implied by the regression (cache
read coalescing, not smem occupancy), the next optimization
candidates ranked by likely impact:

1. **`ldmatrix` for K / V loads.** sm_121a supports
   `ldmatrix.sync.aligned.b8` which gives a single u32-per-lane
   smem→register load in the exact lane pattern the MMA frag
   expects, avoiding the explicit `pack_a_frag` /
   `pack_b_frag_col_major_n8k32` arithmetic. Keeps F7's coalesced
   cache read pattern but trims the middle hop.
2. **Merge the softmax-phase `s_acc *= alpha` rescale into the
   P·V MMA accumulator**. Today it's a dedicated loop over
   `BLOCK_M × head_dim` cells between softmax and V load. Fusing
   it into `rvllm::mma_m16n8k32_e4m3_e4m3_f32(d, a, b)` by passing
   `alpha × s_acc_old` as the `C` operand is a direct win — one
   fewer smem round-trip per tile.
3. **Piecewise CUDA-graph capture for the 60-layer pipeline.**
   Per-launch overhead is 5-10 µs × 60 layers × many kernels per
   layer; graph capture collapses that to one. Most relevant
   *after* the kernel itself shrinks further.

None of these individually have the smem-pressure characteristic
that F9 bet on, but they target the actual observed ceiling.

## Ledger (unchanged by F9)

    config                              TTFT    × baseline
    production default (per-token)      61.7 s  1.0×
    scalar unified kernel               11.7 s  5.3×
    F6 MMA (sliding only)                4.7 s  13.1×
    F7 MMA (sliding + global)            3.8 s  **16.2×**
    vLLM Triton reference                1.8 s  34.3×

F7 remains the production state. The ~2× gap to vLLM is real,
profile-driven work for a follow-up.
