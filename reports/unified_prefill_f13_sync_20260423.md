# Phase F13 — drop post-softmax sync. Below measurement noise.

**Branch:** `rusty_sm121_unified_prefill_mma`
**Date:** 2026-04-23
**Result:** no meaningful win, not shipped.

## Change

The `__syncthreads()` after the softmax phase used to gate the
`s_acc *= alpha` rescale pass that F12 folded into the MMA C
operand. With the rescale gone, nothing between softmax's last
write and the next `__syncthreads` (after V load) reads any
softmax-written smem region — V load writes `s_v_fp8` and
`s_v_scale`, neither of which the softmax phase touches. The
post-softmax sync is formally redundant and was dropped.

Safety checked:

* `s_s`, `s_m`, `s_l`, `s_alpha` — written by softmax, next read by
  the P·V fold. There's a full `__syncthreads` between V load and
  fold which covers visibility.
* Warp-level `__shfl_xor_sync` in the P·V fold — fires after
  another full barrier; removing the softmax-end sync doesn't
  disturb warp coherency.

## Measurement

Live TTFT on the 1836-token chat prompt through rvllm-serve, 4-5
steady runs each:

    config   median (steady-state)
    F12       3.71 – 3.75 s
    F13       3.75 – 3.76 s
    noise     ±0.05 s run-to-run

Below the run-to-run measurement noise floor. Theoretical saving
is ~58 tiles × 60 layers × ~500 ns/sync ≈ 1.7 ms, which is indeed
within noise.

## Reverted

F13 change reverted in the same session. rvllm-serve rebuilt +
restarted on F12 state. Live TTFT back to 3.71-3.75 s.

## What this result means

We've hit the point where single-sync / single-fuse kernel tweaks
stop showing up through rvllm-serve's wall-clock. The remaining
~2× gap to vLLM (F12 at 3.71 s vs vLLM 1.8 s) is not going to be
closed by more of these — it's either:

* **Structurally** different (a Triton-compiled kernel has a
  denser inner loop — fewer scalar ops between MMAs, better
  register allocation, possibly warp specialization).
* **Outside** our kernel (vLLM's launch overhead amortisation,
  scheduler interleaving, etc.).

Either is a rewrite, not an F-series tweak.

## Recommendation

Merge F12 into `rusty_sm121` as the final Phase F commit. The
~2× residual gap is documented; closing it is a separate effort
that likely starts with "write a Triton-equivalent kernel from
scratch" rather than "keep polishing the hand-rolled CUDA one."

## Ledger (unchanged vs F12)

    config                         TTFT      × baseline
    production default             61.7 s    1.0×
    scalar unified                 11.7 s    5.3×
    F6 MMA (sliding only)           4.7 s    13.1×
    F7 MMA (sliding + global)       3.8 s    16.2×
    F12 + alpha fuse                3.71 s   **16.6×**   ← merge target
    F13 sync drop                   3.75 s   — noise
    vLLM Triton reference           1.8 s    34.3×
