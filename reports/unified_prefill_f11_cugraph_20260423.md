# Phase F11 — CUDA graph capture. Measured, not shipped.

**Branch:** `rusty_sm121_unified_prefill_mma`
**Date:** 2026-04-23.

## Hypothesis

Capture the ~600 kernel launches of one prefill into a CUDA graph
and replay with a single `cuGraphLaunch`. Expected to collapse
per-launch overhead (traditionally 5-10 µs × N launches) into a
single small cost.

## Measurement before building

One cudarc-less micro-benchmark — 10 000 `<<<1, 1>>>` no-op
launches, sequential vs captured graph:

    N=10000 launches, 17.55 ms total, 1.75 us/launch
    graph launch (10000 nodes), 6.45 ms total, 0.65 us/launch
    speedup: 2.7x

So on this GB10 + CUDA 13.2 + driver 595.58.03 stack:

* sequential launch cost: **1.75 µs/launch**
* graph launch cost: **0.65 µs/launch**
* graph savings: **~1.1 µs/launch**

For a realistic prefill:

    60 layers × ~10 kernels per layer + sampling tail   ≈ 605 launches
    sequential: 605 × 1.75 µs =  1.06 ms
    graph:      605 × 0.65 µs =  0.39 ms
    saved:                      ~0.67 ms

On F7's 3.8 s TTFT that's **0.018 %**. On decode (~200 ms/step, 60
launches) it's 0.05 %. Either way, sub-millisecond — well below
measurement noise and not worth the graph-capture infrastructure.

## Why CUDA graphs don't move the needle here

Graphs help when a program is dominated by many **small** kernels
where host submission latency bites. Our kernels are the opposite:

* The unified prefill attention kernel takes ~30-60 ms per launch
  at head_dim=256-512.
* The cuBLASLt GEMMs take a few ms each.
* Fused norm / quant / rope kernels are each 0.1-0.3 ms.

Launch overhead is < 0.1 % of per-kernel time for every kernel in
the prefill path. Even a 100 % graph speedup on launch cost would
only reclaim that 0.1 %.

Graph capture also adds real engineering cost:

* Per-prompt-length or per-shape graph cache (prompt_len controls
  grid dimensions on the attention kernel, which can't change
  post-capture without `cuGraphExecKernelNodeSetParams` plumbing).
* Pointer-argument updating across replays (weights pointer moves
  per layer — again `SetParams`).
* No dynamic allocation during capture, so the scratch arena has
  to be pre-sized per shape.

For < 1 ms of savings, none of that is justifiable.

## What *is* still on the table

From the F8 / F9 / F10 reports, one candidate remains that's layout-
insensitive **and** hits a mechanism the F9 measurement actually
flagged:

* **Fuse `s_acc *= alpha` into the P·V MMA `C` operand.** Today the
  alpha rescale is a dedicated pass over `BLOCK_M × head_dim` cells
  that reads + writes `s_acc` between softmax update and V load.
  Every tile round-trips the full accumulator through smem. If the
  MMA unpacker reads `s_acc[m, d]`, multiplies by `s_alpha[row]`,
  adds `d_frag * s_p_scale[row]`, and writes back — one smem R/W
  pair per (m, d) per tile goes away. At head_dim=512 that's
  `16 × 512 = 8 192` f32 smem R/W per tile per block, ≈ 64 KB of
  smem traffic saved per tile.
* Beyond that, the ~2× gap to vLLM is probably in the kernel proper
  — smem bank traffic, load patterns, a denser Triton-compiled
  inner loop. Closing it needs a rewrite more structural than
  bolt-on tweaks, which takes us out of "Phase F" territory.

## Recommendation

Ship F7 as-is (already merged into `rusty_sm121` + `rusty_sm121_
inference_server` earlier today). Optionally attempt the alpha-fuse
in a Phase F12 if you want another shot at trimming the attention
kernel. CUDA graph capture is correctly deferred — possibly useful
later if we ever land chunked prefill + many-small-batch decoding,
but not now.

## Micro-bench artefact

    /tmp/launch_bench.cu   (kept in /tmp; not committed — this is a
                            proof-of-cost not a library contribution)

## Ledger (unchanged by F11)

    config                              TTFT     × baseline
    production default                  61.7 s   1.0×
    scalar unified                      11.7 s   5.3×
    F6 MMA (sliding only)                4.7 s   13.1×
    F7 MMA (sliding + global)            3.8 s   16.2× ← current main
    vLLM Triton reference                1.8 s   34.3×
