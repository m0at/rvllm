# Phase F10 — `ldmatrix` for MMA fragment loads. Abandoned.

**Branch:** `rusty_sm121_unified_prefill_mma` (F10 work sits alongside
F7; no code in the main kernel path).
**Date:** 2026-04-23.

## Goal

Replace the manual `pack_a_frag_row_major_m16k32` /
`pack_b_frag_col_major_n8k32` byte-shuffle arithmetic (~4 u32 /
~2 u32 loads per lane per MMA, with the k-boundary indexing
computed from `lane / 4` + `(lane % 4) * 8`) with a single
`ldmatrix.sync.aligned` instruction that the hardware already knows
how to lane-permute into the MMA operand shape.

## sm_121a `ldmatrix` variants — what works

Scanned the PTX variants nvcc 13.2 will emit for `sm_121a`:

    ldmatrix.sync.aligned.m8n8.x4.shared.b16                         OK
    ldmatrix.sync.aligned.m8n16.x1.b8x16.b6x16_p32.shared            OK  (FP6 + padding)
    ldmatrix.sync.aligned.m16n16.x2.b8.shared                        ptxas FAIL
    ldmatrix.sync.aligned.m16n16.x4.b8.shared                        ptxas FAIL
    ldmatrix.sync.aligned.m8n8.x4.b8.shared                          ptxas FAIL
    ldmatrix.sync.aligned.m8n16.x4.b8x16.b4x16_p64.shared            ptxas FAIL

So sm_121a accepts `.b16` for the `.m8n8` shape only; the
MMA-friendly genuine `.b8` variants (the ones `f8f6f4` MMA
integrates with on Hopper / SM90) aren't in Blackwell-consumer's
PTX feature set.

## Layout mismatch of `.m8n8.x4.b16`

The one variant that assembles has the wrong per-lane output layout
for our `mma.sync.kind::f8f6f4.m16n8k32.row.col` A-fragment.
`kernels/ldmatrix_probe.cu` loads a 16×32-byte test tile and dumps
both paths side-by-side; `v3/tools/ldmatrix_probe_check.py` decodes
them:

    Per-lane 4-u32 manual pack vs ldmatrix.m8n8.x4.b16:
      lanes matching position-for-position: 0/32
      total u32 matching: 32/128

Lane 0 illustrates the problem — manual delivers 4 **distinct** u32s
(k = 0..3, 4..7 for row 0 and row 8); ldmatrix delivers 2 distinct
u32s (k = 0..3 for rows 0/8 + k = 16..19 for rows 0/8), each
repeated twice:

    lane 0 manual:  03020100 83828180 07060504 87868584
    lane 0 ldmat:   03020100 83828180 03020100 83828180

Decoded:
* Manual `a[0]` = A[row=0,   k=0..3]   = `{0x00, 0x01, 0x02, 0x03}` ✓
* Manual `a[2]` = A[row=0,   k=4..7]   = `{0x04, 0x05, 0x06, 0x07}` ✓
* Ldmatrix `r2` = A[row=0,   k=16..19] = `{0x00, 0x01, 0x02, 0x03}` — wrong column range for MMA

Matrix 0 (rows 0..7, k = 0..15) and matrix 1 (rows 8..15, k = 0..15)
match the MMA's `a[0]` / `a[1]` *for lane 0 only*. Matrices 2 / 3
would need to be re-addressed or the smem source pre-swizzled so
that "16 b16 along k" == "the MMA's second 16-byte group of the same
row".

## Why I'm stopping

Two realistic paths to make `.m8n8.x4.b16` work:

1. **Post-ldmatrix register permutation via `__shfl_sync`.** Correct
   but costly — the reshuffle is a 32-lane butterfly with non-trivial
   masks, and CUTLASS (on Hopper) avoids this exactly by using
   `.m8n16.x4.b8` which doesn't exist on sm_121a.
2. **Pre-swizzle smem into 2×(8×16) tiles per A fragment.** Means
   changing how `s_q_fp8` is laid out (today: row-major 16 × 32,
   which is what every subsequent smem consumer in the kernel
   expects).

Either path is a kernel rewrite, not a load-site swap. Given:

* F9's result already showed that kernel-structural changes on this
  branch can backfire in non-obvious ways.
* The expected win on top of F7 is small — F7's `pack_a_frag` is
  already 4 `ld.shared.u32` instructions per lane, which smem can
  service in a few cycles.
* The NVFP4 probe (`rusty_sm121_nvfp4` branch's
  `nvfp4_mma_probe.cu`) uses the same manual-pack pattern we do —
  prior art on the same hardware made the same call.

I'd rather stop here than ship a second regression.

## What next

The F8 / F9 reports pointed at two targets whose mechanism is less
layout-sensitive:

1. **Fuse `s_acc *= alpha` into the P·V MMA C operand.** Today
   the alpha rescale runs as a dedicated pass over
   `BLOCK_M × head_dim` cells between the softmax update and the V
   load. If the unpack at the end of the P·V MMA reads the previous
   `s_acc` value, multiplies by `alpha[row]`, adds `d_frag *
   s_p_scale[row]`, and writes back — one smem round trip per tile
   goes away. No lane-layout hazard.
2. **Piecewise CUDA-graph capture of the 60-layer prefill.**
   Addresses launch overhead directly; affects every kernel, not
   just attention. Needs `cuStreamBeginCapture` plumbing in the
   runtime, no kernel-side changes.

Either of these fits better with what F9 revealed the real ceiling
to be (memory + launch throughput, not smem-resident operand
shuffling).

## Files left behind for reference

    kernels/ldmatrix_probe.cu           — manual vs ldmatrix comparison
    v3/tools/ldmatrix_probe_check.py    — decoder + layout dump

These stay on the branch so the next person looking at ldmatrix on
sm_121a has the probe + the known outcome.

## Ledger (unchanged by F10)

    config                              TTFT     × baseline
    production default (per-token)      61.7 s   1.0×
    scalar unified                      11.7 s   5.3×
    F6 MMA (sliding only)                4.7 s   13.1×
    F7 MMA (sliding + global)            3.8 s   **16.2×**
    vLLM Triton reference                1.8 s   34.3×

F7 is the production state on `rusty_sm121` (merged 2026-04-23).
The experimental branch keeps F8 / F9 / F10 reports as the trail
of attempts.
