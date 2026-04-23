# Unified prefill — tensor-core upgrade (Phase F plan)

**Baseline.** The unified FP8-KV prefill kernel landed on `rusty_sm121`
produces correct output but uses scalar FMAs for the Q·K^T and P·V
matmuls. End-to-end numbers on a 1836-token prompt through rvllm-serve:

| path                                                      | TTFT    |
|-----------------------------------------------------------|---------|
| production default (per-token outer loop)                 | 61.7 s  |
| `RVLLM_BATCH_PREFILL=1` + `RVLLM_UNIFIED_PREFILL=0` (per-qi) | 62.7 s  |
| `RVLLM_BATCH_PREFILL=1` + `RVLLM_UNIFIED_PREFILL=1`       | 15.1 s  |
| vLLM reference (Triton, `mma.sync`)                       |  1.8 s  |

~4× production speedup; still ~8× off vLLM. The delta is the matmul
path — Triton compiles `tl.dot` to `mma.sync.kind::f8f6f4.m16n8k32.row
.col.f32.e4m3.e4m3.f32` (GB10 FP8 tensor core). Our kernel reads FP8
from smem one byte at a time and does scalar f32 FMAs. Tensor cores
are ~32× the throughput of scalar FMA on the same thread count, which
is where the gap lives.

This spec scopes the tensor-core rewrite of the two hot matmuls and
leaves everything else (online softmax, masking, paged KV I/O,
per-slot scale cache) untouched.

## PTX instruction

    mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
      { d0..d3 }, { a0..a3 }, { b0, b1 }, { c0..c3 }

Tile shape per warp: **M=16, N=8, K=32**. One warp issues one MMA.
Registers per lane (32 lanes per warp):

* A fragment — 4 × u32 (16 FP8 bytes) covering a 16×32 row-major slice
  of A. Lane *i* holds rows `{i/4, i/4+8}`, cols `{(i%4)*8 … (i%4)*8+7}`,
  repeated for k=[0..7] then k=[8..15], stored in the 4 u32 slots.
* B fragment — 2 × u32 (8 FP8 bytes) covering an 8×32 col-major slice.
* D / C fragments — 4 × f32 accumulator lanes covering the 16×8 output.

The NVFP4 probe in `kernels/nvfp4_mma_probe.cu` (on `rusty_sm121_nvfp4`)
assembles on GB10 with this lane layout; the FP8 variant swaps `e2m1`
→ `e4m3` and is otherwise byte-identical.

## Matmul shapes we need to produce

For every (`BLOCK_M=16`, `head_dim`, `TILE_SIZE`) configuration the
kernel already handles:

| layer kind | head_dim | TILE_SIZE | Q·Kᵀ output | P·V output |
|------------|----------|-----------|-------------|------------|
| sliding    | 256      | 32        | 16×32       | 16×256     |
| global     | 512      | 16        | 16×16       | 16×512     |

Decomposition into `m16n8k32` tiles (m fixed to BLOCK_M=16, so one row
of MMAs per output):

**Q·Kᵀ — sliding:** output 16×32 = **4** n-tiles of width 8. K-reduction
covers `head_dim=256` in `256 / 32 = 8` k-steps per n-tile, accumulating
into the same D fragment. Total MMAs per Q·Kᵀ per layer = 4·8 = **32**.

**Q·Kᵀ — global:** output 16×16 = 2 n-tiles, k=32, `head_dim=512` →
16 k-steps. Total = 2·16 = **32**.

**P·V — sliding:** output 16×256. P is [16×32] already in registers
from the softmax update (rescaled by alpha). V is [32×256]. Treating
P as A (m16k32) and V as B (n, k=32): n = 256/8 = 32 n-tiles, k=32 =
one k-step per n-tile. Total = **32** MMAs.

**P·V — global:** output 16×512. P is [16×16] (shorter tile). We need
2 k-steps per 16-P-row (since the MMA requires k=32, and P only
provides 16 k-lanes we either pad-zero the second half of K or do
TILE_SIZE=32 on global too and halve the k work). Simplest: still use
TILE_SIZE=16, pad P's second half with zeros, do n=512/8=64 tiles ×
1 k-step = **64** MMAs. Cheap; the real cost is the V loads.

Baseline rough cost per layer per q_block at `head_dim=256`,
`TILE_SIZE=32`, ~60 tiles:

| op                | scalar FMAs (per layer per q_block) | mmas (per layer per q_block) |
|-------------------|------------------------------------:|-----------------------------:|
| Q·Kᵀ              | 60 × 16·32·256 = **7.86 M**         | 60 × 32 = **1 920**          |
| P·V               | 60 × 16·32·256 = 7.86 M             | 60 × 32 = 1 920              |

Each MMA does 16·8·32 = 4 096 FP8 multiplies. 1 920 MMAs = 7.86 M FMAs,
same total, but distributed across tensor cores — the expected speedup
is ≈32×.

## Thread / warp layout

Current kernel: 128 threads / block = 4 warps, laid out as 16 rows × 8
lanes for softmax reductions. That layout doesn't match the MMA warp
requirement (each MMA needs its warp's 32 lanes cooperating).

New plan: 4 warps, each owning a non-overlapping slice of the work:

* **Q·Kᵀ.** 4 warps × (n-tiles / 4) = each warp handles 1-8 n-tiles
  serially. At `head_dim=256`, TILE=32: 4 n-tiles → one per warp. Each
  warp loops through 8 k-steps accumulating in its D fragment, then
  writes S to smem for the softmax phase. Warps stay independent.
* **P·V.** At `head_dim=256`: 32 n-tiles across 4 warps = 8 n-tiles per
  warp, 1 k-step each. Each warp's output D is a 16×8 slice that gets
  added into the running acc; running acc lives in smem (already does)
  and the warp reduces its D into the appropriate acc rows.

Softmax + mask still runs on the 128-thread row layout after the MMA
phase — we convert between layouts via smem writes of the S fragment.
That adds two `__syncthreads` per tile (post-QK, post-PV) — we already
have one there today.

## Operand packing

This is the part that looks mechanical but bites first-time readers.
Two helpers in a small inline shim:

```cuda
__device__ __forceinline__
void pack_a_frag_row_major_m16k32(
    const uint8_t* src,  // [16][32] row-major in smem
    int src_row_stride,  // bytes = head_dim
    uint32_t a[4], int lane);

__device__ __forceinline__
void pack_b_frag_col_major_n8k32(
    const uint8_t* src,  // [8][32] col-major
    int src_row_stride,
    uint32_t b[2], int lane);
```

Packing is a fixed permutation of the lane index; derive it once from
the PTX spec table (same as the NVFP4 probe; differ only in that FP8
E4M3 occupies 1 byte per element vs NVFP4's 4 bits, so packing is
trivially 4 bytes per u32 with no nibble interleaving).

For fragments that straddle multiple MMA calls (e.g. Q is reused
across all n-tiles in a row): load Q into registers ONCE per q_block,
reuse across the inner loop. K fragments change per (n-tile, k-step).

## How Q / K scales get applied

The scale cascade is the correctness gate. Current kernel:

```
s_q[m, d]   = fp8_decode(Q[t, h, d]) * q_scale[t, h] * softmax_scale
s_k_tile[d] = FP8 byte (unmodified)
k_scale[t]  = per-slot f32
S[m, t]     = Σ_d s_q[m, d] * fp8_decode(s_k_tile[t, d])    ← scalar loop
            * k_scale[t]
```

With tensor cores, Q must enter the MMA as FP8. Two options:

* **(A) Keep Q in FP8 in smem, bake scales post-MMA.** Load Q bytes
  unmodified, pass to MMA, multiply the f32 accumulator by
  `q_scale[m] × k_scale[t] × softmax_scale` before softmax. This is
  the vLLM pattern (`KV_QUANT_MODE >= 2`) — they apply
  `k_token_head_scales` after the dot. Straightforward, preserves
  per-row/per-slot scale precision.
* **(B) Pre-multiply Q by its scales into FP8.** Requires quantising
  Q *scaled* — loses precision because `Q × q_scale × softmax_scale`
  no longer lives in [-1, 1] at the same granularity. Bad idea.

Go with (A). Same applies to V: load FP8 unmodified, multiply acc by
`v_scale[t]` post-MMA (fused into the P multiply since P is already
f32: `P_scaled = P × v_scale[:, None]` before casting to FP8 for MMA
input).

Actually P going into MMA is the tricky direction: P is f32 coming
out of softmax. The MMA requires FP8 A operand. We must re-quantise P
to FP8 for the MMA, and that's a precision hit. Options:

1. Accept the re-quant. Apply a dynamic per-row scale on P (max abs
   across the tile, scale into [-1, 1]), store as FP8, undo the scale
   on the accumulator. vLLM does exactly this; the re-quant error is
   well below the other FP8 noise.
2. Skip P's MMA — keep the P·V loop scalar. Only 50 % of the win but
   half the complexity.

Default plan: **option 1** for both matmuls. Start with (2) as a
checkpoint if Q·Kᵀ MMA alone lands cleanly; it's trivially revertable.

## Smem budget

Needs re-check after layout change. Per-warp fragment registers cost
no smem. Current smem (sliding, head=256):

* `s_q` 16 KB — **unchanged** (Q byte tile loaded once, used by all n-tiles)
* `s_k_fp8` 8 KB — unchanged
* `s_v_fp8` 8 KB — unchanged
* `s_s` 2 KB — unchanged (still holds the 16×32 f32 S tensor post-MMA)
* `s_acc` 16 KB — unchanged
* miscellaneous (M, L, alpha, scales) — ~1 KB
* total ≈ 51 KB — same as today, well inside the 99 KB cap.

For global (head=512, tile=16): 82 KB today, similarly unchanged.

## Phases

* **F1.** MMA probe in `kernels/fp8_e4m3_mma_probe.cu` — one
  standalone kernel that runs one `mma.sync` on fixed inputs and
  writes the D fragment. Python harness
  `v3/tools/fp8_mma_probe_check.py` compares against an fp64 NumPy
  reference. Proves PTX assembles + lane layout matches on GB10
  without touching the FA2 kernel. **Target: 1 commit, 2 files.**
* **F2.** Operand-pack helpers (`pack_a_frag_row_major_m16k32`,
  `pack_b_frag_col_major_n8k32`) in a new header included by
  `flash_attention_unified_prefill.cu`. Unit-test via a second probe
  that packs a known input and checks every u32 against the spec.
  **Target: 1 commit, 2-3 files.**
* **F3.** Replace the Q·Kᵀ scalar loop with MMA. P·V stays scalar for
  this commit. Run the existing
  `v3/tools/fa2_unified_prefill_check.py` — scale_rel threshold
  loosens to ~1e-1 to absorb the extra FP8 round. End-to-end TTFT
  expected ≈ 8-10 s on the 1836-token prompt (Q·Kᵀ was ~half the
  cost).
* **F4.** Replace the P·V scalar loop with MMA. P re-quant to FP8
  with per-row max scale. End-to-end TTFT expected ≈ 3-5 s.
* **F5.** Measure against vLLM on the same hardware. Document any
  remaining gap (launch overhead, smem bank conflicts, occupancy).

Rollback is per-phase: each of F3/F4 is gated on
`RVLLM_UNIFIED_PREFILL_MMA=1` (default 0 during bring-up, flip to 1
after correctness signs off). Keep the scalar path live until the
numbers justify removing it — or keep permanently for arches without
FP8 tensor cores.

## Risks + mitigations

* **Precision.** P→FP8 re-quant is the biggest unknown. Mitigation:
  per-row dynamic scale in the MMA branch, but preserve the scalar
  path behind the env toggle for A/B. Run `rvllm-ppl chunk=128`
  after each MMA phase — stay within 0.1 of the scalar baseline or
  revert.
* **Register pressure.** Each warp holds Q + partial acc as MMA
  fragments. With 4 warps × 4+4+4 register fragments per warp ≈ 48
  regs/lane just for fragments, plus scratch for softmax. Should
  stay under the 64-reg occupancy sweet spot on sm_121 but measure
  with `--ptxas-options=-v`.
* **Lane layout bugs.** Hardest to debug — wrong output comes back
  as garbage that passes shape checks. Mitigation: F1 probe with
  fp64 reference + per-lane u32 dumps, before ever running the full
  kernel.
* **GB10 firmware oddities.** The sm_121 board has shown
  smem/v4-load hangs historically (GB10_SPEC §hardware quirks).
  MMA + smem is a new combination; keep a scalar fallback path in
  the kernel for the first few days post-integration until we trust
  the GPU side.

## Non-goals for this branch

* Multi-warp cooperative MMA tiles (ldmatrix + mbarrier). The
  simple "one warp owns its slice" plan above gets us most of the
  speedup with far less infrastructure; async copies + barriers are
  the canonical follow-up if we hit bandwidth limits.
* bf16 accumulator (`.bf16.bf16.f32` mma). FP8 accumulator is f32,
  which is what we want; the bf16 variant is for a different data
  path and doesn't apply.
* head_dim=128 support. Gemma 4 doesn't use it; adding the tile
  shape later is a one-function change.

## File layout (anticipated)

| file                                                  | phase |
|-------------------------------------------------------|-------|
| `kernels/fp8_e4m3_mma_probe.cu`                       | F1    |
| `v3/tools/fp8_mma_probe_check.py`                     | F1    |
| `kernels/fp8_mma_frag_pack.cuh`                       | F2    |
| `kernels/flash_attention_unified_prefill.cu` (edit)   | F3-F4 |
| `v3/tools/fa2_unified_prefill_check.py` (threshold loosened) | F3-F4 |
| `v3/UNIFIED_PREFILL_MMA_PLAN.md`                      | this  |
