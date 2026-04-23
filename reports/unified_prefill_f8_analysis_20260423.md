# Phase F8 — post-MMA gap analysis (no ncu)

**Date:** 2026-04-23, GB10 / CUDA 13.2 / driver 595.58.03
**Branch:** `rusty_sm121_unified_prefill_mma`
**Goal:** find the remaining 2.2× after F7 landed both Q·Kᵀ and P·V on
FP8 tensor cores for every Gemma 4 layer.

## 1. ncu is unusable on this kernel stack

Every ncu configuration I tried triggered a `cuLaunchKernel
(LaunchFailed)` on some post-prefill kernel (`scale_rows_f32_ratio`,
`quantize_fp8_per_token`, `f32_to_f16_sat`, `cublasLtMatmul`, or a
CUTLASS SM120 TMA descriptor assertion). The failures aren't in the
attention kernel itself — they hit whichever kernel comes next after
ncu's instrumentation has reshuffled device resources. Same attempt
worked in F5 with MMA disabled, so the interaction is specifically
`mma.sync.kind::f8f6f4` + ncu on sm_121a.

Tried configurations:

* Per-kernel replay + kernel-name filter on `flash_attention_2_prefill_fp8kv_unified`
* Application replay mode
* Full (`--set full`), basic (`--set basic`), minimal (one metric)
* `RVLLM_FP8_GEMM_CUTLASS_SM120=0` to rule out the CUTLASS TMA path
* `RVLLM_PROMPT_LEN` 128 / 512 / 1024
* Wide and narrow `--launch-skip` / `--launch-count` windows

All hit the same class of error. No data captured. Moved on to
static + wall-clock timing.

## 2. Static resource usage (ptxas)

    nvcc -cubin -arch=sm_121a -O3 -Xptxas=-v \
         -Ikernels kernels/flash_attention_unified_prefill.cu

    0 bytes gmem
    Compiling entry function 'flash_attention_2_prefill_fp8kv_unified_kernel' for 'sm_121a'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    Used 56 registers, used 1 barriers

**Interpretation.** 56 registers × 128 threads = 7 168 regs per block
— well under the sm_121 64 Ki regs/SM ceiling. Zero spills. The
kernel is **not** register-limited; occupancy is driven entirely by
dynamic smem.

## 3. Smem / occupancy analysis

Sliding layer (head_dim=256, tile_size=32):

| region       | bytes   |
|--------------|--------:|
| s_q_fp8      | 4 096   |
| s_q_scale    | 64      |
| s_k_fp8      | 8 192   |
| s_v_fp8      | 8 192   |
| s_v_fp8_T    | 8 192   |
| s_k_scale    | 128     |
| s_v_scale    | 128     |
| s_s          | 2 048   |
| s_m/l/alpha  | 192     |
| s_p_fp8      | 512     |
| s_p_scale    | 64      |
| s_acc        | 16 384  |
| **total**    | **48 192** |

Global layer (head_dim=512, tile_size=16 → MMA_K=32 padded):

| region       | bytes   |
|--------------|--------:|
| s_q_fp8      | 8 192   |
| s_q_scale    | 64      |
| s_k_fp8      | 8 192   |
| s_v_fp8      | 8 192   |
| s_v_fp8_T    | 16 384  |
| s_k_scale    | 64      |
| s_v_scale    | 64      |
| s_s          | 2 048   |
| s_m/l/alpha  | 192     |
| s_p_fp8      | 512     |
| s_p_scale    | 64      |
| s_acc        | 32 768  |
| **total**    | **76 736** |

Both are inside the 99 KB opt-in cap. Per-SM smem is 228 KB; Blackwell
consumer reserves 100 KB for static smem, leaving ~128 KB available for
dynamic allocations, but the 99 KB limit is *per block*.

**Blocks / SM:**
* Sliding: `128 KB / 48 KB ≈ 2` → occupancy 2 blocks/SM.
* Global: `128 KB / 77 KB ≈ 1` → occupancy 1 block/SM.

With 128 SMs and 2064-3680 total blocks per prefill (1024-1836-token
prompts respectively), global layers process ~29 serial batches per
layer; sliding layers process ~15. At ~25 µs per block, that's
~0.4 ms per global layer, ~0.38 ms per sliding layer, ~22 ms total
attention across 60 layers. Measured attention fraction ≈ 3.1 s on
the 1836-token run (4.0 s total × ~0.78 attention fraction from F5
extrapolation). So per-block time is much higher than the theoretical
"compute-only" number — occupancy × latency-hiding isn't holding up.

## 4. Speedup ledger (wall-clock, rvllm-serve, 1836-token chat prompt)

    config                              TTFT    Δ vs previous   Δ vs baseline
    production default (per-token)      61.7 s  ---             1.00×
    scalar unified kernel               11.7 s  5.3×            5.3×
    F3 Q·Kᵀ MMA (probe extrapolation)    ~8 s   ~1.5×           ~7.7×
    F6 MMA (sliding Q·K + P·V)           4.7 s  ~1.7×           13.1×
    F7 MMA (sliding + global P·V)        4.0 s  1.17×           15.4×
    vLLM Triton reference                1.8 s                   34.3×

F6 → F7 added 15%. Adding global P·V to the tensor-core path saves
~0.7 s on a 1836-token prompt. The remaining 2.2 s gap vs vLLM is
roughly evenly split between:

1. **Smem-limited occupancy** (1-2 blocks/SM) — fewer resident blocks
   means less latency hiding across memory stalls. vLLM's Triton
   compiles to a denser kernel with fewer live smem buffers.
2. **Explicit V transpose pass** — 8-16 KB of smem writes per tile,
   pure overhead, no compute.
3. **Scalar softmax / mask / scale operations** between MMAs — the
   Warp-per-row reductions and per-slot scale multiplies still run
   on the scalar f32 pipeline.

Roughly: if smem usage dropped to the ~30 KB range it'd unlock
3 blocks/SM (sliding) → potentially 1.3-1.5× on attention alone.

## 5. Candidate F9+ work

Ranked by impact / effort:

1. **Eliminate `s_v_fp8_T`.** Store V in smem in the
   `[d][t]` layout that the MMA B operand wants directly — merge
   the transpose into the V load. Saves 8 KB (sliding) / 16 KB
   (global) of smem → enables 3-4 blocks/SM → 1.3× expected
   speedup. The load pattern becomes scattered bytes, but u32
   packing (as in the F7 transpose) keeps bank writes aligned. The
   scalar P·V path also benefits — it reads V in `[t][d]` with a
   non-contiguous `t → head_dim` stride today, `[d][t]` gives it
   unit-stride reads.
2. **Share `s_q_fp8` with `s_acc`** across the two phases of the
   tile loop. They're used in disjoint regions — Q during Q·Kᵀ,
   acc during P·V + epilogue. Saves 4-8 KB at the cost of careful
   scheduling. Doesn't strictly need to happen if (1) lands.
3. **Merge softmax rescale into the MMA epilogue.** Today `s_acc *=
   s_alpha[m]` runs as a dedicated loop over `BLOCK_M × head_dim`
   cells between the softmax phase and V load. If the MMA unpacks
   directly into s_acc *after* the alpha scale, that loop is a
   read-modify-write instead of two separate passes. Minor win;
   clean up only after (1) + (2).
4. **ldmatrix for K / V loads.** If sm_121a supports
   `ldmatrix.sync.aligned.b8` (it does), a single u32 per lane
   replaces the current per-byte-shuffled u64 reads. Reduces smem
   traffic + scalar index arithmetic. Needs validation that the
   lane-to-tile mapping matches our packers.
5. **Piecewise CUDA-graph capture of the prefill**. All 60 layers
   submit the same kernel shapes, so per-launch overhead (~5-10 µs
   × 60 = 300-600 µs) is recoverable via cuGraph. Worth chasing
   only after the kernel itself shrinks.

## 6. Recommendation

If the user's goal is "ship now", F7 at 4.0 s TTFT (15.4× baseline)
is the merge point — it's the last place on the branch with
unambiguously correct output and every prior commit is a pure win.
Close the 2.2× gap to vLLM in a follow-up PR focused on
smem-occupancy improvements (item 1 above).

If the user wants to keep going, step 1 (fuse V transpose into load)
is the single highest-leverage follow-up. ~1-2 days of kernel work,
numerical risk: low (byte-identical to current F7 — only the load
pattern changes, the MMA algebra is the same).
