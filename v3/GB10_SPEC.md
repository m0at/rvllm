# GB10 / sm_121 — Blackwell Consumer Target Spec

## Hardware

- **GB10 = "Project DIGITS" = DGX Spark.** Grace ARM CPU + Blackwell consumer
  GPU in a single package. NVIDIA's dev kit for local LLM work.
- Compute capability **12.1** → PTX arch `sm_121`.
  - Distinct from `sm_120` (RTX 5090 / RTX 6000 Blackwell) and `sm_122`
    (RTX 5080 / RTX 5070). Same Blackwell ISA family, different memory +
    firmware profile.
- Memory: **LPDDR5X unified** (shared CPU + GPU), ~273 GB/s advertised,
  ~150 ns latency. No dedicated HBM.
- Clocks: **851 MHz sustained** turning into **507 MHz throttled** after
  ~3 s of sustained compute load. The throttle is firmware-enforced and
  overrides `nvidia-smi -lgc`.

## Required toolchain

- **CUDA 13.0+** for `nvcc` to recognise `-arch=sm_121`. Earlier releases
  accept `sm_120` / `sm_122` only.
- **cudarc 0.19+** on the Rust side. Older cudarc releases (notably 0.12)
  do not know about Blackwell consumer compute caps; do NOT downgrade.

## Hardware quirks we must code around

These were found the hard way (kernel hangs, not compile errors):

1. **128-bit vector loads (`float4`/`uint4`) hang.** Use scalar 32-bit or
   64-bit loads (`uint32_t*` via `__ldg`, or `unsigned long long` via
   `__ldg`). The PR #28 kernels document this with comments like
   "scalar loads only (no float4/v4) to avoid sm_121 hang with shared
   memory".
2. **Shared memory + v4 loads combined → hang.** Either drop the shared
   memory usage, or drop to scalar loads. Not both.
3. **Some shared-memory paths hang on their own.** The PR keeps
   shared-memory variants in `if false { … }` branches as documentation
   for a later bisect.
4. **Static shared memory ≤ 32 bytes** is the only reliable size for the
   `int4_gemv` pattern. Larger static allocations caused observed hangs.
5. **`LaunchConfig::shared_mem_bytes = 4` minimum** (not 0) — "non-zero
   for Blackwell compat; static smem handles the rest".
6. **Exactly one `__syncthreads()`** in the reduction path is safe;
   multiple sync points triggered hangs during PR #28 bring-up.
7. **The native `cvt.rn.f16x2.e4m3x2` PTX instruction** is available and
   fast (3 insn/byte vs 24 branchless), but consumes enough power to
   trigger the firmware clock throttle. Under throttle, the branchless
   path wins. There is no single "best" kernel for GB10 — the dispatcher
   picks based on clock regime.

## SM121 hardware feature matrix

Concrete limits vs Hopper that shape what kernels we can port.
Collected from NVIDIA CUDA docs + cross-referenced against
external SM121 write-ups (e.g. the spark-vllm-mxfp4-docker
SM121 technical guide).

| Feature                 | SM90 (Hopper)                | SM121 (GB10)                 |
|-------------------------|------------------------------|------------------------------|
| MMA instruction         | `wgmma.mma_async` (async WG) | `mma.sync.aligned` (sync)    |
| MMA tile size           | 64×N×32 async                | 16×8×32 sync                 |
| TMA                     | full (load + multicast)      | load only, **no multicast**  |
| DSMEM                   | ✅                           | ❌                           |
| Thread-block cluster    | up to 16 blocks              | **1×1×1 only**               |
| FA3 (WGMMA+TMA)         | native                       | not applicable (we use FA2)  |

Consequence for this branch: our sm_121 kernels use traditional
`mma.sync`-style tiling, scalar loads, no cluster features. The FA2
`FA2_BC=32` change (vs SM90's 64) is arch-conditional precisely
because Blackwell-consumer smem and tile budgets differ.

## Arch suffix (`-arch=sm_121` vs `sm_121a` vs `sm_121f`)

`kernels/build.sh` currently compiles with `-arch=sm_121`. This
produces a PTX target without the `__CUDA_ARCH_FAMILY_SPECIFIC__`
macro. Family-specific instructions that need it include:

- `mma.kind::mxf4.block_scale` (MXFP4 block-scaled MMA)
- `ldmatrix.b8x16.b4x16_p64` (FP4 ldmatrix with format conversion)
- `cvt.e2m1x2` (FP4 ↔ FP16 scalar conversion)

Our current FP8 scalar CVT path (`cvt.rn.f16x2.e4m3x2`) works on
plain `sm_121` — the numerical precision check confirms it. No
switch required for what ships on this branch.

**Follow-up trigger:** when we add FP8 tensor-core MMA
(`mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32`) or any MXFP4
path, switch `build.sh` to `-arch=sm_121f` (family mode, Blackwell
consumer) to enable the hardware paths. `sm_121a` also works but
`sm_121f` is the recommended form going forward.

## Empirical numbers (this DGX Spark, driver 595.58.03, CUDA 13.2)

Measured with `v3/tools/fp8_gemv_bench.py` on M=1 GEMV, three variants:

**Memory-bound regime (N=32768, K=8192 → 256 MB weight, L2-overflow):**
- All three variants: **p50 ≈ 1.09 ms, 247 GB/s effective** — that's
  91% of the 273 GB/s LPDDR5X advertised peak. Kernel choice does
  not matter at this size.

**L2-hot regime (N=2048, K=5120 → 10.5 MB weight, fits in L2):**
- `wpr_native` (cvt.rn.f16x2.e4m3x2):      p50 **20.2 µs** / 518 GB/s
- `wpr` (scalar branchless decode):        p50  28.6 µs  / 367 GB/s
- `wpr_lut` (shared-mem LUT):              p50  40.8 µs  / 257 GB/s

Native wins ~2× against LUT when we're compute-bound on the decode
path — a regime entered whenever the weight tile is cache-resident.

**Clock-regime observation:** Over 15 s of continuous looping, SM
clock stayed at ~2510-2527 MHz, power 42-57 W. The 851 → 507 MHz
firmware throttle documented below did **not** trigger on either
workload size with the current driver / firmware combination. Either
the throttle depends on a thermal envelope that micro-benching
doesn't reach, or it was firmware-specific to the PR #28 reporter's
environment. On this board **`WprNative` is the correct default**;
the regime-routing machinery was removed as dead code.

## Power-profile paradox (PR #28 historical context — not reproduced)

**Note — see "Empirical numbers" above.** The clock-regime paradox
below is the original PR #28 reporter's model of GB10 behaviour. We
built the `WprLut` / `WprNative` variants on top of it. On this
DGX Spark (driver 595.58.03, CUDA 13.2) the described 851 → 507 MHz
firmware throttle **did not trigger** during 15 s of continuous
GEMV looping — clocks stayed at ~2520 MHz the whole time.
`WprNative` wins unconditionally on observed hardware. The original
`ClockRegime` / `select_variant` dispatch policy was removed as
unused — a regime router can be added above `Fp8GemvVariant` later
if a firmware revives the plateau.

Original model (PR #28): at 507 MHz throttled, instruction issue
rate caps memory bandwidth utilisation — *more* instructions per
byte can be better because they keep more loads in flight (LPDDR5X
needs 20+ iterations per thread to saturate bandwidth). At 851 MHz
sustained, *fewer* instructions win because less power draw lets
the firmware keep the clock high. The PR #28 reporter measured the
branchless FP8→f32 path holding 851 MHz while the native CVT path
dropped to 507 MHz after ~3 s.

## Memory model

- LPDDR5X is shared; CUDA driver maps user allocations either as
  host-pinned (zero-copy) or device-resident (DMA-mapped LPDDR5X).
- `cuMemAllocManaged` + `cuMemAdvise(cudaMemAdviseSetPreferredLocation,
  device_id)` is the expected path for the runtime.
- There is no separate `HbmArena`; the v3 `rvllm-mem` crate will need a
  `UnifiedArena` sibling (feature-gated behind `gb10`) that allocates
  from the unified pool.

## FA / attention implications

- **No FA3 SM90** — WGMMA and TMA are Hopper-only. GB10 must use an
  FA2-style kernel.
- head_dim 256 (Gemma 4 sliding) already strains the FA2 kernel's
  shared-memory budget; PR #28 halves `FA2_BC` from 64 to 32 to fit
  within sm_121 smem limits. This change is *not* safe to apply
  unconditionally — SM90 path stays on BC=64. A clean port needs an
  arch-conditional tile size.

## Build targets covered by this branch

Commits in the `rusty_sm121` branch (as of the current state):

1. **Build system:** `kernels/build.sh` recognises `sm_121` when
   `nvcc ≥ 13`, producing `kernels/sm_121/*.ptx` alongside the existing
   `kernels/sm_90/*.ptx` tree. Existing SM90 output paths are untouched.
2. **`rvllm-core::arch`:** new `CompileTarget` enum + compute-capability
   mapping `(12, 1) → Sm121`, plus `from_compute_capability` and
   `as_sm_str` helpers. No existing code paths changed.
3. **New kernel sources** (compile for every arch currently in
   `ALL_ARCHS`):
   - `kernels/dequant_fp8.cu` — FP8 → {F16, BF16} in three scale
     variants (none, per-tensor, blockwise). Pure bit-manipulation +
     `__half2float`; works on every target.
   - `kernels/fp8_gemv.cu` — four blockwise-scale GEMV kernels, from the
     baseline through warp-per-row to the native-CVT variant. The
     native-CVT kernel is guarded by `#if __CUDA_ARCH__ >= 1000` so the
     file still builds for `sm_80..sm_90`.
   - `kernels/int4_gemv.cu` — INT4 GPTQ GEMV with per-group asymmetric
     dequant. Scalar loads, single `__syncthreads()` — compatible with
     every target.
4. **`kernels/cast_fp.cu`** grows BF16 siblings for the existing F16
   casts (`cast_f32_to_bf16_kernel`, `cast_bf16_to_f32_kernel`,
   `round_f32_to_bf16_kernel`). Adds to the file without touching the
   existing kernels.

## Landed on this branch

1. **Runtime arch detection** — `CudaContextHandle::compute_capability()`
   drives `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*)`;
   `bring_up::resolve_kernels_dir` maps the pair to a `CompileTarget` and
   selects `kernels/<sm_xxx>/manifest.json`. Unsupported CC or missing
   subdir = hard `ConfigError` at bring-up (no silent fallback).
2. **`UnifiedArena` in `rvllm-mem`** — new module `unified.rs` gated
   behind `feature = "gb10"`, wraps the `HbmArena` bump bookkeeping
   around `cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL)` followed by
   `cuMemAdvise_v2(SET_PREFERRED_LOCATION, device)` so pages stay on
   the GPU side of the unified pool. `Region<'a>` handles stay
   source-compatible with the HBM path.
3. **Arch-conditional `FA2_BC`** — `kernels/flash_attention.cu` picks
   `FA2_BC=32` when `__CUDA_ARCH__ >= 1000` and stays at 64 otherwise,
   behind `#ifndef FA2_BC` so `-DFA2_BC=<n>` still overrides. Verified
   via `nvcc -E`: sm_90 expands to 64, sm_121 to 32.
4. **Manifest SHA-pinning** — `kernels/gen_manifest.sh` is invoked from
   `build.sh` after every per-arch compile loop, writing
   `kernels/<sm_xxx>/manifest.json` with `{revision, arch, entries}`
   keyed by PTX stem. All new kernels auto-included.
5. **`fp8_gemv` kernel variants** — `rvllm-kernels::gb10_dispatch`
   exposes `Fp8GemvVariant { WprLut, WprNative, WprNativeF16In }`
   as pure kernel-variant enum + `entry_point()` + `available_for()`
   arch gate. Entry-point symbols test-pinned against
   `kernels/fp8_gemv.cu`. (The earlier `ClockRegime` +
   `select_variant` machinery was removed after the observed GB10
   clock behaviour made regime-aware routing dead code — see the
   "Power-profile paradox" note below; a regime router can be
   re-added above this module if a future firmware revives the
   plateau.)
6. **CI compile-check** — new `gb10-check` job in
   `.github/workflows/ci.yml` runs `cargo check` + lib tests for
   `rvllm-core`, `rvllm-mem --features gb10`, and `rvllm-kernels`
   (43 tests). No CUDA needed.
7. **Hardware validation on GB10** — two layers:
   a. **Rust bring-up smoke** (`tests/gb10_hw_smoke.rs`, gated on
      `gb10,cuda`, `#[ignore]`): primary-context retain on CUDA 13.2,
      compute-cap probe → `(12, 1)` → `CompileTarget::Sm121`,
      `UnifiedArena` 64 MiB managed alloc + 3 non-overlapping regions.
   b. **FP8-GEMV numerical check** (`v3/tools/fp8_precision_check.py`):
      launches both `wpr_lut` and `wpr_native` against a pure-numpy
      fp64 reference. On GB10: `wpr_lut scale_rel.max 7e-5`,
      `wpr_native 4e-7` — both far under the `1e-3` gate. Axis-bug
      detector proven sensitive via scale-row-flip poison (drives
      `scale_rel` to 5.0, triggers FAIL).
   c. **FA2 decode numerical check** (`v3/tools/fa2_precision_check.py`):
      launches `flash_attention_2_decode_kernel` with the sm_121
      `FA2_BC = 32` arch-conditional tile width against a naive
      fp64 attention reference. On GB10, 4 shape configurations all
      pass at the f32 epsilon floor:
        head_dim=128 ctx=64    scale_rel.max 1.8e-6
        head_dim=128 ctx=512   scale_rel.max 1.4e-6
        head_dim=256 ctx=128   scale_rel.max 1.1e-6   (uses opt-in
                                                       dynamic smem)
        head_dim=256 ctx=1024  scale_rel.max 1.5e-6   (Gemma 4
                                                       sliding realistic)
      Closes the validation gap for the FA2_BC arch change — it was
      only compile-verified before.
8. **FP8-GEMV microbench** (`v3/tools/fp8_gemv_bench.py`): cuda-events
   latency + 10 Hz nvidia-smi sampling. Measured on GB10 (driver
   595.58.03, CUDA 13.2):
   - Memory-bound (256 MB weight, L2-overflow): all three WPR
     variants converge to **~240 GB/s ≈ 88% of LPDDR5X peak**.
   - L2-hot (10.5 MB weight): `wpr_native 20 µs / 518 GB/s` wins 2×
     against `wpr_lut 41 µs / 257 GB/s`.
   - Clocks stay at ~2520 MHz throughout; no firmware throttle
     triggered (see "Power-profile paradox" section for historical
     PR #28 model).

### Decode fast path (sm_121 FP8 GEMV f16-input)

After the core bring-up landed, a second wave of changes replaces the
4 decode projection GEMMs (QKV, O, gate_up, down) with an sm_121-
specialised f16-input FP8 GEMV that keeps the per-channel block
scale in the epilogue. On Gemma 4 31B fp8-block, measured end-to-end
on GB10: **4.7 → 5.1 tok/s (+8.5%)** — and fixes a latent PPL
regression where the `fp8_gemm_channelscale` cuBLASLt heuristic
`LaunchFailed`s on Blackwell consumer and the fallback collapsed to
a scalar weight scale.

9. **Native CVT in FA2-FP8KV decode** — `kernels/flash_attention.cu`:
   `fp8kv_decode_byte` branches on `__CUDA_ARCH__ >= 1000` and uses
   `__nv_cvt_fp8_to_halfraw` (emits `cvt.rn.f16x2.e4m3x2`) on
   Blackwell, branchless scalar elsewhere. Same source, per-arch
   codegen — no Rust-side dispatch.
10. **F16-input FP8 GEMV** — `kernels/fp8_gemv.cu`:
    `fp8_gemv_blockwise_wpr_native_f16in_kernel` mirrors the existing
    `wpr_native` kernel but consumes f16 activations and emits f16
    output. Native `cvt.f32.f16` on every load; 8 halves per 2×u64
    input read; `Fp8GemvF16InLaunch` in `rvllm-fused` is the
    launcher. Enables skipping activation FP8-quant on the M=1
    decode path — the kernel reads f16 straight from rmsnorm (QKV /
    gate_up via `scratch.delta_f16`, O from `scratch.attn_out`, down
    from GELU f16 scratch) and preserves the per-channel block
    scale that the Sm121 cuBLASLt fallback drops.
11. **F16-input fused norm+residual epilogue** —
    `v3/kernels/fused_norm_add_residual_f16.cu` gains a
    `_f16in_kernel` variant that reads f16 input + no channelscale
    broadcast (we already applied it in the GEMV). Wired into the
    O-proj and down-proj fast paths.
12. **FA2 FP8-KV prefill for sm_121** —
    `kernels/flash_attention.cu` adds
    `flash_attention_2_prefill_fp8kv_kernel` (BC=32) and
    `flash_attention_2_prefill_fp8kv_bc16_kernel` (BC=16 for
    head_dim=512 smem budget). Multi-query causal FP8 prefill with
    per-tensor descales, f16 output. `PagedPrefillFp8Launcher`'s
    Fa2Ptx arm wires them — required after upstream ac72222 split
    prompt prefill into its own phase. Probe TTFT on 8-tok prompt:
    213-216 ms.

### Upstream fix bundled with this branch

- `CudaContextHandle::init` uses `cuDevicePrimaryCtxRetain` +
  `cuCtxSetCurrent` instead of `cuCtxCreate_v2`. cudarc 0.19 only
  cfg-wraps `cuCtxCreate_v2` for CUDA 11.07..12.09, so the legacy
  symbol is missing on CUDA 13. Primary-context retain is ABI-stable
  across CUDA 11/12/13, unblocking the whole `feature = "cuda"`
  path on CUDA 13 — not just the GB10 work. No behavioural change
  on SM90: rvllm always holds one context for the process lifetime.

## Remaining follow-ups (explicitly NOT on this branch)

- ~~Wire `gb10_dispatch::Fp8GemvVariant` into the runtime launch
  path~~: landed (decode fast path #9–11 above).
- ~~Select `UnifiedArena` vs `HbmArena` in `Bringup::load` on
  `CompileTarget`~~: landed (bring-up section #1–2 above).
- End-to-end Gemma-4 PPL on sm_121 — probe run validates 32-token
  decode numerically, a full PPL sweep is the next measurement
  milestone.
- **Native sm_121 FP8 GEMM with row×column-scale epilogue**. The
  current `fp8_gemm_channelscale_or_fallback` on sm_121 falls back to
  `cublaslt.fp8_gemm(..., a_scale, b_scale_scalar)` because cuBLASLt's
  FP8 channelscale heuristic `LaunchFailed`s on Blackwell consumer —
  that drops the per-channel weight scale and costs PPL on the
  prefill / M>1 path. Decode M=1 is already fixed by the f16-input
  GEMV fast path. A proper fix needs its own scoped PR:

    * `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` with multi-stage cp.async
      tiled K/V loads (Blackwell-native FP8 tensor-core MMA).
    * Row×column-scale epilogue (a_scale[M] × b_blockscale[N_blk, K_blk]).
    * Numerical validation against a fp64 reference (new
      `v3/tools/fp8_gemm_channelscale_check.py`) before the perf pass.
    * Tile-shape autotune across the 4 Gemma 4 projection shapes
      (QKV 4608×5376, O 5376×3584, gate_up 41152×5376,
      down 5376×20576).
    * Microbench against the scalar cuBLASLt fallback to confirm
      both **lower PPL** and **not-slower prefill TTFT** — the
      intermediate "extend the f16-input GEMV to M>1" shortcut was
      tried on this branch and regressed prefill ~7× (GEMV parallelism
      collapses at M=8 vs tensor-core mma), so the replacement must
      be real mma.sync, not an extended GEMV.
    * Scope estimate: 2-3 days, ~500-1000 LOC CUDA + validation +
      bench + wiring. Explicitly out of scope for this branch.
- Switch `kernels/build.sh` to `-arch=sm_121f` (family-mode) when a
  future kernel needs FP8 tensor-core MMA or MXFP4 hardware paths.
  Not required for anything on this branch (see "Arch suffix").
- ~~Fix the two pre-existing upstream bugs that blocked
  `cargo check --workspace`~~: `PagedDecode/PrefillParams.window_size_left`
  landed upstream. The three `Gemma4Bringup::{run_ppl, run_bench,
  run_generate}` "missing" methods actually existed (gated
  `#[cfg(feature = "cuda")]`); the bin [[bin]] entries now carry
  `required-features = ["cuda"]` so default `cargo check --workspace`
  skips them cleanly. Also replaced `rvllm_eval.rs`'s inline
  `cuCtxCreate_v2` probe (CUDA 13 resolution failure) with
  `CudaContextHandle::init` — same primary-context-retain fix as
  in `rvllm-mem::context`.

## Non-goals (explicitly not in this branch)

- **No cudarc downgrade.** PR #28 rolled back 0.19 → 0.12 as a
  CUDA-12.6 workaround; that regression is not part of sm_121 support.
- **No CUTLASS deletion.** SM90 CUTLASS path stays intact for H100/H200.
  sm_121 simply has no CUTLASS coverage; it rides on the custom PTX
  kernels introduced here.
- **No Qwen3.5 / Mamba2 scope.** `attn_gate.cu` and `mamba2_ssm.cu` from
  PR #28 are for a separate feature track and are not pulled in here.
- **No `flash_attention_impl.rs` refactor.** PR #28 migrated the
  existing FA3 wrapper to an older cudarc API; that's orthogonal to
  sm_121 and stays out of this branch.
