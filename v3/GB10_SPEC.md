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

## Power-profile paradox

At 507 MHz throttled, instruction issue rate caps memory bandwidth
utilisation. In this regime, *more* instructions per byte can be better
because they keep more loads in flight (LPDDR5X needs 20+ iterations
per thread to saturate bandwidth). At 851 MHz sustained, *fewer*
instructions win because less power draw lets the firmware keep the
clock high. Kernel selection is therefore clock-regime-dependent.

Measured baseline (PR #28 quotes): the branchless FP8→f32 path holds
851 MHz, whereas the native CVT path drops to 507 MHz after ~3 s.

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

## Still TODO (not in this branch yet)

- [x] ~~Runtime arch detection~~ — `CudaContextHandle::compute_capability()`
      drives `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*)`;
      `bring_up::resolve_kernels_dir` maps the pair to a `CompileTarget`
      and selects `kernels/<sm_xxx>/manifest.json`. Unsupported CC or
      missing subdir = hard `ConfigError` at bring-up (no silent fallback).
- [ ] `UnifiedArena` in `rvllm-mem` behind `feature = "gb10"` — the
      current `HbmArena` assumes dedicated HBM.
- [x] ~~Arch-conditional `FA2_BC`~~ — `kernels/flash_attention.cu`
      picks `FA2_BC=32` when `__CUDA_ARCH__ >= 1000` (sm_100/sm_121/sm_122)
      and stays at 64 otherwise. Wrapped in `#ifndef FA2_BC` so an
      explicit `-DFA2_BC=<n>` on the nvcc command line still wins.
      Verified via `nvcc -E`: sm_90 expands to 64, sm_121 to 32.
- [x] ~~Manifest SHA-pinning~~ — `kernels/gen_manifest.sh` is invoked
      from `build.sh` after every per-arch compile loop, writing
      `kernels/<sm_xxx>/manifest.json` with `{revision, arch, entries}`
      keyed by PTX stem (matches `KernelLoader::load_ptx(name)`).
      All new kernels (`fp8_gemv`, `int4_gemv`, `dequant_fp8`) are
      auto-included.
- [ ] GB10-specific dispatch policy in `fp8_gemv`: pick between
      `wpr_lut` (throttled) and `wpr_native` (sustained) based on the
      measured clock or a time-window heuristic.
- [ ] CI: `cargo check --workspace` cross-compile for `sm_121` target
      gated by `feature = "gb10"` (compile-check only; no hardware in
      CI).
- [ ] Bench harness: `rvllm-bench` should log `clocks.sm` +
      `power.draw` at 1 Hz during a run so the clock-regime paradox
      shows up in numbers, not just comments.
- [ ] Validation: run `fp8_precision_check.py` on GB10 hardware to
      confirm the FP8 scale tensor axis — the open GPU PPL bug (v3/SPEC)
      likely interacts with sm_121 in unknown ways.

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
