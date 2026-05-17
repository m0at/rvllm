# SM75 Support

SM75 covers Turing GPUs such as NVIDIA T4 and RTX 6000 Turing. In rvLLM this
is a compatibility target only: the core arch layer can name `sm_75`, map
compute capability 7.5, and report unsupported fast paths, but runtime dispatch
and kernel artifacts are not wired.

## Current Status

- `CompileTarget::from_compute_capability(7, 5)` maps to `Sm75`.
- The manifest arch string is `sm_75`.
- Expected manifest location, once kernels exist, is `kernels/sm_75/manifest.json`.
- SM75 does not support rvLLM FP8 tensor-core routes.
- SM75 does not support the current W4A8 CUTLASS path.
- SM75 does not support FA3. FA3 requires Hopper SM90 WGMMA/TMA.
- SM75 does not use the current FA2 PTX backend, which is SM121-specific today.
- RotorQuant KV has no GPU kernels on any target yet.

## Limits

T4 is a 16 GB GDDR6 device, and RTX 6000 Turing is commonly 24 GB. These cards
do not have native FP8 tensor cores, WGMMA, or TMA. Treat SM75 as an
FP16/INT8-era architecture unless a Turing-specific kernel path is added and
measured.

Do not enable these paths on SM75:

- FP8 tensor-core GEMM or attention.
- H100 W4A8 CUTLASS shared objects.
- FA3 paged attention.
- GB10/SM121 FA2 PTX assumptions.
- RotorQuant KV compression/decompression.

## Kernel Audit

Static scan result for this branch:

- `kernels/cutlass_fp8_gemm*.cu`, `kernels/cutlass_*_autotune.cu`, and
  `kernels/cutlass_w4a8_wrapper.cu` are CUTLASS tensor-op paths targeting
  Hopper or newer assumptions. They are not SM75 candidates.
- `kernels/fa3_sm90_wrapper.cu` and `kernels/flash_attention_3*.cu` are FA3
  Hopper paths and stay rejected on SM75.
- `kernels/fused_rmsnorm_fp8_quant.cu`, `kernels/fused_silu_fp8_quant.cu`, and
  FP8 KV rope/cache kernels use CUDA FP8 storage/conversion types. They are not
  SM75 serving candidates.
- `kernels/flash_attention.cu` contains an FA2-style path, but the runtime
  currently wires that PTX backend for SM121 only. It must not be advertised as
  an SM75 fallback until it is compiled, manifest-listed, and tested for Turing
  shared-memory limits.
- Plain fp16/f32 elementwise fused kernels in `v3/kernels` are the likely first
  SM75 manifest candidates, subject to an actual `-arch=sm_75` build.

## FA2 Fallback Plan

The first viable SM75 attention path should be a separate compatibility backend:

1. Compile `flash_attention.cu` or a trimmed derivative with `-arch=sm_75`.
2. Use FP16 Q/K/V and FP16 or FP32 accumulators first; avoid FP8 KV on Turing.
3. Reduce tile sizes until static shared memory stays under T4 limits.
4. Add `kernels/sm_75/manifest.json` entries and make runtime selection require
   `CompileTarget::Sm75`.
5. Validate with synthetic attention parity, then Gemma 31B PPL smoke, then
   decode throughput. Only after those pass should SM75 attention move out of
   no-go.

## Requirements Before Enabling

1. Build a `kernels/sm_75` PTX tree with `-arch=sm_75`.
2. Generate `kernels/sm_75/manifest.json` with `arch: "sm_75"` and entries for
   every fused kernel the runtime will load.
3. Convert the static audit above into a passing compile matrix.
4. Add an SM75 attention backend. The likely first path is the FA2-style
   compatibility kernel using FP16 KV, with reduced tile sizes and no Hopper
   features.
5. Add GEMM fallbacks that are valid on Turing, such as cuBLAS/cuBLASLt FP16 or
   a measured INT8/INT4 path built specifically for SM75.
6. Add runtime guardrails so unsupported features fail via
   `CompileTarget::require_feature` before loading incompatible artifacts.
7. Validate on actual T4 hardware: manifest load, fused kernels, attention
   correctness, memory budget, PPL smoke, and decode throughput.

Until those are complete, SM75 support should be described as probeable but not
serving-ready.
