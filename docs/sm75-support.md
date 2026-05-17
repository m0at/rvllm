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

## Requirements Before Enabling

1. Build a `kernels/sm_75` PTX tree with `-arch=sm_75`.
2. Generate `kernels/sm_75/manifest.json` with `arch: "sm_75"` and entries for
   every fused kernel the runtime will load.
3. Audit every fused kernel for unsupported CUDA intrinsics, FP8 types, and
   arch guards.
4. Add an SM75 attention backend. The likely first path is an FA2-style
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
