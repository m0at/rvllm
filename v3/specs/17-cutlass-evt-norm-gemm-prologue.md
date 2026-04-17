# Spec 17 — CUTLASS EVT norm+quant+GEMM prologue (QKV)

**Status:** Planned. Not implemented.
**Expected win:** 3–5% decode throughput at N=128.
**Effort:** Real engineering — custom CUTLASS, no cuBLASLt equivalent.

## Problem

Today each transformer layer does these steps back-to-back at the entry:

```
1. fused_add_rmsnorm_fp8_quant  (residual_f16, gamma) -> hidden_fp8, hidden_scale
2. cublasLt FP8 GEMM (bias epilogue)  hidden_fp8 @ qkv_weight -> qkv_out_f16
```

Two kernel launches. The first step's ONLY consumer is the second step's A
input. Its output (`hidden_fp8`) is written to HBM, immediately read back
by the GEMM. Wasted bandwidth and one wasted dispatch per layer.

At N=128 on Qwen2.5-7B (28 layers), that's 28 launches per decode step
that could be folded into the GEMM's prologue.

## Why cuBLASLt can't do this

cuBLASLt's prologue in CUDA 12.x only exposes pre-epilogue scaling
(per-tensor / per-row D scale) and bias add. It does not support a
per-row RMSnorm + per-row FP8 quantize as a prologue on A. This is a
hard limitation of the library.

## Approach — custom CUTLASS 3.x SM90 EVT

CUTLASS 3.x on Hopper has an "EVT" (Epilogue Visitor Tree) framework
that also applies to the *mainloop's load path*. We can attach a
prologue-visitor that:

- Reads raw f16 residual
- Adds f16 residual-in (pre-norm residual stream)
- Runs RMSnorm across hidden dim (warp-reduce sum-of-squares → rsqrt)
- Multiplies by gamma (f16)
- Divides by per-row max-abs → FP8 with per-row scale
- Feeds the FP8 values directly into TMA + WGMMA for the QKV matmul
- Writes the per-row scale to a scratch buffer (needed by FA3 later)

In one kernel: **RMSnorm + quantize + QKV GEMM + bias epilogue.** Five
composed ops in one launch. The output is still `qkv_out` in f16 (or
FP8 with the D-scale path from spec 18 if combined).

## Why this is the "real" engineering

- Requires a new CUTLASS kernel file (cannot use an existing instantiation).
- The prologue visitor must compute a per-row reduction BEFORE the
  mainloop starts on that row (needs a warp-sync barrier — doable but
  nontrivial in CUTLASS 3.x).
- Autotuning: the EVT prologue adds register pressure, which can bump
  the GEMM's optimal tile out of what our current variants cover.
  Expect to need to re-autotune tile shapes once this lands.
- Testing: needs a pure-Rust f32 reference (mirrors our existing
  `fused_add_rmsnorm_fp8_quant_ref`) plus a tight cosine tolerance vs the
  CUTLASS kernel. The rmsnorm reduction in FP8-quantize-land is sensitive
  to per-row scale, so numerical verification is non-trivial.

## Files to write

- `kernels/cutlass_fp8_gemm_rmsnorm_prologue.cu`  — new CUTLASS kernel.
- `kernels/cutlass_fp8_gemm_rmsnorm_prologue.cuh` — the EVT visitor tree.
- `v3/crates/rvllm-cutlass/src/prologue.rs` — Rust dispatch wrapper.
- `v3/crates/rvllm-runtime/src/layer_exec.rs` — swap `fused_add_rmsnorm_fp8_quant` + `cublasLt.fp8_gemm_bias(QKV)` → single `cutlass_fp8_gemm_rmsnorm_bias(QKV)` call.

## Prerequisite

Spec 18 (cuBLASLt FP8 D / FA3 FP8-O) should land first so the output
data type of this fused kernel can be FP8 directly (eliminating another
layer output f16→FP8 roundtrip down the pipeline).

## Verification

- Pure-Rust reference: compose `fused_add_rmsnorm_fp8_quant_ref` + a
  toy FP8 GEMM reference; cosine ≥ 0.999 against the CUTLASS kernel.
- Bench: compare N=128 decode tok/s before and after. Target: +3–5%.
- Layer output check: first 16 sampled tokens match f16-KV + split-kernel
  baseline on a representative prompt.

## Out of scope

- Post-MLP pre-norm fusion (the second fused_add_rmsnorm_fp8_quant, at
  layer step 9 in the current pipeline). That's a separate but
  analogous spec — gate_up GEMM with the same EVT prologue.
- Applying this to O-proj or down-proj (both of which are residual-fused
  and don't have a pre-norm on their input).
