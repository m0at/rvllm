# rvLLM Architecture (April 4, 2026)

This is the current architecture summary after the April 4 dispatch and GEMM-policy fixes.

## End-to-End Flow

```text
HTTP / benchmark request
-> scheduler builds mixed prefill + decode batch
-> gpu_worker uploads metadata
-> gpu_runner selects forward path
-> model layers execute on GPU
-> lm head + sampling / argmax
-> token ids copied back
-> engine updates sequences
```

The dedicated GPU thread still owns CUDA state, graph capture, replay, and the runner.

## The Important Split: `T=1` vs `T>=2`

### `T=1`: batch-1 normal decode

The normal path is now:

```text
CublasGemvDecode
```

That was not true before. The old default still used the legacy fused decode path, and that was one of the main reasons the public docs were wrong.

### `T>=2`: batched decode and prefill

The normal path is now an explicit per-op hybrid GEMM policy:

```text
QKV / O / down  -> cuBLAS or cublasLt
GateUp + SiLU   -> CUTLASS
```

This is the current best policy on H100 for Qwen2.5-7B.

## What Was Wrong Before

Two architectural problems were masking the real performance picture:

1. **Wrong batch-1 default**
   - The standard `T=1` path still went through the older fused decode stack.
   - Fix: normal `T=1` now defaults to `CublasGemvDecode`.

2. **Half-wired batched hybrid policy**
   - The runner wanted a hybrid strategy, but the actual enum and dispatch did not enforce one.
   - CUTLASS presence could change QKV routing even when that was not the intended policy.
   - Fix: `GemmStrategy::Hybrid` is now real and stable.

## Current Layer Stack

### Batch-1 normal decode

```text
RMSNorm
QKV projection
RoPE + KV cache write
attention decode
O-proj
RMSNorm
gate_up
SiLU * Mul
down
```

All projection-heavy ops are handled through the cuBLAS-backed batch-1 path now.

### Batched decode / prefill

```text
RMSNorm
QKV via cuBLAS / cublasLt
RoPE + cache update
attention backend
O-proj via cuBLAS / cublasLt
residual + RMSNorm
GateUp + SiLU via CUTLASS
down via cuBLAS / cublasLt
```

## Current Comparison vs vLLM 0.19.0

Same H100, same Qwen2.5-7B snapshot, direct engine, `output-len=128`:

| N | vLLM 0.19.0 | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 165.5 | 127.9 | 0.77x |
| 32 | 4467.7 | 4407.5 | 0.99x |
| 64 | 7972.1 | 7964.0 | 1.00x |
| 128 | 13903.5 | 13148.3 | 0.95x |

So the architecture story is now:

- single-stream decode still needs work
- batched decode is nearly there
- the current public docs should be talking about the batched hybrid stack, not the old fused-default story

## Relevant Files

- `crates/rvllm-model-runner/src/gpu_runner.rs`
- `crates/rvllm-model-runner/src/gpu_layer/mod.rs`
- `crates/rvllm-model-runner/src/gpu_layer/batched.rs`
- `crates/rvllm-worker/src/gpu_worker.rs`

## Remaining Work

- make the `cublasLt` autotune cache degrade safely on bad cached algos
- keep closing the `N=1` gap
- improve the `N=128` gap without regressing `N=32/64`
