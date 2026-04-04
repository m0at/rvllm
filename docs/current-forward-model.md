# Current Forward Model (April 4, 2026)

This is the current high-level forward-path model for `rvLLM`, not the older pre-fusion trace.

The two important ideas now are:

1. `T=1` and `T>=2` are intentionally different execution regimes.
2. The runner makes an explicit per-op GEMM policy choice instead of letting CUTLASS availability accidentally choose the path.

## Current Path Selection

### Batch-1 (`T=1`)

Normal batch-1 decode now defaults to:

```text
CublasGemvDecode
```

That was a deliberate fix. The previous default still favored the older fused decode path, which lost on the real H100 end-to-end benchmark.

The batch-1 selection order is now:

```text
explicit experimental env paths
-> FP8 decode if FP8 weights are active
-> CublasGemvDecode (default normal path)
-> legacy FusedDecode only if forced
```

Current verified number on H100 / Qwen2.5-7B / `output-len=128`:

- `N=1`: `127.9 tok/s`

### Batched (`T>=2`)

Batched prefill and batched decode use the normal layer stack plus an explicit GEMM policy.

Current default:

```text
GemmStrategy::Hybrid
```

Hybrid means:

```text
QKV        -> cuBLAS / cublasLt
O-proj     -> cuBLAS / cublasLt
GateUp     -> CUTLASS
SiLU       -> fused with GateUp CUTLASS epilogue
Down-proj  -> cuBLAS / cublasLt
```

The old bug was that the runner conceptually wanted this hybrid policy, but the actual enum and dispatch did not encode it cleanly. As a result, QKV could still wander onto CUTLASS just because the shared library was present.

That is fixed now.

## Current Layer Shape

### Batch-1 normal decode

Per layer:

```text
RMSNorm
QKV projection via cuBLAS / cublasLt
RoPE + KV cache write
attention decode
O-proj via cuBLAS / cublasLt
RMSNorm
gate_up via cuBLAS / cublasLt
SiLU * Mul
down via cuBLAS / cublasLt
```

### Batched decode / prefill

Per layer:

```text
RMSNorm
QKV via cuBLAS / cublasLt
bias / layout handling as needed
RoPE + cache update
attention backend
O-proj via cuBLAS / cublasLt
residual + RMSNorm
GateUp + SiLU via CUTLASS
down via cuBLAS / cublasLt
```

## Current Benchmark Truth

Same H100, same Qwen2.5-7B snapshot, `output-len=128`, direct engine:

| N | vLLM 0.19.0 | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 165.5 | 127.9 | 0.77x |
| 32 | 4467.7 | 4407.5 | 0.99x |
| 64 | 7972.1 | 7964.0 | 1.00x |
| 128 | 13903.5 | 13148.3 | 0.95x |

## What Is Still Behind

- `N=1` decode is still materially behind vLLM.
- `N=32` and `N=64` are basically tied now.
- `N=128` is close, but still a few percent behind.
- `cublasLt` autotune cache behavior is still flaky on some shapes and should fall back more aggressively when a cached algo goes bad.

## Relevant Controls

```bash
RVLLM_CUBLAS_DECODE=0|1
RVLLM_BATCHED_GEMM_STRATEGY=cublas|hybrid|cutlass
RVLLM_PERSISTENT_V3=1
RVLLM_FP8_WEIGHTS=1
```

## Bottom Line

The current system is no longer “fused decode by default, CUTLASS when available.”

It is:

- batch-1 normal decode on `CublasGemvDecode`
- batched execution on an explicit hybrid policy
- experimental persistent and megakernel paths kept separate from the normal path
