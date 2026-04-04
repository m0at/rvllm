# Benchmark History

This file starts with the current public benchmark truth, then keeps older numbers only as historical context.

## Current Public Comparison (April 4, 2026)

Model: Qwen2.5-7B f16
GPU: H100 SXM 80GB
Harness: direct engine
Decode length: `output-len=128`

### vLLM 0.19.0 vs rvLLM

| N | vLLM 0.19.0 tok/s | rvLLM tok/s | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 165.5 | 127.9 | 0.77x |
| 32 | 4467.7 | 4407.5 | 0.99x |
| 64 | 7972.1 | 7964.0 | 1.00x |
| 128 | 13903.5 | 13148.3 | 0.95x |

### What changed to get here

Two fixes matter most:

1. **Batch-1 dispatch fix**
   - normal `T=1` decode now defaults to `CublasGemvDecode`
   - this lifted the current normal batch-1 path to `127.9 tok/s`

2. **Batched GEMM policy fix**
   - `GemmStrategy::Hybrid` is now real instead of half-implied
   - current hybrid policy is:
     - QKV: cuBLAS / cublasLt
     - O-proj: cuBLAS / cublasLt
     - GateUp + SiLU: CUTLASS
     - Down-proj: cuBLAS / cublasLt

### Explicit batched strategy sweep

On the same H100 for `N=64`, `output-len=128`:

| Strategy | tok/s |
|---|---:|
| `cublas` | 7965.6 |
| `hybrid` | 8193.3 |
| `cutlass` | 7830.4 |

That sweep is why `Hybrid` is the current default when CUTLASS is available.

## Current Read of the Gap

- `N=1`: still materially behind vLLM
- `N=32`: basically tied
- `N=64`: effectively tied
- `N=128`: still a few percent behind

The remaining issues are no longer “wrong path” bugs in the normal batched stack. The biggest remaining work is:

- better single-stream decode
- safer `cublasLt` autotune fallback when cached algos go bad
- a few more percent at `N=128`

## Historical Context

Older measurements below used different harnesses, older vLLM versions, or pre-fix architecture. Keep them as optimization history, not as the current headline.

### Earlier direct-engine comparison vs vLLM 0.6.3

| N | stock vLLM 0.6.3.post1 | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 133.7 | 120.6 | 0.90x |
| 4 | 543.3 | 427.9 | 0.79x |
| 8 | 926.1 | 845.8 | 0.91x |
| 16 | 1934.5 | 1648.9 | 0.85x |
| 32 | 3197.1 | 3170.0 | 0.99x |

### Earlier H100 direct-engine peak

This was a useful optimization waypoint, but not the current apples-to-apples comparison:

| N | rvLLM tok/s |
|---:|---:|
| 128 | 12312 |

### Earlier lifecycle / HTTP numbers

Those runs were useful for separating direct-engine performance from serving-stack overhead, but they were not re-run against `vLLM 0.19.0` and should not be treated as the current public baseline.
