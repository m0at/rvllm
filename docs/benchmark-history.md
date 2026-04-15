# Benchmark History

This file starts with the current public benchmark truth, then keeps older numbers only as historical context.

## Latest Internal Benchmark (April 14, 2026)

Model: Qwen2.5-7B FP8 E4M3
GPU: H100 SXM 80GB
Harness: rvllm-v2-bench (direct engine, no HTTP)
Decode length: `output-len=512`

### FP8 Optimization Gains

Three kernel optimizations measured against the April 12 v2 FP8 baseline:

| N | Before (tok/s) | After (tok/s) | Gain |
|---:|---:|---:|---:|
| 1 | 130.0 | 145.5 | +11.9% |
| 32 | 3,917.0 | 4,356.3 | +11.2% |
| 64 | 9,743.7 | 10,990.8 | +12.8% |
| 128 | 16,641.4 | 19,137.3 | +15.0% |

### What changed

1. **Vectorized fused SiLU*mul + FP8 quantize** -- 128-bit loads (uint4, 8 halves per load) and 64-bit FP8 stores (uint2, 8 FP8 values per store). Register caching eliminates second-pass global memory reads.

2. **Stream-K / split-K FP8 GEMM autotune variants (v25-v31)** -- CUTLASS StreamKScheduler decomposes K-dimension work across SMs. Split-K=4 on Down projection (M=64,N=3584,K=18944): 28.5% speedup (65.4us -> 46.8us) by going from 28 threadblocks on 132 SMs (21% utilization) to 112 threadblocks (85% utilization).

3. **FP8FastAccum schedule aliases** -- CUTLASS KernelTmaWarpSpecializedFP8FastAccum, Cooperative, and Pingpong variants in autotune pool (v15-v24).

### FP8 Autotune Cache

32 FP8 GEMM variants total. Standalone autotune binary benchmarks all variants per shape. Cache results (36 entries):

- Down projection (K=18944): split-K=4 (v29) wins for most M values, stream-K (v25) for some
- Gate+Up (K=3584): v0/v5 wins (already high SM occupancy, splitting doesn't help)
- O-proj (K=3584): v0 wins
- QKV (K=3584): v0 wins

### Note on methodology

These numbers use `output-len=512` vs the April 12 public comparison which used `output-len=128`. The percentage gains carry over (they're per-decode-step improvements) but the absolute tok/s numbers are not directly comparable to the head-to-head vLLM table.

---

## Current Public Comparison (April 7, 2026)

Model: Qwen2.5-7B f16
GPU: H100 SXM 80GB
Harness: direct engine
Decode length: `output-len=128`

### vLLM 0.19.0 vs rvLLM

| N | vLLM 0.19.0 tok/s | rvLLM tok/s | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 167.5 | 132.7 | 0.79x |
| 32 | 4964.2 | 4494.9 | 0.91x |
| 64 | 9312.6 | 8503.4 | 0.91x |
| 96 | 13085.9 | 10530.6 | 0.80x |
| 128 | 16825.3 | 13718.1 | 0.82x |

### What changed to get here

Two things matter most:

1. **Batch-1 default-path fix**
   - normal `T=1` decode now defaults to the reusable `Batched` path
   - this is still the right architecture change even though current `N=1` is behind vLLM

2. **Batched GEMM policy fix**
   - `GemmStrategy::Hybrid` is now real instead of half-implied
   - current hybrid policy is:
     - QKV: cuBLAS / cublasLt
     - O-proj: cuBLAS / cublasLt
     - GateUp + SiLU: CUTLASS
     - Down-proj: cuBLAS / cublasLt

### Important correction

The earlier `89f`-era "rvLLM beats vLLM at `N=64`" claim is no longer treated as valid.

- the fast `89f` H100 run was real
- but that path was fast because the CUTLASS gate-aux FFN branch skipped the FFN down-projection
- the archived fast H100 CUTLASS library is still kept in the repo for forensic reproducibility

So the current public baseline is the clean current-`main` table above, not the older `9589 tok/s` claim.

### Earlier explicit batched strategy sweep

On the same H100 for `N=64`, `output-len=128`:

| Strategy | tok/s |
|---|---:|
| `cublas` | 7965.6 |
| `hybrid` | 8193.3 |
| `cutlass` | 7830.4 |

That sweep is why `Hybrid` is the current default when CUTLASS is available.

## Current Read of the Gap

- `N=1`: materially behind vLLM
- `N=32`: closer, but still behind
- `N=64`: still behind
- `N=128`: still behind by a wider margin

The biggest remaining work is:

- better single-stream decode
- a correct fast Hopper FFN path that does not skip work
- safer `cublasLt` autotune fallback when cached algos go bad
- more efficiency at `N=64` and `N=128`

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
