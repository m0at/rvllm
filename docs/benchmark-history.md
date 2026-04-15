# Benchmark History

This file starts with the current public benchmark truth, then keeps older numbers only as historical context.

## Verified Baseline (April 15, 2026 -- deploy6.log)

Model: Qwen2.5-7B FP8 E4M3
GPU: H100 SXM 80GB (ssh8.vast.ai:13302)
Harness: rvllm-v2-bench (direct engine, no HTTP)
Commit: 2678aaef2 (reverted to pre-experimental baseline)
Decode length: `output-len=512`, greedy (temperature=0), 3 iterations, CUDA graphs enabled

### rvLLM v2 FP8 -- 512 output tokens (verified)

| N | tok/s | stdev |
|---:|---:|---:|
| 1 | 149.1 | 0.0 |
| 4 | 582.7 | 0.2 |
| 8 | 1,163.6 | 1.8 |
| 16 | 2,345.0 | 3.6 |
| 32 | 4,434.3 | 3.0 |
| 64 | 11,240.0 | 7.1 |
| 128 | 19,259.3 | 33.1 |

### What changed since April 14

1. **FlashAttention-3 SM90 paged-KV decode** -- WGMMA/TMA-accelerated via shared library (libfa3_kernels.so), 7.5us/layer
2. **Revert of experimental changes** -- commits 072d6dffc (FP8FastAccum + CPU hot-path) and 238b9f3a7 (FP8 LM head GEMM) were reverted because 238b9f3a7 caused a massive N=128 regression (19K -> 10K tok/s)
3. **Cherry-pick 4bd92c789** -- re-applies only 072d6dffc (FP8FastAccum Pingpong schedule + CPU hot-path elimination) without the FP8 LM head regression. Benchmark pending.

### Current read of the gap

- rvLLM individual kernels are 2-6x faster per-call (SiLU 3.1x, RMSNorm 2.3x, FA3 6.2x)
- Throughput gap is structural, not kernel-level
- Both engines use the same CUTLASS cutlass_3x_gemm_sm90_fp8 kernel family
- Remaining bottlenecks: F16 LM head GEMM, extra FP8 quantization passes, per-step metadata HtoD, stream.synchronize()

---

## Internal Benchmark (April 14, 2026)

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

---

## Earlier Public Comparison (April 7, 2026) -- RETRACTED

**Note**: This comparison ran vLLM with F16 eager mode (no CUDA graphs, no torch.compile, no FP8). It was not a fair apples-to-apples comparison. Kept here for historical context only. See the April 15 numbers above for the honest comparison.

Model: Qwen2.5-7B f16
GPU: H100 SXM 80GB

| N | vLLM 0.19.0 tok/s | rvLLM tok/s | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 167.5 | 132.7 | 0.79x |
| 32 | 4964.2 | 4494.9 | 0.91x |
| 64 | 9312.6 | 8503.4 | 0.91x |
| 96 | 13085.9 | 10530.6 | 0.80x |
| 128 | 16825.3 | 13718.1 | 0.82x |

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
