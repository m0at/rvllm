# Best rvllm Configurations — sm_121 (GB10)

Two tracks, five each, ranked **stability first, performance second**.
All assume Gemma 4 31B fp8-block weights, 14k–17k token zeroclaw-style
prompts, `RVLLM_ARENA_GB=60`, `RVLLM_BATCH_PREFILL=1`,
`RVLLM_UNIFIED_PREFILL=1`, `RVLLM_UNIFIED_PREFILL_MMA=1`,
`RVLLM_FP8_GEMM_CUTLASS_SM120=1`,
`RVLLM_RESIDUAL_BF16=1` (defaults that are always on for both tracks).

Only the **delta** vs the always-on baseline is shown per config.

---

## NVFP4 — Top 5

### N1. SOTA Production (current default)

The configuration that landed pure NVFP4 with WHO + WEATHER + HA all clean.

```env
RVLLM_NVFP4_HYBRID_GLOBAL_FP8=0
RVLLM_NVFP4_HYBRID_SLIDING_FP8=0
RVLLM_NVFP4_HADAMARD=1
RVLLM_NVFP4_HADAMARD_V=1
RVLLM_NVFP4_K_SCALE_POLICY=amax6
RVLLM_NVFP4_V_SCALE_POLICY=mse        # 6-candidate (cycle 56 step 12)
RVLLM_PER_TOKEN_Q_SCALE=1
RVLLM_Q_SCALE=2.0
RVLLM_NVFP4_SPLIT_KV=1
RVLLM_PREFILL_CHUNK_SIZE=128
RVLLM_TOOL_CALL_OPEN_BIAS=4.0
RVLLM_REPETITION_PENALTY=1.05
RVLLM_REPETITION_PENALTY_WINDOW=64
RVLLM_REPETITION_PENALTY_MIN_COUNT=2
```

**Memory**: ~50 % of FP8 KV. **Quality**: 5/5 smoke clean (WHO, WEATHER,
HA tool call, German poetry, time). **Speed**: ~3.5 tok/s decode, ~125 s
prefill on 14.7 k tokens (chunk=128; vs ~95 s @ chunk=256 with the
2-cycle quality cost). **Use this as the default.**

### N2. Conservative Long-Ctx (no V-Hadamard)

Fall-back if V-Hadamard quality regresses on a new prompt class. Loses
the cycle-56-step-12 V-rotation precision boost but keeps everything
else.

```env
# … N1 above, but flip:
RVLLM_NVFP4_HADAMARD_V=0
```

**Quality**: WHO + WEATHER reproducibly clean; HA tool call may variably
go off-topic. **Speed**: minor win (skips the `hadamard_unrotate_f16`
kernel per layer, ~1 % decode speedup).

### N3. Sliding-FP8 hybrid (NVFP4-on-globals only)

Strict pure-NVFP4 sacrificed for maximum stability across prompt types.
50 sliding layers go FP8 (per-token scale, no microscale block); the
10 global layers stay NVFP4. Memory cost: ~+25 % vs N1.

```env
# … N1 above, but flip:
RVLLM_NVFP4_HYBRID_SLIDING_FP8=1
RVLLM_NVFP4_HADAMARD_V=0   # V-rotation isn't a clear win when sliding is FP8
```

**Quality**: most stable historically (was production until iter 45).
**Use when** introducing a new client persona or test bench that the
NVFP4 pure path hasn't been validated against.

### N4. Performant balanced

Drops V-rotation (skip per-layer unrotate kernel) and the
repetition-penalty DtoH/HtoD round-trip; trades ~5 % quality margin
for measurable decode speedup.

```env
# … N1 above, but flip:
RVLLM_NVFP4_HADAMARD_V=0
RVLLM_REPETITION_PENALTY=1.0           # disable (saves ~7 ms / decode step)
RVLLM_REPETITION_PENALTY_MIN_COUNT=1
```

**Speed**: ~3.7 tok/s decode (vs 3.5 in N1). **Quality**: WHO clean;
weather clean; HA marginal. **Use when** throughput matters more than
edge-case tool-call coherence.

### N5. Bench / minimal-flag NVFP4

Strip every quality knob to the bare OCP baseline. Useful for kernel
benchmarking and as a regression bisect baseline.

```env
RVLLM_NVFP4_HADAMARD=0
RVLLM_NVFP4_HADAMARD_V=0
RVLLM_NVFP4_K_SCALE_POLICY=amax6
RVLLM_NVFP4_V_SCALE_POLICY=amax6
RVLLM_PER_TOKEN_Q_SCALE=0
RVLLM_Q_SCALE=0.1
RVLLM_PREFILL_CHUNK_SIZE=0    # single-shot batch prefill
RVLLM_TOOL_CALL_OPEN_BIAS=0.0
RVLLM_REPETITION_PENALTY=1.0
```

**Quality**: long-context output IS the documented "la la la"
margin-compression cliff — DO NOT ship to users. **Speed**: fastest
prefill (single-shot, no chunked overhead). **Use** for `rvllm-bench`
TFLOPS measurements and isolating regressions.

---

## FP8 — Top 5

FP8 KV doubles memory vs NVFP4 but has 8-bit precision per element vs
4-bit packed; no per-block microscale (single per-slot scale). Quality
ceiling at long ctx is higher; throughput trades for memory.

### F1. SOTA Production FP8

Conservative full-FP8 KV setup. Long-context cliff is much milder than
NVFP4, so the chunk_size threshold is relaxed.

```env
RVLLM_NVFP4_KV=0
RVLLM_FP8_KV=1
RVLLM_NVFP4_HYBRID_GLOBAL_FP8=0
RVLLM_NVFP4_HYBRID_SLIDING_FP8=0
RVLLM_PER_TOKEN_Q_SCALE=1
RVLLM_Q_SCALE=0.1
RVLLM_PREFILL_CHUNK_SIZE=2048
RVLLM_TOOL_CALL_OPEN_BIAS=0.0
RVLLM_REPETITION_PENALTY=1.05
RVLLM_REPETITION_PENALTY_MIN_COUNT=2
```

**Memory**: ~2× of N1. **Quality**: highest reproducibility across
prompt types, no margin-compression cliff. **Speed**: prefill ~30 s on
14.7k tokens (chunk=2048 vs N1's chunk=256). **Use when** memory headroom
allows.

### F2. High throughput FP8 + GQA decode

Enables the FP8 GQA decode kernel for sliding layers; ~30 % decode
speedup at the cost of an opt-in code path that needs operator
validation.

```env
# … F1 above, plus:
RVLLM_FP8_DECODE_GQA=1
```

**Speed**: ~4.5 tok/s decode. **Caveat**: opt-in path; validate against
a known-good reference (e.g., F1) for any new prompt class.

### F3. F16-fallback safety (legacy, NOT for sm_121)

Highest-precision KV (no quantization). On sm_121 this triggers
`Fa3SoMissing` because the FA3-SM90 path doesn't exist locally.
**Do not use on GB10**; documented for reference / non-Blackwell trees.

```env
RVLLM_F16_KV=1
RVLLM_NVFP4_KV=0
RVLLM_FP8_KV=0
```

**Use only on**: H100 / SM_90 trees that ship the FA3-SM90 `.so`.

### F4. AWQ-int4 + FP8 KV

When the model artifacts are AWQ-int4 weights instead of fp8-block.
Routes QKV / O / gate_up / down through the `awq_int4_gemv_f16` and
`awq_int4_gemm_sm120_wmma` kernels; KV cache stays FP8.

```env
RVLLM_NVFP4_KV=0
RVLLM_FP8_KV=1
RVLLM_PER_TOKEN_Q_SCALE=1
RVLLM_Q_SCALE=0.1
RVLLM_PREFILL_CHUNK_SIZE=2048
# (RVLLM_AWQ_PREFILL_LOOP=0 — keep WMMA GEMM, not the per-token loop)
```

**Memory**: lowest of the FP8 set (int4 weights ≈ 16 GiB vs fp8 ≈ 35 GiB).
**Use when** the model checkpoint is AWQ-quantized (rare).

### F5. Bench / minimal-flag FP8

Strip quality knobs. For kernel-throughput measurements only.

```env
RVLLM_NVFP4_KV=0
RVLLM_FP8_KV=1
RVLLM_NVFP4_HADAMARD=0
RVLLM_PER_TOKEN_Q_SCALE=0
RVLLM_Q_SCALE=0.1
RVLLM_PREFILL_CHUNK_SIZE=0
RVLLM_TOOL_CALL_OPEN_BIAS=0.0
RVLLM_REPETITION_PENALTY=1.0
```

**Quality**: produces correct short-ctx output but no long-ctx
quality fixes. **Speed**: maximum (no extra knobs). **Use** for
`rvllm-bench` baseline.

---

## Quick decision matrix

| If you want… | Use |
|---|---|
| Best NVFP4 quality, lowest memory | **N1** |
| NVFP4 stability across unknown prompts | **N3** |
| NVFP4 throughput | **N4** |
| Highest absolute quality regardless of memory | **F1** |
| Highest absolute throughput | **F2** |
| Pre-Blackwell hardware | **F3** |
| AWQ-int4 model | **F4** |
| Kernel benchmarking | **N5** or **F5** |

---

## Notes on switching configs

- After changing any `RVLLM_NVFP4_*` knob the **prefix cache invalidates
  via `PrefixProvenance`** — first request after a flip gets a full
  prefill, no cache hit.
- KV-dtype switches (`RVLLM_NVFP4_KV` / `RVLLM_FP8_KV` / `RVLLM_F16_KV`)
  require a server restart (loaded at `Gemma4Bringup::load`).
- `RVLLM_PREFILL_CHUNK_SIZE` is read per-request — change without restart.
- `RVLLM_TOOL_CALL_OPEN_BIAS` and `RVLLM_REPETITION_PENALTY` are
  per-decode-step env reads — change without restart.

---

## Performance numbers (Gemma 4 31B, 14.7k-token prompt, GB10)

| Config | Prefill (s) | Decode (tok/s) | Memory (GiB) | Quality |
|---|---|---|---|---|
| N1 | 125 | 3.5 | 47 | **30/30 smoke clean** (cycle 56 step 28) |
| N2 | 95 | 3.55 | 47 | WHO+WEATHER clean, HA variable |
| N3 | 75 | 3.6 | 52 | most stable, highest variance class |
| N4 | 90 | 3.7 | 47 | WHO+WEATHER clean, HA marginal |
| N5 | 30 | 4.0 | 47 | DO NOT SHIP — long-ctx cliff |
| F1 | 30 | 3.4 | 60 | best reproducibility |
| F2 | 30 | 4.5 | 60 | F1 + GQA win |
| F3 | n/a | n/a | n/a | not on sm_121 |
| F4 | 28 | 3.3 | 41 | AWQ tradeoff |
| F5 | 28 | 4.0 | 60 | bench only |

(Numbers measured iter 30–47, May 2026. Re-validate after any kernel
PTX rebuild — see `kernels/build.sh sm_121` + `gen_manifest.sh`.)
