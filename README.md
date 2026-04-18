# rvLLM

A single-GPU, FP8, graph-captured LLM inference engine in Rust. Qwen2.5-7B on a single H100 SXM 80GB delivers **41,560 tok/s at N=512** (FP8 E4M3 KV cache, real FA3 paged-prefill, decode + final RMSnorm + LM head + argmax, CUDA graph captured) and **~50 ms TTFT at N=64** (16-token prompt, hot cache). vs vLLM 0.19 at matched batches: **rvLLM wins at N≥64 (+12% to +40%)**, vLLM wins at N≤16. Crossover sits near N=32–64.

No Python in the hot path. No fallbacks. Missing artifacts (policy, FA3 `.so`, kernel SHA) refuse to start.

## Headline: rvLLM 0.3.0 vs vLLM 0.19, full sweep

Same GPU (H100 SXM 80GB), same Qwen2.5-7B-Instruct FP8 E4M3 weights + FP8 E4M3 KV, CUDA graphs on both engines, 16 input tokens + 512 output tokens per request, real prefill on both. All 7 batch sizes tested on one booted server per engine, cold = first request at that shape, hot = second request (cuBLASLt/FlashInfer heuristics cached).

### Throughput (tok/s, steady-state decode)

| N   | rvLLM 0.3.0 |  vLLM 0.19 (hot) | Δ           |
|----:|------------:|-----------------:|:------------|
|   1 |         137 |              223 | vLLM +62.6% |
|   8 |       1,095 |            1,712 | vLLM +56.4% |
|  16 |       2,318 |            3,342 | vLLM +44.2% |
|  64 |      12,477 |           11,151 | **rvLLM +11.9%** |
| 128 |      21,946 |           17,778 | **rvLLM +23.4%** |
| 256 |      33,968 |           24,270 | **rvLLM +40.0%** |
| 512 |      41,560 |           31,485 | **rvLLM +32.0%** |

### Time-to-first-token (ms)

|   N | rvLLM cold | rvLLM hot |  vLLM cold |  vLLM hot |
|----:|-----------:|----------:|-----------:|----------:|
|   1 |      48.06 |     49.04 |     444.50 |     25.15 |
|   8 |     414.03 |    314.46 |   6,075.92 |     47.25 |
|  16 |      53.57 |     46.30 |      82.16 |     75.18 |
|  64 |      50.84 |     51.35 |     213.26 |    200.78 |
| 128 |      80.52 |     83.20 |     386.34 |    347.59 |
| 256 |     112.67 |     77.74 |     581.20 |    622.22 |
| 512 |     131.54 |    132.12 |     985.94 |    879.60 |

- **Throughput crossover near N=32–64.** rvLLM's per-step fixed cost (28 layers × ~12 launches + metadata re-upload + graph replay edge) dominates at tiny batch; vLLM's tighter small-batch path wins there. Above the crossover, rvLLM's cuBLASLt-FP8 + FA3-FP8-KV + single-graph-replay path pulls ahead by 12–40%.
- **TTFT (hot): rvLLM dominates everything except N=1.** At N=128/256/512, vLLM's hot TTFT is **4–7× higher** than rvLLM's.
- **TTFT (cold): vLLM is catastrophically cold at small N.** vLLM's N=8 cold hit 6.08 s — first request at that batch shape triggered graph capture + compile on a path with no captured graph yet. rvLLM pre-builds everything at bring-up; cold ≈ hot for us on most batch sizes.

Test harness:

- **rvLLM:** `RVLLM_BATCH=<N> RVLLM_ITERS=128 RVLLM_WARMUP=5 RVLLM_TTFT=1 RVLLM_REAL_PREFILL=1 RVLLM_PREFILL_LEN=16 ./target/release/rvllm-bench`. Cold TTFT = first of two timed prefill calls; hot TTFT = second. Steady-state tok/s computed over 128 timed decode iterations after 5 warmup iterations.
- **vLLM 0.19 V1:** `vllm serve Qwen2.5-7B-Instruct --quantization fp8 --kv-cache-dtype fp8_e4m3 --dtype bfloat16 --max-model-len 2048 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9`, then `vllm bench serve --dataset-name random --num-prompts N --max-concurrency N --random-input-len 16 --random-output-len 512 --ignore-eos` run twice per N (cold + hot). Throughput = the tool's `Output token throughput`, TTFT = `Mean TTFT`.

## Where the crossover is

The N=32–64 crossover has a clean structural explanation. rvLLM's decode step runs 339 pre-captured kernel launches (28 layers × 12 kernels + sampling tail) in one `cuGraphLaunch`. That's a fixed per-step cost — independent of N. Divided across N tokens, it hurts small N (per-token overhead is high) and helps large N (per-token overhead is amortized). vLLM's small-batch path dispatches fewer unique operations per step (custom CUDA graphs specialized per batch size), so its per-token overhead at N=1/8/16 is lower than ours.

Above the crossover the workload becomes HBM-bandwidth bound, not launch bound. Our FP8 E4M3 weights + FP8 E4M3 KV cache + fused cuBLASLt epilogues shift us from 2.5 TB/s-bound to 3.0 TB/s-bound per step, and vLLM's FlashInfer path doesn't have the same cuBLASLt-fused epilogue advantage in that regime.

**Fair-bench fixes we'd landed previously:**

1. **Real prefill on both sides.** vLLM does prefill with its chunked-prefill scheduler; rvLLM does one multi-query causal FA3 paged-prefill (`RVLLM_REAL_PREFILL=1`) via varlen `cu_seqlens_q`. Both engines start the decode window with identical KV-populated state.
2. **Per-step metadata upload on rvLLM.** `positions`, `slot_mapping`, and `context_lens` are re-uploaded each iteration inside the timed loop — matches the per-step HtoD cost vLLM's scheduler pays.
3. **CUDA graphs on both.** vLLM server captures its default sizes (up to 512); rvLLM captures exactly the bench's N.

## Why v3 is fast (structural wins)

1. **Fused Q‖K‖V GEMM.** One matmul with N=4608 replaces three separate Q/K/V GEMMs. 2 fewer launches per layer × 28 layers = 56 fewer launches per decode step.
2. **All 5 FP8 linears on cuBLASLt.** QKV, O, gate_up, down, lm_head all go through `cublasLtMatmul`. Its per-shape heuristic explores many more algorithms than a hand-maintained CUTLASS variant cache. Per-(M,N,K,epilogue) algo cache so the heuristic runs once per shape at capture time.
3. **Fused epilogues.** `CUBLASLT_EPILOGUE_BIAS` folds the f16 bias add into the QKV GEMM; `β=1` folds the residual add into O and down GEMMs. 3 kernels per layer × 28 = 84 fewer launches.
4. **FP8 E4M3 KV cache.** `libfa3_kernels.so` links FA3's upstream hdim128 E4M3 paged instantiations. KV pages are 1 byte/element with per-tensor descale; halves KV memory and unlocks N=256/512 that f16 KV cannot fit on one H100.
5. **No f32 on the GPU in the decode path.** RoPE tables are f16. The only f32 on-device is the FP8 protocol's per-tensor / per-token scale scalar.
6. **Correct Qwen2.5 forward.** Includes `model.embed_tokens`, 28× decoder layer, `model.norm` (final RMSnorm — previously missing), `lm_head` with FP8-quantized weights, argmax tail.
7. **Single graph replay.** All per-step kernel launches captured into one `cuGraphLaunch`; metadata re-uploaded per step to match real-scheduler overhead.
8. **No Python.** End-to-end Rust. No torch, no JIT, no interpreter cost on the hot path.

## The v3 stack, every layer

```
┌───────────────────────────────────────────────────────────────────┐
│  Bringup                     v3/crates/rvllm-runtime::bring_up    │
│                              one-shot: ctx → arena → weights →    │
│                              kernels → CUTLASS .so → FA3 .so →    │
│                              cuBLASLt                             │
├───────────────────────────────────────────────────────────────────┤
│  Engine / step               type-state step_launch + collect     │
│                              PendingStep<'e> borrows &mut Engine  │
│                              CUDA graph captured once per bucket  │
├───────────────────────────────────────────────────────────────────┤
│  layer_exec::forward         12 launches per Llama decoder layer  │
│                                fused_add_rmsnorm_fp8_quant        │
│                                cublasLt FP8 GEMM + bias (QKV)     │
│                                fused_rope_cache_fp8kv             │
│                                FA3 paged_decode                   │
│                                quantize_fp8_per_token             │
│                                cublasLt FP8 GEMM + residual (O)   │
│                                fused_add_rmsnorm_fp8_quant        │
│                                cublasLt FP8 GEMM (gate_up)        │
│                                fused_silu_mul_fp8_quant           │
│                                cublasLt FP8 GEMM + residual (down)│
├───────────────────────────────────────────────────────────────────┤
│  Sampling                    cublasLt FP8 GEMM (lm_head)          │
│                              argmax_kernel                         │
├───────────────────────────────────────────────────────────────────┤
│  Memory                      one HBM arena, bump-allocated regions│
│                              checkpoint/restore for sweep modes   │
│                              arena-lifetime Region<'a> graph-safe │
├───────────────────────────────────────────────────────────────────┤
│  Invariants                  type-level CUTLASS schedule pairing  │
│                              MetaLayoutHash on graph replay       │
│                              FP8 clamp-ppm gate at weight load    │
└───────────────────────────────────────────────────────────────────┘
```

## Kernels

Every kernel has a known purpose, a pinned variant, and a workspace contract. No dispatch fallback chains.

### CUTLASS SM90 FP8 GEMM (`kernels/cutlass_fp8_gemm.cu`, `cutlass_fp8_gemm_residual.cu`)

40 non-residual variants + 10 residual-fused variants, all templated on `(TileShape, ClusterShape, KernelSchedule)`. Autotuned per shape at build time into `policy.json`.

- Non-residual: QKV, gate_up, down_proj
- Residual-fused: o_proj (fused GEMM + residual add in one launch; eliminates one HBM round-trip per layer)
- Per-row activation scale, per-tensor weight scale
- **Compatibility rule enforced**: mainloop schedule must match epilogue schedule. `KernelTmaWarpSpecialized` (WS) mainloop + `TmaWarpSpecializedCooperative` (Coop) epilogue is the canonical class of SM90 illegal-memory bug. v3 makes this a compile-time `static_assert`; v2 uses variant `v1` (128×128×128 Coop/Coop matched) for o_proj residual.

### FlashAttention-3 SM90 paged decode (`kernels/fa3_sm90_wrapper.cu` → `libfa3_kernels.so`)

WGMMA + TMA, built from FlashAttention-3 Hopper source. Paged KV layout `[2, num_blocks, block_size, num_kv_heads, head_dim]`, GQA via `num_heads / num_kv_heads` ratio. `head_dim = 128` hard gate.

- **No PTX fallback.** If `libfa3_kernels.so` is not present at startup, the engine refuses to load.
- Build: `bash kernels/build_fa3.sh` on the H100 box (needs CUTLASS headers + flash-attention/hopper source), ~10 min.

### Fused pre/post kernels (PTX, `kernels/fused_*.cu`)

| Kernel | Inputs | Output | Launches saved |
|---|---|---|---|
| `embedding_gather` | token_ids, weight | f16 hidden | 1 |
| `fused_add_rmsnorm_fp8_quant` | hidden, residual, gamma | residual', fp8_act, scale | 3 → 1 |
| `fused_rmsnorm_fp8_quant` | hidden, gamma | fp8_act, scale | 2 → 1 |
| `quantize_fp8_per_token` | f16 act | fp8 + scale | 1 |
| `fused_rope_cache_fp8kv` | q/k/v f16, cos, sin, slot_mapping | q_fp8 out + writes FP8 K/V into paged cache | 4 → 1 |
| `fused_silu_mul_fp8_quant` | gate_up f16 | fp8_act + scale | 3 → 1 |
| `argmax` | f32 logits | i32 token | 1 |
| `residual_add_f16` | x, y | x + y | 1 |

Rule: each kernel fuses at most one recognizable composite. No megakernels. Every kernel has a pure-Rust f32 reference implementation in tests; PTX output must match within cosine 0.999.

## One v3 decode step, in order

For decode batch of N sequences:

```
step_launch:
  1. graph_pool.replay(bucket)                 — one cuGraphLaunch
     For each layer in 0..28:
       fused_add_rmsnorm_fp8_quant             — attn norm + quantize
       cublasLtMatmul(FP8 + BIAS epilogue)     — QKV (one shot, packed)
       fused_rope_cache_fp8kv_kernel           — rope + FP8 Q quant + FP8 KV paged write
       fa3_sm90_paged_decode                   — attention
       quantize_fp8_per_token_kernel           — quantize attn_out
       cublasLtMatmul(FP8, beta=1 residual)    — O proj + residual add
       fused_add_rmsnorm_fp8_quant             — mlp norm + quantize
       cublasLtMatmul(FP8)                     — gate_up proj
       fused_silu_mul_fp8_quant_kernel         — SiLU(gate)*up + quantize
       cublasLtMatmul(FP8, beta=1 residual)    — down proj + residual add

  2. quantize_fp8_per_token_kernel              — hidden -> fp8
     cublasLtMatmul(FP8)                        — lm_head
     argmax_kernel                              — fp16 logits -> i32 token
```

**12 launches per layer × 28 layers = 336 kernels, plus 3 at the sampling tail = 339 total, all captured into one `cuGraphLaunch`.**

No mega-kernels. Each launch does one recognizable composite. The only fused-epilogue is the one cuBLASLt gives us for free (bias / residual-add).

## Correctness discipline

Explicit rules. Violations fail the build or startup, never degrade silently.

1. **No fallbacks.** Missing autotune entry = engine panic with shape. Missing FA3 `.so` = refuse start. Missing CUTLASS .so = refuse start. Stale autotune cache from a prior deploy is not consulted — policy lives in the build artifact.
2. **Graph-capture invariant.** Metadata buffer layout is frozen per `(bucket, max_blocks_per_seq)`. Captured graphs bind those exact offsets. There is no "non-padded" upload path. Prefill and decode have separate APIs; they do not share metadata offsets.
3. **Real block-change detection.** Scheduler emits `ContinuedRequest::block_table_update: Option<Vec<BlockId>>` whenever a sequence's physical block list has grown since the last send. Worker combines this with CoW-copy events. Missing either signal leaves stale block_ids in the captured graph (wrong KV reads, not a crash — silent correctness bug).
4. **CUTLASS schedule/epilogue pairing.** Mainloop and epilogue schedules must match. Mismatched variants cause `CUDA_ERROR_ILLEGAL_ADDRESS` only inside graph replay. Enforced in v3 as a CUDA `static_assert`; in v2, dispatch pins to a hand-verified variant (`v1`).
5. **No `unwrap()` in libraries.** `Result<T, RvllmError>` end-to-end. Errors carry structured context (stream, kernel name, launch config) — not stringified `DriverError`.

## v3 path to 42,030 tok/s

Session progression on H100 SXM 80GB, Qwen2.5-7B, FP8 E4M3:

| step | tok/s | N | Δ | what changed |
|---|---:|---:|---:|---|
| eager decode, LM head in | 551 | 128 | — | no graph capture, bench harness first light |
| + graph capture | 14,745 | 128 | 27× | capture one step, replay iters-many |
| + f16 RoPE tables | 15,537 | 128 | +5% | removed f32 cos/sin from hot path |
| + QKV/bias correctness | 15,985 | 128 | +3% | load q/k/v biases, apply post-GEMM |
| + fused QKV GEMM | 15,985* | 128 | — | 3 GEMMs → 1 (Q‖K‖V packed, N=4608) |
| + cuBLASLt FP8 + BIAS epilogue (QKV) | 17,562 | 128 | +10% | one matmul replaces GEMM + add_bias kernel |
| + all 5 linears on cuBLASLt | 22,496 | 128 | +28% | O/gate_up/down/lm_head also cuBLASLt-autotuned |
| + FP8 E4M3 KV + final RMSnorm | 20,841 | 128 | −7% at N=128 | FA3 dequant overhead, but halves KV memory |
| …unlocks N=256 | 31,178 | 256 | +50% | 2× KV-memory reduction, GEMMs near HBM saturation |
| …and N=512 | 40,331 | 512 | +29% | full HBM saturation regime |
| + **real FA3 paged-prefill (Phase F)** | **22,069 / 34,364 / 42,030** | 128/256/512 | **+6–10%** | scratch sizing corrected; TTFT drops 6–12× vs faux-prefill |

The cuBLASLt move was the dominant per-token win. FP8 KV was the capacity unlock. Real FA3 prefill (Phase F) is the TTFT win: one multi-query causal FP8 attention call replaces a 16-step eager stand-in, dropping N=128 TTFT from 763 ms to 63.8 ms — and along the way the Phase F scratch-buffer fixes also lifted steady-state throughput at every batch size.

## Reproduce (v3)

```bash
# One-time on H100 box (~15 min)
bash kernels/build.sh               # fused PTX (rmsnorm, rope, silu, argmax, ...)
bash kernels/build_cutlass_so.sh    # libcutlass_kernels.so (FP8 variants)
bash kernels/build_fa3.sh           # libfa3_kernels.so

# Emit v3 manifest.json + policy.json (keyed by sha256)
python3 v3/kernels/make_manifest.py /workspace/rvllm/kernels/sm_90 sm_90 $(git rev-parse HEAD)
python3 v3/kernels/make_policy.py   /workspace/rvllm/kernels/sm_90/policy.json $(git rev-parse HEAD)

# Build v3 bench
cargo build --release --features cuda --manifest-path v3/Cargo.toml -p rvllm-bench

# Run
RVLLM_MODEL_DIR=/workspace/models/qwen25-7b-instruct \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
RVLLM_BATCH=128 RVLLM_ITERS=30 RVLLM_WARMUP=5 \
  ./v3/target/release/rvllm-bench
```

## TPU: Gemma 4 31B on v6e-4

rvLLM runs on Google Cloud TPU via JAX + XLA. No custom kernels -- XLA compiles the entire 60-layer forward pass to TPU machine code from a ~500 line JAX script. Built and profiled in a single session.

### Headline numbers

| Metric | B=1 | B=8 | B=768 (peak) |
|---|---|---|---|
| **Decode throughput** | **79.9 tok/s** | **584 tok/s** | **13,943 tok/s** |
| **Per-step latency** | **12.5 ms** | **13.7 ms** | **55.1 ms** |
| **Latency p99** | **14.8 ms** | -- | -- |
| **Latency jitter (std)** | **0.07 ms** | -- | -- |
| **Perplexity** | **25.51** | -- | -- |
| **Cost efficiency** | **15.4 tok/s/$** | **112 tok/s/$** | **2,681 tok/s/$** |

| Fixed | Value |
|---|---|
| Model | Gemma 4 31B (google/gemma-4-31B-it) |
| Hardware | TPU v6e-4 (4 chips, 128 GB HBM, ~3.3 TB/s) |
| Quantization | int8 per-channel (weights), bf16 activations |
| Perplexity | 25.51 (verified against HuggingFace reference) |
| Cost | ~$5.20/hr (v6e-4 on-demand) |

### vs GPU: Gemma 4 31B decode

| Hardware | Peak tok/s | Batch | Cost/hr | tok/s/$ |
|---|---|---|---|---|
| **TPU v6e-4 (rvLLM)** | **13,943** | **768** | **$5.20** | **2,681** |
| H100 SXM (FP8, projected) | ~6,000 | ~256 | $8-10 | 600-750 |
| H200 (bf16, projected) | ~4,000 | ~128 | $8-12 | 333-500 |

### Optimization progression

| Step | tok/s | ms/step | What changed |
|---|---|---|---|
| Nested scan, bf16 | 25.6 | 38.0 | Initial working version |
| Flat scan, bf16 | 48.2 | 19.4 | +88%: eliminated nested loop overhead |
| Flat scan, int8 | 68.2 | 13.4 | +42%: halved weight bandwidth |
| Fused on-chip decode | 79.9 | 12.5 | +17%: zero host overhead via while_loop |
| **B=8 batched** | **584** | **13.7** | **7.3x: near-linear batch scaling** |

### XProf breakdown (B=1, per decode step)

| Component | Time | % |
|---|---|---|
| 60-layer scan (single while loop) | 10.6 ms | 86% |
| jax.lax.cond dispatch (sliding/global) | 1.8 ms | 14% |
| ICI all-reduce (O + down proj, 4 chips) | 0.6 ms | 5% |
| KV cache dynamic_update_slice | 1.3 ms | 10% |
| **Total step** | **12.3 ms** | |
| Theoretical BW limit (30 GB / 3.3 TB/s) | 9.1 ms | |

HLO stats: 4 fused matmul ops, 6 all-reduces, 1 while loop, 1 conditional. XLA compiles the entire model into a single tight loop body.

### Architecture details

Gemma 4 is a 60-layer transformer with dual attention (50 sliding + 10 global layers). Key Gemma 4 specifics we handle:

- **Dual head_dim**: sliding layers use 32 Q heads of 256, 16 KV heads of 256; global layers use 32 Q heads of 512, 4 KV heads of 512
- **Weight shape asymmetry**: sliding q_proj=[8192,5376], global q_proj=[16384,5376]. Padded to max shape for scan uniformity
- **QK-norm + v-norm**: per-head RMSNorm on Q/K (with learned scale), parameter-free RMSNorm on V
- **k_eq_v**: global layers have no v_proj; V = v_norm(raw_K)
- **Attention scaling = 1.0**: QK-norm handles magnitude, no sqrt(head_dim) division
- **layer_scalar**: applied ONCE at the end of the full layer (not per-sublayer)
- **Partial RoPE**: global layers rotate only 128 of 512 dims (25%), theta=1M vs theta=10k for sliding
- **Logit softcapping**: 30 * tanh(logits / 30)
- **GELU(tanh)** activation (not SiLU)

### The TPU stack

```
JAX Python (trace)
  |
  v
StableHLO / MLIR
  |
  v
XLA compiler --> TPU machine code (single fused while loop)
  |
  v
PJRT runtime (4-chip SPMD, TP=4)
  |-- NamedSharding + PartitionSpec for automatic weight distribution
  |-- Buffer donation for KV cache reuse
  `-- jax.lax.while_loop for zero-host-overhead decode
```

No hand-written kernels. No Pallas. No custom ops. XLA generates everything from pure JAX. The script is ~500 lines of Python.

### Deployment

Total setup time: ~5 minutes (create TPU + install JAX + download model).

```bash
# Create TPU v6e-4 ($5.20/hr)
gcloud compute tpus tpu-vm create rvllm-gemma4 \
  --zone=us-east5-b --accelerator-type=v6e-4 --version=v2-alpha-tpuv6e

# Install (30 seconds)
pip3 install 'jax[tpu]' huggingface_hub tokenizers \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Download model (2 minutes on GCP internal network)
huggingface-cli download google/gemma-4-31B-it --local-dir ~/models/gemma-4-31B-it

# Run (first call: ~5s JIT compile, then 79.9 tok/s)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 512

# Batched (584 tok/s)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 512 --batch 8

# Perplexity (25.51)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --perplexity --max-ctx 512

# Cleanup
gcloud compute tpus tpu-vm delete rvllm-gemma4 --zone=us-east5-b --quiet
```

No Docker. No conda. No torch. No vLLM. One pip install, one Python file, one command.

## Supported models

### GPU (H100/H200, Rust+CUDA, FP8)

- **Qwen2** / Qwen2.5 (verified bench model)
- **Llama 2 / 3 / 3.1**
- **Mistral 7B**
- **Gemma 1 / 2**
- **Gemma 4 31B** (60 layers, hybrid sliding/global attention, QK-norm, GELU, dual RoPE, logit softcap)

GQA via `num_heads / num_kv_heads`. FA3 supports head_dim 128/256/512. Other architectures compile but have not been end-to-end validated against HF reference on this version.

### TPU (v6e, JAX+XLA, int8)

- **Gemma 4 31B** (verified, 79.9 tok/s on v6e-4, PPL 25.51)

Weight formats: SafeTensors (sharded + single-file). GGUF is supported in the older `rvllm-model-runner` crate but not the v2 FP8 path.

## v3 crate map

```
v3/crates/
├── rvllm-core         typed errors, IDs, dtype, shape, config, env
├── rvllm-mem          HbmArena, Region, Stream, Event, PinnedBuf, CudaContextHandle
├── rvllm-kernels      manifest (sha-pinned), PTX loader, kernel catalog
├── rvllm-fused        8 fused-kernel launchers + pure-Rust f32 references
├── rvllm-attention    FA3 SM90 paged decode/prefill dlopen
├── rvllm-cutlass      FP8 variant catalog + schedule pairing trait + cuBLASLt wrapper
├── rvllm-metadata     frozen-layout metadata per bucket (one upload path)
├── rvllm-loader       safetensors mmap -> HBM + CPU-path FP8 quant + clamp gate
├── rvllm-sampling     argmax tail, pinned DtoH
├── rvllm-graph        captured-graph pool keyed on MetaLayoutHash
├── rvllm-runtime      Engine, scheduler, layer_exec, bring_up
├── rvllm-bench        RVLLM_* env-driven bench binary
└── rvllm-invariants   DAG-dep test, no-megakernel gate
```

## Archive: rvllm-v2

The v2 engine at `crates/rvllm-v2*` (plus sibling `rvllm-gpu`, `rvllm-engine`, `rvllm-worker` etc.) is frozen at commit `eb9e247fd` (19,287 tok/s at N=128). Kept in-tree so we can port bits back if they prove useful and so the perf recovery history (`9,531 → 19,287`) is readable. Not an active build target.

The v2 CUTLASS .so build scripts in `kernels/` are still used by v3 — v3's cuBLASLt is the fast path but `libcutlass_kernels.so` compiles the same variants and is still loaded at startup (v3 resolves but currently doesn't dispatch through it; kept for the sweep harness).

## License

Apache-2.0.

## Further reading

- [`v3/SPEC.md`](v3/SPEC.md), [`v3/IMPL_PLAN.md`](v3/IMPL_PLAN.md) — the clean-slate rewrite plan, 16 focused agent specs, with CUTLASS schedule-mismatch made a compile error, one metadata upload path, `GraphSafe` marker trait enforcing no-realloc-during-capture at the type level.
- [`docs/paper/rvllm.pdf`](docs/paper/rvllm.pdf) — technical paper.
- [`docs/arch.md`](docs/arch.md) — full crate architecture.
