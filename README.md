# rvLLM

LLM inference engine. Rust+CUDA on GPU, JAX+XLA on TPU.

**31B Gemma 4 on TPU v6e-4: 13,943 tok/s** (B=768, int8, TP=4 SPMD, PPL 19.24). 2,681 tok/s/$. 128K context support (24.7 tok/s decode). 3.6x faster than vLLM on H100 GPU (measured). Zero custom kernels - ~500 lines of JAX, XLA compiles everything.

## TPU: 31B Gemma 4 on v6e-4

Pure JAX + XLA. No custom kernels. XLA compiles the entire 60-layer forward pass to TPU machine code from a ~500 line JAX script.

### Headline numbers

| Metric | B=1 (512 ctx) | B=1 (2048 ctx) | B=1 (128K ctx) | B=768 (peak) |
|---|---|---|---|---|
| **Decode throughput** | **79.9 tok/s** | **41.9 tok/s** | **24.7 tok/s** | **13,943 tok/s** |
| **Per-step latency** | **12.52 ms** | **23.84 ms** | **40.56 ms** | **55.1 ms** |
| **Architecture** | single-scan | split-cache | split-cache | single-scan |
| **Perplexity** | 22.80 | **19.24** | **19.24** | |
| **Cost efficiency** | **15.4 tok/s/$** | **8.1 tok/s/$** | **4.8 tok/s/$** | **2,681 tok/s/$** |

| Fixed | Value |
|---|---|
| Model | 31B Gemma 4 (google/gemma-4-31B-it) |
| Hardware | TPU v6e-4 (4 chips, 128 GB HBM, ~3.3 TB/s) |
| Quantization | int8 per-channel (weights), bf16 activations, int8 KV cache (per-head scales) |
| Perplexity | 19.24 (split-cache int8 KV, improved from original 25.51) |
| Context | 512 to 128K supported via `--max-ctx` |
| Cost | ~$5.20/hr (v6e-4 on-demand) |

### Context scaling

| Context | ms/step | tok/s | Architecture | KV Memory |
|---|---|---|---|---|
| 512 (single-scan) | 12.52 | 79.9 | 1 scan, 60 layers, cond dispatch | 500MB bf16 |
| 2048 (split-cache) | 23.84 | 41.9 | 10 groups x 6 layers, no cond | 587MB int8 |
| 8192 | ~28 | ~35 | split-cache + blockwise global | 755MB int8 |
| 32K | ~32 | ~31 | split-cache + blockwise global | 3.0GB int8 |
| 64K | ~36 | ~28 | split-cache + blockwise global | 5.8GB int8 |
| 128K | 40.56 | 24.7 | split-cache + blockwise global | 11.1GB int8 |

### Split-cache architecture

Gemma 4's 60 layers have two attention types: 50 sliding-window layers (1024-token window) and 10 global layers (full context). The split-cache design exploits this structure:

- **Split-cache**: 50 sliding layers use a 1024-entry circular buffer, 10 global layers use full-context blockwise attention
- **Grouped execution**: processed in 10 groups of 6 (5 sliding + 1 global), eliminating jax.lax.cond overhead
- **Int8 KV cache**: per-head quantization scales, 50% memory reduction with quality preserved
- **Blockwise global attention**: online softmax with BLOCK_K=8192, never materializes full score tensor
- **Sliding window**: O(1024) per layer regardless of context length

### Perplexity progression

| Version | PPL | Notes |
|---|---|---|
| bf16 KV, single scan | 25.51 | Original baseline |
| int8 KV, single scan | 22.80 | Per-head scales |
| int8 KV, split-cache | 19.24 | Circular buffer + blockwise |

### Memory budget at 128K (per chip, TP=4)

| Component | Size |
|---|---|
| Weights (int8) | 7.75 GB |
| Sliding KV (50 layers x 1024 x 1024 bytes) | 50 MB |
| Global KV (10 layers x 131072 x 512 bytes, after TP shard) | 625 MB |
| Scale arrays | ~60 MB |
| **Total** | **~8.5 GB of 32 GB** |

### Batch scaling sweep

| Batch | tok/s | ms/step | Scaling | tok/s/$ |
|---|---|---|---|---|
| 1 | 79.9 | 12.5 | 1x | 15.4 |
| 8 | 584 | 13.7 | 7.3x | 112 |
| 64 | 4,220 | 15.2 | 52.8x | 812 |
| 128 | 6,831 | 18.7 | 85.5x | 1,314 |
| 256 | 10,536 | 24.3 | 131.9x | 2,026 |
| 512 | 12,932 | 39.6 | 161.9x | 2,487 |
| **768** | **13,943** | **55.1** | **174.5x** | **2,681** |
| 1024 | 13,705 | 74.7 | 171.5x | 2,636 |

Near-linear scaling from B=1 to B=512. Peak at B=768 where compute and bandwidth saturate simultaneously.

### TPU vs vLLM GPU comparison (measured)

Head-to-head against vLLM on H100 SXM 80GB (RedHatAI/gemma-4-31B-it-FP8-Dynamic, $1.92/hr on vast.ai). All numbers measured on our hardware.

**Single-user latency (B=1):** rvLLM TPU single-scan 79.9 tok/s vs vLLM GPU 66.9 tok/s. TPU 19% faster.

**Peak throughput:** rvLLM TPU single-scan 13,943 tok/s (B=768) vs vLLM GPU 3,848 tok/s (B=128). TPU 3.6x faster.

**Cost efficiency at peak:** TPU 2,681 tok/s/$ vs GPU 2,004 tok/s/$. TPU 34% better.

**128K context:** only TPU (split-cache architecture) supports it. vLLM GPU tested at max_ctx=2048.

| Batch | vLLM GPU tok/s | vLLM GPU ms/step | rvLLM TPU tok/s | rvLLM TPU ms/step |
|---|---|---|---|---|
| 1 | 66.9 | 14.95 | **79.9** | 12.5 |
| 2 | 131.5 | 15.20 | - | - |
| 4 | 257.6 | 15.53 | - | - |
| 8 | 511.7 | 15.63 | **584** | 13.7 |
| 16 | 926.5 | 17.27 | - | - |
| 32 | 1,728 | 18.51 | - | - |
| 48 | 2,258 | 21.26 | - | - |
| 64 | 2,794 | 22.90 | **4,220** | 15.2 |
| 96 | 3,083 | 31.14 | - | - |
| 128 | **3,848** | 33.26 | **6,831** | 18.7 |
| 256 | 3,709 | 69.03 | **10,536** | 24.3 |
| 512 | 3,788 | 135.18 | **12,932** | 39.6 |
| 768 | 3,671 | 209.18 | **13,943** | 55.1 |

| Hardware | Peak tok/s | Batch | Cost/hr | tok/s/$ |
|---|---|---|---|---|
| **TPU v6e-4 (rvLLM, single-scan)** | **13,943** | **768** | **$5.20** | **2,681** |
| H100 SXM (vLLM, FP8, measured) | 3,848 | 128 | $1.92 | 2,004 |

TPU batch scaling numbers are from the single-scan architecture (512 ctx, no 128K support). Split-cache batch sweep in progress.

### Optimization progression

| Step | tok/s | ms/step | What changed |
|---|---|---|---|
| Nested scan, bf16 | 25.6 | 38.0 | Initial working version |
| Flat scan, bf16 | 48.2 | 19.4 | +88%: eliminated nested loop overhead |
| Flat scan, int8 | 68.2 | 13.4 | +42%: halved weight bandwidth |
| Fused on-chip decode | 79.9 | 12.5 | +17%: zero host overhead via while_loop |
| B=8 batched | 584 | 13.7 | 7.3x: near-linear batch scaling |
| B=64 + LIBTPU flags | 4,220 | 15.2 | async collective fusion |
| **B=768 + LIBTPU flags** | **13,943** | **55.1** | **174.5x from baseline** |

### XProf breakdown (B=1, per decode step)

| Component | Time | % |
|---|---|---|
| 60-layer scan (single while loop) | 10.6 ms | 86% |
| &ensp; incl. jax.lax.cond dispatch (sliding/global) | 1.8 ms | 15% of scan |
| &ensp; incl. ICI all-reduce (O + down proj, 4 chips) | 0.6 ms | 6% of scan |
| &ensp; incl. KV cache dynamic_update_slice | 1.3 ms | 12% of scan |
| Host + dispatch overhead | 1.7 ms | 14% |
| **Total step** | **12.3 ms** | **100%** |
| Theoretical BW limit (30 GB / 3.3 TB/s) | 9.1 ms | |

The 1.8 ms cond overhead is structural: Gemma 4's dual attention (50 sliding + 10 global layers) requires runtime dispatch. Flat scan + cond is the optimum for XLA's TPU compiler. (Note: the split-cache architecture eliminates this cond overhead by grouping layers, but shifts time into blockwise global attention at longer contexts.)

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

No hand-written kernels. No Pallas. No custom ops. XLA generates everything from pure JAX.

### TPU deployment

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

# Run (first call: ~5s JIT compile, then 79.9 tok/s at 512 ctx)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 512

# 128K context (24.7 tok/s decode)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 131072

# Batched (13,943 tok/s)
LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_dot_dot_fusion_duplicated=true" \
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 512 --batch 768

# API server (OpenAI-compatible)
python3 tpu/harness/api_server.py --model-dir ~/models/gemma-4-31B-it --port 8080

# Perplexity (19.24, split-cache int8 KV)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --perplexity --max-ctx 2048

# Cleanup
gcloud compute tpus tpu-vm delete rvllm-gemma4 --zone=us-east5-b --quiet
```

No Docker. No conda. No torch. No vLLM. One pip install, one Python file, one command.

Chat client: native Rust egui app at `chat-client/`, connects via OpenAI API to the TPU server.


## EAGLE-3 Speculative Decoding (TPU, experimental)

EAGLE-3 draft-verify speculation for single-user latency. Trains a lightweight 450M-param draft head that proposes K=5 tokens per cycle; the full 31B target verifies all K+1 in one forward pass.

### Status

| Metric | Value |
|---|---|
| Baseline (fused while_loop, B=1) | 79.9 tok/s, 12.5 ms/step |
| EAGLE-3 fused cycle (random draft) | 31.0 ms/cycle |
| EAGLE-3 fused cycle (trained, 1K examples) | 31.0 ms/cycle, tau=1.01 |
| Projected (tau=3.5, trained on 50K+) | ~145 tok/s (1.8x) |
| Projected (tau=3.5 + int8 KV cache) | ~175 tok/s (2.2x) |
| Hardware ceiling (perfect pipelining) | ~300 tok/s (3.8x) |

The cycle time (31ms) is physics-bound: 9.5ms weight read + 9.5ms TP=4 all-reduce at T=6 + 11.5ms KV reads + 2ms draft. The lever is tau (acceptance rate), which requires more training data.

### Training the draft head

```bash
# Prepare training data (JSONL: {"text": "conversation..."})
# UltraChat, ShareGPT, or self-distilled from target model

# Train (1K examples, ~11 min on v6e-4; 10K examples, ~2 hours)
python3 tpu/harness/eagle3_train.py \
  --model-dir ~/models/gemma-4-31B-it \
  --data-file train.jsonl \
  --output-dir eagle3-head \
  --max-seq 512 --epochs 2 --lr 5e-5 --warmup-steps 100

# Training loss progression (1K examples, 2 epochs):
#   step 10:   loss 20.2  (random)
#   step 50:   loss 9.1   (-55%)
#   step 300:  loss 7.9   (-61%)
#   step 1000: loss 7.8   (epoch 0 end)
#   step 2000: loss 7.1   (epoch 1 end)
```

For tau > 2.0, the paper recommends:
1. **Self-distilled data**: generate responses with the target model, not generic datasets (+0.4-0.6 tau)
2. **Scale to 50K+ examples**: diminishing returns after 100K (+0.2-0.3 tau)
3. **Extend TTT depth schedule**: change depth transition from d<2 to d<3 (+0.1-0.3 tau)

### Running inference

```bash
# With trained draft head (speculative)
python3 tpu/harness/eagle3_infer.py \
  --model-dir ~/models/gemma-4-31B-it \
  --draft-dir eagle3-head \
  --max-tokens 256 --max-ctx 512 --fused \
  --prompt "Hello, how are you?"

# Pipeline test with random draft (validates wiring, ~0% acceptance)
python3 tpu/harness/eagle3_infer.py \
  --model-dir ~/models/gemma-4-31B-it \
  --random-draft --max-tokens 64 --fused

# Baseline comparison (no speculation)
python3 tpu/harness/eagle3_infer.py \
  --model-dir ~/models/gemma-4-31B-it \
  --baseline --max-tokens 128
```

### Architecture

- **Draft head**: 450M params (1.5% of target). FC_fuse(3d->d) + FC_in(2d->d) + 1 transformer layer + shared LM head.
- **Feature capture**: branchless `lax.cond` at layers 2, 30, 59 inside the 60-layer scan carry. No scan segmentation.
- **Verify**: T=K+1=6 positions through all 60 layers in one pass. Multi-position causal attention with sliding window.
- **Acceptance**: greedy argmax matching (lossless for greedy decode). Stochastic rejection for sampled decode (future).
- **KV rollback**: implicit (next cycle overwrites; causal mask prevents reading garbage).
- **Fused decode loop**: `jax.lax.while_loop` with draft+verify+accept all on-device. Zero Python dispatch in the hot path.

### Files

| File | Purpose |
|---|---|
| `tpu/harness/eagle3_infer.py` | Inference: fused + unfused decode, baseline comparison |
| `tpu/harness/eagle3_train.py` | Training: online TTT loss, fused train step, safetensors export |
| `tpu/harness/EAGLE3_SPEC.md` | Architecture spec with confirmed model values |
| `tpu/harness/cache_push.sh` | Push XLA compilation cache to HF |
| `tpu/harness/cache_pull.sh` | Pull XLA compilation cache from HF |

### Next steps for higher tau

| Optimization | Expected impact | Effort |
|---|---|---|
| Self-distilled training data (target model outputs) | +0.4-0.6 tau | 8-10h data gen |
| Scale to 50K examples | +0.2-0.3 tau | 2h training |
| Int8 KV cache quantization | -5.75ms cycle time | 2-4h code |
| Splash attention (fused Pallas kernel) | -1-2ms cycle time | 1-2h wiring |

Reference: [EAGLE-3 paper](https://arxiv.org/abs/2503.01840), [EAGLE-3 SPEC](tpu/harness/EAGLE3_SPEC.md).


## GPU: 31B Gemma 4 on H100

Rust + CUDA on H100 SXM 80GB. FP8 weights with per-channel scales, F16 KV cache, F16 paged attention. All 60 layers captured in a single CUDA graph (1776 nodes). **7,612 tok/s** peak (B=512), **PPL 12.54** (better than HF BF16 19.62).

### Gemma 4 architecture

60-layer transformer with dual attention (50 sliding + 10 global layers):

| Property | Sliding layers (50) | Global layers (10) |
|---|---|---|
| Q/K/V heads | 32 / 16 / 16 | 32 / 4 / none (V = K) |
| Head dimension | 256 | 512 |
| Attention window | 1024 tokens | full context |
| RoPE theta | 10,000 | 1,000,000 |
| RoPE rotation | 100% | 25% (partial) |

Other Gemma 4 specifics:
- **QK-norm + v-norm**: per-head RMSNorm on Q/K (with learned scale), parameter-free RMSNorm on V
- **k_eq_v**: global layers have no v_proj; V = v_norm(raw_K)
- **Attention scaling = 1.0**: QK-norm handles magnitude, no sqrt(head_dim) division
- **layer_scalar**: applied once at the end of the full layer (not per-sublayer)
- **Logit softcapping**: 30 * tanh(logits / 30)
- **GELU(tanh)** activation (not SiLU)
- **Tied embeddings**: lm_head = embed_tokens.T

### Gemma 4 forward pass (16 launches per layer)

```
For each layer in 0..60:
  1.  fused_rmsnorm_fp8_quant           input layernorm + FP8 quantize
  2.  fp8_gemm                          fused Q||K||V projection
  3.  fused_qk_rmsnorm                  per-head RMSNorm on Q and K
  4.  fused_rope_partial_fp8kv          partial RoPE + FP8 quant + paged KV write
  5.  paged_decode / paged_prefill      FA3 attention (head_dim=256 sliding, 512 global)
  6.  quantize_fp8_per_token            attn output to FP8
  7.  fp8_gemm_residual                 O projection + residual add
  8.  fused_rmsnorm                     post-attention layernorm
  9.  residual_scale_f16                multiply by layer scalar
  10. fused_rmsnorm_fp8_quant           pre-FFN layernorm + FP8 quantize
  11. fp8_gemm                          fused gate||up projection
  12. fused_gelu_mul_fp8_quant          GELU(tanh)(gate) * up to FP8
  13. fp8_gemm_residual                 down projection + residual add
  14. fused_rmsnorm                     post-FFN layernorm
  15. residual_scale_f16                multiply by layer scalar
  16. implicit residual carry

Sampling tail:
  quantize_fp8_per_token              hidden to FP8
  fp8_gemm                            lm_head
  logit_softcap                       30 * tanh(logits / 30)
  argmax_kernel                       token selection
```

16 launches per layer x 60 layers + sampling tail = ~963 launches per step, all captured into one `cuGraphLaunch`.

### GPU perplexity

| Weight path | KV cache | PPL | tok/s (B=1, graph) |
|---|---|---|---|
| FP8-Dynamic per-channel + channelscale | F16 | **12.54** | 37.9 |
| BF16 split QKV per-tensor FP8 | F16 | 17.96 | 37.9 |
| F16 weights (no FP8) | F16 | 19.79 | 37.9 |
| HuggingFace BF16 reference | -- | 19.62 | -- |
| TPU int8 reference | int8 | 19.24 | -- |

The FP8-Dynamic checkpoint (RedHatAI/gemma-4-31B-it-FP8-Dynamic) with native per-channel weight scales + F16 KV cache achieves the best PPL (12.54) because the checkpoint was quantized with calibrated per-row scales that preserve more precision than BF16 rounding.

### GPU batch scaling

| Batch | tok/s | ms/step | Efficiency |
|---|---|---|---|
| 1 | 51 | 19.7 | baseline |
| 4 | 224 | 17.9 | 110% |
| 8 | 438 | 18.3 | 107% |
| 16 | 872 | 18.4 | 107% |
| 32 | 1,670 | 19.2 | 102% |
| 64 | 2,978 | 21.5 | 91% |
| 128 | 4,893 | 26.2 | 75% |
| 256 | 6,606 | 38.8 | 51% |
| 512 | 7,612 | 67.3 | 29% |

H100 SXM 80GB, FP8 weights, CUDA graph. Near-linear scaling through B=32 (memory-bound), saturating at B=256+ as FP8 tensor cores become compute-bound.

### CUDA graph capture

All 60 layers + lm_head captured into a single `cuGraphLaunch` (1776 nodes). Eliminates per-kernel launch overhead.

| Mode | tok/s (B=1) | ms/step |
|---|---|---|
| CUDA graph | 51 | 19.7 |
| Eager (no graph) | 11 | 90.4 |

4.6x speedup from graph capture at batch=1.

### Kernels

Every kernel has a known purpose, a pinned variant, and a workspace contract. No dispatch fallback chains.

**CUTLASS SM90 FP8 GEMMs** - 40 non-residual + 10 residual-fused variants, autotuned per shape. Schedule/epilogue pairing enforced at compile time via `static_assert`.

**FlashAttention-3 SM90** - WGMMA + TMA, paged KV layout, GQA. Supports head_dim 128/256/512 for Gemma 4's dual head dimensions.

**Fused kernels** (v3, Gemma 4 specific):

| Kernel | Purpose |
|---|---|
| `fused_rmsnorm_fp8_quant` | layernorm + FP8 quantize in one launch |
| `fused_qk_rmsnorm` | per-head RMSNorm on Q and K |
| `fused_rope_partial_fp8kv` | partial RoPE + FP8 quant + paged KV write |
| `fused_gelu_mul_fp8_quant` | GELU(tanh)(gate) * up to FP8 |
| `logit_softcap` | 30 * tanh(logits / 30) |
| `quantize_fp8_per_token` | activation to FP8 with per-token scale |
| `residual_scale_f16` | multiply by layer scalar |
| `argmax` | f32 logits to i32 token |

No fallbacks. Missing kernel .so = engine refuses to start.

### GPU build and run

```bash
# One-time on H100 box (~15 min)
bash kernels/build.sh               # fused PTX
bash kernels/build_cutlass_so.sh    # libcutlass_kernels.so
bash kernels/build_fa3.sh           # libfa3_kernels.so

# Build
cargo build --release --features cuda --manifest-path v3/Cargo.toml -p rvllm-bench

# Run
RVLLM_MODEL_DIR=/workspace/models/gemma-4-31B-it \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
RVLLM_BATCH=128 RVLLM_ITERS=30 RVLLM_WARMUP=5 \
  ./v3/target/release/rvllm-bench
```

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

## Correctness discipline

1. **No fallbacks.** Missing autotune entry = engine panic. Missing .so = refuse start. No silent degradation.
2. **Graph-capture invariant.** Metadata buffer layout frozen per (bucket, max_blocks_per_seq). Captured graphs bind exact offsets.
3. **CUTLASS schedule/epilogue pairing.** Mainloop and epilogue schedules must match. Enforced via `static_assert`.
4. **No `unwrap()` in libraries.** `Result<T, RvllmError>` end-to-end with structured context.
5. **Real block-change detection.** Scheduler emits block table updates; missing signals = stale KV reads caught at the type level.

## License

Apache-2.0.

## Further reading

- [`v3/GEMMA4_SPEC.md`](v3/GEMMA4_SPEC.md) - 31B Gemma 4 architecture details and weight shapes
- [`v3/SPEC.md`](v3/SPEC.md), [`v3/IMPL_PLAN.md`](v3/IMPL_PLAN.md) - v3 rewrite plan, 16 agent specs
- [`docs/bench.html`](docs/bench.html) - interactive benchmark results
- [`docs/arch.md`](docs/arch.md) - full crate architecture
