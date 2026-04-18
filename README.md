# rvLLM

LLM inference engine. Rust+CUDA on GPU, JAX+XLA on TPU.

**31B Gemma 4 on TPU v6e-4: 13,943 tok/s** (B=768, int8, TP=4 SPMD, PPL 25.51). 2,681 tok/s/$. Zero custom kernels - ~500 lines of JAX, XLA compiles everything.

## TPU: 31B Gemma 4 on v6e-4

Pure JAX + XLA. No custom kernels. XLA compiles the entire 60-layer forward pass to TPU machine code from a ~500 line JAX script.

### Headline numbers

| Metric | B=1 | B=8 | B=768 (peak) |
|---|---|---|---|
| **Decode throughput** | **79.9 tok/s** | **584 tok/s** | **13,943 tok/s** |
| **Per-step latency** | **12.5 ms** | **13.7 ms** | **55.1 ms** |
| **Latency p99** | **14.8 ms** | | |
| **Latency jitter (std)** | **0.07 ms** | | |
| **Perplexity** | **25.51** | | |
| **Cost efficiency** | **15.4 tok/s/$** | **112 tok/s/$** | **2,681 tok/s/$** |

| Fixed | Value |
|---|---|
| Model | 31B Gemma 4 (google/gemma-4-31B-it) |
| Hardware | TPU v6e-4 (4 chips, 128 GB HBM, ~3.3 TB/s) |
| Quantization | int8 per-channel (weights), bf16 activations |
| Perplexity | 25.51 (verified against HuggingFace reference) |
| Cost | ~$5.20/hr (v6e-4 on-demand) |

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

### TPU vs GPU cost comparison

| Hardware | Peak tok/s | Batch | Cost/hr | tok/s/$ |
|---|---|---|---|---|
| **TPU v6e-4 (rvLLM)** | **13,943** | **768** | **$5.20** | **2,681** |
| H100 SXM (FP8, projected) | ~6,000 | ~256 | $8-10 | 600-750 |
| H200 (bf16, projected) | ~4,000 | ~128 | $8-12 | 333-500 |

3.5-8x better cost efficiency than GPU.

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

The 1.8 ms cond overhead is structural: Gemma 4's dual attention (50 sliding + 10 global layers) requires runtime dispatch. Flat scan + cond is the optimum for XLA's TPU compiler.

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

# Run (first call: ~5s JIT compile, then 79.9 tok/s)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 512

# Batched (13,943 tok/s)
LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_dot_dot_fusion_duplicated=true" \
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 512 --batch 768

# Perplexity (25.51)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --perplexity --max-ctx 512

# Cleanup
gcloud compute tpus tpu-vm delete rvllm-gemma4 --zone=us-east5-b --quiet
```

No Docker. No conda. No torch. No vLLM. One pip install, one Python file, one command.


## GPU: 31B Gemma 4 on H100

Rust + CUDA. 16-kernel-launch fused pipeline for Gemma 4's dual-attention architecture. All 60 layers captured in a single CUDA graph. FP8 E4M3 weights quantized at load with calibrated per-tensor scales.

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

### GPU perplexity status

The GPU forward pass for 31B Gemma 4 is implemented and runs end-to-end. Perplexity validation against the TPU baseline (25.51) is actively being debugged - tracking logit softcap and F32 precision through lm_head GEMM. Throughput numbers will be posted once perplexity is confirmed correct.

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
