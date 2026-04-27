# rvLLM

LLM inference engine. Rust+CUDA on GPU, JAX+XLA on TPU.

Three Gemma 4 models on TPU v6e-4: **E4B** (16,794 tok/s peak, 78.3 tok/s B=1, PPL 5.87), **26B-A4B MoE** (14,899 tok/s peak), **31B** (9,600 tok/s peak, 128K context). GPU: 31B on H100 at **8,786 tok/s** (FP8, CUDA graph, PPL 14.75). Zero custom kernels on TPU -- ~500 lines of JAX. Native Rust binary on GPU -- zero Python in the serving path.

**[Full benchmarks](https://docs.solidsf.com/docs/bench.html)**

## At a glance

| | E4B (4B) | 26B-A4B (MoE) | 31B TPU | 31B GPU | vLLM H100 |
|---|---|---|---|---|---|
| **B=1 tok/s** | **78.3** | 52.9 | 44.2 | 53 | 66.9 |
| **Peak tok/s** | **16,794** | 14,899 | 9,600 | 8,786 | 3,848 |
| **PPL** | **5.87** | 90.21 | 24.76 | 14.75 | - |
| **Cached TTFT** | **25.9 ms** | 35.3 ms | 73.3 ms | 63 ms | - |
| **Peak tok/s/$** | **3,230** | 2,865 | 1,846 | 4,576 | 2,004 |

TPU: v6e-4, $5.20/hr, int8, max-ctx 2048. GPU: H100 SXM, $1.92/hr, FP8. All measured.

## TPU: Gemma 4 on v6e-4

Pure JAX + XLA. No custom kernels. XLA compiles the entire forward pass to TPU machine code from a ~500 line JAX script. Three models, one codebase.

### Models supported

| Property | E4B (4B) | 26B-A4B (MoE) | 31B |
|---|---|---|---|
| Total / active params | ~4B / 4B | 26B / ~4B | 31B / 31B |
| Layers | 42 | 30 | 60 |
| Hidden size | 2,560 | 2,816 | 5,376 |
| Q / KV heads (sliding) | 8 / 2 | 16 / 8 | 32 / 16 |
| Q / KV heads (global) | 8 / 2 | 16 / 2 (V=K) | 32 / 4 (V=K) |
| Head dim (sliding / global) | 256 / 512 | 256 / 512 | 256 / 512 |
| Sliding window | 512 | 1,024 | 1,024 |
| MoE | none | 128 experts, top-8 | none |
| KV-shared layers | 18 (of 42) | 0 | 0 |
| Per-layer input injection | 256-d gated (5.6 GB embed) | none | none |

### Batch scaling (max-ctx 2048)

| Batch | E4B tok/s | 26B-A4B tok/s | 31B tok/s | vLLM H100 |
|---|---|---|---|---|
| 1 | 78 | 53 | 44 | 66.9 |
| 8 | 542 | 390 | 318 | 515 |
| 64 | 3,661 | 2,662 | 2,112 | 2,794 |
| 128 | 6,298 | 4,915 | 3,853 | 3,848 |
| 256 | 10,214 | 8,192 | 6,246 | 3,709 |
| 512 | 13,773 | 12,390 | 8,550 | 3,788 |
| **768** | **15,514** | **14,899** | **9,600** | 3,671 |
| **1024** | **16,794** | - | - | - |

### 31B context scaling (B=1)

| Context | ms/step | tok/s | Architecture | KV type |
|---|---|---|---|---|
| 512 | 12.79 | 78.2 | Single-scan, 60-layer scan + cond | bf16 |
| 2,048 | 22.6 | 44.2 | Single-scan | bf16 |
| 32K | ~66 | ~15 | Single-scan | bf16 |
| 64K | ~91 | ~11 | Split-cache, 10 groups x 6 | int8 |
| 128K | 40.56 | 24.7 | Split-cache + blockwise global | int8 |

Dual-path architecture auto-switches at the 32K boundary.

### TPU deployment

```bash
# Create TPU v6e-4 ($5.20/hr)
gcloud compute tpus tpu-vm create rvllm-gemma4 \
  --zone=us-east5-b --accelerator-type=v6e-4 --version=v2-alpha-tpuv6e \
  --boot-disk-size=200

# Install (30 seconds)
pip3 install 'jax[tpu]' huggingface_hub tokenizers \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Download model
huggingface-cli download google/gemma-4-E4B-it --local-dir ~/models/gemma-4-E4B-it

# Run E4B (78.3 tok/s B=1)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-E4B-it --max-tokens 200 --max-ctx 2048

# Run 31B batched (9,600 tok/s B=768)
LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_tpu_scoped_vmem_limit_kib=131072" \
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 2048 --batch 768

# 128K context (24.7 tok/s)
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --fused --max-tokens 200 --max-ctx 131072

# API server (OpenAI-compatible) -- imports model symbols from gemma4_tpu_infer.py,
# so --model-dir must be a directory its auto-detect can parse (standard HF layout).
python3 tpu/harness/api_server.py --model-dir ~/models/gemma-4-31B-it --port 8080

# Perplexity
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir ~/models/gemma-4-31B-it --perplexity --max-ctx 2048
```

No Docker. No conda. No torch. No vLLM. One pip install, one Python file, one command.


## EAGLE-3 Speculative Decoding (TPU, experimental)

450M-param draft head proposes K=5 tokens per cycle; the full 31B verifies K+1=6 in one forward pass. Lossless for greedy decode.

| Metric | Value |
|---|---|
| Baseline (B=1, 512 ctx) | 78.2 tok/s, 12.79 ms/step |
| EAGLE-3 fused cycle | 31.0 ms/cycle |
| Projected @ tau=3.5 | ~145 tok/s (1.8x) |
| Hardware ceiling | ~300 tok/s (3.8x) |

Requires 50K+ training examples for production tau. Current: 2K examples, loss 7.1, pipeline validated end-to-end. See [`tpu/harness/EAGLE3_SPEC.md`](tpu/harness/EAGLE3_SPEC.md).


## MiniMax-M2.7-NVFP4 on TPU v6e-8 (experimental)

Scaled the TPU stack up to a 230B-total / 10B-active MoE model (`lukealonso/MiniMax-M2.7-NVFP4`) on a single v6e-8 slice. The measured throughput below is from the legacy JAX reproduction harness; the production path is now Rust + M2 checkpoint loader + PJRT/XLA under `v3/crates/`, with Python/JAX quarantined as reference-only.

### Measured (v6e-8, spot, europe-west4-a, `M2_MOE=shardmap`)

| Metric | Value |
|---|---|
| Model load (136 GB via parallel ThreadPool) | 79 s |
| B=1 | 726 ms/step, 1.4 tok/s |
| **B=8, bf16 KV** | **146 ms/step, 54.7 tok/s** (`RVLLM_M2_MOE_IMPL=auto`, replicate-token MoE + inactive-expert skip) |
| **B=8, int8 KV** | **145 ms/step, 55.1 tok/s** (`RVLLM_M2_KV=int8`) |
| **B=16, int8 KV** | **155 ms/step, 103.5 tok/s** |
| **B=32, int8 KV** | **187 ms/step, 171.3 tok/s** (`replicate_tokens`; now included in `auto`) |
| B=64, int8 KV | OOM at ctx 2048 (`992 MB` allocation, `850 MB` free) |
| B=8, previous all-to-all MoE | 802 ms/step, 10.0 tok/s |
| B=8, opt-in Pallas bf16 matmul (`RVLLM_NVFP4_BACKEND=pallas`, `BN=512`) | 1105 ms/step, 7.2 tok/s |
| Full PPL gate (2047 scored tokens) | **6.73** |
| Correctness gate | **pass** (`0.0` PPL delta, 736-char generation prefix match) |
| Gen sample (256 tok) | Starts coherent, then repeats `\\vec{p}`; decode loop works, long-form coherence still needs attention/spec-decode work |
| Rust PJRT smoke | Compiles and launches an 8-partition B=8 graph; **not a model benchmark** because the layer body is zero/no-op |
| Rust int8 arena boot conversion | **233.1 GB materialized in 214.0 s** at B=8/ctx=2048 with Rayon (`180` worker threads, `181` LWPs, ~69-86 cores active); artifact: `tpu/out/m2/rust_xla/m2_int8_arena_b8_rayon.json` |

### What this demonstrates

- v6e-8 provisioning (spot), 130 GB HF download in 1m45s (authed)
- NVFP4 safetensors loader (modelopt 4-tensor format: packed + weight_scale + weight_scale_2 + input_scale)
- Correct bf16 checkpoint loading: safetensors bf16 payloads are bitcast from raw `uint16`, not numerically cast
- Expert-parallel sharding: 256 experts across 8 chips (32/chip) via `shard_map`; B=8/B=16/B=32 use replicate-token + `psum`, while the legacy path uses `jax.lax.all_to_all`
- `lax.scan` over 62 layers so XLA streams one layer's weights at a time (fits in 32 GB HBM/chip)
- Sigmoid+bias top-8 router with aux-loss-free scoring (`scoring_func: sigmoid`, `use_routing_bias: true`)
- Partial RoPE (rotary_dim=64 of head_dim=128), QK-norm per layer, GQA softmax
- Per-batch legacy JAX compile cache persisted to HuggingFace (`and-y/rvllm-m2-build`) so next boot skips install + JIT for reproduction runs
- Rust M2 runtime path now owns checkpoint indexing, safetensors reads, flat weight-arena planning/upload, decode MLIR generation, PPL/gen token loop, and the OpenAI-compatible server surface
- B=8/B=16/B=32 now use exact replicate-token MoE by default: tokens stay replicated across expert shards, each shard skips inactive local experts with `lax.cond`, and outputs combine with `psum`. This avoids padded all-to-all buckets and empty expert matmuls.
- `RVLLM_M2_KV=int8` adds Gemma-style per-vector int8 KV cache. It preserves the short PPL/coherence gate and keeps B=8/B=16 throughput flat while fitting B=32 at 171 tok/s.
- The Rust int8 arena path now converts MiniMax's NVFP4 expert weights to a flat int8 arena at boot. The first implementation was accidentally single-threaded (`NLWP=1`, one core at ~100% CPU). The Rayon version uses all host cores (`RAYON_NUM_THREADS=180`) and completed the 233.1 GB arena materialization in 214.0 seconds. This is a boot/setup speedup; real decode tok/s still depends on wiring the int8 MXU layer body.

### Rust + XLA status (no JAX runtime)

The Rust path now compiles and launches an 8-partition B=8 decode graph through PJRT/libtpu without Python or JAX. The executed artifact used a zero/no-op Mosaic decode-layer body, so it validates runtime plumbing only. It is **not** a model-quality or real matmul throughput number.

It currently owns:

- MiniMax M2 checkpoint indexing and safetensors schema validation
- Flat `i8` weight-arena planning for the 134 GB NVFP4 checkpoint
- int8/bf16 KV sizing and host input/output specs
- Decode StableHLO/MLIR emission for B=8/B=16/B=32 shapes
- Rust PJRT client wiring to `libtpu.so`
- OpenAI-compatible `/v1/chat/completions` server surface over `rvllm_xla::M2Runtime`

What we found while compiling the Rust MLIR directly on the TPU:

- Arbitrary custom-call names do not work on TPU. `call_target_name = "rvllm.m2.embed"` fails at XLA compile with `Custom emitter for rvllm.m2.embed not found`.
- The TPU backend has a registered lowering emitter for exactly `call_target_name = "tpu_custom_call"`.
- `tpu_custom_call` requires JSON `backend_config` shaped like `{"custom_call_config":{"body":"<base64 Mosaic bytecode>","serialization_format":1,"needs_layout_passes":true},"implicit_sharding":{"type":"MANUAL"}}`.
- The `body` is not our semicolon metadata string. It is serialized Mosaic MLIR bytecode produced by the Mosaic serde pass.
- Commit `8d92d99f8` emits the real `tpu_custom_call` target and JSON backend config from Rust. The TPU compile probe now gets past the custom-emitter lookup and fails at body deserialization: `Failed to deserialize the Mosaic module: Missing or invalid version attribute`. That is expected for the current empty placeholder body.
- Commit `d24a77eef` proves the native StableHLO path: a Rust-emitted B=8, ctx=2048, int8-KV smoke graph compiles through PJRT on `rvllm-m2` with no Python/JAX. The report marks `sweep[0].status = "compiled"` and covers the real M2 runtime signature (`191,069` weight tensors, `134.4 GB` planned arena, `2.08 GB` int8 KV buffer).
- Commit `a2148a9c3` removes the fake embed/final custom calls from the real graph. The full graph now emits native StableHLO placeholders for embed/final and exactly 62 `tpu_custom_call` layer calls. TPU compile now fails only on the fused decode-layer Mosaic body placeholder, which is the intended remaining blocker.
- Commit `5461bc2c2` adds 8-device PJRT argument-list execution. It ran a zero/no-op B=8 graph and writes `tpu/out/m2/rust_xla/m2_rust_xla_b8_zero_body_5461bc2c233f94696472f3a9b4452daf583bf1b4.json`; that artifact must not be compared to real model tok/s.
- Commit `d76482f88` parallelizes the boot-time NVFP4->int8 arena conversion with Rayon. The TPU VM run used `180` Rayon worker threads (`181` LWPs observed) and wrote `tpu/out/m2/rust_xla/m2_int8_arena_b8_rayon.json`: 233.1 GB total arena, 191,069 weight entries, 214.0 s end-to-end. The dense slots were reserved/zero-filled for this bench (`copy_dense_tensors=false`); expert NVFP4 weights/scales were actually converted.
- Therefore the remaining blocker is not Python, tokenization, serving, PJRT, or multi-device execution. It is linking a real serialized Mosaic body for the fused NVFP4 decode-layer kernel, then replacing the current native embed/final placeholders with real native StableHLO math.
- The HF repo `and-y/rvllm-m2-build` is a private dataset containing legacy JAX cache artifacts only. It is useful for reproduction, not for the Rust-native runtime.

### Known issues blocking production

1. **MoE dispatch overhead dominates at B=1** — 726 ms/step is far off the HBM-bandwidth ceiling. `shard_map` + per-expert `nvfp4_matmul` calls fail to fuse into a single MXU-tiled kernel. B=8/B=16/B=32 avoid padded token all-to-all now; B=1 still needs a better small-batch kernel.
2. **Real Rust decode still needs the fused NVFP4 Mosaic body**. `rvllm-server` now calls `rvllm_xla::M2Runtime` from `/v1/chat/completions`, and Rust PJRT can execute the 8-partition graph. The current executable artifact uses a zero layer body; production needs the fused NVFP4 decode-layer body serialized into TPU Mosaic custom-call format.
3. **Long-form generation degenerates**. The full gate scores PPL 6.73 and passes prefix matching, but the 256-token angular-momentum sample falls into repetitive math text after the coherent opening.

### Build list

1. Rust emitter: use TPU's real `tpu_custom_call` target and JSON backend-config contract
2. Native StableHLO fallback for embed/final logits so only the hard NVFP4 layer needs Mosaic
3. Mosaic/MLIR custom-call fused NVFP4->MXU matmul kernel; high-level Pallas was slower/crash-prone, so this needs serialized Mosaic body control
4. Batched prefill -> 16x TTFT (20-token prompt currently 14s)
5. Flatten API/infer wrappers onto the same scanned full-bench forward path
6. Async collective flags (`LIBTPU_INIT_ARGS`) -> around 5% at larger B
7. EAGLE-3 speculative decode (scaffold exists, needs draft-head training)

### Reproduce

One-shot deploy (requires v6e-8 quota in `europe-west4-a`):

```bash
SPOT=1 ZONE=europe-west4-a HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  bash tpu/harness/deploy_m2_tpu.sh
```

Rust-only compile/execute probe on the VM. This path does not run Python or JAX. The body below is a zero-body runtime probe, not the real NVFP4 matmul:

```bash
cd $HOME/runs/$SHA/v3
cargo run --release -p rvllm-xla --features tpu --bin m2_rust_decode_bench -- \
  --model-dir ../tpu/harness/m2_checkpoint_schema \
  --artifact-dir /tmp/m2_rust_xla \
  --out /tmp/m2_rust_xla/final.json \
  --decode-layer-body /tmp/m2_raw_layer_body/body_hidden_any_local_weight_b8.mlir \
  --decode-layer-body-format lowered \
  --execute-decode \
  --batch 8 \
  --ctx 2048 \
  --iters 10 --warmup 3 \
  --kv-cache int8 \
  --moe-impl raw-mosaic-sharded-weight
```

Full v6e-4-equivalent M2 run artifacts are in [`tpu/out/m2/full_equiv_bb800cc21/`](tpu/out/m2/full_equiv_bb800cc21/).


## GPU: 31B Gemma 4 on H100

Rust + CUDA on H100 SXM 80GB. FP8 weights with per-channel scales + CUTLASS channelscale epilogue, F16 KV cache, F16 paged attention (FA3 SM90). All 60 layers captured in a single CUDA graph (~935 nodes). **8,786 tok/s** peak (B=512), **PPL 14.75**, **TTFT 63 ms**.

### GPU batch scaling

| Batch | tok/s | ms/step | Scaling |
|---|---|---|---|
| 1 | 53 | 18.7 | 1.0x |
| 8 | 434 | 18.4 | 8.2x |
| 32 | 1,743 | 18.4 | 32.9x |
| 64 | 3,265 | 19.6 | 61.6x |
| 128 | 5,802 | 22.1 | 109.5x |
| 256 | 7,808 | 32.8 | 147.3x |
| **512** | **8,786** | 58.3 | **165.8x** |

### rvLLM vs vLLM on H100 (measured)

| Batch | rvLLM tok/s | vLLM tok/s | Delta |
|---|---|---|---|
| 1 | 53 | 69 | -23% |
| 32 | 1,743 | 1,748 | ~0% |
| 64 | **3,265** | 3,130 | +4% |
| **128** | **5,802** | 4,689 | **+24%** |
| 256 | **7,808** | 7,077 | +10% |
| 512 | **8,786** | 8,243 | +7% |

rvLLM overtakes vLLM at B=64 and leads by 24% at B=128.

### GPU perplexity

| Weight path | KV cache | PPL | tok/s (B=1) |
|---|---|---|---|
| **FP8-Dynamic + CUTLASS channelscale epilogue** | F16 | **14.75** | 53 |
| BF16 split QKV per-tensor FP8 | F16 | 17.96 | 37.9 |
| F16 weights (no FP8) | F16 | 19.79 | 37.9 |
| HuggingFace BF16 reference | -- | 19.62 | -- |

### Gemma 4 forward pass (14 launches per layer)

```
For each layer in 0..60:
  1.  fused_rmsnorm_fp8_quant           input layernorm + FP8 quantize
  2.  cutlass_fp8_gemm_channelscale     fused Q||K||V + channelscale epilogue
  3.  fused_qkv_rmsnorm                 Q/K norm (learned) + V norm (parameter-free)
  4.  fused_rope_partial_f16kv          partial RoPE + F16 KV cache write
  5.  paged_decode (FA3 SM90)           attention (head_dim=256 sliding, 512 global)
  6.  quantize_fp8_per_token            attn output to FP8
  7.  fp8_gemm                          O projection
  8.  fused_norm_add_residual           channelscale + rmsnorm + residual add
  9.  fused_rmsnorm_fp8_quant           pre-FFN layernorm + FP8 quantize
  10. cutlass_fp8_gemm_channelscale     fused gate||up + channelscale epilogue
  11. fused_gelu_mul_fp8_quant          GELU(tanh)(gate) * up to FP8
  12. fp8_gemm                          down projection
  13. fused_norm_add_residual           channelscale + rmsnorm + residual + layer_scalar

Sampling tail:
  fused_rmsnorm                       final layernorm
  f16_gemm_f32                        lm_head
  logit_softcap                       30 * tanh(logits / 30)
  argmax_kernel                       token selection
```

### Kernel fusion summary

Four rounds of fusion + custom CUTLASS epilogue reduced graph nodes from 1776 to ~935 (47% reduction):

| Fusion | Kernels eliminated | Nodes saved |
|---|---|---|
| f32_to_bf16 + rmsnorm + vector_add -> fused_norm_add_residual | 3 -> 1 (x2/layer) | 240 |
| scale_cols_f32 fused into norm+add kernel (O-proj, down) | 1 -> 0 (x2/layer) | 120 |
| residual_scale_f16 fused into post-ff norm+add | 1 -> 0 (x1/layer) | 60 |
| vnorm_f16 fused into qk_rmsnorm -> fused_qkv_rmsnorm | 2 -> 1 (x1/layer) | 60 |
| CUTLASS channelscale epilogue (QKV, gate_up) | 3 -> 1 (x2/layer) | 240+ |

The CUTLASS channelscale kernel uses a custom SM90 EVT epilogue that applies per-token activation scale (ColBroadcast) and per-channel weight scale (RowBroadcast) directly in the GEMM epilogue while the accumulator is still F32, then casts to F16.

**Help wanted:** The current CUTLASS kernel uses a 128x128x128 tile which is suboptimal for low-batch decode (M <= 16). A smaller tile variant (e.g. 64x64x128) would improve B=1-8 throughput. PRs welcome for additional tile shapes with autotune selection.

### GPU build and run

```bash
# One-time on H100 box (~15 min)
bash kernels/build.sh               # fused PTX
bash kernels/build_cutlass_so.sh    # libcutlass_kernels.so
bash kernels/build_fa3.sh           # libfa3_kernels.so

# Build
cargo build --release --features cuda --manifest-path v3/Cargo.toml -p rvllm-bench
```

The three build scripts populate `kernels/sm_90/` with the PTX blobs, `libcutlass_kernels.so`, `libfa3_kernels.so`, and `policy.json`. That directory is intentionally empty in the repo -- it is build output. The `RVLLM_*` env paths below resolve against it.

**Required env vars** (every rvLLM GPU binary takes the same set):

| Variable | What it points at |
|---|---|
| `RVLLM_MODEL_DIR` | HuggingFace-layout model directory (config.json, safetensors, tokenizer.json) |
| `RVLLM_KERNELS_DIR` | directory holding the built PTX blobs |
| `RVLLM_CUTLASS_SO` | path to `libcutlass_kernels.so` |
| `RVLLM_FA3_SO` | path to `libfa3_kernels.so` |
| `RVLLM_POLICY` | `policy.json` emitted by the autotune pass |

```bash
# Run
RVLLM_MODEL_DIR=/workspace/models/gemma-4-31B-it \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
RVLLM_BATCH=128 RVLLM_ITERS=30 RVLLM_WARMUP=5 \
  ./v3/target/release/rvllm-bench
```

`RVLLM_BATCH`, `RVLLM_ITERS`, and `RVLLM_WARMUP` are bench-only knobs; the five variables in the table above are the minimum set for any rvLLM GPU binary.

### Native HTTP server (experimental)

The native OpenAI-compatible server lives in [`v3/crates/rvllm-serve`](/Users/andy/rvllm/v3/crates/rvllm-serve) and produces a `rvllm-server` binary. For M2 it owns `/health`, `/v1/models`, and `/v1/chat/completions` over `rvllm_xla::M2Runtime`. Until tokenizer/chat-template support lands, M2 chat accepts `prompt_token_ids`; on non-TPU builds or missing Mosaic custom-call bodies it returns a backend-unavailable response. No Python server is on the new serving path.

### Kernels

Every kernel has a known purpose, a pinned variant, and a workspace contract. No dispatch fallback chains.

| Kernel | Purpose |
|---|---|
| `cutlass_fp8_gemm_channelscale` | SM90 FP8 GEMM with EVT channelscale epilogue (QKV, gate_up) |
| `fused_rmsnorm_fp8_quant` | layernorm + FP8 quantize in one launch |
| `fused_qkv_rmsnorm` | per-head RMSNorm on Q, K (learned) and V (parameter-free) |
| `fused_rope_partial_f16kv` | partial RoPE + F16 KV cache write |
| `fused_gelu_mul_fp8_quant` | GELU(tanh)(gate) * up to FP8 |
| `fused_norm_add_residual` | channelscale + RMSNorm + residual add (+ optional layer_scalar) |
| `logit_softcap` | 30 * tanh(logits / 30) |
| `quantize_fp8_per_token` | activation to FP8 with per-token scale |
| `argmax` | f32 logits to i32 token |

No fallbacks. Missing kernel .so = engine refuses to start.

## v3 crate map

```
v3/crates/
  rvllm-core         typed errors, IDs, dtype, shape, config, env
  rvllm-mem          HbmArena, Region, Stream, Event, PinnedBuf, CudaContextHandle
  rvllm-kernels      manifest (sha-pinned), PTX loader, kernel catalog
  rvllm-fused        8 fused-kernel launchers + pure-Rust f32 references
  rvllm-attention    FA3 SM90 paged decode/prefill dlopen
  rvllm-cutlass      FP8 variant catalog + schedule pairing trait + cuBLASLt wrapper
  rvllm-metadata     frozen-layout metadata per bucket (one upload path)
  rvllm-loader       safetensors mmap -> HBM + CPU-path FP8 quant + clamp gate
  rvllm-sampling     argmax tail, pinned DtoH
  rvllm-graph        captured-graph pool keyed on MetaLayoutHash
  rvllm-runtime      Engine, scheduler, layer_exec, bring_up
  rvllm-serve        HTTP / OpenAI-compatible serve loop (bin: rvllm-server)
  rvllm-bench        RVLLM_* env-driven bench binary
  rvllm-deploy       SHA-pinned tarball + manifest.json verification (bin: rvllm-deploy) -- Phase D scaffold
  rvllm-invariants   DAG-dep test, no-megakernel gate
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

- [`docs/bench.html`](https://docs.solidsf.com/docs/bench.html) - interactive benchmark results with charts
- [`v3/GEMMA4_SPEC.md`](v3/GEMMA4_SPEC.md) - 31B Gemma 4 architecture details and weight shapes
- [`v3/SPEC.md`](v3/SPEC.md), [`v3/IMPL_PLAN.md`](v3/IMPL_PLAN.md) - v3 rewrite plan, 16 agent specs
- [`tpu/harness/EAGLE3_SPEC.md`](tpu/harness/EAGLE3_SPEC.md) - EAGLE-3 speculative decoding spec
- [`docs/arch.md`](docs/arch.md) - full crate architecture
