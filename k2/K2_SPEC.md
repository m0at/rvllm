# Kimi K2.6 on rvLLM -- CPU-offload MoE inference

## Target hardware
- 1x A40 48GB (SM86) + 1TB host RAM + 128 vCPUs @ 3.7GHz
- vast.ai instance 35318164 (ssh4.vast.ai:38164)
- Actual successful bench box on April 21, 2026: 1x A40 48GB + ~2.0TiB RAM + 2.4TB disk
- Meaningful rerun target: 1x H100/H200 80GB + >=900GB available host RAM + >=1.2TB free disk

## Model architecture (DeepSeek V3 variant)
- 1T total params, 32B activated, 61 layers (layer 0 dense, 1-60 MoE)
- MLA attention: q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, 64 heads
- MoE: 384 routed experts (INT4 group=32 symmetric), top-8, sigmoid scoring, routed_scaling_factor=2.827
- 1 shared expert per MoE layer: intermediate_size=18432, SiLU (BF16, NOT quantized)
- Dense layer 0: intermediate_size=18432, SiLU (BF16)
- Vocab: 163840, RoPE theta=50000, YaRN scaling (factor=64, orig_max=4096)
- KV cache per token: (512 + 64) * 2 bytes = 1152 bytes (MLA compressed)

## Quantization (from config)
- Method: compressed-tensors, pack-quantized, INT4 group_size=32 symmetric
- Quantized: routed expert weights ONLY
- NOT quantized (BF16): attention, shared_experts, dense MLP, lm_head, embeddings

## Artifact sizes
- HF safetensors repo (`moonshotai/Kimi-K2.6`): ~555G on disk
- GGUF Q4_X repo (`ubergarm/Kimi-K2.6-GGUF`, `Q4_X/`): ~544G on disk
- Practical download budget for one box: at least 1.2TB free disk before logs, env, and scratch

## Memory layout

### GPU (A40 48GB)
- Attention weights BF16: ~12.2GB (61 layers)
- Embedding BF16: 2.2GB
- LM head BF16: 2.2GB
- Router weights BF16: 0.3GB (60 MoE layers)
- Shared expert INT8 (quantized at load): ~23.7GB (60 MoE layers)
- Dense MLP layer 0 BF16: ~0.8GB
- Total: ~41.4GB, leaves ~6.6GB for KV cache (~95K tokens)

### CPU (1TB RAM)
- Routed expert weights INT4 mmap'd: ~535GB
- Dequant scratch: ~1.5GB (8 experts at FP32)

## Per-step data flow (B=1 decode)
```
For each layer 0..60:
  GPU: RMSNorm(x)
  GPU: MLA attention (q_a_proj -> q_a_norm -> q_b_proj -> split q_nope/q_rope
                       kv_a_proj -> split c_kv/k_rope -> kv_a_norm -> kv_b_proj
                       -> attention -> o_proj)
  GPU: residual add
  GPU: RMSNorm(x)
  if layer == 0 (dense):
    GPU: dense MLP (gate_proj, up_proj, SiLU, down_proj)
  else (MoE):
    GPU: router sigmoid -> top-8 selection
    GPU: shared expert MLP (INT8 cuBLAS)
    CPU: 8 routed expert MLPs (INT4 dequant + matmul, rayon parallel)
    GPU: weighted combine + residual
GPU: final RMSNorm -> lm_head -> sample
```

## Observed A40 status (April 21, 2026)
- KTransformers + SGLang + `KT_METHOD=LLAMAFILE` does serve K2.6 correctly on the A40 box.
- Smoke, TTFT, throughput, and chunked PPL all completed end-to-end.
- Result quality is not the blocker; speed is.

### Observed micro-benchmark
- Artifact: `/workspace/k2_bench_final_micro.json` on instance `35318164`
- Smoke: 1 token in 12.25s (`0.0816 tok/s`)
- TTFT: `26894.6 ms`
- Stream: 4 tokens in 68.56s (`0.0583 tok/s`)
- Throughput @ conc=1: `0.0354 tok/s`
- Throughput @ conc=2: `0.0395 tok/s`
- Throughput @ conc=4: `0.0478 tok/s`
- Throughput @ conc=8: `0.0517 tok/s`
- Chunked PPL: `105.58` over 58 scored tokens with 64-char chunks

### Practical conclusion
- The A40 path is a functional proof, not a meaningful production benchmark.
- As of April 21, 2026, the visible 1x H100/H200 Vast offers with enough disk did not also have enough host RAM for this offload setup.
- A useful rerun needs a box that satisfies both the GPU target and the storage/RAM target at the same time.

## Current implementation notes

### Files
- `k2/deploy.sh` - remote preflight plus HF and GGUF background downloads
- `k2/kt_a40_launch.sh` - SGLang + KTransformers launch wrapper for the A40 proof box
- `k2/kt_bench.py` - smoke, TTFT, throughput, and chunked-PPL benchmark harness
- `k2/infer.py`, `k2/model.py`, `k2/loader.py` - earlier Python prototype files kept in-tree, but not the served path used for the final proof

### Runtime caveat
- Large-prefill or full-context fallback in the current SGLang KTransformers path can hit a missing LLAMAFILE wrapper method (`submit_write_weight_scale_to_buffer`) inside the MoE full-context path.
- The current workaround is chunked PPL with small enough chunks to stay below `kt_gpu_prefill_token_threshold`.
