# Throughput Optimization Spec: Close the vLLM Gap

## Problem
rvLLM plateaus at ~8,300 tok/s while Python vLLM reaches ~12,700 tok/s at high N on the same A100 80GB. The gap is kernel efficiency and scheduling, not VRAM.

## Root Causes (suspected)
1. Per-step memory allocations (alloc_zeros called every forward pass)
2. CUDA graph capture/replay not wired into production decode path
3. Separate Q/K/V projections instead of fused QKV
4. Separate gate/up projections instead of fused gate_up
5. CPU-GPU synchronization points between layers
6. Kernel launch overhead (many small kernels per layer)
7. FlashAttention kernel not warp-specialized
8. Scheduler doesn't overlap prefill chunks with decode
9. No async overlap between CPU scheduling and GPU compute
10. KV cache memory access patterns not coalesced

## Agent Assignments

| Agent | Optimization | Files |
|-------|-------------|-------|
| 1 | Pre-allocate activation buffers (reuse across steps) | gpu_runner.rs |
| 2 | Pre-allocate layer scratch buffers | gpu_layer.rs |
| 3 | Fused QKV projection kernel (single GEMM) | kernels/fused_qkv.cu, gpu_layer.rs |
| 4 | Fused gate+up projection kernel | kernels/fused_gate_up.cu, gpu_layer.rs |
| 5 | Wire CUDA graph capture into decode forward path | gpu_worker.rs, graph_runner.rs |
| 6 | Wire CUDA graph replay for decode steps | gpu_worker.rs, graph_runner.rs |
| 7 | Remove per-layer device.synchronize() calls | gpu_layer.rs, gpu_runner.rs |
| 8 | Async HtoD metadata upload (overlap with compute) | gpu_runner.rs |
| 9 | FA2 decode kernel: warp shuffle reduction | kernels/flash_attention.cu |
| 10 | FA2 decode kernel: vectorized KV cache loads | kernels/flash_attention.cu |
| 11 | Fused residual+RMSNorm (wire existing kernel) | gpu_layer.rs |
| 12 | Fused SiLU*mul+down_proj | gpu_layer.rs |
| 13 | Batch argmax across all sequences | kernels/argmax.cu |
| 14 | Scheduler: overlap prefill with decode batches | scheduler.rs |
| 15 | Engine: async step (submit next while current runs) | gpu_engine.rs |
| 16 | KV cache: vectorized reshape_and_cache (float4) | kernels/reshape_and_cache.cu |
| 17 | RMSNorm kernel: vectorized loads (float4) | kernels/rms_norm.cu |
| 18 | Embedding gather: vectorized loads (float4) | kernels/embedding_gather.cu |
| 19 | Reduce HtoD transfers (pack metadata into single buffer) | gpu_runner.rs |
| 20 | Profile: add CUDA events timing to identify actual bottleneck | gpu_runner.rs, gpu_layer.rs |

## Rules
- Each agent edits ONLY their assigned files
- New kernel files go in kernels/ and must be registered in kernel_loader.rs KERNEL_FUNCTIONS
- All optimizations must be gated behind feature flags or runtime checks where possible
- Existing behavior must not break (777+ tests must pass)
- Focus on the DECODE path (that's where high-N throughput matters)
