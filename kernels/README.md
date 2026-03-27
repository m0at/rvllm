# vllm-rs CUDA Kernels

CUDA kernel sources for GPU-accelerated vLLM operations. Compiled to PTX and loaded at runtime via the CUDA driver API.

## Kernels

| File | Kernel | Purpose |
|------|--------|---------|
| `paged_attention.cu` | `paged_attention_v2_kernel` | PagedAttention V2 with block tables and online softmax |
| `rms_norm.cu` | `rms_norm_kernel` | RMSNorm with shared memory reduction |
| `rotary_embedding.cu` | `rotary_embedding_kernel` | RoPE with GQA support (separate num_heads / num_kv_heads) |
| `activation.cu` | `silu_kernel`, `fused_silu_mul_kernel`, `gelu_kernel` | Activation functions for MLP layers |
| `copy_blocks.cu` | `copy_blocks_kernel` | KV cache block copy for beam search / prefix caching |
| `softmax.cu` | `softmax_kernel` | Numerically stable softmax with warp-level reduction |

## Building

```bash
# Default: compile for A100 (sm_80)
./build.sh

# Custom architecture
CUDA_ARCH=sm_90 ./build.sh   # H100
CUDA_ARCH=sm_89 ./build.sh   # L40/RTX 4090
```

Produces `.ptx` files alongside the `.cu` sources. The Rust runtime loads these via `cuModuleLoadDataEx`.

## Integration

From Rust, load PTX at runtime:

```rust
let ptx = include_str!("../kernels/paged_attention.ptx");
// or load from file at runtime for development
```
