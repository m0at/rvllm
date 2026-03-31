# 07: FP8 Quantization

Research document covering FP8 (E4M3) quantization as implemented in rvLLM -- weight quantization, KV cache compression, CUDA kernel design, cublasLt integration, and the end-to-end inference path.

## 1. FP8 E4M3 Format Overview

FP8 E4M3 is an 8-bit floating-point format with 1 sign bit, 4 exponent bits, and 3 mantissa bits:

```
[S | EEEE | MMM]
 7   6..3   2..0
```

Key properties:
- **Exponent bias:** 7 (same as IEEE convention for 4-bit exponent)
- **Normal range:** 2^(-6) to 448 (1.875 * 2^8)
- **Subnormal minimum:** 2^(-9) = 1.953125e-3
- **Max representable:** 448.0 (exp=15, mantissa=6 -- exp=15 mantissa=7 is reserved as NaN)
- **Precision:** ~3.6% relative error in the normal range (3 mantissa bits give 8 representable values per power of 2)

Compared to FP16 (5 exponent, 10 mantissa), FP8 E4M3 halves storage at the cost of reduced dynamic range and precision. The 4-bit exponent preserves range well ([-448, 448] vs FP16's [-65504, 65504]), while the 3-bit mantissa means values are quantized to the nearest 1/8th within each exponent band.

### Why E4M3, not E5M2?

E5M2 (5 exponent, 2 mantissa) has wider range but only 4 representable values per exponent band vs 8 for E4M3. For weight and KV cache quantization, the narrower range of E4M3 is sufficient (values are scaled to [-448, 448]), and the extra mantissa bit yields ~2x finer granularity. E5M2 is more useful for gradients during training where the wider dynamic range matters.

rTriton's IR supports both `ScalarType::F8E4M3` and `ScalarType::F8E5M2`, but all inference paths use E4M3 exclusively.

## 2. Architecture of FP8 in rvLLM

FP8 quantization operates at two independent levels:

1. **Weight quantization** -- Model weights stored as FP8 with per-row scales, halving weight memory and bandwidth during decode-path GEMV/GEMM operations.
2. **KV cache quantization** -- Key/value activations stored as FP8 with per-head scales in the paged cache, halving the dominant memory consumer during long-context inference.

These are independently controlled:
- Weights: `RVLLM_FP8_WEIGHTS=1` environment variable
- KV cache: `--kv-cache-dtype fp8` CLI flag or `RVLLM_FP8_KV=1`

### Crate layout

| Crate | FP8 role |
|-------|----------|
| `rvllm-gpu/src/fp8_quantize.rs` | CPU-side weight quantization (f16 -> FP8 E4M3 with per-row scales) |
| `rvllm-gpu/src/cublaslt_ops.rs` | cublasLt FP8 GEMM dispatch with cached plans |
| `rvllm-quant/src/dequant/fp8.rs` | CPU-side FP8 dequantization with compile-time LUT |
| `rvllm-quant/src/method.rs` | `QuantMethod::FP8` variant in quantization enum |
| `rvllm-quant/src/gemm.rs` | CPU-fallback fused dequant+GEMV for FP8 weights |
| `rvllm-kv-cache/src/fp8_cache.rs` | FP8 paged KV cache engine (GPU + CPU buffers) |
| `rvllm-model-runner/src/gpu_runner.rs` | Startup quantization: f16 weights -> FP8 on GPU |
| `rvllm-model-runner/src/gpu_layer.rs` | Forward pass integration for both FP8 GEMV and cublasLt paths |
| `rvllm-config/src/cache.rs` | `kv_cache_dtype` configuration field |
| `rtriton/src/ir.rs` | `ScalarType::F8E4M3` / `F8E5M2` in JIT IR |

### CUDA kernel files

| Kernel file | Kernels | Purpose |
|-------------|---------|---------|
| `kernels/gemv_fp8.cu` | `gemv_fp8_kernel`, `fused_add_norm_fp8_gemv_kernel`, `fused_norm_fp8_gemv_kernel`, `fused_silu_down_fp8_gemv_kernel` | M=1 decode GEMV with FP8 weights |
| `kernels/fp8_kv.cu` | `quantize_kv_kernel`, `dequantize_kv_kernel`, `quantize_paged_kv_kernel`, `dequantize_paged_kv_kernel` | KV cache FP8 quantize/dequantize |
| `kernels/cast_fp.cu` | `cast_f32_to_f16_kernel`, `cast_f16_to_f32_kernel` | Precision cast helpers |

## 3. Weight Quantization Path

### 3.1 Startup: f16 -> FP8 on CPU

When `RVLLM_FP8_WEIGHTS=1`, weight quantization occurs during model loading in `gpu_runner.rs`. The process for each layer:

1. Copy f16 weight from GPU to host
2. Quantize to FP8 E4M3 with per-row scales via `quantize_weight_fp8()`
3. Upload FP8 bytes and f16 scale vector back to GPU
4. Store as `CudaSlice<u8>` (data) + `CudaSlice<f16>` (scales)

All four projection matrices per layer are quantized:
- **Fused QKV:** `[qkv_dim, hidden]` -- query, key, value projections concatenated
- **O-proj:** `[hidden, q_dim]` -- output projection
- **Fused gate+up:** `[intermediate*2, hidden]` -- gate and up projections concatenated
- **Down:** `[hidden, intermediate]` -- MLP down projection

After quantization, an FP8 input scratch buffer is allocated:
```rust
let max_k = *[hidden, q_dim, intermediate].iter().max().unwrap();
self.fp8_input_scratch = Some(unsafe { stream.alloc::<u8>(max_k) }?);
```

This scratch is reused for casting activations to FP8 before cublasLt GEMM calls.

### 3.2 Per-row scaling

The CPU quantization in `fp8_quantize.rs` uses per-row dynamic scaling:

```rust
// For each row of the weight matrix:
let absmax = row.iter().map(|v| v.to_f32().abs()).fold(0.0, f32::max);
let scale = absmax / 448.0;  // FP8_E4M3_MAX
let inv_scale = 1.0 / scale;

// Quantize: fp8_byte = float_to_fp8_e4m3(f16_val * inv_scale)
```

Per-row scaling (vs per-tensor) is critical because different output neurons have different activation magnitudes. A single scale factor would waste precision on rows with small values.

The scale factors are stored as f16, adding `out_dim * 2` bytes overhead per weight matrix (negligible vs the 50% weight size reduction).

### 3.3 CPU dequantization with compile-time LUT

The `dequant/fp8.rs` module provides a 256-entry lookup table built at compile time:

```rust
pub(crate) const FP8_E4M3_LUT: [f32; 256] = build_fp8_lut();
```

This maps every possible FP8 byte to its f32 value in a single array lookup, avoiding runtime bit manipulation. Dequantization is then a table lookup + scale multiply, processed 8 elements at a time for SIMD-friendly access patterns.

This is used as the CPU fallback path and for testing. The GPU path uses hardware FP8 conversion or CUDA intrinsics.

## 4. CUDA Kernels for FP8 Weight GEMV

The file `kernels/gemv_fp8.cu` implements four kernels for M=1 decode with FP8 weights. All use the same core pattern: warp-per-row GEMV with 4-wide FP8 byte loads.

### 4.1 FP8 -> float conversion

On sm_89+ (Ada, Hopper), hardware conversion via `__nv_fp8_e4m3`:

```cuda
__device__ __forceinline__ float fp8e4m3_to_float(unsigned char fp8) {
    return float(*reinterpret_cast<const __nv_fp8_e4m3*>(&fp8));
}
```

This compiles to a single instruction on supported hardware.

### 4.2 Standalone GEMV (`gemv_fp8_kernel`)

```
y[out_dim] = weight_fp8[out_dim, in_dim] @ x[in_dim] * scale[out_dim]
Grid:  ((out_dim + 7) / 8, 1, 1)
Block: (256, 1, 1)  -- 8 warps, one row per warp
```

Each warp computes one output element. Within a warp, 32 lanes cooperate on the dot product over `in_dim`, loading 4 consecutive FP8 bytes per lane per iteration (128 elements per warp iteration). Warp shuffle reduction produces the final sum.

Key optimization: FP8 weights are 1 byte each vs 2 bytes for f16, so the kernel reads half the memory for the weight matrix. Since M=1 GEMV is entirely memory-bandwidth-bound, this yields a theoretical 2x speedup (in practice ~1.5-1.9x due to scale loads and conversion overhead).

### 4.3 Fused add + RMSNorm + GEMV (`fused_add_norm_fp8_gemv_kernel`)

Two-phase kernel:
1. **Phase 1:** Residual add + RMSNorm into shared memory (all threads cooperate)
2. **Phase 2:** GEMV with FP8 weights reading from shared memory

```
Shared memory: hidden_size * sizeof(float) + 8 * sizeof(float)
```

This eliminates 2 separate kernel launches and 2 global memory round-trips (write residual, write normed, read normed for GEMV). The normalized activations live in shared memory and are consumed directly by the GEMV phase.

Block 0 additionally writes the residual output to global memory (needed by subsequent layers).

### 4.4 Fused SiLU + down projection (`fused_silu_down_fp8_gemv_kernel`)

Single kernel that computes `SiLU(gate) * up` inline during the GEMV for the down projection:

```cuda
float g = __half2float(gate[i]);
float s = g / (1.0f + __expf(-g)) * __half2float(up[i]);
acc += fp8e4m3_to_float(w_row[i]) * row_sc * s;
```

This avoids materializing the intermediate `SiLU(gate) * up` tensor to global memory.

### 4.5 Launch configuration

All kernels use:
- `FP8_THREADS = 256` (8 warps)
- `FP8_RPB = 8` (rows per block = warps per block)
- `__launch_bounds__(256)` for occupancy hints

The 8 rows-per-block design means each block produces 8 output values, keeping SM occupancy high even for moderate `out_dim`.

## 5. cublasLt FP8 GEMM Path

For batched decode (M > 1) or when the fused GEMV kernels don't apply, rvLLM uses NVIDIA's cublasLt library for FP8 tensor core GEMMs.

### 5.1 Integration in `cublaslt_ops.rs`

The `CublasLtOps` struct wraps cublasLt with:
- A persistent 4 MiB workspace for split-K heuristics
- A plan cache keyed by `(m, n, k)` that stores descriptors + selected algorithm

```rust
pub fn fp8_gemm_a_bt_raw(
    &self, m: usize, n: usize, k: usize,
    input_fp8_ptr: u64, weight_fp8_ptr: u64, output_f16_ptr: u64,
) -> Result<()>
```

Layout: `C_f16[m,n] = A_fp8[m,k] @ B_fp8[n,k]^T`

The first call for a given shape creates matrix layouts (`CUDA_R_8F_E4M3` for inputs, `CUDA_R_16F` for output), runs the heuristic algorithm search, and caches the result. Subsequent calls with the same shape dispatch directly.

### 5.2 Forward path (cublasLt FP8 decode)

When FP8 weights are available and batch size is 1, the decode path in `gpu_layer.rs` follows this sequence:

```
Step 1: RMSNorm (separate kernel)
Step 2: Cast normed f16 -> FP8, then cublasLt FP8 GEMM for QKV
Step 3: QKV bias (if present)
Step 4: RoPE
Step 5: Attention (FlashAttention-3 decode)
Step 6: Cast attn_out -> FP8, cublasLt FP8 GEMM for O-proj
Step 7: Post-attention RMSNorm
Step 8: Cast normed2 -> FP8, cublasLt FP8 GEMM for gate+up
Step 9: SiLU * mul
Step 10: Cast fused_act -> FP8, cublasLt FP8 GEMM for down proj
```

Each GEMM requires a preceding f16-to-FP8 cast into the scratch buffer. This is a simple elementwise kernel (`cast_f16_to_fp8_kernel`) launched with ceil(n/256) blocks.

The cublasLt path is gated on `input.fp8_input_scratch_ptr != 0` and all four FP8 weight sets being present.

### 5.3 Fused GEMV fallback path

When cublasLt is not available or when shared memory is sufficient, the code falls back to the fused GEMV kernels from `gemv_fp8.cu`. This path fuses more operations (add+norm+GEMV) but requires `hidden_size * 4 + 32` bytes of shared memory per block. For large hidden sizes (>12288), this exceeds the 48 KiB default and requires `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` adjustment.

### 5.4 Mega kernel path

There is also a "mega kernel" path (`fused_cute_oproj_add_norm_gateup_fp8_gemv`) that fuses O-proj + residual add + norm + gate+up projection into a single launch. This is gated on O-proj fitting in L2 cache (`hidden * q_dim < 4 MB` in FP8 bytes). For 7B+ models where O-proj exceeds L2 capacity, every block redundantly reads the full O-proj weight, making it ~100x slower -- hence the size gate.

## 6. KV Cache FP8 Quantization

### 6.1 Motivation

The KV cache is often the single largest memory consumer during inference with long contexts. For a Qwen2.5-7B model with 32 layers, 8 KV heads, and head_dim=128:

```
Per-token KV (FP16): 32 layers * 2 (K+V) * 8 heads * 128 dim * 2 bytes = 128 KiB
Per-token KV (FP8):  32 layers * 2 (K+V) * 8 heads * 128 dim * 1 byte  = 64 KiB + scales
```

The scale overhead is `32 * 2 * 8 * 4 bytes = 2 KiB` per token (per-head f32 scales), so FP8 achieves approximately 48% savings.

### 6.2 `FP8CacheEngine`

The FP8 paged KV cache (`fp8_cache.rs`) is a drop-in replacement for the standard FP16 `CacheEngine`. Cache layout:

```
data:   [num_blocks, block_size, num_heads, head_dim] as u8 (FP8 E4M3)
scales: [num_blocks, block_size, num_heads] as f32 (per-head dynamic scale)
```

Allocation is per-layer, with both GPU and CPU staging buffers for offloading:

```rust
pub struct FP8CacheEngine {
    pub gpu_cache_data: Vec<(GpuBuffer<u8>, GpuBuffer<u8>)>,       // per-layer (K, V) u8
    pub gpu_cache_scales: Vec<(GpuBuffer<f32>, GpuBuffer<f32>)>,   // per-layer (K, V) f32
    pub cpu_cache_data: Vec<(CpuBuffer<u8>, CpuBuffer<u8>)>,       // CPU swap
    pub cpu_cache_scales: Vec<(Vec<f32>, Vec<f32>)>,                // CPU swap scales
}
```

### 6.3 Per-head dynamic scaling

Unlike weight quantization (static per-row scales computed once at load time), KV cache quantization uses dynamic per-head scaling computed at write time:

```rust
pub fn quantize_heads(input: &[f32], num_heads: usize, head_dim: usize) -> (Vec<u8>, Vec<f32>) {
    for h in 0..num_heads {
        let absmax = head_slice.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = (absmax / FP8_E4M3_MAX).max(1e-12);
        // quantize: fp8_val = float_to_fp8_e4m3(val * (1.0 / scale))
    }
}
```

Each attention head gets its own scale factor because different heads attend to different patterns and can have very different activation magnitudes. A shared scale would waste precision on heads with smaller activations.

### 6.4 GPU kernels for paged cache

The `fp8_kv.cu` kernels handle quantization/dequantization with paged addressing:

**`quantize_paged_kv_kernel`:** Quantizes f32 source cache into FP8 destination cache with slot-based addressing. Each thread block handles one (slot, head) pair. Two-phase: first compute per-head absmax via warp reduction, then quantize.

**`dequantize_paged_kv_kernel`:** Inverse operation, restoring f32 values from FP8 + scales.

The paged variants accept a `slot_mapping` array that maps token indices to physical cache slots, supporting the block-level memory management of the paged attention system.

### 6.5 Memory budget calculation

The `FP8CacheConfig` computes per-block memory including scale overhead:

```rust
pub fn block_bytes(&self) -> usize {
    let elements = block_size * num_heads * head_dim;
    let data_bytes = 2 * elements * dtype.element_bytes();  // K + V
    let scale_bytes = match dtype {
        FP8 => 2 * block_size * num_heads * size_of::<f32>(),  // K + V scales
        FP16 => 0,
    };
    data_bytes + scale_bytes
}
```

This is used by the block allocator to determine how many blocks fit in the GPU memory budget.

## 7. CPU Quantization Fallback

The `rvllm-quant` crate provides a complete CPU-side FP8 path used for:
- Model loading from FP8-quantized checkpoints
- Testing and validation
- CPU-only inference fallback

### `QuantMethod::FP8`

FP8 is a first-class member of the `QuantMethod` enum alongside GPTQ, AWQ, SqueezeLLM, and GGUF variants:

```rust
pub enum QuantMethod {
    None, GPTQ, AWQ, SqueezeLLM, FP8,
    GgufQ4_0, GgufQ4KM, GgufQ5_0, GgufQ5KM, GgufQ8_0,
}
```

Properties: 8 bits, no group size (per-tensor or per-row scaling), classified as quantized.

### `QuantizedLinear`

The `QuantizedLinear` layer wraps `QuantizedWeight` and provides `forward()` (fused dequant+GEMV) and `dequantize()` methods. For FP8 weights, the GEMV dequantization path uses the compile-time LUT:

```rust
QuantMethod::FP8 => {
    let scale = weight.scales[0];  // per-tensor scale
    for (dst, &byte) in buf.iter_mut().zip(src.iter()) {
        *dst = FP8_E4M3_LUT[byte as usize] * scale;
    }
}
```

The CPU path uses per-tensor scaling (single scale factor for the whole weight) rather than per-row, since per-row scales would require a scale lookup per output row during the GEMV -- acceptable overhead on GPU but slower on CPU.

## 8. When to Use FP8

### FP8 weights (`RVLLM_FP8_WEIGHTS=1`)

**Good for:**
- Single-stream decode latency (M=1 GEMV is memory-bandwidth-bound)
- Reducing weight memory footprint by ~50%
- Low-concurrency serving where decode dominates

**Not useful for:**
- High-throughput batch serving (M >= 8 with continuous batching). At these batch sizes, f16 tensor core GEMMs already saturate compute, and FP8 adds cast overhead with no bandwidth benefit.
- Prefill (large M, compute-bound, tensor cores dominate)

The codebase includes an explicit warning:
> "FP8 weights: improves single-stream decode latency but does NOT improve batched throughput. For high-concurrency serving, f16 is equivalent or faster."

### FP8 KV cache (`--kv-cache-dtype fp8`)

**Good for:**
- Long-context workloads where KV cache dominates VRAM
- Fitting larger batch sizes (more blocks = more concurrent sequences)
- Any workload -- the perplexity impact is minimal with per-head dynamic scaling

**Trade-offs:**
- Scale factor overhead (~3% additional memory per token)
- Quantize/dequantize kernel overhead on each cache write and attention read
- Slightly reduced attention precision (3-bit mantissa per KV element)

## 9. Precision Analysis

### FP8 E4M3 error characteristics

Within each exponent band, FP8 E4M3 has 8 evenly-spaced representable values. The maximum relative quantization error is:

```
max_relative_error = 1 / (2 * 8) = 6.25%  (half a step in the 3-bit mantissa)
```

In practice, per-row/per-head scaling maps the weight/activation distribution to use the full FP8 range, keeping average relative error well below the maximum.

The test suite validates round-trip error:
- Weight quantization: <15% relative error for values > 0.01 (generous tolerance for E4M3)
- KV cache: <25% relative error + 0.5 absolute tolerance (dominated by small values near zero)

### Impact on model quality

FP8 E4M3 weight quantization is generally considered "near-lossless" for inference:
- Llama-family and Qwen models show < 0.1 perplexity increase with FP8 weights
- KV cache FP8 has minimal impact on generation quality with per-head scaling
- The main risk is outlier activations that exceed the [-448, 448] range after scaling, which are clamped to the maximum representable value

## 10. Potential Improvements

### Near-term

1. **Per-row scales for CPU fallback GEMM.** Currently uses per-tensor scaling, losing precision vs the GPU per-row path.

2. **Fused FP8 attention.** Currently the FP8 KV cache is dequantized before attention. A fused `dequantize_and_dot_kernel` is declared in the header comments of `fp8_kv.cu` but not yet fully integrated -- this would avoid materializing the dequantized KV tensors.

3. **Static FP8 checkpoint support.** Currently FP8 weights are quantized from f16 at startup. Supporting native FP8 safetensors (as produced by tools like `llm-compressor` or `AutoFP8`) would eliminate the startup quantization cost.

### Medium-term

4. **FP8 prefill GEMMs.** Use cublasLt FP8 for prefill (large M) where the 2x bandwidth reduction could help for very large K dimensions, even though prefill is compute-bound.

5. **SmoothQuant-style activation scaling.** Pre-compute per-channel activation scales from calibration data and fold them into weight scales, reducing quantization error for outlier channels.

6. **Block-wise FP8 scaling.** Instead of per-row (weights) or per-head (KV cache), use smaller block sizes (e.g., 128-element groups) for finer-grained scaling, similar to how GPTQ/AWQ use group_size=128.

### Hardware-specific

7. **SM_89+ native FP8 tensor cores.** Ada and Hopper GPUs have native FP8 tensor core operations. The current cublasLt path already uses these when available, but custom Triton/CUTLASS kernels could exploit FP8 tensor cores in fused attention kernels.

8. **FP8 for Hopper TMA.** Hopper's Tensor Memory Accelerator can load FP8 data directly into shared memory with format conversion, potentially eliminating the explicit cast kernel before cublasLt GEMMs.
