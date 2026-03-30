//! CudaCacheEngine: allocates real GPU memory for KV cache blocks via cudarc.
//!
//! This is the CUDA-backed counterpart to [`CacheEngine`](super::CacheEngine),
//! which uses the abstract `GpuAllocator` trait. `CudaCacheEngine` works
//! directly with `cudarc::driver::CudaSlice<half::f16>` for zero-copy kernel
//! interop and avoids the overhead of the mock-gpu abstraction.
//!
//! The KV cache stores values in f16, halving VRAM usage compared to f32 and
//! doubling the number of blocks we can allocate for a given memory budget.
//! CUDA kernels that read/write the cache accept f16 pointers directly;
//! computation (attention scores, softmax) still happens in f32 for precision.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use tracing::{debug, info};

use rvllm_core::prelude::{BlockId, LLMError, Result};

/// Per-layer paged KV cache engine backed by real CUDA device memory.
///
/// Each transformer layer owns a `(key, value)` pair of `CudaSlice<f16>`
/// buffers. The buffers are logically divided into fixed-size blocks:
///
///   `[num_blocks, block_size, num_heads, head_dim]`  (flattened)
///
/// Block-level operations (copy, swap-in, swap-out) transfer data between
/// GPU and host without going through the `GpuBuffer` abstraction.
pub struct CudaCacheEngine {
    /// Per-layer (key_cache, value_cache) on GPU in f16.
    gpu_cache: Vec<(CudaSlice<f16>, CudaSlice<f16>)>,
    /// CPU-side staging buffers for swap operations, per layer (key, value) in f16.
    cpu_cache: Vec<(Vec<f16>, Vec<f16>)>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaCacheEngine {
    /// Allocate GPU and CPU KV cache buffers for all layers.
    ///
    /// # Errors
    /// Returns `LLMError::GpuError` if any CUDA allocation fails.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let elements_per_block = block_size * num_heads * head_dim;
        let gpu_total = num_gpu_blocks * elements_per_block;
        let cpu_total = num_cpu_blocks * elements_per_block;

        info!(
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            gpu_bytes = gpu_total * std::mem::size_of::<f16>() * 2,
            "CudaCacheEngine: allocating f16 KV cache"
        );

        let mut gpu_cache = Vec::with_capacity(num_layers);
        let mut cpu_cache = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            debug!(layer, gpu_total, "allocating f16 CUDA KV cache");

            let key_gpu: CudaSlice<f16> = stream.alloc_zeros(gpu_total).map_err(|e| {
                LLMError::GpuError(format!("CUDA key cache alloc failed layer {layer}: {e}"))
            })?;
            let val_gpu: CudaSlice<f16> = stream.alloc_zeros(gpu_total).map_err(|e| {
                LLMError::GpuError(format!("CUDA value cache alloc failed layer {layer}: {e}"))
            })?;
            gpu_cache.push((key_gpu, val_gpu));

            debug!(layer, cpu_total, "allocating f16 CPU staging cache");
            let key_cpu = vec![f16::ZERO; cpu_total];
            let val_cpu = vec![f16::ZERO; cpu_total];
            cpu_cache.push((key_cpu, val_cpu));
        }

        Ok(Self {
            gpu_cache,
            cpu_cache,
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            context,
            stream,
        })
    }

    /// Number of elements per cache block (dtype-independent count).
    pub fn elements_per_block(&self) -> usize {
        self.block_size * self.num_heads * self.head_dim
    }

    /// Reference to the CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Reference to the CUDA stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Tokens per cache block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of allocated GPU blocks.
    pub fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    /// Number of allocated CPU staging blocks.
    pub fn num_cpu_blocks(&self) -> usize {
        self.num_cpu_blocks
    }

    /// Access the per-layer GPU cache slices (f16).
    pub fn gpu_cache(&self) -> &[(CudaSlice<f16>, CudaSlice<f16>)] {
        &self.gpu_cache
    }

    /// Mutable access to the per-layer GPU cache slices (f16).
    pub fn gpu_cache_mut(&mut self) -> &mut [(CudaSlice<f16>, CudaSlice<f16>)] {
        &mut self.gpu_cache
    }

    /// Copy blocks within GPU cache. Each `(src, dst)` pair copies a full
    /// block across all layers by round-tripping through the host.
    ///
    /// A kernel-based copy_blocks (Agent 9) can replace this with a direct
    /// device-to-device copy for better performance.
    pub fn copy_blocks(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let epb = self.elements_per_block();

        for &(src_id, dst_id) in mapping {
            let src = src_id.0 as usize;
            let dst = dst_id.0 as usize;

            if src >= self.num_gpu_blocks || dst >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "copy_blocks: block index out of range (src={src}, dst={dst}, max={})",
                    self.num_gpu_blocks
                )));
            }

            let src_off = src * epb;
            let dst_off = dst * epb;

            for (key_buf, val_buf) in &mut self.gpu_cache {
                let mut key_host = self
                    .stream
                    .clone_dtoh(key_buf)
                    .map_err(|e| LLMError::GpuError(format!("copy_blocks dtoh key: {e}")))?;
                let src_slice: Vec<f16> = key_host[src_off..src_off + epb].to_vec();
                key_host[dst_off..dst_off + epb].copy_from_slice(&src_slice);
                self.stream
                    .memcpy_htod(&key_host, key_buf)
                    .map_err(|e| LLMError::GpuError(format!("copy_blocks htod key: {e}")))?;

                let mut val_host = self
                    .stream
                    .clone_dtoh(val_buf)
                    .map_err(|e| LLMError::GpuError(format!("copy_blocks dtoh val: {e}")))?;
                let src_slice: Vec<f16> = val_host[src_off..src_off + epb].to_vec();
                val_host[dst_off..dst_off + epb].copy_from_slice(&src_slice);
                self.stream
                    .memcpy_htod(&val_host, val_buf)
                    .map_err(|e| LLMError::GpuError(format!("copy_blocks htod val: {e}")))?;
            }
        }

        debug!(
            pairs = mapping.len(),
            "CudaCacheEngine copy_blocks complete"
        );
        Ok(())
    }

    /// Swap blocks from CPU staging cache into GPU cache.
    /// Each `(cpu_block, gpu_block)` copies CPU -> GPU across all layers.
    pub fn swap_in(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let epb = self.elements_per_block();

        for &(cpu_id, gpu_id) in mapping {
            let cpu_idx = cpu_id.0 as usize;
            let gpu_idx = gpu_id.0 as usize;

            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_in: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }
            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_in: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }

            let cpu_off = cpu_idx * epb;
            let gpu_off = gpu_idx * epb;

            for (layer_idx, ((key_gpu, val_gpu), (key_cpu, val_cpu))) in self
                .gpu_cache
                .iter_mut()
                .zip(self.cpu_cache.iter())
                .enumerate()
            {
                let mut key_host = self.stream.clone_dtoh(key_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_in dtoh key layer {layer_idx}: {e}"))
                })?;
                key_host[gpu_off..gpu_off + epb].copy_from_slice(&key_cpu[cpu_off..cpu_off + epb]);
                self.stream
                    .memcpy_htod(&key_host, key_gpu)
                    .map_err(|e| {
                        LLMError::GpuError(format!("swap_in htod key layer {layer_idx}: {e}"))
                    })?;

                let mut val_host = self.stream.clone_dtoh(val_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_in dtoh val layer {layer_idx}: {e}"))
                })?;
                val_host[gpu_off..gpu_off + epb].copy_from_slice(&val_cpu[cpu_off..cpu_off + epb]);
                self.stream
                    .memcpy_htod(&val_host, val_gpu)
                    .map_err(|e| {
                        LLMError::GpuError(format!("swap_in htod val layer {layer_idx}: {e}"))
                    })?;
            }
        }

        debug!(pairs = mapping.len(), "CudaCacheEngine swap_in complete");
        Ok(())
    }

    /// Swap blocks from GPU cache out to CPU staging cache.
    /// Each `(gpu_block, cpu_block)` copies GPU -> CPU across all layers.
    pub fn swap_out(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let epb = self.elements_per_block();

        for &(gpu_id, cpu_id) in mapping {
            let gpu_idx = gpu_id.0 as usize;
            let cpu_idx = cpu_id.0 as usize;

            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_out: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }
            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_out: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }

            let gpu_off = gpu_idx * epb;
            let cpu_off = cpu_idx * epb;

            for (layer_idx, ((key_gpu, val_gpu), (key_cpu, val_cpu))) in self
                .gpu_cache
                .iter()
                .zip(self.cpu_cache.iter_mut())
                .enumerate()
            {
                let key_host = self.stream.clone_dtoh(key_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_out dtoh key layer {layer_idx}: {e}"))
                })?;
                key_cpu[cpu_off..cpu_off + epb].copy_from_slice(&key_host[gpu_off..gpu_off + epb]);

                let val_host = self.stream.clone_dtoh(val_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_out dtoh val layer {layer_idx}: {e}"))
                })?;
                val_cpu[cpu_off..cpu_off + epb].copy_from_slice(&val_host[gpu_off..gpu_off + epb]);

                let _ = layer_idx;
            }
        }

        debug!(pairs = mapping.len(), "CudaCacheEngine swap_out complete");
        Ok(())
    }

    /// Compute the maximum number of GPU blocks that fit in `available_bytes`
    /// given the current cache configuration. Cache is stored in f16.
    pub fn max_blocks_for_memory(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        available_bytes: usize,
    ) -> usize {
        let elements_per_block = block_size * num_heads * head_dim;
        // key + value, each f16, across all layers
        let bytes_per_block = 2 * num_layers * elements_per_block * std::mem::size_of::<f16>();
        if bytes_per_block == 0 {
            return 0;
        }
        available_bytes / bytes_per_block
    }
}

/// Per-layer paged KV cache engine storing data in FP8 E4M3 (u8) with
/// per-head f32 scale factors, backed by real CUDA device memory.
///
/// Halves the VRAM footprint compared to [`CudaCacheEngine`]'s f16 storage.
/// The cache layout is:
///   data:   `[num_blocks, block_size, num_heads, head_dim]` as u8
///   scales: `[num_blocks, block_size, num_heads]` as f32
///
/// On write (`reshape_and_cache_fp8`), f16 K/V are quantized to FP8.
/// On read (`dequantize_block`), FP8 data is expanded back to f16 for
/// the attention kernel. A fused FP8 attention kernel can skip this later.
pub struct CudaFP8CacheEngine {
    /// Per-layer (key_data, value_data) as u8 on GPU.
    gpu_cache_data: Vec<(CudaSlice<u8>, CudaSlice<u8>)>,
    /// Per-layer (key_scales, value_scales) as f32 on GPU.
    gpu_cache_scales: Vec<(CudaSlice<f32>, CudaSlice<f32>)>,
    /// CPU-side staging for data (u8).
    cpu_cache_data: Vec<(Vec<u8>, Vec<u8>)>,
    /// CPU-side staging for scales (f32).
    cpu_cache_scales: Vec<(Vec<f32>, Vec<f32>)>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaFP8CacheEngine {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let data_elements = block_size * num_heads * head_dim;
        let scale_elements = block_size * num_heads;
        let gpu_data_total = num_gpu_blocks * data_elements;
        let gpu_scale_total = num_gpu_blocks * scale_elements;
        let cpu_data_total = num_cpu_blocks * data_elements;
        let cpu_scale_total = num_cpu_blocks * scale_elements;

        let fp16_equiv_bytes = gpu_data_total * 2 * std::mem::size_of::<f16>();
        let fp8_bytes = gpu_data_total * 2 + gpu_scale_total * 2 * std::mem::size_of::<f32>();

        info!(
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            fp16_equiv_bytes,
            fp8_bytes,
            savings_pct = ((fp16_equiv_bytes as f64 - fp8_bytes as f64) / fp16_equiv_bytes as f64 * 100.0) as u32,
            "CudaFP8CacheEngine: allocating FP8 KV cache"
        );

        let mut gpu_cache_data = Vec::with_capacity(num_layers);
        let mut gpu_cache_scales = Vec::with_capacity(num_layers);
        let mut cpu_cache_data = Vec::with_capacity(num_layers);
        let mut cpu_cache_scales = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            debug!(layer, gpu_data_total, "allocating FP8 CUDA KV cache");

            let key_data: CudaSlice<u8> = stream.alloc_zeros(gpu_data_total).map_err(|e| {
                LLMError::GpuError(format!("FP8 CUDA key data alloc layer {layer}: {e}"))
            })?;
            let val_data: CudaSlice<u8> = stream.alloc_zeros(gpu_data_total).map_err(|e| {
                LLMError::GpuError(format!("FP8 CUDA value data alloc layer {layer}: {e}"))
            })?;
            gpu_cache_data.push((key_data, val_data));

            let key_scales: CudaSlice<f32> = stream.alloc_zeros(gpu_scale_total).map_err(|e| {
                LLMError::GpuError(format!("FP8 CUDA key scales alloc layer {layer}: {e}"))
            })?;
            let val_scales: CudaSlice<f32> = stream.alloc_zeros(gpu_scale_total).map_err(|e| {
                LLMError::GpuError(format!("FP8 CUDA value scales alloc layer {layer}: {e}"))
            })?;
            gpu_cache_scales.push((key_scales, val_scales));

            cpu_cache_data.push((vec![0u8; cpu_data_total], vec![0u8; cpu_data_total]));
            cpu_cache_scales.push((vec![0.0f32; cpu_scale_total], vec![0.0f32; cpu_scale_total]));
        }

        Ok(Self {
            gpu_cache_data,
            gpu_cache_scales,
            cpu_cache_data,
            cpu_cache_scales,
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            context,
            stream,
        })
    }

    pub fn data_elements_per_block(&self) -> usize {
        self.block_size * self.num_heads * self.head_dim
    }

    pub fn scale_elements_per_block(&self) -> usize {
        self.block_size * self.num_heads
    }

    pub fn context(&self) -> &Arc<CudaContext> { &self.context }
    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
    pub fn num_layers(&self) -> usize { self.num_layers }
    pub fn num_heads(&self) -> usize { self.num_heads }
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn block_size(&self) -> usize { self.block_size }
    pub fn num_gpu_blocks(&self) -> usize { self.num_gpu_blocks }

    pub fn gpu_cache_data(&self) -> &[(CudaSlice<u8>, CudaSlice<u8>)] {
        &self.gpu_cache_data
    }

    pub fn gpu_cache_scales(&self) -> &[(CudaSlice<f32>, CudaSlice<f32>)] {
        &self.gpu_cache_scales
    }

    /// Quantize f16 key/value tensors and scatter into the FP8 paged cache.
    ///
    /// `key` and `value` are flat f16 host tensors `[num_tokens, num_heads * head_dim]`.
    /// `slot_mapping` maps each token to a flat slot in the cache.
    /// Stages the quantization on the host then uploads. A CUDA kernel version
    /// (using `quantize_paged_kv_kernel` from fp8_kv.cu) can replace this.
    pub fn reshape_and_cache_fp8(
        &mut self,
        key: &[f16],
        value: &[f16],
        layer: usize,
        slot_mapping: &[i32],
    ) -> Result<()> {
        let head_stride = self.num_heads * self.head_dim;
        let num_tokens = slot_mapping.len();

        if key.len() != num_tokens * head_stride {
            return Err(LLMError::GpuError(format!(
                "fp8 reshape_and_cache: key len {} != num_tokens({}) * head_stride({})",
                key.len(), num_tokens, head_stride
            )));
        }
        if value.len() != num_tokens * head_stride {
            return Err(LLMError::GpuError(format!(
                "fp8 reshape_and_cache: value len {} != num_tokens({}) * head_stride({})",
                value.len(), num_tokens, head_stride
            )));
        }

        let data_block_stride = self.block_size * head_stride;
        let scale_block_stride = self.block_size * self.num_heads;

        let (key_data_gpu, val_data_gpu) = &mut self.gpu_cache_data[layer];
        let (key_scale_gpu, val_scale_gpu) = &mut self.gpu_cache_scales[layer];

        let mut key_data = self.stream.clone_dtoh(key_data_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape dtoh key data: {e}")))?;
        let mut val_data = self.stream.clone_dtoh(val_data_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape dtoh val data: {e}")))?;
        let mut key_scales = self.stream.clone_dtoh(key_scale_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape dtoh key scales: {e}")))?;
        let mut val_scales = self.stream.clone_dtoh(val_scale_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape dtoh val scales: {e}")))?;

        for (token_idx, &slot) in slot_mapping.iter().enumerate() {
            if slot < 0 { continue; }
            let slot = slot as usize;
            let block_idx = slot / self.block_size;
            let block_offset = slot % self.block_size;

            let data_cache_off = block_idx * data_block_stride + block_offset * head_stride;
            let scale_cache_off = block_idx * scale_block_stride + block_offset * self.num_heads;
            let src_off = token_idx * head_stride;

            if data_cache_off + head_stride > key_data.len() {
                return Err(LLMError::GpuError(format!(
                    "fp8 reshape: cache offset {} exceeds buffer len {}",
                    data_cache_off + head_stride, key_data.len()
                )));
            }

            // f16 -> f32 -> FP8 quantize with per-head scaling
            let key_f32: Vec<f32> = key[src_off..src_off + head_stride]
                .iter().map(|v| v.to_f32()).collect();
            let val_f32: Vec<f32> = value[src_off..src_off + head_stride]
                .iter().map(|v| v.to_f32()).collect();

            let (key_quant, key_sc) = crate::fp8_cache::quantize_heads(&key_f32, self.num_heads, self.head_dim);
            let (val_quant, val_sc) = crate::fp8_cache::quantize_heads(&val_f32, self.num_heads, self.head_dim);

            key_data[data_cache_off..data_cache_off + head_stride].copy_from_slice(&key_quant);
            val_data[data_cache_off..data_cache_off + head_stride].copy_from_slice(&val_quant);
            key_scales[scale_cache_off..scale_cache_off + self.num_heads].copy_from_slice(&key_sc);
            val_scales[scale_cache_off..scale_cache_off + self.num_heads].copy_from_slice(&val_sc);
        }

        self.stream.memcpy_htod(&key_data, key_data_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape htod key data: {e}")))?;
        self.stream.memcpy_htod(&val_data, val_data_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape htod val data: {e}")))?;
        self.stream.memcpy_htod(&key_scales, key_scale_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape htod key scales: {e}")))?;
        self.stream.memcpy_htod(&val_scales, val_scale_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 reshape htod val scales: {e}")))?;

        debug!(num_tokens, layer, "CudaFP8CacheEngine reshape_and_cache_fp8 complete");
        Ok(())
    }

    /// Dequantize an entire block from FP8 back to f16 for the attention kernel.
    ///
    /// Returns `(key_f16, value_f16)` each of length `block_size * num_heads * head_dim`.
    pub fn dequantize_block(
        &self,
        layer: usize,
        block_idx: usize,
    ) -> Result<(Vec<f16>, Vec<f16>)> {
        if block_idx >= self.num_gpu_blocks {
            return Err(LLMError::GpuError(format!(
                "fp8 dequantize_block: block {block_idx} out of range (max={})",
                self.num_gpu_blocks
            )));
        }

        let head_stride = self.num_heads * self.head_dim;
        let depb = self.data_elements_per_block();
        let sepb = self.scale_elements_per_block();
        let data_off = block_idx * depb;
        let scale_off = block_idx * sepb;

        let (key_data_gpu, val_data_gpu) = &self.gpu_cache_data[layer];
        let (key_scale_gpu, val_scale_gpu) = &self.gpu_cache_scales[layer];

        let key_data = self.stream.clone_dtoh(key_data_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 deq dtoh key data: {e}")))?;
        let val_data = self.stream.clone_dtoh(val_data_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 deq dtoh val data: {e}")))?;
        let key_scales = self.stream.clone_dtoh(key_scale_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 deq dtoh key scales: {e}")))?;
        let val_scales = self.stream.clone_dtoh(val_scale_gpu)
            .map_err(|e| LLMError::GpuError(format!("fp8 deq dtoh val scales: {e}")))?;

        let mut key_f16 = vec![f16::ZERO; depb];
        let mut val_f16 = vec![f16::ZERO; depb];

        for tok in 0..self.block_size {
            let tok_data_off = data_off + tok * head_stride;
            let tok_scale_off = scale_off + tok * self.num_heads;
            let out_off = tok * head_stride;

            let key_f32 = crate::fp8_cache::dequantize_heads(
                &key_data[tok_data_off..tok_data_off + head_stride],
                &key_scales[tok_scale_off..tok_scale_off + self.num_heads],
                self.num_heads,
                self.head_dim,
            );
            let val_f32 = crate::fp8_cache::dequantize_heads(
                &val_data[tok_data_off..tok_data_off + head_stride],
                &val_scales[tok_scale_off..tok_scale_off + self.num_heads],
                self.num_heads,
                self.head_dim,
            );

            for i in 0..head_stride {
                key_f16[out_off + i] = f16::from_f32(key_f32[i]);
                val_f16[out_off + i] = f16::from_f32(val_f32[i]);
            }
        }

        Ok((key_f16, val_f16))
    }

    /// Dequantize multiple blocks for attention. Returns per-block f16 vectors.
    pub fn dequantize_blocks(
        &self,
        layer: usize,
        block_indices: &[usize],
    ) -> Result<(Vec<Vec<f16>>, Vec<Vec<f16>>)> {
        let mut keys = Vec::with_capacity(block_indices.len());
        let mut vals = Vec::with_capacity(block_indices.len());
        for &bidx in block_indices {
            let (k, v) = self.dequantize_block(layer, bidx)?;
            keys.push(k);
            vals.push(v);
        }
        Ok((keys, vals))
    }

    /// Copy blocks within GPU FP8 cache (data + scales).
    pub fn copy_blocks(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let depb = self.data_elements_per_block();
        let sepb = self.scale_elements_per_block();

        for &(src_id, dst_id) in mapping {
            let src = src_id.0 as usize;
            let dst = dst_id.0 as usize;

            if src >= self.num_gpu_blocks || dst >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "fp8 copy_blocks: block out of range (src={src}, dst={dst}, max={})",
                    self.num_gpu_blocks
                )));
            }

            let data_src = src * depb;
            let data_dst = dst * depb;
            let scale_src = src * sepb;
            let scale_dst = dst * sepb;

            for ((kd, vd), (ks, vs)) in self.gpu_cache_data.iter_mut()
                .zip(self.gpu_cache_scales.iter_mut())
            {
                let mut kd_host = self.stream.clone_dtoh(kd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy dtoh: {e}")))?;
                let slice: Vec<u8> = kd_host[data_src..data_src + depb].to_vec();
                kd_host[data_dst..data_dst + depb].copy_from_slice(&slice);
                self.stream.memcpy_htod(&kd_host, kd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy htod: {e}")))?;

                let mut vd_host = self.stream.clone_dtoh(vd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy dtoh: {e}")))?;
                let slice: Vec<u8> = vd_host[data_src..data_src + depb].to_vec();
                vd_host[data_dst..data_dst + depb].copy_from_slice(&slice);
                self.stream.memcpy_htod(&vd_host, vd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy htod: {e}")))?;

                let mut ks_host = self.stream.clone_dtoh(ks)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy dtoh: {e}")))?;
                let slice: Vec<f32> = ks_host[scale_src..scale_src + sepb].to_vec();
                ks_host[scale_dst..scale_dst + sepb].copy_from_slice(&slice);
                self.stream.memcpy_htod(&ks_host, ks)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy htod: {e}")))?;

                let mut vs_host = self.stream.clone_dtoh(vs)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy dtoh: {e}")))?;
                let slice: Vec<f32> = vs_host[scale_src..scale_src + sepb].to_vec();
                vs_host[scale_dst..scale_dst + sepb].copy_from_slice(&slice);
                self.stream.memcpy_htod(&vs_host, vs)
                    .map_err(|e| LLMError::GpuError(format!("fp8 copy htod: {e}")))?;
            }
        }

        debug!(pairs = mapping.len(), "CudaFP8CacheEngine copy_blocks complete");
        Ok(())
    }

    /// Swap blocks from CPU to GPU (data + scales).
    pub fn swap_in(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let depb = self.data_elements_per_block();
        let sepb = self.scale_elements_per_block();

        for &(cpu_id, gpu_id) in mapping {
            let cpu_idx = cpu_id.0 as usize;
            let gpu_idx = gpu_id.0 as usize;

            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "fp8 swap_in: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }
            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "fp8 swap_in: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }

            let cpu_data_off = cpu_idx * depb;
            let gpu_data_off = gpu_idx * depb;
            let cpu_scale_off = cpu_idx * sepb;
            let gpu_scale_off = gpu_idx * sepb;

            for (((kd, vd), (ks, vs)), ((ckd, cvd), (cks, cvs))) in self.gpu_cache_data.iter_mut()
                .zip(self.gpu_cache_scales.iter_mut())
                .zip(self.cpu_cache_data.iter().zip(self.cpu_cache_scales.iter()))
            {
                let mut kd_host = self.stream.clone_dtoh(kd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in dtoh: {e}")))?;
                kd_host[gpu_data_off..gpu_data_off + depb]
                    .copy_from_slice(&ckd[cpu_data_off..cpu_data_off + depb]);
                self.stream.memcpy_htod(&kd_host, kd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in htod: {e}")))?;

                let mut vd_host = self.stream.clone_dtoh(vd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in dtoh: {e}")))?;
                vd_host[gpu_data_off..gpu_data_off + depb]
                    .copy_from_slice(&cvd[cpu_data_off..cpu_data_off + depb]);
                self.stream.memcpy_htod(&vd_host, vd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in htod: {e}")))?;

                let mut ks_host = self.stream.clone_dtoh(ks)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in dtoh: {e}")))?;
                ks_host[gpu_scale_off..gpu_scale_off + sepb]
                    .copy_from_slice(&cks[cpu_scale_off..cpu_scale_off + sepb]);
                self.stream.memcpy_htod(&ks_host, ks)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in htod: {e}")))?;

                let mut vs_host = self.stream.clone_dtoh(vs)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in dtoh: {e}")))?;
                vs_host[gpu_scale_off..gpu_scale_off + sepb]
                    .copy_from_slice(&cvs[cpu_scale_off..cpu_scale_off + sepb]);
                self.stream.memcpy_htod(&vs_host, vs)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_in htod: {e}")))?;
            }
        }

        debug!(pairs = mapping.len(), "CudaFP8CacheEngine swap_in complete");
        Ok(())
    }

    /// Swap blocks from GPU to CPU (data + scales).
    pub fn swap_out(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let depb = self.data_elements_per_block();
        let sepb = self.scale_elements_per_block();

        for &(gpu_id, cpu_id) in mapping {
            let gpu_idx = gpu_id.0 as usize;
            let cpu_idx = cpu_id.0 as usize;

            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "fp8 swap_out: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }
            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "fp8 swap_out: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }

            let gpu_data_off = gpu_idx * depb;
            let cpu_data_off = cpu_idx * depb;
            let gpu_scale_off = gpu_idx * sepb;
            let cpu_scale_off = cpu_idx * sepb;

            for (((kd, vd), (ks, vs)), ((ckd, cvd), (cks, cvs))) in self.gpu_cache_data.iter()
                .zip(self.gpu_cache_scales.iter())
                .zip(self.cpu_cache_data.iter_mut().zip(self.cpu_cache_scales.iter_mut()))
            {
                let kd_host = self.stream.clone_dtoh(kd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_out dtoh: {e}")))?;
                ckd[cpu_data_off..cpu_data_off + depb]
                    .copy_from_slice(&kd_host[gpu_data_off..gpu_data_off + depb]);

                let vd_host = self.stream.clone_dtoh(vd)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_out dtoh: {e}")))?;
                cvd[cpu_data_off..cpu_data_off + depb]
                    .copy_from_slice(&vd_host[gpu_data_off..gpu_data_off + depb]);

                let ks_host = self.stream.clone_dtoh(ks)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_out dtoh: {e}")))?;
                cks[cpu_scale_off..cpu_scale_off + sepb]
                    .copy_from_slice(&ks_host[gpu_scale_off..gpu_scale_off + sepb]);

                let vs_host = self.stream.clone_dtoh(vs)
                    .map_err(|e| LLMError::GpuError(format!("fp8 swap_out dtoh: {e}")))?;
                cvs[cpu_scale_off..cpu_scale_off + sepb]
                    .copy_from_slice(&vs_host[gpu_scale_off..gpu_scale_off + sepb]);
            }
        }

        debug!(pairs = mapping.len(), "CudaFP8CacheEngine swap_out complete");
        Ok(())
    }

    /// Maximum FP8 blocks that fit in `available_bytes`.
    pub fn max_blocks_for_memory(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        available_bytes: usize,
    ) -> usize {
        let data_per_block = block_size * num_heads * head_dim; // u8
        let scale_per_block = block_size * num_heads; // f32
        // key + value data (u8) + key + value scales (f32)
        let bytes_per_block = num_layers * (2 * data_per_block + 2 * scale_per_block * std::mem::size_of::<f32>());
        if bytes_per_block == 0 { return 0; }
        available_bytes / bytes_per_block
    }
}
