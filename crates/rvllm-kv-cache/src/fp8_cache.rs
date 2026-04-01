//! FP8 (E4M3) quantized KV cache engine.
//!
//! Stores key and value tensors as u8 (FP8 E4M3 format) with per-head f32
//! scale factors. This halves the VRAM footprint of the KV cache compared to
//! FP16 storage -- often the single largest memory consumer during inference.
//!
//! FP8 E4M3: 1 sign, 4 exponent, 3 mantissa bits. Range [-448, 448].
//! Dynamic per-head scaling preserves accuracy across varying activation
//! magnitudes.
//!
//! ## Usage
//!
//! Enable with `--kv-cache-dtype fp8` at the engine level. The
//! [`FP8CacheEngine`] drop-in replaces [`CacheEngine`](super::CacheEngine)
//! when configured.

use half::f16;
use rvllm_core::prelude::{BlockId, LLMError, Result};
use rvllm_gpu::prelude::{CpuBuffer, GpuAllocator, GpuBuffer, GpuStream};
use std::mem;
use tracing::{debug, info};

/// Maximum representable magnitude in FP8 E4M3 format.
const FP8_E4M3_MAX: f32 = 448.0;

/// KV cache dtype selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheDtype {
    /// Standard FP16 storage (2 bytes per element).
    FP16,
    /// FP8 E4M3 storage (1 byte per element + per-head scale).
    FP8,
}

impl KVCacheDtype {
    /// Parse from CLI string (e.g., `--kv-cache-dtype fp8`).
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fp8" | "fp8_e4m3" => Some(Self::FP8),
            "fp16" | "half" | "auto" => Some(Self::FP16),
            _ => None,
        }
    }

    /// Bytes per cache element (excluding scale overhead).
    pub fn element_bytes(&self) -> usize {
        match self {
            Self::FP16 => mem::size_of::<f16>(),
            Self::FP8 => 1, // u8
        }
    }
}

/// FP8 cache configuration, extends [`super::CacheConfig`] with dtype awareness.
pub struct FP8CacheConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub dtype: KVCacheDtype,
}

impl FP8CacheConfig {
    /// Create a new FP8-aware cache config.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: KVCacheDtype,
    ) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            block_size,
            dtype,
        }
    }

    /// Bytes consumed by a single KV block (key + value) for one layer,
    /// accounting for FP8 quantization savings and scale factor overhead.
    pub fn block_bytes(&self) -> usize {
        let elements = self.block_size * self.num_heads * self.head_dim;
        let data_bytes = 2 * elements * self.dtype.element_bytes();
        let scale_bytes = match self.dtype {
            KVCacheDtype::FP8 => {
                // Per-head scale for each token slot in the block, for both K and V.
                2 * self.block_size * self.num_heads * mem::size_of::<f32>()
            }
            KVCacheDtype::FP16 => 0,
        };
        data_bytes + scale_bytes
    }

    /// Total bytes consumed by a single block across all layers.
    pub fn total_block_bytes(&self) -> usize {
        self.num_layers * self.block_bytes()
    }

    /// Maximum number of blocks that fit in `available_bytes`.
    pub fn num_blocks_from_memory(&self, available_bytes: usize) -> usize {
        let per_block = self.total_block_bytes();
        if per_block == 0 {
            return 0;
        }
        available_bytes / per_block
    }
}

/// Convert a single f32 value to FP8 E4M3, stored as u8.
///
/// The input should be pre-scaled (divided by the per-head scale factor)
/// so that it falls within [-448, 448].
fn float_to_fp8_e4m3(val: f32) -> u8 {
    let val = val.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
    let sign: u8 = if val < 0.0 { 0x80 } else { 0 };
    let abs_val = val.abs();

    // Subnormal threshold for E4M3: 2^-9 ~ 1.953125e-3
    const SUBNORMAL_UNIT: f32 = 1.953_125e-3;

    if abs_val < SUBNORMAL_UNIT {
        // Zero or rounds to zero
        return sign; // mantissa = 0
    }

    if abs_val < 2.0 * SUBNORMAL_UNIT {
        // Subnormal range: mantissa in [1..7], value = mantissa * 2^-9
        let mantissa = (abs_val / SUBNORMAL_UNIT + 0.5) as u8;
        let mantissa = mantissa.min(7);
        return sign | mantissa;
    }

    // Normal: 1.mmm * 2^(exp-7), exp in [1..15]
    // Find the exponent
    let bits = val.abs().to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp8_exp = fp32_exp + 7;

    if fp8_exp <= 0 {
        // Subnormal in E4M3
        let mantissa = (abs_val / SUBNORMAL_UNIT + 0.5) as u8;
        return sign | mantissa.min(7);
    }

    if fp8_exp > 15 {
        // Overflow: max finite E4M3 = sign | 0x7E (exp=15, mantissa=6)
        return sign | 0x7E;
    }

    // 3-bit mantissa from fp32's 23-bit mantissa, with rounding
    let fp32_mantissa = bits & 0x7F_FFFF;
    let mut mantissa = ((fp32_mantissa + (1 << 19)) >> 20) as u8;
    let mut exp = fp8_exp as u8;

    if mantissa >= 8 {
        mantissa = 0;
        exp += 1;
        if exp > 15 {
            return sign | 0x7E;
        }
    }

    // Clamp to avoid NaN (exp=15, mantissa=7)
    if exp == 15 && mantissa > 6 {
        mantissa = 6;
    }

    sign | ((exp & 0xF) << 3) | (mantissa & 0x7)
}

/// Convert FP8 E4M3 (u8) back to f32.
fn fp8_e4m3_to_float(fp8: u8) -> f32 {
    let sign: f32 = if fp8 & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (fp8 >> 3) & 0xF;
    let mantissa = fp8 & 0x7;

    if exp == 0 {
        if mantissa == 0 {
            return 0.0;
        }
        // Subnormal: mantissa * 2^-9
        return sign * (mantissa as f32) * 1.953_125e-3;
    }

    if exp == 15 && mantissa == 7 {
        // NaN sentinel -- return 0 for safety
        return 0.0;
    }

    // Normal: (1 + mantissa/8) * 2^(exp-7)
    let fmantissa = 1.0 + (mantissa as f32) / 8.0;
    sign * fmantissa * (2.0f32).powi(exp as i32 - 7)
}

/// Quantize a slice of f32 values into FP8 E4M3 with per-head dynamic scaling.
///
/// `input` is shaped `[num_heads, head_dim]` (one token's worth).
/// Returns `(quantized_data, scales)` where `scales` has `num_heads` entries.
pub fn quantize_heads(input: &[f32], num_heads: usize, head_dim: usize) -> (Vec<u8>, Vec<f32>) {
    let mut output = vec![0u8; num_heads * head_dim];
    let mut scales = vec![0.0f32; num_heads];

    for h in 0..num_heads {
        let base = h * head_dim;
        let head_slice = &input[base..base + head_dim];

        // Dynamic per-head scaling: absmax / FP8_MAX
        let absmax = head_slice.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = (absmax / FP8_E4M3_MAX).max(1e-12);
        scales[h] = scale;

        let inv_scale = 1.0 / scale;
        for d in 0..head_dim {
            output[base + d] = float_to_fp8_e4m3(head_slice[d] * inv_scale);
        }
    }

    (output, scales)
}

/// Dequantize a slice of FP8 E4M3 values back to f32.
///
/// `input` is `[num_heads, head_dim]` as u8, `scales` is `[num_heads]`.
pub fn dequantize_heads(
    input: &[u8],
    scales: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * head_dim];

    for h in 0..num_heads {
        let base = h * head_dim;
        let scale = scales[h];
        for d in 0..head_dim {
            output[base + d] = fp8_e4m3_to_float(input[base + d]) * scale;
        }
    }

    output
}

/// Per-layer FP8 paged KV cache engine.
///
/// Stores key/value data as u8 (FP8 E4M3) on the GPU with per-head f32
/// scale factors, achieving ~50% VRAM savings over FP16 caches.
///
/// The cache layout mirrors [`CacheEngine`](super::CacheEngine):
///   data:   `[num_blocks, block_size, num_heads, head_dim]` as u8
///   scales: `[num_blocks, block_size, num_heads]` as f32
pub struct FP8CacheEngine {
    /// Per-layer (key_data, value_data) as u8 on GPU.
    pub gpu_cache_data: Vec<(GpuBuffer<u8>, GpuBuffer<u8>)>,
    /// Per-layer (key_scales, value_scales) as f32 on GPU.
    pub gpu_cache_scales: Vec<(GpuBuffer<f32>, GpuBuffer<f32>)>,
    /// Per-layer CPU staging buffers for data (u8).
    pub cpu_cache_data: Vec<(CpuBuffer<u8>, CpuBuffer<u8>)>,
    /// Per-layer CPU staging buffers for scales (f32).
    pub cpu_cache_scales: Vec<(Vec<f32>, Vec<f32>)>,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
}

impl FP8CacheEngine {
    /// Allocate FP8 KV cache for all layers.
    ///
    /// # Errors
    /// Returns `LLMError::MemoryError` if any allocation fails.
    pub fn new<A: GpuAllocator>(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        allocator: &A,
    ) -> Result<Self> {
        let data_elements = block_size * num_heads * head_dim;
        let scale_elements = block_size * num_heads;
        let gpu_data_total = num_gpu_blocks * data_elements;
        let gpu_scale_total = num_gpu_blocks * scale_elements;
        let cpu_data_total = num_cpu_blocks * data_elements;
        let cpu_scale_total = num_cpu_blocks * scale_elements;

        let fp16_equiv_bytes = num_gpu_blocks * data_elements * 2 * mem::size_of::<f16>();
        let fp8_bytes =
            num_gpu_blocks * data_elements * 2 + num_gpu_blocks * scale_elements * 2 * 4;

        info!(
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            fp16_equiv_bytes,
            fp8_bytes,
            savings_pct = ((fp16_equiv_bytes as f64 - fp8_bytes as f64) / fp16_equiv_bytes as f64
                * 100.0) as u32,
            "FP8CacheEngine: allocating quantized KV cache"
        );

        let mut gpu_cache_data = Vec::with_capacity(num_layers);
        let mut gpu_cache_scales = Vec::with_capacity(num_layers);
        let mut cpu_cache_data = Vec::with_capacity(num_layers);
        let mut cpu_cache_scales = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            debug!(layer, gpu_data_total, "allocating FP8 GPU KV cache");

            let key_data = allocator.alloc::<u8>(gpu_data_total).map_err(|e| {
                LLMError::MemoryError(format!("FP8 GPU key data alloc layer {layer}: {e}"))
            })?;
            let val_data = allocator.alloc::<u8>(gpu_data_total).map_err(|e| {
                LLMError::MemoryError(format!("FP8 GPU value data alloc layer {layer}: {e}"))
            })?;
            gpu_cache_data.push((key_data, val_data));

            let key_scales = allocator.alloc::<f32>(gpu_scale_total).map_err(|e| {
                LLMError::MemoryError(format!("FP8 GPU key scales alloc layer {layer}: {e}"))
            })?;
            let val_scales = allocator.alloc::<f32>(gpu_scale_total).map_err(|e| {
                LLMError::MemoryError(format!("FP8 GPU value scales alloc layer {layer}: {e}"))
            })?;
            gpu_cache_scales.push((key_scales, val_scales));

            let key_cpu_data = CpuBuffer::<u8>::new(cpu_data_total);
            let val_cpu_data = CpuBuffer::<u8>::new(cpu_data_total);
            cpu_cache_data.push((key_cpu_data, val_cpu_data));

            let key_cpu_scales = vec![0.0f32; cpu_scale_total];
            let val_cpu_scales = vec![0.0f32; cpu_scale_total];
            cpu_cache_scales.push((key_cpu_scales, val_cpu_scales));
        }

        Ok(Self {
            gpu_cache_data,
            gpu_cache_scales,
            cpu_cache_data,
            cpu_cache_scales,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
        })
    }

    /// Number of u8 data elements per cache block.
    pub fn data_elements_per_block(&self) -> usize {
        self.block_size * self.num_heads * self.head_dim
    }

    /// Number of f32 scale elements per cache block.
    pub fn scale_elements_per_block(&self) -> usize {
        self.block_size * self.num_heads
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

    /// Number of GPU blocks.
    pub fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    /// Quantize and write f16 key/value tensors into the FP8 paged cache.
    ///
    /// `key` and `value` are flat f16 tensors of shape `[num_tokens, num_heads * head_dim]`.
    /// `slot_mapping` maps each token to a flat slot index in the cache.
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
            return Err(LLMError::MemoryError(format!(
                "fp8 reshape_and_cache: key length {} != num_tokens({}) * head_stride({})",
                key.len(),
                num_tokens,
                head_stride
            )));
        }
        if value.len() != num_tokens * head_stride {
            return Err(LLMError::MemoryError(format!(
                "fp8 reshape_and_cache: value length {} != num_tokens({}) * head_stride({})",
                value.len(),
                num_tokens,
                head_stride
            )));
        }

        let data_block_stride = self.block_size * head_stride;
        let scale_block_stride = self.block_size * self.num_heads;

        let (key_data_buf, val_data_buf) = &mut self.gpu_cache_data[layer];
        let (key_scale_buf, val_scale_buf) = &mut self.gpu_cache_scales[layer];

        let mut key_data = key_data_buf.copy_to_host()?;
        let mut val_data = val_data_buf.copy_to_host()?;
        let mut key_scales = key_scale_buf.copy_to_host()?;
        let mut val_scales = val_scale_buf.copy_to_host()?;

        for (token_idx, &slot) in slot_mapping.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let slot = slot as usize;
            let block_idx = slot / self.block_size;
            let block_offset = slot % self.block_size;

            let data_cache_offset = block_idx * data_block_stride + block_offset * head_stride;
            let scale_cache_offset = block_idx * scale_block_stride + block_offset * self.num_heads;
            let src_offset = token_idx * head_stride;

            if data_cache_offset + head_stride > key_data.len() {
                return Err(LLMError::MemoryError(format!(
                    "fp8 reshape_and_cache: cache offset {} exceeds buffer len {}",
                    data_cache_offset + head_stride,
                    key_data.len()
                )));
            }

            // Convert f16 -> f32, quantize to FP8 with per-head scales
            let key_f32: Vec<f32> = key[src_offset..src_offset + head_stride]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let val_f32: Vec<f32> = value[src_offset..src_offset + head_stride]
                .iter()
                .map(|v| v.to_f32())
                .collect();

            let (key_quant, key_sc) = quantize_heads(&key_f32, self.num_heads, self.head_dim);
            let (val_quant, val_sc) = quantize_heads(&val_f32, self.num_heads, self.head_dim);

            key_data[data_cache_offset..data_cache_offset + head_stride]
                .copy_from_slice(&key_quant);
            val_data[data_cache_offset..data_cache_offset + head_stride]
                .copy_from_slice(&val_quant);
            key_scales[scale_cache_offset..scale_cache_offset + self.num_heads]
                .copy_from_slice(&key_sc);
            val_scales[scale_cache_offset..scale_cache_offset + self.num_heads]
                .copy_from_slice(&val_sc);
        }

        key_data_buf.copy_from_host(&key_data)?;
        val_data_buf.copy_from_host(&val_data)?;
        key_scale_buf.copy_from_host(&key_scales)?;
        val_scale_buf.copy_from_host(&val_scales)?;

        debug!(num_tokens, layer, "fp8 reshape_and_cache complete");
        Ok(())
    }

    /// Dequantize and read back a single token's key/value from the FP8 cache
    /// as f32 vectors (each of length `num_heads * head_dim`).
    pub fn dequantize_token(&self, layer: usize, slot: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let head_stride = self.num_heads * self.head_dim;
        let data_block_stride = self.block_size * head_stride;
        let scale_block_stride = self.block_size * self.num_heads;

        let block_idx = slot / self.block_size;
        let block_offset = slot % self.block_size;

        let data_offset = block_idx * data_block_stride + block_offset * head_stride;
        let scale_offset = block_idx * scale_block_stride + block_offset * self.num_heads;

        let (key_data_buf, val_data_buf) = &self.gpu_cache_data[layer];
        let (key_scale_buf, val_scale_buf) = &self.gpu_cache_scales[layer];

        let key_data = key_data_buf.copy_to_host()?;
        let val_data = val_data_buf.copy_to_host()?;
        let key_scales = key_scale_buf.copy_to_host()?;
        let val_scales = val_scale_buf.copy_to_host()?;

        if data_offset + head_stride > key_data.len() {
            return Err(LLMError::MemoryError(format!(
                "fp8 dequantize_token: slot {} exceeds cache capacity",
                slot
            )));
        }

        let key_f32 = dequantize_heads(
            &key_data[data_offset..data_offset + head_stride],
            &key_scales[scale_offset..scale_offset + self.num_heads],
            self.num_heads,
            self.head_dim,
        );
        let val_f32 = dequantize_heads(
            &val_data[data_offset..data_offset + head_stride],
            &val_scales[scale_offset..scale_offset + self.num_heads],
            self.num_heads,
            self.head_dim,
        );

        Ok((key_f32, val_f32))
    }

    /// Copy blocks within GPU FP8 cache. Each `(src, dst)` pair copies a full
    /// block (data + scales) across all layers.
    pub fn copy_blocks(
        &mut self,
        mapping: &[(BlockId, BlockId)],
        _stream: &GpuStream,
    ) -> Result<()> {
        let depb = self.data_elements_per_block();
        let sepb = self.scale_elements_per_block();

        for &(src_id, dst_id) in mapping {
            let src = src_id.0 as usize;
            let dst = dst_id.0 as usize;

            if src >= self.num_gpu_blocks || dst >= self.num_gpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "fp8 copy_blocks: block index out of range (src={src}, dst={dst}, max={})",
                    self.num_gpu_blocks
                )));
            }

            let data_src_off = src * depb;
            let data_dst_off = dst * depb;
            let scale_src_off = src * sepb;
            let scale_dst_off = dst * sepb;

            for ((key_d, val_d), (key_s, val_s)) in self
                .gpu_cache_data
                .iter_mut()
                .zip(self.gpu_cache_scales.iter_mut())
            {
                // Data copy
                let mut kd = key_d.copy_to_host()?;
                let src_slice: Vec<u8> = kd[data_src_off..data_src_off + depb].to_vec();
                kd[data_dst_off..data_dst_off + depb].copy_from_slice(&src_slice);
                key_d.copy_from_host(&kd)?;

                let mut vd = val_d.copy_to_host()?;
                let src_slice: Vec<u8> = vd[data_src_off..data_src_off + depb].to_vec();
                vd[data_dst_off..data_dst_off + depb].copy_from_slice(&src_slice);
                val_d.copy_from_host(&vd)?;

                // Scale copy
                let mut ks = key_s.copy_to_host()?;
                let src_slice: Vec<f32> = ks[scale_src_off..scale_src_off + sepb].to_vec();
                ks[scale_dst_off..scale_dst_off + sepb].copy_from_slice(&src_slice);
                key_s.copy_from_host(&ks)?;

                let mut vs = val_s.copy_to_host()?;
                let src_slice: Vec<f32> = vs[scale_src_off..scale_src_off + sepb].to_vec();
                vs[scale_dst_off..scale_dst_off + sepb].copy_from_slice(&src_slice);
                val_s.copy_from_host(&vs)?;
            }
        }

        debug!(pairs = mapping.len(), "fp8 copy_blocks complete");
        Ok(())
    }

    /// Swap blocks from CPU FP8 cache into GPU FP8 cache.
    pub fn swap_in(&mut self, mapping: &[(BlockId, BlockId)], _stream: &GpuStream) -> Result<()> {
        let depb = self.data_elements_per_block();
        let sepb = self.scale_elements_per_block();

        for &(cpu_id, gpu_id) in mapping {
            let cpu_idx = cpu_id.0 as usize;
            let gpu_idx = gpu_id.0 as usize;

            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "fp8 swap_in: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }
            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "fp8 swap_in: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }

            let cpu_data_off = cpu_idx * depb;
            let gpu_data_off = gpu_idx * depb;
            let cpu_scale_off = cpu_idx * sepb;
            let gpu_scale_off = gpu_idx * sepb;

            for (((key_d, val_d), (key_s, val_s)), ((cpu_kd, cpu_vd), (cpu_ks, cpu_vs))) in self
                .gpu_cache_data
                .iter_mut()
                .zip(self.gpu_cache_scales.iter_mut())
                .zip(self.cpu_cache_data.iter().zip(self.cpu_cache_scales.iter()))
            {
                let mut kd = key_d.copy_to_host()?;
                kd[gpu_data_off..gpu_data_off + depb]
                    .copy_from_slice(&cpu_kd.as_slice()[cpu_data_off..cpu_data_off + depb]);
                key_d.copy_from_host(&kd)?;

                let mut vd = val_d.copy_to_host()?;
                vd[gpu_data_off..gpu_data_off + depb]
                    .copy_from_slice(&cpu_vd.as_slice()[cpu_data_off..cpu_data_off + depb]);
                val_d.copy_from_host(&vd)?;

                let mut ks = key_s.copy_to_host()?;
                ks[gpu_scale_off..gpu_scale_off + sepb]
                    .copy_from_slice(&cpu_ks[cpu_scale_off..cpu_scale_off + sepb]);
                key_s.copy_from_host(&ks)?;

                let mut vs = val_s.copy_to_host()?;
                vs[gpu_scale_off..gpu_scale_off + sepb]
                    .copy_from_slice(&cpu_vs[cpu_scale_off..cpu_scale_off + sepb]);
                val_s.copy_from_host(&vs)?;
            }
        }

        debug!(pairs = mapping.len(), "fp8 swap_in complete");
        Ok(())
    }

    /// Swap blocks from GPU FP8 cache out to CPU FP8 cache.
    pub fn swap_out(&mut self, mapping: &[(BlockId, BlockId)], _stream: &GpuStream) -> Result<()> {
        let depb = self.data_elements_per_block();
        let sepb = self.scale_elements_per_block();

        for &(gpu_id, cpu_id) in mapping {
            let gpu_idx = gpu_id.0 as usize;
            let cpu_idx = cpu_id.0 as usize;

            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "fp8 swap_out: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }
            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "fp8 swap_out: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }

            let gpu_data_off = gpu_idx * depb;
            let cpu_data_off = cpu_idx * depb;
            let gpu_scale_off = gpu_idx * sepb;
            let cpu_scale_off = cpu_idx * sepb;

            for (((key_d, val_d), (key_s, val_s)), ((cpu_kd, cpu_vd), (cpu_ks, cpu_vs))) in self
                .gpu_cache_data
                .iter()
                .zip(self.gpu_cache_scales.iter())
                .zip(
                    self.cpu_cache_data
                        .iter_mut()
                        .zip(self.cpu_cache_scales.iter_mut()),
                )
            {
                let kd = key_d.copy_to_host()?;
                cpu_kd.as_mut_slice()[cpu_data_off..cpu_data_off + depb]
                    .copy_from_slice(&kd[gpu_data_off..gpu_data_off + depb]);

                let vd = val_d.copy_to_host()?;
                cpu_vd.as_mut_slice()[cpu_data_off..cpu_data_off + depb]
                    .copy_from_slice(&vd[gpu_data_off..gpu_data_off + depb]);

                let ks = key_s.copy_to_host()?;
                cpu_ks[cpu_scale_off..cpu_scale_off + sepb]
                    .copy_from_slice(&ks[gpu_scale_off..gpu_scale_off + sepb]);

                let vs = val_s.copy_to_host()?;
                cpu_vs[cpu_scale_off..cpu_scale_off + sepb]
                    .copy_from_slice(&vs[gpu_scale_off..gpu_scale_off + sepb]);
            }
        }

        debug!(pairs = mapping.len(), "fp8 swap_out complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_dtype_parsing() {
        assert_eq!(KVCacheDtype::from_str_opt("fp8"), Some(KVCacheDtype::FP8));
        assert_eq!(
            KVCacheDtype::from_str_opt("fp8_e4m3"),
            Some(KVCacheDtype::FP8)
        );
        assert_eq!(KVCacheDtype::from_str_opt("fp16"), Some(KVCacheDtype::FP16));
        assert_eq!(KVCacheDtype::from_str_opt("auto"), Some(KVCacheDtype::FP16));
        assert_eq!(KVCacheDtype::from_str_opt("nonsense"), None);
    }

    #[test]
    fn fp8_round_trip_zero() {
        let fp8 = float_to_fp8_e4m3(0.0);
        let back = fp8_e4m3_to_float(fp8);
        assert_eq!(back, 0.0);
    }

    #[test]
    fn fp8_round_trip_one() {
        let fp8 = float_to_fp8_e4m3(1.0);
        let back = fp8_e4m3_to_float(fp8);
        assert!((back - 1.0).abs() < 0.1, "got {back}");
    }

    #[test]
    fn fp8_round_trip_negative() {
        let fp8 = float_to_fp8_e4m3(-2.5);
        let back = fp8_e4m3_to_float(fp8);
        assert!((back - (-2.5)).abs() < 0.5, "got {back}");
    }

    #[test]
    fn fp8_clamps_overflow() {
        let fp8 = float_to_fp8_e4m3(1000.0);
        let back = fp8_e4m3_to_float(fp8);
        assert!(back <= FP8_E4M3_MAX, "got {back}");
        assert!(back > 400.0, "should be near max, got {back}");
    }

    #[test]
    fn fp8_small_values() {
        // Very small value should round-trip approximately
        let fp8 = float_to_fp8_e4m3(0.01);
        let back = fp8_e4m3_to_float(fp8);
        assert!((back - 0.01).abs() < 0.005, "got {back}");
    }

    #[test]
    fn quantize_dequantize_heads_accuracy() {
        let num_heads = 4;
        let head_dim = 8;
        let input: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| (i as f32 - 16.0) * 0.5)
            .collect();

        let (quantized, scales) = quantize_heads(&input, num_heads, head_dim);
        let dequantized = dequantize_heads(&quantized, &scales, num_heads, head_dim);

        // Check approximate round-trip accuracy.
        // FP8 E4M3 has 3 mantissa bits => worst-case quantization step is
        // absmax / 8 per head (since the significand has 8 steps for normals).
        for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            let err = (orig - deq).abs();
            let head_idx = i / head_dim;
            // Compute the absmax for this head from the original data
            let head_base = head_idx * head_dim;
            let head_absmax = input[head_base..head_base + head_dim]
                .iter()
                .fold(0.0f32, |acc, &v| acc.max(v.abs()));
            // Worst-case error: ~1 quantization step = absmax / 8
            let tol = head_absmax / 8.0 + 0.05;
            assert!(
                err <= tol,
                "element {i}: orig={orig}, deq={deq}, err={err}, tol={tol}"
            );
        }
    }

    #[test]
    fn fp8_cache_config_block_bytes() {
        let cfg = FP8CacheConfig::new(32, 32, 128, 16, KVCacheDtype::FP8);
        let fp16_cfg = FP8CacheConfig::new(32, 32, 128, 16, KVCacheDtype::FP16);

        // FP8 should use roughly half the bytes of FP16 (plus small scale overhead)
        assert!(cfg.block_bytes() < fp16_cfg.block_bytes());

        // FP16: 2 * 16 * 32 * 128 * 2 = 262144
        assert_eq!(fp16_cfg.block_bytes(), 2 * 16 * 32 * 128 * 2);

        // FP8 data: 2 * 16 * 32 * 128 * 1 = 131072
        // FP8 scales: 2 * 16 * 32 * 4 = 4096
        // Total: 135168
        assert_eq!(cfg.block_bytes(), 131072 + 4096);
    }

    #[test]
    fn fp8_cache_config_memory_savings() {
        let fp8 = FP8CacheConfig::new(32, 32, 128, 16, KVCacheDtype::FP8);
        let fp16 = FP8CacheConfig::new(32, 32, 128, 16, KVCacheDtype::FP16);

        let avail = 1 << 30; // 1 GB
        let fp8_blocks = fp8.num_blocks_from_memory(avail);
        let fp16_blocks = fp16.num_blocks_from_memory(avail);

        // FP8 should fit roughly 2x as many blocks
        assert!(
            fp8_blocks > fp16_blocks,
            "fp8={fp8_blocks} should > fp16={fp16_blocks}"
        );
        let ratio = fp8_blocks as f64 / fp16_blocks as f64;
        assert!(ratio > 1.8, "expected ~2x improvement, got {ratio:.2}x");
    }

    #[test]
    fn fp8_element_bytes() {
        assert_eq!(KVCacheDtype::FP8.element_bytes(), 1);
        assert_eq!(KVCacheDtype::FP16.element_bytes(), 2);
    }

    #[test]
    fn fp8_round_trip_various_magnitudes() {
        let test_values = [
            0.0, 1.0, -1.0, 0.5, -0.5, 10.0, -10.0, 100.0, -100.0, 448.0, -448.0,
        ];
        for &val in &test_values {
            let fp8 = float_to_fp8_e4m3(val);
            let back = fp8_e4m3_to_float(fp8);
            let err = (val - back).abs();
            let tol = val.abs() * 0.2 + 0.01; // ~12.5% relative + small absolute
            assert!(err <= tol, "value {val}: got {back}, err={err}, tol={tol}");
        }
    }
}
