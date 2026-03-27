//! FlashAttention-2 backend with paged KV cache support.
//!
//! Provides two paths:
//! - **CUDA path** (`cuda` feature): dispatches to `flash_attention_2_kernel` /
//!   `flash_attention_2_decode_kernel` via cudarc. Auto-selected for SM >= 8.0.
//! - **CPU reference path** (always available): tiled online-softmax attention
//!   that mirrors the CUDA algorithm for correctness testing on Mac / CI.
//!
//! The CPU path is the default when `cfg(not(feature = "cuda"))`.

use half::f16;
use tracing::debug;
#[cfg(feature = "cuda")]
use tracing::trace;
use rvllm_core::prelude::{LLMError, Result};

use crate::backend::AttentionBackend;
use crate::buffer::GpuBuffer;

/// KV tile width -- matches `FA2_BC` in `flash_attention.cu`.
const TILE_BC: usize = 64;

/// Supported head dimensions.
const SUPPORTED_HEAD_DIMS: &[usize] = &[64, 96, 128];

/// Configuration for the FlashAttention-2 backend.
#[derive(Debug, Clone)]
pub struct FlashAttention2Config {
    /// Whether to apply causal masking.
    pub causal: bool,
    /// Number of KV heads (for GQA; if equal to num_heads, standard MHA).
    pub num_kv_heads: Option<usize>,
}

impl Default for FlashAttention2Config {
    fn default() -> Self {
        Self {
            causal: true,
            num_kv_heads: None,
        }
    }
}

/// FlashAttention-2 backend with tiled SRAM computation and online softmax.
///
/// When the `cuda` feature is enabled and a CUDA device is available, this
/// dispatches to the GPU kernels. Otherwise it falls back to a CPU reference
/// implementation that uses the same tiled online-softmax algorithm.
pub struct FlashAttention2 {
    config: FlashAttention2Config,
    #[cfg(feature = "cuda")]
    cuda_state: Option<CudaFlashState>,
}

#[cfg(feature = "cuda")]
struct CudaFlashState {
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
    stream: cudarc::driver::CudaStream,
}

impl FlashAttention2 {
    /// Create a new FlashAttention-2 backend with default configuration.
    pub fn new() -> Self {
        debug!("initializing FlashAttention-2 backend (CPU reference)");
        Self {
            config: FlashAttention2Config::default(),
            #[cfg(feature = "cuda")]
            cuda_state: None,
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(config: FlashAttention2Config) -> Self {
        debug!(?config, "initializing FlashAttention-2 backend");
        Self {
            config,
            #[cfg(feature = "cuda")]
            cuda_state: None,
        }
    }

    /// Create with CUDA device for GPU execution.
    #[cfg(feature = "cuda")]
    pub fn with_cuda(
        device: std::sync::Arc<cudarc::driver::CudaDevice>,
        ptx_bytes: &[u8],
        config: FlashAttention2Config,
    ) -> Result<Self> {
        let ptx_str = std::str::from_utf8(ptx_bytes)
            .map_err(|e| LLMError::GpuError(format!("PTX not valid UTF-8: {e}")))?;

        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(ptx_str),
                "flash_attention",
                &[
                    "flash_attention_2_kernel",
                    "flash_attention_2_decode_kernel",
                ],
            )
            .map_err(|e| LLMError::GpuError(format!("failed to load flash_attention PTX: {e}")))?;

        let stream = device
            .fork_default_stream()
            .map_err(|e| LLMError::GpuError(format!("failed to create CUDA stream: {e}")))?;

        debug!("FlashAttention-2 CUDA backend initialized");
        Ok(Self {
            config,
            cuda_state: Some(CudaFlashState { device, stream }),
        })
    }

    /// Validate that head_dim is one of the supported sizes.
    fn validate_head_dim(head_dim: usize) -> Result<()> {
        if !SUPPORTED_HEAD_DIMS.contains(&head_dim) {
            return Err(LLMError::GpuError(format!(
                "FlashAttention-2 supports head_dim in {SUPPORTED_HEAD_DIMS:?}, got {head_dim}"
            )));
        }
        Ok(())
    }

    /// CPU reference implementation of FlashAttention-2 with paged KV cache.
    ///
    /// Uses the same tiled online-softmax algorithm as the CUDA kernel:
    /// - Stream K/V in tiles of size `TILE_BC`
    /// - Online softmax with running max/sum (Milakov & Gimelshein)
    /// - No full N x N attention matrix materialized
    fn forward_cpu(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        if query.shape.len() != 3 {
            return Err(LLMError::GpuError(format!(
                "query must be 3-D [num_tokens, num_heads, head_dim], got {} dims",
                query.shape.len()
            )));
        }
        if key_cache.shape.len() != 4 {
            return Err(LLMError::GpuError(format!(
                "key_cache must be 4-D [num_blocks, block_size, num_kv_heads, head_dim], got {} dims",
                key_cache.shape.len()
            )));
        }

        let num_tokens = query.shape[0];
        let num_heads = query.shape[1];
        let head_dim = query.shape[2];
        let block_size = key_cache.shape[1];
        let num_kv_heads = self.config.num_kv_heads.unwrap_or(num_heads);
        let num_seqs = context_lens.data.len();

        Self::validate_head_dim(head_dim)?;

        if num_seqs == 0 {
            return Ok(GpuBuffer {
                data: Vec::new(),
                shape: vec![0, num_heads, head_dim],
            });
        }

        let max_blocks_per_seq = block_tables.shape.get(1).copied().unwrap_or(0);
        let mut output = vec![f16::ZERO; num_tokens * num_heads * head_dim];

        let kv_head_ratio = if num_kv_heads < num_heads {
            num_heads / num_kv_heads
        } else {
            1
        };

        // Process each sequence
        let mut token_offset = 0usize;
        for seq_idx in 0..num_seqs {
            let ctx_len = (context_lens.data[seq_idx] as usize).min(max_context_len);
            let q_len = if seq_idx + 1 < num_seqs {
                1 // decode: 1 token per sequence
            } else {
                (num_tokens - token_offset).max(1)
            };

            let num_tiles = (ctx_len + TILE_BC - 1) / TILE_BC;

            for qi in 0..q_len {
                let q_pos = token_offset + qi;
                if q_pos >= num_tokens {
                    break;
                }

                for h in 0..num_heads {
                    let kv_h = if num_kv_heads == num_heads {
                        h
                    } else {
                        h / kv_head_ratio
                    };

                    // Load query vector (pre-scaled)
                    let q_base = (q_pos * num_heads + h) * head_dim;
                    let q_vec: Vec<f32> = (0..head_dim)
                        .map(|d| query.data[q_base + d].to_f32() * scale)
                        .collect();

                    // Online softmax state
                    let mut row_max = f32::NEG_INFINITY;
                    let mut row_sum = 0.0f32;
                    let mut acc = vec![0.0f32; head_dim];

                    // Tile over KV positions
                    for tile in 0..num_tiles {
                        let tile_start = tile * TILE_BC;
                        let tile_len = TILE_BC.min(ctx_len - tile_start);

                        // Compute Q * K^T for this tile
                        let mut scores = Vec::with_capacity(tile_len);
                        for t in 0..tile_len {
                            let kv_pos = tile_start + t;
                            let page_idx = kv_pos / block_size;
                            let page_off = kv_pos % block_size;
                            if page_idx >= max_blocks_per_seq {
                                scores.push(f32::NEG_INFINITY);
                                continue;
                            }
                            let phys_block =
                                block_tables.data[seq_idx * max_blocks_per_seq + page_idx]
                                    as usize;
                            let k_base = ((phys_block * block_size + page_off) * num_kv_heads
                                + kv_h)
                                * head_dim;

                            let dot: f32 = (0..head_dim)
                                .map(|d| q_vec[d] * key_cache.data[k_base + d].to_f32())
                                .sum();

                            // Causal mask: mask out if kv_pos > query's absolute position
                            if self.config.causal {
                                let q_abs_pos = ctx_len - q_len + qi;
                                if kv_pos > q_abs_pos {
                                    scores.push(f32::NEG_INFINITY);
                                    continue;
                                }
                            }

                            scores.push(dot);
                        }

                        // Online softmax: update running max
                        let tile_max = scores
                            .iter()
                            .cloned()
                            .fold(f32::NEG_INFINITY, f32::max);

                        let new_max = row_max.max(tile_max);
                        if new_max > row_max && row_max > f32::NEG_INFINITY {
                            let correction = (row_max - new_max).exp();
                            for a in acc.iter_mut() {
                                *a *= correction;
                            }
                            row_sum *= correction;
                        }
                        row_max = new_max;

                        // Exponentiate scores
                        let exp_scores: Vec<f32> = scores
                            .iter()
                            .map(|&s| {
                                if s > f32::NEG_INFINITY + 1.0 {
                                    (s - row_max).exp()
                                } else {
                                    0.0
                                }
                            })
                            .collect();

                        let tile_sum: f32 = exp_scores.iter().sum();
                        row_sum += tile_sum;

                        // Accumulate P * V
                        for (t, &w) in exp_scores.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let kv_pos = tile_start + t;
                            let page_idx = kv_pos / block_size;
                            let page_off = kv_pos % block_size;
                            if page_idx >= max_blocks_per_seq {
                                continue;
                            }
                            let phys_block =
                                block_tables.data[seq_idx * max_blocks_per_seq + page_idx]
                                    as usize;
                            let v_base = ((phys_block * block_size + page_off) * num_kv_heads
                                + kv_h)
                                * head_dim;

                            for d in 0..head_dim {
                                acc[d] += w * value_cache.data[v_base + d].to_f32();
                            }
                        }
                    }

                    // Normalize and write output
                    let inv_sum = if row_sum > 0.0 { 1.0 / row_sum } else { 0.0 };
                    let o_base = (q_pos * num_heads + h) * head_dim;
                    for d in 0..head_dim {
                        output[o_base + d] = f16::from_f32(acc[d] * inv_sum);
                    }
                }
            }
            token_offset += q_len;
        }

        Ok(GpuBuffer {
            data: output,
            shape: vec![num_tokens, num_heads, head_dim],
        })
    }

    /// CUDA dispatch for FlashAttention-2.
    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

        let state = self
            .cuda_state
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("CUDA state not initialized".into()))?;

        if query.shape.len() != 3 {
            return Err(LLMError::GpuError(format!(
                "query must be 3-D, got {} dims",
                query.shape.len()
            )));
        }
        if key_cache.shape.len() != 4 {
            return Err(LLMError::GpuError(format!(
                "key_cache must be 4-D, got {} dims",
                key_cache.shape.len()
            )));
        }

        let num_tokens = query.shape[0];
        let num_heads = query.shape[1];
        let head_dim = query.shape[2];
        let block_size = key_cache.shape[1];
        let num_kv_heads = self.config.num_kv_heads.unwrap_or(num_heads);
        let num_seqs = context_lens.data.len();
        let max_blocks_per_seq = block_tables.shape.get(1).copied().unwrap_or(0);

        Self::validate_head_dim(head_dim)?;

        if num_seqs == 0 {
            return Ok(GpuBuffer {
                data: Vec::new(),
                shape: vec![0, num_heads, head_dim],
            });
        }

        trace!(
            num_seqs,
            num_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_context_len,
            "FlashAttention2::forward_cuda"
        );

        let dev = &state.device;

        // Upload buffers
        let q_f32: Vec<f32> = query.data.iter().map(|v| v.to_f32()).collect();
        let k_f32: Vec<f32> = key_cache.data.iter().map(|v| v.to_f32()).collect();
        let v_f32: Vec<f32> = value_cache.data.iter().map(|v| v.to_f32()).collect();

        let d_query = dev
            .htod_sync_copy(&q_f32)
            .map_err(|e| LLMError::GpuError(format!("htod query: {e}")))?;
        let d_key = dev
            .htod_sync_copy(&k_f32)
            .map_err(|e| LLMError::GpuError(format!("htod key_cache: {e}")))?;
        let d_val = dev
            .htod_sync_copy(&v_f32)
            .map_err(|e| LLMError::GpuError(format!("htod value_cache: {e}")))?;
        let d_block_tables = dev
            .htod_sync_copy(&block_tables.data)
            .map_err(|e| LLMError::GpuError(format!("htod block_tables: {e}")))?;
        let d_context_lens = dev
            .htod_sync_copy(&context_lens.data)
            .map_err(|e| LLMError::GpuError(format!("htod context_lens: {e}")))?;

        let output_len = num_tokens * num_heads * head_dim;
        let d_output: CudaSlice<f32> = dev
            .alloc_zeros(output_len)
            .map_err(|e| LLMError::GpuError(format!("output alloc: {e}")))?;

        // Shared memory: 2 * TILE_BC * head_dim + TILE_BC + (128/32) floats
        let smem_bytes = ((2 * TILE_BC * head_dim + TILE_BC + 4) * std::mem::size_of::<f32>()) as u32;

        let is_decode = num_tokens == num_seqs; // one token per sequence

        if is_decode {
            let func = dev
                .get_func("flash_attention", "flash_attention_2_decode_kernel")
                .ok_or_else(|| {
                    LLMError::GpuError("flash_attention_2_decode_kernel not found".into())
                })?;

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: smem_bytes,
            };

            unsafe {
                func.launch_on_stream(
                    &state.stream,
                    cfg,
                    (
                        &d_output,
                        &d_query,
                        &d_key,
                        &d_val,
                        &d_block_tables,
                        &d_context_lens,
                        scale,
                        num_heads as i32,
                        num_kv_heads as i32,
                        head_dim as i32,
                        block_size as i32,
                        max_blocks_per_seq as i32,
                    ),
                )
                .map_err(|e| {
                    LLMError::GpuError(format!("flash_attention_2_decode_kernel launch: {e}"))
                })?;
            }
        } else {
            // Prefill path: build seq_start_pos
            let mut seq_starts = Vec::with_capacity(num_seqs);
            let mut pos = 0i32;
            for i in 0..num_seqs {
                seq_starts.push(pos);
                let ctx = context_lens.data[i];
                pos += if i + 1 < num_seqs { 1 } else { (num_tokens as i32) - pos };
            }
            let d_seq_starts = dev
                .htod_sync_copy(&seq_starts)
                .map_err(|e| LLMError::GpuError(format!("htod seq_starts: {e}")))?;

            let func = dev
                .get_func("flash_attention", "flash_attention_2_kernel")
                .ok_or_else(|| {
                    LLMError::GpuError("flash_attention_2_kernel not found".into())
                })?;

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: smem_bytes,
            };

            unsafe {
                func.launch_on_stream(
                    &state.stream,
                    cfg,
                    (
                        &d_output,
                        &d_query,
                        &d_key,
                        &d_val,
                        &d_block_tables,
                        &d_context_lens,
                        &d_seq_starts,
                        scale,
                        num_heads as i32,
                        num_kv_heads as i32,
                        head_dim as i32,
                        block_size as i32,
                        max_context_len as i32,
                        max_blocks_per_seq as i32,
                        num_tokens as i32,
                        if self.config.causal { 1i32 } else { 0i32 },
                    ),
                )
                .map_err(|e| {
                    LLMError::GpuError(format!("flash_attention_2_kernel launch: {e}"))
                })?;
            }
        }

        // Sync and download
        dev.wait_for(&state.stream)
            .map_err(|e| LLMError::GpuError(format!("stream sync: {e}")))?;

        let host_output: Vec<f32> = dev
            .dtoh_sync_copy(&d_output)
            .map_err(|e| LLMError::GpuError(format!("dtoh output: {e}")))?;

        let f16_output: Vec<f16> = host_output.iter().map(|v| f16::from_f32(*v)).collect();

        Ok(GpuBuffer {
            data: f16_output,
            shape: vec![num_tokens, num_heads, head_dim],
        })
    }
}

impl Default for FlashAttention2 {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionBackend for FlashAttention2 {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        #[cfg(feature = "cuda")]
        if self.cuda_state.is_some() {
            return self.forward_cuda(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                max_context_len,
                scale,
            );
        }

        self.forward_cpu(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            max_context_len,
            scale,
        )
    }

    fn name(&self) -> &str {
        #[cfg(feature = "cuda")]
        if self.cuda_state.is_some() {
            return "FlashAttention2-CUDA";
        }
        "FlashAttention2-CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f16(val: f32) -> f16 {
        f16::from_f32(val)
    }

    fn make_const_buf(val: f32, shape: Vec<usize>) -> GpuBuffer<f16> {
        let n: usize = shape.iter().product();
        GpuBuffer {
            data: vec![make_f16(val); n],
            shape,
        }
    }

    #[test]
    fn name_cpu() {
        let fa = FlashAttention2::new();
        assert_eq!(fa.name(), "FlashAttention2-CPU");
    }

    #[test]
    fn rejects_unsupported_head_dim() {
        let fa = FlashAttention2::new();
        // head_dim=32 is not in SUPPORTED_HEAD_DIMS
        let query = make_const_buf(1.0, vec![1, 4, 32]);
        let kc = make_const_buf(1.0, vec![1, 16, 4, 32]);
        let vc = make_const_buf(1.0, vec![1, 16, 4, 32]);
        let bt = GpuBuffer { data: vec![0], shape: vec![1, 1] };
        let cl: GpuBuffer<i32> = GpuBuffer { data: vec![1], shape: vec![1] };
        assert!(fa.forward(&query, &kc, &vc, &bt, &cl, 1, 0.125).is_err());
    }

    #[test]
    fn rejects_bad_query_dims() {
        let fa = FlashAttention2::new();
        let query = GpuBuffer { data: vec![make_f16(1.0); 16], shape: vec![4, 4] };
        let kc = make_const_buf(1.0, vec![1, 1, 1, 64]);
        let vc = make_const_buf(1.0, vec![1, 1, 1, 64]);
        let bt = GpuBuffer { data: vec![0], shape: vec![1, 1] };
        let cl: GpuBuffer<i32> = GpuBuffer { data: vec![1], shape: vec![1] };
        assert!(fa.forward(&query, &kc, &vc, &bt, &cl, 1, 0.125).is_err());
    }

    #[test]
    fn empty_batch_returns_empty() {
        let fa = FlashAttention2::new();
        let query = GpuBuffer { data: Vec::new(), shape: vec![0, 4, 64] };
        let kc = GpuBuffer { data: Vec::new(), shape: vec![0, 16, 4, 64] };
        let vc = GpuBuffer { data: Vec::new(), shape: vec![0, 16, 4, 64] };
        let bt = GpuBuffer { data: Vec::new(), shape: vec![0, 0] };
        let cl: GpuBuffer<i32> = GpuBuffer { data: Vec::new(), shape: vec![0] };
        let out = fa.forward(&query, &kc, &vc, &bt, &cl, 0, 0.125).unwrap();
        assert!(out.data.is_empty());
    }

    #[test]
    fn uniform_attention_single_seq() {
        // With uniform K and V (all 1s), query=[1,1,...], scale=1/sqrt(head_dim):
        // All attention weights should be equal, output should be the value vector.
        let num_heads = 2;
        let head_dim = 64;
        let block_size = 16;
        let ctx_len = 4;
        let num_blocks_needed = (ctx_len + block_size - 1) / block_size; // 1

        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: false,
            num_kv_heads: None,
        });

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Query: [1, num_heads, head_dim] all ones
        let query = make_const_buf(1.0, vec![1, num_heads, head_dim]);

        // KV cache: [num_blocks, block_size, num_heads, head_dim]
        // Fill the first block with 1s (ctx_len=4 tokens used)
        let kc = make_const_buf(1.0, vec![num_blocks_needed, block_size, num_heads, head_dim]);
        let vc = make_const_buf(1.0, vec![num_blocks_needed, block_size, num_heads, head_dim]);

        // Block tables: seq 0 uses physical block 0
        let bt = GpuBuffer {
            data: vec![0i32],
            shape: vec![1, 1],
        };
        let cl: GpuBuffer<i32> = GpuBuffer {
            data: vec![ctx_len as i32],
            shape: vec![1],
        };

        let out = fa
            .forward(&query, &kc, &vc, &bt, &cl, ctx_len, scale)
            .unwrap();

        assert_eq!(out.shape, vec![1, num_heads, head_dim]);
        for &v in &out.data {
            let f = v.to_f32();
            assert!(
                (f - 1.0).abs() < 0.02,
                "expected ~1.0, got {f}"
            );
        }
    }

    #[test]
    fn causal_mask_blocks_future() {
        // 2 query tokens attending to 2 KV positions with causal masking.
        // Token 0 can only see position 0.
        // Token 1 can see positions 0 and 1.
        let num_heads = 1;
        let head_dim = 64;
        let block_size = 16;
        let ctx_len = 2;

        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: true,
            num_kv_heads: None,
        });

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Query: [2, 1, 64]
        let query = make_const_buf(1.0, vec![2, num_heads, head_dim]);

        // K: all ones so dot products are equal (head_dim * scale)
        let kc = make_const_buf(1.0, vec![1, block_size, num_heads, head_dim]);
        // V: position 0 has value 0.5, position 1 has value 1.5
        let mut vc_data = vec![f16::ZERO; block_size * num_heads * head_dim];
        for d in 0..head_dim {
            vc_data[0 * num_heads * head_dim + d] = make_f16(0.5);
            vc_data[1 * num_heads * head_dim + d] = make_f16(1.5);
        }
        let vc = GpuBuffer {
            data: vc_data,
            shape: vec![1, block_size, num_heads, head_dim],
        };

        let bt = GpuBuffer { data: vec![0i32], shape: vec![1, 1] };
        let cl: GpuBuffer<i32> = GpuBuffer { data: vec![ctx_len as i32], shape: vec![1] };

        let out = fa
            .forward(&query, &kc, &vc, &bt, &cl, ctx_len, scale)
            .unwrap();

        assert_eq!(out.shape, vec![2, num_heads, head_dim]);

        // Token 0 (first query): can only see pos 0, output should be ~0.5
        for d in 0..head_dim {
            let v = out.data[0 * num_heads * head_dim + d].to_f32();
            assert!(
                (v - 0.5).abs() < 0.05,
                "token 0, dim {d}: expected ~0.5, got {v}"
            );
        }

        // Token 1 (second query): sees pos 0 and 1, equal weights => average of 0.5 and 1.5 = 1.0
        for d in 0..head_dim {
            let v = out.data[1 * num_heads * head_dim + d].to_f32();
            assert!(
                (v - 1.0).abs() < 0.05,
                "token 1, dim {d}: expected ~1.0, got {v}"
            );
        }
    }

    #[test]
    fn multi_block_paged_attention() {
        // Test with context spanning multiple physical blocks.
        let num_heads = 1;
        let head_dim = 64;
        let block_size = 4;
        let ctx_len = 10; // spans 3 blocks (4+4+2)

        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: false,
            num_kv_heads: None,
        });

        let scale = 1.0 / (head_dim as f32).sqrt();

        let query = make_const_buf(1.0, vec![1, num_heads, head_dim]);

        // Allocate 5 physical blocks, use blocks 2, 0, 4 for this sequence
        let num_phys_blocks = 5;
        let kc = make_const_buf(1.0, vec![num_phys_blocks, block_size, num_heads, head_dim]);
        let vc = make_const_buf(1.0, vec![num_phys_blocks, block_size, num_heads, head_dim]);

        let bt = GpuBuffer {
            data: vec![2i32, 0, 4], // physical block IDs
            shape: vec![1, 3],
        };
        let cl: GpuBuffer<i32> = GpuBuffer {
            data: vec![ctx_len as i32],
            shape: vec![1],
        };

        let out = fa
            .forward(&query, &kc, &vc, &bt, &cl, ctx_len, scale)
            .unwrap();

        assert_eq!(out.shape, vec![1, num_heads, head_dim]);
        // Uniform K and V => output should be ~1.0
        for &v in &out.data {
            let f = v.to_f32();
            assert!(
                (f - 1.0).abs() < 0.02,
                "expected ~1.0, got {f}"
            );
        }
    }

    #[test]
    fn gqa_kv_head_mapping() {
        // GQA: 4 query heads, 2 KV heads. Heads 0,1 -> KV head 0; Heads 2,3 -> KV head 1.
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let block_size = 16;
        let ctx_len = 2;

        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: false,
            num_kv_heads: Some(num_kv_heads),
        });

        let scale = 1.0 / (head_dim as f32).sqrt();

        let query = make_const_buf(1.0, vec![1, num_heads, head_dim]);
        let kc = make_const_buf(1.0, vec![1, block_size, num_kv_heads, head_dim]);
        let vc = make_const_buf(1.0, vec![1, block_size, num_kv_heads, head_dim]);

        let bt = GpuBuffer { data: vec![0i32], shape: vec![1, 1] };
        let cl: GpuBuffer<i32> = GpuBuffer { data: vec![ctx_len as i32], shape: vec![1] };

        let out = fa
            .forward(&query, &kc, &vc, &bt, &cl, ctx_len, scale)
            .unwrap();

        assert_eq!(out.shape, vec![1, num_heads, head_dim]);
        for &v in &out.data {
            let f = v.to_f32();
            assert!(
                (f - 1.0).abs() < 0.02,
                "expected ~1.0, got {f}"
            );
        }
    }

    #[test]
    fn head_dim_96_works() {
        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: false,
            num_kv_heads: None,
        });
        let head_dim = 96;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let query = make_const_buf(1.0, vec![1, 1, head_dim]);
        let kc = make_const_buf(1.0, vec![1, 16, 1, head_dim]);
        let vc = make_const_buf(1.0, vec![1, 16, 1, head_dim]);
        let bt = GpuBuffer { data: vec![0i32], shape: vec![1, 1] };
        let cl: GpuBuffer<i32> = GpuBuffer { data: vec![2], shape: vec![1] };

        let out = fa.forward(&query, &kc, &vc, &bt, &cl, 2, scale).unwrap();
        assert_eq!(out.shape, vec![1, 1, 96]);
    }

    #[test]
    fn head_dim_128_works() {
        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: false,
            num_kv_heads: None,
        });
        let head_dim = 128;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let query = make_const_buf(1.0, vec![1, 1, head_dim]);
        let kc = make_const_buf(1.0, vec![1, 16, 1, head_dim]);
        let vc = make_const_buf(1.0, vec![1, 16, 1, head_dim]);
        let bt = GpuBuffer { data: vec![0i32], shape: vec![1, 1] };
        let cl: GpuBuffer<i32> = GpuBuffer { data: vec![3], shape: vec![1] };

        let out = fa.forward(&query, &kc, &vc, &bt, &cl, 3, scale).unwrap();
        assert_eq!(out.shape, vec![1, 1, 128]);
    }

    #[test]
    fn long_context_tiling() {
        // Test with context > TILE_BC to exercise multi-tile path
        let num_heads = 1;
        let head_dim = 64;
        let block_size = 16;
        let ctx_len = 200; // > TILE_BC=64, exercises 4 tiles
        let num_blocks = (ctx_len + block_size - 1) / block_size; // 13

        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: false,
            num_kv_heads: None,
        });

        let scale = 1.0 / (head_dim as f32).sqrt();

        let query = make_const_buf(1.0, vec![1, num_heads, head_dim]);
        let kc = make_const_buf(1.0, vec![num_blocks, block_size, num_heads, head_dim]);
        let vc = make_const_buf(1.0, vec![num_blocks, block_size, num_heads, head_dim]);

        // Identity block table
        let bt_data: Vec<i32> = (0..num_blocks as i32).collect();
        let bt = GpuBuffer {
            data: bt_data,
            shape: vec![1, num_blocks],
        };
        let cl: GpuBuffer<i32> = GpuBuffer {
            data: vec![ctx_len as i32],
            shape: vec![1],
        };

        let out = fa
            .forward(&query, &kc, &vc, &bt, &cl, ctx_len, scale)
            .unwrap();

        assert_eq!(out.shape, vec![1, num_heads, head_dim]);
        for &v in &out.data {
            let f = v.to_f32();
            assert!(
                (f - 1.0).abs() < 0.02,
                "expected ~1.0, got {f}"
            );
        }
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FlashAttention2>();
    }

    #[test]
    fn multi_seq_decode() {
        // 3 sequences in decode mode (1 token each)
        let num_heads = 2;
        let head_dim = 64;
        let block_size = 8;

        let fa = FlashAttention2::with_config(FlashAttention2Config {
            causal: false,
            num_kv_heads: None,
        });

        let scale = 1.0 / (head_dim as f32).sqrt();

        // 3 tokens total, 3 seqs
        let query = make_const_buf(1.0, vec![3, num_heads, head_dim]);

        // Physical blocks: 4 blocks available
        let kc = make_const_buf(1.0, vec![4, block_size, num_heads, head_dim]);
        let vc = make_const_buf(1.0, vec![4, block_size, num_heads, head_dim]);

        // Block tables: [3, 2] -- each seq uses up to 2 blocks
        let bt = GpuBuffer {
            data: vec![0, 1, 2, 3, 0, 2],
            shape: vec![3, 2],
        };
        // Context lengths: 5, 3, 8
        let cl: GpuBuffer<i32> = GpuBuffer {
            data: vec![5, 3, 8],
            shape: vec![3],
        };

        let out = fa
            .forward(&query, &kc, &vc, &bt, &cl, 8, scale)
            .unwrap();

        assert_eq!(out.shape, vec![3, num_heads, head_dim]);
        // Uniform KV => all outputs ~1.0
        for &v in &out.data {
            let f = v.to_f32();
            assert!(
                (f - 1.0).abs() < 0.02,
                "expected ~1.0, got {f}"
            );
        }
    }
}
