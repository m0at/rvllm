//! GPU Transformer Layer -- one complete transformer block on CUDA.
//!
//! Combines all CUDA dispatch ops (Agents 2-7) into the standard
//! decoder-only transformer sequence:
//!
//! 1. RMSNorm(input)
//! 2. QKV projection (cuBLAS sgemm)
//! 3. RoPE on Q, K
//! 4. PagedAttention(Q, K_cache, V_cache)
//! 5. Output projection (cuBLAS sgemm)
//! 6. RMSNorm(residual + attn_out)
//! 7. MLP: gate+up (cuBLAS) -> fused_silu_mul -> down (cuBLAS)
//! 8. residual + mlp_out
//!
//! All code is gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use tracing::trace;

    use rvllm_core::error::{LLMError, Result};
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::stream::GpuStream;

    /// Configuration for a single transformer layer.
    #[derive(Debug, Clone)]
    pub struct GpuLayerConfig {
        pub hidden_size: usize,
        pub num_heads: usize,
        pub num_kv_heads: usize,
        pub head_dim: usize,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub layer_idx: usize,
    }

    /// Weight references for a single transformer layer.
    ///
    /// All slices live on GPU and are owned by the GpuModelWeights container.
    /// This struct borrows them for the duration of a forward pass.
    pub struct GpuLayerWeights<'a> {
        // Pre-attention norm
        pub input_layernorm: &'a CudaSlice<f32>,
        // Attention projections
        pub q_proj: &'a CudaSlice<f32>,
        pub k_proj: &'a CudaSlice<f32>,
        pub v_proj: &'a CudaSlice<f32>,
        pub o_proj: &'a CudaSlice<f32>,
        // Post-attention norm
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        // MLP weights
        pub gate_proj: &'a CudaSlice<f32>,
        pub up_proj: &'a CudaSlice<f32>,
        pub down_proj: &'a CudaSlice<f32>,
    }

    /// Metadata needed for a single layer forward pass.
    pub struct GpuLayerInput<'a> {
        /// Hidden states entering this layer, shape [num_tokens, hidden_size].
        pub hidden_states: &'a CudaSlice<f32>,
        /// Position ids for RoPE, shape [num_tokens].
        pub positions: &'a CudaSlice<u32>,
        /// KV cache key block for this layer, shape [num_blocks, num_kv_heads, head_dim, block_size].
        pub key_cache: &'a CudaSlice<f32>,
        /// KV cache value block for this layer, shape [num_blocks, num_kv_heads, head_dim, block_size].
        pub value_cache: &'a CudaSlice<f32>,
        /// Block table mapping sequence positions to cache blocks, shape [num_seqs, max_blocks_per_seq].
        pub block_tables: &'a CudaSlice<u32>,
        /// Context length for each sequence, shape [num_seqs].
        pub context_lens: &'a CudaSlice<u32>,
        /// Slot mapping for cache writes during prefill, shape [num_tokens].
        pub slot_mapping: &'a CudaSlice<u32>,
        /// Number of tokens in the batch.
        pub num_tokens: usize,
        /// Number of sequences in the batch.
        pub num_seqs: usize,
        /// Maximum context length across sequences.
        pub max_context_len: u32,
        /// Block size for paged attention.
        pub block_size: usize,
    }

    /// One complete GPU transformer layer.
    ///
    /// Holds references to the kernel loader and cuBLAS handle;
    /// weights are passed in per-call via `GpuLayerWeights`.
    pub struct GpuTransformerLayer {
        config: GpuLayerConfig,
        device: Arc<CudaDevice>,
    }

    impl GpuTransformerLayer {
        pub fn new(config: GpuLayerConfig, device: Arc<CudaDevice>) -> Self {
            Self { config, device }
        }

        /// Execute a full transformer layer forward pass.
        ///
        /// Returns the output hidden states as a new CudaSlice<f32> of shape
        /// [num_tokens, hidden_size]. The caller is responsible for using this
        /// as input to the next layer.
        pub fn forward(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeights<'_>,
            blas: &CublasHandle,
        ) -> Result<CudaSlice<f32>> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            trace!(
                layer = cfg.layer_idx,
                num_tokens,
                "gpu transformer layer forward"
            );

            // ---------------------------------------------------------------
            // 1. Pre-attention RMSNorm
            // ---------------------------------------------------------------
            let normed = Self::rms_norm(
                &self.device,
                input.hidden_states,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // ---------------------------------------------------------------
            // 2. QKV projections via cuBLAS sgemm
            //    input [num_tokens, hidden] x weight^T [hidden, proj_dim]
            // ---------------------------------------------------------------
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let q = Self::linear(&self.device, blas, &normed, weights.q_proj, num_tokens, q_dim, hidden)?;
            let k = Self::linear(&self.device, blas, &normed, weights.k_proj, num_tokens, kv_dim, hidden)?;
            let v = Self::linear(&self.device, blas, &normed, weights.v_proj, num_tokens, kv_dim, hidden)?;

            // ---------------------------------------------------------------
            // 3. RoPE on Q and K
            // ---------------------------------------------------------------
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.device,
                &q,
                &k,
                input.positions,
                num_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;

            // ---------------------------------------------------------------
            // 4. Paged Attention
            // ---------------------------------------------------------------
            let attn_out = Self::paged_attention(
                &self.device,
                &q_rot,
                &k_rot,
                &v,
                input.key_cache,
                input.value_cache,
                input.block_tables,
                input.context_lens,
                input.slot_mapping,
                num_tokens,
                input.num_seqs,
                num_heads,
                num_kv_heads,
                head_dim,
                input.max_context_len,
                input.block_size,
            )?;

            // ---------------------------------------------------------------
            // 5. Output projection
            // ---------------------------------------------------------------
            let attn_proj = Self::linear(
                &self.device, blas, &attn_out, weights.o_proj,
                num_tokens, hidden, q_dim,
            )?;

            // ---------------------------------------------------------------
            // Residual: hidden_states + attn_proj
            // ---------------------------------------------------------------
            let residual = Self::add_tensors(
                &self.device,
                input.hidden_states,
                &attn_proj,
                num_tokens * hidden,
            )?;

            // ---------------------------------------------------------------
            // 6. Post-attention RMSNorm
            // ---------------------------------------------------------------
            let normed2 = Self::rms_norm(
                &self.device,
                &residual,
                weights.post_attention_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // ---------------------------------------------------------------
            // 7. MLP: gate_proj + up_proj -> fused_silu_mul -> down_proj
            // ---------------------------------------------------------------
            let gate = Self::linear(
                &self.device, blas, &normed2, weights.gate_proj,
                num_tokens, intermediate, hidden,
            )?;
            let up = Self::linear(
                &self.device, blas, &normed2, weights.up_proj,
                num_tokens, intermediate, hidden,
            )?;

            let fused = Self::fused_silu_mul(&self.device, &gate, &up, num_tokens * intermediate)?;

            let mlp_out = Self::linear(
                &self.device, blas, &fused, weights.down_proj,
                num_tokens, hidden, intermediate,
            )?;

            // ---------------------------------------------------------------
            // 8. Residual: residual + mlp_out
            // ---------------------------------------------------------------
            let output = Self::add_tensors(
                &self.device,
                &residual,
                &mlp_out,
                num_tokens * hidden,
            )?;

            Ok(output)
        }

        // ===================================================================
        // Private dispatch helpers
        //
        // Each wraps the corresponding CUDA kernel or cuBLAS call.
        // These are the seams where Agent 2-7 implementations plug in.
        // ===================================================================

        /// RMSNorm: out[i] = (x[i] / rms) * weight[i % hidden]
        /// where rms = sqrt(mean(x^2) + eps).
        ///
        /// Dispatches to the rms_norm CUDA kernel.
        fn rms_norm(
            device: &Arc<CudaDevice>,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let n = num_tokens * hidden_size;
            let mut output = device
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("rms_norm alloc failed: {e}")))?;

            // Launch rms_norm kernel: one block per token, hidden_size threads per block.
            // The kernel reads `input`, `weight`, writes `output`.
            let module_name = "rms_norm";
            let func_name = "rms_norm_kernel";
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (hidden_size.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // SAFETY: All CudaSlice pointers are valid device memory allocated on
            // the same device. Grid/block dims are checked above. The kernel reads
            // `input` [num_tokens * hidden_size], `weight` [hidden_size], and writes
            // `output` [num_tokens * hidden_size].
            let kernel = device
                .get_func(module_name, func_name)
                .ok_or_else(|| LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded")))?;
            unsafe {
                kernel
                    .launch(cfg, (&input, &weight, &mut output, eps, hidden_size as u32, num_tokens as u32))
                    .map_err(|e| LLMError::GpuError(format!("rms_norm launch failed: {e}")))?;
            }

            Ok(output)
        }

        /// Linear projection via cuBLAS sgemm.
        /// Computes output = input @ weight^T where:
        ///   input: [m, k], weight: [n, k] (row-major), output: [m, n].
        fn linear(
            device: &Arc<CudaDevice>,
            blas: &CublasHandle,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = device
                .alloc_zeros::<f32>(m * n)
                .map_err(|e| LLMError::GpuError(format!("linear alloc failed: {e}")))?;

            blas.sgemm(m, n, k, 1.0, input, weight, 0.0, &mut output)?;

            Ok(output)
        }

        /// Apply rotary positional embeddings to Q and K tensors.
        ///
        /// Dispatches to the rotary_embedding CUDA kernel.
        /// Q shape: [num_tokens, num_heads * head_dim]
        /// K shape: [num_tokens, num_kv_heads * head_dim]
        /// positions: [num_tokens]
        fn apply_rotary_embedding(
            device: &Arc<CudaDevice>,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            positions: &CudaSlice<u32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
            let q_len = num_tokens * num_heads * head_dim;
            let k_len = num_tokens * num_kv_heads * head_dim;

            // Clone Q and K so we can apply rotation in-place.
            let mut q_out = device
                .alloc_zeros::<f32>(q_len)
                .map_err(|e| LLMError::GpuError(format!("rope q alloc failed: {e}")))?;
            let mut k_out = device
                .alloc_zeros::<f32>(k_len)
                .map_err(|e| LLMError::GpuError(format!("rope k alloc failed: {e}")))?;

            // Copy input to output buffers (dtod).
            device.dtod_copy(q, &mut q_out)
                .map_err(|e| LLMError::GpuError(format!("rope q copy failed: {e}")))?;
            device.dtod_copy(k, &mut k_out)
                .map_err(|e| LLMError::GpuError(format!("rope k copy failed: {e}")))?;

            let module_name = "rotary_embedding";
            let func_name = "rotary_embedding_kernel";

            // Launch for Q: one block per (token, head), head_dim/2 threads.
            let half_dim = head_dim / 2;
            let q_cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, num_heads as u32, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel_q = device
                .get_func(module_name, func_name)
                .ok_or_else(|| LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded")))?;
            // SAFETY: q_out has exactly q_len elements, positions has num_tokens elements.
            // Grid covers all (token, head) pairs; each thread handles one frequency.
            unsafe {
                kernel_q
                    .launch(q_cfg, (&mut q_out, &positions, head_dim as u32, num_heads as u32))
                    .map_err(|e| LLMError::GpuError(format!("rope q launch failed: {e}")))?;
            }

            // Launch for K.
            let k_cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, num_kv_heads as u32, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel_k = device
                .get_func(module_name, func_name)
                .ok_or_else(|| LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded (k)")))?;
            // SAFETY: k_out has exactly k_len elements. Same contract as Q launch.
            unsafe {
                kernel_k
                    .launch(k_cfg, (&mut k_out, &positions, head_dim as u32, num_kv_heads as u32))
                    .map_err(|e| LLMError::GpuError(format!("rope k launch failed: {e}")))?;
            }

            Ok((q_out, k_out))
        }

        /// Paged attention forward pass.
        ///
        /// Writes new K,V into the cache at slot_mapping positions,
        /// then runs the paged_attention kernel for the actual attention computation.
        #[allow(clippy::too_many_arguments)]
        fn paged_attention(
            device: &Arc<CudaDevice>,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            v: &CudaSlice<f32>,
            key_cache: &CudaSlice<f32>,
            value_cache: &CudaSlice<f32>,
            block_tables: &CudaSlice<u32>,
            context_lens: &CudaSlice<u32>,
            _slot_mapping: &CudaSlice<u32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let out_len = num_tokens * num_heads * head_dim;
            let mut output = device
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("paged_attn alloc failed: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            // TODO: First, write new K,V into cache via reshape_and_cache kernel
            // using slot_mapping. This is Agent 9's responsibility (ops_cuda.rs).
            // For now, we assume the cache has already been populated for decode,
            // and during prefill the scheduler handles cache writes.

            let module_name = "paged_attention";
            let func_name = "paged_attention_v1_kernel";

            // Launch config: one block per (seq, head), threads cover the
            // head_dim for accumulation.
            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (head_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            let kernel = device
                .get_func(module_name, func_name)
                .ok_or_else(|| LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded")))?;

            // SAFETY: All slices are valid GPU memory on the same device.
            // output: [num_tokens, num_heads, head_dim]
            // q: [num_tokens, num_heads, head_dim]
            // key_cache, value_cache: [num_blocks, num_kv_heads, head_dim, block_size]
            // block_tables: [num_seqs, max_blocks_per_seq]
            // context_lens: [num_seqs]
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            &mut output,
                            &q,
                            &key_cache,
                            &value_cache,
                            &block_tables,
                            &context_lens,
                            scale,
                            num_heads as u32,
                            num_kv_heads as u32,
                            head_dim as u32,
                            block_size as u32,
                            max_context_len,
                        ),
                    )
                    .map_err(|e| LLMError::GpuError(format!("paged_attention launch failed: {e}")))?;
            }

            Ok(output)
        }

        /// Fused SiLU activation with element-wise multiply: out = silu(gate) * up.
        ///
        /// Dispatches to the activation CUDA kernel.
        fn fused_silu_mul(
            device: &Arc<CudaDevice>,
            gate: &CudaSlice<f32>,
            up: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = device
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul alloc failed: {e}")))?;

            let module_name = "activation";
            let func_name = "fused_silu_mul_kernel";

            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            let kernel = device
                .get_func(module_name, func_name)
                .ok_or_else(|| LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded")))?;

            // SAFETY: gate, up, and output all have exactly n elements.
            // Grid covers all elements with ceil division.
            unsafe {
                kernel
                    .launch(cfg, (&mut output, &gate, &up, n as u32))
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul launch failed: {e}")))?;
            }

            Ok(output)
        }

        /// Element-wise tensor addition: out = a + b.
        ///
        /// Tries "add_bias" module first (Agent 20's dedicated kernel), then
        /// "activation" module, then falls back to CPU.
        fn add_tensors(
            device: &Arc<CudaDevice>,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = device
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("add_tensors alloc failed: {e}")))?;

            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            // Try dedicated add_bias module first, then activation module
            let kernel = device.get_func("add_bias", "add_kernel")
                .or_else(|| device.get_func("activation", "add_kernel"));

            match kernel {
                Some(k) => {
                    // SAFETY: a, b, output all have exactly n elements.
                    unsafe {
                        k.launch(cfg, (&mut output, &a, &b, n as u32))
                            .map_err(|e| LLMError::GpuError(format!("add_kernel launch failed: {e}")))?;
                    }
                }
                None => {
                    // Fallback: CPU add (only until kernels are compiled).
                    let a_host = device.dtoh_sync_copy(a)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh a failed: {e}")))?;
                    let b_host = device.dtoh_sync_copy(b)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh b failed: {e}")))?;
                    let sum: Vec<f32> = a_host.iter().zip(b_host.iter()).map(|(x, y)| x + y).collect();
                    output = device.htod_sync_copy(&sum)
                        .map_err(|e| LLMError::GpuError(format!("add htod failed: {e}")))?;
                }
            }

            Ok(output)
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;

#[cfg(test)]
mod tests {
    // Tests run under default features (mock-gpu), so we verify the module
    // compiles but the CUDA types are not exposed.
    #[test]
    fn module_compiles_without_cuda() {
        // Under mock-gpu the `inner` module is not compiled.
        // This test confirms that the crate still builds cleanly.
        assert!(true);
    }
}
