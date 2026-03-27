//! GPU forward pass orchestrator (Agent 13).
//!
//! `GpuModelRunner` drives the full Llama-family forward pass on CUDA:
//! token embedding lookup -> N transformer layers -> final RMSNorm -> LM head -> logits.
//!
//! All CUDA code is gated behind `#[cfg(feature = "cuda")]`. Under `mock-gpu`
//! (the default), this module provides a compile-compatible stub that returns
//! an error at runtime so existing Mac-side tests keep working.

use std::collections::HashMap;

use tracing::{debug, trace};

use crate::bridge::{LLMError, Result};
use crate::runner::ModelRunnerConfig;

// ---------------------------------------------------------------------------
// ModelRunnerConfig re-used from runner.rs -- already has all the fields we
// need (num_layers, hidden_size, num_heads, etc.).
// ---------------------------------------------------------------------------

// =========================================================================
//  CUDA implementation
// =========================================================================
#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync, LaunchConfig};
    use tracing::{debug, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    // -- Types from other agents (will be real once those agents land) ------
    // Agent 1: KernelLoader
    use rvllm_gpu::kernel_loader::KernelLoader;
    // Agent 11: GpuModelWeights
    use rvllm_model_loader::gpu_weights::GpuModelWeights;
    // Agent 8: CudaCacheEngine
    use rvllm_kv_cache::engine_cuda::CudaCacheEngine;
    // Agent 6: CudaLinearLayer
    use crate::layers::linear_cuda::CudaLinearLayer;
    // Agent 2: CudaRMSNorm
    use crate::layers::norm_cuda::CudaRMSNorm;
    // Agent 3: CudaRotaryEmbedding
    use crate::layers::rotary_cuda::CudaRotaryEmbedding;
    // Agent 12: GpuTransformerLayer
    use crate::gpu_layer::GpuTransformerLayer;
    // cuBLAS from rvllm-gpu
    use rvllm_gpu::prelude::CublasHandle;

    /// GPU model runner -- orchestrates the full causal-LM forward pass on CUDA.
    ///
    /// Holds all GPU-resident weights, the KV cache engine, cuBLAS handle,
    /// kernel loader, and per-layer transformer blocks. Created once at model
    /// load time and reused for every forward call.
    pub struct GpuModelRunner {
        weights: GpuModelWeights,
        cache: CudaCacheEngine,
        blas: CublasHandle,
        loader: KernelLoader,
        config: ModelRunnerConfig,
        device: Arc<CudaDevice>,
        stream: CudaStream,
        /// Per-layer transformer blocks, built at construction.
        layers: Vec<GpuTransformerLayer>,
        /// Embedding table on GPU: [vocab_size, hidden_size] row-major f32.
        embed_tokens: CudaSlice<f32>,
        /// Final RMSNorm weight on GPU: [hidden_size].
        final_norm_weight: CudaSlice<f32>,
        /// LM head weight on GPU: [vocab_size, hidden_size] row-major f32.
        lm_head_weight: CudaSlice<f32>,
        /// RMSNorm epsilon.
        rms_norm_eps: f32,
    }

    impl GpuModelRunner {
        /// Build the runner from loaded GPU weights and pre-initialized components.
        ///
        /// # Errors
        /// Returns `LLMError::GpuError` if any GPU allocation or weight lookup fails.
        pub fn new(
            weights: GpuModelWeights,
            cache: CudaCacheEngine,
            blas: CublasHandle,
            loader: KernelLoader,
            config: ModelRunnerConfig,
            device: Arc<CudaDevice>,
        ) -> Result<Self> {
            debug!(
                num_layers = config.num_layers,
                hidden = config.hidden_size,
                vocab = config.vocab_size,
                "GpuModelRunner::new"
            );

            let stream = CudaStream::new(Arc::clone(&device)).map_err(|e| {
                LLMError::GpuError(format!("stream creation failed: {e}"))
            })?;

            // Extract top-level weights from the weight container.
            let embed_tokens = weights
                .get("model.embed_tokens.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.embed_tokens.weight".into()))?
                .clone();

            let final_norm_weight = weights
                .get("model.norm.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.norm.weight".into()))?
                .clone();

            let lm_head_weight = weights
                .get("lm_head.weight")
                .ok_or_else(|| LLMError::GpuError("missing lm_head.weight".into()))?
                .clone();

            // Build per-layer transformer blocks (Agent 12).
            let mut layers = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let layer = GpuTransformerLayer::new(
                    i,
                    &weights,
                    &config,
                    &blas,
                    &loader,
                    Arc::clone(&device),
                )?;
                layers.push(layer);
            }

            let rms_norm_eps = 1e-5_f32; // standard for Llama-family

            Ok(Self {
                weights,
                cache,
                blas,
                loader,
                config,
                device,
                stream,
                layers,
                embed_tokens,
                final_norm_weight,
                lm_head_weight,
                rms_norm_eps,
            })
        }

        /// Execute a single forward pass, returning logits on CPU.
        ///
        /// * `token_ids`    -- token IDs for this step, len = num_tokens
        /// * `positions`    -- position ID per token, len = num_tokens
        /// * `block_tables` -- per-sequence block table for paged attention
        /// * `context_lens` -- per-sequence context length
        ///
        /// Returns `Vec<f32>` of shape `[num_tokens, vocab_size]` in row-major order.
        pub fn forward(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            block_tables: &[Vec<u32>],
            context_lens: &[u32],
        ) -> Result<Vec<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            debug!(num_tokens, "GpuModelRunner::forward");

            // ------------------------------------------------------------------
            // Step 1: Token embedding lookup on GPU.
            //
            // embed_tokens is [vocab_size, hidden_size]. We gather rows indexed
            // by token_ids to produce hidden_states [num_tokens, hidden_size].
            // ------------------------------------------------------------------
            let mut hidden_states = self.embedding_lookup(token_ids)?;

            // ------------------------------------------------------------------
            // Step 2: N transformer layers (Agent 12).
            //
            // Each layer takes hidden_states and returns updated hidden_states,
            // writing K/V into the paged cache.
            // ------------------------------------------------------------------
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                trace!(layer = layer_idx, "gpu transformer layer");
                hidden_states = layer.forward(
                    &hidden_states,
                    positions,
                    block_tables,
                    context_lens,
                    &self.cache,
                    &self.blas,
                    &self.loader,
                    &self.stream,
                    layer_idx,
                )?;
            }

            // ------------------------------------------------------------------
            // Step 3: Final RMSNorm.
            // ------------------------------------------------------------------
            let normed = CudaRMSNorm::forward(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                &self.stream,
            )?;

            // ------------------------------------------------------------------
            // Step 4: LM head projection -> logits [num_tokens, vocab_size].
            //
            // logits = normed @ lm_head_weight^T
            // normed: [num_tokens, hidden_size]
            // lm_head_weight: [vocab_size, hidden_size]
            // output: [num_tokens, vocab_size]
            // ------------------------------------------------------------------
            let logits_gpu = CudaLinearLayer::forward(
                &normed,
                &self.lm_head_weight,
                None, // no bias on LM head
                num_tokens,
                vocab_size,
                hidden_size,
                &self.blas,
            )?;

            // ------------------------------------------------------------------
            // Step 5: Transfer logits back to CPU.
            // ------------------------------------------------------------------
            // SAFETY: CudaSlice<f32>::copy_to_host performs a synchronous DtoH
            // copy through cudarc, which handles the device synchronization.
            let logits_cpu = self
                .device
                .dtoh_sync_copy(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH failed: {e}")))?;

            debug!(
                logits_len = logits_cpu.len(),
                expected = num_tokens * vocab_size,
                "forward complete"
            );

            Ok(logits_cpu)
        }

        /// Gather embedding rows for the given token IDs.
        ///
        /// Copies token_ids to GPU, then launches a gather kernel (or falls
        /// back to a host-side gather + HtoD copy).
        fn embedding_lookup(&self, token_ids: &[u32]) -> Result<CudaSlice<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;
            let output_len = num_tokens * hidden_size;

            // Upload token_ids to device.
            let token_ids_gpu = self
                .device
                .htod_sync_copy(token_ids)
                .map_err(|e| LLMError::GpuError(format!("token_ids HtoD failed: {e}")))?;

            // Try launching the embedding gather kernel if the kernel loader has it.
            // Fallback: do embedding lookup on CPU and upload.
            // The dedicated kernel avoids a full embedding table DtoH round-trip.
            //
            // For now we use the fallback path which is correct and simple.
            // A dedicated CUDA kernel (embedding_gather.cu) can be added later
            // for better throughput on large batch sizes.
            let embed_host = self
                .device
                .dtoh_sync_copy(&self.embed_tokens)
                .map_err(|e| LLMError::GpuError(format!("embed DtoH failed: {e}")))?;

            let mut output = vec![0.0f32; output_len];
            for (t, &tid) in token_ids.iter().enumerate() {
                let src_start = tid as usize * hidden_size;
                let src_end = src_start + hidden_size;
                let dst_start = t * hidden_size;
                if src_end <= embed_host.len() {
                    output[dst_start..dst_start + hidden_size]
                        .copy_from_slice(&embed_host[src_start..src_end]);
                }
                // Out-of-range tokens stay as zeros (same behavior as CPU path).
            }

            let output_gpu = self
                .device
                .htod_sync_copy(&output)
                .map_err(|e| LLMError::GpuError(format!("embed output HtoD failed: {e}")))?;

            Ok(output_gpu)
        }

        /// Access the underlying model config.
        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        /// Access the cache engine (for external cache management).
        pub fn cache(&self) -> &CudaCacheEngine {
            &self.cache
        }

        /// Access the cache engine mutably (for swap/copy ops).
        pub fn cache_mut(&mut self) -> &mut CudaCacheEngine {
            &mut self.cache
        }
    }
}

// Re-export under cuda feature gate.
#[cfg(feature = "cuda")]
pub use cuda_impl::GpuModelRunner;

// =========================================================================
//  Mock-GPU stub (default feature)
// =========================================================================
#[cfg(not(feature = "cuda"))]
mod mock_impl {
    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    /// Stub GpuModelRunner for non-CUDA builds.
    ///
    /// Allows downstream code to reference the type without conditional
    /// compilation everywhere. All methods return an error at runtime.
    pub struct GpuModelRunner {
        config: ModelRunnerConfig,
    }

    impl GpuModelRunner {
        /// Returns an error -- real CUDA is required.
        pub fn forward(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _block_tables: &[Vec<u32>],
            _context_lens: &[u32],
        ) -> Result<Vec<f32>> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use mock_impl::GpuModelRunner;

// =========================================================================
//  Tests (run under mock-gpu / default features)
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_runner_returns_error() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 2,
                hidden_size: 64,
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 16,
                intermediate_size: 128,
                vocab_size: 100,
                max_position: 512,
                dtype: "float32".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            let result = runner.forward(&[1, 2, 3], &[0, 1, 2], &[], &[]);
            assert!(result.is_err());
            let err_msg = format!("{}", result.unwrap_err());
            assert!(err_msg.contains("cuda"));
        }
    }

    #[test]
    fn config_accessible() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 4,
                hidden_size: 256,
                num_heads: 8,
                num_kv_heads: 8,
                head_dim: 32,
                intermediate_size: 512,
                vocab_size: 32000,
                max_position: 2048,
                dtype: "float16".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            assert_eq!(runner.config().num_layers, 4);
            assert_eq!(runner.config().vocab_size, 32000);
        }
    }
}
