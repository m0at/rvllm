//! GPU forward pass orchestrator (Agent 13).
//!
//! `GpuModelRunner` drives the full Llama-family forward pass on CUDA:
//! token embedding lookup -> N transformer layers -> final RMSNorm -> LM head -> logits.
//!
//! All CUDA code is gated behind `#[cfg(feature = "cuda")]`. Under `mock-gpu`
//! (the default), this module provides a compile-compatible stub that returns
//! an error at runtime so existing Mac-side tests keep working.

// =========================================================================
//  CUDA implementation
// =========================================================================
#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
    use tracing::{debug, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    use rvllm_gpu::kernel_loader::KernelLoader;
    use rvllm_model_loader::gpu_weights::GpuModelWeights;
    use rvllm_kv_cache::engine_cuda::CudaCacheEngine;
    use crate::layers::linear_cuda::CudaLinearLayer;
    use crate::layers::norm_cuda::CudaRMSNorm;
    use crate::gpu_layer::{GpuLayerConfig, GpuLayerInput, GpuLayerWeights, GpuTransformerLayer};
    use rvllm_gpu::prelude::CublasHandle;

    pub struct GpuModelRunner {
        weights: GpuModelWeights,
        cache: CudaCacheEngine,
        blas: CublasHandle,
        loader: KernelLoader,
        config: ModelRunnerConfig,
        device: Arc<CudaDevice>,
        stream: CudaStream,
        layers: Vec<GpuTransformerLayer>,
        embed_tokens: CudaSlice<f32>,
        final_norm_weight: CudaSlice<f32>,
        lm_head_weight: CudaSlice<f32>,
        rms_norm_eps: f32,
    }

    impl GpuModelRunner {
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

            // cudarc 0.12: streams are created via fork_default_stream()
            let stream = device.fork_default_stream()
                .map_err(|e| LLMError::GpuError(format!("stream creation failed: {e}")))?;

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

            let mut layers = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let layer_cfg = GpuLayerConfig {
                    hidden_size: config.hidden_size,
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                    intermediate_size: config.intermediate_size,
                    rms_norm_eps: 1e-5_f32,
                    layer_idx: i,
                };
                layers.push(GpuTransformerLayer::new(layer_cfg, Arc::clone(&device)));
            }

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
                rms_norm_eps: 1e-5_f32,
            })
        }

        pub fn forward(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            block_tables: &[Vec<u32>],
            context_lens: &[u32],
        ) -> Result<Vec<f32>> {
            let num_tokens = token_ids.len();
            let num_seqs = context_lens.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            debug!(num_tokens, "GpuModelRunner::forward");

            // Upload positions to GPU (reused across all layers)
            let positions_gpu: CudaSlice<u32> = self.device
                .htod_sync_copy(positions)
                .map_err(|e| LLMError::GpuError(format!("positions HtoD: {e}")))?;

            // Upload context_lens to GPU
            let context_lens_gpu: CudaSlice<u32> = self.device
                .htod_sync_copy(context_lens)
                .map_err(|e| LLMError::GpuError(format!("context_lens HtoD: {e}")))?;

            // Flatten block_tables to [num_seqs, max_blocks_per_seq] row-major
            let max_blocks = block_tables.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
            let mut flat_bt = vec![0u32; num_seqs * max_blocks];
            for (s, row) in block_tables.iter().enumerate() {
                for (b, &blk) in row.iter().enumerate() {
                    flat_bt[s * max_blocks + b] = blk;
                }
            }
            let block_tables_gpu: CudaSlice<u32> = self.device
                .htod_sync_copy(&flat_bt)
                .map_err(|e| LLMError::GpuError(format!("block_tables HtoD: {e}")))?;

            // Zero slot_mapping -- cache writes not yet wired, only decode path used
            let slot_mapping_gpu: CudaSlice<u32> = self.device
                .alloc_zeros(num_tokens)
                .map_err(|e| LLMError::GpuError(format!("slot_mapping alloc: {e}")))?;

            let max_context_len = context_lens.iter().copied().max().unwrap_or(0);

            // Step 1: token embedding lookup
            let mut hidden_states = self.embedding_lookup(token_ids)?;

            // Step 2: transformer layers
            let gpu_cache = self.cache.gpu_cache();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                trace!(layer = layer_idx, "gpu transformer layer");
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let weights = self.layer_weights(layer_idx)?;
                let input = GpuLayerInput {
                    hidden_states: &hidden_states,
                    positions: &positions_gpu,
                    key_cache,
                    value_cache,
                    block_tables: &block_tables_gpu,
                    context_lens: &context_lens_gpu,
                    slot_mapping: &slot_mapping_gpu,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                };
                hidden_states = layer.forward(&input, &weights, &self.blas)?;
            }

            // Step 3: final RMSNorm
            let normed = CudaRMSNorm::forward(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                &self.stream,
            )?;

            // Step 4: LM head  normed [num_tokens, hidden] @ lm_head^T [hidden, vocab]
            let logits_gpu = CudaLinearLayer::forward_once(
                &normed,
                &self.lm_head_weight,
                None,
                num_tokens,
                vocab_size,
                hidden_size,
                &self.blas,
            )?;

            // Step 5: DtoH
            let logits_cpu = self.device
                .dtoh_sync_copy(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;

            debug!(logits_len = logits_cpu.len(), expected = num_tokens * vocab_size, "forward complete");
            Ok(logits_cpu)
        }

        /// Per-layer weight references into the GPU weight map.
        fn layer_weights(&self, i: usize) -> Result<GpuLayerWeights<'_>> {
            let g = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights.get(name).ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeights {
                input_layernorm:           g(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj:                    g(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj:                    g(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj:                    g(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj:                    g(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                post_attention_layernorm:  g(&format!("model.layers.{i}.post_attention_layernorm.weight"))?,
                gate_proj:                 g(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj:                   g(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj:                 g(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }

        fn embedding_lookup(&self, token_ids: &[u32]) -> Result<CudaSlice<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;

            // CPU gather fallback -- embed_gather.cu kernel can replace this later
            let embed_host = self.device
                .dtoh_sync_copy(&self.embed_tokens)
                .map_err(|e| LLMError::GpuError(format!("embed DtoH: {e}")))?;

            let mut output = vec![0.0f32; num_tokens * hidden_size];
            for (t, &tid) in token_ids.iter().enumerate() {
                let src = tid as usize * hidden_size;
                if src + hidden_size <= embed_host.len() {
                    output[t * hidden_size..t * hidden_size + hidden_size]
                        .copy_from_slice(&embed_host[src..src + hidden_size]);
                }
            }

            self.device
                .htod_sync_copy(&output)
                .map_err(|e| LLMError::GpuError(format!("embed HtoD: {e}")))
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn cache(&self) -> &CudaCacheEngine {
            &self.cache
        }

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
