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
/// Output of a forward pass -- either full logits or just argmax token IDs.
#[derive(Debug, Clone)]
pub enum ForwardOutput {
    /// Full logits buffer: [num_tokens * vocab_size] f32.
    Logits(Vec<f32>),
    /// GPU-side argmax token IDs: [num_tokens] i32 (greedy fast path).
    TokenIds(Vec<i32>),
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use half::f16;
    use tracing::{debug, info, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    use crate::gpu_layer::{
        GpuLayerConfig, GpuLayerInput, GpuLayerWeights, GpuLayerWeightsF16, GpuTransformerLayer,
    };
    use crate::layers::linear_cuda::CudaLinearLayer;
    use crate::layers::norm_cuda::CudaRMSNorm;
    use rvllm_gpu::kernel_loader::KernelLoader;
    use rvllm_gpu::prelude::CublasHandle;
    use rvllm_kv_cache::engine_cuda::CudaCacheEngine;
    use rvllm_model_loader::gpu_weights::GpuModelWeights;

    use super::ForwardOutput;

    pub struct GpuModelRunner {
        weights: GpuModelWeights,
        cache: CudaCacheEngine,
        blas: CublasHandle,
        loader: Arc<KernelLoader>,
        config: ModelRunnerConfig,
        device: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        /// Separate stream for async HtoD metadata uploads, overlaps with
        /// the previous step's compute still running on `stream`.
        copy_stream: Arc<CudaStream>,
        layers: Vec<GpuTransformerLayer>,
        embed_tokens: CudaSlice<f32>,
        final_norm_weight: CudaSlice<f32>,
        lm_head_weight: CudaSlice<f32>,
        rms_norm_eps: f32,
        /// Precomputed RoPE cos table on GPU: [max_position, head_dim/2]
        rope_cos: CudaSlice<f32>,
        /// Precomputed RoPE sin table on GPU: [max_position, head_dim/2]
        rope_sin: CudaSlice<f32>,
        /// When true, use hgemm with f16 projection weights instead of sgemm.
        use_fp16: bool,
    }

    impl GpuModelRunner {
        pub fn new(
            weights: GpuModelWeights,
            cache: CudaCacheEngine,
            blas: CublasHandle,
            loader: KernelLoader,
            config: ModelRunnerConfig,
            device: Arc<CudaContext>,
            stream: Arc<CudaStream>,
        ) -> Result<Self> {
            let loader = Arc::new(loader);
            debug!(
                num_layers = config.num_layers,
                hidden = config.hidden_size,
                vocab = config.vocab_size,
                "GpuModelRunner::new"
            );

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
                .or_else(|| weights.get("model.embed_tokens.weight"))
                .ok_or_else(|| {
                    LLMError::GpuError(
                        "missing lm_head.weight and model.embed_tokens.weight".into(),
                    )
                })?
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
                layers.push(GpuTransformerLayer::new(layer_cfg, Arc::clone(&stream), Arc::clone(&loader)));
            }

            // Precompute RoPE cos/sin tables
            let head_dim = config.head_dim;
            let max_pos = config.max_position.min(8192);
            let half_dim = head_dim / 2;
            let rope_theta = config.rope_theta;
            let mut cos_table = vec![0.0f32; max_pos * half_dim];
            let mut sin_table = vec![0.0f32; max_pos * half_dim];
            for pos in 0..max_pos {
                for i in 0..half_dim {
                    let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                    let theta = pos as f32 * freq;
                    cos_table[pos * half_dim + i] = theta.cos();
                    sin_table[pos * half_dim + i] = theta.sin();
                }
            }
            let rope_cos = stream
                .clone_htod(&cos_table)
                .map_err(|e| LLMError::GpuError(format!("rope cos HtoD: {e}")))?;
            let rope_sin = stream
                .clone_htod(&sin_table)
                .map_err(|e| LLMError::GpuError(format!("rope sin HtoD: {e}")))?;
            info!(max_pos, half_dim, "RoPE tables uploaded to GPU");

            // Create a dedicated copy stream for async HtoD metadata uploads.
            // This stream runs concurrently with compute on `stream`, letting
            // metadata transfers overlap with the previous step's GPU work.
            let copy_stream = device
                .new_stream()
                .map_err(|e| LLMError::GpuError(format!("copy_stream create: {e}")))?;
            info!("copy_stream created for async metadata uploads");

            Ok(Self {
                weights,
                cache,
                blas,
                loader,
                config,
                device,
                stream,
                copy_stream,
                layers,
                embed_tokens,
                final_norm_weight,
                lm_head_weight,
                rms_norm_eps: 1e-5_f32,
                rope_cos,
                rope_sin,
                use_fp16: false,
            })
        }

        pub fn forward(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
        ) -> Result<Vec<f32>> {
            match self.forward_ex(token_ids, positions, attn_meta, is_prefill, false)? {
                ForwardOutput::Logits(logits) => Ok(logits),
                ForwardOutput::TokenIds(_) => unreachable!("greedy_only=false must return Logits"),
            }
        }

        /// Extended forward: when `greedy_only` is true, runs argmax on GPU and
        /// returns only token IDs (num_tokens * 4 bytes DtoH instead of
        /// num_tokens * vocab_size * 4 bytes).
        pub fn forward_ex(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
            greedy_only: bool,
        ) -> Result<ForwardOutput> {
            let num_tokens = token_ids.len();
            let num_seqs = attn_meta.context_lens.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            debug!(num_tokens, num_seqs, is_prefill, greedy_only, "GpuModelRunner::forward_ex");

            // ---- Async metadata upload on copy_stream ----
            // All HtoD transfers below run on copy_stream, which executes
            // concurrently with any leftover compute from the previous step
            // still running on the main stream. The main stream will wait
            // for copy_stream before the embedding lookup that first needs
            // this data.

            // Upload positions to GPU as i32 (CUDA kernels expect int*)
            let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
            let positions_gpu: CudaSlice<i32> = self
                .copy_stream
                .clone_htod(&pos_i32)
                .map_err(|e| LLMError::GpuError(format!("positions HtoD: {e}")))?;

            // Upload context_lens as i32
            let cl_i32: Vec<i32> = attn_meta.context_lens.iter().map(|&c| c as i32).collect();
            let context_lens_gpu: CudaSlice<i32> = self
                .copy_stream
                .clone_htod(&cl_i32)
                .map_err(|e| LLMError::GpuError(format!("context_lens HtoD: {e}")))?;

            // Flatten block_tables to [num_seqs, max_blocks_per_seq] row-major as i32
            let max_blocks = attn_meta
                .block_tables
                .iter()
                .map(|r| r.len())
                .max()
                .unwrap_or(1)
                .max(1);
            let mut flat_bt = vec![0i32; num_seqs * max_blocks];
            for (s, row) in attn_meta.block_tables.iter().enumerate() {
                for (b, &blk) in row.iter().enumerate() {
                    flat_bt[s * max_blocks + b] = blk as i32;
                }
            }
            let block_tables_gpu: CudaSlice<i32> = self
                .copy_stream
                .clone_htod(&flat_bt)
                .map_err(|e| LLMError::GpuError(format!("block_tables HtoD: {e}")))?;

            // Upload slot_mapping as i32
            let sm_i32: Vec<i32> = attn_meta.slot_mapping.iter().map(|&s| s as i32).collect();
            let slot_mapping_gpu: CudaSlice<i32> = self
                .copy_stream
                .clone_htod(&sm_i32)
                .map_err(|e| LLMError::GpuError(format!("slot_mapping HtoD: {e}")))?;

            let max_context_len = attn_meta.max_context_len;

            // Build seq_start_pos from query_lens (not context_lens).
            // query_lens[i] = prompt_len for prefill, 1 for decode. Sum == num_tokens.
            let mut seq_starts_host = Vec::with_capacity(num_seqs + 1);
            let mut pos = 0i32;
            for &ql in &attn_meta.query_lens {
                seq_starts_host.push(pos);
                pos += ql as i32;
            }
            seq_starts_host.push(num_tokens as i32); // sentinel
            let seq_start_pos_gpu: CudaSlice<i32> = self
                .copy_stream
                .clone_htod(&seq_starts_host)
                .map_err(|e| LLMError::GpuError(format!("seq_start_pos HtoD: {e}")))?;

            // ---- Sync point: main stream waits for copy_stream ----
            // Ensure all metadata is resident on GPU before embedding lookup
            // and subsequent layer compute that reads these buffers.
            self.stream
                .join(&self.copy_stream)
                .map_err(|e| LLMError::GpuError(format!("stream join copy_stream: {e}")))?;

            // Step 1: token embedding lookup
            info!("gpu_runner: embedding lookup");
            let mut hidden_states = self.embedding_lookup(token_ids)?;

            // Step 2: transformer layers
            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if layer_idx == 0 || layer_idx == num_layers - 1 {
                    info!(layer = layer_idx, use_fp16 = self.use_fp16, "gpu_runner: layer start");
                }
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
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
                    is_prefill,
                    seq_start_pos: &seq_start_pos_gpu,
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                hidden_states = if self.use_fp16 {
                    let weights = self.layer_weights_f16(layer_idx)?;
                    layer.forward_f16(&input, &weights, &self.blas)?
                } else {
                    let weights = self.layer_weights(layer_idx)?;
                    layer.forward(&input, &weights, &self.blas)?
                };
                if layer_idx == 0 || layer_idx == num_layers - 1 {
                    info!(layer = layer_idx, "gpu_runner: layer done");
                }
            }

            // Step 3: final RMSNorm (all on stream 0, no sync needed)
            let normed = CudaRMSNorm::forward(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                &self.stream,
            )?;

            // Step 4: LM head  normed [num_tokens, hidden] @ lm_head^T [hidden, vocab]
            let logits_gpu = if self.use_fp16 {
                let lm_f16 = self
                    .weights
                    .get_f16("lm_head.weight")
                    .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                if let Some(lm_w) = lm_f16 {
                    CudaLinearLayer::forward_once_f16(
                        &normed, lm_w, num_tokens, vocab_size, hidden_size, &self.blas, &self.loader,
                    )?
                } else {
                    // Fallback: lm_head not available as f16, use f32
                    CudaLinearLayer::forward_once(
                        &normed, &self.lm_head_weight, None,
                        num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                }
            } else {
                CudaLinearLayer::forward_once(
                    &normed, &self.lm_head_weight, None,
                    num_tokens, vocab_size, hidden_size, &self.blas,
                )?
            };

            // Step 5: greedy fast path -- argmax on GPU, copy only token IDs
            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self
                    .stream
                    .clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax token_ids DtoH: {e}")))?;
                debug!(
                    num_tokens,
                    "forward_ex complete (greedy, {} bytes DtoH)",
                    num_tokens * 4
                );
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Step 5 (fallback): full logits DtoH for temperature>0 sampling
            let logits_cpu = self
                .stream
                .clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;

            debug!(
                logits_len = logits_cpu.len(),
                expected = num_tokens * vocab_size,
                "forward_ex complete (full logits)"
            );
            Ok(ForwardOutput::Logits(logits_cpu))
        }

        /// Launch argmax kernel on GPU, returning [num_tokens] i32 token IDs.
        fn gpu_argmax(
            &self,
            logits_gpu: &CudaSlice<f32>,
            num_tokens: usize,
            vocab_size: usize,
        ) -> Result<CudaSlice<i32>> {
            let kernel = self
                .loader
                .get_func("argmax", "argmax_kernel")?;

            let output: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(num_tokens)
                .map_err(|e| LLMError::GpuError(format!("argmax alloc: {e}")))?;

            let block_dim = vocab_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&kernel)
                    .arg(logits_gpu)
                    .arg(&output)
                    .arg(&(vocab_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("argmax_kernel launch: {e}")))?;
            }

            Ok(output)
        }

        /// Per-layer weight references into the GPU weight map.
        fn layer_weights(&self, i: usize) -> Result<GpuLayerWeights<'_>> {
            let g = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeights {
                input_layernorm: g(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                post_attention_layernorm: g(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: g(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }

        fn embedding_lookup(&self, token_ids: &[u32]) -> Result<CudaSlice<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;

            let kernel = self
                .loader
                .get_func("embedding_gather", "embedding_gather_kernel")?;

            let output = self
                .stream
                .alloc_zeros::<f32>(num_tokens * hidden_size)
                .map_err(|e| LLMError::GpuError(format!("embed alloc: {e}")))?;

            let ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
            let ids_gpu = self
                .stream
                .clone_htod(&ids_i32)
                .map_err(|e| LLMError::GpuError(format!("token_ids HtoD: {e}")))?;

            let block_dim = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&kernel)
                    .arg(&output)
                    .arg(&self.embed_tokens)
                    .arg(&ids_gpu)
                    .arg(&(hidden_size as i32))
                    .arg(&(self.config.vocab_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
            }

            Ok(output)
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

        /// Enable f16 inference mode.
        ///
        /// After calling this, `forward_ex` will use hgemm with f16 projection
        /// weights for all linear layers. The `GpuModelWeights` must already
        /// have f16 weights populated via `insert_f16` / the f16 loader.
        pub fn enable_fp16(&mut self) {
            self.use_fp16 = true;
            info!(use_fp16 = true, "GpuModelRunner: fp16 mode enabled");
        }

        pub fn use_fp16(&self) -> bool {
            self.use_fp16
        }

        /// Per-layer f16 weight references into the GPU weight map.
        fn layer_weights_f16(&self, i: usize) -> Result<GpuLayerWeightsF16<'_>> {
            let g_f16 = |name: &str| -> Result<&CudaSlice<f16>> {
                self.weights
                    .get_f16(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing f16 weight: {name}")))
            };
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeightsF16 {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g_f16(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g_f16(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g_f16(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g_f16(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                post_attention_layernorm: g_f32(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: g_f16(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g_f16(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g_f16(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
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
    use super::ForwardOutput;

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
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
        ) -> Result<Vec<f32>> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn forward_ex(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
            _greedy_only: bool,
        ) -> Result<ForwardOutput> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn enable_fp16(&mut self) {}

        pub fn use_fp16(&self) -> bool {
            false
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
                rope_theta: 10000.0,
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
                rope_theta: 10000.0,
                dtype: "float16".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            assert_eq!(runner.config().num_layers, 4);
            assert_eq!(runner.config().vocab_size, 32000);
        }
    }
}
