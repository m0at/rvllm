//! ModelRunner: orchestrates the forward pass.

use std::sync::Arc;

use rvllm_core::types::Dtype;
use tracing::debug;

use crate::architectures::{create_model, Architecture};
use crate::bridge::{
    AttentionBackend, CacheEngine, GpuAllocator, GpuBuffer, LLMError, ModelWeights, Result,
};
use crate::input::ModelInput;

/// Static configuration for the model runner, derived from the model config.
#[derive(Debug, Clone)]
pub struct ModelRunnerConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position: usize,
    pub rope_theta: f32,
    pub dtype: Dtype,
    pub architecture: String,
    /// Per-layer type for hybrid models: "full_attention" or "linear_attention".
    /// Empty for standard transformer models.
    pub layer_types: Vec<String>,
    /// Fraction of head_dim that gets RoPE (1.0 for standard, 0.25 for Qwen3.5).
    pub partial_rotary_factor: f32,
    /// Qwen3.5: q_proj produces 2x head_dim (query + output gate).
    pub has_attn_output_gate: bool,
    /// RMSNorm epsilon (Qwen3.5 uses 1e-6, Llama uses 1e-5).
    pub rms_norm_eps: f32,
}

/// Drives the transformer forward pass: embed -> layers -> LM head -> logits.
pub struct ModelRunner {
    pub config: ModelRunnerConfig,
    model: Box<dyn Architecture>,
    attention: Box<dyn AttentionBackend>,
    cache: Arc<CacheEngine>,
    #[allow(dead_code)]
    gpu: Arc<dyn GpuAllocator>,
}

impl ModelRunner {
    pub fn new(
        weights: ModelWeights,
        config: ModelRunnerConfig,
        attention: Box<dyn AttentionBackend>,
        cache: Arc<CacheEngine>,
        gpu: Arc<dyn GpuAllocator>,
    ) -> Result<Self> {
        debug!(arch = %config.architecture, "creating model runner");
        let model = create_model(&config.architecture, weights, &config)?;
        Ok(Self {
            config,
            model,
            attention,
            cache,
            gpu,
        })
    }

    /// Execute a single forward pass, returning logits [batch, vocab].
    pub fn execute_model(&self, input: ModelInput) -> Result<GpuBuffer<f32>> {
        debug!(
            num_tokens = input.num_tokens(),
            is_prefill = input.is_prefill,
            "execute_model"
        );

        if input.token_ids.is_empty() {
            return Err(LLMError::ModelError("empty input".into()));
        }

        self.model
            .forward(&input, &self.cache, self.attention.as_ref())
    }
}
