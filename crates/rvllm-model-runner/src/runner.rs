//! ModelRunner: orchestrates the forward pass.

use std::sync::Arc;

use tracing::debug;

use crate::bridge::{
    AttentionBackend, CacheEngine, GpuAllocator, GpuBuffer, LLMError, ModelWeights, Result,
};
use crate::input::ModelInput;
use crate::architectures::{create_model, Architecture};

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
    pub rms_norm_eps: f32,
    pub dtype: String,
    pub architecture: String,
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
