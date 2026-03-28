//! Worker configuration.

/// Configuration for a single GPU worker instance.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// GPU device ordinal this worker owns.
    pub device_id: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_model_len: usize,
    /// Tokens per KV cache block.
    pub block_size: usize,
    /// Fraction of GPU memory to use for KV cache.
    pub gpu_memory_utilization: f32,
    /// Tensor parallel rank of this worker.
    pub rank: usize,
    /// Tensor parallel world size.
    pub tensor_parallel_size: usize,
    /// Pipeline parallel world size.
    pub pipeline_parallel_size: usize,
    /// Model architecture name (e.g. "llama").
    pub architecture: String,
    /// Data type string (e.g. "float16").
    pub dtype: String,
    /// RoPE theta parameter.
    pub rope_theta: f32,
    /// Fraction of head_dim that gets RoPE (Phi: 0.5, others: 1.0).
    pub partial_rotary_factor: f32,
    /// Soft-capping value for attention logits (Gemma 2). 0.0 = disabled.
    pub attn_logit_softcapping: f32,
    /// Number of MoE experts (Mixtral: 8, DeepSeek: 64). 0 = dense.
    pub num_local_experts: usize,
    /// Number of experts activated per token (Mixtral: 2, DeepSeek: 6).
    pub num_experts_per_tok: usize,
    /// KV cache data type: "auto", "fp8", "fp8_e4m3".
    pub kv_cache_dtype: String,
    /// Enable prefix caching.
    pub enable_prefix_caching: bool,
}

impl WorkerConfig {
    /// Build a `ModelRunnerConfig` from this worker config.
    pub fn model_runner_config(&self) -> rvllm_model_runner::ModelRunnerConfig {
        rvllm_model_runner::ModelRunnerConfig {
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position: self.max_model_len,
            dtype: self.dtype.clone(),
            architecture: self.architecture.clone(),
            rope_theta: self.rope_theta,
            rms_norm_eps: 1e-6, // TODO: read from model config.json
        }
    }

    /// Build a `CacheConfig` from this worker config.
    pub fn cache_config(&self) -> rvllm_kv_cache::CacheConfig {
        rvllm_kv_cache::CacheConfig::new(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )
    }
}
