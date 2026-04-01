//! Model configuration.

use rvllm_core::types::Dtype;
use serde::{Deserialize, Serialize};

/// Conservative default context length used when the caller does not provide one.
pub const DEFAULT_MAX_MODEL_LEN: usize = 2048;
/// Safe auto-expansion cap for Llama-family models.
pub const AUTO_LLAMA_MAX_MODEL_LEN_CAP: usize = 8192;

/// Configuration for the model itself: paths, dtype, length limits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfigImpl {
    /// Path to model weights (HuggingFace repo id or local path).
    pub model_path: String,
    /// Optional override for tokenizer path (defaults to model_path).
    pub tokenizer_path: Option<String>,
    /// Data type for model weights and compute.
    pub dtype: Dtype,
    /// Maximum sequence length the model supports.
    pub max_model_len: usize,
    /// Whether `max_model_len` came from an explicit user override.
    #[serde(skip, default)]
    pub max_model_len_explicit: bool,
    /// Whether to trust remote code when loading the model.
    pub trust_remote_code: bool,
}

impl Default for ModelConfigImpl {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            dtype: Dtype::Auto,
            max_model_len: DEFAULT_MAX_MODEL_LEN,
            max_model_len_explicit: false,
            trust_remote_code: false,
        }
    }
}

impl ModelConfigImpl {
    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::default()
    }
}

/// Builder for [`ModelConfigImpl`].
#[derive(Debug, Default)]
pub struct ModelConfigBuilder(ModelConfigImpl);

impl ModelConfigBuilder {
    /// Set the model path.
    pub fn model_path(mut self, v: impl Into<String>) -> Self {
        self.0.model_path = v.into();
        self
    }

    /// Set the tokenizer path.
    pub fn tokenizer_path(mut self, v: impl Into<String>) -> Self {
        self.0.tokenizer_path = Some(v.into());
        self
    }

    /// Set the dtype.
    pub fn dtype(mut self, v: Dtype) -> Self {
        self.0.dtype = v;
        self
    }

    /// Set the max model length.
    pub fn max_model_len(mut self, v: usize) -> Self {
        self.0.max_model_len = v;
        self.0.max_model_len_explicit = true;
        self
    }

    /// Set trust remote code.
    pub fn trust_remote_code(mut self, v: bool) -> Self {
        self.0.trust_remote_code = v;
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> ModelConfigImpl {
        self.0
    }
}

/// Resolve an effective max model length for the runtime.
///
/// By default rvLLM uses a conservative 2048-token limit. For Llama-family
/// checkpoints we can safely expand to the declared HF context length when it
/// stays within the currently-supported 8K window.
pub fn resolve_runtime_max_model_len(
    requested: usize,
    requested_was_explicit: bool,
    architecture: &str,
    declared: Option<usize>,
) -> usize {
    if requested_was_explicit
        || requested != DEFAULT_MAX_MODEL_LEN
        || architecture != "LlamaForCausalLM"
    {
        return requested;
    }

    match declared {
        Some(len) if (DEFAULT_MAX_MODEL_LEN..=AUTO_LLAMA_MAX_MODEL_LEN_CAP).contains(&len) => len,
        _ => requested,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_max_model_len_is_conservative() {
        assert_eq!(
            ModelConfigImpl::default().max_model_len,
            DEFAULT_MAX_MODEL_LEN
        );
    }

    #[test]
    fn llama_auto_expands_to_declared_len() {
        assert_eq!(
            resolve_runtime_max_model_len(
                DEFAULT_MAX_MODEL_LEN,
                false,
                "LlamaForCausalLM",
                Some(4096),
            ),
            4096
        );
        assert_eq!(
            resolve_runtime_max_model_len(
                DEFAULT_MAX_MODEL_LEN,
                false,
                "LlamaForCausalLM",
                Some(8192),
            ),
            8192
        );
    }

    #[test]
    fn llama_auto_does_not_shrink_or_overexpand() {
        assert_eq!(
            resolve_runtime_max_model_len(
                DEFAULT_MAX_MODEL_LEN,
                false,
                "LlamaForCausalLM",
                Some(2048),
            ),
            DEFAULT_MAX_MODEL_LEN
        );
        assert_eq!(
            resolve_runtime_max_model_len(
                DEFAULT_MAX_MODEL_LEN,
                false,
                "LlamaForCausalLM",
                Some(131072),
            ),
            DEFAULT_MAX_MODEL_LEN
        );
    }

    #[test]
    fn explicit_override_wins() {
        assert_eq!(
            resolve_runtime_max_model_len(1024, true, "LlamaForCausalLM", Some(4096)),
            1024
        );
        assert_eq!(
            resolve_runtime_max_model_len(
                DEFAULT_MAX_MODEL_LEN,
                true,
                "LlamaForCausalLM",
                Some(4096),
            ),
            DEFAULT_MAX_MODEL_LEN
        );
    }

    #[test]
    fn non_llama_models_keep_requested_len() {
        assert_eq!(
            resolve_runtime_max_model_len(
                DEFAULT_MAX_MODEL_LEN,
                false,
                "Qwen2ForCausalLM",
                Some(32768),
            ),
            DEFAULT_MAX_MODEL_LEN
        );
    }
}
