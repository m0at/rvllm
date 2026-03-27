//! Model configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the model itself: paths, dtype, length limits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfigImpl {
    /// Path to model weights (HuggingFace repo id or local path).
    pub model_path: String,
    /// Optional override for tokenizer path (defaults to model_path).
    pub tokenizer_path: Option<String>,
    /// Data type for model weights (e.g. "float16", "bfloat16", "auto").
    pub dtype: String,
    /// Maximum sequence length the model supports.
    pub max_model_len: usize,
    /// Whether to trust remote code when loading the model.
    pub trust_remote_code: bool,
}

impl Default for ModelConfigImpl {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            dtype: "auto".into(),
            max_model_len: 2048,
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
    pub fn dtype(mut self, v: impl Into<String>) -> Self {
        self.0.dtype = v.into();
        self
    }

    /// Set the max model length.
    pub fn max_model_len(mut self, v: usize) -> Self {
        self.0.max_model_len = v;
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
