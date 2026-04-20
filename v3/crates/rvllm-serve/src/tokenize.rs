//! Tokenizer + chat-template wrapper.
//!
//! Skeleton for Phase 3. Real implementation loads:
//!   * `tokenizer.json` via the `tokenizers` crate
//!   * `chat_template` string from `tokenizer_config.json`, rendered
//!     through `minijinja` per request.
//!
//! Separated so integration tests can stub a fake tokenizer that
//! maps bytes → IDs one-to-one.

use std::path::Path;
use std::sync::Arc;

use crate::error::ApiError;
use crate::openai::chat::ChatMessage;

/// Shared tokenizer handle. Cloneable (`Arc` inside).
#[derive(Clone)]
pub struct TokenizerHandle {
    inner: Arc<TokenizerInner>,
}

#[allow(dead_code)]
struct TokenizerInner {
    tokenizer: tokenizers::Tokenizer,
    /// Optional chat_template string; `None` means fall back to a
    /// built-in Gemma 4 template.
    chat_template: Option<String>,
    /// EOS / stop token ids harvested from `tokenizer_config.json`.
    eos_token_ids: Vec<u32>,
    /// BOS token id, prepended to prompts when the chat template
    /// doesn't emit one.
    bos_token_id: Option<u32>,
}

impl TokenizerHandle {
    /// Load `tokenizer.json` (and optionally `tokenizer_config.json`)
    /// from a HF model directory.
    ///
    /// Phase 3 will flesh out chat-template + special-token handling.
    /// For now we just load the tokenizer.json so callers can compile.
    pub fn load(model_dir: &Path) -> Result<Self, ApiError> {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| {
            ApiError::Internal(format!(
                "tokenizer.json load from {}: {e}",
                tok_path.display()
            ))
        })?;
        Ok(Self {
            inner: Arc::new(TokenizerInner {
                tokenizer,
                chat_template: None,
                eos_token_ids: Vec::new(),
                bos_token_id: None,
            }),
        })
    }

    /// Encode raw text into token IDs. Skips special tokens by default
    /// (the chat template inserts them where appropriate).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ApiError> {
        let enc = self
            .inner
            .tokenizer
            .encode(text, false)
            .map_err(|e| ApiError::Tokenize(format!("{e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, ApiError> {
        self.inner
            .tokenizer
            .decode(ids, /*skip_special_tokens*/ true)
            .map_err(|e| ApiError::Tokenize(format!("{e}")))
    }

    /// Render a chat-template around `messages` and return prompt IDs.
    /// Phase 3 implements the minijinja path; for now this is a stub.
    pub fn render_chat(&self, _messages: &[ChatMessage]) -> Result<Vec<u32>, ApiError> {
        Err(ApiError::Internal(
            "chat template rendering not yet implemented — phase 3".into(),
        ))
    }

    /// Token IDs that should terminate generation.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.inner.eos_token_ids
    }
}
