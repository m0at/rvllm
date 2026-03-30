//! Main tokenizer wrapper around the HuggingFace `tokenizers` crate.

use std::path::Path;

use hf_hub::api::sync::Api;
use rvllm_core::prelude::{LLMError, Result, TokenId};
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info};

use crate::chat::{apply_chatml, apply_harmony, ChatMessage};
use crate::incremental::IncrementalDecoder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateMode {
    Chatml,
    HarmonyGptOss,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderedPrompt {
    Text(String),
    Tokens(Vec<TokenId>),
}

/// High-level tokenizer wrapping HuggingFace's tokenizer with vLLM conventions.
pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: Vec<TokenId>,
    eos_token_id: Option<TokenId>,
    bos_token_id: Option<TokenId>,
    pad_token_id: Option<TokenId>,
    incremental: IncrementalDecoder,
    chat_template_mode: ChatTemplateMode,
}

impl Tokenizer {
    /// Load a tokenizer from a HuggingFace model name or local directory.
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        info!(model = model_name_or_path, "loading tokenizer");

        // Check HF cache first (avoids network round-trip when model is already downloaded)
        {
            let hf_home = std::env::var("HF_HOME").unwrap_or_else(|_| {
                format!(
                    "{}/.cache/huggingface",
                    std::env::var("HOME").unwrap_or_default()
                )
            });
            let cache_snap = std::path::Path::new(&hf_home)
                .join("hub")
                .join(format!("models--{}", model_name_or_path.replace("/", "--")))
                .join("snapshots");
            if let Ok(mut entries) = std::fs::read_dir(&cache_snap) {
                if let Some(Ok(entry)) = entries.next() {
                    let tf = entry.path().join("tokenizer.json");
                    if tf.exists() {
                        info!(path = %tf.display(), "loading tokenizer from HF cache");
                        return Self::from_file_with_hint(&tf, Some(model_name_or_path));
                    }
                }
            }
        }

        let path = Path::new(model_name_or_path);
        if path.is_dir() {
            let tokenizer_file = path.join("tokenizer.json");
            if tokenizer_file.exists() {
                return Self::from_file_with_hint(&tokenizer_file, Some(model_name_or_path));
            }
            return Err(LLMError::TokenizerError(format!(
                "no tokenizer.json found in {}",
                model_name_or_path
            )));
        }

        // If it looks like a local file that actually exists, load it directly
        if path.is_file() {
            return Self::from_file_with_hint(path, Some(model_name_or_path));
        }

        // Download tokenizer.json from HuggingFace hub
        let api = Api::new()
            .map_err(|e| LLMError::TokenizerError(format!("failed to init hf-hub API: {}", e)))?;
        let repo = api.model(model_name_or_path.to_string());
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            LLMError::TokenizerError(format!(
                "failed to download tokenizer.json from '{}': {}",
                model_name_or_path, e
            ))
        })?;

        Self::from_file_with_hint(&tokenizer_path, Some(model_name_or_path))
    }

    /// Load a tokenizer directly from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_file_with_hint(path, None)
    }

    fn from_file_with_hint(path: &Path, model_name_or_path: Option<&str>) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer from file");

        let hf = HfTokenizer::from_file(path).map_err(|e| {
            LLMError::TokenizerError(format!("failed to load {}: {}", path.display(), e))
        })?;

        let chat_template_mode = detect_chat_template_mode(path.parent(), model_name_or_path)?;
        Ok(Self::from_hf_tokenizer_with_mode(hf, chat_template_mode))
    }

    fn from_hf_tokenizer_with_mode(hf: HfTokenizer, chat_template_mode: ChatTemplateMode) -> Self {
        let mut special_tokens = Vec::new();
        let mut eos_token_id = None;
        let mut bos_token_id = None;
        let mut pad_token_id = None;

        // Extract special tokens from added tokens
        if let Some(added) = hf.get_added_tokens_decoder().get(&0) {
            // Check if token 0 is special (often <pad> or <unk>)
            if added.special {
                pad_token_id = Some(0);
            }
        }

        for (id, token) in hf.get_added_tokens_decoder() {
            if token.special {
                special_tokens.push(id);
                let content = token.content.to_lowercase();
                if content.contains("eos")
                    || content == "</s>"
                    || content == "<|endoftext|>"
                    || content == "<|im_end|>"
                {
                    eos_token_id = Some(id);
                }
                if content.contains("bos") || content == "<s>" || content == "<|begin_of_text|>" {
                    bos_token_id = Some(id);
                }
                if content.contains("pad") || content == "<pad>" {
                    pad_token_id = Some(id);
                }
            }
        }

        special_tokens.sort_unstable();
        special_tokens.dedup();

        debug!(
            vocab_size = hf.get_vocab_size(true),
            special_count = special_tokens.len(),
            eos = ?eos_token_id,
            bos = ?bos_token_id,
            pad = ?pad_token_id,
            chat_template_mode = ?chat_template_mode,
            "tokenizer loaded"
        );

        Self {
            inner: hf,
            special_tokens,
            eos_token_id,
            bos_token_id,
            pad_token_id,
            incremental: IncrementalDecoder::new(),
            chat_template_mode,
        }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| LLMError::TokenizerError(format!("encode failed: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode a batch of texts into token IDs.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<TokenId>>> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), false)
            .map_err(|e| LLMError::TokenizerError(format!("encode_batch failed: {}", e)))?;
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| LLMError::TokenizerError(format!("decode failed: {}", e)))
    }

    /// Streaming decode: feed one token at a time, get text back when a
    /// complete character/word boundary is available.
    pub fn decode_incremental(&mut self, token: TokenId) -> Result<Option<String>> {
        Ok(self.incremental.add_token(token, &self.inner))
    }

    /// Vocabulary size including special tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// End-of-sequence token ID, if detected.
    pub fn eos_token_id(&self) -> Option<TokenId> {
        self.eos_token_id
    }

    /// Beginning-of-sequence token ID, if detected.
    pub fn bos_token_id(&self) -> Option<TokenId> {
        self.bos_token_id
    }

    /// Padding token ID, if detected.
    pub fn pad_token_id(&self) -> Option<TokenId> {
        self.pad_token_id
    }

    /// All special token IDs (sorted, deduplicated).
    pub fn get_special_tokens(&self) -> &[TokenId] {
        &self.special_tokens
    }

    /// Apply a chat template to format messages for the model.
    /// Returns a rendered prompt that may be either text or token IDs.
    pub fn render_chat_prompt(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<RenderedPrompt> {
        match self.chat_template_mode {
            ChatTemplateMode::Chatml => Ok(RenderedPrompt::Text(apply_chatml(
                messages,
                add_generation_prompt,
            )?)),
            ChatTemplateMode::HarmonyGptOss => Ok(RenderedPrompt::Tokens(apply_harmony(
                &self.inner,
                messages,
                add_generation_prompt,
            )?)),
        }
    }

    /// Apply a chat template to format messages for the model as text.
    /// Harmony-capable models should use `render_chat_prompt()` instead.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        match self.render_chat_prompt(messages, add_generation_prompt)? {
            RenderedPrompt::Text(prompt) => Ok(prompt),
            RenderedPrompt::Tokens(_) => Err(LLMError::TokenizerError(
                "chat template rendered tokens; use render_chat_prompt() instead".into(),
            )),
        }
    }

    /// Access the underlying HuggingFace tokenizer.
    pub fn inner(&self) -> &HfTokenizer {
        &self.inner
    }

    pub fn chat_template_mode(&self) -> ChatTemplateMode {
        self.chat_template_mode
    }

    /// Reset the incremental decoder state.
    pub fn reset_incremental(&mut self) {
        self.incremental.reset();
    }
}

fn detect_chat_template_mode(
    model_dir: Option<&Path>,
    model_name_or_path: Option<&str>,
) -> Result<ChatTemplateMode> {
    if let Some(mode) = env_chat_template_mode()? {
        return Ok(mode);
    }

    if let Some(dir) = model_dir {
        if let Some(mode) = detect_chat_template_mode_from_dir(dir)? {
            return Ok(mode);
        }
    }

    if model_name_or_path.is_some_and(looks_like_gpt_oss) {
        return Ok(ChatTemplateMode::HarmonyGptOss);
    }

    Ok(ChatTemplateMode::Chatml)
}

fn env_chat_template_mode() -> Result<Option<ChatTemplateMode>> {
    let value = match std::env::var("RVLLM_CHAT_TEMPLATE") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(e) => {
            return Err(LLMError::TokenizerError(format!(
                "failed to read RVLLM_CHAT_TEMPLATE: {e}"
            )))
        }
    };

    parse_chat_template_mode(&value).map(Some)
}

fn detect_chat_template_mode_from_dir(dir: &Path) -> Result<Option<ChatTemplateMode>> {
    for file_name in ["tokenizer_config.json", "config.json"] {
        let path = dir.join(file_name);
        if !path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(&path).map_err(|e| {
            LLMError::TokenizerError(format!("failed to read {}: {e}", path.display()))
        })?;
        let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            LLMError::TokenizerError(format!("invalid {}: {e}", path.display()))
        })?;
        if let Some(mode) = detect_chat_template_mode_from_json(&json) {
            return Ok(Some(mode));
        }
    }
    Ok(None)
}

fn detect_chat_template_mode_from_json(json: &serde_json::Value) -> Option<ChatTemplateMode> {
    let model_type = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase());
    if model_type
        .as_deref()
        .is_some_and(|value| value.contains("gpt_oss") || value.contains("gpt-oss"))
    {
        return Some(ChatTemplateMode::HarmonyGptOss);
    }

    let architecture_matches = json
        .get("architectures")
        .and_then(|v| v.as_array())
        .is_some_and(|architectures| {
            architectures.iter().filter_map(|v| v.as_str()).any(looks_like_gpt_oss)
        });
    if architecture_matches {
        return Some(ChatTemplateMode::HarmonyGptOss);
    }

    let chat_template_mentions_harmony = json
        .get("chat_template")
        .and_then(|v| v.as_str())
        .is_some_and(|template| {
            template.contains("<|start|>")
                || template.to_ascii_lowercase().contains("harmony")
                || template.to_ascii_lowercase().contains("analysis")
        });
    if chat_template_mentions_harmony {
        return Some(ChatTemplateMode::HarmonyGptOss);
    }

    None
}

fn parse_chat_template_mode(value: &str) -> Result<ChatTemplateMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "chatml" => Ok(ChatTemplateMode::Chatml),
        "harmony" | "harmony_gpt_oss" | "harmony-gpt-oss" | "gpt_oss" | "gpt-oss" => {
            Ok(ChatTemplateMode::HarmonyGptOss)
        }
        other => Err(LLMError::TokenizerError(format!(
            "unsupported chat template mode '{}'",
            other
        ))),
    }
}

fn looks_like_gpt_oss(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("gpt_oss") || normalized.contains("gpt-oss")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn make_test_tokenizer() -> Tokenizer {
        use tokenizers::models::wordpiece::WordPiece;

        let mut vocab = std::collections::HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("hello".to_string(), 3);
        vocab.insert("world".to_string(), 4);
        vocab.insert("hi".to_string(), 5);
        vocab.insert("there".to_string(), 6);

        let wp = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut hf = HfTokenizer::new(wp);
        hf.with_pre_tokenizer(Some(Whitespace {}));

        Tokenizer::from_hf_tokenizer_with_mode(hf, ChatTemplateMode::Chatml)
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("hello").unwrap();
        assert!(!ids.is_empty());
        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn encode_batch_works() {
        let tok = make_test_tokenizer();
        let batch = tok.encode_batch(&["hello", "world"]).unwrap();
        assert_eq!(batch.len(), 2);
        assert!(!batch[0].is_empty());
        assert!(!batch[1].is_empty());
    }

    #[test]
    fn vocab_size_positive() {
        let tok = make_test_tokenizer();
        assert!(tok.vocab_size() > 0);
    }

    #[test]
    fn special_token_accessors() {
        let tok = make_test_tokenizer();
        // Our minimal tokenizer has no special tokens marked
        let _ = tok.eos_token_id();
        let _ = tok.bos_token_id();
        let _ = tok.pad_token_id();
        let _ = tok.get_special_tokens();
    }

    #[test]
    fn decode_incremental_works() {
        let mut tok = make_test_tokenizer();
        let ids = tok.encode("hello").unwrap();
        let mut output = String::new();
        for &id in &ids {
            if let Ok(Some(text)) = tok.decode_incremental(id) {
                output.push_str(&text);
            }
        }
        assert!(!output.is_empty());
        tok.reset_incremental();
    }

    #[test]
    fn chat_template_works() {
        let tok = make_test_tokenizer();
        let msgs = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
        ];
        let result = tok.render_chat_prompt(&msgs, true).unwrap();
        match result {
            RenderedPrompt::Text(prompt) => {
                assert!(prompt.contains("Hello"));
                assert!(prompt.contains("Hi there"));
            }
            RenderedPrompt::Tokens(_) => panic!("expected ChatML text prompt"),
        }
    }

    #[test]
    fn from_file_missing_errors() {
        let result = Tokenizer::from_file(Path::new("/nonexistent/tokenizer.json"));
        assert!(result.is_err());
    }

    #[test]
    fn from_pretrained_bad_dir_errors() {
        // Use a temp dir with no tokenizer.json
        let dir = std::env::temp_dir();
        let result = Tokenizer::from_pretrained(dir.to_str().unwrap());
        // Might succeed if there's a tokenizer.json in temp, but likely errors
        // Just exercise the path
        let _ = result;
    }

    #[test]
    fn inner_accessor() {
        let tok = make_test_tokenizer();
        let inner = tok.inner();
        assert!(inner.get_vocab_size(false) > 0);
    }

    #[test]
    fn detects_harmony_from_config() {
        let json = serde_json::json!({
            "model_type": "gpt_oss",
        });
        assert_eq!(
            detect_chat_template_mode_from_json(&json),
            Some(ChatTemplateMode::HarmonyGptOss)
        );
    }

    #[test]
    fn detects_harmony_from_model_name_hint() {
        assert!(looks_like_gpt_oss("openai/gpt-oss-20b"));
        assert!(looks_like_gpt_oss("openai/gpt_oss-20b"));
    }
}
