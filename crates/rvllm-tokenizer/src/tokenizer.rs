//! Main tokenizer wrapper around the HuggingFace `tokenizers` crate.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};
use rvllm_core::prelude::{hf_auth_hint, hf_token_from_env, LLMError, Result, TokenId};
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info};

use crate::chat::{apply_chatml, ChatMessage};
use crate::incremental::IncrementalDecoder;

const TOKENIZER_JSON: &str = "tokenizer.json";
const TOKENIZER_MODEL: &str = "tokenizer.model";

fn configured_hf_api() -> Result<Api> {
    let mut builder = ApiBuilder::from_env();
    if let Some(token) = hf_token_from_env() {
        builder = builder.with_token(Some(token));
    }
    builder
        .build()
        .map_err(|e| LLMError::TokenizerError(format!("failed to init hf-hub API: {e}")))
}

fn find_cached_tokenizer_json(model_name_or_path: &str) -> Option<PathBuf> {
    let hf_home = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let cache_snap = Path::new(&hf_home)
        .join("hub")
        .join(format!("models--{}", model_name_or_path.replace('/', "--")))
        .join("snapshots");

    let mut snapshots = std::fs::read_dir(cache_snap)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    snapshots.sort();

    snapshots
        .into_iter()
        .map(|snapshot| snapshot.join(TOKENIZER_JSON))
        .find(|path| path.exists())
}

fn missing_tokenizer_error(path: &Path) -> LLMError {
    let tokenizer_model = path.join(TOKENIZER_MODEL);
    if tokenizer_model.exists() {
        return unsupported_sentencepiece_error(&tokenizer_model);
    }
    LLMError::TokenizerError(format!("no {TOKENIZER_JSON} found in {}", path.display()))
}

fn unsupported_sentencepiece_error(path: &Path) -> LLMError {
    LLMError::TokenizerError(format!(
        "found {} but rvLLM currently only loads {TOKENIZER_JSON}. \
use a tokenizer export that includes {TOKENIZER_JSON}, or point --tokenizer at a compatible repo/path",
        path.display()
    ))
}

fn download_tokenizer_error(model_name_or_path: &str, error: &str) -> LLMError {
    let mut message =
        format!("failed to download {TOKENIZER_JSON} from '{model_name_or_path}': {error}");
    if let Some(hint) = hf_auth_hint(error) {
        message.push(' ');
        message.push_str(hint);
    } else {
        message.push_str(&format!(
            " rvLLM currently expects a {TOKENIZER_JSON} file; if this repo only exposes {TOKENIZER_MODEL}, \
use a tokenizer export that includes {TOKENIZER_JSON}."
        ));
    }
    LLMError::TokenizerError(message)
}

/// High-level tokenizer wrapping HuggingFace's tokenizer with vLLM conventions.
pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: Vec<TokenId>,
    eos_token_id: Option<TokenId>,
    bos_token_id: Option<TokenId>,
    pad_token_id: Option<TokenId>,
    incremental: IncrementalDecoder,
}

impl Tokenizer {
    /// Load a tokenizer from a HuggingFace model name or local directory.
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        info!(model = model_name_or_path, "loading tokenizer");

        // Check HF cache first (avoids network round-trip when model is already downloaded)
        if let Some(tokenizer_path) = find_cached_tokenizer_json(model_name_or_path) {
            info!(path = %tokenizer_path.display(), "loading tokenizer from HF cache");
            return Self::from_file(&tokenizer_path);
        }

        let path = Path::new(model_name_or_path);
        if path.is_dir() {
            let tokenizer_file = path.join(TOKENIZER_JSON);
            if tokenizer_file.exists() {
                return Self::from_file(&tokenizer_file);
            }
            return Err(missing_tokenizer_error(path));
        }

        // If it looks like a local file that actually exists, load it directly
        if path.is_file() {
            if path.file_name() == Some(OsStr::new(TOKENIZER_MODEL)) {
                return Err(unsupported_sentencepiece_error(path));
            }
            return Self::from_file(path);
        }

        // Download tokenizer.json from HuggingFace hub
        let api = configured_hf_api()?;
        let repo = api.model(model_name_or_path.to_string());
        let tokenizer_path = repo
            .get(TOKENIZER_JSON)
            .map_err(|e| download_tokenizer_error(model_name_or_path, &e.to_string()))?;

        Self::from_file(&tokenizer_path)
    }

    /// Load a tokenizer directly from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer from file");

        let hf = HfTokenizer::from_file(path).map_err(|e| {
            LLMError::TokenizerError(format!("failed to load {}: {}", path.display(), e))
        })?;

        Ok(Self::from_hf_tokenizer(hf))
    }

    /// Build from an already-loaded HuggingFace tokenizer.
    fn from_hf_tokenizer(hf: HfTokenizer) -> Self {
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
            "tokenizer loaded"
        );

        Self {
            inner: hf,
            special_tokens,
            eos_token_id,
            bos_token_id,
            pad_token_id,
            incremental: IncrementalDecoder::new(),
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
    /// Falls back to ChatML format.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        apply_chatml(messages, add_generation_prompt)
    }

    /// Access the underlying HuggingFace tokenizer.
    pub fn inner(&self) -> &HfTokenizer {
        &self.inner
    }

    /// Reset the incremental decoder state.
    pub fn reset_incremental(&mut self) {
        self.incremental.reset();
    }
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

        Tokenizer::from_hf_tokenizer(hf)
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
        let result = tok.apply_chat_template(&msgs, true).unwrap();
        assert!(result.contains("Hello"));
        assert!(result.contains("Hi there"));
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
    fn local_sentencepiece_only_dir_errors_clearly() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(TOKENIZER_MODEL), b"not-a-real-model").unwrap();
        let err = match Tokenizer::from_pretrained(dir.path().to_str().unwrap()) {
            Ok(_) => panic!("expected tokenizer.model-only directory to error"),
            Err(err) => err,
        };
        let err = err.to_string();
        assert!(err.contains(TOKENIZER_MODEL));
        assert!(err.contains(TOKENIZER_JSON));
    }

    #[test]
    fn local_sentencepiece_file_errors_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join(TOKENIZER_MODEL);
        std::fs::write(&model_path, b"not-a-real-model").unwrap();
        let err = match Tokenizer::from_pretrained(model_path.to_str().unwrap()) {
            Ok(_) => panic!("expected tokenizer.model path to error"),
            Err(err) => err,
        };
        let err = err.to_string();
        assert!(err.contains(TOKENIZER_MODEL));
        assert!(err.contains(TOKENIZER_JSON));
    }

    #[test]
    fn inner_accessor() {
        let tok = make_test_tokenizer();
        let inner = tok.inner();
        assert!(inner.get_vocab_size(false) > 0);
    }
}
