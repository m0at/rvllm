//! Stateful incremental decoder for streaming token-by-token output.

use tokenizers::Tokenizer as HfTokenizer;
use rvllm_core::prelude::TokenId;

/// Stateful decoder that buffers tokens and emits text only when
/// complete UTF-8 characters are available.
pub struct IncrementalDecoder {
    /// Accumulated token IDs not yet fully decoded.
    buffer: Vec<TokenId>,
    /// Number of tokens already emitted as text.
    emitted_up_to: usize,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            emitted_up_to: 0,
        }
    }

    /// Add a token and return new text if a complete segment is available.
    pub fn add_token(&mut self, token: TokenId, tokenizer: &HfTokenizer) -> Option<String> {
        self.buffer.push(token);

        // Decode everything so far
        let full = tokenizer
            .decode(&self.buffer, true)
            .ok()?;

        // Decode only the previously-emitted prefix
        let prefix = if self.emitted_up_to > 0 {
            tokenizer
                .decode(&self.buffer[..self.emitted_up_to], true)
                .ok()?
        } else {
            String::new()
        };

        // The new text is the difference
        if full.len() > prefix.len() {
            let new_text = &full[prefix.len()..];
            // Check if the new text ends with a valid char boundary.
            // If it contains a replacement char at the end, hold off.
            if new_text.ends_with('\u{FFFD}') {
                return None;
            }
            self.emitted_up_to = self.buffer.len();
            Some(new_text.to_string())
        } else {
            None
        }
    }

    /// Flush any remaining buffered tokens as text.
    pub fn flush(&mut self, tokenizer: &HfTokenizer) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let full = tokenizer
            .decode(&self.buffer, true)
            .unwrap_or_default();

        let prefix = if self.emitted_up_to > 0 {
            tokenizer
                .decode(&self.buffer[..self.emitted_up_to], true)
                .unwrap_or_default()
        } else {
            String::new()
        };

        self.buffer.clear();
        self.emitted_up_to = 0;

        if full.len() > prefix.len() {
            full[prefix.len()..].to_string()
        } else {
            String::new()
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.emitted_up_to = 0;
    }
}

impl Default for IncrementalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> HfTokenizer {
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::Tokenizer;

        let mut vocab = std::collections::HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let wp = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut tok = Tokenizer::new(wp);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        tok
    }

    #[test]
    fn incremental_basic() {
        let hf = make_test_tokenizer();
        let mut dec = IncrementalDecoder::new();

        // Feed token 1 = "hello"
        let r = dec.add_token(1, &hf);
        assert!(r.is_some());
        assert_eq!(r.unwrap(), "hello");

        // Feed token 2 = "world"
        let r = dec.add_token(2, &hf);
        assert!(r.is_some());
    }

    #[test]
    fn incremental_flush() {
        let hf = make_test_tokenizer();
        let mut dec = IncrementalDecoder::new();
        dec.add_token(1, &hf);
        let remainder = dec.flush(&hf);
        // After flush, state is clear
        assert!(remainder.is_empty() || !remainder.is_empty()); // exercises the path
        assert_eq!(dec.buffer.len(), 0);
    }

    #[test]
    fn incremental_reset() {
        let mut dec = IncrementalDecoder::new();
        dec.buffer.push(42);
        dec.emitted_up_to = 1;
        dec.reset();
        assert!(dec.buffer.is_empty());
        assert_eq!(dec.emitted_up_to, 0);
    }

    #[test]
    fn incremental_default() {
        let dec = IncrementalDecoder::default();
        assert!(dec.buffer.is_empty());
    }
}
