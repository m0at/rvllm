//! Draft model runner for generating speculative tokens.
//!
//! Provides the `DraftModel` trait for pluggable draft model implementations,
//! and `DraftModelRunner` as the default (placeholder) implementation.

use rvllm_core::prelude::{LLMError, Result, TokenId};

use crate::config::SpeculativeConfig;

/// A single draft token with its probability distribution.
#[derive(Debug, Clone)]
pub struct DraftToken {
    /// The token id selected by the draft model.
    pub token_id: TokenId,
    /// Log-probability of the selected token.
    pub logprob: f32,
    /// Full probability distribution from the draft model over the vocabulary.
    pub draft_probs: Vec<f32>,
}

/// Trait for draft model implementations.
///
/// The draft model generates K candidate tokens autoregressively. Each token
/// must include its full probability distribution so the verifier can perform
/// speculative sampling (the accept/reject step requires comparing p_draft
/// vs p_target at each position).
pub trait DraftModel: Send {
    /// Generate `num_tokens` draft tokens starting from `input_tokens`.
    ///
    /// Returns K `DraftToken`s, each containing the selected token_id and the
    /// full draft probability distribution at that position. The draft model
    /// runs autoregressively: token i is appended to context before generating
    /// token i+1.
    fn generate(&self, input_tokens: &[TokenId], num_tokens: usize) -> Result<Vec<DraftToken>>;

    /// Vocabulary size of the draft model.
    fn vocab_size(&self) -> usize;
}

/// Default draft model runner (placeholder implementation).
///
/// In production this would wrap a real smaller model (e.g., same architecture
/// with fewer layers, or a separate small LM like TinyLlama). The placeholder
/// uses deterministic token selection for testing.
pub struct DraftModelRunner {
    config: SpeculativeConfig,
    vocab_size: usize,
}

impl DraftModelRunner {
    /// Create a new draft model runner from config.
    ///
    /// In a real implementation this would load the draft model weights.
    pub fn new(config: SpeculativeConfig) -> Result<Self> {
        if config.draft_model_path.is_empty() {
            return Err(LLMError::ConfigError(
                "draft_model_path must not be empty".into(),
            ));
        }
        tracing::info!(
            path = %config.draft_model_path,
            k = config.num_speculative_tokens,
            "initializing draft model runner"
        );
        Ok(Self {
            config,
            vocab_size: 32000, // placeholder vocab size
        })
    }

    /// Generate `num_tokens` draft tokens autoregressively from `input_tokens`.
    ///
    /// Each draft token includes the full probability distribution so the
    /// verifier can perform speculative sampling.
    pub fn generate_draft_tokens(
        &self,
        input_tokens: &[TokenId],
        num_tokens: usize,
    ) -> Result<Vec<DraftToken>> {
        self.generate(input_tokens, num_tokens)
    }

    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }
}

impl DraftModel for DraftModelRunner {
    fn generate(&self, input_tokens: &[TokenId], num_tokens: usize) -> Result<Vec<DraftToken>> {
        if input_tokens.is_empty() {
            return Err(LLMError::ModelError(
                "cannot generate drafts from empty input".into(),
            ));
        }

        tracing::debug!(
            input_len = input_tokens.len(),
            num_tokens,
            "generating draft tokens"
        );

        let mut drafts = Vec::with_capacity(num_tokens);
        let mut context: Vec<TokenId> = input_tokens.to_vec();

        for i in 0..num_tokens {
            // Placeholder: deterministic distribution. A real implementation
            // runs the draft model forward pass here.
            let mut probs = vec![0.0f32; self.vocab_size];
            let last = *context.last().unwrap_or(&0);
            let selected = ((last as usize + i + 1) % self.vocab_size) as TokenId;
            probs[selected as usize] = 1.0;

            drafts.push(DraftToken {
                token_id: selected,
                logprob: 0.0, // log(1.0)
                draft_probs: probs,
            });

            context.push(selected);
        }

        Ok(drafts)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SpeculativeConfig {
        SpeculativeConfig::new("/models/draft".into(), 3)
    }

    #[test]
    fn new_requires_model_path() {
        let cfg = SpeculativeConfig {
            draft_model_path: String::new(),
            ..test_config()
        };
        assert!(DraftModelRunner::new(cfg).is_err());
    }

    #[test]
    fn new_succeeds_with_valid_config() {
        let runner = DraftModelRunner::new(test_config()).unwrap();
        assert_eq!(runner.vocab_size(), 32000);
    }

    #[test]
    fn generate_draft_tokens_basic() {
        let runner = DraftModelRunner::new(test_config()).unwrap();
        let input = vec![100, 200, 300];
        let drafts = runner.generate_draft_tokens(&input, 3).unwrap();
        assert_eq!(drafts.len(), 3);
        for d in &drafts {
            assert_eq!(d.draft_probs.len(), 32000);
            let sum: f32 = d.draft_probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn generate_draft_tokens_empty_input() {
        let runner = DraftModelRunner::new(test_config()).unwrap();
        assert!(runner.generate_draft_tokens(&[], 3).is_err());
    }

    #[test]
    fn generate_draft_tokens_zero_tokens() {
        let runner = DraftModelRunner::new(test_config()).unwrap();
        let drafts = runner.generate_draft_tokens(&[1, 2], 0).unwrap();
        assert!(drafts.is_empty());
    }
}
