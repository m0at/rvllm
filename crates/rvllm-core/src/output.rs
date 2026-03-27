//! Completion and request output types.

use serde::{Deserialize, Serialize};

use crate::types::{FinishReason, LogProb, RequestId, TokenId};

/// A single completion sequence produced by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionOutput {
    /// Index of this completion within the request (for best_of > 1).
    pub index: usize,
    /// Decoded text.
    pub text: String,
    /// Token ids generated.
    pub token_ids: Vec<TokenId>,
    /// Sum of log-probabilities over the generated tokens.
    pub cumulative_logprob: LogProb,
    /// Per-position top log-probabilities, if requested.
    pub logprobs: Option<Vec<Vec<(TokenId, LogProb)>>>,
    /// Why generation stopped.
    pub finish_reason: Option<FinishReason>,
}

/// Full output for a single request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOutput {
    /// The request that produced this output.
    pub request_id: RequestId,
    /// Original prompt text.
    pub prompt: String,
    /// Tokenized prompt.
    pub prompt_token_ids: Vec<TokenId>,
    /// Per-position logprobs for the prompt tokens (when `echo: true`).
    pub prompt_logprobs: Option<Vec<Vec<(TokenId, LogProb)>>>,
    /// One or more completion sequences.
    pub outputs: Vec<CompletionOutput>,
    /// Whether all sequences have finished.
    pub finished: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_output_serde() {
        let co = CompletionOutput {
            index: 0,
            text: "hello".into(),
            token_ids: vec![1, 2, 3],
            cumulative_logprob: -1.5,
            logprobs: None,
            finish_reason: Some(FinishReason::Stop),
        };
        let json = serde_json::to_string(&co).unwrap();
        let co2: CompletionOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(co2.text, "hello");
        assert_eq!(co2.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn request_output_serde() {
        let ro = RequestOutput {
            request_id: RequestId(1),
            prompt: "test prompt".into(),
            prompt_token_ids: vec![10, 20],
            prompt_logprobs: None,
            outputs: vec![],
            finished: false,
        };
        let json = serde_json::to_string(&ro).unwrap();
        let ro2: RequestOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(ro2.prompt, "test prompt");
        assert!(!ro2.finished);
    }

    #[test]
    fn send_sync_assertions() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CompletionOutput>();
        assert_send_sync::<RequestOutput>();
    }
}
