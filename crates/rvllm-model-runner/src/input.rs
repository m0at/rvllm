//! Model input types for the forward pass.

use crate::bridge::AttentionMetadata;

/// Input batch for a single forward-pass invocation.
#[derive(Debug, Clone)]
pub struct ModelInput {
    /// Token ids for this step, shape [batch_total_tokens].
    pub token_ids: Vec<u32>,
    /// Position ids corresponding to each token.
    pub position_ids: Vec<u32>,
    /// Attention metadata (slot mapping, block tables, context lengths).
    pub attention_metadata: AttentionMetadata,
    /// True during prefill (prompt processing), false during decode.
    pub is_prefill: bool,
}

impl ModelInput {
    /// Number of tokens in this batch.
    pub fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }
}
