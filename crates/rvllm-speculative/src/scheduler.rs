//! Speculative scheduling: coordinate draft generation and target verification.

use rvllm_core::prelude::{LLMError, Result, TokenId};
use rvllm_sequence::Sequence;

use crate::config::SpeculativeConfig;
use crate::draft::{DraftModelRunner, DraftToken};

/// The output of a speculative scheduling step: draft tokens per sequence
/// plus the concatenated input for the target model verification pass.
#[derive(Debug, Clone)]
pub struct SpeculativeStep {
    /// Per-sequence draft tokens (K tokens each).
    pub draft_tokens: Vec<Vec<DraftToken>>,
    /// Concatenated token sequence for the target model to verify.
    /// For each sequence: original tokens + K draft tokens.
    pub target_input: Vec<TokenId>,
}

/// Coordinates draft model execution and target model verification preparation.
pub struct SpeculativeScheduler {
    config: SpeculativeConfig,
    draft_runner: DraftModelRunner,
}

impl SpeculativeScheduler {
    pub fn new(config: SpeculativeConfig) -> Result<Self> {
        let draft_runner = DraftModelRunner::new(config.clone())?;
        Ok(Self {
            config,
            draft_runner,
        })
    }

    /// Run the draft model K steps for each sequence, then prepare the
    /// combined input for the target model to verify all K+1 positions.
    pub fn prepare_draft_and_target(
        &self,
        sequences: &[Sequence],
    ) -> Result<SpeculativeStep> {
        if sequences.is_empty() {
            return Err(LLMError::SchedulerError(
                "no sequences to schedule".into(),
            ));
        }

        let k = self.config.num_speculative_tokens;
        let mut all_drafts = Vec::with_capacity(sequences.len());
        let mut target_input = Vec::new();

        for seq in sequences {
            let tokens = seq.get_token_ids();
            if tokens.is_empty() {
                return Err(LLMError::SchedulerError(
                    "sequence has no tokens".into(),
                ));
            }

            tracing::debug!(
                seq_id = %seq.seq_id,
                token_count = tokens.len(),
                k,
                "generating draft tokens for sequence"
            );

            let drafts = self.draft_runner.generate_draft_tokens(&tokens, k)?;

            // Target input: original context + all draft tokens
            target_input.extend_from_slice(&tokens);
            for d in &drafts {
                target_input.push(d.token_id);
            }

            all_drafts.push(drafts);
        }

        Ok(SpeculativeStep {
            draft_tokens: all_drafts,
            target_input,
        })
    }

    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::prelude::SequenceId;

    fn test_config() -> SpeculativeConfig {
        SpeculativeConfig::new("/models/draft".into(), 3)
    }

    fn make_seq(id: u64, prompt: Vec<TokenId>) -> Sequence {
        Sequence::new(SequenceId(id), prompt)
    }

    #[test]
    fn prepare_draft_and_target_basic() {
        let scheduler = SpeculativeScheduler::new(test_config()).unwrap();
        let seqs = vec![make_seq(1, vec![10, 20, 30])];
        let step = scheduler.prepare_draft_and_target(&seqs).unwrap();

        assert_eq!(step.draft_tokens.len(), 1);
        assert_eq!(step.draft_tokens[0].len(), 3);
        // target_input = original 3 tokens + 3 draft tokens
        assert_eq!(step.target_input.len(), 6);
        assert_eq!(&step.target_input[..3], &[10, 20, 30]);
    }

    #[test]
    fn prepare_empty_sequences_fails() {
        let scheduler = SpeculativeScheduler::new(test_config()).unwrap();
        assert!(scheduler.prepare_draft_and_target(&[]).is_err());
    }

    #[test]
    fn prepare_multiple_sequences() {
        let scheduler = SpeculativeScheduler::new(test_config()).unwrap();
        let seqs = vec![
            make_seq(1, vec![10, 20]),
            make_seq(2, vec![30, 40, 50]),
        ];
        let step = scheduler.prepare_draft_and_target(&seqs).unwrap();
        assert_eq!(step.draft_tokens.len(), 2);
        assert_eq!(step.draft_tokens[0].len(), 3);
        assert_eq!(step.draft_tokens[1].len(), 3);
        // (2+3) + (3+3) = 11
        assert_eq!(step.target_input.len(), 11);
    }
}
