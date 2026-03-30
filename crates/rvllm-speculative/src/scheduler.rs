//! Speculative scheduling: coordinate draft generation and target verification.
//!
//! The scheduler handles batched speculative decoding across multiple sequences.
//! For each sequence it:
//!   1. Runs the draft model for K steps
//!   2. Builds the concatenated verification input for the target model
//!   3. Tracks per-sequence offsets so verification results can be demuxed

use rvllm_core::prelude::{LLMError, Result, TokenId};
use rvllm_sequence::Sequence;

use crate::config::SpeculativeConfig;
use crate::draft::{DraftModelRunner, DraftToken};
use crate::verification::{verify_tokens, VerificationResult};

/// Per-sequence metadata for a speculative step.
#[derive(Debug, Clone)]
pub struct SeqSpecInfo {
    /// Index into the batch.
    pub seq_index: usize,
    /// Number of context tokens (before draft tokens) for this sequence.
    pub context_len: usize,
    /// Number of draft tokens generated for this sequence.
    pub num_draft_tokens: usize,
    /// Start offset of this sequence's tokens in the concatenated target_input.
    pub input_offset: usize,
    /// Total tokens for this sequence in target_input (context_len + num_draft_tokens).
    pub input_len: usize,
}

/// The output of a speculative scheduling step: draft tokens per sequence
/// plus the concatenated input for the target model verification pass.
#[derive(Debug, Clone)]
pub struct SpeculativeStep {
    /// Per-sequence draft tokens (K tokens each).
    pub draft_tokens: Vec<Vec<DraftToken>>,
    /// Concatenated token sequence for the target model to verify.
    /// For each sequence: original tokens + K draft tokens.
    pub target_input: Vec<TokenId>,
    /// Per-sequence metadata for demuxing verification results.
    pub seq_info: Vec<SeqSpecInfo>,
}

impl SpeculativeStep {
    /// Extract target probability distributions for a specific sequence
    /// from a flat batch of target_probs.
    ///
    /// The target model returns probabilities for each position in target_input.
    /// This extracts the K+1 distributions relevant to one sequence's
    /// verification (the last K+1 positions of that sequence's slice).
    pub fn extract_target_probs_for_seq(
        &self,
        seq_index: usize,
        all_target_probs: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let info = &self.seq_info[seq_index];
        let k = info.num_draft_tokens;
        // We need the K+1 distributions at the end of this sequence's slice:
        // positions [context_len-1 .. context_len+K-1] relative to the sequence start,
        // which covers K draft positions + 1 bonus position.
        // In the flat target_probs, this sequence's positions start at input_offset.
        let verify_start = info.input_offset + info.context_len - 1;
        let verify_end = verify_start + k + 1;

        if verify_end <= all_target_probs.len() {
            all_target_probs[verify_start..verify_end].to_vec()
        } else {
            // Fallback: return whatever is available
            let start = verify_start.min(all_target_probs.len());
            let end = verify_end.min(all_target_probs.len());
            all_target_probs[start..end].to_vec()
        }
    }

    /// Verify all sequences against target probabilities, returning per-sequence
    /// verification results.
    pub fn verify_all(
        &self,
        all_target_probs: &[Vec<f32>],
    ) -> Vec<VerificationResult> {
        let mut results = Vec::with_capacity(self.draft_tokens.len());

        for (i, drafts) in self.draft_tokens.iter().enumerate() {
            let draft_probs: Vec<Vec<f32>> = drafts.iter().map(|d| d.draft_probs.clone()).collect();
            let draft_token_ids: Vec<TokenId> = drafts.iter().map(|d| d.token_id).collect();
            let target_probs = self.extract_target_probs_for_seq(i, all_target_probs);

            let result = verify_tokens(&draft_probs, &target_probs, &draft_token_ids);
            results.push(result);
        }

        results
    }
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
    pub fn prepare_draft_and_target(&self, sequences: &[Sequence]) -> Result<SpeculativeStep> {
        if sequences.is_empty() {
            return Err(LLMError::SchedulerError("no sequences to schedule".into()));
        }

        let k = self.config.num_speculative_tokens;
        let mut all_drafts = Vec::with_capacity(sequences.len());
        let mut target_input = Vec::new();
        let mut seq_info = Vec::with_capacity(sequences.len());

        for (idx, seq) in sequences.iter().enumerate() {
            let tokens = seq.get_token_ids();
            if tokens.is_empty() {
                return Err(LLMError::SchedulerError("sequence has no tokens".into()));
            }

            tracing::debug!(
                seq_id = %seq.seq_id,
                token_count = tokens.len(),
                k,
                "generating draft tokens for sequence"
            );

            let drafts = self.draft_runner.generate_draft_tokens(&tokens, k)?;
            let input_offset = target_input.len();
            let context_len = tokens.len();

            // Target input: original context + all draft tokens
            target_input.extend_from_slice(&tokens);
            for d in &drafts {
                target_input.push(d.token_id);
            }

            seq_info.push(SeqSpecInfo {
                seq_index: idx,
                context_len,
                num_draft_tokens: drafts.len(),
                input_offset,
                input_len: context_len + drafts.len(),
            });

            all_drafts.push(drafts);
        }

        Ok(SpeculativeStep {
            draft_tokens: all_drafts,
            target_input,
            seq_info,
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

        // Check seq_info
        assert_eq!(step.seq_info.len(), 1);
        assert_eq!(step.seq_info[0].context_len, 3);
        assert_eq!(step.seq_info[0].num_draft_tokens, 3);
        assert_eq!(step.seq_info[0].input_offset, 0);
        assert_eq!(step.seq_info[0].input_len, 6);
    }

    #[test]
    fn prepare_empty_sequences_fails() {
        let scheduler = SpeculativeScheduler::new(test_config()).unwrap();
        assert!(scheduler.prepare_draft_and_target(&[]).is_err());
    }

    #[test]
    fn prepare_multiple_sequences() {
        let scheduler = SpeculativeScheduler::new(test_config()).unwrap();
        let seqs = vec![make_seq(1, vec![10, 20]), make_seq(2, vec![30, 40, 50])];
        let step = scheduler.prepare_draft_and_target(&seqs).unwrap();
        assert_eq!(step.draft_tokens.len(), 2);
        assert_eq!(step.draft_tokens[0].len(), 3);
        assert_eq!(step.draft_tokens[1].len(), 3);
        // (2+3) + (3+3) = 11
        assert_eq!(step.target_input.len(), 11);

        // Check offsets
        assert_eq!(step.seq_info[0].input_offset, 0);
        assert_eq!(step.seq_info[0].input_len, 5); // 2 context + 3 draft
        assert_eq!(step.seq_info[1].input_offset, 5);
        assert_eq!(step.seq_info[1].input_len, 6); // 3 context + 3 draft
    }

    #[test]
    fn verify_all_self_speculation() {
        let scheduler = SpeculativeScheduler::new(test_config()).unwrap();
        let seqs = vec![make_seq(1, vec![10, 20, 30])];
        let step = scheduler.prepare_draft_and_target(&seqs).unwrap();

        // Build per-position target probs for the full target_input.
        // extract_target_probs_for_seq will pull out positions [context_len-1 .. context_len+K).
        // For self-speculation the target must match the draft at each position.
        let vocab = 32000;
        let k = 3;
        let mut all_target_probs = Vec::new();

        // Positions 0..context_len-1 don't matter for verification, fill with dummy.
        let context_len = step.seq_info[0].context_len; // 3
        for _ in 0..context_len - 1 {
            let mut probs = vec![0.0f32; vocab];
            probs[0] = 1.0;
            all_target_probs.push(probs);
        }

        // Positions context_len-1 .. context_len-1+K+1: must match draft distributions.
        // The draft produced tokens at these positions. Copy draft distributions directly.
        for i in 0..k {
            all_target_probs.push(step.draft_tokens[0][i].draft_probs.clone());
        }
        // Bonus position (K+1th): just put mass on some token
        let mut bonus_probs = vec![0.0f32; vocab];
        bonus_probs[42] = 1.0;
        all_target_probs.push(bonus_probs);

        let results = step.verify_all(&all_target_probs);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_accepted, k);
        assert!(results[0].bonus_token.is_some());
        assert_eq!(results[0].bonus_token.unwrap(), 42);
    }
}
