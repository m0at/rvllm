use serde::{Deserialize, Serialize};
use rvllm_core::prelude::{SequenceId, TokenId};

use crate::status::SequenceStatus;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    pub seq_id: SequenceId,
    pub prompt_token_ids: Vec<TokenId>,
    pub output_token_ids: Vec<TokenId>,
    pub status: SequenceStatus,
    pub cumulative_logprob: f32,
    pub logprobs: Vec<(TokenId, f32)>,
    pub num_computed_tokens: usize,
}

impl Sequence {
    pub fn new(seq_id: SequenceId, prompt_token_ids: Vec<TokenId>) -> Self {
        Self {
            seq_id,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            status: SequenceStatus::Waiting,
            cumulative_logprob: 0.0,
            logprobs: Vec::new(),
            num_computed_tokens: 0,
        }
    }

    pub fn append_token(&mut self, token_id: TokenId, logprob: f32) {
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
        self.logprobs.push((token_id, logprob));
    }

    pub fn get_len(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    pub fn get_output_len(&self) -> usize {
        self.output_token_ids.len()
    }

    pub fn get_token_ids(&self) -> Vec<TokenId> {
        let mut ids = self.prompt_token_ids.clone();
        ids.extend_from_slice(&self.output_token_ids);
        ids
    }

    pub fn get_last_token_id(&self) -> Option<TokenId> {
        self.output_token_ids
            .last()
            .or(self.prompt_token_ids.last())
            .copied()
    }

    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    pub fn set_status(&mut self, status: SequenceStatus) -> rvllm_core::prelude::Result<()> {
        self.status.validate_transition(status)?;
        self.status = status;
        Ok(())
    }

    /// Number of tokens not yet computed (for chunked prefill).
    pub fn num_new_tokens(&self) -> usize {
        self.get_len().saturating_sub(self.num_computed_tokens)
    }

    /// Number of KV-cache blocks needed for the full sequence at the given block size.
    pub fn get_num_blocks(&self, block_size: usize) -> usize {
        let total = self.get_len();
        total.div_ceil(block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq() -> Sequence {
        Sequence::new(SequenceId(1), vec![10, 20, 30])
    }

    #[test]
    fn new_sequence_defaults() {
        let s = make_seq();
        assert_eq!(s.get_prompt_len(), 3);
        assert_eq!(s.get_output_len(), 0);
        assert_eq!(s.get_len(), 3);
        assert_eq!(s.status, SequenceStatus::Waiting);
        assert_eq!(s.cumulative_logprob, 0.0);
    }

    #[test]
    fn append_token() {
        let mut s = make_seq();
        s.append_token(42, -0.5);
        s.append_token(43, -1.0);
        assert_eq!(s.get_output_len(), 2);
        assert_eq!(s.get_len(), 5);
        assert!((s.cumulative_logprob - (-1.5)).abs() < 1e-6);
        assert_eq!(s.logprobs.len(), 2);
    }

    #[test]
    fn get_token_ids_concat() {
        let mut s = make_seq();
        s.append_token(99, -0.1);
        assert_eq!(s.get_token_ids(), vec![10, 20, 30, 99]);
    }

    #[test]
    fn last_token_id() {
        let s = make_seq();
        assert_eq!(s.get_last_token_id(), Some(30));

        let mut s2 = make_seq();
        s2.append_token(55, -0.1);
        assert_eq!(s2.get_last_token_id(), Some(55));

        let empty = Sequence::new(SequenceId(0), vec![]);
        assert_eq!(empty.get_last_token_id(), None);
    }

    #[test]
    fn set_status_valid() {
        let mut s = make_seq();
        assert!(s.set_status(SequenceStatus::Running).is_ok());
        assert_eq!(s.status, SequenceStatus::Running);
        assert!(s.set_status(SequenceStatus::FinishedStopped).is_ok());
        assert!(s.is_finished());
    }

    #[test]
    fn set_status_invalid_from_finished() {
        let mut s = make_seq();
        s.set_status(SequenceStatus::FinishedAborted).unwrap();
        assert!(s.set_status(SequenceStatus::Running).is_err());
    }

    #[test]
    fn num_new_tokens() {
        let mut s = make_seq();
        assert_eq!(s.num_new_tokens(), 3);
        s.num_computed_tokens = 2;
        assert_eq!(s.num_new_tokens(), 1);
        s.append_token(77, -0.1);
        assert_eq!(s.num_new_tokens(), 2);
    }

    #[test]
    fn get_num_blocks() {
        let mut s = make_seq();
        // 3 tokens, block_size=2 -> ceil(3/2) = 2
        assert_eq!(s.get_num_blocks(2), 2);
        s.append_token(1, -0.1);
        // 4 tokens, block_size=2 -> 2
        assert_eq!(s.get_num_blocks(2), 2);
        // 4 tokens, block_size=3 -> ceil(4/3) = 2
        assert_eq!(s.get_num_blocks(3), 2);
    }

    #[test]
    fn serde_roundtrip() {
        let mut s = make_seq();
        s.append_token(42, -0.5);
        let json = serde_json::to_string(&s).unwrap();
        let s2: Sequence = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.seq_id, s.seq_id);
        assert_eq!(s2.output_token_ids, s.output_token_ids);
    }
}
