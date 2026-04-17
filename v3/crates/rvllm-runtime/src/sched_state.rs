//! Request state machine per spec 07.
//!
//! Transitions are explicit: `Queued → Prefilling → Decoding → Finished`.
//! `Aborted` reachable from any state.

use rvllm_core::{ReqId, TokenId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReqState {
    Queued,
    Prefilling,
    Decoding,
    Finished,
    Aborted,
}

#[derive(Debug)]
pub struct Request {
    pub id: ReqId,
    pub state: ReqState,
    pub prompt_tokens: Vec<TokenId>,
    pub output_tokens: Vec<TokenId>,
    pub max_output_tokens: u32,
}

impl Request {
    pub fn new(id: ReqId, prompt_tokens: Vec<TokenId>, max_output_tokens: u32) -> Self {
        Self {
            id,
            state: ReqState::Queued,
            prompt_tokens,
            output_tokens: Vec::new(),
            max_output_tokens,
        }
    }

    pub fn is_alive(&self) -> bool {
        !matches!(self.state, ReqState::Finished | ReqState::Aborted)
    }

    pub fn is_decoding(&self) -> bool {
        matches!(self.state, ReqState::Decoding)
    }

    pub fn context_len(&self) -> u32 {
        (self.prompt_tokens.len() + self.output_tokens.len()) as u32
    }

    pub fn push_output(&mut self, tok: TokenId) {
        self.output_tokens.push(tok);
        if self.output_tokens.len() as u32 >= self.max_output_tokens {
            self.state = ReqState::Finished;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_finishes_at_max_tokens() {
        let mut r = Request::new(ReqId(1), vec![TokenId(0); 4], 3);
        r.state = ReqState::Decoding;
        r.push_output(TokenId(1));
        r.push_output(TokenId(2));
        assert!(r.is_decoding());
        r.push_output(TokenId(3));
        assert_eq!(r.state, ReqState::Finished);
        assert!(!r.is_alive());
    }
}
