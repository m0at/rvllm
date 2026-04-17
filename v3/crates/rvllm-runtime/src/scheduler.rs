//! Scheduler per spec 07.
//!
//! Emits one `BatchPlan` per step of exactly one variant (`Prefill`,
//! `Decode`, or `Idle`). No mixed prefill+decode in the same step —
//! that was one of the metadata-coupling sources in v2.

use rvllm_core::{ReqId, TokenId};

use crate::sched_state::{ReqState, Request};

/// Bucket list for decode. Must match graph-capture buckets.
pub const DECODE_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256];

/// Smallest decode bucket that holds `actual` sequences.
pub fn bucket_for(actual: u32) -> Option<u32> {
    DECODE_BUCKETS.iter().copied().find(|&b| b >= actual)
}

/// Scheduler output for one step.
#[derive(Debug)]
pub enum BatchPlan {
    Idle,
    Prefill {
        req_ids: Vec<ReqId>,
        prompt_tokens_flat: Vec<TokenId>,
        cu_seqlens_q: Vec<u32>,
    },
    Decode {
        req_ids: Vec<ReqId>,
        bucket: u32,
        last_tokens: Vec<TokenId>,
        positions: Vec<u32>,
        context_lens: Vec<u32>,
    },
}

pub struct Scheduler {
    requests: Vec<Request>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            requests: Vec::with_capacity(256),
        }
    }

    pub fn enqueue(&mut self, req: Request) {
        self.requests.push(req);
    }

    pub fn num_alive(&self) -> usize {
        self.requests.iter().filter(|r| r.is_alive()).count()
    }

    /// Pick the next step's plan. Prefill wins over decode when any
    /// request is in `Queued` or `Prefilling` state.
    pub fn schedule(&mut self) -> BatchPlan {
        // Prefill: move Queued → Prefilling; emit one plan for all of them.
        let mut to_prefill: Vec<usize> = Vec::new();
        for (i, r) in self.requests.iter_mut().enumerate() {
            if matches!(r.state, ReqState::Queued | ReqState::Prefilling) {
                r.state = ReqState::Prefilling;
                to_prefill.push(i);
            }
        }
        if !to_prefill.is_empty() {
            let mut req_ids = Vec::with_capacity(to_prefill.len());
            let mut prompt_tokens_flat: Vec<TokenId> = Vec::new();
            let mut cu_seqlens_q: Vec<u32> = Vec::with_capacity(to_prefill.len() + 1);
            cu_seqlens_q.push(0);
            for &i in &to_prefill {
                let r = &self.requests[i];
                req_ids.push(r.id);
                prompt_tokens_flat.extend(r.prompt_tokens.iter().copied());
                cu_seqlens_q.push(prompt_tokens_flat.len() as u32);
            }
            // Mark as decoding starting next step (prefill completes in one step).
            for &i in &to_prefill {
                self.requests[i].state = ReqState::Decoding;
            }
            return BatchPlan::Prefill {
                req_ids,
                prompt_tokens_flat,
                cu_seqlens_q,
            };
        }

        // Decode: collect Decoding requests into smallest bucket.
        let active: Vec<&Request> = self.requests.iter().filter(|r| r.is_decoding()).collect();
        if active.is_empty() {
            return BatchPlan::Idle;
        }
        let actual = active.len() as u32;
        let Some(bucket) = bucket_for(actual) else {
            return BatchPlan::Idle; // too many — scheduler must preempt
        };
        let mut req_ids = Vec::with_capacity(active.len());
        let mut last_tokens = Vec::with_capacity(active.len());
        let mut positions = Vec::with_capacity(active.len());
        let mut context_lens = Vec::with_capacity(active.len());
        for r in &active {
            req_ids.push(r.id);
            last_tokens.push(*r.output_tokens.last().unwrap_or(&r.prompt_tokens[r.prompt_tokens.len() - 1]));
            positions.push(r.context_len() - 1);
            context_lens.push(r.context_len());
        }
        BatchPlan::Decode {
            req_ids,
            bucket,
            last_tokens,
            positions,
            context_lens,
        }
    }

    /// Commit per-seq outputs from a completed decode step.
    pub fn commit_decode(&mut self, req_tokens: &[(ReqId, TokenId)]) {
        for &(id, tok) in req_tokens {
            if let Some(r) = self.requests.iter_mut().find(|r| r.id == id) {
                r.push_output(tok);
            }
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_rounds_up() {
        assert_eq!(bucket_for(1), Some(1));
        assert_eq!(bucket_for(3), Some(4));
        assert_eq!(bucket_for(100), Some(128));
        assert_eq!(bucket_for(256), Some(256));
        assert_eq!(bucket_for(257), None);
    }

    #[test]
    fn schedule_emits_prefill_then_decode() {
        let mut s = Scheduler::new();
        s.enqueue(Request::new(ReqId(1), vec![TokenId(10), TokenId(11)], 4));
        s.enqueue(Request::new(ReqId(2), vec![TokenId(20), TokenId(21)], 4));
        match s.schedule() {
            BatchPlan::Prefill { req_ids, .. } => assert_eq!(req_ids.len(), 2),
            other => panic!("expected Prefill, got {other:?}"),
        }
        // After commit of first prefill round, next schedule is Decode.
        match s.schedule() {
            BatchPlan::Decode { req_ids, bucket, .. } => {
                assert_eq!(req_ids.len(), 2);
                assert_eq!(bucket, 2);
            }
            other => panic!("expected Decode, got {other:?}"),
        }
    }
}
