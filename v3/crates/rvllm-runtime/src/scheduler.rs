//! Scheduler per spec 07.
//!
//! Emits one `BatchPlan` per step of exactly one variant (`Prefill`,
//! `Decode`, or `Idle`). No mixed prefill+decode in the same step —
//! that was one of the metadata-coupling sources in v2.

use rvllm_core::{ReqId, TokenId};

use crate::sched_state::{ReqState, Request};

/// Bucket list for decode. Must match graph-capture buckets.
pub const DECODE_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256];

/// Largest single-step decode batch the scheduler can pack. Above this
/// the scheduler clamps to the largest bucket and defers the
/// remainder to the next step (see `schedule()`).
pub const MAX_DECODE_BATCH: u32 = 256;

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
    /// Round-robin cursor over decoding requests. When more than
    /// `MAX_DECODE_BATCH` are alive concurrently, each `schedule()`
    /// call walks a different window so requests 257+ are not
    /// starved waiting for the head of the queue to finish. The
    /// cursor is interpreted modulo the number of decoding requests
    /// at scheduling time.
    decode_cursor: usize,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            requests: Vec::with_capacity(256),
            decode_cursor: 0,
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
        //
        // When more than `MAX_DECODE_BATCH` are alive at once we clamp
        // to the largest bucket AND rotate the start cursor each call
        // so requests 257+ are not starved waiting for the head of
        // the queue to finish (the previous `take(256)` always picked
        // the same first 256 in iteration order). The deadlock guard
        // is the clamp itself — without it `bucket_for(actual)`
        // returned `None` and `schedule()` reported `Idle` even
        // though the alive set was non-empty.
        let decoding_idxs: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter_map(|(i, r)| if r.is_decoding() { Some(i) } else { None })
            .collect();
        if decoding_idxs.is_empty() {
            self.decode_cursor = 0;
            return BatchPlan::Idle;
        }
        let total = decoding_idxs.len();
        let cap = (MAX_DECODE_BATCH as usize).min(total);
        // Round-robin window: start at `decode_cursor % total`, take
        // `cap` items with wrap-around.
        let start = self.decode_cursor % total;
        let window: Vec<usize> = (0..cap)
            .map(|k| decoding_idxs[(start + k) % total])
            .collect();
        // Advance the cursor by the window size, modulo total. When
        // total <= cap (the typical case) we land back at 0; when
        // total > cap we walk forward and eventually wrap, giving
        // each request a turn.
        self.decode_cursor = (start + cap) % total;
        let active: Vec<&Request> = window.iter().map(|&i| &self.requests[i]).collect();
        let actual = active.len() as u32;
        // After clamping `actual <= MAX_DECODE_BATCH`, so bucket_for
        // never returns None — the `expect` here is a load-bearing
        // invariant guard, not a fallible path.
        let bucket = bucket_for(actual)
            .expect("active.len() <= MAX_DECODE_BATCH, every bucket entry covers it");
        let mut req_ids = Vec::with_capacity(active.len());
        let mut last_tokens = Vec::with_capacity(active.len());
        let mut positions = Vec::with_capacity(active.len());
        let mut context_lens = Vec::with_capacity(active.len());
        for r in &active {
            req_ids.push(r.id);
            // Lazy fallback: `unwrap_or` would eagerly index
            // `prompt_tokens[len - 1]` and panic with `usize`
            // underflow if both vecs are empty. `or_else` + `.last()`
            // keeps the failure mode an explicit `expect`.
            last_tokens.push(
                *r.output_tokens
                    .last()
                    .or_else(|| r.prompt_tokens.last())
                    .expect("scheduled request has neither prompt nor output tokens"),
            );
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
    fn schedule_clamps_above_max_batch_and_defers_rest() {
        // Regression: > MAX_DECODE_BATCH active decodes used to make
        // schedule() return Idle, deadlocking the runtime since
        // requests stayed Decoding forever. Now the first
        // MAX_DECODE_BATCH are scheduled; the remainder are picked
        // up by the next schedule() call.
        let mut s = Scheduler::new();
        let n = (MAX_DECODE_BATCH as u32) + 10;
        for i in 0..n {
            s.enqueue(Request::new(ReqId(i as u64), vec![TokenId(i)], 4));
        }
        // First call: prefill all queued requests.
        match s.schedule() {
            BatchPlan::Prefill { .. } => {}
            other => panic!("expected Prefill, got {other:?}"),
        }
        // Second call: must emit Decode for the first MAX_DECODE_BATCH,
        // NOT Idle.
        match s.schedule() {
            BatchPlan::Decode { req_ids, bucket, .. } => {
                assert_eq!(req_ids.len() as u32, MAX_DECODE_BATCH);
                assert_eq!(bucket, MAX_DECODE_BATCH);
            }
            BatchPlan::Idle => panic!(
                "schedule() returned Idle while {n} active requests are alive — \
                 this is the >256 deadlock"
            ),
            other => panic!("unexpected plan: {other:?}"),
        }
    }

    #[test]
    fn schedule_rotates_window_to_avoid_starvation() {
        // Above MAX_DECODE_BATCH, the previous implementation kept
        // serving the first 256 decoding requests every call until
        // they finished, starving the rest. The rotating cursor
        // guarantees that successive schedule() calls walk through
        // every request, so each gets a turn within
        // ceil(total / MAX_DECODE_BATCH) calls.
        let mut s = Scheduler::new();
        let n = (MAX_DECODE_BATCH as u32) + 10;
        for i in 0..n {
            s.enqueue(Request::new(ReqId(i as u64), vec![TokenId(i)], 4));
        }
        // Drain prefill.
        match s.schedule() {
            BatchPlan::Prefill { .. } => {}
            other => panic!("expected Prefill, got {other:?}"),
        }
        // Collect req-id sets across two consecutive decode steps.
        let scheduled = |s: &mut Scheduler| -> std::collections::HashSet<u64> {
            match s.schedule() {
                BatchPlan::Decode { req_ids, .. } => {
                    req_ids.into_iter().map(|r| r.0).collect()
                }
                other => panic!("expected Decode, got {other:?}"),
            }
        };
        let first = scheduled(&mut s);
        let second = scheduled(&mut s);
        assert_eq!(first.len(), MAX_DECODE_BATCH as usize);
        assert_eq!(second.len(), MAX_DECODE_BATCH as usize);
        // First window starts at index 0, takes 256 → ids [0..256).
        // Cursor advances to 256, second window wraps and covers
        // [256, 266) ∪ [0, 246). The 10 tail requests (256..266) MUST
        // appear in the second window — that is the starvation guard.
        for tail in (MAX_DECODE_BATCH as u64)..(n as u64) {
            assert!(
                second.contains(&tail),
                "tail request {tail} was starved across the second schedule() call"
            );
        }
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
            BatchPlan::Decode {
                req_ids, bucket, ..
            } => {
                assert_eq!(req_ids.len(), 2);
                assert_eq!(bucket, 2);
            }
            other => panic!("expected Decode, got {other:?}"),
        }
    }
}
