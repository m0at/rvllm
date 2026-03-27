//! End-to-end smoke tests for rvllm (mock-gpu).
//!
//! Exercises the full LLMEngine pipeline:
//!   create engine with mock weights -> add request -> step() until done -> verify output.
//!
//! These tests run with default features (mock-gpu) and require no GPU.

use std::collections::HashMap;

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer as HfTokenizer;

use rvllm_config::EngineConfig;
use rvllm_core::prelude::{FinishReason, RequestId, SamplingParams, TokenId};
use rvllm_engine::{ExecutorInput, LLMEngine, SamplerOutput, Scheduler, SchedulerOutputs};
use rvllm_sequence::SequenceGroup;
use rvllm_tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Mock scheduler -- returns all pending groups each step
// ---------------------------------------------------------------------------

struct MockScheduler {
    groups: Vec<SequenceGroup>,
}

impl MockScheduler {
    fn new() -> Self {
        Self { groups: Vec::new() }
    }
}

impl Scheduler for MockScheduler {
    fn add_seq_group(&mut self, seq_group: SequenceGroup) {
        self.groups.push(seq_group);
    }

    fn abort_seq_group(&mut self, request_id: &RequestId) {
        self.groups.retain(|g| g.request_id != *request_id);
    }

    fn schedule(&mut self) -> SchedulerOutputs {
        let groups = self.groups.clone();
        self.groups.retain(|g| !g.is_finished());
        let num_tokens = groups
            .iter()
            .flat_map(|g| g.get_seqs())
            .map(|s| s.num_new_tokens().max(1))
            .sum();
        SchedulerOutputs {
            scheduled_seq_groups: groups,
            num_batched_tokens: num_tokens,
            preempted: false,
        }
    }

    fn has_unfinished_seqs(&self) -> bool {
        !self.groups.is_empty()
    }

    fn get_num_unfinished_seq_groups(&self) -> usize {
        self.groups.len()
    }
}

// ---------------------------------------------------------------------------
// Mock executor -- returns a fixed token, then EOS after max_calls
// ---------------------------------------------------------------------------

struct MockExecutor {
    token_id: TokenId,
    calls: usize,
    max_calls: usize,
}

impl MockExecutor {
    fn new(token_id: TokenId, max_calls: usize) -> Self {
        Self {
            token_id,
            calls: 0,
            max_calls,
        }
    }
}

impl rvllm_engine::Executor for MockExecutor {
    fn execute_model(&mut self, input: ExecutorInput) -> rvllm_core::prelude::Result<Vec<SamplerOutput>> {
        self.calls += 1;
        let mut outputs = Vec::new();
        for meta in &input.seq_group_metadata {
            for &seq_id in meta.seq_data.keys() {
                let tid = if self.calls >= self.max_calls {
                    0 // EOS-like token to terminate
                } else {
                    self.token_id
                };
                outputs.push(SamplerOutput {
                    seq_id,
                    token_id: tid,
                    logprob: -0.5,
                    top_logprobs: None,
                });
            }
        }
        Ok(outputs)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_test_tokenizer() -> Tokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert(" ".to_string(), 2);
    vocab.insert("!".to_string(), 3);
    vocab.insert("[UNK]".to_string(), 4);

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();

    let mut hf = HfTokenizer::new(bpe);
    hf.with_pre_tokenizer(Some(Whitespace {}));

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    hf.save(&path, false).unwrap();
    Tokenizer::from_file(&path).unwrap()
}

fn make_engine(max_executor_calls: usize) -> LLMEngine {
    let config = EngineConfig::default();
    let tokenizer = make_test_tokenizer();
    let scheduler = Box::new(MockScheduler::new());
    let executor = Box::new(MockExecutor::new(1, max_executor_calls));
    LLMEngine::new(config, executor, scheduler, tokenizer).unwrap()
}

// ===========================================================================
// Tests
// ===========================================================================

/// Smoke test: create engine, add a single request, run to completion,
/// verify the RequestOutput contains generated tokens.
#[test]
fn e2e_single_request_completes() {
    let mut engine = make_engine(100);
    let mut params = SamplingParams::default();
    params.max_tokens = 5;

    engine
        .add_request(RequestId(1), "hello".to_string(), params)
        .unwrap();

    let outputs = engine.run().unwrap();

    assert_eq!(outputs.len(), 1, "expected exactly one completed request");
    let output = &outputs[0];
    assert_eq!(output.request_id, RequestId(1));
    assert!(output.finished, "request should be marked finished");
    assert!(
        !output.outputs.is_empty(),
        "should have at least one completion"
    );

    let completion = &output.outputs[0];
    assert!(
        !completion.token_ids.is_empty(),
        "completion should have generated tokens"
    );
    assert!(
        completion.finish_reason.is_some(),
        "completion should have a finish reason"
    );
}

/// Verify that step() produces incremental outputs before the request finishes.
#[test]
fn e2e_step_produces_incremental_output() {
    let mut engine = make_engine(100);
    let mut params = SamplingParams::default();
    params.max_tokens = 3;

    engine
        .add_request(RequestId(1), "hello".to_string(), params)
        .unwrap();

    // First step should produce output but not be finished yet
    let step1 = engine.step().unwrap();
    assert!(!step1.is_empty(), "first step should produce output");
    // The request may or may not be finished after 1 step depending on max_tokens
    // but it should have a RequestOutput
    assert_eq!(step1[0].request_id, RequestId(1));
}

/// Multiple requests can be added and all complete.
#[test]
fn e2e_multiple_requests_complete() {
    let mut engine = make_engine(100);
    let mut params = SamplingParams::default();
    params.max_tokens = 3;

    engine
        .add_request(RequestId(1), "hello".to_string(), params.clone())
        .unwrap();
    engine
        .add_request(RequestId(2), "world".to_string(), params)
        .unwrap();

    let outputs = engine.run().unwrap();

    assert_eq!(outputs.len(), 2, "both requests should complete");
    assert!(outputs.iter().all(|o| o.finished));

    let ids: Vec<RequestId> = outputs.iter().map(|o| o.request_id).collect();
    assert!(ids.contains(&RequestId(1)));
    assert!(ids.contains(&RequestId(2)));
}

/// Finish reason is Length when max_tokens is hit.
#[test]
fn e2e_finish_reason_length() {
    let mut engine = make_engine(100);
    let mut params = SamplingParams::default();
    params.max_tokens = 2;

    engine
        .add_request(RequestId(1), "hello".to_string(), params)
        .unwrap();

    let outputs = engine.run().unwrap();
    assert_eq!(outputs.len(), 1);

    let completion = &outputs[0].outputs[0];
    assert_eq!(
        completion.finish_reason,
        Some(FinishReason::Length),
        "should finish due to max_tokens"
    );
    assert_eq!(
        completion.token_ids.len(),
        2,
        "should have exactly max_tokens tokens"
    );
}

/// Verify prompt and prompt_token_ids are preserved in the output.
#[test]
fn e2e_output_preserves_prompt() {
    let mut engine = make_engine(100);
    let mut params = SamplingParams::default();
    params.max_tokens = 1;

    engine
        .add_request(RequestId(1), "hello".to_string(), params)
        .unwrap();

    let outputs = engine.run().unwrap();
    let output = &outputs[0];

    assert_eq!(output.prompt, "hello");
    assert!(
        !output.prompt_token_ids.is_empty(),
        "prompt_token_ids should not be empty"
    );
}

/// Aborting a request removes it from the scheduler so no further
/// forward passes are issued.  The internal request map still holds
/// the aborted entry (with finish_reason = Abort) until step() cleans
/// it up.  After one more step, the engine should be idle.
#[test]
fn e2e_abort_request() {
    let mut engine = make_engine(100);
    let mut params = SamplingParams::default();
    params.max_tokens = 10;

    engine
        .add_request(RequestId(1), "hello".to_string(), params)
        .unwrap();

    // Run one step so the request gets scheduled and produces output.
    let _ = engine.step().unwrap();

    // Now abort -- removes from scheduler.
    engine.abort_request(&RequestId(1));

    // The step() that follows should see no scheduled groups (empty schedule)
    // and return nothing. The cleanup path in step() only fires for groups
    // that were scheduled, so we call step twice to ensure the bookkeeping
    // settles.
    let out = engine.step().unwrap();
    assert!(
        out.is_empty(),
        "step after abort should produce no output"
    );
}

/// Empty engine returns empty output on step().
#[test]
fn e2e_empty_engine_step() {
    let mut engine = make_engine(100);
    let outputs = engine.step().unwrap();
    assert!(
        outputs.is_empty(),
        "step on empty engine should return no outputs"
    );
    assert!(!engine.has_unfinished());
}
