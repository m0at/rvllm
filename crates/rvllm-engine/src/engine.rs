//! Synchronous inference engine composing scheduler, executor, and tokenizer.
//!
//! Uses real types from dependency crates where their APIs are compatible:
//! - `rvllm_executor` -- `Executor` trait, `ExecutorInput`, `SamplerOutput`, `ExecutorFactory`
//! - `rvllm_tokenizer` -- `Tokenizer` for encode/decode (via constructor)
//! - `rvllm_sequence` -- `Sequence`, `SequenceGroup`, `SequenceGroupMetadata`
//!
//! The engine defines its own `Scheduler` and `Executor` traits as internal
//! abstractions. `ExecutorAdapter` wraps the real async `rvllm_executor::Executor`
//! into the engine's sync trait.
//!
//! NOTE: `rvllm_scheduler::Scheduler` uses a different `SequenceGroup` type
//! (from rvllm-block-manager) and currently has API mismatches. Once those are
//! resolved, a `SchedulerAdapter` can bridge the real scheduler here.

use std::collections::HashMap;
use std::time::Instant;

use tracing::{debug, info};

use rvllm_config::EngineConfig;
use rvllm_core::prelude::{
    FinishReason, LLMError, LogProb, RequestId, RequestOutput, Result, SamplingParams, SequenceId,
    TokenId,
};
use rvllm_sequence::{Sequence, SequenceGroup, SequenceGroupMetadata};
use rvllm_tokenizer::Tokenizer;

use crate::beam_search::{top_k_from_logprobs, BeamSearchState};
use crate::output::{OutputProcessor, SequenceOutputState};

// ---------------------------------------------------------------------------
// Re-exports of real executor types from rvllm-executor
// ---------------------------------------------------------------------------

/// The real async executor trait from rvllm-executor.
pub use rvllm_executor::Executor as RealExecutorTrait;
/// The real executor configuration.
pub use rvllm_executor::ExecutorConfig;
/// Factory for creating executors based on config.
pub use rvllm_executor::ExecutorFactory;
/// The real executor input type.
pub use rvllm_executor::ExecutorInput as RealExecutorInput;
/// The real batch-level sampler output.
pub use rvllm_executor::SamplerOutput as RealSamplerOutput;

// ---------------------------------------------------------------------------
// Engine-level types
// ---------------------------------------------------------------------------

/// Output produced by a single scheduling step.
#[derive(Debug, Clone)]
pub struct SchedulerOutputs {
    /// Sequence groups scheduled for this iteration.
    pub scheduled_seq_groups: Vec<SequenceGroup>,
    /// Number of batched tokens across all scheduled groups.
    pub num_batched_tokens: usize,
    /// Whether any blocks were preempted.
    pub preempted: bool,
}

/// Input fed to the executor for a single forward pass.
#[derive(Debug, Clone)]
pub struct ExecutorInput {
    /// Per-group metadata (token ids, block tables, sampling params).
    pub seq_group_metadata: Vec<SequenceGroupMetadata>,
}

/// Per-sequence sampler output from one forward pass.
/// Engine-level granularity: one output per sequence.
/// Adapted from the batch-level `rvllm_executor::SamplerOutput`.
#[derive(Debug, Clone)]
pub struct SamplerOutput {
    pub seq_id: SequenceId,
    pub token_id: TokenId,
    pub logprob: LogProb,
    pub top_logprobs: Option<Vec<(TokenId, LogProb)>>,
}

// ---------------------------------------------------------------------------
// Trait definitions for Scheduler and Executor
// ---------------------------------------------------------------------------

/// Scheduler trait -- drives sequence admission, preemption, and block allocation.
pub trait Scheduler: Send {
    fn add_seq_group(&mut self, seq_group: SequenceGroup);
    fn abort_seq_group(&mut self, request_id: &RequestId);
    fn schedule(&mut self) -> SchedulerOutputs;
    fn has_unfinished_seqs(&self) -> bool;
    fn get_num_unfinished_seq_groups(&self) -> usize;
}

/// Executor trait -- runs the model forward pass and sampling.
/// This is the engine's own sync abstraction. Use `ExecutorAdapter` to wrap
/// the real async `rvllm_executor::Executor` trait.
pub trait Executor: Send {
    fn execute_model(&mut self, input: ExecutorInput) -> Result<Vec<SamplerOutput>>;
}

// ---------------------------------------------------------------------------
// ExecutorAdapter -- wraps real rvllm_executor::Executor (async) into sync
// ---------------------------------------------------------------------------

/// Adapter wrapping a boxed `rvllm_executor::Executor` (async trait) to
/// implement the engine's synchronous `Executor` trait.
///
/// Converts between:
/// - Engine's `ExecutorInput` (uses `rvllm_sequence::SequenceGroupMetadata`)
///   and real `rvllm_executor::ExecutorInput` (uses executor's stub `SequenceGroupMetadata`)
/// - Real `rvllm_executor::SamplerOutput` (batch-level: `Vec<u32>`, `Vec<f32>`)
///   and engine's per-sequence `SamplerOutput`
pub struct ExecutorAdapter {
    inner: Box<dyn RealExecutorTrait>,
    rt: tokio::runtime::Handle,
}

impl ExecutorAdapter {
    /// Wrap a real executor with a tokio runtime handle for async bridging.
    pub fn new(inner: Box<dyn RealExecutorTrait>, rt: tokio::runtime::Handle) -> Self {
        Self { inner, rt }
    }

    /// Create from an `ExecutorConfig` using the factory.
    pub fn from_config(config: ExecutorConfig, rt: tokio::runtime::Handle) -> Result<Self> {
        let inner = ExecutorFactory::create(config)
            .map_err(|e| LLMError::GpuError(format!("executor factory: {}", e)))?;
        Ok(Self { inner, rt })
    }

    /// Number of free GPU KV-cache blocks (from rank-0 worker).
    pub fn num_available_gpu_blocks(&self) -> usize {
        self.inner.num_available_gpu_blocks()
    }

    /// Number of free CPU KV-cache blocks (from rank-0 worker).
    pub fn num_available_cpu_blocks(&self) -> usize {
        self.inner.num_available_cpu_blocks()
    }
}

impl Executor for ExecutorAdapter {
    fn execute_model(&mut self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
        // Build a mapping from position -> seq_ids so we can attribute outputs
        let mut seq_id_map: Vec<SequenceId> = Vec::new();
        for meta in &input.seq_group_metadata {
            for &sid in meta.seq_data.keys() {
                seq_id_map.push(sid);
            }
        }

        // Convert engine's ExecutorInput -> real rvllm_executor::ExecutorInput
        let real_input = RealExecutorInput {
            seq_group_metadata_list: input
                .seq_group_metadata
                .iter()
                .map(|m| rvllm_executor::SequenceGroupMetadata {
                    request_id: m.request_id,
                    is_prompt: m.is_prompt,
                    seq_data: m.seq_data.clone(),
                    sampling_params: m.sampling_params.clone(),
                    block_tables: m.block_tables.clone(),
                })
                .collect(),
            blocks_to_swap_in: Vec::new(),
            blocks_to_swap_out: Vec::new(),
            blocks_to_copy: Vec::new(),
        };

        // Bridge async -> sync
        let real_outputs = self.rt.block_on(self.inner.execute_model(real_input))?;

        // Convert real SamplerOutput (batch-level) -> engine SamplerOutput (per-seq)
        let mut outputs = Vec::new();
        for real_out in &real_outputs {
            for (idx, (&tid, &lp)) in real_out
                .token_ids
                .iter()
                .zip(real_out.logprobs.iter())
                .enumerate()
            {
                // Map output position back to seq_id using our pre-built map
                let seq_id = seq_id_map
                    .get(idx)
                    .copied()
                    .unwrap_or(SequenceId(idx as u64));
                outputs.push(SamplerOutput {
                    seq_id,
                    token_id: tid,
                    logprob: lp,
                    top_logprobs: real_out.top_logprobs.get(idx).cloned(),
                });
            }
        }

        Ok(outputs)
    }
}

// ---------------------------------------------------------------------------
// EngineRequest -- internal bookkeeping per in-flight request
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct EngineRequest {
    #[allow(dead_code)]
    request_id: RequestId,
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    sampling_params: SamplingParams,
    seq_states: Vec<SequenceOutputState>,
    beam_search: Option<BeamSearchState>,
}

// ---------------------------------------------------------------------------
// LLMEngine
// ---------------------------------------------------------------------------

/// Synchronous inference engine that drives the scheduler/executor loop.
pub struct LLMEngine {
    config: EngineConfig,
    executor: Box<dyn Executor>,
    scheduler: Box<dyn Scheduler>,
    tokenizer: Tokenizer,
    requests: HashMap<RequestId, EngineRequest>,
    next_seq_id: u64,
}

impl LLMEngine {
    /// Create a new engine from its constituent parts.
    pub fn new(
        config: EngineConfig,
        executor: Box<dyn Executor>,
        scheduler: Box<dyn Scheduler>,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        info!("LLMEngine initializing");
        Ok(Self {
            config,
            executor,
            scheduler,
            tokenizer,
            requests: HashMap::new(),
            next_seq_id: 0,
        })
    }

    /// Create an engine wired to the real `rvllm_executor` crate.
    ///
    /// Uses `ExecutorAdapter` to wrap the async `rvllm_executor::Executor` trait
    /// into the engine's sync `Executor` trait.
    pub fn with_real_executor(
        config: EngineConfig,
        executor_config: ExecutorConfig,
        scheduler: Box<dyn Scheduler>,
        tokenizer: Tokenizer,
        rt: tokio::runtime::Handle,
    ) -> Result<Self> {
        let executor = ExecutorAdapter::from_config(executor_config, rt)?;
        Self::new(config, Box::new(executor), scheduler, tokenizer)
    }

    /// Submit a new generation request.
    ///
    /// Tokenizes the prompt, creates a `SequenceGroup`, and hands it to
    /// the scheduler.
    pub fn add_request(
        &mut self,
        request_id: RequestId,
        prompt: String,
        sampling_params: SamplingParams,
    ) -> Result<()> {
        info!(%request_id, prompt_len = prompt.len(), "adding request");

        let prompt_token_ids = self.tokenizer.encode(&prompt)?;
        debug!(%request_id, num_tokens = prompt_token_ids.len(), "prompt tokenized");

        if prompt_token_ids.is_empty() {
            return Err(LLMError::TokenizerError(
                "prompt produced zero tokens".into(),
            ));
        }

        let num_seqs = sampling_params.best_of.max(1);
        let mut seqs = Vec::with_capacity(num_seqs);
        let mut seq_states = Vec::with_capacity(num_seqs);

        for _ in 0..num_seqs {
            let seq_id = SequenceId(self.next_seq_id);
            self.next_seq_id += 1;
            seqs.push(Sequence::new(seq_id, prompt_token_ids.clone()));
            seq_states.push(SequenceOutputState::new());
        }
        let initial_seq_ids: Vec<SequenceId> = seqs.iter().map(|s| s.seq_id).collect();

        let seq_group = SequenceGroup::new(
            request_id,
            seqs,
            sampling_params.clone(),
            Instant::now(),
            prompt.clone(),
        );

        self.scheduler.add_seq_group(seq_group);

        self.requests.insert(
            request_id,
            EngineRequest {
                request_id,
                prompt,
                prompt_token_ids,
                beam_search: if sampling_params.use_beam_search {
                    Some(BeamSearchState::new(
                        request_id,
                        num_seqs,
                        sampling_params.max_tokens,
                        1.0,
                        false,
                        &initial_seq_ids,
                    ))
                } else {
                    None
                },
                sampling_params,
                seq_states,
            },
        );

        Ok(())
    }

    /// Abort a request, removing it from the scheduler and internal tracking.
    pub fn abort_request(&mut self, request_id: &RequestId) {
        info!(%request_id, "aborting request");
        self.scheduler.abort_seq_group(request_id);
        if let Some(req) = self.requests.get_mut(request_id) {
            for state in &mut req.seq_states {
                if state.finish_reason.is_none() {
                    state.finish_reason = Some(FinishReason::Abort);
                }
            }
        }
    }

    /// Run a single scheduling + execution step.
    ///
    /// Returns `RequestOutput` for any request that made progress (streaming)
    /// or that has finished.
    pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
        debug!("engine step begin");

        // 1. Schedule
        let sched_out = self.scheduler.schedule();
        debug!(
            num_groups = sched_out.scheduled_seq_groups.len(),
            num_tokens = sched_out.num_batched_tokens,
            "scheduler output"
        );

        if sched_out.scheduled_seq_groups.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Build executor input
        let input = self.build_executor_input(&sched_out);

        // 3. Execute model forward pass
        let sampler_outputs = self.executor.execute_model(input)?;
        debug!(num_outputs = sampler_outputs.len(), "executor returned");

        // 4. Process outputs: update states, check stop, detokenize
        let mut results = Vec::new();
        let eos = self.tokenizer.eos_token_id();

        // Index sampler outputs by seq_id for fast lookup
        let output_map: HashMap<SequenceId, &SamplerOutput> =
            sampler_outputs.iter().map(|o| (o.seq_id, o)).collect();

        for group in &sched_out.scheduled_seq_groups {
            let request_id = group.request_id;
            let req = match self.requests.get_mut(&request_id) {
                Some(r) => r,
                None => continue,
            };

            let output = if req.sampling_params.use_beam_search {
                Self::process_beam_search_group(&self.tokenizer, req, &output_map, eos)
            } else {
                for (seq_idx, seq) in group.get_seqs().iter().enumerate() {
                    if seq.is_finished() {
                        continue;
                    }

                    if let Some(sampled) = output_map.get(&seq.seq_id) {
                        let decoded = self
                            .tokenizer
                            .decode(&[sampled.token_id])
                            .unwrap_or_default();

                        if let Some(state) = req.seq_states.get_mut(seq_idx) {
                            OutputProcessor::process_token(
                                state,
                                sampled.token_id,
                                sampled.logprob,
                                sampled.top_logprobs.clone(),
                                &decoded,
                                &req.sampling_params,
                                eos,
                            );
                        }
                    }
                }

                let mut output = OutputProcessor::build_request_output(
                    request_id,
                    &req.prompt,
                    &req.prompt_token_ids,
                    &req.seq_states,
                );

                if output.finished
                    && req.sampling_params.best_of > 1
                    && !req.sampling_params.use_beam_search
                {
                    output = crate::best_of_n::build_best_of_n_output(output, &req.seq_states);
                }

                output
            };

            results.push(output);
        }

        // Clean up finished requests -- remove from both internal map and scheduler
        let finished_ids: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, req)| {
                if let Some(beam) = &req.beam_search {
                    beam.is_finished()
                } else {
                    req.seq_states.iter().all(|s| s.is_finished())
                }
            })
            .map(|(&id, _)| id)
            .collect();

        for id in &finished_ids {
            self.requests.remove(id);
            self.scheduler.abort_seq_group(id);
        }

        debug!(num_outputs = results.len(), "step complete");
        Ok(results)
    }

    /// Blocking loop: call `step()` until all requests are finished.
    pub fn run(&mut self) -> Result<Vec<RequestOutput>> {
        info!("engine run loop starting");
        let mut all_outputs = Vec::new();

        while self.has_unfinished() {
            let step_outputs = self.step()?;
            for output in step_outputs {
                if output.finished {
                    all_outputs.push(output);
                }
            }
        }

        info!(
            num_completed = all_outputs.len(),
            "engine run loop finished"
        );
        Ok(all_outputs)
    }

    /// Whether there are any unfinished requests.
    pub fn has_unfinished(&self) -> bool {
        self.scheduler.has_unfinished_seqs() || !self.requests.is_empty()
    }

    /// Access the engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    // -- private helpers --

    fn process_beam_search_group(
        tokenizer: &Tokenizer,
        req: &mut EngineRequest,
        output_map: &HashMap<SequenceId, &SamplerOutput>,
        eos: Option<TokenId>,
    ) -> RequestOutput {
        let beam = req
            .beam_search
            .as_mut()
            .expect("beam search state missing for beam request");

        let mut expansions = HashMap::new();
        for active in &beam.active_beams {
            let Some(sampled) = output_map.get(&active.seq_id) else {
                continue;
            };

            let candidates = sampled
                .top_logprobs
                .as_ref()
                .filter(|top| !top.is_empty())
                .map(|top| top_k_from_logprobs(top, beam.num_beams))
                .unwrap_or_else(|| vec![(sampled.token_id, sampled.logprob)]);

            let decoded_candidates = candidates
                .into_iter()
                .map(|(token_id, logprob)| {
                    let decoded = tokenizer.decode(&[token_id]).unwrap_or_default();
                    (token_id, logprob, decoded, eos == Some(token_id))
                })
                .collect::<Vec<_>>();
            expansions.insert(active.seq_id, decoded_candidates);
        }

        let step_result = beam.step(&expansions);
        let mut recycled = step_result.seqs_to_free.into_iter();
        for op in step_result.fork_ops {
            let seq_id = recycled.next().unwrap_or(op.parent_seq_id);
            beam.set_beam_seq_id(op.new_beam_idx, seq_id);
        }

        beam.build_output(&req.prompt, &req.prompt_token_ids, 1)
    }

    fn build_executor_input(&self, sched_out: &SchedulerOutputs) -> ExecutorInput {
        let mut metadata = Vec::with_capacity(sched_out.scheduled_seq_groups.len());

        for group in &sched_out.scheduled_seq_groups {
            let Some(req) = self.requests.get(&group.request_id) else {
                continue;
            };
            let is_prompt = if let Some(beam) = &req.beam_search {
                beam.active_beams.iter().all(|b| b.token_ids.is_empty())
            } else {
                group.get_seqs().iter().any(|s| s.get_output_len() == 0)
            };

            let mut seq_data = HashMap::new();
            let block_tables = HashMap::new();

            if let Some(beam) = &req.beam_search {
                for active in &beam.active_beams {
                    seq_data.insert(
                        active.seq_id,
                        rvllm_sequence::SequenceData {
                            prompt_token_ids: req.prompt_token_ids.clone(),
                            output_token_ids: active.token_ids.clone(),
                            cumulative_logprob: active.cumulative_logprob,
                        },
                    );
                }
            } else {
                for seq in group.get_seqs() {
                    if seq.is_finished() {
                        continue;
                    }
                    seq_data.insert(
                        seq.seq_id,
                        rvllm_sequence::SequenceData {
                            prompt_token_ids: seq.prompt_token_ids.clone(),
                            output_token_ids: seq.output_token_ids.clone(),
                            cumulative_logprob: seq.cumulative_logprob,
                        },
                    );
                }
            }

            let mut sampling_params = group.sampling_params.clone();
            if sampling_params.use_beam_search {
                sampling_params.logprobs = Some(
                    sampling_params
                        .logprobs
                        .unwrap_or(0)
                        .max(sampling_params.best_of),
                );
            }

            metadata.push(SequenceGroupMetadata {
                request_id: group.request_id,
                is_prompt,
                seq_data,
                sampling_params,
                block_tables,
            });
        }

        ExecutorInput {
            seq_group_metadata: metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer as HfTokenizer;

    // -- Mock scheduler --

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
            // Remove finished groups
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

    // -- Mock executor that always returns a fixed token --

    struct MockExecutor {
        token_id: TokenId,
        logprob: LogProb,
        calls: usize,
        max_calls: usize,
    }

    impl MockExecutor {
        fn new(token_id: TokenId, max_calls: usize) -> Self {
            Self {
                token_id,
                logprob: -0.5,
                calls: 0,
                max_calls,
            }
        }
    }

    struct BeamExecutor;

    impl Executor for BeamExecutor {
        fn execute_model(&mut self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
            let mut outputs = Vec::new();
            for meta in &input.seq_group_metadata {
                for (&seq_id, seq_data) in &meta.seq_data {
                    let (token_id, logprob, top_logprobs) =
                        match seq_data.output_token_ids.as_slice() {
                            [] if seq_id == SequenceId(0) => {
                                (1, -0.1, Some(vec![(1, -0.1), (2, -0.2)]))
                            }
                            [] => (3, -5.0, Some(vec![(3, -5.0), (4, -6.0)])),
                            [1] => (0, -0.1, Some(vec![(0, -0.1)])),
                            [2] => (0, -1.0, Some(vec![(0, -1.0)])),
                            _ => (0, -10.0, Some(vec![(0, -10.0)])),
                        };
                    outputs.push(SamplerOutput {
                        seq_id,
                        token_id,
                        logprob,
                        top_logprobs,
                    });
                }
            }
            Ok(outputs)
        }
    }

    impl Executor for MockExecutor {
        fn execute_model(&mut self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
            self.calls += 1;
            let mut outputs = Vec::new();
            for meta in &input.seq_group_metadata {
                for &seq_id in meta.seq_data.keys() {
                    // After max_calls, return EOS-like token (id=0) to terminate
                    let tid = if self.calls >= self.max_calls {
                        0
                    } else {
                        self.token_id
                    };
                    outputs.push(SamplerOutput {
                        seq_id,
                        token_id: tid,
                        logprob: self.logprob,
                        top_logprobs: None,
                    });
                }
            }
            Ok(outputs)
        }
    }

    fn make_test_tokenizer() -> Tokenizer {
        let mut vocab = std::collections::HashMap::new();
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

        Tokenizer::from_file(std::path::Path::new("/dev/null")).unwrap_or_else(|_| {
            // Construct via the public API path -- we use a workaround
            // since from_hf_tokenizer is private. Write to temp file.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("tokenizer.json");
            hf.save(&path, false).unwrap();
            Tokenizer::from_file(&path).unwrap()
        })
    }

    fn make_engine(max_executor_calls: usize) -> LLMEngine {
        let config = EngineConfig::default();
        let tokenizer = make_test_tokenizer();
        let scheduler = Box::new(MockScheduler::new());
        let executor = Box::new(MockExecutor::new(1, max_executor_calls));
        LLMEngine::new(config, executor, scheduler, tokenizer).unwrap()
    }

    fn make_beam_engine() -> LLMEngine {
        let config = EngineConfig::default();
        let tokenizer = make_test_tokenizer();
        let scheduler = Box::new(MockScheduler::new());
        let executor = Box::new(BeamExecutor);
        LLMEngine::new(config, executor, scheduler, tokenizer).unwrap()
    }

    #[test]
    fn add_request_tokenizes_prompt() {
        let mut engine = make_engine(5);
        let result =
            engine.add_request(RequestId(1), "hello".to_string(), SamplingParams::default());
        assert!(result.is_ok());
        assert!(engine.has_unfinished());
        assert!(engine.requests.contains_key(&RequestId(1)));
    }

    #[test]
    fn abort_request_removes() {
        let mut engine = make_engine(5);
        engine
            .add_request(RequestId(1), "hello".to_string(), SamplingParams::default())
            .unwrap();
        engine.abort_request(&RequestId(1));
        // States should be marked as aborted
        let req = engine.requests.get(&RequestId(1)).unwrap();
        assert!(req
            .seq_states
            .iter()
            .all(|s| s.finish_reason == Some(FinishReason::Abort)));
    }

    #[test]
    fn step_produces_output() {
        let mut engine = make_engine(5);
        engine
            .add_request(RequestId(1), "hello".to_string(), SamplingParams::default())
            .unwrap();

        let outputs = engine.step().unwrap();
        assert!(!outputs.is_empty());
        assert_eq!(outputs[0].request_id, RequestId(1));
    }

    #[test]
    fn run_completes_with_max_tokens() {
        let mut engine = make_engine(100);
        let mut params = SamplingParams::default();
        params.max_tokens = 3;
        engine
            .add_request(RequestId(1), "hello".to_string(), params)
            .unwrap();

        let outputs = engine.run().unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].finished);
    }

    #[test]
    fn has_unfinished_false_when_empty() {
        let engine = make_engine(5);
        assert!(!engine.has_unfinished());
    }

    #[test]
    fn build_executor_input_from_scheduler_outputs() {
        let mut engine = make_engine(5);
        engine
            .add_request(RequestId(0), "hello".to_string(), SamplingParams::default())
            .unwrap();
        let seq = Sequence::new(SequenceId(0), vec![1, 2, 3]);
        let group = SequenceGroup::new(
            RequestId(0),
            vec![seq],
            SamplingParams::default(),
            Instant::now(),
            "test".into(),
        );
        let sched_out = SchedulerOutputs {
            scheduled_seq_groups: vec![group],
            num_batched_tokens: 3,
            preempted: false,
        };
        let input = engine.build_executor_input(&sched_out);
        assert_eq!(input.seq_group_metadata.len(), 1);
        assert_eq!(input.seq_group_metadata[0].request_id, RequestId(0));
        assert!(input.seq_group_metadata[0].is_prompt);
    }

    #[test]
    fn run_beam_search_selects_single_best_output() {
        let mut engine = make_beam_engine();
        let mut params = SamplingParams::default();
        params.max_tokens = 2;
        params.best_of = 2;
        params.use_beam_search = true;
        engine
            .add_request(RequestId(7), "hello".to_string(), params)
            .unwrap();

        let outputs = engine.run().unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].finished);
        assert_eq!(outputs[0].outputs.len(), 1);
        assert_eq!(outputs[0].outputs[0].token_ids, vec![1, 0]);
    }

    // -- Test ExecutorAdapter type-level wiring --

    #[test]
    fn executor_adapter_from_config() {
        // Verify the adapter can be constructed from an ExecutorConfig.
        // This exercises the type-level wiring without requiring a GPU.
        let config = ExecutorConfig {
            num_gpus: 1,
            model_name: "test-model".into(),
            ..ExecutorConfig::default()
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let adapter = ExecutorAdapter::from_config(config, rt.handle().clone());
        assert!(adapter.is_ok());

        let adapter = adapter.unwrap();
        assert!(adapter.num_available_gpu_blocks() > 0);
        assert!(adapter.num_available_cpu_blocks() > 0);
    }
}
