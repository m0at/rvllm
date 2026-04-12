use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use rvllm_core::prelude::{
    CompletionOutput, FinishReason, RequestId, RequestOutput, SamplingParams, SequenceId, TokenId,
};
use rvllm_tokenizer::Tokenizer;

use crate::scheduler::{BlockManagerOps, Scheduler};
use crate::types::{ForwardOutput, SchedulerOutput, StepDiff, V2RequestOutput};
use crate::worker::Worker;

#[allow(dead_code)]
struct EngineRequest {
    request_id: RequestId,
    seq_id: SequenceId,
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    sampling_params: SamplingParams,
    output_token_ids: Vec<TokenId>,
    finished: bool,
    finish_reason: Option<FinishReason>,
}

pub struct StepPending {
    sched_out: SchedulerOutput,
}

pub struct Engine<B: BlockManagerOps> {
    scheduler: Scheduler<B>,
    worker: Worker,
    tokenizer: Tokenizer,
    requests: HashMap<RequestId, EngineRequest>,
    next_request_id: AtomicU64,
    eos_token_id: Option<TokenId>,
}

impl<B: BlockManagerOps> Engine<B> {
    pub fn new(scheduler: Scheduler<B>, worker: Worker, tokenizer: Tokenizer) -> Self {
        let eos_token_id = tokenizer.eos_token_id();
        Self {
            scheduler,
            worker,
            tokenizer,
            requests: HashMap::new(),
            next_request_id: AtomicU64::new(1),
            eos_token_id,
        }
    }

    pub fn add_request(
        &mut self,
        prompt: String,
        sampling_params: SamplingParams,
    ) -> Result<RequestId, EngineError> {
        let prompt_token_ids = self
            .tokenizer
            .encode(&prompt)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;

        if prompt_token_ids.is_empty() {
            return Err(EngineError::Tokenizer(
                "prompt produced zero tokens".into(),
            ));
        }

        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let seq_id = self.scheduler.add_request(
            request_id,
            prompt_token_ids.clone(),
            sampling_params.clone(),
        );

        self.requests.insert(
            request_id,
            EngineRequest {
                request_id,
                seq_id,
                prompt,
                prompt_token_ids,
                sampling_params,
                output_token_ids: Vec::new(),
                finished: false,
                finish_reason: None,
            },
        );
        Ok(request_id)
    }

    pub fn abort_request(&mut self, request_id: RequestId) {
        self.scheduler.abort_request(request_id);
        if let Some(req) = self.requests.get_mut(&request_id) {
            req.finished = true;
            req.finish_reason = Some(FinishReason::Abort);
        }
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work() || self.requests.values().any(|r| !r.finished)
    }

    pub fn step(&mut self) -> Result<Vec<V2RequestOutput>, EngineError> {
        let sched_out = self.scheduler.schedule();
        if sched_out.diff.is_empty() {
            return Ok(Vec::new());
        }

        let fwd_output = self
            .worker
            .step(&sched_out.diff)
            .map_err(|e| EngineError::Worker(e.to_string()))?;
        Ok(self.process_forward_output(&sched_out, &fwd_output))
    }

    pub fn step_launch(&mut self) -> Result<Option<StepPending>, EngineError> {
        let sched_out = self.scheduler.schedule();
        if sched_out.diff.is_empty() {
            return Ok(None);
        }

        self.worker
            .step_launch(&sched_out.diff)
            .map_err(|e| EngineError::Worker(e.to_string()))?;

        Ok(Some(StepPending { sched_out }))
    }

    pub fn step_collect(
        &mut self,
        pending: Option<StepPending>,
    ) -> Result<Vec<V2RequestOutput>, EngineError> {
        let pending = match pending {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };

        let fwd_output = self
            .worker
            .step_collect()
            .map_err(|e| EngineError::Worker(e.to_string()))?;
        Ok(self.process_forward_output(&pending.sched_out, &fwd_output))
    }

    fn process_forward_output(
        &mut self,
        sched_out: &SchedulerOutput,
        fwd_output: &ForwardOutput,
    ) -> Vec<V2RequestOutput> {
        let diff = &sched_out.diff;
        let mut step_results: Vec<(SequenceId, TokenId, bool)> =
            Vec::with_capacity(diff.added.len() + diff.continued.len());
        let mut request_outputs: Vec<V2RequestOutput> = Vec::new();
        let mut token_idx = 0;

        // Process added (prefill) requests
        for added in &diff.added {
            let is_last_chunk = added.token_chunk.end >= added.prompt_token_ids.len();
            if token_idx < fwd_output.token_ids.len() {
                let token_id = fwd_output.token_ids[token_idx];
                if is_last_chunk {
                    let finished = self.check_finish(added.request_id, token_id);
                    let reason = if finished {
                        Some(self.determine_finish_reason(added.request_id, token_id))
                    } else {
                        None
                    };
                    step_results.push((added.seq_id, token_id, finished));
                    if let Some(req) = self.requests.get_mut(&added.request_id) {
                        req.output_token_ids.push(token_id);
                        if finished {
                            req.finished = true;
                            req.finish_reason = reason;
                        }
                    }
                } else {
                    step_results.push((added.seq_id, 0, false));
                }
            }
            token_idx += 1;
        }

        // Process continued (decode) requests
        for cont in &diff.continued {
            if token_idx < fwd_output.token_ids.len() {
                let token_id = fwd_output.token_ids[token_idx];
                let finished = self.check_finish(cont.request_id, token_id);
                let reason = if finished {
                    Some(self.determine_finish_reason(cont.request_id, token_id))
                } else {
                    None
                };
                step_results.push((cont.seq_id, token_id, finished));
                if let Some(req) = self.requests.get_mut(&cont.request_id) {
                    req.output_token_ids.push(token_id);
                    if finished {
                        req.finished = true;
                        req.finish_reason = reason;
                    }
                }
            }
            token_idx += 1;
        }

        self.scheduler.process_step_result(&step_results);
        self.build_request_outputs(diff, fwd_output, &mut request_outputs);
        self.cleanup_finished();

        request_outputs
    }

    fn build_request_outputs(
        &self,
        diff: &StepDiff,
        fwd_output: &ForwardOutput,
        outputs: &mut Vec<V2RequestOutput>,
    ) {
        let mut token_idx = 0;

        for added in &diff.added {
            let is_last_chunk = added.token_chunk.end >= added.prompt_token_ids.len();
            let logprob = fwd_output.logprobs.get(token_idx).copied().unwrap_or(0.0);
            token_idx += 1;

            if !is_last_chunk {
                continue;
            }

            if let Some(req) = self.requests.get(&added.request_id) {
                let output_text = self.decode_output_tokens(req);
                outputs.push(V2RequestOutput {
                    request_id: added.request_id,
                    output_text,
                    output_token_ids: req.output_token_ids.clone(),
                    finished: req.finished,
                    finish_reason: req.finish_reason,
                    logprobs: vec![logprob],
                });
            }
        }

        for cont in &diff.continued {
            let logprob = fwd_output.logprobs.get(token_idx).copied().unwrap_or(0.0);
            token_idx += 1;

            if let Some(req) = self.requests.get(&cont.request_id) {
                let output_text = self.decode_output_tokens(req);
                outputs.push(V2RequestOutput {
                    request_id: cont.request_id,
                    output_text,
                    output_token_ids: req.output_token_ids.clone(),
                    finished: req.finished,
                    finish_reason: req.finish_reason,
                    logprobs: vec![logprob],
                });
            }
        }
    }

    fn decode_output_tokens(&self, req: &EngineRequest) -> String {
        if req.output_token_ids.is_empty() {
            return String::new();
        }
        if req.finished || !req.sampling_params.stop_strings.is_empty() {
            self.tokenizer
                .decode(&req.output_token_ids)
                .unwrap_or_default()
        } else {
            String::new()
        }
    }

    fn cleanup_finished(&mut self) {
        let finished_ids: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, req)| req.finished)
            .map(|(&id, _)| id)
            .collect();
        for id in &finished_ids {
            self.requests.remove(id);
        }
    }

    fn check_finish(&self, request_id: RequestId, token_id: TokenId) -> bool {
        let req = match self.requests.get(&request_id) {
            Some(r) => r,
            None => return false,
        };

        if let Some(eos) = self.eos_token_id {
            if token_id == eos && !req.sampling_params.ignore_eos {
                return true;
            }
        }

        let new_output_len = req.output_token_ids.len() + 1;
        if new_output_len >= req.sampling_params.max_tokens {
            return true;
        }

        if !req.sampling_params.stop_strings.is_empty() {
            let mut check_ids = req.output_token_ids.clone();
            check_ids.push(token_id);
            if let Ok(text) = self.tokenizer.decode(&check_ids) {
                for stop in &req.sampling_params.stop_strings {
                    if text.contains(stop.as_str()) {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn determine_finish_reason(&self, request_id: RequestId, token_id: TokenId) -> FinishReason {
        let req = match self.requests.get(&request_id) {
            Some(r) => r,
            None => return FinishReason::Stop,
        };

        if let Some(eos) = self.eos_token_id {
            if token_id == eos {
                return FinishReason::Stop;
            }
        }

        let new_output_len = req.output_token_ids.len() + 1;
        if new_output_len >= req.sampling_params.max_tokens {
            return FinishReason::Length;
        }

        FinishReason::Stop
    }

    /// Convert a V2RequestOutput to the core RequestOutput for API compatibility.
    pub fn to_core_output(&self, v2out: &V2RequestOutput) -> Option<RequestOutput> {
        // For finished requests already cleaned up, reconstruct from the output itself
        let prompt = String::new();
        let prompt_token_ids = Vec::new();

        Some(RequestOutput {
            request_id: v2out.request_id,
            prompt,
            prompt_token_ids,
            prompt_logprobs: None,
            outputs: vec![CompletionOutput {
                index: 0,
                text: v2out.output_text.clone(),
                token_ids: v2out.output_token_ids.clone(),
                cumulative_logprob: v2out.logprobs.iter().sum(),
                logprobs: None,
                finish_reason: v2out.finish_reason,
            }],
            finished: v2out.finished,
        })
    }

    /// Synchronize the GPU compute stream. Blocks until all enqueued work completes.
    pub fn sync(&self) -> Result<(), EngineError> {
        self.worker
            .sync()
            .map_err(|e| EngineError::Worker(e.to_string()))
    }

    pub fn num_active_requests(&self) -> usize {
        self.requests.len()
    }

    pub fn request_id_counter(&self) -> &AtomicU64 {
        &self.next_request_id
    }
}

#[derive(Debug)]
pub enum EngineError {
    Tokenizer(String),
    Worker(String),
    Scheduler(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Tokenizer(msg) => write!(f, "tokenizer error: {msg}"),
            EngineError::Worker(msg) => write!(f, "worker error: {msg}"),
            EngineError::Scheduler(msg) => write!(f, "scheduler error: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}
