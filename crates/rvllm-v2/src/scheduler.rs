use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::types::{
    AddedRequest, BlockId, BlockOps, ContinuedRequest, RequestId, SamplingParams, SchedulerOutput,
    SequenceId, StepDiff, TokenId,
};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_prefill_chunk: usize,
    pub preemption_mode: PreemptionMode,
    pub block_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_prefill_chunk: 0,
            preemption_mode: PreemptionMode::Recompute,
            block_size: 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionMode {
    Swap,
    Recompute,
}

#[derive(Debug, Clone)]
struct InternalRequest {
    request_id: RequestId,
    seq_id: SequenceId,
    prompt_token_ids: Vec<TokenId>,
    output_token_ids: Vec<TokenId>,
    sampling_params: SamplingParams,
    arrival_time: Instant,
    num_prompt_computed: usize,
    was_running: bool,
    last_new_token: Option<TokenId>,
    finished: bool,
}

impl InternalRequest {
    fn seq_len(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    fn prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    fn remaining_prefill(&self) -> usize {
        self.prompt_len().saturating_sub(self.num_prompt_computed)
    }

    fn is_prefilling(&self) -> bool {
        self.remaining_prefill() > 0
    }

    fn blocks_needed(&self, block_size: usize) -> usize {
        let total = self.seq_len();
        (total + block_size - 1) / block_size
    }
}

pub trait BlockManagerOps {
    fn allocate(&mut self, seq_id: SequenceId, num_tokens: usize) -> Vec<BlockId>;
    fn allocate_incremental(&mut self, seq_id: SequenceId, new_tokens: usize) -> Vec<BlockId>;
    fn free(&mut self, seq_id: SequenceId);
    fn get_block_table(&self, seq_id: SequenceId) -> Option<&[BlockId]>;
    fn get_block_table_update(&self, seq_id: SequenceId) -> Option<Vec<BlockId>>;
    fn mark_table_sent(&mut self, seq_id: SequenceId);
    fn cow_if_needed(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)>;
    fn swap_out(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)>;
    fn swap_in(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)>;
    fn can_allocate(&self, num_blocks: usize) -> bool;
    fn above_watermark(&self) -> bool;
}

pub struct Scheduler<B: BlockManagerOps> {
    config: SchedulerConfig,
    block_manager: B,
    waiting: VecDeque<RequestId>,
    running: Vec<RequestId>,
    swapped: VecDeque<RequestId>,
    requests: HashMap<RequestId, InternalRequest>,
    next_seq_id: u64,
}

const MAX_PREEMPTIONS: usize = 4;

impl<B: BlockManagerOps> Scheduler<B> {
    pub fn new(config: SchedulerConfig, block_manager: B) -> Self {
        Self {
            config,
            block_manager,
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: VecDeque::new(),
            requests: HashMap::new(),
            next_seq_id: 1,
        }
    }

    pub fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<TokenId>,
        sampling_params: SamplingParams,
    ) -> SequenceId {
        let seq_id = SequenceId(self.next_seq_id);
        self.next_seq_id += 1;

        let req = InternalRequest {
            request_id,
            seq_id,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            sampling_params,
            arrival_time: Instant::now(),
            num_prompt_computed: 0,
            was_running: false,
            last_new_token: None,
            finished: false,
        };

        self.requests.insert(request_id, req);
        self.waiting.push_back(request_id);
        seq_id
    }

    pub fn abort_request(&mut self, request_id: RequestId) {
        self.waiting.retain(|id| *id != request_id);
        self.swapped.retain(|id| *id != request_id);

        if let Some(pos) = self.running.iter().position(|id| *id == request_id) {
            self.running.swap_remove(pos);
            if let Some(req) = self.requests.get(&request_id) {
                self.block_manager.free(req.seq_id);
            }
        }

        self.requests.remove(&request_id);
    }

    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut diff = StepDiff {
            added: Vec::new(),
            removed: Vec::new(),
            continued: Vec::new(),
            block_ops: BlockOps::default(),
        };
        let mut num_batched_tokens = 0usize;

        self.retire_finished(&mut diff);
        self.preempt_if_needed(&mut diff);
        self.try_swap_in(&mut diff);
        self.admit_waiting(&mut diff);
        self.build_continued(&mut diff, &mut num_batched_tokens);
        self.cow_for_running(&mut diff);

        for rid in &self.running {
            if let Some(req) = self.requests.get_mut(rid) {
                req.was_running = true;
                req.last_new_token = None;
            }
        }

        SchedulerOutput {
            diff,
            num_running: self.running.len(),
            num_waiting: self.waiting.len() + self.swapped.len(),
            total_batched_tokens: num_batched_tokens,
        }
    }

    pub fn process_step_result(&mut self, results: &[(SequenceId, TokenId, bool)]) {
        for &(seq_id, token_id, finished) in results {
            let req = match self.requests.values_mut().find(|r| r.seq_id == seq_id) {
                Some(r) => r,
                None => continue,
            };

            if req.is_prefilling() {
                req.num_prompt_computed = req.num_prompt_computed.saturating_add(1);
                if finished {
                    req.finished = true;
                }
                continue;
            }

            req.output_token_ids.push(token_id);
            req.last_new_token = Some(token_id);

            if finished {
                req.finished = true;
            }

            self.block_manager.allocate_incremental(seq_id, 1);
        }
    }

    fn retire_finished(&mut self, diff: &mut StepDiff) {
        let mut still_running = Vec::with_capacity(self.running.len());

        for rid in std::mem::take(&mut self.running) {
            let finished = self.requests.get(&rid).map_or(true, |r| r.finished);
            if finished {
                diff.removed.push(rid);
                if let Some(req) = self.requests.get(&rid) {
                    self.block_manager.free(req.seq_id);
                }
                self.requests.remove(&rid);
            } else {
                still_running.push(rid);
            }
        }

        self.running = still_running;
    }

    fn preempt_if_needed(&mut self, diff: &mut StepDiff) {
        let mut preempted = 0usize;

        while !self.running.is_empty()
            && !self.block_manager.above_watermark()
            && preempted < MAX_PREEMPTIONS
        {
            let victim_id = self.running.pop().unwrap();

            let seq_id = match self.requests.get(&victim_id) {
                Some(r) => r.seq_id,
                None => continue,
            };

            match self.config.preemption_mode {
                PreemptionMode::Swap => {
                    let pairs = self.block_manager.swap_out(seq_id);
                    diff.block_ops.swap_out.extend(pairs);
                    self.swapped.push_back(victim_id);
                }
                PreemptionMode::Recompute => {
                    self.block_manager.free(seq_id);
                    if let Some(req) = self.requests.get_mut(&victim_id) {
                        req.num_prompt_computed = 0;
                        req.output_token_ids.clear();
                    }
                    self.waiting.push_back(victim_id);
                }
            }

            if self.requests.get(&victim_id).is_some_and(|r| r.was_running) {
                diff.removed.push(victim_id);
            }

            preempted += 1;
        }
    }

    fn try_swap_in(&mut self, diff: &mut StepDiff) {
        let mut still_swapped = VecDeque::new();

        while let Some(rid) = self.swapped.pop_front() {
            if self.running.len() >= self.config.max_num_seqs {
                still_swapped.push_back(rid);
                continue;
            }

            let req = match self.requests.get(&rid) {
                Some(r) => r,
                None => continue,
            };

            let needed = req.blocks_needed(self.config.block_size);
            if self.block_manager.can_allocate(needed) {
                let pairs = self.block_manager.swap_in(req.seq_id);
                diff.block_ops.swap_in.extend(pairs);

                let block_table = self
                    .block_manager
                    .get_block_table(req.seq_id)
                    .map(|t| t.to_vec())
                    .unwrap_or_default();

                let chunk_range = if req.is_prefilling() {
                    let chunk_size = self.chunk_size(req);
                    let start = req.num_prompt_computed;
                    let end = (start + chunk_size).min(req.prompt_len());
                    start..end
                } else {
                    0..0
                };

                diff.added.push(AddedRequest {
                    request_id: req.request_id,
                    seq_id: req.seq_id,
                    prompt_token_ids: req.prompt_token_ids.clone(),
                    sampling_params: req.sampling_params.clone(),
                    block_table,
                    is_prefill: req.is_prefilling(),
                    token_chunk: chunk_range,
                });

                self.running.push(rid);
            } else {
                still_swapped.push_back(rid);
                break;
            }
        }

        while let Some(g) = self.swapped.pop_front() {
            still_swapped.push_back(g);
        }
        self.swapped = still_swapped;
    }

    fn admit_waiting(&mut self, diff: &mut StepDiff) {
        let mut still_waiting = VecDeque::new();

        while let Some(rid) = self.waiting.pop_front() {
            if self.running.len() >= self.config.max_num_seqs {
                still_waiting.push_back(rid);
                continue;
            }

            let req = match self.requests.get(&rid) {
                Some(r) => r,
                None => continue,
            };

            let chunk_size = self.chunk_size(req);
            let needed_tokens = if req.is_prefilling() {
                (req.num_prompt_computed + chunk_size).min(req.prompt_len())
            } else {
                req.seq_len()
            };
            let needed_blocks =
                (needed_tokens + self.config.block_size - 1) / self.config.block_size;

            if !self.block_manager.can_allocate(needed_blocks) {
                still_waiting.push_back(rid);
                continue;
            }

            let block_table = self.block_manager.allocate(req.seq_id, needed_tokens);
            let start = req.num_prompt_computed;
            let end = (start + chunk_size).min(req.prompt_len());

            diff.added.push(AddedRequest {
                request_id: req.request_id,
                seq_id: req.seq_id,
                prompt_token_ids: req.prompt_token_ids.clone(),
                sampling_params: req.sampling_params.clone(),
                block_table,
                is_prefill: true,
                token_chunk: start..end,
            });

            if let Some(req_mut) = self.requests.get_mut(&rid) {
                req_mut.num_prompt_computed = end;
            }

            self.running.push(rid);
        }

        self.waiting = still_waiting;
    }

    fn build_continued(&mut self, diff: &mut StepDiff, num_batched_tokens: &mut usize) {
        for added in &diff.added {
            let chunk_len = added.token_chunk.end - added.token_chunk.start;
            if chunk_len > 0 {
                *num_batched_tokens += chunk_len;
            } else {
                *num_batched_tokens += 1;
            }
        }

        let added_ids: Vec<RequestId> = diff.added.iter().map(|a| a.request_id).collect();

        for rid in &self.running {
            if added_ids.contains(rid) {
                continue;
            }

            let req = match self.requests.get(rid) {
                Some(r) => r,
                None => continue,
            };

            if req.is_prefilling() {
                let chunk_size = self.chunk_size(req);
                let start = req.num_prompt_computed;
                let end = (start + chunk_size).min(req.prompt_len());

                let block_table = self
                    .block_manager
                    .get_block_table(req.seq_id)
                    .map(|t| t.to_vec())
                    .unwrap_or_default();

                let new_end_tokens = end;
                let current_blocks = block_table.len();
                let needed_blocks =
                    (new_end_tokens + self.config.block_size - 1) / self.config.block_size;
                if needed_blocks > current_blocks {
                    self.block_manager
                        .allocate_incremental(req.seq_id, needed_blocks - current_blocks);
                }

                let updated_table = self
                    .block_manager
                    .get_block_table(req.seq_id)
                    .map(|t| t.to_vec())
                    .unwrap_or_default();

                diff.added.push(AddedRequest {
                    request_id: req.request_id,
                    seq_id: req.seq_id,
                    prompt_token_ids: req.prompt_token_ids.clone(),
                    sampling_params: req.sampling_params.clone(),
                    block_table: updated_table,
                    is_prefill: true,
                    token_chunk: start..end,
                });

                if let Some(req_mut) = self.requests.get_mut(rid) {
                    req_mut.num_prompt_computed = end;
                }

                *num_batched_tokens += end - start;
                continue;
            }

            let new_token = match req.last_new_token {
                Some(t) => t,
                None => 0,
            };

            let block_table_update = self.block_manager.get_block_table_update(req.seq_id);

            diff.continued.push(ContinuedRequest {
                request_id: req.request_id,
                seq_id: req.seq_id,
                new_token_id: new_token,
                block_table_update,
            });

            self.block_manager.mark_table_sent(req.seq_id);
            *num_batched_tokens += 1;
        }
    }

    fn cow_for_running(&mut self, diff: &mut StepDiff) {
        for rid in &self.running {
            let seq_id = match self.requests.get(rid) {
                Some(r) => r.seq_id,
                None => continue,
            };

            let copies = self.block_manager.cow_if_needed(seq_id);
            diff.block_ops.copies.extend(copies);
        }
    }

    fn chunk_size(&self, req: &InternalRequest) -> usize {
        let remaining = req.remaining_prefill();
        if remaining == 0 {
            return 1;
        }
        if self.config.max_prefill_chunk > 0 {
            remaining.min(self.config.max_prefill_chunk)
        } else {
            remaining
        }
    }
}
