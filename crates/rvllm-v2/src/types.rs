use std::fmt;
use std::ops::Range;

use serde::{Deserialize, Serialize};

pub use rvllm_core::prelude::{
    FinishReason, RequestId, RequestOutput, ResponseFormat, SamplingParams, SequenceId, TokenId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blk-{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceStatus {
    Running,
    Finished(FinishReason),
    Swapped,
    Waiting,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepDiff {
    pub added: Vec<AddedRequest>,
    pub removed: Vec<RequestId>,
    pub continued: Vec<ContinuedRequest>,
    pub block_ops: BlockOps,
}

impl StepDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.continued.is_empty()
            && self.block_ops.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddedRequest {
    pub request_id: RequestId,
    pub seq_id: SequenceId,
    pub prompt_token_ids: Vec<TokenId>,
    pub sampling_params: SamplingParams,
    pub block_table: Vec<BlockId>,
    pub is_prefill: bool,
    pub token_chunk: Range<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuedRequest {
    pub request_id: RequestId,
    pub seq_id: SequenceId,
    pub new_token_id: TokenId,
    pub block_table_update: Option<Vec<BlockId>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockOps {
    pub copies: Vec<(BlockId, BlockId)>,
    pub swap_in: Vec<(BlockId, BlockId)>,
    pub swap_out: Vec<(BlockId, BlockId)>,
}

impl BlockOps {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.copies.is_empty() && self.swap_in.is_empty() && self.swap_out.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub request_id: RequestId,
    pub seq_id: SequenceId,
    pub prompt_token_ids: Vec<TokenId>,
    pub output_token_ids: Vec<TokenId>,
    pub sampling_params: SamplingParams,
    pub block_table: Vec<BlockId>,
    pub is_prefill: bool,
    pub num_computed_tokens: usize,
    pub token_chunk: Range<usize>,
}

impl WorkerRequest {
    pub fn seq_len(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    pub fn next_position(&self) -> u32 {
        self.seq_len().saturating_sub(1) as u32
    }

    pub fn current_slot(&self, block_size: usize) -> u32 {
        let pos = self.seq_len().saturating_sub(1);
        let block_idx = pos / block_size;
        let block_offset = pos % block_size;
        let block_id = self.block_table[block_idx];
        block_id.0 * block_size as u32 + block_offset as u32
    }

    pub fn last_token_id(&self) -> TokenId {
        self.output_token_ids
            .last()
            .copied()
            .unwrap_or(*self.prompt_token_ids.last().unwrap())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuBatchInput {
    pub num_seqs: usize,
    pub num_prefill_seqs: usize,
    pub num_decode_seqs: usize,
    pub seq_ids: Vec<u64>,
    pub token_ids: Vec<TokenId>,
    pub position_ids: Vec<u32>,
    pub slot_mapping: Vec<u32>,
    pub context_lens: Vec<u32>,
    pub query_lens: Vec<u32>,
    pub is_all_greedy: bool,
    pub block_tables_flat: Vec<u32>,
    pub max_blocks_per_seq: usize,
    pub prefill_tokens: Vec<TokenId>,
    pub prefill_positions: Vec<u32>,
    pub prefill_slot_mapping: Vec<u32>,
    pub is_all_decode: bool,
    pub is_all_prefill: bool,
    pub max_context_len: u32,
}

impl GpuBatchInput {
    pub fn new() -> Self {
        Self {
            is_all_decode: true,
            is_all_prefill: true,
            is_all_greedy: true,
            ..Self::default()
        }
    }

    pub fn clear(&mut self) {
        self.num_seqs = 0;
        self.num_prefill_seqs = 0;
        self.num_decode_seqs = 0;
        self.seq_ids.clear();
        self.token_ids.clear();
        self.position_ids.clear();
        self.slot_mapping.clear();
        self.context_lens.clear();
        self.query_lens.clear();
        self.is_all_greedy = true;
        self.block_tables_flat.clear();
        self.max_blocks_per_seq = 0;
        self.prefill_tokens.clear();
        self.prefill_positions.clear();
        self.prefill_slot_mapping.clear();
        self.is_all_decode = true;
        self.is_all_prefill = true;
        self.max_context_len = 0;
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForwardOutput {
    pub token_ids: Vec<TokenId>,
    pub logprobs: Vec<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerOutput {
    pub diff: StepDiff,
    pub num_running: usize,
    pub num_waiting: usize,
    pub total_batched_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2RequestOutput {
    pub request_id: RequestId,
    pub output_text: String,
    pub output_token_ids: Vec<TokenId>,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Vec<f32>,
}
