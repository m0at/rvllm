//! Scheduler output types describing what the engine should execute.

use rvllm_core::prelude::BlockId;

use crate::scheduler::SequenceGroup;

/// A sequence group scheduled for this iteration, with its token chunk size.
#[derive(Debug, Clone)]
pub struct ScheduledSequenceGroup {
    pub seq_group: SequenceGroup,
    /// True when this scheduled step is processing prompt tokens.
    pub is_prefill: bool,
    /// Number of tokens to process this step (chunked prefill support).
    pub token_chunk_size: usize,
}

/// Complete output of a single scheduling step.
#[derive(Debug, Clone)]
pub struct SchedulerOutputs {
    /// Groups selected to run this iteration.
    pub scheduled_seq_groups: Vec<ScheduledSequenceGroup>,
    /// Block pairs to swap from CPU -> GPU.
    pub blocks_to_swap_in: Vec<(BlockId, BlockId)>,
    /// Block pairs to swap from GPU -> CPU.
    pub blocks_to_swap_out: Vec<(BlockId, BlockId)>,
    /// Block pairs to copy (copy-on-write).
    pub blocks_to_copy: Vec<(BlockId, BlockId)>,
    /// Total tokens in this batch.
    pub num_batched_tokens: usize,
    /// How many of the scheduled groups are prefills (not yet decoding).
    pub num_prefill_groups: usize,
}

impl SchedulerOutputs {
    pub fn empty() -> Self {
        Self {
            scheduled_seq_groups: Vec::new(),
            blocks_to_swap_in: Vec::new(),
            blocks_to_swap_out: Vec::new(),
            blocks_to_copy: Vec::new(),
            num_batched_tokens: 0,
            num_prefill_groups: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.scheduled_seq_groups.is_empty()
    }
}
