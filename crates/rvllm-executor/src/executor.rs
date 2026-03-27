//! Executor trait and input types.

use async_trait::async_trait;
use rvllm_core::prelude::{BlockId, Result};

use crate::{SamplerOutput, SequenceGroupMetadata};

/// Input bundle for a single executor step.
#[derive(Debug, Clone)]
pub struct ExecutorInput {
    pub seq_group_metadata_list: Vec<SequenceGroupMetadata>,
    pub blocks_to_swap_in: Vec<(BlockId, BlockId)>,
    pub blocks_to_swap_out: Vec<(BlockId, BlockId)>,
    pub blocks_to_copy: Vec<(BlockId, BlockId)>,
}

/// Core trait for dispatching model execution across one or more GPUs.
#[async_trait]
pub trait Executor: Send + Sync {
    /// Run one forward pass across all workers and return sampled outputs.
    async fn execute_model(&self, input: ExecutorInput) -> Result<Vec<SamplerOutput>>;

    /// Verify that all workers are alive and responsive.
    async fn check_health(&self) -> Result<()>;

    /// Gracefully shut down all workers.
    async fn shutdown(&self) -> Result<()>;

    /// Number of free GPU KV-cache blocks (from rank-0 worker).
    fn num_available_gpu_blocks(&self) -> usize;

    /// Number of free CPU KV-cache blocks (from rank-0 worker).
    fn num_available_cpu_blocks(&self) -> usize;
}
