#![forbid(unsafe_code)]
//! Multi-GPU executor orchestration for vllm-rs.
//!
//! Provides the `Executor` trait and concrete implementations for single-GPU
//! and multi-GPU inference dispatch, along with a factory for automatic
//! selection based on configuration.

pub mod config;
pub mod executor;
pub mod factory;
pub mod multi_gpu;
pub mod single_gpu;
pub mod tensor_parallel;

pub use config::ExecutorConfig;
pub use executor::{Executor, ExecutorInput};
pub use factory::ExecutorFactory;
pub use multi_gpu::MultiGpuExecutor;
pub use single_gpu::SingleGpuExecutor;
pub use tensor_parallel::{
    classify_parallel_style, ColumnParallelLinear, ParallelStyle, RowParallelLinear,
    TensorParallelConfig, TransformerLayerParallel,
};

// ---------------------------------------------------------------------------
// Wire to real types from rvllm-sequence where possible.
// ---------------------------------------------------------------------------

/// Re-export from rvllm_sequence::SequenceGroupMetadata.
pub use rvllm_sequence::SequenceGroupMetadata;

// ---------------------------------------------------------------------------
// SamplerOutput: kept local because rvllm_sampling::SamplerOutput has a
// different shape (single token_id + logprob + top_logprobs) while this
// executor-level output bundles multiple token_ids per group.
// TODO: unify with rvllm_sampling::SamplerOutput once APIs converge
// ---------------------------------------------------------------------------

/// Executor-level sampler output for a sequence group.
#[derive(Debug, Clone)]
pub struct SamplerOutput {
    pub token_ids: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
}

// ---------------------------------------------------------------------------
// Worker stubs: kept local because adding rvllm-worker as a dependency would
// pull in rvllm-model-runner, rvllm-kv-cache, rvllm-gpu. The executor uses
// these via async channels, so the interface is narrow.
// TODO: wire to rvllm_worker::{Worker, WorkerInput, WorkerOutput} once the
// dependency chain is acceptable
// ---------------------------------------------------------------------------

/// Placeholder for `rvllm_worker::WorkerConfig`.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub rank: usize,
    pub gpu_id: usize,
    pub model_name: String,
}

/// Input to a single worker execution step.
#[derive(Debug, Clone)]
pub struct WorkerInput {
    pub seq_group_metadata_list: Vec<SequenceGroupMetadata>,
    pub blocks_to_swap_in: Vec<(rvllm_core::prelude::BlockId, rvllm_core::prelude::BlockId)>,
    pub blocks_to_swap_out: Vec<(rvllm_core::prelude::BlockId, rvllm_core::prelude::BlockId)>,
    pub blocks_to_copy: Vec<(rvllm_core::prelude::BlockId, rvllm_core::prelude::BlockId)>,
}

/// Full output from one worker execution step.
#[derive(Debug, Clone)]
pub struct WorkerOutput {
    pub sampler_outputs: Vec<SamplerOutput>,
}

/// Placeholder for `rvllm_worker::Worker`.
pub struct Worker {
    pub config: WorkerConfig,
    gpu_blocks: usize,
    cpu_blocks: usize,
}

impl Worker {
    pub fn new(config: WorkerConfig) -> rvllm_core::prelude::Result<Self> {
        tracing::info!(
            rank = config.rank,
            gpu = config.gpu_id,
            "worker initialized"
        );
        Ok(Self {
            config,
            gpu_blocks: 256,
            cpu_blocks: 128,
        })
    }

    pub async fn execute_model(
        &self,
        input: WorkerInput,
    ) -> rvllm_core::prelude::Result<WorkerOutput> {
        tracing::debug!(
            rank = self.config.rank,
            seqs = input.seq_group_metadata_list.len(),
            "worker executing model"
        );
        Ok(WorkerOutput {
            sampler_outputs: vec![SamplerOutput {
                token_ids: vec![0; input.seq_group_metadata_list.len()],
                logprobs: vec![0.0; input.seq_group_metadata_list.len()],
                top_logprobs: vec![Vec::new(); input.seq_group_metadata_list.len()],
            }],
        })
    }

    pub async fn check_health(&self) -> rvllm_core::prelude::Result<()> {
        Ok(())
    }

    pub fn num_available_gpu_blocks(&self) -> usize {
        self.gpu_blocks
    }

    pub fn num_available_cpu_blocks(&self) -> usize {
        self.cpu_blocks
    }
}
