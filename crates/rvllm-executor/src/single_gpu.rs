//! Single-GPU executor -- delegates directly to one Worker.

use async_trait::async_trait;
use rvllm_core::prelude::Result;

use crate::config::ExecutorConfig;
use crate::executor::{Executor, ExecutorInput};
use crate::{SamplerOutput, Worker, WorkerConfig, WorkerInput};

/// Executor backed by a single GPU worker.
pub struct SingleGpuExecutor {
    worker: Worker,
}

impl SingleGpuExecutor {
    pub fn new(config: ExecutorConfig) -> Result<Self> {
        let worker_cfg = WorkerConfig {
            rank: 0,
            gpu_id: 0,
            model_name: config.model_name.clone(),
        };
        let worker = Worker::new(worker_cfg)?;
        tracing::info!("single-gpu executor ready");
        Ok(Self { worker })
    }
}

#[async_trait]
impl Executor for SingleGpuExecutor {
    async fn execute_model(&self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
        let worker_input = WorkerInput {
            seq_group_metadata_list: input.seq_group_metadata_list,
            blocks_to_swap_in: input.blocks_to_swap_in,
            blocks_to_swap_out: input.blocks_to_swap_out,
            blocks_to_copy: input.blocks_to_copy,
        };
        let output = self.worker.execute_model(worker_input).await?;
        Ok(output.sampler_outputs)
    }

    async fn check_health(&self) -> Result<()> {
        self.worker.check_health().await
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("single-gpu executor shutting down");
        Ok(())
    }

    fn num_available_gpu_blocks(&self) -> usize {
        self.worker.num_available_gpu_blocks()
    }

    fn num_available_cpu_blocks(&self) -> usize {
        self.worker.num_available_cpu_blocks()
    }
}
