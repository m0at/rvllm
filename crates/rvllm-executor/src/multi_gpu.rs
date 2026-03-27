//! Multi-GPU executor -- broadcasts to N workers via channels.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use rvllm_core::prelude::{LLMError, Result};

use crate::config::ExecutorConfig;
use crate::executor::{Executor, ExecutorInput};
use crate::{SamplerOutput, Worker, WorkerConfig, WorkerInput};

/// Message sent from the executor to each worker thread.
enum WorkerCommand {
    Execute {
        input: WorkerInput,
        reply: oneshot::Sender<Result<Vec<SamplerOutput>>>,
    },
    HealthCheck {
        reply: oneshot::Sender<Result<()>>,
    },
    Shutdown,
}

/// Handle for a worker running on its own tokio task.
struct WorkerHandle {
    tx: mpsc::Sender<WorkerCommand>,
    gpu_blocks: usize,
    cpu_blocks: usize,
}

/// Executor that fans out to multiple GPU workers.
pub struct MultiGpuExecutor {
    handles: Vec<WorkerHandle>,
    _nccl_stub: (), // Placeholder for future NCCL communicator
}

impl MultiGpuExecutor {
    pub fn new(config: ExecutorConfig) -> Result<Self> {
        if config.num_gpus < 2 {
            return Err(LLMError::ConfigError(
                "MultiGpuExecutor requires num_gpus >= 2".into(),
            ));
        }

        // TODO: Initialize NCCL communicator across ranks
        tracing::info!(num_gpus = config.num_gpus, "setting up NCCL communicator (stub)");

        let mut handles = Vec::with_capacity(config.num_gpus);

        for rank in 0..config.num_gpus {
            let worker_cfg = WorkerConfig {
                rank,
                gpu_id: rank,
                model_name: config.model_name.clone(),
            };
            let worker = Worker::new(worker_cfg)?;
            let gpu_blocks = worker.num_available_gpu_blocks();
            let cpu_blocks = worker.num_available_cpu_blocks();

            let (tx, mut rx) = mpsc::channel::<WorkerCommand>(8);

            // Spawn a dedicated task per worker
            tokio::spawn(async move {
                while let Some(cmd) = rx.recv().await {
                    match cmd {
                        WorkerCommand::Execute { input, reply } => {
                            let result = worker.execute_model(input).await;
                            let mapped = result.map(|o| o.sampler_outputs);
                            let _ = reply.send(mapped);
                        }
                        WorkerCommand::HealthCheck { reply } => {
                            let _ = reply.send(worker.check_health().await);
                        }
                        WorkerCommand::Shutdown => {
                            tracing::info!(rank = worker.config.rank, "worker shutting down");
                            break;
                        }
                    }
                }
            });

            handles.push(WorkerHandle {
                tx,
                gpu_blocks,
                cpu_blocks,
            });
        }

        tracing::info!(
            num_workers = handles.len(),
            "multi-gpu executor ready"
        );

        Ok(Self {
            handles,
            _nccl_stub: (),
        })
    }
}

#[async_trait]
impl Executor for MultiGpuExecutor {
    async fn execute_model(&self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
        let input = Arc::new(input);
        let mut receivers = Vec::with_capacity(self.handles.len());

        // Broadcast the input to all workers
        for handle in &self.handles {
            let worker_input = WorkerInput {
                seq_group_metadata_list: input.seq_group_metadata_list.clone(),
                blocks_to_swap_in: input.blocks_to_swap_in.clone(),
                blocks_to_swap_out: input.blocks_to_swap_out.clone(),
                blocks_to_copy: input.blocks_to_copy.clone(),
            };

            let (reply_tx, reply_rx) = oneshot::channel();
            handle
                .tx
                .send(WorkerCommand::Execute {
                    input: worker_input,
                    reply: reply_tx,
                })
                .await
                .map_err(|_| LLMError::GpuError("worker channel closed".into()))?;

            receivers.push(reply_rx);
        }

        // Collect all results; return rank-0 output
        let mut rank0_output = None;
        for (rank, rx) in receivers.into_iter().enumerate() {
            let result = rx
                .await
                .map_err(|_| LLMError::GpuError(format!("worker rank {rank} dropped reply")))?;
            let output = result?;
            if rank == 0 {
                rank0_output = Some(output);
            }
        }

        rank0_output.ok_or_else(|| LLMError::GpuError("no rank-0 output".into()))
    }

    async fn check_health(&self) -> Result<()> {
        for (rank, handle) in self.handles.iter().enumerate() {
            let (reply_tx, reply_rx) = oneshot::channel();
            handle
                .tx
                .send(WorkerCommand::HealthCheck { reply: reply_tx })
                .await
                .map_err(|_| {
                    LLMError::GpuError(format!("worker rank {rank} unresponsive"))
                })?;

            reply_rx
                .await
                .map_err(|_| LLMError::GpuError(format!("worker rank {rank} dropped reply")))??;
        }
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        for handle in &self.handles {
            let _ = handle.tx.send(WorkerCommand::Shutdown).await;
        }
        tracing::info!("multi-gpu executor shut down");
        Ok(())
    }

    fn num_available_gpu_blocks(&self) -> usize {
        // Report from rank 0
        self.handles.first().map_or(0, |h| h.gpu_blocks)
    }

    fn num_available_cpu_blocks(&self) -> usize {
        self.handles.first().map_or(0, |h| h.cpu_blocks)
    }
}
