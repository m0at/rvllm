//! Executor factory -- selects single vs multi-GPU based on config.

use rvllm_core::prelude::Result;

use crate::config::ExecutorConfig;
use crate::executor::Executor;
use crate::multi_gpu::MultiGpuExecutor;
use crate::single_gpu::SingleGpuExecutor;

/// Constructs the appropriate executor for the given configuration.
pub struct ExecutorFactory;

impl ExecutorFactory {
    /// Create a boxed `Executor`.
    ///
    /// Returns `SingleGpuExecutor` when `num_gpus == 1`,
    /// `MultiGpuExecutor` otherwise.
    pub fn create(config: ExecutorConfig) -> Result<Box<dyn Executor>> {
        if config.num_gpus <= 1 {
            tracing::info!("factory: selecting single-gpu executor");
            let exec = SingleGpuExecutor::new(config)?;
            Ok(Box::new(exec))
        } else {
            tracing::info!(num_gpus = config.num_gpus, "factory: selecting multi-gpu executor");
            let exec = MultiGpuExecutor::new(config)?;
            Ok(Box::new(exec))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use rvllm_core::prelude::{RequestId, SamplingParams, SequenceId};
    use rvllm_sequence::SequenceData;

    fn make_test_metadata(request_id: u64, is_prompt: bool) -> crate::SequenceGroupMetadata {
        let mut seq_data = HashMap::new();
        seq_data.insert(
            SequenceId(1),
            SequenceData {
                prompt_token_ids: vec![1, 2, 3],
                output_token_ids: vec![],
                cumulative_logprob: 0.0,
            },
        );
        crate::SequenceGroupMetadata {
            request_id: RequestId(request_id),
            is_prompt,
            seq_data,
            sampling_params: SamplingParams::default(),
            block_tables: HashMap::new(),
        }
    }

    #[test]
    fn factory_returns_single_for_one_gpu() {
        let config = ExecutorConfig {
            num_gpus: 1,
            model_name: "test-model".into(),
            ..Default::default()
        };
        let exec = ExecutorFactory::create(config);
        assert!(exec.is_ok());
        let exec = exec.unwrap();
        assert!(exec.num_available_gpu_blocks() > 0);
    }

    #[tokio::test]
    async fn factory_returns_multi_for_two_gpus() {
        let config = ExecutorConfig {
            num_gpus: 2,
            model_name: "test-model".into(),
            ..Default::default()
        };
        let exec = ExecutorFactory::create(config);
        assert!(exec.is_ok());
        let exec = exec.unwrap();
        assert!(exec.num_available_gpu_blocks() > 0);
        exec.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn single_gpu_execute_and_health() {
        let config = ExecutorConfig {
            num_gpus: 1,
            model_name: "test".into(),
            ..Default::default()
        };
        let exec = ExecutorFactory::create(config).unwrap();

        exec.check_health().await.unwrap();

        let input = crate::executor::ExecutorInput {
            seq_group_metadata_list: vec![make_test_metadata(1, true)],
            blocks_to_swap_in: vec![],
            blocks_to_swap_out: vec![],
            blocks_to_copy: vec![],
        };
        let outputs = exec.execute_model(input).await.unwrap();
        assert!(!outputs.is_empty());

        exec.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn multi_gpu_execute_and_health() {
        let config = ExecutorConfig {
            num_gpus: 3,
            model_name: "test".into(),
            ..Default::default()
        };
        let exec = ExecutorFactory::create(config).unwrap();

        exec.check_health().await.unwrap();

        let input = crate::executor::ExecutorInput {
            seq_group_metadata_list: vec![make_test_metadata(42, false)],
            blocks_to_swap_in: vec![],
            blocks_to_swap_out: vec![],
            blocks_to_copy: vec![],
        };
        let outputs = exec.execute_model(input).await.unwrap();
        assert!(!outputs.is_empty());

        exec.shutdown().await.unwrap();
    }

    #[test]
    fn multi_gpu_rejects_single() {
        let config = ExecutorConfig {
            num_gpus: 1,
            model_name: "test".into(),
            ..Default::default()
        };
        let result = MultiGpuExecutor::new(config);
        assert!(result.is_err());
    }
}
