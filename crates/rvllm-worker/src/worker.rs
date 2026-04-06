//! Worker: single-GPU execution context.

use std::sync::Arc;

use rvllm_core::prelude::{BlockId, LLMError, Result, TokenId};
use rvllm_gpu::prelude::GpuStream;
use rvllm_kv_cache::CacheEngine;
use rvllm_model_runner::bridge::{GpuAllocator, MockAttentionBackend, ModelWeights};
use rvllm_model_runner::ModelRunner;
use tracing::{debug, info};

use crate::config::WorkerConfig;
use crate::input::{self, SequenceGroupMetadata};

/// Input to a single worker execution step.
#[derive(Debug, Clone)]
pub struct WorkerInput {
    pub seq_group_metadata_list: Vec<SequenceGroupMetadata>,
    pub blocks_to_swap_in: Vec<(BlockId, BlockId)>,
    pub blocks_to_swap_out: Vec<(BlockId, BlockId)>,
    pub blocks_to_copy: Vec<(BlockId, BlockId)>,
}

/// Output of a single sampled token per sequence.
#[derive(Debug, Clone)]
pub struct SamplerOutput {
    pub seq_id: u64,
    pub token_id: TokenId,
    pub logprob: f32,
}

/// Full output from one worker execution step.
#[derive(Debug, Clone)]
pub struct WorkerOutput {
    pub outputs: Vec<SamplerOutput>,
}

/// Single-GPU worker that owns a model runner, cache engine, and GPU stream.
pub struct Worker {
    pub model_runner: Option<ModelRunner>,
    pub cache_engine: Option<CacheEngine>,
    pub device_id: usize,
    pub gpu: Arc<dyn GpuAllocator>,
    /// Stream for KV cache operations (swap, copy).
    cache_stream: GpuStream,
    config: WorkerConfig,
}

impl Worker {
    /// Create a new worker bound to the given GPU device.
    pub fn new(config: WorkerConfig, gpu: Arc<dyn GpuAllocator>) -> Result<Self> {
        let device_id = config.device_id;
        info!(device_id, rank = config.rank, "creating worker");

        let cache_stream = GpuStream::new(device_id)
            .map_err(|e| LLMError::GpuError(format!("failed to create GPU stream: {e}")))?;

        Ok(Self {
            model_runner: None,
            cache_engine: None,
            device_id,
            gpu,
            cache_stream,
            config,
        })
    }

    /// Load model weights into the model runner.
    pub fn init_model(&mut self, model_weights: ModelWeights) -> Result<()> {
        info!(device_id = self.device_id, "initializing model");

        let mr_config = self.config.model_runner_config();
        let cache_elements = self.config.num_kv_heads * self.config.head_dim * 16;
        let cache = Arc::new(rvllm_model_runner::bridge::CacheEngine::new(
            self.config.num_layers,
            cache_elements,
        ));
        let attention = Box::new(MockAttentionBackend);

        let runner =
            ModelRunner::new(model_weights, mr_config, attention, cache, self.gpu.clone())?;

        self.model_runner = Some(runner);
        info!(device_id = self.device_id, "model initialized");
        Ok(())
    }

    /// Allocate GPU and CPU KV cache blocks.
    pub fn init_cache(&mut self, num_gpu_blocks: usize, num_cpu_blocks: usize) -> Result<()> {
        info!(
            device_id = self.device_id,
            num_gpu_blocks, num_cpu_blocks, "initializing KV cache"
        );

        // TODO: use the real GPU allocator once CacheEngine::new accepts dyn GpuAllocator
        #[cfg(feature = "cuda")]
        let kv_alloc = rvllm_gpu::cuda_allocator::CudaGpuAllocator::new(self.device_id)?;
        #[cfg(not(feature = "cuda"))]
        let kv_alloc = rvllm_gpu::prelude::MockGpuAllocator::new(1 << 30);

        let engine = CacheEngine::new(
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            &kv_alloc,
        )?;

        self.cache_engine = Some(engine);
        info!(device_id = self.device_id, "KV cache initialized");
        Ok(())
    }

    /// Execute one step: apply cache ops, run forward pass, sample tokens.
    pub fn execute_model(&mut self, input: WorkerInput) -> Result<WorkerOutput> {
        debug!(
            device_id = self.device_id,
            num_groups = input.seq_group_metadata_list.len(),
            swap_in = input.blocks_to_swap_in.len(),
            swap_out = input.blocks_to_swap_out.len(),
            copy = input.blocks_to_copy.len(),
            "execute_model"
        );

        // 1. Apply cache operations
        if let Some(cache) = &mut self.cache_engine {
            if !input.blocks_to_swap_in.is_empty() {
                cache.swap_in(&input.blocks_to_swap_in, &self.cache_stream)?;
            }
            if !input.blocks_to_swap_out.is_empty() {
                cache.swap_out(&input.blocks_to_swap_out, &self.cache_stream)?;
            }
            if !input.blocks_to_copy.is_empty() {
                cache.copy_blocks(&input.blocks_to_copy, &self.cache_stream)?;
            }
        }

        // 2. Prepare model input tensors
        let model_input =
            input::prepare_input(&input.seq_group_metadata_list, self.config.block_size)?;

        // 3. Run forward pass
        let runner = self
            .model_runner
            .as_ref()
            .ok_or_else(|| LLMError::ModelError("model not initialized".into()))?;
        let logits = runner.execute_model(model_input)?;

        // 4. Sample tokens from logits
        let outputs = sample_from_logits(
            &logits.data,
            &input.seq_group_metadata_list,
            runner.config.vocab_size,
        );

        debug!(
            device_id = self.device_id,
            num_outputs = outputs.len(),
            "execute_model done"
        );
        Ok(WorkerOutput { outputs })
    }

    /// Profile available GPU memory and compute how many KV cache blocks fit.
    pub fn profile_num_available_blocks(
        &self,
        gpu_memory_utilization: f32,
        gpu_memory_reserve_bytes: usize,
    ) -> Result<(usize, usize)> {
        let free_bytes = self.gpu.free_gpu_bytes();
        let free_after_reserve = free_bytes.saturating_sub(gpu_memory_reserve_bytes);
        let available = (free_after_reserve as f32 * gpu_memory_utilization) as usize;

        let cache_cfg = self.config.cache_config();
        let total_block_bytes = cache_cfg.total_block_bytes();

        let num_gpu_blocks = if total_block_bytes > 0 {
            available / total_block_bytes
        } else {
            0
        };

        // CPU blocks: use 4 GiB as default swap space
        let cpu_swap_bytes = 4 * 1024 * 1024 * 1024_usize;
        let num_cpu_blocks = if total_block_bytes > 0 {
            cpu_swap_bytes / total_block_bytes
        } else {
            0
        };

        info!(
            free_bytes,
            gpu_memory_reserve_bytes,
            free_after_reserve,
            available, num_gpu_blocks, num_cpu_blocks, "profiled available blocks"
        );

        Ok((num_gpu_blocks, num_cpu_blocks))
    }

    /// Dummy forward pass to trigger lazy GPU initialization.
    pub fn warm_up(&self) -> Result<()> {
        info!(device_id = self.device_id, "warming up worker");

        if let Some(runner) = &self.model_runner {
            let dummy_input = rvllm_model_runner::ModelInput {
                token_ids: vec![0],
                position_ids: vec![0],
                attention_metadata: rvllm_model_runner::bridge::AttentionMetadata {
                    slot_mapping: vec![0],
                    context_lens: vec![1],
                    query_lens: vec![1],
                    block_tables: vec![vec![0]],
                    max_context_len: 1,
                },
                is_prefill: true,
            };
            let _ = runner.execute_model(dummy_input)?;
        }

        info!(device_id = self.device_id, "warm-up complete");
        Ok(())
    }
}

/// Greedy sample one token per sequence from the logits buffer.
fn sample_from_logits(
    logits: &[f32],
    metadata: &[SequenceGroupMetadata],
    vocab_size: usize,
) -> Vec<SamplerOutput> {
    let mut outputs = Vec::new();
    let mut offset = 0;

    for group in metadata {
        for (seq_id, seq_data) in &group.seq_data {
            let num_tokens = if group.is_prompt {
                seq_data.prompt_token_ids.len()
            } else {
                1
            };

            // Logits for the last token of this sequence
            let last_logit_start = offset + (num_tokens - 1) * vocab_size;
            let last_logit_end = last_logit_start + vocab_size;

            if last_logit_end <= logits.len() {
                let seq_logits = &logits[last_logit_start..last_logit_end];
                let (token_id, logprob) = greedy_sample(seq_logits);
                outputs.push(SamplerOutput {
                    seq_id: seq_id.0,
                    token_id,
                    logprob,
                });
            } else {
                // Logits too short (mock model may return fewer);
                // produce a dummy token.
                outputs.push(SamplerOutput {
                    seq_id: seq_id.0,
                    token_id: 0,
                    logprob: 0.0,
                });
            }

            offset += num_tokens * vocab_size;
        }
    }

    outputs
}

/// Argmax + log-probability of the selected token.
fn greedy_sample(logits: &[f32]) -> (TokenId, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }

    let mut best_idx = 0;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }

    // Compute log-softmax for the selected token
    let max = best_val;
    let lse = max + logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    let logprob = best_val - lse;

    (best_idx as TokenId, logprob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sample_basic() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let (token, lp) = greedy_sample(&logits);
        assert_eq!(token, 3);
        assert!(lp < 0.0);
    }

    #[test]
    fn greedy_sample_empty() {
        let (token, lp) = greedy_sample(&[]);
        assert_eq!(token, 0);
        assert_eq!(lp, 0.0);
    }

    #[test]
    fn greedy_sample_single() {
        let (token, lp) = greedy_sample(&[5.0]);
        assert_eq!(token, 0);
        assert!((lp - 0.0).abs() < 1e-6);
    }

    #[test]
    fn worker_output_construction() {
        let output = WorkerOutput {
            outputs: vec![
                SamplerOutput {
                    seq_id: 1,
                    token_id: 42,
                    logprob: -0.5,
                },
                SamplerOutput {
                    seq_id: 2,
                    token_id: 99,
                    logprob: -1.2,
                },
            ],
        };
        assert_eq!(output.outputs.len(), 2);
        assert_eq!(output.outputs[0].token_id, 42);
    }

    #[test]
    fn worker_input_construction() {
        let input = WorkerInput {
            seq_group_metadata_list: vec![],
            blocks_to_swap_in: vec![(BlockId(0), BlockId(1))],
            blocks_to_swap_out: vec![],
            blocks_to_copy: vec![(BlockId(2), BlockId(3))],
        };
        assert_eq!(input.blocks_to_swap_in.len(), 1);
        assert_eq!(input.blocks_to_copy.len(), 1);
    }

    #[test]
    fn sample_from_logits_basic() {
        use rvllm_core::prelude::{RequestId, SequenceId};
        use rvllm_sequence::SequenceData;
        use std::collections::HashMap;

        let vocab_size = 4;
        // 2 sequences, decode mode (1 token each)
        let logits = vec![
            0.1, 0.9, 0.2, 0.3, // seq 0 -> token 1
            0.5, 0.1, 0.8, 0.2, // seq 1 -> token 2
        ];
        let sd0 = SequenceData {
            prompt_token_ids: vec![1],
            output_token_ids: vec![2],
            cumulative_logprob: 0.0,
            seq_len: 2,
            last_token_id: 2,
        };
        let sd1 = SequenceData {
            prompt_token_ids: vec![3],
            output_token_ids: vec![4],
            cumulative_logprob: 0.0,
            seq_len: 2,
            last_token_id: 4,
        };
        let groups = vec![
            SequenceGroupMetadata {
                request_id: RequestId(0),
                is_prompt: false,
                seq_data: [(SequenceId(10), sd0)].into_iter().collect(),
                block_tables: HashMap::new(),
                sampling_params: rvllm_core::prelude::SamplingParams::default(),
            },
            SequenceGroupMetadata {
                request_id: RequestId(1),
                is_prompt: false,
                seq_data: [(SequenceId(20), sd1)].into_iter().collect(),
                block_tables: HashMap::new(),
                sampling_params: rvllm_core::prelude::SamplingParams::default(),
            },
        ];

        let outputs = sample_from_logits(&logits, &groups, vocab_size);
        assert_eq!(outputs.len(), 2);
        // HashMap iteration order is nondeterministic, so check both are present
        let ids: std::collections::HashSet<u64> = outputs.iter().map(|o| o.seq_id).collect();
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
    }

    #[test]
    fn sample_from_logits_short_logits_fallback() {
        use rvllm_core::prelude::{RequestId, SequenceId};
        use rvllm_sequence::SequenceData;
        use std::collections::HashMap;

        let sd = SequenceData {
            prompt_token_ids: vec![1],
            output_token_ids: vec![],
            cumulative_logprob: 0.0,
            seq_len: 1,
            last_token_id: 1,
        };
        let groups = vec![SequenceGroupMetadata {
            request_id: RequestId(0),
            is_prompt: false,
            seq_data: [(SequenceId(5), sd)].into_iter().collect(),
            block_tables: HashMap::new(),
            sampling_params: rvllm_core::prelude::SamplingParams::default(),
        }];

        // Empty logits => fallback to dummy
        let outputs = sample_from_logits(&[], &groups, 100);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].token_id, 0);
    }
}
