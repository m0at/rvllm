//! CUDA graph runner for decode steps.
//!
//! Wraps the forward pass so that decode steps (single token per sequence) are
//! captured into CUDA graphs on first encounter and replayed on subsequent
//! steps, eliminating kernel launch overhead.
//!
//! Only decode steps are graphed -- prefill varies in sequence length and cannot
//! be captured into a fixed graph. Input tensors are padded to the nearest
//! cached batch size so the same graph can serve multiple actual batch sizes.

use std::collections::HashMap;

use tracing::{debug, info, trace, warn};

use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::cuda_graph::{padded_batch_size, CudaGraphPool, GRAPH_BATCH_SIZES};
use rvllm_gpu::stream::GpuStream;
use rvllm_model_runner::bridge::AttentionMetadata;
use rvllm_model_runner::input::ModelInput;

/// Configuration for the graph runner.
#[derive(Debug, Clone)]
pub struct GraphRunnerConfig {
    /// Maximum batch size to capture graphs for.
    pub max_batch_size: usize,
    /// Whether to enable graph capture/replay.
    pub enabled: bool,
    /// Vocabulary size (needed for output padding).
    pub vocab_size: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
}

impl Default for GraphRunnerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            enabled: true,
            vocab_size: 32000,
            hidden_size: 4096,
        }
    }
}

/// Manages CUDA graph capture and replay for decode steps.
///
/// Sits between the scheduler and the actual model forward pass. For decode
/// batches it pads input to the nearest cached batch size, replays a captured
/// graph if available, and strips padding from the output. For prefill or
/// oversized batches it falls through to the normal forward path.
pub struct GraphRunner {
    pool: CudaGraphPool,
    config: GraphRunnerConfig,
    /// Tracks which batch sizes have been captured.
    captured: HashMap<usize, bool>,
}

impl GraphRunner {
    /// Create a new graph runner.
    pub fn new(config: GraphRunnerConfig) -> Self {
        info!(
            max_batch_size = config.max_batch_size,
            enabled = config.enabled,
            "creating GraphRunner"
        );
        let pool = CudaGraphPool::new(config.max_batch_size);
        Self {
            pool,
            config,
            captured: HashMap::new(),
        }
    }

    /// Whether graph replay is enabled and the batch can use it.
    pub fn can_use_graph(&self, input: &ModelInput) -> bool {
        if !self.config.enabled || !self.pool.is_enabled() {
            return false;
        }
        // Only decode steps (not prefill)
        if input.is_prefill {
            return false;
        }
        let batch_size = input.num_tokens();
        padded_batch_size(batch_size)
            .map(|p| p <= self.config.max_batch_size)
            .unwrap_or(false)
    }

    /// Pad a decode ModelInput to the nearest graph-cached batch size.
    ///
    /// Adds dummy tokens (id=0) with zero-valued attention metadata to fill
    /// the batch to the padded size. The caller must strip these extra outputs
    /// after the forward pass.
    pub fn pad_input(&self, input: &ModelInput) -> Result<(ModelInput, usize)> {
        let actual = input.num_tokens();
        let padded = padded_batch_size(actual).ok_or_else(|| {
            LLMError::GpuError(format!(
                "batch size {} exceeds max graphable size {}",
                actual,
                *GRAPH_BATCH_SIZES.last().unwrap()
            ))
        })?;

        if padded == actual {
            trace!(batch_size = actual, "no padding needed");
            return Ok((input.clone(), actual));
        }

        let pad_count = padded - actual;
        debug!(actual, padded, pad_count, "padding decode input for graph");

        let mut token_ids = input.token_ids.clone();
        let mut position_ids = input.position_ids.clone();
        let mut slot_mapping = input.attention_metadata.slot_mapping.clone();
        let mut context_lens = input.attention_metadata.context_lens.clone();
        let mut block_tables = input.attention_metadata.block_tables.clone();

        // Pad with dummy entries. Use -1 for slot_mapping so cache_write
        // skips padded tokens (kernel checks slot >= 0).
        for _ in 0..pad_count {
            token_ids.push(0);
            position_ids.push(0);
            slot_mapping.push(u32::MAX); // -1 as u32, kernel interprets as negative i32
            context_lens.push(0); // 0 context = attention kernel skips this seq
            block_tables.push(vec![0]);
        }

        let max_context_len = context_lens.iter().copied().max().unwrap_or(0);

        Ok((
            ModelInput {
                token_ids,
                position_ids,
                attention_metadata: AttentionMetadata {
                    slot_mapping,
                    query_lens: vec![1; context_lens.len()],
                    context_lens,
                    block_tables,
                    max_context_len,
                },
                is_prefill: false,
            },
            actual,
        ))
    }

    /// Strip padding from the logits output.
    ///
    /// Given logits of shape `[padded_batch, vocab_size]`, returns only the
    /// first `actual_batch * vocab_size` elements.
    pub fn unpad_logits(&self, logits: &[f32], actual_batch: usize) -> Vec<f32> {
        let vocab = self.config.vocab_size;
        let end = actual_batch * vocab;
        if end <= logits.len() {
            logits[..end].to_vec()
        } else {
            warn!(
                actual_batch,
                vocab,
                logits_len = logits.len(),
                "logits shorter than expected after unpadding"
            );
            logits.to_vec()
        }
    }

    /// Access the underlying graph pool.
    pub fn pool(&self) -> &CudaGraphPool {
        &self.pool
    }

    /// Mutable access to the graph pool (for capture/insert).
    pub fn pool_mut(&mut self) -> &mut CudaGraphPool {
        &mut self.pool
    }

    /// Check if a graph has been captured for the given batch size.
    pub fn has_graph_for(&self, batch_size: usize) -> bool {
        self.pool.has_graph(batch_size)
    }

    /// Check if a graph exists for this EXACT batch size (no padding lookup).
    pub fn has_graph_for_exact(&self, batch_size: usize) -> bool {
        self.pool.get_exact(batch_size).is_some()
    }

    /// Record that a graph capture was attempted for a batch size.
    pub fn mark_captured(&mut self, padded_batch_size: usize) {
        self.captured.insert(padded_batch_size, true);
    }

    /// Whether capture has been attempted for this padded batch size.
    pub fn was_capture_attempted(&self, padded_batch_size: usize) -> bool {
        self.captured
            .get(&padded_batch_size)
            .copied()
            .unwrap_or(false)
    }

    /// Capture a graph by running a forward pass on `stream`, then store it.
    ///
    /// `forward_fn` should execute the full decode forward pass for the given
    /// padded input. All kernel launches during `forward_fn` are captured.
    pub fn capture_graph<F>(
        &mut self,
        stream: &GpuStream,
        padded_batch_size: usize,
        forward_fn: F,
    ) -> Result<()>
    where
        F: Fn() -> Result<()>,
    {
        if self.was_capture_attempted(padded_batch_size) {
            trace!(
                padded_batch_size,
                "graph capture already attempted, skipping"
            );
            return Ok(());
        }

        info!(padded_batch_size, "capturing CUDA graph for decode");

        // Warm up: run once without capture to ensure lazy initialization is done.
        forward_fn()?;
        stream.synchronize()?;

        // Now capture.
        self.pool.begin_capture(stream)?;
        forward_fn()?;
        let graph = self.pool.end_capture(stream, padded_batch_size)?;
        self.pool.insert(graph);
        self.mark_captured(padded_batch_size);

        info!(padded_batch_size, "CUDA graph captured successfully");
        Ok(())
    }

    /// Replay a cached graph for the given decode step.
    ///
    /// Returns `true` if a graph was replayed, `false` if the caller should
    /// fall back to the normal forward path.
    pub fn try_replay(&self, stream: &GpuStream, actual_batch_size: usize) -> Result<bool> {
        match self.pool.get(actual_batch_size) {
            Some(graph) => {
                trace!(
                    actual_batch_size,
                    padded = graph.batch_size(),
                    "replaying cached CUDA graph"
                );
                graph.replay(stream)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Disable CUDA graph capture and replay.
    pub fn disable(&mut self) {
        self.config.enabled = false;
        self.pool.disable();
    }

    /// Enable CUDA graph capture and replay.
    pub fn enable(&mut self) {
        self.config.enabled = true;
        self.pool.enable();
    }

    /// Whether graph mode is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Clear all cached graphs (e.g., after model reload).
    pub fn clear(&mut self) {
        self.pool.clear();
        self.captured.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decode_input(batch_size: usize) -> ModelInput {
        ModelInput {
            token_ids: vec![42; batch_size],
            position_ids: vec![10; batch_size],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0; batch_size],
                context_lens: vec![11; batch_size],
                block_tables: vec![vec![0]; batch_size],
                query_lens: vec![1; batch_size],
                max_context_len: 11,
            },
            is_prefill: false,
        }
    }

    fn make_prefill_input(seq_len: usize) -> ModelInput {
        ModelInput {
            token_ids: (0..seq_len as u32).collect(),
            position_ids: (0..seq_len as u32).collect(),
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0; seq_len],
                context_lens: vec![seq_len as u32],
                query_lens: vec![seq_len as u32],
                block_tables: vec![vec![0]],
                max_context_len: seq_len as u32,
            },
            is_prefill: true,
        }
    }

    #[test]
    fn can_use_graph_decode_only() {
        let runner = GraphRunner::new(GraphRunnerConfig::default());
        let decode = make_decode_input(4);
        let prefill = make_prefill_input(128);

        assert!(runner.can_use_graph(&decode));
        assert!(!runner.can_use_graph(&prefill));
    }

    #[test]
    fn can_use_graph_respects_max_batch() {
        let runner = GraphRunner::new(GraphRunnerConfig {
            max_batch_size: 8,
            ..Default::default()
        });

        assert!(runner.can_use_graph(&make_decode_input(8)));
        assert!(!runner.can_use_graph(&make_decode_input(16)));
    }

    #[test]
    fn can_use_graph_disabled() {
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());
        runner.disable();

        assert!(!runner.can_use_graph(&make_decode_input(4)));

        runner.enable();
        assert!(runner.can_use_graph(&make_decode_input(4)));
    }

    #[test]
    fn pad_input_exact() {
        let runner = GraphRunner::new(GraphRunnerConfig::default());
        let input = make_decode_input(4);
        let (padded, actual) = runner.pad_input(&input).unwrap();
        assert_eq!(actual, 4);
        assert_eq!(padded.num_tokens(), 4);
        assert_eq!(padded.token_ids, input.token_ids);
    }

    #[test]
    fn pad_input_rounds_up() {
        let runner = GraphRunner::new(GraphRunnerConfig::default());
        let input = make_decode_input(3);
        let (padded, actual) = runner.pad_input(&input).unwrap();
        assert_eq!(actual, 3);
        assert_eq!(padded.num_tokens(), 4); // rounded up to 4
        assert_eq!(padded.token_ids[..3], vec![42, 42, 42]);
        assert_eq!(padded.token_ids[3], 0); // padding token
    }

    #[test]
    fn pad_input_too_large() {
        let runner = GraphRunner::new(GraphRunnerConfig::default());
        let input = make_decode_input(512);
        assert!(runner.pad_input(&input).is_err());
    }

    #[test]
    fn unpad_logits_strips_padding() {
        let runner = GraphRunner::new(GraphRunnerConfig {
            vocab_size: 4,
            ..Default::default()
        });
        // Padded logits: batch=4, vocab=4 => 16 elements
        let logits: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let unpadded = runner.unpad_logits(&logits, 3);
        assert_eq!(unpadded.len(), 12); // 3 * 4
        assert_eq!(unpadded, &logits[..12]);
    }

    #[test]
    fn unpad_logits_no_padding_needed() {
        let runner = GraphRunner::new(GraphRunnerConfig {
            vocab_size: 4,
            ..Default::default()
        });
        let logits: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let unpadded = runner.unpad_logits(&logits, 4);
        assert_eq!(unpadded.len(), 16);
    }

    #[test]
    fn capture_and_replay_mock() {
        let stream = GpuStream::new(0).unwrap();
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());

        // Capture a graph for batch size 8
        let call_count = std::sync::atomic::AtomicUsize::new(0);
        runner
            .capture_graph(&stream, 8, || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })
            .unwrap();

        // forward_fn called twice: once for warmup, once during capture
        assert_eq!(call_count.load(std::sync::atomic::Ordering::Relaxed), 2);

        assert!(runner.has_graph_for(8));
        assert!(runner.has_graph_for(5)); // rounds up to 8
        assert!(runner.was_capture_attempted(8));

        // Replay
        let replayed = runner.try_replay(&stream, 6).unwrap();
        assert!(replayed);
    }

    #[test]
    fn try_replay_no_graph() {
        let stream = GpuStream::new(0).unwrap();
        let runner = GraphRunner::new(GraphRunnerConfig::default());

        let replayed = runner.try_replay(&stream, 4).unwrap();
        assert!(!replayed);
    }

    #[test]
    fn clear_removes_everything() {
        let stream = GpuStream::new(0).unwrap();
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());

        runner.capture_graph(&stream, 4, || Ok(())).unwrap();
        assert!(runner.has_graph_for(4));

        runner.clear();
        assert!(!runner.has_graph_for(4));
        assert!(!runner.was_capture_attempted(4));
    }

    #[test]
    fn skip_duplicate_capture() {
        let stream = GpuStream::new(0).unwrap();
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());
        let call_count = std::sync::atomic::AtomicUsize::new(0);

        // First capture
        runner
            .capture_graph(&stream, 4, || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        let first = call_count.load(std::sync::atomic::Ordering::Relaxed);

        // Second capture attempt for same size -- should be skipped
        runner
            .capture_graph(&stream, 4, || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        let second = call_count.load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(first, second, "duplicate capture should be skipped");
    }

    #[test]
    fn graph_runner_config_default() {
        let cfg = GraphRunnerConfig::default();
        assert_eq!(cfg.max_batch_size, 32);
        assert!(cfg.enabled);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
    }
}
