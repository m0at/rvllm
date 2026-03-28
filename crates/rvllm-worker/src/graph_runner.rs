//! CUDA graph runner for decode steps.
//!
//! Wraps the forward pass so that decode steps (single token per sequence) are
//! captured into CUDA graphs on first encounter and replayed on subsequent
//! steps, eliminating kernel launch overhead.
//!
//! Only decode steps are graphed -- prefill varies in sequence length and cannot
//! be captured into a fixed graph. Input tensors are padded to the nearest
//! cached batch size so the same graph can serve multiple actual batch sizes.
//!
//! ## Capture/Replay Protocol
//!
//! On capture, the padded ModelInput is stored persistently so that the host
//! buffers referenced by CUDA graph memcpy nodes remain valid. On replay:
//!
//! 1. The persistent input buffers are updated in-place (same heap addresses).
//! 2. The CUDA graph is replayed -- memcpy nodes re-read from the persistent
//!    host buffers, picking up the new token/position/metadata values.
//! 3. The cached logits from the last forward are returned (the graph replays
//!    the full pipeline including the DtoH copy into a persistent output).
//!
//! For the mock (non `cuda-graphs`) build, capture/replay are no-ops so the
//! normal forward path runs unconditionally and the output is cached.

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

/// Persistent input buffers for a captured CUDA graph.
///
/// These are heap-allocated Vecs whose backing storage addresses are baked into
/// the CUDA graph's memcpy nodes during capture. On replay, we mutate the
/// contents in-place (same capacity, same pointer) so the graph reads fresh
/// data from the same host addresses.
struct GraphInputBuffers {
    /// Padded ModelInput whose Vec fields are at stable heap addresses.
    input: ModelInput,
    /// The padded batch size this was captured for.
    padded_batch_size: usize,
}

impl GraphInputBuffers {
    fn new(padded_input: ModelInput, padded_bs: usize) -> Self {
        Self {
            input: padded_input,
            padded_batch_size: padded_bs,
        }
    }

    /// Update the persistent buffers with new decode data, keeping Vec
    /// allocations (and therefore heap pointers) stable.
    ///
    /// `actual` is the real batch size; entries beyond `actual` keep their
    /// padding values (zeros).
    fn update_in_place(&mut self, new_input: &ModelInput, actual: usize) {
        let padded = self.padded_batch_size;
        debug_assert!(actual <= padded);

        // Copy real token data into the persistent buffer
        self.input.token_ids[..actual].copy_from_slice(&new_input.token_ids[..actual]);
        self.input.position_ids[..actual].copy_from_slice(&new_input.position_ids[..actual]);
        self.input.attention_metadata.slot_mapping[..actual]
            .copy_from_slice(&new_input.attention_metadata.slot_mapping[..actual]);
        self.input.attention_metadata.context_lens[..actual]
            .copy_from_slice(&new_input.attention_metadata.context_lens[..actual]);

        // query_lens are always 1 for decode, already set
        // block_tables: copy row by row
        for i in 0..actual {
            let src = &new_input.attention_metadata.block_tables[i];
            let dst = &mut self.input.attention_metadata.block_tables[i];
            let copy_len = src.len().min(dst.len());
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
            // Extend dst if src is longer
            if src.len() > dst.len() {
                dst.extend_from_slice(&src[dst.len()..]);
            }
        }

        // Update max_context_len
        self.input.attention_metadata.max_context_len = self
            .input
            .attention_metadata
            .context_lens
            .iter()
            .copied()
            .max()
            .unwrap_or(0);

        // Zero padding entries (keep them as benign values)
        for i in actual..padded {
            self.input.token_ids[i] = 0;
            self.input.position_ids[i] = 0;
            self.input.attention_metadata.slot_mapping[i] = 0;
            self.input.attention_metadata.context_lens[i] = 1;
        }

        trace!(actual, padded, "graph input buffers updated in-place");
    }
}

/// Cached output from the last forward pass through a captured graph.
struct GraphOutputCache {
    /// The logits from the last forward (padded batch size * vocab_size).
    logits: Vec<f32>,
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
    /// Persistent input buffers per padded batch size, kept at stable heap
    /// addresses so CUDA graph memcpy nodes can re-read from them on replay.
    input_buffers: HashMap<usize, GraphInputBuffers>,
    /// Cached logits output per padded batch size from the last forward/replay.
    output_cache: HashMap<usize, GraphOutputCache>,
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
            input_buffers: HashMap::new(),
            output_cache: HashMap::new(),
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

        // Pad with dummy entries
        for _ in 0..pad_count {
            token_ids.push(0);
            position_ids.push(0);
            slot_mapping.push(0);
            context_lens.push(1);
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

    /// Store a padded ModelInput as the persistent input buffer for a batch
    /// size. Must be called before `capture_decode_graph` so the buffer is
    /// allocated at a stable heap address before capture begins.
    pub fn store_input_buffer(&mut self, padded_input: ModelInput, padded_bs: usize) {
        debug!(padded_bs, "storing persistent graph input buffer");
        self.input_buffers
            .insert(padded_bs, GraphInputBuffers::new(padded_input, padded_bs));
    }

    /// Get a reference to the persistent input for a padded batch size.
    pub fn get_input_buffer(&self, padded_bs: usize) -> Option<&ModelInput> {
        self.input_buffers.get(&padded_bs).map(|b| &b.input)
    }

    /// Update the persistent input buffer in-place for graph replay.
    /// The Vec backing stores keep their heap addresses so CUDA graph memcpy
    /// nodes read fresh data from the same pointers on replay.
    pub fn update_input_buffer(
        &mut self,
        padded_bs: usize,
        new_input: &ModelInput,
        actual_batch: usize,
    ) -> Result<()> {
        let buf = self.input_buffers.get_mut(&padded_bs).ok_or_else(|| {
            LLMError::GpuError(format!(
                "no persistent input buffer for padded_bs={}",
                padded_bs
            ))
        })?;
        buf.update_in_place(new_input, actual_batch);
        Ok(())
    }

    /// Cache the logits output from a forward pass for a padded batch size.
    pub fn cache_output(&mut self, padded_bs: usize, logits: Vec<f32>) {
        trace!(
            padded_bs,
            logits_len = logits.len(),
            "caching graph output"
        );
        self.output_cache.insert(
            padded_bs,
            GraphOutputCache { logits },
        );
    }

    /// Retrieve cached logits for a padded batch size, unpadded to actual_batch.
    pub fn get_cached_output(&self, padded_bs: usize, actual_batch: usize) -> Option<Vec<f32>> {
        self.output_cache.get(&padded_bs).map(|cache| {
            self.unpad_logits(&cache.logits, actual_batch)
        })
    }

    /// Capture a graph by running a forward pass on `stream`, then store it.
    ///
    /// `forward_fn` should execute the full decode forward pass for the given
    /// padded input. All kernel launches during `forward_fn` are captured.
    /// The forward_fn must use the persistent input buffers (from
    /// `get_input_buffer`) so that CUDA graph memcpy nodes reference stable
    /// host addresses.
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

    /// Capture a CUDA graph for decode, returning the logits from the capture
    /// forward pass. This variant stores the output in the cache.
    ///
    /// `forward_fn` must return the full logits Vec from the forward pass.
    pub fn capture_decode_graph<F>(
        &mut self,
        stream: &GpuStream,
        padded_batch_size: usize,
        forward_fn: F,
    ) -> Result<Vec<f32>>
    where
        F: Fn() -> Result<Vec<f32>>,
    {
        if self.was_capture_attempted(padded_batch_size) {
            trace!(
                padded_batch_size,
                "graph capture already attempted, skipping"
            );
            // Return cached output if available
            if let Some(cache) = self.output_cache.get(&padded_batch_size) {
                return Ok(cache.logits.clone());
            }
            return Err(LLMError::GpuError(
                "capture attempted but no cached output".into(),
            ));
        }

        info!(padded_batch_size, "capturing CUDA graph for decode");

        // Warm up: run forward once without capture to JIT-compile kernels.
        let warmup_logits = forward_fn()?;
        stream.synchronize()?;

        // Capture: run forward again under stream capture.
        self.pool.begin_capture(stream)?;
        let capture_logits = forward_fn()?;
        let graph = self.pool.end_capture(stream, padded_batch_size)?;
        self.pool.insert(graph);
        self.mark_captured(padded_batch_size);

        // Cache the output from the capture run.
        self.cache_output(padded_batch_size, capture_logits.clone());

        info!(padded_batch_size, "CUDA graph captured successfully");
        // Return the warmup logits (capture logits may have stale data from
        // the graph instantiation, warmup is the real result).
        Ok(warmup_logits)
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

    /// Full replay cycle: update persistent inputs, replay the graph, return
    /// unpadded logits from the output cache.
    ///
    /// Returns `Some(logits)` if replay succeeded with cached output available,
    /// `None` if the caller should fall back to the normal forward path.
    ///
    /// On mock/no-op builds where the pool does not do real capture, the output
    /// cache will be empty (no logits cached) so this returns `None` and the
    /// caller runs the real forward. On real `cuda-graphs` builds, the output
    /// cache is populated during capture and refreshed on replay.
    pub fn try_replay_decode(
        &mut self,
        stream: &GpuStream,
        model_input: &ModelInput,
        actual_batch: usize,
    ) -> Result<Option<Vec<f32>>> {
        let padded_bs = match padded_batch_size(actual_batch) {
            Some(p) => p,
            None => return Ok(None),
        };

        // Must have both a captured graph and cached output to replay.
        if !self.pool.has_graph(padded_bs) {
            return Ok(None);
        }
        if !self.output_cache.contains_key(&padded_bs) {
            // Graph exists but no output cache -- mock build or capture that
            // didn't produce cacheable output. Fall through to real forward.
            return Ok(None);
        }

        // Update persistent input buffers so the graph's memcpy nodes
        // pick up fresh data from the same host addresses on replay.
        if self.input_buffers.contains_key(&padded_bs) {
            self.update_input_buffer(padded_bs, model_input, actual_batch)?;
        }

        // Replay the CUDA graph (re-executes all captured kernels).
        let replayed = self.try_replay(stream, actual_batch)?;
        if !replayed {
            return Ok(None);
        }

        // Synchronize to ensure the replayed graph (including any DtoH of
        // logits) has completed before reading the output.
        stream.synchronize()?;

        // Return the cached output unpadded to actual batch size.
        match self.get_cached_output(padded_bs, actual_batch) {
            Some(logits) => {
                debug!(
                    actual_batch,
                    padded_bs,
                    logits_len = logits.len(),
                    "graph replay complete, returning logits"
                );
                Ok(Some(logits))
            }
            None => {
                warn!(padded_bs, "graph replayed but output cache miss");
                Ok(None)
            }
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

    /// Access the configuration.
    pub fn config(&self) -> &GraphRunnerConfig {
        &self.config
    }

    /// Clear all cached graphs (e.g., after model reload).
    pub fn clear(&mut self) {
        self.pool.clear();
        self.captured.clear();
        self.input_buffers.clear();
        self.output_cache.clear();
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
        let input = make_decode_input(64);
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

    #[test]
    fn store_and_retrieve_input_buffer() {
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());
        let input = make_decode_input(4);

        runner.store_input_buffer(input.clone(), 4);

        let stored = runner.get_input_buffer(4).unwrap();
        assert_eq!(stored.token_ids, input.token_ids);
        assert_eq!(stored.position_ids, input.position_ids);

        assert!(runner.get_input_buffer(8).is_none());
    }

    #[test]
    fn update_input_buffer_in_place() {
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());
        let input = make_decode_input(4);
        runner.store_input_buffer(input, 4);

        // Create new input with different values
        let mut new_input = make_decode_input(3);
        new_input.token_ids = vec![99, 100, 101];
        new_input.position_ids = vec![20, 21, 22];

        runner.update_input_buffer(4, &new_input, 3).unwrap();

        let stored = runner.get_input_buffer(4).unwrap();
        // First 3 entries updated
        assert_eq!(stored.token_ids[0], 99);
        assert_eq!(stored.token_ids[1], 100);
        assert_eq!(stored.token_ids[2], 101);
        // Fourth entry is padding (zero)
        assert_eq!(stored.token_ids[3], 0);
        assert_eq!(stored.position_ids[0], 20);
        assert_eq!(stored.position_ids[3], 0);
    }

    #[test]
    fn cache_and_retrieve_output() {
        let mut runner = GraphRunner::new(GraphRunnerConfig {
            vocab_size: 4,
            ..Default::default()
        });

        let logits: Vec<f32> = (0..16).map(|i| i as f32).collect();
        runner.cache_output(4, logits.clone());

        // Retrieve unpadded for actual_batch=3
        let unpadded = runner.get_cached_output(4, 3).unwrap();
        assert_eq!(unpadded.len(), 12); // 3 * vocab_size(4)
        assert_eq!(unpadded, &logits[..12]);

        // Retrieve full for actual_batch=4
        let full = runner.get_cached_output(4, 4).unwrap();
        assert_eq!(full.len(), 16);

        // Non-existent batch size
        assert!(runner.get_cached_output(8, 3).is_none());
    }

    #[test]
    fn capture_decode_graph_caches_output() {
        let stream = GpuStream::new(0).unwrap();
        let mut runner = GraphRunner::new(GraphRunnerConfig {
            vocab_size: 4,
            ..Default::default()
        });

        let logits: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let logits_clone = logits.clone();

        let result = runner
            .capture_decode_graph(&stream, 4, || Ok(logits_clone.clone()))
            .unwrap();

        // Returns warmup logits
        assert_eq!(result.len(), 16);

        // Output is cached
        assert!(runner.get_cached_output(4, 4).is_some());
        assert!(runner.was_capture_attempted(4));
        assert!(runner.has_graph_for(4));
    }

    #[test]
    fn try_replay_decode_no_graph_returns_none() {
        let stream = GpuStream::new(0).unwrap();
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());
        let input = make_decode_input(4);

        let result = runner.try_replay_decode(&stream, &input, 4).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn try_replay_decode_no_output_cache_returns_none() {
        let stream = GpuStream::new(0).unwrap();
        let mut runner = GraphRunner::new(GraphRunnerConfig::default());

        // Capture with old API (no output caching)
        runner.capture_graph(&stream, 4, || Ok(())).unwrap();
        assert!(runner.has_graph_for(4));

        let input = make_decode_input(4);
        let result = runner.try_replay_decode(&stream, &input, 4).unwrap();
        // No cached output, so returns None
        assert!(result.is_none());
    }

    #[test]
    fn clear_removes_input_and_output_caches() {
        let stream = GpuStream::new(0).unwrap();
        let mut runner = GraphRunner::new(GraphRunnerConfig {
            vocab_size: 4,
            ..Default::default()
        });

        let input = make_decode_input(4);
        runner.store_input_buffer(input, 4);
        runner.cache_output(4, vec![0.0; 16]);
        runner.capture_graph(&stream, 4, || Ok(())).unwrap();

        runner.clear();

        assert!(runner.get_input_buffer(4).is_none());
        assert!(runner.get_cached_output(4, 4).is_none());
        assert!(!runner.has_graph_for(4));
        assert!(!runner.was_capture_attempted(4));
    }

    #[test]
    fn config_accessor() {
        let runner = GraphRunner::new(GraphRunnerConfig {
            vocab_size: 128,
            hidden_size: 256,
            ..Default::default()
        });
        assert_eq!(runner.config().vocab_size, 128);
        assert_eq!(runner.config().hidden_size, 256);
    }
}
