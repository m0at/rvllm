use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use tracing::{trace, info, warn};

use rvllm_core::prelude::LLMError;

use crate::input::InputBuilder;
use crate::kv_cache::{CudaKVCache, KVCacheEngine};
use crate::runner::GpuModelRunner;
use rvllm_gpu::cuda_graph::{CudaGraphPool, padded_batch_size};

use crate::types::{
    BlockOps, ForwardOutput, GpuBatchInput, RequestId, SamplingParams, StepDiff,
    TokenId, WorkerRequest,
};

// ===================================================================
// WorkerConfig
// ===================================================================

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub block_size: usize,
    pub max_batch_size: usize,
    pub vocab_size: usize,
    pub graph_enabled: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            max_batch_size: 256,
            vocab_size: 32000,
            graph_enabled: true,
        }
    }
}

// ===================================================================
// WorkerError
// ===================================================================

#[derive(Debug)]
pub enum WorkerError {
    Cuda(String),
    Runner(String),
    NoPending,
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerError::Cuda(s) => write!(f, "CUDA error: {s}"),
            WorkerError::Runner(s) => write!(f, "runner error: {s}"),
            WorkerError::NoPending => write!(f, "no pending forward output"),
        }
    }
}

impl std::error::Error for WorkerError {}

pub type Result<T> = std::result::Result<T, WorkerError>;

impl From<LLMError> for WorkerError {
    fn from(e: LLMError) -> Self {
        WorkerError::Runner(e.to_string())
    }
}

// ===================================================================
// Constants
// ===================================================================

/// Number of forward calls before we begin graph capture.
const GRAPH_WARMUP_CALLS: usize = 3;

// ===================================================================
// PendingForward -- tracks an async launch for later collection
// ===================================================================

struct PendingForward {
    num_seqs: usize,
    is_decode: bool,
    stored_output: Option<ForwardOutput>,
}

// ===================================================================
// Worker: the stateful GPU worker
// ===================================================================

pub struct Worker {
    // GPU infrastructure
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    runner: GpuModelRunner,
    kv_cache: CudaKVCache,

    // Persistent state -- vLLM-0.19 style
    requests: HashMap<RequestId, WorkerRequest>,
    block_size: usize,

    // Input builder (reusable, cleared each step)
    input_builder: InputBuilder,

    // Graph capture state
    forward_count: usize,
    graph_enabled: bool,

    // Async pipeline state
    pending: Option<PendingForward>,

    // CUDA graph pool for decode
    graph_pool: CudaGraphPool,
    warmup_count: HashMap<usize, usize>,

    // Sampler config
    vocab_size: usize,
}

impl Worker {
    pub fn new(
        mut runner: GpuModelRunner,
        kv_cache: CudaKVCache,
        input_builder: InputBuilder,
        config: WorkerConfig,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Self {
        let graph_pool = CudaGraphPool::new(config.max_batch_size);

        // Pre-allocate cuBLAS workspace for graph capture
        if config.graph_enabled {
            if let Err(e) = runner.prepare_for_graph_capture() {
                warn!("cuBLAS graph workspace alloc failed, disabling graphs: {e}");
            } else {
                info!("cuBLAS workspace ready for CUDA graph capture");
            }
        }

        Self {
            context,
            stream,
            runner,
            kv_cache,
            requests: HashMap::new(),
            block_size: config.block_size,
            input_builder,
            forward_count: 0,
            graph_enabled: config.graph_enabled,
            pending: None,
            graph_pool,
            warmup_count: HashMap::new(),
            vocab_size: config.vocab_size,
        }
    }

    // =================================================================
    // Primary step: diff in, ForwardOutput out
    // =================================================================

    /// Execute one step: apply diff, run block ops, build input, forward, sample.
    pub fn step(&mut self, diff: &StepDiff) -> Result<ForwardOutput> {
        // 1. Apply diff to persistent state
        self.apply_diff(diff);

        // 2. Execute block operations (copies, swaps) before forward
        self.execute_block_ops(&diff.block_ops)?;

        // 3. Early out if nothing to forward
        if self.requests.is_empty() {
            return Ok(ForwardOutput::default());
        }

        // 4. Build GPU input from persistent state (borrow, don't clone)
        let input_ref = self.input_builder.build(&self.requests, self.block_size);
        let is_all_decode = input_ref.is_all_decode;
        let num_seqs = input_ref.num_seqs;

        // 5. Fast path: CUDA graph replay (no clone needed -- disjoint field borrows)
        if is_all_decode && self.graph_enabled {
            if let Some(padded) = padded_batch_size(num_seqs) {
                if self.graph_pool.has_graph(num_seqs) {
                    self.runner.upload_metadata_padded(input_ref, padded)?;
                    let graph = self.graph_pool.get(num_seqs).unwrap();
                    graph.replay_on(self.runner.cuda_stream())
                        .map_err(|e| WorkerError::Runner(format!("graph replay: {e}")))?;
                    let token_ids_i32 = self.runner.read_graph_output(num_seqs)?;
                    let token_ids: Vec<TokenId> =
                        token_ids_i32.iter().map(|&t| t as TokenId).collect();
                    self.forward_count += 1;
                    return Ok(ForwardOutput { token_ids, logprobs: Vec::new() });
                }
            }
        }

        // Slow path: clone input for methods that take &mut self on Worker
        let input = input_ref.clone();

        if input.is_all_decode {
            let actual_batch = input.num_seqs;

            // Try CUDA graph capture for decode batches
            if self.graph_enabled {
                if let Some(padded) = padded_batch_size(actual_batch) {
                    let count = self.warmup_count.entry(padded).or_insert(0);
                    *count += 1;
                    if *count == GRAPH_WARMUP_CALLS {
                        match self.capture_decode_graph(&input, actual_batch, padded) {
                            Ok(output) => {
                                self.forward_count += 1;
                                return Ok(output);
                            }
                            Err(e) => {
                                warn!(padded, "graph capture failed, disabling for this bucket: {e}");
                            }
                        }
                    }
                }
            }

            // Normal forward (pre-graph-capture warmup)
            let token_ids_i32 = self.runner
                .forward_greedy(&input, &self.kv_cache)
                .map_err(|e| WorkerError::Runner(format!("{e}")))?;
            let token_ids: Vec<TokenId> = token_ids_i32.iter().map(|&t| t as TokenId).collect();
            Ok(ForwardOutput {
                token_ids,
                logprobs: Vec::new(),
            })
        } else {
            // Mixed batch (prefill + decode): still use GPU argmax to avoid massive logits DtoH
            let token_ids_i32 = self.runner
                .forward_greedy(&input, &self.kv_cache)
                .map_err(|e| WorkerError::Runner(format!("{e}")))?;
            let token_ids: Vec<TokenId> = token_ids_i32.iter().map(|&t| t as TokenId).collect();
            Ok(ForwardOutput {
                token_ids,
                logprobs: Vec::new(),
            })
        }
    }

    // =================================================================
    // Async pipeline: launch / collect
    // =================================================================

    /// Launch the forward pass asynchronously. Returns immediately.
    /// For decode: enqueues all GPU work (no DtoH, no sync) for true overlap.
    /// For mixed: runs synchronously and stores the result.
    /// Caller must call step_collect() to get the result.
    pub fn step_launch(&mut self, diff: &StepDiff) -> Result<()> {
        self.apply_diff(diff);
        self.execute_block_ops(&diff.block_ops)?;

        if self.requests.is_empty() {
            self.pending = None;
            return Ok(());
        }

        let input = self.input_builder.build(&self.requests, self.block_size).clone();
        let num_seqs = input.num_seqs;

        if input.is_all_decode {
            // True async: enqueue GPU work only (no DtoH, no sync)
            if self.graph_enabled {
                if let Some(padded) = padded_batch_size(num_seqs) {
                    if self.graph_pool.has_graph(num_seqs) {
                        self.runner.upload_metadata_padded(&input, padded)?;
                        let graph = self.graph_pool.get(num_seqs).unwrap();
                        graph.replay_on(self.runner.cuda_stream())
                            .map_err(|e| WorkerError::Runner(format!("graph replay: {e}")))?;
                        self.forward_count += 1;
                        self.pending = Some(PendingForward { num_seqs, is_decode: true, stored_output: None });
                        return Ok(());
                    }
                }
            }
            // No graph available: launch forward async (GPU kernels only, no DtoH)
            self.runner.forward_greedy_launch(&input, &self.kv_cache)
                .map_err(|e| WorkerError::Runner(format!("{e}")))?;
            self.forward_count += 1;
            self.pending = Some(PendingForward { num_seqs, is_decode: true, stored_output: None });
        } else {
            // Mixed prefill+decode: GPU argmax (no full logits DtoH)
            self.runner.forward_greedy_launch(&input, &self.kv_cache)
                .map_err(|e| WorkerError::Runner(format!("{e}")))?;
            self.forward_count += 1;
            self.pending = Some(PendingForward { num_seqs, is_decode: true, stored_output: None });
        }

        Ok(())
    }

    /// Collect the result of a previously launched step.
    /// For decode: reads GPU argmax results (DtoH + sync).
    /// For mixed: returns the pre-computed result from step_launch.
    pub fn step_collect(&mut self) -> Result<ForwardOutput> {
        let pending = self
            .pending
            .take()
            .ok_or(WorkerError::NoPending)?;

        if pending.num_seqs == 0 {
            return Ok(ForwardOutput::default());
        }

        if pending.is_decode {
            // Just read the argmax token IDs (DtoH + sync)
            let token_ids_i32 = self.runner.read_graph_output(pending.num_seqs)
                .map_err(|e| WorkerError::Runner(format!("{e}")))?;
            let token_ids: Vec<TokenId> =
                token_ids_i32.iter().map(|&t| t as TokenId).collect();
            Ok(ForwardOutput { token_ids, logprobs: Vec::new() })
        } else {
            // Mixed: result already computed in step_launch
            Ok(pending.stored_output.unwrap_or_default())
        }
    }

    pub fn requests(&self) -> &HashMap<RequestId, WorkerRequest> {
        &self.requests
    }

    pub fn num_active_requests(&self) -> usize {
        self.requests.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn forward_count(&self) -> usize {
        self.forward_count
    }

    /// Apply diff only (for spec decode: separate diff from forward).
    pub fn step_apply_diff(&mut self, diff: &StepDiff) -> Result<()> {
        self.apply_diff(diff);
        self.execute_block_ops(&diff.block_ops)
    }

    /// Run forward only on current request state (for spec decode fallback).
    pub fn step_forward_only(&mut self) -> Result<ForwardOutput> {
        if self.requests.is_empty() {
            return Ok(ForwardOutput::default());
        }
        let input = self.input_builder.build(&self.requests, self.block_size).clone();
        if input.is_all_decode {
            let token_ids_i32 = self.runner
                .forward_greedy(&input, &self.kv_cache)
                .map_err(|e| WorkerError::Runner(format!("{e}")))?;
            let token_ids: Vec<TokenId> = token_ids_i32.iter().map(|&t| t as TokenId).collect();
            self.forward_count += 1;
            Ok(ForwardOutput { token_ids, logprobs: Vec::new() })
        } else {
            self.forward_count += 1;
            let logits = self.execute_forward(&input)?;
            Ok(self.sample_greedy(&logits, &input))
        }
    }

    // =================================================================
    // apply_diff: the core state mutation
    // =================================================================

    fn apply_diff(&mut self, diff: &StepDiff) {
        // Remove finished/preempted requests FIRST so IDs can be reused
        for request_id in &diff.removed {
            if self.requests.remove(request_id).is_some() {
                trace!(%request_id, "removed request from worker state");
            }
        }

        // Add new requests (prefill phase)
        for added in &diff.added {
            let req = WorkerRequest {
                request_id: added.request_id,
                seq_id: added.seq_id,
                prompt_token_ids: added.prompt_token_ids.clone(),
                output_token_ids: Vec::new(),
                sampling_params: added.sampling_params.clone(),
                block_table: added.block_table.clone(),
                is_prefill: added.is_prefill,
                num_computed_tokens: 0,
                token_chunk: added.token_chunk.clone(),
            };
            trace!(
                request_id = %added.request_id,
                seq_id = %added.seq_id,
                prompt_len = added.prompt_token_ids.len(),
                "added request to worker state"
            );
            self.requests.insert(added.request_id, req);
        }

        // Update continuing decode requests
        for cont in &diff.continued {
            if let Some(req) = self.requests.get_mut(&cont.request_id) {
                if cont.new_token_id != 0 {
                    req.output_token_ids.push(cont.new_token_id);
                }
                if let Some(new_table) = &cont.block_table_update {
                    req.block_table.clone_from(new_table);
                }
                req.is_prefill = false;
                req.num_computed_tokens = req.seq_len();
                req.token_chunk = 0..0;
                trace!(
                    request_id = %cont.request_id,
                    new_token = cont.new_token_id,
                    seq_len = req.seq_len(),
                    "continued request in worker state"
                );
            }
        }
    }

    // =================================================================
    // Block operations
    // =================================================================

    fn execute_block_ops(&mut self, ops: &BlockOps) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        if !ops.copies.is_empty() {
            trace!(count = ops.copies.len(), "executing CoW block copies");
            self.kv_cache.copy_blocks(&ops.copies);
        }

        if !ops.swap_in.is_empty() {
            trace!(count = ops.swap_in.len(), "executing swap-in (CPU->GPU)");
            self.kv_cache.swap_in(&ops.swap_in);
        }

        if !ops.swap_out.is_empty() {
            trace!(count = ops.swap_out.len(), "executing swap-out (GPU->CPU)");
            self.kv_cache.swap_out(&ops.swap_out);
        }

        Ok(())
    }

    // =================================================================
    // Forward pass
    // =================================================================

    fn execute_forward(&mut self, input: &GpuBatchInput) -> Result<Vec<f32>> {
        self.forward_count += 1;

        // Raw forward through the model runner. The runner handles:
        // - Embedding lookup
        // - Packed metadata upload (single HtoD memcpy)
        // - Layer loop with double-buffered scratch
        // - Final RMSNorm + LM head GEMM
        // - DtoH logits transfer
        self.runner
            .forward(input, &self.kv_cache)
            .map_err(|e| WorkerError::Runner(format!("{e}")))
    }

    // =================================================================
    // Sampling: greedy argmax on CPU (simplest correct path)
    // =================================================================

    fn sample_greedy(&self, logits: &[f32], input: &GpuBatchInput) -> ForwardOutput {
        let vocab = self.vocab_size;
        let num_seqs = input.num_seqs;
        let mut token_ids = Vec::with_capacity(num_seqs);
        let mut logprobs = Vec::with_capacity(num_seqs);

        for i in 0..num_seqs {
            let start = i * vocab;
            let end = (start + vocab).min(logits.len());
            if start >= logits.len() {
                token_ids.push(0);
                logprobs.push(f32::NEG_INFINITY);
                continue;
            }

            let seq_logits = &logits[start..end];
            let (best_idx, best_logit) = seq_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &f32::NEG_INFINITY));

            token_ids.push(best_idx as TokenId);
            logprobs.push(*best_logit);
        }

        ForwardOutput { token_ids, logprobs }
    }

    // =================================================================
    // CUDA graph capture
    // =================================================================

    fn capture_decode_graph(
        &mut self,
        input: &GpuBatchInput,
        actual_batch: usize,
        padded_batch: usize,
    ) -> Result<ForwardOutput> {
        // Cap context for graph capture: use min(max_seq_len, 2048) to avoid
        // excessive split-K (choose_num_splits(32768) = 16, wasteful for short contexts).
        // Graphs are recaptured if actual context exceeds this.
        let max_ctx = self.runner.max_seq_len().min(2048) as u32;

        // Warmup forward (stabilizes cuBLAS algorithm selection)
        self.runner.upload_metadata_padded(input, padded_batch)?;
        self.runner
            .forward_gpu_only(padded_batch, padded_batch, max_ctx, &self.kv_cache)?;
        self.sync_stream()?;

        // Re-upload metadata for capture
        self.runner.upload_metadata_padded(input, padded_batch)?;

        // Begin capture
        self.graph_pool
            .begin_capture_on(self.runner.cuda_stream())
            .map_err(|e| WorkerError::Runner(format!("begin_capture: {e}")))?;

        // Forward inside capture region
        let fwd_result = self.runner.forward_gpu_only(
            padded_batch,
            padded_batch,
            max_ctx,
            &self.kv_cache,
        );

        match fwd_result {
            Ok(()) => {
                let graph = self
                    .graph_pool
                    .end_capture_on(self.runner.cuda_stream(), padded_batch)
                    .map_err(|e| WorkerError::Runner(format!("end_capture: {e}")))?;
                self.graph_pool.insert(graph);
                info!(padded_batch, actual_batch, "CUDA graph captured for decode");

                // Read output from the warmup+capture forward
                let token_ids_i32 = self.runner.read_graph_output(actual_batch)?;
                let token_ids: Vec<TokenId> =
                    token_ids_i32.iter().map(|&t| t as TokenId).collect();
                Ok(ForwardOutput {
                    token_ids,
                    logprobs: Vec::new(),
                })
            }
            Err(e) => {
                // End capture to clean up stream state
                let _ = self
                    .graph_pool
                    .end_capture_on(self.runner.cuda_stream(), padded_batch);
                warn!(padded_batch, "graph capture forward failed: {e}");
                Err(WorkerError::Runner(format!(
                    "graph capture forward failed: {e}"
                )))
            }
        }
    }

    // =================================================================
    // Stream sync
    // =================================================================

    fn sync_stream(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| WorkerError::Cuda(format!("stream sync: {e}")))
    }

    /// Synchronize the compute stream. Blocks until all enqueued GPU work completes.
    pub fn sync(&self) -> Result<()> {
        self.sync_stream()
    }

    /// Run forward_greedy with a custom input (e.g., spec decode verify batch).
    pub fn forward_greedy_custom(&mut self, input: &GpuBatchInput) -> Result<Vec<i32>> {
        self.runner
            .forward_greedy(input, &self.kv_cache)
            .map_err(|e| WorkerError::Runner(format!("{e}")))
    }

    /// Access the KV cache (e.g., for spec decode).
    pub fn kv_cache(&self) -> &CudaKVCache {
        &self.kv_cache
    }

    // =================================================================
    // Pre-capture CUDA graph buckets
    // =================================================================

    /// Pre-capture decode graphs for all standard batch-size buckets.
    /// Call once after model load, before serving requests.
    pub fn pre_capture_decode_graphs(&mut self, max_batch: usize) -> Result<()> {
        use rvllm_gpu::cuda_graph::GRAPH_BATCH_SIZES;

        if !self.graph_enabled {
            info!("graphs disabled, skipping pre-capture");
            return Ok(());
        }

        let max_ctx = self.runner.max_seq_len().min(2048) as u32;

        for &bucket in GRAPH_BATCH_SIZES {
            if bucket > max_batch {
                break;
            }

            // Build dummy decode input for this bucket size
            let dummy = GpuBatchInput {
                num_seqs: bucket,
                num_prefill_seqs: 0,
                num_decode_seqs: bucket,
                seq_ids: (0..bucket as u64).collect(),
                token_ids: vec![0; bucket],
                position_ids: vec![0; bucket],
                slot_mapping: vec![0; bucket],
                context_lens: vec![1; bucket],
                query_lens: vec![1; bucket],
                sampling_params: vec![SamplingParams::default(); bucket],
                block_tables_flat: vec![0; bucket * 8],
                max_blocks_per_seq: 8,
                is_all_decode: true,
                is_all_prefill: false,
                max_context_len: max_ctx,
                prefill_tokens: Vec::new(),
                prefill_positions: Vec::new(),
                prefill_slot_mapping: Vec::new(),
            };

            // Warmup forward
            self.runner.upload_metadata_padded(&dummy, bucket)?;
            self.runner
                .forward_gpu_only(bucket, bucket, max_ctx, &self.kv_cache)
                .map_err(|e| WorkerError::Runner(format!("graph warmup: {e}")))?;
            self.sync_stream()?;

            // Capture
            self.runner.upload_metadata_padded(&dummy, bucket)?;
            self.graph_pool
                .begin_capture_on(self.runner.cuda_stream())
                .map_err(|e| WorkerError::Runner(format!("begin_capture: {e}")))?;

            match self
                .runner
                .forward_gpu_only(bucket, bucket, max_ctx, &self.kv_cache)
            {
                Ok(()) => {
                    let graph = self
                        .graph_pool
                        .end_capture_on(self.runner.cuda_stream(), bucket)
                        .map_err(|e| WorkerError::Runner(format!("end_capture: {e}")))?;
                    self.graph_pool.insert(graph);
                    trace!(bucket, "pre-captured decode graph");
                }
                Err(e) => {
                    let _ = self
                        .graph_pool
                        .end_capture_on(self.runner.cuda_stream(), bucket);
                    warn!(bucket, "graph pre-capture failed: {e}");
                }
            }
        }

        info!(
            count = self.graph_pool.len(),
            "decode graph pre-capture complete"
        );
        Ok(())
    }
}
