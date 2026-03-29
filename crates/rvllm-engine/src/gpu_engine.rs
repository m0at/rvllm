//! GPU-accelerated inference engine composing scheduler, GPU worker, and tokenizer.
//!
//! Reads model config.json from HuggingFace cache to set correct architecture
//! parameters, loads weights to GPU, and drives the scheduling/output loop.
//!
//! Gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::{HashMap, VecDeque};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use tracing::{debug, info, trace, warn};

    use rvllm_block_manager::prefix_cache::{self, PrefixCache};
    use rvllm_block_manager::{BlockManager, MemoryPool};
    use rvllm_config::EngineConfig;
    use rvllm_core::prelude::{
        BlockId, FinishReason, LLMError, LogProb, RequestId, RequestOutput, Result, SamplingParams,
        SequenceId, TokenId,
    };
    use rvllm_sequence::{
        Sequence, SequenceData, SequenceGroup, SequenceGroupMetadata, SequenceStatus,
    };
    use rvllm_tokenizer::Tokenizer;
    use rvllm_worker::gpu_worker::{GpuWorker, GpuWorkerOutput};

    use crate::beam_search::{top_k_from_logprobs, BeamHypothesis, BeamSearchState};
    use crate::hf_snapshot;
    use crate::output::{OutputProcessor, SequenceOutputState};

    #[derive(Debug)]
    struct EngineBlockPool {
        free_list: Mutex<VecDeque<BlockId>>,
        total: usize,
    }

    impl EngineBlockPool {
        fn new(total: usize) -> Self {
            let mut free_list = VecDeque::with_capacity(total);
            for idx in 0..total {
                free_list.push_back(BlockId(idx as u32));
            }
            Self {
                free_list: Mutex::new(free_list),
                total,
            }
        }
    }

    impl MemoryPool for EngineBlockPool {
        fn allocate(&self) -> Option<BlockId> {
            self.free_list.lock().ok()?.pop_front()
        }

        fn free(&self, block_id: BlockId) {
            if let Ok(mut free_list) = self.free_list.lock() {
                free_list.push_back(block_id);
            }
        }

        fn free_blocks(&self) -> usize {
            self.free_list
                .lock()
                .map(|free_list| free_list.len())
                .unwrap_or(0)
        }

        fn total_blocks(&self) -> usize {
            self.total
        }
    }

    #[derive(Debug)]
    struct BeamGroupUpdate {
        output: RequestOutput,
        active_beams: Vec<BeamHypothesis>,
        fork_pairs: Vec<(SequenceId, SequenceId)>,
    }

    // ------------------------------------------------------------------
    // HuggingFace model config reading
    // ------------------------------------------------------------------

    /// Subset of fields from a HuggingFace config.json that we need.
    #[derive(Debug, Clone)]
    struct HfModelConfig {
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        num_hidden_layers: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        tie_word_embeddings: bool,
        architecture: String,
        rope_theta: f32,
        partial_rotary_factor: f32,
        attn_logit_softcapping: f32,
        num_local_experts: usize,
        num_experts_per_tok: usize,
    }

    fn resolve_model_dir(model_name: &str) -> Result<PathBuf> {
        hf_snapshot::ensure_snapshot(model_name)
    }

    fn read_model_config(model_dir: &Path) -> Result<HfModelConfig> {
        let config_path = model_dir.join("config.json");
        let content = std::fs::read_to_string(&config_path).map_err(|e| {
            LLMError::ModelError(format!("failed to read {}: {e}", config_path.display()))
        })?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| LLMError::ModelError(format!("invalid config.json: {e}")))?;

        let get_usize = |key: &str, default: usize| -> usize {
            json.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            json.get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default)
        };

        let hidden_size = get_usize("hidden_size", 4096);
        let num_attention_heads = get_usize("num_attention_heads", 32);
        let head_dim = hidden_size / num_attention_heads;

        let architecture = json
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("LlamaForCausalLM")
            .to_string();

        Ok(HfModelConfig {
            hidden_size,
            intermediate_size: get_usize("intermediate_size", 11008),
            num_attention_heads,
            num_key_value_heads: get_usize("num_key_value_heads", num_attention_heads),
            num_hidden_layers: get_usize("num_hidden_layers", 32),
            vocab_size: get_usize("vocab_size", 32000),
            rms_norm_eps: get_f32("rms_norm_eps", 1e-5),
            tie_word_embeddings: json
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            architecture,
            rope_theta: get_f32("rope_theta", 10000.0),
            partial_rotary_factor: get_f32("partial_rotary_factor", 1.0),
            attn_logit_softcapping: get_f32("attn_logit_softcapping", 0.0),
            num_local_experts: get_usize("num_local_experts", 0),
            num_experts_per_tok: get_usize("num_experts_per_tok", 0),
        })
    }

    // ------------------------------------------------------------------
    // Internal request bookkeeping
    // ------------------------------------------------------------------

    #[derive(Debug)]
    struct EngineRequest {
        #[allow(dead_code)]
        request_id: RequestId,
        prompt: String,
        prompt_token_ids: Vec<TokenId>,
        sampling_params: SamplingParams,
        seq_states: Vec<SequenceOutputState>,
        beam_search: Option<BeamSearchState>,
    }

    // ------------------------------------------------------------------
    // Minimal FIFO scheduler
    // ------------------------------------------------------------------

    struct FifoScheduler {
        /// Waiting queue: groups not yet scheduled.
        waiting: Vec<SequenceGroup>,
        /// Running queue: groups currently being executed (already scheduled).
        running: Vec<SequenceGroup>,
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
        /// Number of free GPU KV-cache blocks available for new admissions.
        /// Updated by the engine each step before scheduling.
        free_block_count: usize,
        /// Minimum free blocks required to admit a new sequence group.
        /// Prevents OOM by keeping a watermark so running sequences can grow.
        watermark_blocks: usize,
    }

    impl FifoScheduler {
        fn new(max_num_seqs: usize, max_num_batched_tokens: usize) -> Self {
            Self {
                waiting: Vec::new(),
                running: Vec::new(),
                max_num_seqs,
                max_num_batched_tokens,
                free_block_count: 0,
                watermark_blocks: 0,
            }
        }

        fn set_block_budget(&mut self, free_blocks: usize, watermark: usize) {
            self.free_block_count = free_blocks;
            self.watermark_blocks = watermark;
        }

        fn add_seq_group(&mut self, group: SequenceGroup) {
            self.waiting.push(group);
        }

        fn abort_seq_group(&mut self, request_id: &RequestId) {
            self.waiting.retain(|g| g.request_id != *request_id);
            self.running.retain(|g| g.request_id != *request_id);
        }

        fn cloned_seqs_for_request(&self, request_id: &RequestId) -> Vec<Sequence> {
            self.waiting
                .iter()
                .chain(self.running.iter())
                .find(|group| group.request_id == *request_id)
                .map(|group| group.get_seqs().to_vec())
                .unwrap_or_default()
        }

        fn has_unfinished_seqs(&self) -> bool {
            !self.waiting.is_empty() || !self.running.is_empty()
        }

        fn live_seq_ids(&self) -> std::collections::HashSet<SequenceId> {
            self.waiting
                .iter()
                .chain(self.running.iter())
                .flat_map(|g| g.get_seqs().iter().map(|s| s.seq_id))
                .collect()
        }

        /// Append a generated token to a sequence in the scheduler.
        fn update_seq_token(&mut self, seq_id: SequenceId, token_id: TokenId, logprob: f32) {
            for group in self.running.iter_mut().chain(self.waiting.iter_mut()) {
                for seq in group.get_seqs_mut() {
                    if seq.seq_id == seq_id {
                        seq.append_token(token_id, logprob);
                        return;
                    }
                }
            }
        }

        /// Replace a sequence's tracked state wholesale.
        fn replace_seq_state(
            &mut self,
            seq_id: SequenceId,
            prompt_token_ids: Vec<TokenId>,
            output_token_ids: Vec<TokenId>,
            cumulative_logprob: f32,
            status: SequenceStatus,
        ) {
            for group in self.running.iter_mut().chain(self.waiting.iter_mut()) {
                for seq in group.get_seqs_mut() {
                    if seq.seq_id == seq_id {
                        seq.prompt_token_ids = prompt_token_ids;
                        seq.output_token_ids = output_token_ids;
                        seq.cumulative_logprob = cumulative_logprob;
                        seq.logprobs.clear();
                        seq.num_computed_tokens = 0;
                        seq.status = status;
                        return;
                    }
                }
            }
        }

        /// Mark a sequence as finished so `schedule()` can purge it.
        fn finish_seq(&mut self, seq_id: SequenceId, status: SequenceStatus) {
            for group in self.running.iter_mut().chain(self.waiting.iter_mut()) {
                for seq in group.get_seqs_mut() {
                    if seq.seq_id == seq_id {
                        let _ = seq.set_status(status);
                        return;
                    }
                }
            }
        }

        fn get_num_unfinished_seq_groups(&self) -> usize {
            self.waiting.len() + self.running.len()
        }

        fn schedule(&mut self) -> (Vec<SequenceGroup>, usize) {
            // Purge finished groups from both queues.
            self.running.retain(|g| !g.is_finished());
            self.waiting.retain(|g| !g.is_finished());

            let mut total_tokens: usize = 0;

            // Running groups are always re-scheduled (they need their next token).
            for group in &self.running {
                let tokens_this: usize = group
                    .get_seqs()
                    .iter()
                    .filter(|s| !s.is_finished())
                    .map(|s| {
                        if s.get_output_len() == 0 {
                            s.get_len()
                        } else {
                            1
                        }
                    })
                    .sum();
                total_tokens += tokens_this;
            }

            // Promote waiting groups into running up to budget limits.
            // Also gate on block availability: don't admit new groups when
            // free blocks are below the watermark so running sequences have
            // room to grow during decode.
            while !self.waiting.is_empty() {
                if self.running.len() >= self.max_num_seqs {
                    break;
                }
                if self.free_block_count < self.watermark_blocks {
                    debug!(
                        free = self.free_block_count,
                        watermark = self.watermark_blocks,
                        "scheduler: holding back waiting groups -- below block watermark"
                    );
                    break;
                }
                let group = &self.waiting[0];
                let tokens_this: usize = group
                    .get_seqs()
                    .iter()
                    .filter(|s| !s.is_finished())
                    .map(|s| {
                        if s.get_output_len() == 0 {
                            s.get_len()
                        } else {
                            1
                        }
                    })
                    .sum();
                if total_tokens + tokens_this > self.max_num_batched_tokens {
                    break;
                }
                // Each new group needs at least 1 block per sequence.
                let seqs_in_group = group.get_seqs().iter().filter(|s| !s.is_finished()).count();
                if self.free_block_count < seqs_in_group {
                    break;
                }
                self.free_block_count = self.free_block_count.saturating_sub(seqs_in_group);
                total_tokens += tokens_this;
                self.running.push(self.waiting.remove(0));
            }

            // Return clones of running groups for execution.
            let selected: Vec<SequenceGroup> = self.running.iter().cloned().collect();
            (selected, total_tokens)
        }
    }

    // ------------------------------------------------------------------
    // GpuLLMEngine
    // ------------------------------------------------------------------

    /// Shared request queue for async overlap. The async engine pushes
    /// new requests here; the GPU engine drains them during GPU wait.
    pub type RequestQueue = std::sync::Arc<std::sync::Mutex<Vec<PendingRequest>>>;

    /// Shared abort queue. The async engine pushes request IDs to abort
    /// here; the GPU engine drains them at the start of each step.
    pub type AbortQueue = std::sync::Arc<std::sync::Mutex<Vec<RequestId>>>;

    /// A request buffered for async processing.
    pub struct PendingRequest {
        pub request_id: RequestId,
        pub prompt: String,
        pub params: SamplingParams,
    }

    pub struct GpuLLMEngine {
        config: EngineConfig,
        scheduler: FifoScheduler,
        worker: GpuWorker,
        tokenizer: Tokenizer,
        requests: HashMap<RequestId, EngineRequest>,
        next_request_id: std::sync::Arc<AtomicU64>,
        next_seq_id: u64,
        prefix_cache: Option<PrefixCache>,
        /// Total number of GPU KV-cache blocks available.
        num_gpu_blocks: u32,
        /// Shared GPU block ID pool used by both beam and non-beam paths.
        gpu_block_pool: Arc<EngineBlockPool>,
        /// Per-sequence block tables that persist across step() calls.
        seq_block_tables: HashMap<SequenceId, Vec<BlockId>>,
        /// Shared queue for new requests arriving during GPU compute.
        request_queue: Option<RequestQueue>,
        /// Shared queue for abort requests arriving during GPU compute.
        abort_queue: Option<AbortQueue>,
        /// Beam-search-only block manager for CoW KV cache sharing.
        beam_block_manager: BlockManager,
        /// Cache block copy operations to apply before the next worker step.
        pending_beam_copy_blocks: Vec<(BlockId, BlockId)>,
    }

    /// Pending state from step_launch, consumed by step_collect.
    pub struct StepPending {
        pub scheduled_groups: Vec<SequenceGroup>,
        pub metadata: Vec<SequenceGroupMetadata>,
        /// Async batch size from execute_launch (Some = async DtoH pending).
        pub actual_batch: Option<usize>,
    }

    impl GpuLLMEngine {
        pub fn new(config: EngineConfig) -> Result<Self> {
            let model_name = &config.model.model_path;
            info!(model = %model_name, "GpuLLMEngine: initializing");

            // 1. Resolve model directory
            let model_dir = resolve_model_dir(model_name)?;
            info!(model_dir = %model_dir.display(), "resolved model directory");

            // 2. Read model config.json
            let hf_config = read_model_config(&model_dir)?;
            info!(
                arch = %hf_config.architecture,
                hidden = hf_config.hidden_size,
                layers = hf_config.num_hidden_layers,
                heads = hf_config.num_attention_heads,
                kv_heads = hf_config.num_key_value_heads,
                vocab = hf_config.vocab_size,
                intermediate = hf_config.intermediate_size,
                "model config loaded"
            );

            let head_dim = hf_config.hidden_size / hf_config.num_attention_heads;

            // 3. Tokenizer
            let tokenizer_path = config.model.tokenizer_path.as_deref().unwrap_or(model_name);
            let tokenizer = Tokenizer::from_pretrained(tokenizer_path)?;

            // 4. Build WorkerConfig from real model config
            let worker_config = rvllm_worker::WorkerConfig {
                device_id: 0,
                num_layers: hf_config.num_hidden_layers,
                num_kv_heads: hf_config.num_key_value_heads,
                head_dim,
                hidden_size: hf_config.hidden_size,
                num_attention_heads: hf_config.num_attention_heads,
                intermediate_size: hf_config.intermediate_size,
                vocab_size: hf_config.vocab_size,
                max_model_len: config.model.max_model_len,
                block_size: config.cache.block_size,
                gpu_memory_utilization: config.cache.gpu_memory_utilization,
                rank: 0,
                tensor_parallel_size: config.parallel.tensor_parallel_size,
                pipeline_parallel_size: config.parallel.pipeline_parallel_size,
                architecture: hf_config.architecture.clone(),
                dtype: config.model.dtype,
                rms_norm_eps: hf_config.rms_norm_eps,
                rope_theta: hf_config.rope_theta,
                partial_rotary_factor: hf_config.partial_rotary_factor,
                attn_logit_softcapping: hf_config.attn_logit_softcapping,
                num_local_experts: hf_config.num_local_experts,
                num_experts_per_tok: hf_config.num_experts_per_tok,
                kv_cache_dtype: config.cache.kv_cache_dtype.clone(),
                enable_prefix_caching: config.cache.enable_prefix_caching,
            };

            // 5. Create GPU worker
            let mut worker = GpuWorker::new(worker_config)
                .map_err(|e| LLMError::GpuError(format!("GpuWorker creation failed: {e}")))?;

            // 6. Init model config
            worker.init_model(&model_dir)?;

            // 7. Load weights to GPU
            info!("loading model weights to GPU...");
            worker.load_weights(&model_dir)?;
            info!("model weights loaded successfully");

            // 8. Profile GPU memory and init KV cache + GPU model runner
            let (num_gpu_blocks, num_cpu_blocks) =
                worker.profile_num_available_blocks(config.cache.gpu_memory_utilization)?;
            worker.init_cache(num_gpu_blocks, num_cpu_blocks)?;

            // 8. Scheduler
            let scheduler = FifoScheduler::new(
                config.scheduler.max_num_seqs,
                config.scheduler.max_num_batched_tokens,
            );

            let prefix_cache = if config.cache.enable_prefix_caching {
                let block_size = config.cache.block_size;
                let max_cached = 1024; // max cached prefix blocks
                info!(block_size, max_cached, "prefix caching enabled");
                Some(PrefixCache::new(block_size, max_cached))
            } else {
                None
            };

            info!(
                num_gpu_blocks,
                num_cpu_blocks,
                block_size = config.cache.block_size,
                max_num_seqs = config.scheduler.max_num_seqs,
                "GpuLLMEngine: ready for inference"
            );
            let gpu_block_pool = Arc::new(EngineBlockPool::new(num_gpu_blocks));
            let beam_gpu_pool: Arc<dyn MemoryPool> = gpu_block_pool.clone();
            let beam_cpu_pool: Arc<dyn MemoryPool> = Arc::new(EngineBlockPool::new(num_cpu_blocks));
            let mut beam_block_manager =
                BlockManager::new(beam_gpu_pool, beam_cpu_pool, config.cache.block_size);
            beam_block_manager.set_watermark(0.0);
            if config.cache.enable_prefix_caching {
                beam_block_manager.enable_prefix_caching(1024);
            }
            Ok(Self {
                config,
                scheduler,
                worker,
                tokenizer,
                requests: HashMap::new(),
                next_request_id: std::sync::Arc::new(AtomicU64::new(1)),
                next_seq_id: 0,
                prefix_cache,
                num_gpu_blocks: num_gpu_blocks as u32,
                gpu_block_pool,
                seq_block_tables: HashMap::new(),
                request_queue: None,
                abort_queue: None,
                beam_block_manager,
                pending_beam_copy_blocks: Vec::new(),
            })
        }

        pub fn add_request(
            &mut self,
            request_id: RequestId,
            prompt: String,
            params: SamplingParams,
        ) -> Result<()> {
            info!(%request_id, prompt_len = prompt.len(), "GpuLLMEngine: add_request");

            let prompt_token_ids = self.tokenizer.encode(&prompt)?;
            debug!(%request_id, num_tokens = prompt_token_ids.len(), "prompt tokenized");

            if prompt_token_ids.is_empty() {
                return Err(LLMError::TokenizerError(
                    "prompt produced zero tokens".into(),
                ));
            }

            self.insert_request(request_id, prompt, prompt_token_ids, params)
        }

        pub fn add_request_auto_id(
            &mut self,
            prompt: String,
            params: SamplingParams,
        ) -> Result<RequestId> {
            let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
            self.add_request(request_id, prompt, params)?;
            Ok(request_id)
        }

        fn insert_request(
            &mut self,
            request_id: RequestId,
            prompt: String,
            prompt_token_ids: Vec<TokenId>,
            params: SamplingParams,
        ) -> Result<()> {
            let num_seqs = params.best_of.max(1);
            let mut seqs = Vec::with_capacity(num_seqs);
            let mut seq_states = Vec::with_capacity(num_seqs);
            let mut initial_seq_ids = Vec::with_capacity(num_seqs);

            for _ in 0..num_seqs {
                let seq_id = SequenceId(self.next_seq_id);
                self.next_seq_id += 1;
                seqs.push(Sequence::new(seq_id, prompt_token_ids.clone()));
                seq_states.push(SequenceOutputState::new());
                initial_seq_ids.push(seq_id);
            }
            if params.use_beam_search {
                for seq in seqs.iter_mut().skip(1) {
                    let _ = seq.set_status(SequenceStatus::FinishedAborted);
                }
            }

            let seq_group = SequenceGroup::new(
                request_id,
                seqs,
                params.clone(),
                Instant::now(),
                prompt.clone(),
            );
            self.scheduler.add_seq_group(seq_group);

            let beam_search = if params.use_beam_search {
                Some(BeamSearchState::new(
                    request_id,
                    num_seqs,
                    params.max_tokens,
                    params.length_penalty,
                    params.early_stopping,
                    &initial_seq_ids,
                ))
            } else {
                None
            };

            self.requests.insert(
                request_id,
                EngineRequest {
                    request_id,
                    prompt,
                    prompt_token_ids,
                    sampling_params: params,
                    seq_states,
                    beam_search,
                },
            );

            Ok(())
        }

        pub fn abort_request(&mut self, request_id: &RequestId) {
            info!(%request_id, "GpuLLMEngine: aborting request");
            let seqs = self.scheduler.cloned_seqs_for_request(request_id);
            self.scheduler.abort_seq_group(request_id);
            if let Some(req) = self.requests.get_mut(request_id) {
                if req.sampling_params.use_beam_search {
                    for seq in &seqs {
                        self.beam_block_manager.free(seq);
                    }
                }
                for state in &mut req.seq_states {
                    if state.finish_reason.is_none() {
                        state.finish_reason = Some(FinishReason::Abort);
                    }
                }
            }
        }

        /// Launch GPU work for one step. Returns pending state if work was
        /// launched, None if nothing to schedule. GPU computes asynchronously
        /// after this returns (~60us for graph replay path).
        pub fn step_launch(&mut self) -> Result<Option<StepPending>> {
            let (scheduled_groups, metadata, aborted_seqs) = match self.prepare_step() {
                Some(v) => v,
                None => return Ok(None),
            };

            if !aborted_seqs.is_empty() {
                for group in &scheduled_groups {
                    for (seq_idx, seq) in group.get_seqs().iter().enumerate() {
                        if aborted_seqs.contains(&seq.seq_id) {
                            if let Some(req) = self.requests.get_mut(&group.request_id) {
                                if let Some(state) = req.seq_states.get_mut(seq_idx) {
                                    state.finish_reason = Some(FinishReason::Abort);
                                }
                            }
                        }
                    }
                }
            }

            // Launch worker (returns quickly for async graph replay path)
            let actual_batch = self.worker.execute_launch(&metadata)
                .map_err(|e| LLMError::GpuError(format!("worker launch failed: {e}")))?;

            Ok(Some(StepPending { scheduled_groups, metadata, actual_batch }))
        }

        /// Collect GPU results and process outputs. Call after step_launch.
        /// If pending is None, returns empty (nothing was launched).
        pub fn step_collect(&mut self, pending: Option<StepPending>) -> Result<Vec<RequestOutput>> {
            let pending = match pending {
                Some(p) => p,
                None => return Ok(Vec::new()),
            };

            let worker_outputs = self.worker.execute_collect(pending.actual_batch, &pending.metadata)
                .map_err(|e| LLMError::GpuError(format!("worker collect failed: {e}")))?;

            // Prefix caching
            if let Some(ref mut pc) = self.prefix_cache {
                let block_size = self.config.cache.block_size;
                for meta in &pending.metadata {
                    if meta.is_prompt {
                        for (_seq_id, seq_data) in &meta.seq_data {
                            let num_full_blocks = seq_data.prompt_token_ids.len() / block_size;
                            let block_ids: Vec<rvllm_core::prelude::BlockId> = (0..num_full_blocks)
                                .map(|i| rvllm_core::prelude::BlockId(i as u32))
                                .collect();
                            let _ = prefix_cache::register_prefix_blocks(
                                pc, &seq_data.prompt_token_ids, &block_ids, block_size,
                            );
                        }
                    }
                }
            }

            let results = self.process_worker_outputs(&pending.scheduled_groups, &worker_outputs)?;
            self.recycle_dead_blocks();
            Ok(results)
        }

        /// Set the shared request queue for async overlap.
        pub fn set_request_queue(&mut self, q: RequestQueue) {
            self.request_queue = Some(q);
        }

        /// Set the shared abort queue for async overlap.
        pub fn set_abort_queue(&mut self, q: AbortQueue) {
            self.abort_queue = Some(q);
        }

        /// Drain buffered requests and aborts from shared queues into the engine.
        fn drain_request_queue(&mut self) {
            if let Some(ref q) = self.abort_queue {
                let aborts: Vec<RequestId> = {
                    let mut lock = q.lock().unwrap();
                    std::mem::take(&mut *lock)
                };
                for rid in aborts {
                    self.abort_request(&rid);
                }
            }
            if let Some(ref q) = self.request_queue {
                let requests: Vec<PendingRequest> = {
                    let mut lock = q.lock().unwrap();
                    std::mem::take(&mut *lock)
                };
                for req in requests {
                    let _ = self.add_request(req.request_id, req.prompt, req.params);
                }
            }
        }

        /// Step with overlap: runs `during_gpu` while GPU computes.
        /// Same correctness as step() -- scheduler state is consistent because
        /// the closure runs AFTER prepare_step but BEFORE process_worker_outputs.
        /// The closure should only drain NEW requests, not touch current sequences.
        pub fn step_with_overlap<F: FnOnce()>(&mut self, during_gpu: F) -> Result<Vec<RequestOutput>> {
            let (scheduled_groups, metadata, aborted_seqs) = match self.prepare_step() {
                Some(v) => v,
                None => { during_gpu(); return Ok(Vec::new()); }
            };

            if !aborted_seqs.is_empty() {
                for group in &scheduled_groups {
                    for (seq_idx, seq) in group.get_seqs().iter().enumerate() {
                        if aborted_seqs.contains(&seq.seq_id) {
                            if let Some(req) = self.requests.get_mut(&group.request_id) {
                                if let Some(state) = req.seq_states.get_mut(seq_idx) {
                                    state.finish_reason = Some(FinishReason::Abort);
                                }
                            }
                        }
                    }
                }
            }

            let requires_cache_ops = !self.pending_beam_copy_blocks.is_empty()
                || metadata
                    .iter()
                    .any(|meta| meta.sampling_params.use_beam_search);
            let worker_outputs = if requires_cache_ops {
                let blocks_to_copy = std::mem::take(&mut self.pending_beam_copy_blocks);
                let outputs = self
                    .worker
                    .execute_with_cache_ops(&metadata, &[], &[], &blocks_to_copy)
                    .map_err(|e| LLMError::GpuError(format!("worker execute failed: {e}")))?;
                during_gpu();
                outputs
            } else {
                self.worker
                    .execute_with_overlap(&metadata, during_gpu)
                    .map_err(|e| LLMError::GpuError(format!("worker execute failed: {e}")))?
            };

            // Prefix caching
            if let Some(ref mut pc) = self.prefix_cache {
                let block_size = self.config.cache.block_size;
                for meta in &metadata {
                    if meta.is_prompt {
                        for (_seq_id, seq_data) in &meta.seq_data {
                            let num_full_blocks = seq_data.prompt_token_ids.len() / block_size;
                            let block_ids: Vec<rvllm_core::prelude::BlockId> = (0..num_full_blocks)
                                .map(|i| rvllm_core::prelude::BlockId(i as u32))
                                .collect();
                            let _ = prefix_cache::register_prefix_blocks(
                                pc, &seq_data.prompt_token_ids, &block_ids, block_size,
                            );
                        }
                    }
                }
            }

            let results = self.process_worker_outputs(&scheduled_groups, &worker_outputs)?;
            self.recycle_dead_blocks();
            Ok(results)
        }

        pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
            // Drain any buffered requests from the shared queue before scheduling
            self.drain_request_queue();
            let t0 = std::time::Instant::now();
            let result = self.step_with_overlap(|| {});
            if std::env::var("RVLLM_PROFILE").is_ok() {
                let el = t0.elapsed();
                if el.as_millis() > 0 {
                    tracing::info!("PROFILE step total: {:.3}ms", el.as_secs_f64() * 1000.0);
                }
            }
            result
        }

        pub fn step_old(&mut self) -> Result<Vec<RequestOutput>> {
            let (scheduled_groups, metadata, aborted_seqs) = match self.prepare_step() {
                Some(v) => v,
                None => return Ok(Vec::new()),
            };

            if !aborted_seqs.is_empty() {
                for group in &scheduled_groups {
                    for (seq_idx, seq) in group.get_seqs().iter().enumerate() {
                        if aborted_seqs.contains(&seq.seq_id) {
                            if let Some(req) = self.requests.get_mut(&group.request_id) {
                                if let Some(state) = req.seq_states.get_mut(seq_idx) {
                                    state.finish_reason = Some(FinishReason::Abort);
                                }
                            }
                        }
                    }
                }
            }

            info!(
                num_groups = metadata.len(),
                "gpu_engine: calling worker.execute"
            );
            let blocks_to_copy = std::mem::take(&mut self.pending_beam_copy_blocks);
            let worker_outputs = self
                .worker
                .execute_with_cache_ops(&metadata, &[], &[], &blocks_to_copy)
                .map_err(|e| LLMError::GpuError(format!("worker execute failed: {e}")))?;
            trace!(
                num_outputs = worker_outputs.outputs.len(),
                "gpu_engine: worker.execute returned"
            );

            // Prefix caching: after prefill, register new prefix blocks.
            if let Some(ref mut pc) = self.prefix_cache {
                let block_size = self.config.cache.block_size;
                for meta in &metadata {
                    if meta.is_prompt {
                        for (_seq_id, seq_data) in &meta.seq_data {
                            let num_full_blocks = seq_data.prompt_token_ids.len() / block_size;
                            let block_ids: Vec<rvllm_core::prelude::BlockId> = (0..num_full_blocks)
                                .map(|i| rvllm_core::prelude::BlockId(i as u32))
                                .collect();
                            let newly_cached = prefix_cache::register_prefix_blocks(
                                pc,
                                &seq_data.prompt_token_ids,
                                &block_ids,
                                block_size,
                            );
                            if !newly_cached.is_empty() {
                                debug!(
                                    newly_cached = newly_cached.len(),
                                    "registered prefix blocks in cache"
                                );
                            }
                        }
                    }
                }
            }

            let results = self.process_worker_outputs(&scheduled_groups, &worker_outputs)?;
            self.recycle_dead_blocks();

            debug!(num_outputs = results.len(), "GpuLLMEngine: step complete");
            Ok(results)
        }

        pub fn run(&mut self) -> Result<Vec<RequestOutput>> {
            info!("GpuLLMEngine: run loop starting");
            let mut all_outputs = Vec::new();
            while self.has_unfinished() {
                let results = self.step()?;
                for output in results {
                    if output.finished {
                        all_outputs.push(output);
                    }
                }
            }
            info!(
                num_completed = all_outputs.len(),
                "GpuLLMEngine: run loop finished"
            );
            Ok(all_outputs)
        }

        /// Prepare one step: schedule, build metadata, handle prefix cache lookups.
        /// Returns None if there's nothing to schedule.
        fn prepare_step(
            &mut self,
        ) -> Option<(
            Vec<SequenceGroup>,
            Vec<SequenceGroupMetadata>,
            std::collections::HashSet<SequenceId>,
        )> {
            let free_count = self.gpu_block_pool.free_blocks();
            let watermark = (self.num_gpu_blocks as usize / 25).max(1);
            self.scheduler.set_block_budget(free_count, watermark);

            let (scheduled_groups, num_tokens) = self.scheduler.schedule();
            debug!(
                num_groups = scheduled_groups.len(),
                num_tokens, "pipelined scheduler output"
            );
            if scheduled_groups.is_empty() {
                return None;
            }

            let (metadata, aborted_seqs) = self.build_metadata(&scheduled_groups);
            if metadata.is_empty() {
                return None;
            }

            // Prefix caching: check for matching prefix blocks before prefill.
            if let Some(ref mut pc) = self.prefix_cache {
                for meta in &metadata {
                    if meta.is_prompt {
                        for (_seq_id, seq_data) in &meta.seq_data {
                            let hits = pc.count_hits(&seq_data.prompt_token_ids);
                            if hits > 0 {
                                let block_size = self.config.cache.block_size;
                                let cached_tokens = hits * block_size;
                                debug!(
                                    hits,
                                    cached_tokens,
                                    prompt_len = seq_data.prompt_token_ids.len(),
                                    "prefix cache hit: reusing cached KV blocks"
                                );
                            }
                        }
                    }
                }
            }

            trace!(
                num_groups = metadata.len(),
                "pipelined: submitting to GPU thread"
            );
            Some((scheduled_groups, metadata, aborted_seqs))
        }

        fn build_beam_group_update(
            tokenizer: &Tokenizer,
            req: &mut EngineRequest,
            output_map: &HashMap<u64, (TokenId, LogProb, Vec<(TokenId, LogProb)>)>,
            eos: Option<TokenId>,
        ) -> BeamGroupUpdate {
            let beam = req
                .beam_search
                .as_mut()
                .expect("beam search state missing for beam request");
            let mut expansions = HashMap::new();

            for active in &beam.active_beams {
                let Some((token_id, logprob, top_lps)) = output_map.get(&active.seq_id.0) else {
                    continue;
                };

                let candidates = if top_lps.is_empty() {
                    vec![(*token_id, *logprob)]
                } else {
                    top_k_from_logprobs(top_lps, beam.num_beams)
                };

                let decoded_candidates = candidates
                    .into_iter()
                    .map(|(candidate_token_id, candidate_logprob)| {
                        let decoded = tokenizer.decode(&[candidate_token_id]).unwrap_or_default();
                        (
                            candidate_token_id,
                            candidate_logprob,
                            decoded,
                            eos == Some(candidate_token_id),
                        )
                    })
                    .collect::<Vec<_>>();
                expansions.insert(active.seq_id, decoded_candidates);
            }

            let step_result = beam.step(&expansions);
            let mut recycled = step_result.seqs_to_free.into_iter();
            let mut fork_pairs = Vec::with_capacity(step_result.fork_ops.len());
            for op in step_result.fork_ops {
                let seq_id = recycled
                    .next()
                    .or_else(|| beam.take_available_seq_id())
                    .unwrap_or(op.parent_seq_id);
                beam.set_beam_seq_id(op.new_beam_idx, seq_id);
                fork_pairs.push((op.parent_seq_id, seq_id));
            }
            for seq_id in recycled {
                beam.recycle_seq_id(seq_id);
            }

            BeamGroupUpdate {
                output: beam.build_output(&req.prompt, &req.prompt_token_ids, 1),
                active_beams: beam.active_beams.clone(),
                fork_pairs,
            }
        }

        fn apply_beam_group_update(
            &mut self,
            request_id: RequestId,
            group: &SequenceGroup,
            update: &BeamGroupUpdate,
        ) -> Result<()> {
            let prompt_token_ids = self
                .requests
                .get(&request_id)
                .map(|req| req.prompt_token_ids.clone())
                .ok_or_else(|| LLMError::SchedulerError("beam request disappeared".into()))?;

            let child_seq_ids: std::collections::HashSet<SequenceId> = update
                .fork_pairs
                .iter()
                .map(|(_, child_seq_id)| *child_seq_id)
                .collect();
            for child_seq_id in &child_seq_ids {
                if let Some(old_child_seq) = group
                    .get_seqs()
                    .iter()
                    .find(|seq| seq.seq_id == *child_seq_id)
                {
                    self.beam_block_manager.free(old_child_seq);
                }
            }

            for (parent_seq_id, child_seq_id) in &update.fork_pairs {
                let parent_seq = group
                    .get_seqs()
                    .iter()
                    .find(|seq| seq.seq_id == *parent_seq_id)
                    .ok_or_else(|| {
                        LLMError::SchedulerError(format!(
                            "missing parent sequence {} for beam fork",
                            parent_seq_id.0
                        ))
                    })?;
                let mut child_seq = Sequence::new(*child_seq_id, prompt_token_ids.clone());
                child_seq.status = SequenceStatus::Running;
                self.beam_block_manager.fork(parent_seq, &mut child_seq)?;
            }

            let active_seq_ids: std::collections::HashSet<SequenceId> = update
                .active_beams
                .iter()
                .map(|active| active.seq_id)
                .collect();

            for active in &update.active_beams {
                self.scheduler.replace_seq_state(
                    active.seq_id,
                    prompt_token_ids.clone(),
                    active.token_ids.clone(),
                    active.cumulative_logprob,
                    SequenceStatus::Running,
                );

                let mut seq = Sequence::new(active.seq_id, prompt_token_ids.clone());
                seq.output_token_ids = active.token_ids.clone();
                seq.cumulative_logprob = active.cumulative_logprob;
                seq.status = SequenceStatus::Running;
                self.beam_block_manager.allocate(&seq)?;
                let _ = self.beam_block_manager.cow_if_needed(&seq)?;
            }

            self.pending_beam_copy_blocks
                .extend(self.beam_block_manager.get_copy_on_write_blocks());

            for seq in group.get_seqs() {
                if !active_seq_ids.contains(&seq.seq_id) {
                    self.scheduler
                        .finish_seq(seq.seq_id, SequenceStatus::FinishedAborted);
                    if !child_seq_ids.contains(&seq.seq_id) {
                        self.beam_block_manager.free(seq);
                    }
                }
            }

            Ok(())
        }

        /// Process worker outputs: update scheduler, build request outputs,
        /// clean up finished requests.
        fn process_worker_outputs(
            &mut self,
            scheduled_groups: &[SequenceGroup],
            worker_outputs: &GpuWorkerOutput,
        ) -> Result<Vec<RequestOutput>> {
            let mut output_map: HashMap<u64, (TokenId, LogProb, Vec<(TokenId, LogProb)>)> =
                HashMap::new();
            for wo in &worker_outputs.outputs {
                output_map.insert(wo.seq_id, (wo.token_id, wo.logprob, wo.top_logprobs.clone()));
            }

            let mut results = Vec::with_capacity(scheduled_groups.len());
            let eos = self.tokenizer.eos_token_id();

            for group in scheduled_groups {
                let request_id = group.request_id;
                let use_beam_search = self
                    .requests
                    .get(&request_id)
                    .map(|req| req.sampling_params.use_beam_search)
                    .unwrap_or(false);

                let output = if use_beam_search {
                    let update = {
                        let req = match self.requests.get_mut(&request_id) {
                            Some(r) => r,
                            None => continue,
                        };
                        Self::build_beam_group_update(&self.tokenizer, req, &output_map, eos)
                    };
                    self.apply_beam_group_update(request_id, group, &update)?;
                    update.output
                } else {
                    let req = match self.requests.get_mut(&request_id) {
                        Some(r) => r,
                        None => continue,
                    };
                    let logprobs_requested =
                        req.sampling_params.logprobs.map(|n| n > 0).unwrap_or(false);
                    let needs_text = !req.sampling_params.stop_strings.is_empty();

                    for (seq_idx, seq) in group.get_seqs().iter().enumerate() {
                        if seq.is_finished() {
                            continue;
                        }

                        if let Some((token_id, logprob, top_lps)) = output_map.get(&seq.seq_id.0) {
                            let decoded = if needs_text {
                                self.tokenizer.decode(&[*token_id]).unwrap_or_default()
                            } else {
                                String::new()
                            };
                            self.scheduler
                                .update_seq_token(seq.seq_id, *token_id, *logprob);

                            let top_logprobs = if logprobs_requested && !top_lps.is_empty() {
                                Some(top_lps.clone())
                            } else {
                                None
                            };

                            if let Some(state) = req.seq_states.get_mut(seq_idx) {
                                OutputProcessor::process_token(
                                    state,
                                    *token_id,
                                    *logprob,
                                    top_logprobs,
                                    &decoded,
                                    &req.sampling_params,
                                    eos,
                                );
                                if let Some(reason) = state.finish_reason {
                                    let status = match reason {
                                        FinishReason::Stop => SequenceStatus::FinishedStopped,
                                        FinishReason::Length => SequenceStatus::FinishedLength,
                                        FinishReason::Abort => SequenceStatus::FinishedAborted,
                                    };
                                    self.scheduler.finish_seq(seq.seq_id, status);
                                }
                            }
                        }
                    }

                    if !needs_text {
                        let all_finished = req.seq_states.iter().all(|s| s.is_finished());
                        if all_finished {
                            for state in &mut req.seq_states {
                                if !state.token_ids.is_empty() && state.text.is_empty() {
                                    state.text = self
                                        .tokenizer
                                        .decode(&state.token_ids)
                                        .unwrap_or_default();
                                }
                            }
                        }
                    }

                    OutputProcessor::build_request_output(
                        request_id,
                        &req.prompt,
                        &req.prompt_token_ids,
                        &req.seq_states,
                    )
                };
                results.push(output);
            }

            // Clean up finished requests.
            let finished_ids: Vec<RequestId> = self
                .requests
                .iter()
                .filter(|(_, req)| {
                    if let Some(beam) = &req.beam_search {
                        beam.is_finished()
                    } else {
                        req.seq_states.iter().all(|s| s.is_finished())
                    }
                })
                .map(|(&id, _)| id)
                .collect();
            for id in &finished_ids {
                let seqs = self.scheduler.cloned_seqs_for_request(id);
                if self
                    .requests
                    .get(id)
                    .map(|req| req.sampling_params.use_beam_search)
                    .unwrap_or(false)
                {
                    for seq in &seqs {
                        self.beam_block_manager.free(seq);
                    }
                }
                self.requests.remove(id);
                self.scheduler.abort_seq_group(id);
            }

            Ok(results)
        }

        /// Recycle KV cache blocks from sequences no longer tracked by scheduler.
        fn recycle_dead_blocks(&mut self) {
            let live_seq_ids: std::collections::HashSet<SequenceId> = self.scheduler.live_seq_ids();
            let dead_sids: Vec<SequenceId> = self
                .seq_block_tables
                .keys()
                .filter(|sid| !live_seq_ids.contains(sid))
                .copied()
                .collect();
            for sid in dead_sids {
                if let Some(blocks) = self.seq_block_tables.remove(&sid) {
                    debug!(
                        seq_id = sid.0,
                        num_blocks = blocks.len(),
                        "recycling blocks from finished sequence"
                    );
                    for b in blocks {
                        self.gpu_block_pool.free(b);
                    }
                }
            }
        }

        /// Check for unfinished sequences (scheduler-side only, doesn't
        /// account for in-flight GPU work).
        fn has_unfinished_excluding_worker(&self) -> bool {
            self.scheduler.has_unfinished_seqs() || !self.requests.is_empty()
        }

        pub fn has_unfinished(&self) -> bool {
            self.scheduler.has_unfinished_seqs() || !self.requests.is_empty()
        }

        pub fn num_unfinished(&self) -> usize {
            self.scheduler.get_num_unfinished_seq_groups()
        }

        pub fn config(&self) -> &EngineConfig {
            &self.config
        }

        /// Get a shared handle to the request ID counter (for async-side ID assignment).
        pub fn request_id_counter(&self) -> std::sync::Arc<AtomicU64> {
            self.next_request_id.clone()
        }

        /// Build per-group metadata with block allocation.  Returns the
        /// metadata list plus a set of sequence IDs that were aborted because
        /// blocks could not be allocated.
        fn build_metadata(
            &mut self,
            groups: &[SequenceGroup],
        ) -> (
            Vec<SequenceGroupMetadata>,
            std::collections::HashSet<SequenceId>,
        ) {
            let block_size = self.config.cache.block_size;
            let mut metadata = Vec::with_capacity(groups.len());
            let mut aborted_seqs: std::collections::HashSet<SequenceId> =
                std::collections::HashSet::new();

            for group in groups {
                let Some(req) = self.requests.get(&group.request_id) else {
                    continue;
                };
                let use_beam_search = req.sampling_params.use_beam_search;
                let beam_prompt_token_ids = req.prompt_token_ids.clone();
                let beam_active_beams = req
                    .beam_search
                    .as_ref()
                    .map(|beam| beam.active_beams.clone())
                    .unwrap_or_default();
                let is_prompt = if use_beam_search {
                    beam_active_beams.iter().all(|beam| beam.token_ids.is_empty())
                } else {
                    group.get_seqs()
                        .iter()
                        .filter(|seq| !seq.is_finished())
                        .any(|seq| seq.get_output_len() == 0)
                };
                let mut seq_data = HashMap::new();
                let mut block_tables = HashMap::new();

                if use_beam_search {
                    for active in &beam_active_beams {
                        let mut seq = Sequence::new(active.seq_id, beam_prompt_token_ids.clone());
                        seq.output_token_ids = active.token_ids.clone();
                        seq.cumulative_logprob = active.cumulative_logprob;
                        seq.status = SequenceStatus::Running;

                        let beam_alloc = self.beam_block_manager.allocate(&seq);
                        if beam_alloc.is_err() {
                            warn!(
                                seq_id = seq.seq_id.0,
                                "beam block allocation failed: aborting sequence"
                            );
                            self.scheduler
                                .finish_seq(seq.seq_id, SequenceStatus::FinishedAborted);
                            self.beam_block_manager.free(&seq);
                            aborted_seqs.insert(seq.seq_id);
                            continue;
                        }

                        let table = match self.beam_block_manager.get_block_table(seq.seq_id) {
                            Some(table) => table.iter().map(|block| block.block_id).collect(),
                            None => {
                                self.scheduler
                                    .finish_seq(seq.seq_id, SequenceStatus::FinishedAborted);
                                self.beam_block_manager.free(&seq);
                                aborted_seqs.insert(seq.seq_id);
                                continue;
                            }
                        };
                        block_tables.insert(seq.seq_id, table);
                        seq_data.insert(
                            seq.seq_id,
                            SequenceData {
                                prompt_token_ids: beam_prompt_token_ids.clone(),
                                output_token_ids: active.token_ids.clone(),
                                cumulative_logprob: active.cumulative_logprob,
                            },
                        );
                    }
                } else {
                    for seq in group.get_seqs() {
                        if seq.is_finished() {
                            continue;
                        }

                        let total_tokens = seq.prompt_token_ids.len() + seq.output_token_ids.len();
                        // +1 headroom: pre-allocate for the token about to be generated this step
                        let needed_blocks = (total_tokens + 1 + block_size - 1) / block_size;

                        // Reuse existing blocks, append new ones if needed
                        let existing = self.seq_block_tables.entry(seq.seq_id).or_default();
                        let mut alloc_failed = false;
                        while existing.len() < needed_blocks {
                            let block_id = if let Some(block_id) = self.gpu_block_pool.allocate() {
                                block_id.0
                            } else {
                                warn!(
                                    seq_id = seq.seq_id.0,
                                    needed = needed_blocks,
                                    have = existing.len(),
                                    num_gpu_blocks = self.num_gpu_blocks,
                                    free_blocks = self.gpu_block_pool.free_blocks(),
                                    "block allocation failed: no free GPU KV-cache blocks, aborting sequence"
                                );
                                alloc_failed = true;
                                break;
                            };
                            existing.push(BlockId(block_id));
                        }

                        if alloc_failed {
                            // Recycle whatever blocks this sequence had -- it cannot
                            // proceed without its full allocation.
                            if let Some(blocks) = self.seq_block_tables.remove(&seq.seq_id) {
                                for b in blocks {
                                    self.gpu_block_pool.free(b);
                                }
                            }
                            // Mark finished so the scheduler drops it next round.
                            self.scheduler
                                .finish_seq(seq.seq_id, SequenceStatus::FinishedAborted);
                            aborted_seqs.insert(seq.seq_id);
                            continue;
                        }

                        block_tables.insert(seq.seq_id, existing.clone());
                        seq_data.insert(
                            seq.seq_id,
                            SequenceData {
                                prompt_token_ids: seq.prompt_token_ids.clone(),
                                output_token_ids: seq.output_token_ids.clone(),
                                cumulative_logprob: seq.cumulative_logprob,
                            },
                        );
                    }
                }

                // Only emit metadata if the group still has live sequences.
                if !seq_data.is_empty() {
                    let mut sampling_params = group.sampling_params.clone();
                    if sampling_params.use_beam_search {
                        sampling_params.logprobs = Some(
                            sampling_params
                                .logprobs
                                .unwrap_or(0)
                                .max(sampling_params.best_of),
                        );
                    }
                    metadata.push(SequenceGroupMetadata {
                        request_id: group.request_id,
                        is_prompt,
                        seq_data,
                        sampling_params,
                        block_tables,
                    });
                }
            }
            (metadata, aborted_seqs)
        }
    }

    unsafe impl Send for GpuLLMEngine {}
}

#[cfg(feature = "cuda")]
pub use inner::{AbortQueue, GpuLLMEngine, PendingRequest, RequestQueue};
