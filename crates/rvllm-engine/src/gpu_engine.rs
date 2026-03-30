//! GPU-accelerated inference engine composing scheduler, GPU worker, and tokenizer.
//!
//! Reads model config.json from HuggingFace cache to set correct architecture
//! parameters, loads weights to GPU, and drives the scheduling/output loop.
//!
//! Gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    use tracing::{debug, info, trace, warn};

    use rvllm_block_manager::prefix_cache::{self, PrefixCache};
    use rvllm_config::EngineConfig;
    use rvllm_core::prelude::{
        BlockId, FinishReason, LLMError, LogProb, RequestId, RequestOutput, Result, SamplingParams,
        SequenceId, TokenId,
    };
    use rvllm_sequence::{Sequence, SequenceData, SequenceGroup, SequenceGroupMetadata, SequenceStatus};
    use rvllm_tokenizer::Tokenizer;
    use rvllm_worker::gpu_worker::{GpuWorker, GpuWorkerOutput};

    use rvllm_speculative::{SelfDraftModel, SpeculativeConfig, SpeculativeEngine, TargetModel};

    use crate::hf_snapshot;
    use crate::output::{OutputProcessor, SequenceOutputState};

    // ------------------------------------------------------------------
    // HuggingFace model config reading
    // ------------------------------------------------------------------

    /// Subset of fields from a HuggingFace config.json that we need.
    #[derive(Debug, Clone)]
    struct HfModelConfig {
        model_type: String,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        num_hidden_layers: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
        rms_norm_eps: f32,
        tie_word_embeddings: bool,
        architecture: String,
        rope_theta: f32,
        partial_rotary_factor: f32,
        attn_logit_softcapping: f32,
        attention_bias: bool,
        sliding_window: Option<usize>,
        layer_types: Vec<String>,
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

        // Models like Qwen3.5 nest their parameters under "text_config".
        // Try text_config first, fall back to top-level.
        let text_json = json.get("text_config").unwrap_or(&json);

        let get_usize = |key: &str, default: usize| -> usize {
            text_json
                .get(key)
                .or_else(|| json.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            text_json
                .get(key)
                .or_else(|| json.get(key))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default)
        };
        let get_string = |key: &str, default: &str| -> String {
            json.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or(default)
                .to_string()
        };
        let get_string_vec = |key: &str| -> Vec<String> {
            json.get(key)
                .and_then(|v| v.as_array())
                .map(|vals| {
                    vals.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };

        let hidden_size = get_usize("hidden_size", 4096);
        let num_attention_heads = get_usize("num_attention_heads", 32);
        let head_dim = get_usize(
            "head_dim",
            hidden_size.checked_div(num_attention_heads).unwrap_or(0),
        );

        let architecture = json
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("LlamaForCausalLM")
            .to_string();

        let tie_word_embeddings = text_json
            .get("tie_word_embeddings")
            .or_else(|| json.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let rope_theta = text_json
            .get("rope_parameters")
            .and_then(|v| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or_else(|| get_f32("rope_theta", 10000.0));

        let partial_rotary_factor = text_json
            .get("rope_parameters")
            .and_then(|v| v.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or_else(|| get_f32("partial_rotary_factor", 1.0));

        Ok(HfModelConfig {
            model_type: get_string("model_type", "llama"),
            hidden_size,
            intermediate_size: get_usize("intermediate_size", 11008),
            num_attention_heads,
            head_dim,
            num_key_value_heads: get_usize("num_key_value_heads", num_attention_heads),
            num_hidden_layers: get_usize("num_hidden_layers", 32),
            vocab_size: get_usize("vocab_size", 32000),
            max_position_embeddings: get_usize("max_position_embeddings", 2048),
            rms_norm_eps: get_f32("rms_norm_eps", 1e-5),
            tie_word_embeddings,
            architecture,
            rope_theta,
            partial_rotary_factor,
            attn_logit_softcapping: get_f32("attn_logit_softcapping", 0.0),
            attention_bias: json
                .get("attention_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            sliding_window: json
                .get("sliding_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            layer_types: get_string_vec("layer_types"),
            num_local_experts: get_usize("num_local_experts", 0),
            num_experts_per_tok: json
                .get("num_experts_per_tok")
                .or_else(|| json.get("experts_per_token"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(0),
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
        /// Persistent block allocation with recycling.
        next_block_id: u32,
        /// Free list of recycled block IDs.
        free_blocks: Vec<u32>,
        /// Per-sequence block tables that persist across step() calls.
        seq_block_tables: HashMap<SequenceId, Vec<BlockId>>,
        /// Shared queue for new requests arriving during GPU compute.
        request_queue: Option<RequestQueue>,
        /// Shared queue for abort requests arriving during GPU compute.
        abort_queue: Option<AbortQueue>,
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
                model_type = %hf_config.model_type,
                hidden = hf_config.hidden_size,
                layers = hf_config.num_hidden_layers,
                heads = hf_config.num_attention_heads,
                kv_heads = hf_config.num_key_value_heads,
                vocab = hf_config.vocab_size,
                intermediate = hf_config.intermediate_size,
                "model config loaded"
            );

            let quant_method = rvllm_quant::detect_quant_method(&model_dir)?;
            if quant_method.is_quantized() && hf_config.architecture != "GptOssForCausalLM" {
                return Err(LLMError::ModelError(format!(
                    "quantized model checkpoints are not wired into the GPU engine yet (detected {})",
                    quant_method
                )));
            }

            // 3. Tokenizer
            let tokenizer_path = config.model.tokenizer_path.as_deref().unwrap_or(model_name);
            let tokenizer = Tokenizer::from_pretrained(tokenizer_path)?;

            // 4. Build WorkerConfig from real model config
            let worker_config = rvllm_worker::WorkerConfig {
                device_id: 0,
                num_layers: hf_config.num_hidden_layers,
                num_kv_heads: hf_config.num_key_value_heads,
                head_dim: hf_config.head_dim,
                hidden_size: hf_config.hidden_size,
                num_attention_heads: hf_config.num_attention_heads,
                intermediate_size: hf_config.intermediate_size,
                vocab_size: hf_config.vocab_size,
                max_model_len: config
                    .model
                    .max_model_len
                    .min(hf_config.max_position_embeddings),
                rms_norm_eps: hf_config.rms_norm_eps,
                block_size: config.cache.block_size,
                gpu_memory_utilization: config.cache.gpu_memory_utilization,
                rank: 0,
                tensor_parallel_size: config.parallel.tensor_parallel_size,
                pipeline_parallel_size: config.parallel.pipeline_parallel_size,
                architecture: hf_config.architecture.clone(),
                dtype: config.model.dtype,
                rope_theta: hf_config.rope_theta,
                partial_rotary_factor: hf_config.partial_rotary_factor,
                attn_logit_softcapping: hf_config.attn_logit_softcapping,
                attention_bias: hf_config.attention_bias,
                sliding_window: hf_config.sliding_window,
                layer_types: hf_config.layer_types.clone(),
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

            let spec_enabled = Self::speculative_enabled();
            let fp8_kv = worker.use_fp8_kv();
            info!(
                num_gpu_blocks,
                num_cpu_blocks,
                block_size = config.cache.block_size,
                max_num_seqs = config.scheduler.max_num_seqs,
                fp8_kv,
                speculative = spec_enabled,
                "GpuLLMEngine: ready for inference"
            );
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
                next_block_id: 0,
                free_blocks: Vec::new(),
                seq_block_tables: HashMap::new(),
                request_queue: None,
                abort_queue: None,
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

            for _ in 0..num_seqs {
                let seq_id = SequenceId(self.next_seq_id);
                self.next_seq_id += 1;
                seqs.push(Sequence::new(seq_id, prompt_token_ids.clone()));
                seq_states.push(SequenceOutputState::new());
            }

            let seq_group = SequenceGroup::new(
                request_id,
                seqs,
                params.clone(),
                Instant::now(),
                prompt.clone(),
            );
            self.scheduler.add_seq_group(seq_group);

            self.requests.insert(
                request_id,
                EngineRequest {
                    request_id,
                    prompt,
                    prompt_token_ids,
                    sampling_params: params,
                    seq_states,
                },
            );

            Ok(())
        }

        pub fn abort_request(&mut self, request_id: &RequestId) {
            info!(%request_id, "GpuLLMEngine: aborting request");
            self.scheduler.abort_seq_group(request_id);
            if let Some(req) = self.requests.get_mut(request_id) {
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

            let results = self.process_worker_outputs(&pending.scheduled_groups, &worker_outputs);
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

            let worker_outputs = self.worker
                .execute_with_overlap(&metadata, during_gpu)
                .map_err(|e| LLMError::GpuError(format!("worker execute failed: {e}")))?;

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

            let results = self.process_worker_outputs(&scheduled_groups, &worker_outputs);
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

            let worker_outputs = self
                .worker
                .execute(&metadata)
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

            let results = self.process_worker_outputs(&scheduled_groups, &worker_outputs);
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
            info!(num_completed = all_outputs.len(), "GpuLLMEngine: run loop finished");
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
            let free_count = self.free_blocks.len()
                + (self.num_gpu_blocks.saturating_sub(self.next_block_id)) as usize;
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

        /// Process worker outputs: update scheduler, build request outputs,
        /// clean up finished requests.
        fn process_worker_outputs(
            &mut self,
            scheduled_groups: &[SequenceGroup],
            worker_outputs: &GpuWorkerOutput,
        ) -> Vec<RequestOutput> {
            let mut output_map: HashMap<u64, (TokenId, LogProb, &[(TokenId, LogProb)])> =
                HashMap::with_capacity(worker_outputs.outputs.len());
            for wo in &worker_outputs.outputs {
                output_map.insert(
                    wo.seq_id,
                    (wo.token_id, wo.logprob, &wo.top_logprobs),
                );
            }

            let mut results = Vec::with_capacity(scheduled_groups.len());
            let eos = self.tokenizer.eos_token_id();

            for group in scheduled_groups {
                let request_id = group.request_id;
                let req = match self.requests.get_mut(&request_id) {
                    Some(r) => r,
                    None => continue,
                };

                let logprobs_requested =
                    req.sampling_params.logprobs.map(|n| n > 0).unwrap_or(false);

                // Only decode token text when stop_strings require it;
                // EOS and max_tokens checks only need token IDs.
                let needs_text = !req.sampling_params.stop_strings.is_empty();

                for (seq_idx, seq) in group.get_seqs().iter().enumerate() {
                    if seq.is_finished() {
                        continue;
                    }

                    if let Some((token_id, logprob, top_lps)) = output_map.get(&seq.seq_id.0) {
                        // Defer tokenizer.decode() when no stop_strings are configured.
                        // For greedy decode the token ID is sufficient for the scheduler;
                        // full text is reconstructed when the response is sent to the client.
                        let decoded = if needs_text {
                            self.tokenizer.decode(&[*token_id]).unwrap_or_default()
                        } else {
                            String::new()
                        };
                        self.scheduler
                            .update_seq_token(seq.seq_id, *token_id, *logprob);

                        let top_logprobs = if logprobs_requested && !top_lps.is_empty() {
                            Some(top_lps.to_vec())
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

                // Lazy text reconstruction: when decode was deferred (no stop_strings),
                // batch-decode accumulated token_ids into text for finished sequences
                // or for any output that will be returned to the client.
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

                let output = OutputProcessor::build_request_output(
                    request_id,
                    &req.prompt,
                    &req.prompt_token_ids,
                    &req.seq_states,
                );
                results.push(output);
            }

            // Clean up finished requests.
            let finished_ids: Vec<RequestId> = self
                .requests
                .iter()
                .filter(|(_, req)| req.seq_states.iter().all(|s| s.is_finished()))
                .map(|(&id, _)| id)
                .collect();
            for id in &finished_ids {
                self.requests.remove(id);
                self.scheduler.abort_seq_group(id);
            }

            results
        }

        /// Recycle KV cache blocks from sequences no longer tracked by scheduler.
        fn recycle_dead_blocks(&mut self) {
            let live_seq_ids: std::collections::HashSet<SequenceId> =
                self.scheduler.live_seq_ids();
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
                        self.free_blocks.push(b.0);
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
        ) -> (Vec<SequenceGroupMetadata>, std::collections::HashSet<SequenceId>) {
            let block_size = self.config.cache.block_size;
            let mut metadata = Vec::with_capacity(groups.len());
            let mut aborted_seqs: std::collections::HashSet<SequenceId> =
                std::collections::HashSet::new();

            for group in groups {
                let is_prompt = group.get_seqs().iter().any(|s| s.get_output_len() == 0);
                let mut seq_data = HashMap::new();
                let mut block_tables = HashMap::new();

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
                        let block_id = if let Some(recycled) = self.free_blocks.pop() {
                            recycled
                        } else if self.next_block_id < self.num_gpu_blocks {
                            let id = self.next_block_id;
                            self.next_block_id += 1;
                            id
                        } else {
                            warn!(
                                seq_id = seq.seq_id.0,
                                needed = needed_blocks,
                                have = existing.len(),
                                num_gpu_blocks = self.num_gpu_blocks,
                                free_blocks = self.free_blocks.len(),
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
                                self.free_blocks.push(b.0);
                            }
                        }
                        // Mark finished so the scheduler drops it next round.
                        self.scheduler.finish_seq(seq.seq_id, SequenceStatus::FinishedAborted);
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

                // Only emit metadata if the group still has live sequences.
                if !seq_data.is_empty() {
                    metadata.push(SequenceGroupMetadata {
                        request_id: group.request_id,
                        is_prompt,
                        seq_data,
                        sampling_params: group.sampling_params.clone(),
                        block_tables,
                    });
                }
            }
            (metadata, aborted_seqs)
        }
    }

    unsafe impl Send for GpuLLMEngine {}

    // ------------------------------------------------------------------
    // GpuTargetModel: adapter for speculative decoding
    // ------------------------------------------------------------------

    /// Wraps the GPU worker's forward pass as a [`TargetModel`] for
    /// speculative decoding verification.
    ///
    /// Constructs a synthetic single-sequence metadata batch from the token
    /// list, runs a full forward pass through the GPU model, and extracts
    /// probability distributions for the requested verification positions.
    pub struct GpuTargetModel {
        worker: *mut GpuWorker,
        vocab_size: usize,
        block_size: usize,
    }

    // SAFETY: GpuTargetModel is only used on the same thread as the engine
    // that owns the GpuWorker. The raw pointer avoids borrow checker issues
    // with the engine holding both the worker and the speculative engine.
    unsafe impl Send for GpuTargetModel {}

    impl GpuTargetModel {
        /// Create from a mutable reference to GpuWorker.
        /// Caller must ensure the GpuWorker outlives this adapter.
        pub unsafe fn new(worker: &mut GpuWorker, vocab_size: usize, block_size: usize) -> Self {
            Self {
                worker: worker as *mut GpuWorker,
                vocab_size,
                block_size,
            }
        }

        fn worker(&mut self) -> &mut GpuWorker {
            // SAFETY: invariant maintained by caller of new()
            unsafe { &mut *self.worker }
        }
    }

    impl TargetModel for GpuTargetModel {
        fn forward_verify(
            &mut self,
            tokens: &[TokenId],
            num_verify_positions: usize,
        ) -> Result<Vec<Vec<f32>>> {
            if tokens.is_empty() || num_verify_positions == 0 {
                return Ok(Vec::new());
            }

            // Build a single-sequence metadata for the full token list.
            // Treat as a prefill (all tokens processed at once).
            let seq_id = SequenceId(u64::MAX - 1); // sentinel
            let request_id = RequestId(u64::MAX - 1);

            let mut seq_data_map = HashMap::new();
            seq_data_map.insert(
                seq_id,
                SequenceData {
                    prompt_token_ids: tokens.to_vec(),
                    output_token_ids: Vec::new(),
                    cumulative_logprob: 0.0,
                },
            );

            // Allocate enough blocks for the full token sequence.
            let needed_blocks = (tokens.len() + self.block_size - 1) / self.block_size;
            let block_table: Vec<BlockId> = (0..needed_blocks as u32).map(BlockId).collect();
            let mut block_tables = HashMap::new();
            block_tables.insert(seq_id, block_table);

            let metadata = vec![SequenceGroupMetadata {
                request_id,
                is_prompt: true,
                seq_data: seq_data_map,
                sampling_params: SamplingParams::default(),
                block_tables,
            }];

            // Run forward pass to get logits for all positions.
            let logits = self.worker().forward_logits(&metadata)?;

            // Extract probability distributions for the last `num_verify_positions`.
            // logits layout: [num_tokens, vocab_size]
            let num_tokens = tokens.len();
            let vs = self.vocab_size;

            if logits.len() < num_tokens * vs {
                return Err(LLMError::ModelError(format!(
                    "logits too short: got {} elements, expected {}",
                    logits.len(),
                    num_tokens * vs,
                )));
            }

            let mut result = Vec::with_capacity(num_verify_positions);
            let start_pos = num_tokens.saturating_sub(num_verify_positions);

            for pos in start_pos..num_tokens {
                let offset = pos * vs;
                let token_logits = &logits[offset..offset + vs];

                // Convert logits to probabilities via softmax.
                let max_logit = token_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = token_logits.iter().map(|&l| (l - max_logit).exp()).collect();
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }
                result.push(probs);
            }

            Ok(result)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    // ------------------------------------------------------------------
    // Speculative decoding integration on GpuLLMEngine
    // ------------------------------------------------------------------

    impl GpuLLMEngine {
        /// Check if speculative decoding is enabled via env var or config.
        fn speculative_enabled() -> bool {
            std::env::var("RVLLM_SPECULATIVE").map_or(false, |v| v == "1")
        }

        /// Get the draft model path from env or default.
        fn speculative_draft_model() -> Option<String> {
            std::env::var("RVLLM_SPECULATIVE_DRAFT").ok()
        }

        /// Get the number of speculative tokens (K) from env or default to 3.
        fn speculative_k() -> usize {
            std::env::var("RVLLM_SPECULATIVE_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3)
        }

        /// Get the number of draft layers from env or default to num_layers / 4.
        fn speculative_draft_layers(total_layers: usize) -> usize {
            std::env::var("RVLLM_SPECULATIVE_DRAFT_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(total_layers / 4)
        }

        /// Whether to use self-draft (no separate model).
        fn speculative_self_draft() -> bool {
            std::env::var("RVLLM_SPECULATIVE_SELF_DRAFT").map_or(false, |v| v == "1")
                || Self::speculative_draft_model().is_none()
        }

        /// Build a SelfDraftModel that calls partial forward on the GPU worker.
        ///
        /// SAFETY: The returned model holds a raw pointer to the worker.
        /// Caller must ensure the worker outlives the draft model and that
        /// no other mutable access occurs concurrently.
        unsafe fn build_self_draft(
            worker: &mut GpuWorker,
            vocab_size: usize,
            block_size: usize,
            num_draft_layers: usize,
        ) -> Box<dyn rvllm_speculative::DraftModel> {
            struct SendPtr(usize);
            unsafe impl Send for SendPtr {}
            impl SendPtr {
                unsafe fn get(&self) -> &mut GpuWorker { &mut *(self.0 as *mut GpuWorker) }
            }
            let worker_ptr = SendPtr(worker as *mut GpuWorker as usize);

            let partial_fn: rvllm_speculative::self_draft::PartialForwardFn =
                Box::new(move |tokens: &[TokenId], max_layers: usize| {
                    let w = unsafe { worker_ptr.get() };

                    let seq_id = SequenceId(u64::MAX - 2);
                    let request_id = RequestId(u64::MAX - 2);

                    let mut seq_data_map = HashMap::new();
                    seq_data_map.insert(
                        seq_id,
                        SequenceData {
                            prompt_token_ids: tokens.to_vec(),
                            output_token_ids: Vec::new(),
                            cumulative_logprob: 0.0,
                        },
                    );

                    let needed_blocks = (tokens.len() + block_size - 1) / block_size;
                    let block_table: Vec<BlockId> =
                        (0..needed_blocks as u32).map(BlockId).collect();
                    let mut block_tables = HashMap::new();
                    block_tables.insert(seq_id, block_table);

                    let metadata = vec![SequenceGroupMetadata {
                        request_id,
                        is_prompt: true,
                        seq_data: seq_data_map,
                        sampling_params: SamplingParams::default(),
                        block_tables,
                    }];

                    w.forward_logits_partial(&metadata, max_layers)
                });

            Box::new(SelfDraftModel::new(num_draft_layers, vocab_size, partial_fn))
        }

        /// Run a single request through speculative decoding.
        /// Returns the generated token IDs (excluding prompt).
        ///
        /// When `RVLLM_SPECULATIVE_SELF_DRAFT=1` or no draft model path is set,
        /// uses self-draft (first N layers of the target model). Otherwise uses
        /// the placeholder DraftModelRunner with the configured draft model path.
        pub fn generate_speculative(
            &mut self,
            prompt_tokens: &[TokenId],
            max_tokens: usize,
            eos_token_id: TokenId,
        ) -> Result<Vec<TokenId>> {
            if !Self::speculative_enabled() {
                return Err(LLMError::ConfigError(
                    "speculative decoding not enabled (set RVLLM_SPECULATIVE=1)".into(),
                ));
            }

            let k = Self::speculative_k();
            let vocab_size = self.worker.vocab_size();
            let block_size = self.config.cache.block_size;
            let use_self_draft = Self::speculative_self_draft();
            let total_layers = self.worker.num_layers();
            let draft_layers = Self::speculative_draft_layers(total_layers);

            info!(
                k,
                max_tokens,
                prompt_len = prompt_tokens.len(),
                self_draft = use_self_draft,
                draft_layers,
                "starting speculative decoding"
            );

            // Build draft model.
            let draft: Box<dyn rvllm_speculative::DraftModel> = if use_self_draft {
                // SAFETY: see build_self_draft docs.
                unsafe { Self::build_self_draft(&mut self.worker, vocab_size, block_size, draft_layers) }
            } else {
                let draft_path = Self::speculative_draft_model().unwrap_or_default();
                let cfg = SpeculativeConfig::new(draft_path, k);
                Box::new(rvllm_speculative::DraftModelRunner::new(cfg)?)
            };

            let spec_config = SpeculativeConfig::new("self-draft".into(), k);

            // SAFETY: GpuTargetModel borrows worker mutably via raw pointer.
            // We don't access self.worker while the SpeculativeEngine is alive.
            let target = unsafe { GpuTargetModel::new(&mut self.worker, vocab_size, block_size) };

            let mut spec_engine = SpeculativeEngine::with_draft(spec_config, target, draft)?;

            let output_tokens =
                spec_engine.generate(prompt_tokens, max_tokens, |tok| tok == eos_token_id)?;

            let metrics = spec_engine.metrics();
            info!(
                accepted = metrics.total_accepted_tokens,
                drafted = metrics.total_draft_tokens,
                steps = metrics.total_steps,
                rate = %format!("{:.1}%", metrics.acceptance_rate() * 100.0),
                speedup = %format!("{:.2}x", metrics.speedup_ratio()),
                "speculative decoding complete"
            );

            Ok(output_tokens)
        }

        /// Benchmark speculative decoding with self-draft vs standard decode.
        ///
        /// Runs both paths on the same prompt and reports comparative metrics.
        /// Set RVLLM_SPECULATIVE=1 before calling.
        pub fn benchmark_speculative(
            &mut self,
            prompt_tokens: &[TokenId],
            max_tokens: usize,
            eos_token_id: TokenId,
        ) -> Result<SpeculativeBenchResult> {
            let k = Self::speculative_k();
            let vocab_size = self.worker.vocab_size();
            let block_size = self.config.cache.block_size;
            let total_layers = self.worker.num_layers();
            let draft_layers = Self::speculative_draft_layers(total_layers);

            info!(
                k,
                draft_layers,
                total_layers,
                max_tokens,
                "benchmarking speculative decoding"
            );

            // --- Standard decode baseline ---
            let std_start = Instant::now();
            let std_tokens = self.generate_standard(prompt_tokens, max_tokens, eos_token_id)?;
            let std_elapsed = std_start.elapsed();
            let std_tok_per_sec = std_tokens.len() as f64 / std_elapsed.as_secs_f64();

            // --- Speculative decode ---
            let spec_config = SpeculativeConfig::new("self-draft".into(), k);
            let draft = unsafe {
                Self::build_self_draft(&mut self.worker, vocab_size, block_size, draft_layers)
            };
            let target = unsafe { GpuTargetModel::new(&mut self.worker, vocab_size, block_size) };
            let mut spec_engine = SpeculativeEngine::with_draft(spec_config, target, draft)?;

            let spec_start = Instant::now();
            let spec_tokens =
                spec_engine.generate(prompt_tokens, max_tokens, |tok| tok == eos_token_id)?;
            let spec_elapsed = spec_start.elapsed();
            let spec_tok_per_sec = spec_tokens.len() as f64 / spec_elapsed.as_secs_f64();

            let metrics = spec_engine.metrics().clone();

            let result = SpeculativeBenchResult {
                standard_tokens: std_tokens.len(),
                standard_elapsed_ms: std_elapsed.as_millis() as u64,
                standard_tok_per_sec: std_tok_per_sec,
                speculative_tokens: spec_tokens.len(),
                speculative_elapsed_ms: spec_elapsed.as_millis() as u64,
                speculative_tok_per_sec: spec_tok_per_sec,
                acceptance_rate: metrics.acceptance_rate(),
                speedup_ratio: metrics.speedup_ratio(),
                draft_layers,
                k,
                wall_clock_speedup: spec_tok_per_sec / std_tok_per_sec.max(1e-9),
            };

            info!(
                std_tps = %format!("{:.1}", result.standard_tok_per_sec),
                spec_tps = %format!("{:.1}", result.speculative_tok_per_sec),
                accept = %format!("{:.1}%", result.acceptance_rate * 100.0),
                speedup = %format!("{:.2}x", result.wall_clock_speedup),
                "speculative benchmark complete"
            );

            Ok(result)
        }

        /// Simple autoregressive decode loop (no speculation) for benchmarking baseline.
        fn generate_standard(
            &mut self,
            prompt_tokens: &[TokenId],
            max_tokens: usize,
            eos_token_id: TokenId,
        ) -> Result<Vec<TokenId>> {
            let vocab_size = self.worker.vocab_size();
            let block_size = self.config.cache.block_size;
            let mut context = prompt_tokens.to_vec();
            let mut output = Vec::with_capacity(max_tokens);

            for _ in 0..max_tokens {
                let seq_id = SequenceId(u64::MAX - 3);
                let request_id = RequestId(u64::MAX - 3);

                let mut seq_data_map = HashMap::new();
                seq_data_map.insert(
                    seq_id,
                    SequenceData {
                        prompt_token_ids: context.clone(),
                        output_token_ids: Vec::new(),
                        cumulative_logprob: 0.0,
                    },
                );

                let needed_blocks = (context.len() + block_size - 1) / block_size;
                let block_table: Vec<BlockId> =
                    (0..needed_blocks as u32).map(BlockId).collect();
                let mut block_tables = HashMap::new();
                block_tables.insert(seq_id, block_table);

                let metadata = vec![SequenceGroupMetadata {
                    request_id,
                    is_prompt: true,
                    seq_data: seq_data_map,
                    sampling_params: SamplingParams::default(),
                    block_tables,
                }];

                let logits = self.worker.forward_logits(&metadata)?;
                let num_tokens = context.len();
                let offset = (num_tokens - 1) * vocab_size;
                if logits.len() < offset + vocab_size {
                    break;
                }
                let token_logits = &logits[offset..offset + vocab_size];

                // Greedy argmax
                let mut best_tok = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in token_logits.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_tok = i as u32;
                    }
                }

                output.push(best_tok);
                context.push(best_tok);
                if best_tok == eos_token_id {
                    break;
                }
            }

            Ok(output)
        }
    }

    /// Results from benchmarking speculative vs standard decode.
    #[derive(Debug, Clone)]
    pub struct SpeculativeBenchResult {
        pub standard_tokens: usize,
        pub standard_elapsed_ms: u64,
        pub standard_tok_per_sec: f64,
        pub speculative_tokens: usize,
        pub speculative_elapsed_ms: u64,
        pub speculative_tok_per_sec: f64,
        pub acceptance_rate: f64,
        pub speedup_ratio: f64,
        pub draft_layers: usize,
        pub k: usize,
        pub wall_clock_speedup: f64,
    }
}

#[cfg(feature = "cuda")]
pub use inner::{AbortQueue, GpuLLMEngine, GpuTargetModel, PendingRequest, RequestQueue, SpeculativeBenchResult};
