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

    use tracing::{debug, info, warn};

    use rvllm_config::EngineConfig;
    use rvllm_core::prelude::{
        FinishReason, LLMError, LogProb, RequestId, RequestOutput, Result,
        SamplingParams, SequenceId, TokenId,
    };
    use rvllm_sequence::{Sequence, SequenceData, SequenceGroup, SequenceGroupMetadata};
    use rvllm_tokenizer::Tokenizer;
    use rvllm_block_manager::prefix_cache::{self, PrefixCache};
    use rvllm_worker::gpu_worker::GpuWorker;

    use crate::output::{OutputProcessor, SequenceOutputState};

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
        let path = Path::new(model_name);
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }

        // Look in HF cache
        let cache_dir = dirs_hf_cache();
        let repo_dir_name = format!("models--{}", model_name.replace('/', "--"));
        let repo_path = cache_dir.join(&repo_dir_name).join("snapshots");

        if repo_path.is_dir() {
            // Find the first snapshot
            if let Ok(entries) = std::fs::read_dir(&repo_path) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let snap_path = entry.path();
                    if snap_path.is_dir() {
                        // Check for config.json
                        if snap_path.join("config.json").exists() {
                            info!(path = %snap_path.display(), "found model in HF cache");
                            return Ok(snap_path);
                        }
                    }
                }
            }
        }

        // Try downloading via hf-hub
        info!(model = model_name, "downloading model from HuggingFace");
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            LLMError::ModelError(format!("failed to init hf-hub: {e}"))
        })?;
        let repo = api.model(model_name.to_string());

        // Download config.json and model files
        let _config_path = repo.get("config.json").map_err(|e| {
            LLMError::ModelError(format!("failed to download config.json: {e}"))
        })?;
        let _model_path = repo.get("model.safetensors").map_err(|e| {
            LLMError::ModelError(format!("failed to download model.safetensors: {e}"))
        })?;

        // Re-check cache after download
        if repo_path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&repo_path) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let snap_path = entry.path();
                    if snap_path.is_dir() && snap_path.join("config.json").exists() {
                        return Ok(snap_path);
                    }
                }
            }
        }

        Err(LLMError::ModelError(format!(
            "could not resolve model directory for '{}'", model_name
        )))
    }

    fn dirs_hf_cache() -> PathBuf {
        if let Ok(cache) = std::env::var("HF_HOME") {
            return PathBuf::from(cache).join("hub");
        }
        if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
            return PathBuf::from(cache);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        PathBuf::from(home).join(".cache/huggingface/hub")
    }

    fn read_model_config(model_dir: &Path) -> Result<HfModelConfig> {
        let config_path = model_dir.join("config.json");
        let content = std::fs::read_to_string(&config_path).map_err(|e| {
            LLMError::ModelError(format!("failed to read {}: {e}", config_path.display()))
        })?;
        let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            LLMError::ModelError(format!("invalid config.json: {e}"))
        })?;

        let get_usize = |key: &str, default: usize| -> usize {
            json.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            json.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default)
        };

        let hidden_size = get_usize("hidden_size", 4096);
        let num_attention_heads = get_usize("num_attention_heads", 32);
        let head_dim = hidden_size / num_attention_heads;

        let architecture = json.get("architectures")
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
            tie_word_embeddings: json.get("tie_word_embeddings")
                .and_then(|v| v.as_bool()).unwrap_or(false),
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
    }

    // ------------------------------------------------------------------
    // Minimal FIFO scheduler
    // ------------------------------------------------------------------

    struct FifoScheduler {
        groups: Vec<SequenceGroup>,
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
    }

    impl FifoScheduler {
        fn new(max_num_seqs: usize, max_num_batched_tokens: usize) -> Self {
            Self { groups: Vec::new(), max_num_seqs, max_num_batched_tokens }
        }

        fn add_seq_group(&mut self, group: SequenceGroup) {
            self.groups.push(group);
        }

        fn abort_seq_group(&mut self, request_id: &RequestId) {
            self.groups.retain(|g| g.request_id != *request_id);
        }

        fn has_unfinished_seqs(&self) -> bool {
            !self.groups.is_empty()
        }

        /// Append a generated token to a sequence in the scheduler.
        fn update_seq_token(&mut self, seq_id: SequenceId, token_id: TokenId, logprob: f32) {
            for group in &mut self.groups {
                for seq in group.get_seqs_mut() {
                    if seq.seq_id == seq_id {
                        seq.append_token(token_id, logprob);
                        return;
                    }
                }
            }
        }

        fn get_num_unfinished_seq_groups(&self) -> usize {
            self.groups.len()
        }

        fn schedule(&mut self) -> (Vec<SequenceGroup>, usize) {
            self.groups.retain(|g| !g.is_finished());

            let mut selected = Vec::new();
            let mut total_tokens: usize = 0;

            for group in &self.groups {
                if selected.len() >= self.max_num_seqs { break; }
                let tokens_this: usize = group.get_seqs().iter()
                    .filter(|s| !s.is_finished())
                    .map(|s| if s.get_output_len() == 0 { s.get_len() } else { 1 })
                    .sum();
                if total_tokens + tokens_this > self.max_num_batched_tokens { break; }
                total_tokens += tokens_this;
                selected.push(group.clone());
            }

            (selected, total_tokens)
        }
    }

    // ------------------------------------------------------------------
    // GpuLLMEngine
    // ------------------------------------------------------------------

    pub struct GpuLLMEngine {
        config: EngineConfig,
        scheduler: FifoScheduler,
        worker: GpuWorker,
        tokenizer: Tokenizer,
        requests: HashMap<RequestId, EngineRequest>,
        next_request_id: AtomicU64,
        next_seq_id: u64,
        prefix_cache: Option<PrefixCache>,
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
            let tokenizer_path = config.model.tokenizer_path.as_deref()
                .unwrap_or(model_name);
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
                dtype: config.model.dtype.clone(),
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

            info!("GpuLLMEngine: ready for inference");
            Ok(Self {
                config,
                scheduler,
                worker,
                tokenizer,
                requests: HashMap::new(),
                next_request_id: AtomicU64::new(1),
                next_seq_id: 0,
                prefix_cache,
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
                return Err(LLMError::TokenizerError("prompt produced zero tokens".into()));
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
                request_id, seqs, params.clone(), Instant::now(), prompt.clone(),
            );
            self.scheduler.add_seq_group(seq_group);

            self.requests.insert(request_id, EngineRequest {
                request_id, prompt, prompt_token_ids,
                sampling_params: params, seq_states,
            });

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

        pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
            debug!("GpuLLMEngine: step begin");

            let (scheduled_groups, num_tokens) = self.scheduler.schedule();
            debug!(num_groups = scheduled_groups.len(), num_tokens, "scheduler output");

            if scheduled_groups.is_empty() {
                return Ok(Vec::new());
            }

            let metadata = Self::build_metadata(&scheduled_groups);

            // Prefix caching: before prefill, check for matching prefix blocks
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

            let worker_outputs = self.worker.execute(&metadata)
                .map_err(|e| LLMError::GpuError(format!("worker execute failed: {e}")))?;

            // Prefix caching: after prefill, register new prefix blocks
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

            let mut output_map: HashMap<u64, (TokenId, LogProb, Vec<(TokenId, LogProb)>)> = HashMap::new();
            for wo in &worker_outputs.outputs {
                output_map.insert(wo.seq_id, (wo.token_id, wo.logprob, wo.top_logprobs.clone()));
            }

            let mut results = Vec::new();
            let eos = self.tokenizer.eos_token_id();

            for group in &scheduled_groups {
                let request_id = group.request_id;
                let req = match self.requests.get_mut(&request_id) {
                    Some(r) => r,
                    None => continue,
                };

                let logprobs_requested = req.sampling_params.logprobs
                    .map(|n| n > 0)
                    .unwrap_or(false);

                for (seq_idx, seq) in group.get_seqs().iter().enumerate() {
                    if seq.is_finished() { continue; }

                    if let Some((token_id, logprob, top_lps)) = output_map.get(&seq.seq_id.0) {
                        let decoded = self.tokenizer.decode(&[*token_id]).unwrap_or_default();

                        // Update the scheduler's sequence with the new token
                        self.scheduler.update_seq_token(seq.seq_id, *token_id, *logprob);

                        let top_logprobs = if logprobs_requested && !top_lps.is_empty() {
                            Some(top_lps.clone())
                        } else {
                            None
                        };

                        if let Some(state) = req.seq_states.get_mut(seq_idx) {
                            OutputProcessor::process_token(
                                state, *token_id, *logprob, top_logprobs, &decoded,
                                &req.sampling_params, eos,
                            );
                        }
                    }
                }

                let output = OutputProcessor::build_request_output(
                    request_id, &req.prompt, &req.prompt_token_ids, &req.seq_states,
                );
                results.push(output);
            }

            // Clean up finished requests
            let finished_ids: Vec<RequestId> = self.requests.iter()
                .filter(|(_, req)| req.seq_states.iter().all(|s| s.is_finished()))
                .map(|(&id, _)| id)
                .collect();
            for id in &finished_ids {
                self.requests.remove(id);
                self.scheduler.abort_seq_group(id);
            }

            debug!(num_outputs = results.len(), "GpuLLMEngine: step complete");
            Ok(results)
        }

        pub fn run(&mut self) -> Result<Vec<RequestOutput>> {
            info!("GpuLLMEngine: run loop starting");
            let mut all_outputs = Vec::new();
            while self.has_unfinished() {
                let step_outputs = self.step()?;
                for output in step_outputs {
                    if output.finished {
                        all_outputs.push(output);
                    }
                }
            }
            info!(num_completed = all_outputs.len(), "GpuLLMEngine: run loop finished");
            Ok(all_outputs)
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

        fn build_metadata(groups: &[SequenceGroup]) -> Vec<SequenceGroupMetadata> {
            let mut metadata = Vec::with_capacity(groups.len());
            for group in groups {
                let is_prompt = group.get_seqs().iter().any(|s| s.get_output_len() == 0);
                let mut seq_data = HashMap::new();
                let block_tables = HashMap::new();

                for seq in group.get_seqs() {
                    if seq.is_finished() { continue; }
                    seq_data.insert(seq.seq_id, SequenceData {
                        prompt_token_ids: seq.prompt_token_ids.clone(),
                        output_token_ids: seq.output_token_ids.clone(),
                        cumulative_logprob: seq.cumulative_logprob,
                    });
                }

                metadata.push(SequenceGroupMetadata {
                    request_id: group.request_id,
                    is_prompt,
                    seq_data,
                    sampling_params: group.sampling_params.clone(),
                    block_tables,
                });
            }
            metadata
        }
    }

    unsafe impl Send for GpuLLMEngine {}
}

#[cfg(feature = "cuda")]
pub use inner::GpuLLMEngine;
