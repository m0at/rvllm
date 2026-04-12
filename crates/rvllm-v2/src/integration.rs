use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use rvllm_core::prelude::{RequestId, SamplingParams};
use rvllm_tokenizer::Tokenizer;

use crate::block_manager::{BlockManager, BlockManagerConfig};
use crate::engine::{Engine, EngineError, StepPending};
use crate::input::InputBuilder;
use crate::kv_cache::{CacheConfig, CudaKVCache};
use crate::runner::{GpuModelRunner, RunnerConfig};
use crate::scheduler::{PreemptionMode, Scheduler, SchedulerConfig};
use crate::types::V2RequestOutput;
use crate::worker::{Worker, WorkerConfig};

pub type ConcreteEngine = Engine<BlockManager>;

#[derive(Debug, Clone)]
pub struct V2EngineConfig {
    pub model_path: String,
    pub tokenizer_path: Option<String>,

    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_model_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    pub block_size: usize,
    pub gpu_memory_utilization: f32,
    pub gpu_memory_reserve_gb: f32,
    pub swap_space_gb: f32,
    pub num_gpu_blocks_override: Option<usize>,
    pub num_cpu_blocks_override: Option<usize>,
    pub prefix_cache_blocks: usize,

    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_prefill_chunk: usize,
    pub preemption_mode: PreemptionMode,

    pub device_id: usize,
    pub watermark: f32,
    pub graph_enabled: bool,
    pub fp8_weights: bool,
}

impl Default for V2EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            num_layers: 32,
            num_kv_heads: 8,
            num_attention_heads: 32,
            head_dim: 128,
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_model_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            block_size: 64,
            gpu_memory_utilization: 0.90,
            gpu_memory_reserve_gb: 0.0,
            swap_space_gb: 4.0,
            num_gpu_blocks_override: None,
            num_cpu_blocks_override: None,
            prefix_cache_blocks: 0,
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_prefill_chunk: 128,
            preemption_mode: PreemptionMode::Recompute,
            device_id: 0,
            watermark: 0.04,
            graph_enabled: true,
            fp8_weights: false,
        }
    }
}

impl V2EngineConfig {
    /// Read model config.json and fill in architecture params.
    /// Accepts either a local directory or a HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B").
    pub fn from_model_path(model_path: &str) -> Self {
        let resolved = resolve_hf_model_path(model_path);
        let config_path = std::path::Path::new(&resolved).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", config_path.display()));
        let v: serde_json::Value =
            serde_json::from_str(&config_str).expect("invalid config.json");

        let num_layers = v["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let hidden_size = v["hidden_size"].as_u64().unwrap_or(4096) as usize;
        let num_attention_heads = v["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let num_kv_heads = v["num_key_value_heads"].as_u64().unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = v["intermediate_size"].as_u64().unwrap_or(11008) as usize;
        let vocab_size = v["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let head_dim = hidden_size / num_attention_heads;
        let rms_norm_eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let rope_theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;

        Self {
            model_path: resolved,
            num_layers,
            num_kv_heads,
            num_attention_heads,
            head_dim,
            hidden_size,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            ..Default::default()
        }
    }

    fn scheduler_config(&self) -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: self.max_num_seqs,
            max_num_batched_tokens: self.max_num_batched_tokens,
            max_prefill_chunk: self.max_prefill_chunk,
            preemption_mode: self.preemption_mode,
            block_size: self.block_size,
        }
    }

    fn block_manager_config(
        &self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> BlockManagerConfig {
        BlockManagerConfig {
            num_gpu_blocks,
            num_cpu_blocks,
            block_size: self.block_size,
            watermark: self.watermark,
            prefix_cache_blocks: self.prefix_cache_blocks,
        }
    }

    fn cache_config(&self, num_gpu_blocks: usize, num_cpu_blocks: usize) -> CacheConfig {
        CacheConfig {
            num_layers: self.num_layers,
            num_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            block_size: self.block_size,
            num_gpu_blocks,
            num_cpu_blocks,
        }
    }

    fn runner_config(&self) -> RunnerConfig {
        RunnerConfig {
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            block_size: self.block_size,
            max_seq_len: self.max_model_len,
            max_num_seqs: self.max_num_seqs,
            max_num_batched_tokens: self.max_num_batched_tokens,
        }
    }

    fn worker_config(&self) -> WorkerConfig {
        WorkerConfig {
            block_size: self.block_size,
            max_batch_size: self.max_num_seqs,
            vocab_size: self.vocab_size,
            graph_enabled: self.graph_enabled,
        }
    }
}

pub struct IncomingRequest {
    pub prompt: String,
    pub sampling_params: SamplingParams,
    pub response_tx: tokio::sync::oneshot::Sender<Vec<V2RequestOutput>>,
}

fn profile_block_counts(config: &V2EngineConfig, available_gpu_bytes: usize) -> (usize, usize) {
    if let Some(gpu_override) = config.num_gpu_blocks_override {
        let cpu_override = config.num_cpu_blocks_override.unwrap_or(256);
        return (gpu_override, cpu_override);
    }

    let bytes_per_kv_block =
        2 * config.block_size * config.num_kv_heads * config.head_dim * 2;
    let bytes_per_block_all_layers = bytes_per_kv_block * config.num_layers;

    if bytes_per_block_all_layers == 0 {
        return (0, 0);
    }

    let usable = (available_gpu_bytes as f32 * config.gpu_memory_utilization) as usize
        - (config.gpu_memory_reserve_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let num_gpu_blocks = usable / bytes_per_block_all_layers;

    let cpu_budget = (config.swap_space_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let num_cpu_blocks = cpu_budget / bytes_per_block_all_layers;

    (num_gpu_blocks, num_cpu_blocks)
}

pub struct ServingHandle {
    request_tx: mpsc::UnboundedSender<IncomingRequest>,
    join_handle: tokio::task::JoinHandle<()>,
}

impl ServingHandle {
    pub fn sender(&self) -> mpsc::UnboundedSender<IncomingRequest> {
        self.request_tx.clone()
    }

    pub async fn shutdown(self) {
        drop(self.request_tx);
        let _ = self.join_handle.await;
    }
}

pub fn initialize(config: V2EngineConfig) -> ServingHandle {
    let (request_tx, request_rx) = mpsc::unbounded_channel();

    let join_handle = tokio::task::spawn_blocking(move || {
        run_serving_loop(config, request_rx);
    });

    ServingHandle {
        request_tx,
        join_handle,
    }
}

/// Full initialization sequence. Public for benchmarking.
///
/// Calls todo!() stubs for CUDA init and model loading.
pub fn init_engine(config: &V2EngineConfig) -> ConcreteEngine {
    let (context, stream) = init_cuda(config.device_id);
    info!(device = config.device_id, "CUDA context initialized");

    let runner_config = config.runner_config();
    let mut runner = load_model_and_build_runner(&runner_config, &config.model_path, Arc::clone(&stream));
    info!(layers = config.num_layers, "model loaded");

    if config.fp8_weights {
        runner.enable_fp8_weights().expect("FP8 weight quantization failed");
    }

    let available_gpu_bytes = query_available_gpu_memory(&context);
    let (num_gpu_blocks, num_cpu_blocks) = profile_block_counts(config, available_gpu_bytes);
    info!(num_gpu_blocks, num_cpu_blocks, "KV cache block counts computed");

    let cache_config = config.cache_config(num_gpu_blocks, num_cpu_blocks);
    let kv_cache =
        CudaKVCache::new(&cache_config, Arc::clone(&context), Arc::clone(&stream))
            .expect("KV cache allocation failed");
    info!("KV cache allocated");

    let input_builder = InputBuilder::new();

    let worker_config = config.worker_config();
    let worker = Worker::new(
        runner,
        kv_cache,
        input_builder,
        worker_config,
        Arc::clone(&context),
        Arc::clone(&stream),
    );
    info!("worker assembled");

    if config.graph_enabled {
        let sizes = pre_capture_batch_sizes(config.max_num_seqs);
        info!(?sizes, "CUDA graphs will be captured lazily for these batch sizes");
    }

    let bm_config = config.block_manager_config(num_gpu_blocks, num_cpu_blocks);
    let block_manager = BlockManager::new(bm_config);

    let sched_config = config.scheduler_config();
    let scheduler = Scheduler::new(sched_config, block_manager);

    let tokenizer_path = config
        .tokenizer_path
        .as_deref()
        .unwrap_or(&config.model_path);
    let tokenizer =
        Tokenizer::from_pretrained(tokenizer_path).expect("failed to load tokenizer");

    Engine::new(scheduler, worker, tokenizer)
}

fn run_serving_loop(
    config: V2EngineConfig,
    mut request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
) {
    let mut engine = init_engine(&config);
    info!("engine initialized, entering serving loop");

    let mut response_map: HashMap<RequestId, tokio::sync::oneshot::Sender<Vec<V2RequestOutput>>> =
        HashMap::new();

    loop {
        let mut drained = 0u64;
        while let Ok(incoming) = request_rx.try_recv() {
            match engine.add_request(incoming.prompt, incoming.sampling_params) {
                Ok(rid) => {
                    response_map.insert(rid, incoming.response_tx);
                    drained += 1;
                }
                Err(e) => {
                    warn!("failed to add request: {e}");
                }
            }
        }
        if drained > 0 {
            debug!(drained, "drained incoming requests");
        }

        if !engine.has_pending_work() {
            match request_rx.blocking_recv() {
                Some(incoming) => {
                    match engine.add_request(incoming.prompt, incoming.sampling_params) {
                        Ok(rid) => {
                            response_map.insert(rid, incoming.response_tx);
                        }
                        Err(e) => {
                            warn!("failed to add request: {e}");
                        }
                    }
                }
                None => {
                    info!("request channel closed, shutting down");
                    break;
                }
            }
        }

        let outputs = match engine.step() {
            Ok(outputs) => outputs,
            Err(e) => {
                warn!("engine step failed: {e}");
                continue;
            }
        };

        for output in &outputs {
            if output.finished {
                if let Some(tx) = response_map.remove(&output.request_id) {
                    let _ = tx.send(vec![output.clone()]);
                }
            }
        }
    }
}

pub struct AsyncEngine {
    engine: ConcreteEngine,
    response_map: HashMap<RequestId, tokio::sync::oneshot::Sender<Vec<V2RequestOutput>>>,
}

impl AsyncEngine {
    pub fn new(engine: ConcreteEngine) -> Self {
        Self {
            engine,
            response_map: HashMap::new(),
        }
    }

    pub fn add_request(
        &mut self,
        prompt: String,
        sampling_params: SamplingParams,
        response_tx: tokio::sync::oneshot::Sender<Vec<V2RequestOutput>>,
    ) -> Result<RequestId, EngineError> {
        let id = self.engine.add_request(prompt, sampling_params)?;
        self.response_map.insert(id, response_tx);
        Ok(id)
    }

    pub fn has_pending_work(&self) -> bool {
        self.engine.has_pending_work()
    }

    pub fn step_launch(&mut self) -> Result<Option<StepPending>, EngineError> {
        self.engine.step_launch()
    }

    pub fn step_collect(
        &mut self,
        pending: Option<StepPending>,
    ) -> Result<Vec<V2RequestOutput>, EngineError> {
        self.engine.step_collect(pending)
    }

    pub fn step(&mut self) -> Result<Vec<V2RequestOutput>, EngineError> {
        self.engine.step()
    }

    pub fn route_outputs(&mut self, outputs: &[V2RequestOutput]) {
        for output in outputs {
            if output.finished {
                if let Some(tx) = self.response_map.remove(&output.request_id) {
                    let _ = tx.send(vec![output.clone()]);
                }
            }
        }
    }
}

fn pre_capture_batch_sizes(max_num_seqs: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut n = 1;
    while n <= max_num_seqs {
        sizes.push(n);
        if n >= max_num_seqs {
            break;
        }
        n *= 2;
    }
    if sizes.last() != Some(&max_num_seqs) {
        sizes.push(max_num_seqs);
    }
    sizes
}

fn init_cuda(
    device_id: usize,
) -> (
    Arc<cudarc::driver::CudaContext>,
    Arc<cudarc::driver::CudaStream>,
) {
    let ctx =
        cudarc::driver::CudaContext::new(device_id).expect("failed to create CUDA context");
    // Disable cudarc's per-CudaSlice event tracking. It records read/write
    // CudaEvents on every allocation, which causes
    // CUDA_ERROR_STREAM_CAPTURE_ISOLATION during graph capture (events from
    // pre-capture work create illegal cross-phase dependencies). We use a
    // single stream so event-based multi-stream sync is unnecessary.
    unsafe { ctx.disable_event_tracking(); }
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    (ctx, stream)
}

fn load_model_and_build_runner(
    config: &RunnerConfig,
    model_path: &str,
    stream: Arc<cudarc::driver::CudaStream>,
) -> GpuModelRunner {
    use cudarc::driver::CudaSlice;
    use half::f16;
    use rvllm_core::prelude::LLMError;
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::cublaslt_ops::CublasLtOps;
    use rvllm_gpu::cublas_autotune::{CublasAutotuner, GemmDtype};
    use rvllm_gpu::kernel_loader::KernelLoader;

    let path = std::path::Path::new(model_path);

    // 1. Load raw weights (safetensors -> GPU f16)
    let raw_weights: std::collections::HashMap<String, CudaSlice<f16>> =
        rvllm_model_loader::gpu_loader::load_weights_to_gpu(path, &stream)
            .expect("failed to load model weights to GPU");
    info!("loaded {} weight tensors to GPU", raw_weights.len());

    // Helper: find weight by name with model. prefix fallback
    let get_weight = |name: &str| -> CudaSlice<f16> {
        raw_weights
            .get(name)
            .or_else(|| {
                name.strip_prefix("model.")
                    .and_then(|rest| raw_weights.get(&format!("model.language_model.{rest}")))
            })
            .unwrap_or_else(|| panic!("missing weight: {name}"))
            .clone()
    };

    // 2. Extract global weights
    let embed_tokens = get_weight("model.embed_tokens.weight");
    let final_norm_weight = get_weight("model.norm.weight");
    let lm_head_weight = raw_weights
        .get("lm_head.weight")
        .cloned()
        .unwrap_or_else(|| get_weight("model.embed_tokens.weight"));

    // 3. Extract + fuse per-layer weights
    let num_layers = config.num_layers;
    let mut fused_qkv = Vec::with_capacity(num_layers);
    let mut fused_gate_up = Vec::with_capacity(num_layers);
    let mut o_proj = Vec::with_capacity(num_layers);
    let mut down_proj = Vec::with_capacity(num_layers);
    let mut input_ln = Vec::with_capacity(num_layers);
    let mut post_ln = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let prefix = format!("model.layers.{i}");

        // QKV fusion: concat q_proj, k_proj, v_proj along output dim
        let q = get_weight(&format!("{prefix}.self_attn.q_proj.weight"));
        let k = get_weight(&format!("{prefix}.self_attn.k_proj.weight"));
        let v = get_weight(&format!("{prefix}.self_attn.v_proj.weight"));
        let q_host = stream.clone_dtoh(&q).expect("dtoh q");
        let k_host = stream.clone_dtoh(&k).expect("dtoh k");
        let v_host = stream.clone_dtoh(&v).expect("dtoh v");
        let mut qkv_host = Vec::with_capacity(q_host.len() + k_host.len() + v_host.len());
        qkv_host.extend_from_slice(&q_host);
        qkv_host.extend_from_slice(&k_host);
        qkv_host.extend_from_slice(&v_host);
        let qkv_gpu = stream.clone_htod(&qkv_host).expect("htod qkv");
        fused_qkv.push(qkv_gpu);

        // GateUp fusion: concat gate_proj, up_proj
        let gate = get_weight(&format!("{prefix}.mlp.gate_proj.weight"));
        let up = get_weight(&format!("{prefix}.mlp.up_proj.weight"));
        let gate_host = stream.clone_dtoh(&gate).expect("dtoh gate");
        let up_host = stream.clone_dtoh(&up).expect("dtoh up");
        let mut gu_host = Vec::with_capacity(gate_host.len() + up_host.len());
        gu_host.extend_from_slice(&gate_host);
        gu_host.extend_from_slice(&up_host);
        let gu_gpu = stream.clone_htod(&gu_host).expect("htod gate_up");
        fused_gate_up.push(gu_gpu);

        o_proj.push(get_weight(&format!("{prefix}.self_attn.o_proj.weight")));
        down_proj.push(get_weight(&format!("{prefix}.mlp.down_proj.weight")));
        input_ln.push(get_weight(&format!("{prefix}.input_layernorm.weight")));
        post_ln.push(get_weight(&format!(
            "{prefix}.post_attention_layernorm.weight"
        )));
    }
    info!(num_layers, "per-layer weights extracted and fused");

    // 4. Build RoPE tables
    let half_dim = config.head_dim / 2;
    let max_pos = config.max_seq_len.min(32768);
    let rope_theta = 1_000_000.0f32; // Qwen2.5 default
    let mut cos_table = vec![0.0f32; max_pos * half_dim];
    let mut sin_table = vec![0.0f32; max_pos * half_dim];
    for pos in 0..max_pos {
        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / config.head_dim as f32);
            let theta = pos as f32 * freq;
            cos_table[pos * half_dim + i] = theta.cos();
            sin_table[pos * half_dim + i] = theta.sin();
        }
    }
    let rope_cos = stream.clone_htod(&cos_table).expect("rope cos htod");
    let rope_sin = stream.clone_htod(&sin_table).expect("rope sin htod");
    info!(max_pos, half_dim, "RoPE tables uploaded");

    // 5. Create kernel loader + cuBLAS
    let kernel_dir_str = std::env::var("RVLLM_PTX_DIR").unwrap_or_else(|_| "kernels".into());
    let kernel_dir = std::path::Path::new(&kernel_dir_str);
    let loader = KernelLoader::new(
        stream.context().clone(),
        Arc::clone(&stream),
        kernel_dir,
    )
    .expect("failed to create kernel loader");
    loader.validate_required_kernels();

    let cublas =
        CublasHandle::new(Arc::clone(&stream)).expect("failed to create cuBLAS handle");

    // 5b. cuBLASLt + autotuning for decode GEMM shapes
    let lt_ops = {
        let lt = CublasLtOps::new(Arc::clone(&stream)).expect("cublasLt init");
        let q_dim = config.num_heads * config.head_dim;
        let qkv_dim = q_dim + 2 * config.num_kv_heads * config.head_dim;
        let gate_up_dim = config.intermediate_size * 2;

        // Query GPU name for autotune cache key
        let gpu_name = query_gpu_name();
        info!(%gpu_name, "autotuning cublasLt algorithms for decode shapes");

        let tuner = CublasAutotuner::autotune_model(
            &lt,
            GemmDtype::F16,
            config.hidden_size,
            q_dim,
            qkv_dim,
            config.intermediate_size,
            gate_up_dim,
            &gpu_name,
        )
        .expect("cublasLt autotune failed");

        info!(algos = tuner.len(), "autotuned cublasLt algorithms");
        lt.install_autotuned(&tuner);
        Some(lt)
    };

    // 6. Try loading CUTLASS
    let cutlass = {
        use rvllm_gpu::cutlass_ffi::CutlassKernels;
        let candidates = [
            "kernels/sm_90/libcutlass_kernels.so",
            "kernels/sm_80/libcutlass_kernels.so",
        ];
        let mut loaded = None;
        for p in &candidates {
            let path = std::path::Path::new(p);
            if path.exists() {
                if let Ok(k) = CutlassKernels::load(path) {
                    info!(?path, "CUTLASS loaded");
                    loaded = Some(Arc::new(k));
                    break;
                }
            }
        }
        loaded
    };

    // 7. Create transformer layers
    let loader = Arc::new(loader);
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let layer_cfg = crate::layer::LayerConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta,
            max_position: max_pos,
            block_size: config.block_size,
        };
        layers.push(crate::layer::GpuTransformerLayer::new(
            layer_cfg,
            Arc::clone(&stream),
            Arc::clone(&loader),
        ));
    }

    // 8. Construct v2 runner
    GpuModelRunner::new(
        config.clone(),
        layers,
        cutlass,
        cublas,
        lt_ops,
        stream,
        loader,
        embed_tokens,
        lm_head_weight,
        final_norm_weight,
        fused_qkv,
        fused_gate_up,
        o_proj,
        down_proj,
        input_ln,
        post_ln,
        rope_cos,
        rope_sin,
    )
    .expect("failed to create GpuModelRunner")
}

fn query_available_gpu_memory(context: &cudarc::driver::CudaContext) -> usize {
    context.bind_to_thread().expect("bind cuda context");
    let (free, _total) = cudarc::driver::result::mem_get_info().expect("cuMemGetInfo");
    free
}

fn query_gpu_name() -> String {
    let mut name = [0i8; 256];
    unsafe {
        cudarc::driver::sys::cuDeviceGetName(name.as_mut_ptr(), 256, 0);
    }
    let bytes: Vec<u8> = name.iter().take_while(|&&b| b != 0).map(|&b| b as u8).collect();
    String::from_utf8(bytes).unwrap_or_else(|_| "unknown_gpu".to_string())
}

/// Resolve a HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B") to a local cache path.
/// If the path already exists as a directory, returns it unchanged.
fn resolve_hf_model_path(model_id: &str) -> String {
    let p = std::path::Path::new(model_id);
    if p.is_dir() {
        return model_id.to_string();
    }
    // HF cache layout: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>/
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let cache_dir = std::path::Path::new(&home).join(".cache/huggingface/hub");
    let repo_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = cache_dir.join(&repo_dir_name).join("snapshots");
    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        let mut snaps: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        snaps.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));
        if let Some(latest) = snaps.first() {
            let resolved = latest.path().to_string_lossy().to_string();
            eprintln!("Resolved model '{}' -> {}", model_id, resolved);
            return resolved;
        }
    }
    model_id.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_capture_sizes_power_of_two() {
        assert_eq!(pre_capture_batch_sizes(8), vec![1, 2, 4, 8]);
    }

    #[test]
    fn pre_capture_sizes_non_power_of_two() {
        assert_eq!(pre_capture_batch_sizes(10), vec![1, 2, 4, 8, 10]);
    }

    #[test]
    fn pre_capture_sizes_one() {
        assert_eq!(pre_capture_batch_sizes(1), vec![1]);
    }

    #[test]
    fn pre_capture_sizes_256() {
        let sizes = pre_capture_batch_sizes(256);
        assert_eq!(sizes, vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);
    }

    #[test]
    fn block_count_with_override() {
        let config = V2EngineConfig {
            num_gpu_blocks_override: Some(500),
            num_cpu_blocks_override: Some(100),
            ..Default::default()
        };
        let (gpu, cpu) = profile_block_counts(&config, 0);
        assert_eq!(gpu, 500);
        assert_eq!(cpu, 100);
    }

    #[test]
    fn block_count_from_memory() {
        let config = V2EngineConfig {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            gpu_memory_utilization: 0.90,
            gpu_memory_reserve_gb: 0.0,
            swap_space_gb: 4.0,
            ..Default::default()
        };
        let available = 40usize * 1024 * 1024 * 1024;
        let (gpu, cpu) = profile_block_counts(&config, available);
        // bytes_per_kv_block = 2(K+V) * 16 * 8 * 128 * 2(f16) = 65536
        // bytes_per_block_all_layers = 65536 * 32 = 2,097,152
        // 40 GiB * 0.9 = 38,654,705,664 / 2,097,152 = 18,432
        assert_eq!(gpu, 18432);
        assert!(cpu > 0);
    }

    #[test]
    fn default_config_values() {
        let c = V2EngineConfig::default();
        assert_eq!(c.block_size, 64);
        assert_eq!(c.max_num_seqs, 256);
        assert_eq!(c.gpu_memory_utilization, 0.90);
        assert_eq!(c.watermark, 0.04);
        assert!(c.graph_enabled);
    }

    #[test]
    fn config_produces_matching_sub_configs() {
        let c = V2EngineConfig::default();

        let sc = c.scheduler_config();
        assert_eq!(sc.max_num_seqs, c.max_num_seqs);
        assert_eq!(sc.max_num_batched_tokens, c.max_num_batched_tokens);
        assert_eq!(sc.block_size, c.block_size);

        let bmc = c.block_manager_config(1000, 200);
        assert_eq!(bmc.num_gpu_blocks, 1000);
        assert_eq!(bmc.num_cpu_blocks, 200);
        assert_eq!(bmc.block_size, c.block_size);
        assert_eq!(bmc.watermark, c.watermark);

        let cc = c.cache_config(1000, 200);
        assert_eq!(cc.num_layers, c.num_layers);
        assert_eq!(cc.num_heads, c.num_kv_heads);
        assert_eq!(cc.num_gpu_blocks, 1000);

        let rc = c.runner_config();
        assert_eq!(rc.num_layers, c.num_layers);
        assert_eq!(rc.num_heads, c.num_attention_heads);
        assert_eq!(rc.num_kv_heads, c.num_kv_heads);
        assert_eq!(rc.block_size, c.block_size);
        assert_eq!(rc.max_seq_len, c.max_model_len);

        let wc = c.worker_config();
        assert_eq!(wc.block_size, c.block_size);
        assert_eq!(wc.max_batch_size, c.max_num_seqs);
        assert_eq!(wc.vocab_size, c.vocab_size);
        assert!(wc.graph_enabled);
    }
}
