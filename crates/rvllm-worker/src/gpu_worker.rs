//! GPU Worker: single-device execution context for real CUDA inference.
//!
//! Loads model weights to GPU via rvllm-model-loader's gpu_loader, then runs
//! a real transformer forward pass using cuBLAS SGEMM for each layer.
//! Supports Qwen2/Llama-family architectures with RoPE, KV cache, and GQA.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::{debug, info, trace, warn};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use rvllm_core::prelude::{BlockId, LLMError, Result, SamplingParams, TokenId};
use rvllm_gpu::prelude::{CublasHandle, CudaGpuAllocator, GpuStream};

use rvllm_kv_cache::CacheEngine;
use rvllm_model_runner::gpu_runner::{
    DecodeExecutionPlan, DecodeGraphAction, DecodeGraphDispatch, DecodeGraphRuntimeState,
    ForwardOutput,
};
use rvllm_model_runner::input::ModelInput;
use rvllm_model_runner::ModelRunnerConfig;
use rvllm_sampling::batch::make_rng;
use rvllm_sampling::guided::{GuidedDecodingState, VocabTable};
use rvllm_sampling::sampler::Sampler;
use rvllm_sequence::{SequenceData, SequenceGroupMetadata};

use rvllm_core::prelude::ResponseFormat;

use crate::config::WorkerConfig;
use crate::graph_runner::{GraphRunner, GraphRunnerConfig};
use crate::input;

/// Output of one GPU worker step, mapping each sequence to its sampled token.
#[derive(Debug, Clone)]
pub struct GpuWorkerOutput {
    pub outputs: Vec<GpuSamplerResult>,
}

/// Per-sequence sampling result from the GPU worker.
#[derive(Debug, Clone)]
pub struct GpuSamplerResult {
    pub seq_id: u64,
    pub token_id: TokenId,
    pub logprob: f32,
    pub top_logprobs: Vec<(TokenId, f32)>,
}

/// Per-layer weights stored as CudaSlice<f16> on GPU.
struct LayerWeights {
    input_layernorm: CudaSlice<half::f16>,
    post_attention_layernorm: CudaSlice<half::f16>,
    q_proj_weight: CudaSlice<half::f16>,
    q_proj_bias: Option<CudaSlice<half::f16>>,
    k_proj_weight: CudaSlice<half::f16>,
    k_proj_bias: Option<CudaSlice<half::f16>>,
    v_proj_weight: CudaSlice<half::f16>,
    v_proj_bias: Option<CudaSlice<half::f16>>,
    o_proj_weight: CudaSlice<half::f16>,
    gate_proj_weight: CudaSlice<half::f16>,
    up_proj_weight: CudaSlice<half::f16>,
    down_proj_weight: CudaSlice<half::f16>,
}

/// All model weights on GPU (f16).
struct GpuModelWeights {
    embed_tokens: CudaSlice<half::f16>,
    layers: Vec<LayerWeights>,
    norm_weight: CudaSlice<half::f16>,
    lm_head_weight: Option<CudaSlice<half::f16>>, // None if tie_word_embeddings
    tie_word_embeddings: bool,
}

/// Precomputed cos/sin tables for Rotary Position Embeddings.
struct RopeTable {
    /// cos_table[pos * half_dim + i] = cos(pos * freq_i)
    cos_table: Vec<f32>,
    /// sin_table[pos * half_dim + i] = sin(pos * freq_i)
    sin_table: Vec<f32>,
    half_dim: usize,
    max_seq_len: usize,
}

impl RopeTable {
    fn new(head_dim: usize, max_seq_len: usize, rope_theta: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_table = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let theta = pos as f32 * freq;
                cos_table[pos * half_dim + i] = theta.cos();
                sin_table[pos * half_dim + i] = theta.sin();
            }
        }

        Self {
            cos_table,
            sin_table,
            half_dim,
            max_seq_len,
        }
    }

    /// Apply RoPE to a Q or K vector in-place.
    /// `data` layout: [num_tokens, num_heads * head_dim]
    /// `positions`: position ID for each token
    fn apply(
        &self,
        data: &mut [f32],
        positions: &[u32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let row_stride = num_heads * head_dim;
        for t in 0..num_tokens {
            let pos = positions[t] as usize;
            if pos >= self.max_seq_len {
                continue;
            }
            for h in 0..num_heads {
                let base = t * row_stride + h * head_dim;
                for i in 0..self.half_dim {
                    let cos_val = self.cos_table[pos * self.half_dim + i];
                    let sin_val = self.sin_table[pos * self.half_dim + i];
                    let x0 = data[base + 2 * i];
                    let x1 = data[base + 2 * i + 1];
                    data[base + 2 * i] = x0 * cos_val - x1 * sin_val;
                    data[base + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

/// KV cache: stores keys and values for all layers across generation steps.
struct KVCache {
    /// key_cache[layer]: flat vec of [cached_tokens * num_kv_heads * head_dim]
    key_cache: Vec<Vec<f32>>,
    /// value_cache[layer]: same layout
    value_cache: Vec<Vec<f32>>,
    /// Number of tokens currently cached
    cached_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KVCache {
    fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            key_cache: vec![Vec::new(); num_layers],
            value_cache: vec![Vec::new(); num_layers],
            cached_len: 0,
            num_kv_heads,
            head_dim,
        }
    }

    fn clear(&mut self) {
        for layer in &mut self.key_cache {
            layer.clear();
        }
        for layer in &mut self.value_cache {
            layer.clear();
        }
        self.cached_len = 0;
    }

    /// Append new K,V data for a layer. k_new/v_new: [num_new_tokens, num_kv_heads * head_dim]
    fn append(&mut self, layer_idx: usize, k_new: &[f32], v_new: &[f32]) {
        self.key_cache[layer_idx].extend_from_slice(k_new);
        self.value_cache[layer_idx].extend_from_slice(v_new);
    }

    /// Mark that we've finished appending for all layers this step.
    fn advance(&mut self, num_new_tokens: usize) {
        self.cached_len += num_new_tokens;
    }

    /// Get full cached K for a layer (including what was just appended).
    fn keys(&self, layer_idx: usize) -> &[f32] {
        &self.key_cache[layer_idx]
    }

    /// Get full cached V for a layer.
    fn values(&self, layer_idx: usize) -> &[f32] {
        &self.value_cache[layer_idx]
    }

    /// Total tokens in cache (after advance).
    fn len(&self) -> usize {
        self.cached_len
    }
}

/// FP8-quantized KV cache: stores keys and values as u8 + f32 scales.
struct Fp8KVCache {
    /// key_data[layer]: quantized u8 vec of [cached_tokens * num_kv_heads * head_dim]
    key_data: Vec<Vec<u8>>,
    /// value_data[layer]: same layout
    value_data: Vec<Vec<u8>>,
    /// key_scales[layer]: per-token per-head scales [cached_tokens * num_kv_heads]
    key_scales: Vec<Vec<f32>>,
    /// value_scales[layer]: same layout
    value_scales: Vec<Vec<f32>>,
    cached_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Fp8KVCache {
    fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            key_data: vec![Vec::new(); num_layers],
            value_data: vec![Vec::new(); num_layers],
            key_scales: vec![Vec::new(); num_layers],
            value_scales: vec![Vec::new(); num_layers],
            cached_len: 0,
            num_kv_heads,
            head_dim,
        }
    }

    fn clear(&mut self) {
        for v in &mut self.key_data {
            v.clear();
        }
        for v in &mut self.value_data {
            v.clear();
        }
        for v in &mut self.key_scales {
            v.clear();
        }
        for v in &mut self.value_scales {
            v.clear();
        }
        self.cached_len = 0;
    }

    /// Quantize and append new K,V for a layer.
    /// k_new/v_new: [num_new_tokens, num_kv_heads * head_dim] as f32.
    fn append_quantized(&mut self, layer_idx: usize, k_new: &[f32], v_new: &[f32]) {
        let stride = self.num_kv_heads * self.head_dim;
        let num_tokens = k_new.len() / stride;
        for t in 0..num_tokens {
            let k_slice = &k_new[t * stride..(t + 1) * stride];
            let v_slice = &v_new[t * stride..(t + 1) * stride];
            let (kq, ks) =
                rvllm_kv_cache::quantize_heads(k_slice, self.num_kv_heads, self.head_dim);
            let (vq, vs) =
                rvllm_kv_cache::quantize_heads(v_slice, self.num_kv_heads, self.head_dim);
            self.key_data[layer_idx].extend_from_slice(&kq);
            self.value_data[layer_idx].extend_from_slice(&vq);
            self.key_scales[layer_idx].extend_from_slice(&ks);
            self.value_scales[layer_idx].extend_from_slice(&vs);
        }
    }

    fn advance(&mut self, num_new_tokens: usize) {
        self.cached_len += num_new_tokens;
    }

    /// Dequantize all cached K for a layer back to f32.
    fn keys_dequantized(&self, layer_idx: usize) -> Vec<f32> {
        let stride = self.num_kv_heads * self.head_dim;
        let total_tokens = self.key_data[layer_idx].len() / stride;
        let mut out = Vec::with_capacity(total_tokens * stride);
        for t in 0..total_tokens {
            let data = &self.key_data[layer_idx][t * stride..(t + 1) * stride];
            let scales =
                &self.key_scales[layer_idx][t * self.num_kv_heads..(t + 1) * self.num_kv_heads];
            out.extend_from_slice(&rvllm_kv_cache::dequantize_heads(
                data,
                scales,
                self.num_kv_heads,
                self.head_dim,
            ));
        }
        out
    }

    /// Dequantize all cached V for a layer back to f32.
    fn values_dequantized(&self, layer_idx: usize) -> Vec<f32> {
        let stride = self.num_kv_heads * self.head_dim;
        let total_tokens = self.value_data[layer_idx].len() / stride;
        let mut out = Vec::with_capacity(total_tokens * stride);
        for t in 0..total_tokens {
            let data = &self.value_data[layer_idx][t * stride..(t + 1) * stride];
            let scales =
                &self.value_scales[layer_idx][t * self.num_kv_heads..(t + 1) * self.num_kv_heads];
            out.extend_from_slice(&rvllm_kv_cache::dequantize_heads(
                data,
                scales,
                self.num_kv_heads,
                self.head_dim,
            ));
        }
        out
    }

    fn len(&self) -> usize {
        self.cached_len
    }
}

/// Single-GPU worker that owns the model weights and cuBLAS handle for
/// real inference on one CUDA device.
pub struct GpuWorker {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cublas: CublasHandle,
    cache_stream: GpuStream,
    compute_stream: GpuStream,
    cache_engine: Option<CacheEngine>,
    sampler: Sampler,
    config: WorkerConfig,
    device_id: usize,
    runner_config: Option<ModelRunnerConfig>,
    vocab_size: usize,
    model_weights: Option<GpuModelWeights>,
    rms_norm_eps: f32,
    rope_table: Option<RopeTable>,
    kv_cache: Option<KVCache>,
    fp8_kv_cache: Option<Fp8KVCache>,
    use_fp8_kv: bool,
    /// GPU-resident FP8 cache engine (created when use_fp8_kv is true).
    #[cfg(feature = "cuda")]
    fp8_cache_engine: Option<rvllm_kv_cache::CudaFP8CacheEngine>,
    guided_states: HashMap<u64, GuidedDecodingState>,
    vocab_table: Option<VocabTable>,
    /// Raw f16 weight map preserved for deferred GpuModelRunner construction.
    #[cfg(feature = "cuda")]
    raw_weight_map: Option<HashMap<String, CudaSlice<half::f16>>>,
    #[cfg(feature = "cuda")]
    raw_weight_shapes: Option<HashMap<String, Vec<usize>>>,
    /// GPU-resident model runner (full forward pass on GPU, no CPU attention fallback).
    /// Constructed in init_cache() once cache geometry is known.
    #[cfg(feature = "cuda")]
    gpu_model_runner: Option<rvllm_model_runner::gpu_runner::GpuModelRunner>,
    /// CUDA graph runner for decode step capture/replay.
    graph_runner: GraphRunner,
    /// GpuStream wrapping the main compute stream (same Arc as self.stream).
    /// Used to replay CUDA graphs on the correct stream (the runner's stream).
    #[cfg(feature = "cuda")]
    runner_stream: GpuStream,
    /// Number of forward calls so far (for warmup before graph capture).
    forward_count: usize,
    /// Pinned host buffer for async DtoH pipeline. Allocated on first use.
    #[cfg(feature = "cuda")]
    pinned_output: Option<cudarc::driver::PinnedHostSlice<i32>>,
    /// Stored sync output for execute_launch/execute_collect pipeline.
    pending_sync_output: Option<(ForwardOutput, Vec<SequenceGroupMetadata>)>,
    decode_input_scratch: input::DecodeInputScratch,
}

impl GpuWorker {
    pub fn new(config: WorkerConfig) -> Result<Self> {
        let device_id = config.device_id;
        info!(device_id, "creating GpuWorker");

        let context = CudaContext::new(device_id).map_err(|e| {
            LLMError::GpuError(format!("failed to init CUDA device {}: {}", device_id, e))
        })?;

        // Disable cudarc's per-CudaSlice event tracking. Event tracking records
        // read/write CudaEvents on every allocation, which causes
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION during graph capture (events from
        // pre-capture work create illegal cross-phase dependencies). Since we use
        // a single stream, event-based multi-stream sync is unnecessary.
        #[cfg(feature = "cuda")]
        unsafe {
            context.disable_event_tracking();
        }

        // Use a non-default stream for all GPU operations. The legacy default
        // stream (stream 0) does NOT support cuStreamBeginCapture, which is
        // required for CUDA graph capture/replay.
        let stream = context
            .new_stream()
            .map_err(|e| LLMError::GpuError(format!("failed to create CUDA stream: {e}")))?;

        // Set memory pool to never release freed memory (instant reuse).
        // By default CUDA's pool may return memory to the OS aggressively;
        // threshold=u64::MAX keeps all freed allocations in the pool.
        unsafe {
            use cudarc::driver::sys::{
                cuDeviceGetDefaultMemPool, cuMemPoolSetAttribute, CUmemPool_attribute, CUmemoryPool,
            };
            let mut pool: CUmemoryPool = std::ptr::null_mut();
            let res = cuDeviceGetDefaultMemPool(&mut pool, device_id as i32);
            if res == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                let mut threshold: u64 = u64::MAX;
                cuMemPoolSetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut std::ffi::c_void,
                );
                info!(device_id, "memory pool release threshold set to u64::MAX");
            } else {
                warn!(
                    device_id,
                    "failed to configure memory pool release threshold"
                );
            }
        }

        let cublas = CublasHandle::new(stream.clone())
            .map_err(|e| LLMError::GpuError(format!("failed to create CublasHandle: {}", e)))?;

        let cache_stream = GpuStream::new(device_id)
            .map_err(|e| LLMError::GpuError(format!("failed to create cache stream: {}", e)))?;

        let compute_stream = GpuStream::new(device_id)
            .map_err(|e| LLMError::GpuError(format!("failed to create compute stream: {}", e)))?;

        info!(device_id, "GpuWorker CUDA resources initialized");

        // Resolve Dtype::Auto based on GPU compute capability
        let mut config = config;
        if config.dtype.is_auto() {
            let devices = rvllm_gpu::device::list_devices();
            let sm_major = devices.first().map(|d| d.compute_capability.0).unwrap_or(8);
            config.dtype = config.dtype.resolve(sm_major);
            info!(dtype = %config.dtype, sm_major, "resolved dtype from Auto");
        }

        let use_fp8_kv = matches!(
            config.kv_cache_dtype.to_lowercase().as_str(),
            "fp8" | "fp8_e4m3"
        ) || std::env::var("RVLLM_FP8_KV").map_or(false, |v| v == "1");
        if use_fp8_kv {
            info!("FP8 KV cache enabled (halves KV VRAM via FP8 E4M3 quantization)");
        }

        let graph_runner = GraphRunner::new(GraphRunnerConfig {
            max_batch_size: 256,
            enabled: true,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
        });

        // GpuStream that wraps the same Arc<CudaStream> used by the model runner.
        // Graph replay must happen on this stream so that HtoD metadata uploads
        // (also on this stream) are ordered before the graph's kernels.
        #[cfg(feature = "cuda")]
        let runner_stream = GpuStream::from_arc(device_id, context.clone(), stream.clone());

        Ok(Self {
            context,
            stream,
            cublas,
            cache_stream,
            compute_stream,
            cache_engine: None,
            sampler: Sampler::new(),
            config,
            device_id,
            runner_config: None,
            vocab_size: 0,
            model_weights: None,
            rms_norm_eps: 1e-6,
            rope_table: None,
            kv_cache: None,
            fp8_kv_cache: None,
            use_fp8_kv,
            #[cfg(feature = "cuda")]
            fp8_cache_engine: None,
            guided_states: HashMap::new(),
            vocab_table: None,
            #[cfg(feature = "cuda")]
            raw_weight_map: None,
            #[cfg(feature = "cuda")]
            raw_weight_shapes: None,
            #[cfg(feature = "cuda")]
            gpu_model_runner: None,
            graph_runner,
            #[cfg(feature = "cuda")]
            runner_stream,
            forward_count: 0,
            #[cfg(feature = "cuda")]
            pinned_output: None,
            pending_sync_output: None,
            decode_input_scratch: input::DecodeInputScratch::new(),
        })
    }

    fn prepare_model_input(&mut self, metadata: &[SequenceGroupMetadata]) -> Result<ModelInput> {
        input::prepare_input_reuse(
            &mut self.decode_input_scratch,
            metadata,
            self.config.block_size,
        )
    }

    #[cfg(feature = "cuda")]
    fn decode_graph_dispatch(
        &self,
        batch: usize,
        is_prefill: bool,
        is_decode: bool,
    ) -> Result<DecodeGraphDispatch> {
        let runner = self
            .gpu_model_runner
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;
        let execution = runner.decode_execution_plan(batch, is_prefill, is_decode);
        Ok(runner.decode_graph_dispatch(
            batch,
            is_prefill,
            is_decode,
            DecodeGraphRuntimeState {
                graphs_enabled: self.graph_runner.is_enabled(),
                exact_graph_available: self.graph_runner.has_graph_for_exact(execution.graph_tokens),
                warmup_complete: self.forward_count > Self::GRAPH_WARMUP_CALLS,
                capture_attempted: self
                    .graph_runner
                    .was_capture_attempted(execution.graph_tokens),
            },
        ))
    }

    #[cfg(feature = "cuda")]
    fn try_gpu_forward_persistent_decode(
        &mut self,
        metadata: &[SequenceGroupMetadata],
        _greedy_only: bool,
    ) -> Result<Option<ForwardOutput>> {
        if metadata.is_empty() || metadata.iter().any(|g| g.is_prompt) {
            return Ok(None);
        }

        let batch: usize = metadata.iter().map(|g| g.seq_data.len()).sum();
        if batch == 0 {
            return Ok(None);
        }

        let dispatch = self.decode_graph_dispatch(batch, false, true)?;
        if !dispatch.execution.use_batched_v2 || matches!(dispatch.action, DecodeGraphAction::Raw)
        {
            return Ok(None);
        }

        let runner = self
            .gpu_model_runner
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;
        input::prepare_decode_persistent_reuse(
            &mut self.decode_input_scratch,
            metadata,
            self.config.block_size,
            runner.graph_max_blocks(),
        )?;

        let output = match dispatch.action {
            DecodeGraphAction::Replay => self.replay_decode_graph(dispatch.execution, None)?,
            DecodeGraphAction::Capture => match self.capture_decode_graph(dispatch.execution, None) {
                Ok(output) => output,
                Err(e) => {
                    warn!(
                        graph_batch = dispatch.execution.graph_tokens,
                        "persistent decode graph capture failed, falling back: {e}"
                    );
                    return Ok(None);
                }
            },
            DecodeGraphAction::Raw => return Ok(None),
        };

        Ok(Some(output))
    }

    #[cfg(not(feature = "cuda"))]
    fn try_gpu_forward_persistent_decode(
        &mut self,
        _metadata: &[SequenceGroupMetadata],
        _greedy_only: bool,
    ) -> Result<Option<ForwardOutput>> {
        Ok(None)
    }

    /// Convenience constructor from EngineConfig and model path.
    pub fn from_engine_config(
        model_path: &str,
        config: &rvllm_config::EngineConfig,
    ) -> Result<Self> {
        let worker_config = worker_config_from_engine(model_path, config);
        Self::new(worker_config)
    }

    /// Load model weights from safetensors files on disk directly to GPU memory.
    pub fn load_weights(&mut self, model_path: &Path) -> Result<()> {
        info!(device_id = self.device_id, path = %model_path.display(), "loading weights to GPU");

        let (all_weights_f16, all_weight_shapes) =
            rvllm_model_loader::gpu_loader::load_weights_to_gpu_with_shapes(
                model_path,
                &self.stream,
            )
            .map_err(|e| LLMError::GpuError(format!("weight loading failed: {e}")))?;

        info!(
            "loaded {} weight tensors to GPU (f16)",
            all_weights_f16.len()
        );

        #[cfg(feature = "cuda")]
        {
            self.raw_weight_map = Some(all_weights_f16.clone());
            self.raw_weight_shapes = Some(all_weight_shapes.clone());
        }

        // Weights are stored only in raw_weight_map for GpuModelRunner.
        // No local GpuModelWeights copy needed (CPU forward path removed).

        // Init RoPE tables
        let rope_theta = self.config.rope_theta;
        let max_seq_len = self.config.max_model_len.min(8192);
        self.rope_table = Some(RopeTable::new(
            self.config.head_dim,
            max_seq_len,
            rope_theta,
        ));
        info!(
            head_dim = self.config.head_dim,
            max_seq_len, rope_theta, "RoPE tables initialized"
        );

        // Init KV cache (FP8 or f32)
        if self.use_fp8_kv {
            self.fp8_kv_cache = Some(Fp8KVCache::new(
                self.config.num_layers,
                self.config.num_kv_heads,
                self.config.head_dim,
            ));
            info!("FP8 quantized KV cache initialized");
        } else {
            self.kv_cache = Some(KVCache::new(
                self.config.num_layers,
                self.config.num_kv_heads,
                self.config.head_dim,
            ));
            info!("KV cache initialized");
        }

        info!(
            device_id = self.device_id,
            "model weights loaded to GPU successfully"
        );
        Ok(())
    }

    /// Initialise the model runner config.
    pub fn init_model(&mut self, model_path: &Path) -> Result<()> {
        info!(device_id = self.device_id, path = %model_path.display(), "GpuWorker init_model");
        let mr_config = self.config.model_runner_config();
        self.vocab_size = mr_config.vocab_size;
        self.runner_config = Some(mr_config);
        Ok(())
    }

    /// Allocate GPU and CPU KV cache blocks after profiling available memory.
    pub fn init_cache(&mut self, num_gpu_blocks: usize, num_cpu_blocks: usize) -> Result<()> {
        info!(
            device_id = self.device_id,
            num_gpu_blocks, num_cpu_blocks, "GpuWorker init_cache"
        );

        #[cfg(feature = "cuda")]
        {
            use rvllm_model_loader::gpu_weights::GpuModelWeights as LoaderWeights;

            let mut raw_map = self.raw_weight_map.take().ok_or_else(|| {
                LLMError::GpuError("raw weight map not available -- call load_weights first".into())
            })?;
            if self.config.architecture == "Qwen3_5ForConditionalGeneration" {
                self.install_qwen35_compat_weights_f16(&mut raw_map)?;
            }
            let raw_shapes = self.raw_weight_shapes.take().unwrap_or_default();
            let loader_weights = LoaderWeights::new(raw_map, raw_shapes);

            let block_size = self.config.block_size;
            let cache = rvllm_kv_cache::engine_cuda::CudaCacheEngine::new(
                self.config.num_layers,
                self.config.num_kv_heads,
                self.config.head_dim,
                block_size,
                num_gpu_blocks,
                num_cpu_blocks,
                self.context.clone(),
                self.stream.clone(),
            )
            .map_err(|e| LLMError::GpuError(format!("CudaCacheEngine init: {e}")))?;

            // Create FP8 cache engine alongside the standard f16 cache when enabled.
            // The f16 CudaCacheEngine is still needed by GpuModelRunner for the
            // forward pass. The FP8 engine shadows it for long-term KV storage,
            // halving VRAM for the KV cache at the cost of quantization noise.
            if self.use_fp8_kv {
                let fp8_cache = rvllm_kv_cache::CudaFP8CacheEngine::new(
                    self.config.num_layers,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    block_size,
                    num_gpu_blocks,
                    num_cpu_blocks,
                    self.context.clone(),
                    self.stream.clone(),
                )
                .map_err(|e| LLMError::GpuError(format!("CudaFP8CacheEngine init: {e}")))?;
                self.fp8_cache_engine = Some(fp8_cache);
                info!("FP8 GPU cache engine initialized alongside standard f16 cache");
            }

            let runner_blas = CublasHandle::new(self.stream.clone())
                .map_err(|e| LLMError::GpuError(format!("runner cublas: {e}")))?;

            // KernelLoader: try build output dir, fall back to empty (PTX loaded at runtime)
            let ptx_dir = Self::find_ptx_dir();
            let mut loader = rvllm_gpu::kernel_loader::KernelLoader::new(
                self.context.clone(),
                self.stream.clone(),
                &ptx_dir.unwrap_or_else(|| std::path::PathBuf::from("/nonexistent")),
            )
            .map_err(|e| LLMError::GpuError(format!("kernel loader: {e}")))?;

            // JIT compile fused CuTE kernels for this model's specific dimensions.
            // Uses nvcc + CUTLASS headers. Cached to ~/.cache/rvllm/fusion/.
            Self::jit_compile_fused_kernels(&mut loader, &self.config)?;

            // Hard validation: all required kernels must be loaded. No silent fallbacks.
            loader.validate_required_kernels();

            let mr_config = self.config.model_runner_config();
            let mut runner = rvllm_model_runner::gpu_runner::GpuModelRunner::new(
                loader_weights,
                cache,
                runner_blas,
                loader,
                mr_config,
                self.context.clone(),
                self.stream.clone(),
            )?;

            if let Err(e) = runner.fuse_weights() {
                warn!("weight fusion failed: {e} -- trying unfused fp16 runtime");
                if let Err(e2) = runner.prepare_fp16_runtime() {
                    warn!("fp16 runtime preparation failed: {e2} -- keeping unfused default path");
                }
            }

            // Pre-allocate cuBLAS workspace for CUDA graph capture.
            // This must happen before any graph capture attempt.
            if let Err(e) = runner.prepare_for_graph_capture() {
                warn!("cuBLAS graph workspace setup failed: {e} -- graph capture disabled");
                self.graph_runner.pool_mut().disable();
            }

            self.gpu_model_runner = Some(runner);
            info!(
                "GPU model runner initialized with {} GPU blocks (block_size={})",
                num_gpu_blocks, block_size
            );

            // Pre-capture CUDA graphs for common decode batch sizes to avoid
            // mid-generation capture stalls.
            if let Err(e) = self.precapture_decode_graphs() {
                warn!("pre-capture failed: {e} -- graphs will be captured lazily");
            }
        }

        Ok(())
    }

    /// Pre-capture CUDA graphs for common decode batch sizes at startup.
    /// Eliminates mid-generation graph capture stalls by populating the graph
    /// pool with pre-built graphs for standard bucket sizes.
    #[cfg(feature = "cuda")]
    fn precapture_decode_graphs(&mut self) -> Result<()> {
        use rvllm_model_runner::bridge::AttentionMetadata;

        const BUCKET_SIZES: &[usize] = &[
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
            152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
        ];
        // Max batch size matches the GraphRunnerConfig created in new() (256).
        let max_batch = 256usize;

        let runner = self.gpu_model_runner.as_ref().ok_or_else(|| {
            LLMError::GpuError("GPU model runner not initialized for precapture".into())
        })?;

        if !self.graph_runner.is_enabled() {
            info!("graph capture disabled, skipping pre-capture");
            return Ok(());
        }

        let cuda_stream = runner.cuda_stream().clone();
        let t0 = std::time::Instant::now();
        let mut captured = 0usize;
        let mut skipped = 0usize;

        for &n in BUCKET_SIZES {
            if n > max_batch {
                break;
            }
            let plan = runner.decode_execution_plan(n, false, true);

            // Skip if already captured (shouldn't happen at init, but be safe)
            if self.graph_runner.has_graph_for_exact(plan.graph_tokens) {
                continue;
            }

            if !plan.use_graphed_decode {
                debug!(n, "skipping pre-capture for non-graph decode path");
                skipped += 1;
                continue;
            }

            let token_ids = vec![0u32; n];
            let positions = vec![0u32; n];
            let attn_meta = AttentionMetadata {
                context_lens: vec![1; n],
                block_tables: vec![vec![0u32]; n],
                slot_mapping: vec![0; n],
                query_lens: vec![1; n],
                max_context_len: 1,
            };

            // Warmup forward (outside capture)
            let upload_result = if plan.use_batched_v2 {
                runner.upload_decode_metadata_v2(&token_ids, &positions, &attn_meta, plan.graph_tokens)
            } else if plan.graph_tokens == n {
                runner.upload_metadata(&token_ids, &positions, &attn_meta)
            } else {
                runner.upload_metadata_padded(&token_ids, &positions, &attn_meta, plan.graph_tokens)
            };
            if let Err(e) = upload_result {
                warn!(n, "precapture upload failed: {e}");
                skipped += 1;
                continue;
            }
            if let Err(e) = runner.forward_gpu_only(plan.graph_tokens, plan.graph_tokens, 1, false)
            {
                let msg = format!("{e}");
                if msg.contains("AUTOTUNED") || msg.contains("NOT_SUPPORTED") {
                    tracing::error!(n, "FATAL: {msg}");
                    return Err(e);
                }
                warn!(n, "precapture warmup forward failed: {e}");
                skipped += 1;
                continue;
            }

            // Sync before capture
            if let Err(e) = cuda_stream.synchronize() {
                warn!(n, "precapture sync failed: {e}");
                skipped += 1;
                continue;
            }

            // Re-upload for capture
            let reupload_result = if plan.use_batched_v2 {
                runner.upload_decode_metadata_v2(&token_ids, &positions, &attn_meta, plan.graph_tokens)
            } else if plan.graph_tokens == n {
                runner.upload_metadata(&token_ids, &positions, &attn_meta)
            } else {
                runner.upload_metadata_padded(&token_ids, &positions, &attn_meta, plan.graph_tokens)
            };
            if let Err(e) = reupload_result {
                warn!(n, "precapture re-upload failed: {e}");
                skipped += 1;
                continue;
            }

            // Begin capture
            if let Err(e) = self.graph_runner.pool_mut().begin_capture_on(&cuda_stream) {
                warn!(n, "precapture begin_capture failed: {e}");
                self.graph_runner.mark_captured(plan.graph_tokens);
                skipped += 1;
                continue;
            }

            // Forward inside capture
            match runner.forward_gpu_only(plan.graph_tokens, plan.graph_tokens, 1, false) {
                Ok(()) => match self
                    .graph_runner
                    .pool_mut()
                    .end_capture_on(&cuda_stream, plan.graph_tokens)
                {
                    Ok(graph) => {
                        self.graph_runner.pool_mut().insert(graph);
                        self.graph_runner.mark_captured(plan.graph_tokens);
                        captured += 1;
                    }
                    Err(e) => {
                        warn!(n, "precapture end_capture failed: {e}");
                        self.graph_runner.mark_captured(plan.graph_tokens);
                        skipped += 1;
                    }
                },
                Err(e) => {
                    warn!(n, "precapture forward failed: {e}");
                    let _ = self
                        .graph_runner
                        .pool_mut()
                        .end_capture_on(&cuda_stream, plan.graph_tokens);
                    self.graph_runner.mark_captured(plan.graph_tokens);
                    skipped += 1;
                }
            }
        }

        let elapsed = t0.elapsed();
        info!(
            captured,
            skipped,
            elapsed_ms = elapsed.as_millis(),
            "CUDA graph pre-capture complete"
        );
        Ok(())
    }

    /// JIT compile fused CuTE kernels for this model's dimensions.
    /// Checks disk cache first (~/.cache/rvllm/fusion/). On miss, instantiates
    /// CuTE templates with compile-time constants and invokes nvcc.
    #[cfg(feature = "cuda")]
    fn jit_compile_fused_kernels(
        loader: &mut rvllm_gpu::kernel_loader::KernelLoader,
        config: &WorkerConfig,
    ) -> Result<()> {
        use rvllm_fusion::cache::KernelCache;
        use rvllm_fusion::jit::JitCompiler;

        // Find CUTLASS headers for CuTE includes
        let cutlass_dirs: Vec<std::path::PathBuf> = [
            "./cutlass/include",
            "./cutlass/tools/util/include",
            "/root/rvllm/cutlass/include",
            "/root/rvllm/cutlass/tools/util/include",
            "/root/cutlass/include",
            "/root/cutlass/tools/util/include",
            "/usr/local/cutlass/include",
            "/opt/cutlass/include",
        ]
        .iter()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
        .collect();

        let jit = if cutlass_dirs.is_empty() {
            // No CUTLASS headers -- try basic JIT without CuTE
            match JitCompiler::new() {
                Ok(j) => j,
                Err(e) => {
                    warn!("JIT compiler not available ({e}), skipping fused kernel compilation");
                    return Ok(());
                }
            }
        } else {
            match JitCompiler::new() {
                Ok(mut j) => {
                    // Re-create with include dirs
                    let arch = j.arch().to_string();
                    JitCompiler::with_config(
                        std::path::PathBuf::from(j.nvcc_path()),
                        arch,
                        cutlass_dirs,
                    )
                }
                Err(e) => {
                    warn!("JIT compiler not available ({e}), skipping fused kernel compilation");
                    return Ok(());
                }
            }
        };

        let cache_dir = Self::dirs_or_home().join("fusion");
        let cache = KernelCache::new(cache_dir);

        let hidden = config.hidden_size;
        let qkv_dim = config.num_attention_heads * config.head_dim
            + 2 * config.num_kv_heads * config.head_dim;
        let gate_up_dim = config.intermediate_size * 2;
        let intermediate = config.intermediate_size;
        let eps = format!("{:e}", config.rms_norm_eps);

        // CuTE templates embedded at compile time
        let templates: &[(&str, &str, &str, usize, usize)] = &[
            (
                "fused_cute_norm_qkv_gemv",
                "norm_gemv",
                include_str!("../../rvllm-fusion/templates/cute_norm_gemv.cu.template"),
                hidden,
                qkv_dim,
            ),
            (
                "fused_cute_add_norm_qkv_gemv",
                "add_norm_gemv",
                include_str!("../../rvllm-fusion/templates/cute_add_norm_gemv.cu.template"),
                hidden,
                qkv_dim,
            ),
            (
                "fused_cute_add_norm_gateup_gemv",
                "add_norm_gemv",
                include_str!("../../rvllm-fusion/templates/cute_add_norm_gemv.cu.template"),
                hidden,
                gate_up_dim,
            ),
            (
                "fused_cute_silu_down_gemv",
                "silu_gemv",
                include_str!("../../rvllm-fusion/templates/cute_silu_mul_gemv.cu.template"),
                hidden,
                intermediate,
            ),
        ];

        let t0 = std::time::Instant::now();
        let mut compiled = 0;
        let mut cached = 0;

        for &(kernel_name, _tag, template, dim_a, dim_b) in templates {
            // Build cache key from kernel name + dimensions + arch
            let cache_key = KernelCache::key_for(kernel_name, &[dim_a, dim_b], jit.arch());

            // Check cache
            if let Some(ptx_bytes) = cache.get(&cache_key) {
                if loader.load_ptx(kernel_name, &ptx_bytes).is_ok() {
                    cached += 1;
                    continue;
                }
            }

            // Instantiate template
            let source = template
                .replace("{{KERNEL_NAME}}", kernel_name)
                .replace("{{HIDDEN_SIZE}}", &dim_a.to_string())
                .replace("{{OUT_DIM}}", &dim_b.to_string())
                .replace("{{INTERMEDIATE_SIZE}}", &dim_b.to_string())
                .replace("{{THREADS}}", "256")
                .replace("{{EPS}}", &eps);

            // Compile
            match jit.compile_to_ptx(&source, kernel_name) {
                Ok(ptx_bytes) => {
                    // Cache to disk
                    if let Err(e) = cache.put(&cache_key, &ptx_bytes) {
                        warn!("failed to cache fused kernel {kernel_name}: {e}");
                    }
                    // Load into kernel loader
                    if let Err(e) = loader.load_ptx(kernel_name, &ptx_bytes) {
                        warn!("failed to load fused kernel {kernel_name}: {e}");
                    } else {
                        compiled += 1;
                    }
                }
                Err(e) => {
                    warn!("JIT compile failed for {kernel_name}: {e}");
                }
            }
        }

        let elapsed = t0.elapsed();
        if compiled > 0 || cached > 0 {
            info!(
                compiled,
                cached,
                elapsed_ms = elapsed.as_millis(),
                "fused kernels ready"
            );
        }
        Ok(())
    }

    /// Get cache directory (HOME/.cache/rvllm or /tmp/rvllm)
    #[cfg(feature = "cuda")]
    fn dirs_or_home() -> std::path::PathBuf {
        std::env::var("HOME")
            .map(|h| std::path::PathBuf::from(h).join(".cache").join("rvllm"))
            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/rvllm"))
    }

    #[cfg(feature = "cuda")]
    fn alias_weight<T: Clone>(weights: &mut HashMap<String, T>, dst: &str, src: &str) {
        if !weights.contains_key(dst) {
            if let Some(v) = weights.get(src).cloned() {
                weights.insert(dst.to_string(), v);
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn copy_range_f16(
        &self,
        src: &CudaSlice<half::f16>,
        offset: usize,
        len: usize,
    ) -> Result<CudaSlice<half::f16>> {
        let host = self
            .stream
            .clone_dtoh(src)
            .map_err(|e| LLMError::GpuError(format!("compat dtoh f16: {e}")))?;
        let slice = host[offset..offset + len].to_vec();
        self.stream
            .clone_htod(&slice)
            .map_err(|e| LLMError::GpuError(format!("compat htod f16: {e}")))
    }

    #[cfg(feature = "cuda")]
    fn install_qwen35_compat_weights_f16(
        &self,
        weights: &mut HashMap<String, CudaSlice<half::f16>>,
    ) -> Result<()> {
        let hidden = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        Self::alias_weight(
            weights,
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        );
        Self::alias_weight(
            weights,
            "model.norm.weight",
            "model.language_model.norm.weight",
        );
        for i in 0..self.config.num_layers {
            let old = format!("model.layers.{i}");
            let base = format!("model.language_model.layers.{i}");
            Self::alias_weight(
                weights,
                &format!("{old}.input_layernorm.weight"),
                &format!("{base}.input_layernorm.weight"),
            );
            Self::alias_weight(
                weights,
                &format!("{old}.post_attention_layernorm.weight"),
                &format!("{base}.post_attention_layernorm.weight"),
            );
            Self::alias_weight(
                weights,
                &format!("{old}.mlp.gate_proj.weight"),
                &format!("{base}.mlp.gate_proj.weight"),
            );
            Self::alias_weight(
                weights,
                &format!("{old}.mlp.up_proj.weight"),
                &format!("{base}.mlp.up_proj.weight"),
            );
            Self::alias_weight(
                weights,
                &format!("{old}.mlp.down_proj.weight"),
                &format!("{base}.mlp.down_proj.weight"),
            );
            let full_q = format!("{base}.self_attn.q_proj.weight");
            let full_k = format!("{base}.self_attn.k_proj.weight");
            let full_v = format!("{base}.self_attn.v_proj.weight");
            let full_o = format!("{base}.self_attn.o_proj.weight");
            let lin_qkv = format!("{base}.linear_attn.in_proj_qkv.weight");
            let lin_o = format!("{base}.linear_attn.out_proj.weight");
            if let Some(src_q) = weights.get(&full_q).cloned() {
                weights.insert(
                    format!("{old}.self_attn.q_proj.weight"),
                    self.copy_range_f16(&src_q, 0, q_dim * hidden)?,
                );
                Self::alias_weight(weights, &format!("{old}.self_attn.k_proj.weight"), &full_k);
                Self::alias_weight(weights, &format!("{old}.self_attn.v_proj.weight"), &full_v);
                Self::alias_weight(weights, &format!("{old}.self_attn.o_proj.weight"), &full_o);
            } else if let Some(src_qkv) = weights.get(&lin_qkv).cloned() {
                weights.insert(
                    format!("{old}.self_attn.q_proj.weight"),
                    self.copy_range_f16(&src_qkv, 0, q_dim * hidden)?,
                );
                weights.insert(
                    format!("{old}.self_attn.k_proj.weight"),
                    self.copy_range_f16(&src_qkv, q_dim * hidden, kv_dim * hidden)?,
                );
                weights.insert(
                    format!("{old}.self_attn.v_proj.weight"),
                    self.copy_range_f16(&src_qkv, (q_dim + kv_dim) * hidden, kv_dim * hidden)?,
                );
                Self::alias_weight(weights, &format!("{old}.self_attn.o_proj.weight"), &lin_o);
            } else {
                return Err(LLMError::ModelError(format!(
                    "Qwen3.5 compat f16: missing attention weights for layer {i}"
                )));
            }
        }
        Ok(())
    }

    /// Search for compiled PTX directory from build output.
    #[cfg(feature = "cuda")]
    fn find_ptx_dir() -> Option<std::path::PathBuf> {
        // Check env var first
        if let Ok(dir) = std::env::var("RVLLM_PTX_DIR") {
            let p = std::path::PathBuf::from(&dir);
            if p.exists() {
                info!(dir, "found PTX dir from RVLLM_PTX_DIR");
                return Some(p);
            }
        }
        // Check common locations
        for dir in &[
            "./target/ptx",
            "/root/rvllm/target/ptx",
            "./target/release/build",
            "./target/debug/build",
        ] {
            let p = std::path::PathBuf::from(dir);
            if p.join("embedding_gather_f16.ptx").exists() {
                info!(dir, "found PTX dir");
                return Some(p);
            }
            // Also check build output subdirs
            if let Ok(entries) = std::fs::read_dir(&p) {
                for entry in entries.flatten() {
                    let ptx_path = entry.path().join("out/ptx");
                    if ptx_path.join("embedding_gather_f16.ptx").exists() {
                        return Some(ptx_path);
                    }
                }
            }
        }
        None
    }

    /// Profile available GPU memory and return (num_gpu_blocks, num_cpu_blocks).
    pub fn profile_num_available_blocks(
        &self,
        gpu_memory_utilization: f32,
        gpu_memory_reserve_bytes: usize,
    ) -> Result<(usize, usize)> {
        let (free, _total) = self
            .context
            .mem_get_info()
            .map_err(|e| LLMError::GpuError(format!("mem_get_info failed: {e}")))?;
        let free_after_reserve = free.saturating_sub(gpu_memory_reserve_bytes);
        let available = (free_after_reserve as f32 * gpu_memory_utilization) as usize;

        let cache_cfg = self.config.cache_config();
        let total_block_bytes = cache_cfg.total_block_bytes();

        // When FP8 KV is enabled, we allocate both f16 (for GpuModelRunner) and
        // FP8 (shadow) caches. FP8 data is u8 + f32 scales per head, roughly
        // ~55% of f16 size. Account for both when computing block budget.
        let effective_block_bytes = if self.use_fp8_kv {
            // f16 cache: total_block_bytes
            // FP8 cache: u8 data (half of f16 data) + f32 scales overhead
            let fp8_data_bytes = total_block_bytes / 2; // u8 vs f16
            let scales_per_block = self.config.block_size * self.config.num_kv_heads;
            let fp8_scale_bytes = scales_per_block * 4 * 2 * self.config.num_layers; // f32 * 2 (K+V) * layers
            total_block_bytes + fp8_data_bytes + fp8_scale_bytes / self.config.num_layers
        } else {
            total_block_bytes
        };

        let num_gpu_blocks = if effective_block_bytes > 0 {
            available / effective_block_bytes
        } else {
            0
        };
        let num_cpu_blocks = 128;

        info!(
            free,
            gpu_memory_reserve_bytes,
            free_after_reserve,
            available,
            num_gpu_blocks,
            num_cpu_blocks,
            fp8 = self.use_fp8_kv,
            "profiled available blocks"
        );
        Ok((num_gpu_blocks, num_cpu_blocks))
    }

    /// Clear the KV cache (call between requests).
    pub fn clear_kv_cache(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.clear();
        }
        if let Some(ref mut cache) = self.fp8_kv_cache {
            cache.clear();
        }
    }

    /// Whether FP8 KV cache quantization is active.
    pub fn use_fp8_kv(&self) -> bool {
        self.use_fp8_kv
    }

    /// Access the GPU-resident FP8 cache engine (if FP8 KV is enabled).
    #[cfg(feature = "cuda")]
    pub fn fp8_cache_engine(&self) -> Option<&rvllm_kv_cache::CudaFP8CacheEngine> {
        self.fp8_cache_engine.as_ref()
    }

    /// Mutable access to the GPU-resident FP8 cache engine.
    #[cfg(feature = "cuda")]
    pub fn fp8_cache_engine_mut(&mut self) -> Option<&mut rvllm_kv_cache::CudaFP8CacheEngine> {
        self.fp8_cache_engine.as_mut()
    }

    /// FP8 pre-forward: dequantize FP8 cache blocks to f16 cache for attention reads.
    ///
    /// Collects unique block indices from the attention metadata's block_tables
    /// and restores them from the FP8 shadow cache into the runner's f16 cache.
    #[cfg(feature = "cuda")]
    fn fp8_pre_forward_dequantize(&mut self, model_input: &ModelInput) -> Result<()> {
        let fp8_engine = match self.fp8_cache_engine.as_ref() {
            Some(e) => e,
            None => return Ok(()),
        };
        let runner = match self.gpu_model_runner.as_mut() {
            Some(r) => r,
            None => return Ok(()),
        };

        // Collect unique block indices from block_tables
        let mut block_set = std::collections::HashSet::new();
        for table in &model_input.attention_metadata.block_tables {
            for &block_id in table {
                block_set.insert(block_id as usize);
            }
        }
        let block_indices: Vec<usize> = block_set.into_iter().collect();

        if block_indices.is_empty() {
            return Ok(());
        }

        let num_layers = fp8_engine.num_layers();
        let cache = runner.cache_mut();
        for layer in 0..num_layers {
            let (key_cache, value_cache) = &mut cache.gpu_cache_mut()[layer];
            fp8_engine.dequantize_blocks_to_f16_cache(
                layer,
                &block_indices,
                key_cache,
                value_cache,
            )?;
        }

        trace!(
            blocks = block_indices.len(),
            "FP8 pre-forward dequantize complete"
        );
        Ok(())
    }

    /// FP8 post-forward: quantize newly written f16 cache blocks to FP8 shadow.
    ///
    /// Derives the written block indices from slot_mapping and stores them
    /// compressed in the FP8 engine.
    #[cfg(feature = "cuda")]
    fn fp8_post_forward_quantize(&mut self, model_input: &ModelInput) -> Result<()> {
        let fp8_engine = match self.fp8_cache_engine.as_mut() {
            Some(e) => e,
            None => return Ok(()),
        };
        let runner = match self.gpu_model_runner.as_ref() {
            Some(r) => r,
            None => return Ok(()),
        };

        let block_size = self.config.block_size;
        let mut block_set = std::collections::HashSet::new();
        for &slot in &model_input.attention_metadata.slot_mapping {
            let block_idx = slot as usize / block_size;
            block_set.insert(block_idx);
        }
        // Also include blocks from block_tables (they were read and may have
        // been updated by reshape_and_cache during the forward pass).
        for table in &model_input.attention_metadata.block_tables {
            for &block_id in table {
                block_set.insert(block_id as usize);
            }
        }
        let block_indices: Vec<usize> = block_set.into_iter().collect();

        if block_indices.is_empty() {
            return Ok(());
        }

        let num_layers = fp8_engine.num_layers();
        let cache = runner.cache();
        for layer in 0..num_layers {
            let (key_cache, value_cache) = &cache.gpu_cache()[layer];
            fp8_engine.quantize_f16_to_fp8(layer, &block_indices, key_cache, value_cache)?;
        }

        trace!(
            blocks = block_indices.len(),
            "FP8 post-forward quantize complete"
        );
        Ok(())
    }

    /// Vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Check if all requests in the batch can use greedy (temperature=0) GPU argmax.
    /// Returns false if any request uses temperature>0, guided decoding, or repetition penalty.
    fn all_greedy(metadata: &[SequenceGroupMetadata]) -> bool {
        metadata.iter().all(|g| {
            let p = &g.sampling_params;
            p.temperature == 0.0
                && matches!(p.response_format, ResponseFormat::Text)
                && p.repetition_penalty == 1.0
                && p.frequency_penalty == 0.0
                && p.presence_penalty == 0.0
        })
    }

    /// Run a forward pass returning raw logits (no sampling).
    /// Used by speculative decoding to get probability distributions for
    /// verification against draft model outputs.
    pub fn forward_logits(&mut self, metadata: &[SequenceGroupMetadata]) -> Result<Vec<f32>> {
        if metadata.is_empty() {
            return Ok(Vec::new());
        }
        let model_input = self.prepare_model_input(metadata)?;
        self.gpu_forward(&model_input)
    }

    /// Launch GPU work and return immediately. GPU computes asynchronously.
    /// Call `execute_collect` to sync and get results.
    /// Returns None if nothing to launch (empty metadata or non-graph path).
    pub fn execute_launch(&mut self, metadata: &[SequenceGroupMetadata]) -> Result<Option<usize>> {
        if metadata.is_empty() {
            return Ok(None);
        }
        let greedy_only = Self::all_greedy(metadata);
        if let Some(fwd_output) = self.try_gpu_forward_persistent_decode(metadata, greedy_only)? {
            match fwd_output {
                ForwardOutput::TokenIdsPending { actual_batch } => return Ok(Some(actual_batch)),
                _ => {
                    self.pending_sync_output = Some((fwd_output, metadata.to_vec()));
                    return Ok(None);
                }
            }
        }
        let model_input = self.prepare_model_input(metadata)?;
        let fwd_output = self.gpu_forward_ex(&model_input, greedy_only)?;
        match fwd_output {
            ForwardOutput::TokenIdsPending { actual_batch } => Ok(Some(actual_batch)),
            _ => {
                // Non-async path (prefill, capture, non-graph). Store output for collect.
                self.pending_sync_output = Some((fwd_output, metadata.to_vec()));
                Ok(None)
            }
        }
    }

    /// Collect results after execute_launch. Syncs GPU if async DtoH pending.
    pub fn execute_collect(
        &mut self,
        actual_batch: Option<usize>,
        metadata: &[SequenceGroupMetadata],
    ) -> Result<GpuWorkerOutput> {
        if let Some(batch) = actual_batch {
            // Async path: sync and read from pinned buffer
            let outputs = self.sample_pending_tokens_from_pinned(batch, metadata)?;
            return Ok(GpuWorkerOutput { outputs });
        }
        // Sync path: output was stored in pending_sync_output
        if let Some((fwd_output, _stored_meta)) = self.pending_sync_output.take() {
            let outputs = match fwd_output {
                ForwardOutput::TokenIds(ref token_ids) => {
                    self.sample_tokens_from_gpu_argmax(token_ids, metadata)?
                }
                ForwardOutput::Logits(ref logits) => self.sample_tokens(logits, metadata)?,
                ForwardOutput::TokenIdsPending { .. } => unreachable!(),
            };
            return Ok(GpuWorkerOutput { outputs });
        }
        Ok(GpuWorkerOutput {
            outputs: Vec::new(),
        })
    }

    /// Execute with overlap: runs `during_gpu` closure while GPU computes.
    /// The closure gets ~4ms of CPU time during the async graph replay path.
    /// For sync paths (prefill, capture), the closure runs after GPU finishes.
    pub fn execute_with_overlap<F: FnOnce()>(
        &mut self,
        metadata: &[SequenceGroupMetadata],
        during_gpu: F,
    ) -> Result<GpuWorkerOutput> {
        if metadata.is_empty() {
            during_gpu();
            return Ok(GpuWorkerOutput {
                outputs: Vec::new(),
            });
        }

        let greedy_only = Self::all_greedy(metadata);
        let fwd_output =
            if let Some(fwd_output) = self.try_gpu_forward_persistent_decode(metadata, greedy_only)?
            {
                fwd_output
            } else {
                let model_input = self.prepare_model_input(metadata)?;
                self.gpu_forward_ex(&model_input, greedy_only)?
            };

        let outputs = match fwd_output {
            ForwardOutput::TokenIdsPending { actual_batch } => {
                // GPU is computing async. Run the overlap closure NOW.
                during_gpu();
                // THEN sync and read results.
                self.sample_pending_tokens_from_pinned(actual_batch, metadata)?
            }
            ForwardOutput::TokenIds(ref token_ids) => {
                during_gpu();
                self.sample_tokens_from_gpu_argmax(token_ids, metadata)?
            }
            ForwardOutput::Logits(ref logits) => {
                during_gpu();
                self.sample_tokens(logits, metadata)?
            }
        };
        Ok(GpuWorkerOutput { outputs })
    }

    /// Execute one inference step with real GPU matmuls.
    pub fn execute(&mut self, metadata: &[SequenceGroupMetadata]) -> Result<GpuWorkerOutput> {
        if metadata.is_empty() {
            return Ok(GpuWorkerOutput {
                outputs: Vec::new(),
            });
        }

        let t_start = std::time::Instant::now();

        let greedy_only = Self::all_greedy(metadata);
        let fwd_output =
            if let Some(fwd_output) = self.try_gpu_forward_persistent_decode(metadata, greedy_only)?
            {
                fwd_output
            } else {
                let model_input = self.prepare_model_input(metadata)?;
                let fwd_output = self.gpu_forward_ex(&model_input, greedy_only)?;
                let t_input = t_start.elapsed();
                let t_forward = t_start.elapsed();
                let outputs = match fwd_output {
                    ForwardOutput::TokenIds(ref token_ids) => {
                        self.sample_tokens_from_gpu_argmax(token_ids, metadata)?
                    }
                    ForwardOutput::TokenIdsPending { actual_batch } => {
                        self.sample_pending_tokens_from_pinned(actual_batch, metadata)?
                    }
                    ForwardOutput::Logits(ref logits) => self.sample_tokens(logits, metadata)?,
                };
                let t_sample = t_start.elapsed();
                if self.forward_count % 64 == 0 && self.forward_count > 0 {
                    let input_us = t_input.as_micros();
                    let forward_us = (t_forward - t_input).as_micros();
                    let sample_us = (t_sample - t_forward).as_micros();
                    let total_us = t_sample.as_micros();
                    info!(
                        input_us,
                        forward_us,
                        sample_us,
                        total_us,
                        tokens = metadata.iter().map(|g| g.seq_data.len()).sum::<usize>(),
                        greedy = greedy_only,
                        "TIMING execute"
                    );
                }
                return Ok(GpuWorkerOutput { outputs });
            };
        let t_input = t_start.elapsed();
        let t_forward = t_start.elapsed();

        let outputs = match fwd_output {
            ForwardOutput::TokenIds(ref token_ids) => {
                self.sample_tokens_from_gpu_argmax(token_ids, metadata)?
            }
            ForwardOutput::TokenIdsPending { actual_batch } => {
                // Async DtoH was enqueued. Sync now and read from pinned buffer.
                self.sample_pending_tokens_from_pinned(actual_batch, metadata)?
            }
            ForwardOutput::Logits(ref logits) => self.sample_tokens(logits, metadata)?,
        };
        let t_sample = t_start.elapsed();

        // Periodic timing report (every 64 steps)
        self.forward_count += 0; // already incremented in gpu_forward_ex
        if self.forward_count % 64 == 0 && self.forward_count > 0 {
            let input_us = t_input.as_micros();
            let forward_us = (t_forward - t_input).as_micros();
            let sample_us = (t_sample - t_forward).as_micros();
            let total_us = t_sample.as_micros();
            info!(
                input_us,
                forward_us,
                sample_us,
                total_us,
                tokens = metadata.iter().map(|g| g.seq_data.len()).sum::<usize>(),
                greedy = greedy_only,
                "TIMING execute"
            );
        }

        Ok(GpuWorkerOutput { outputs })
    }

    /// Launch GPU forward pass and enqueue async DtoH, returning immediately.
    ///
    /// Returns `Some(GpuWorkerOutput)` if the forward path is synchronous
    /// (non-graph path, capture path), or `None` if async DtoH is pending
    /// and the caller should do CPU work before calling `collect_async_output`.
    pub fn execute_async(
        &mut self,
        metadata: &[SequenceGroupMetadata],
    ) -> Result<Option<GpuWorkerOutput>> {
        if metadata.is_empty() {
            return Ok(Some(GpuWorkerOutput {
                outputs: Vec::new(),
            }));
        }

        let greedy_only = Self::all_greedy(metadata);
        let fwd_output =
            if let Some(fwd_output) = self.try_gpu_forward_persistent_decode(metadata, greedy_only)?
            {
                fwd_output
            } else {
                let model_input = self.prepare_model_input(metadata)?;
                self.gpu_forward_ex(&model_input, greedy_only)?
            };

        match fwd_output {
            ForwardOutput::TokenIdsPending { .. } => {
                // Async DtoH in flight -- caller should do CPU work then call collect
                Ok(None)
            }
            ForwardOutput::TokenIds(ref token_ids) => {
                let outputs = self.sample_tokens_from_gpu_argmax(token_ids, metadata)?;
                Ok(Some(GpuWorkerOutput { outputs }))
            }
            ForwardOutput::Logits(ref logits) => {
                let outputs = self.sample_tokens(logits, metadata)?;
                Ok(Some(GpuWorkerOutput { outputs }))
            }
        }
    }

    /// Synchronize the GPU stream and read the pending async DtoH output.
    ///
    /// Call after `execute_async` returned `None` (indicating async DtoH
    /// is in flight). Does CPU work between the async launch and this call
    /// to overlap with GPU compute + DtoH transfer.
    pub fn collect_async_output(
        &mut self,
        metadata: &[SequenceGroupMetadata],
    ) -> Result<GpuWorkerOutput> {
        // Determine actual batch size from metadata
        let actual_batch: usize = metadata.iter().map(|g| g.seq_data.len()).sum();
        let outputs = self.sample_pending_tokens_from_pinned(actual_batch, metadata)?;
        Ok(GpuWorkerOutput { outputs })
    }

    /// Execute one step with explicit cache operations.
    pub fn execute_with_cache_ops(
        &mut self,
        metadata: &[SequenceGroupMetadata],
        _blocks_to_swap_in: &[(BlockId, BlockId)],
        _blocks_to_swap_out: &[(BlockId, BlockId)],
        _blocks_to_copy: &[(BlockId, BlockId)],
    ) -> Result<GpuWorkerOutput> {
        self.execute(metadata)
    }

    /// Run the raw model forward pass (no graph logic).
    fn raw_gpu_forward(&self, model_input: &ModelInput) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            let runner = self.gpu_model_runner.as_ref().ok_or_else(|| {
                LLMError::GpuError(
                    "GPU model runner not initialized -- build with --features cuda".into(),
                )
            })?;

            return runner.forward(
                &model_input.token_ids,
                &model_input.position_ids,
                &model_input.attention_metadata,
                model_input.is_prefill,
            );
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(LLMError::GpuError(
                "GPU forward pass requires --features cuda. CPU fallback is disabled.".into(),
            ));
        }
    }

    /// Extended raw forward: supports greedy_only flag to skip logits DtoH.
    fn raw_gpu_forward_ex(
        &self,
        model_input: &ModelInput,
        greedy_only: bool,
    ) -> Result<ForwardOutput> {
        #[cfg(feature = "cuda")]
        {
            let runner = self.gpu_model_runner.as_ref().ok_or_else(|| {
                LLMError::GpuError(
                    "GPU model runner not initialized -- build with --features cuda".into(),
                )
            })?;

            return runner.forward_ex(
                &model_input.token_ids,
                &model_input.position_ids,
                &model_input.attention_metadata,
                model_input.is_prefill,
                greedy_only,
            );
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (model_input, greedy_only);
            return Err(LLMError::GpuError(
                "GPU forward pass requires --features cuda. CPU fallback is disabled.".into(),
            ));
        }
    }

    /// Warmup threshold: skip graph capture for the first N forward calls.
    const GRAPH_WARMUP_CALLS: usize = 3;

    /// Real GPU forward pass: tries CUDA graph replay for decode steps,
    /// falls back to normal forward, and captures graphs after warmup.
    fn gpu_forward(&mut self, model_input: &ModelInput) -> Result<Vec<f32>> {
        match self.gpu_forward_ex(model_input, false)? {
            ForwardOutput::Logits(logits) => Ok(logits),
            ForwardOutput::TokenIds(_) | ForwardOutput::TokenIdsPending { .. } => {
                unreachable!("greedy_only=false must return Logits")
            }
        }
    }

    /// Extended GPU forward with greedy fast path and CUDA graph capture/replay.
    ///
    /// For decode steps (is_prefill=false, all query_lens=1) with greedy sampling:
    /// 1. Upload metadata into persistent GPU buffers (stable pointers).
    /// 2. If a graph exists for this padded batch size: replay it.
    /// 3. If past warmup and no graph: capture one via forward_graph_body.
    /// 4. Otherwise: run forward_graph_body without capture.
    ///
    /// Metadata uploads happen OUTSIDE graph capture/replay so the memcpy_htod
    /// updates the persistent buffer contents in-place. The graph's kernels then
    /// read fresh data from the same GPU pointers that were baked in at capture.

    fn gpu_forward_ex(
        &mut self,
        model_input: &ModelInput,
        greedy_only: bool,
    ) -> Result<ForwardOutput> {
        self.forward_count += 1;

        // FP8 KV pre-forward: dequantize FP8 blocks back to f16 cache so
        // the attention kernel reads correct data.
        #[cfg(feature = "cuda")]
        if self.use_fp8_kv {
            self.fp8_pre_forward_dequantize(model_input)?;
        }

        // Use graphs for any pure decode step (all query_lens == 1).
        // Sampling params don't matter -- graphs capture the forward pass
        // (GEMMs, attention, norms), not the sampling/logit processing.
        let is_decode = !model_input.is_prefill
            && model_input
                .attention_metadata
                .query_lens
                .iter()
                .all(|&q| q == 1);
        let batch = model_input.num_tokens();
        let result = if !is_decode || !self.graph_runner.is_enabled() {
            self.raw_gpu_forward_ex(model_input, greedy_only)
        } else {
            #[cfg(feature = "cuda")]
            {
                let dispatch =
                    self.decode_graph_dispatch(batch, model_input.is_prefill, is_decode)?;

                match dispatch.action {
                    DecodeGraphAction::Raw => self.raw_gpu_forward_ex(model_input, greedy_only),
                    DecodeGraphAction::Replay => {
                        self.replay_decode_graph(dispatch.execution, Some(model_input))
                    }
                    DecodeGraphAction::Capture => match self
                        .capture_decode_graph(dispatch.execution, Some(model_input))
                    {
                        Ok(output) => Ok(output),
                        Err(e) => {
                            warn!(graph_batch = dispatch.execution.graph_tokens, "graph capture failed, raw forward: {e}");
                            self.raw_gpu_forward_ex(model_input, greedy_only)
                        }
                    },
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                self.raw_gpu_forward_ex(model_input, greedy_only)
            }
        };

        // FP8 KV post-forward: quantize newly written f16 cache blocks to FP8.
        // Errors here are non-fatal (the f16 cache is still correct for this step).
        #[cfg(feature = "cuda")]
        if self.use_fp8_kv {
            if result.is_ok() {
                if let Err(e) = self.fp8_post_forward_quantize(model_input) {
                    warn!("FP8 post-forward quantize failed (non-fatal): {e}");
                }
            }
        }

        result
    }

    #[cfg(feature = "cuda")]
    fn upload_decode_graph_metadata(
        &mut self,
        plan: DecodeExecutionPlan,
        model_input: Option<&ModelInput>,
    ) -> Result<u32> {
        let padded_batch = plan.graph_tokens;
        let runner = self
            .gpu_model_runner
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;

        if plan.use_batched_v2 {
            let scratch = &self.decode_input_scratch;
            runner.upload_decode_metadata_v2_flat(
                &scratch.token_ids,
                &scratch.position_ids,
                &scratch.query_lens,
                &scratch.context_lens,
                &scratch.slot_mapping,
                &scratch.block_tables_flat,
                padded_batch,
            )?;
            return Ok(scratch.max_context_len());
        }

        let model_input = model_input.ok_or_else(|| {
            LLMError::GpuError("generic decode graph path requires model input".into())
        })?;
        if plan.actual_tokens == padded_batch {
            runner.upload_metadata(
                &model_input.token_ids,
                &model_input.position_ids,
                &model_input.attention_metadata,
            )?;
        } else {
            runner.upload_metadata_padded(
                &model_input.token_ids,
                &model_input.position_ids,
                &model_input.attention_metadata,
                padded_batch,
            )?;
        }
        Ok(model_input.attention_metadata.max_context_len)
    }

    #[cfg(feature = "cuda")]
    fn replay_decode_graph(
        &mut self,
        plan: DecodeExecutionPlan,
        model_input: Option<&ModelInput>,
    ) -> Result<ForwardOutput> {
        let actual_batch = plan.actual_tokens;
        let padded_batch = plan.graph_tokens;
        self.ensure_pinned_output_capacity(actual_batch)?;
        self.upload_decode_graph_metadata(plan, model_input)?;
        let runner = self
            .gpu_model_runner
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;

        let graph = self
            .graph_runner
            .pool()
            .get_exact(padded_batch)
            .ok_or_else(|| {
                LLMError::GpuError(format!("no graph for padded batch {padded_batch}"))
            })?;
        // Replay on the runner's stream. The HtoD metadata upload (above) was
        // also enqueued on this stream, so CUDA ordering guarantees the kernels
        // see the updated buffers.
        graph.replay(&self.runner_stream)?;

        let pinned = self
            .pinned_output
            .as_mut()
            .ok_or_else(|| LLMError::GpuError("pinned output buffer not allocated".into()))?;
        let dst = pinned
            .as_mut_slice()
            .map_err(|e| LLMError::GpuError(format!("pinned buf write: {e}")))?;
        runner.read_graph_output_async(actual_batch, dst)?;
        Ok(ForwardOutput::TokenIdsPending { actual_batch })
    }

    #[cfg(feature = "cuda")]
    fn capture_decode_graph(
        &mut self,
        plan: DecodeExecutionPlan,
        model_input: Option<&ModelInput>,
    ) -> Result<ForwardOutput> {
        let actual_batch = plan.actual_tokens;
        let padded_batch = plan.graph_tokens;
        self.ensure_pinned_output_capacity(actual_batch)?;

        let runner = self
            .gpu_model_runner
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;

        if !plan.use_graphed_decode {
            return Err(LLMError::GpuError(format!(
                "graph capture unsupported for batch {padded_batch}"
            )));
        }

        info!(
            padded_batch,
            actual_batch, "capturing CUDA graph for padded batch size"
        );

        let max_context_len = self.upload_decode_graph_metadata(plan, model_input)?;
        runner.forward_gpu_only(padded_batch, padded_batch, max_context_len, false)?;

        let cuda_stream = runner.cuda_stream().clone();
        cuda_stream
            .synchronize()
            .map_err(|e| LLMError::GpuError(format!("pre-capture sync: {e}")))?;

        self.upload_decode_graph_metadata(plan, model_input)?;

        let capture_result = self.graph_runner.pool_mut().begin_capture_on(&cuda_stream);
        if let Err(e) = capture_result {
            warn!(padded_batch, "graph capture begin failed: {e}");
            self.graph_runner.mark_captured(padded_batch);
            return Err(LLMError::GpuError(format!("graph capture failed: {e}")));
        }

        let fwd_result =
            runner.forward_gpu_only(padded_batch, padded_batch, max_context_len, false);

        match fwd_result {
            Ok(()) => {
                let graph = self
                    .graph_runner
                    .pool_mut()
                    .end_capture_on(&cuda_stream, padded_batch)?;
                self.graph_runner.pool_mut().insert(graph);
                self.graph_runner.mark_captured(padded_batch);
                info!(padded_batch, "CUDA graph captured for padded batch");
                let pinned = self.pinned_output.as_mut().ok_or_else(|| {
                    LLMError::GpuError("pinned output buffer not allocated".into())
                })?;
                let dst = pinned
                    .as_mut_slice()
                    .map_err(|e| LLMError::GpuError(format!("pinned buf write: {e}")))?;
                runner.read_graph_output_async(actual_batch, dst)?;
                Ok(ForwardOutput::TokenIdsPending { actual_batch })
            }
            Err(e) => {
                warn!(padded_batch, "graph capture forward failed: {e}");
                let _ = self
                    .graph_runner
                    .pool_mut()
                    .end_capture_on(&cuda_stream, padded_batch);
                self.graph_runner.mark_captured(padded_batch);
                Err(e)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ensure_pinned_output_capacity(&mut self, need: usize) -> Result<()> {
        let have = self.pinned_output.as_ref().map_or(0, |b| b.len());
        if have >= need {
            return Ok(());
        }
        let alloc_len = need.max(1);
        let buf = unsafe { self.context.alloc_pinned::<i32>(alloc_len) }
            .map_err(|e| LLMError::GpuError(format!("pinned output alloc: {e}")))?;
        self.pinned_output = Some(buf);
        Ok(())
    }

    /// Legacy CPU forward pass removed -- f16 only, use GpuModelRunner.
    #[allow(dead_code)]
    fn gpu_forward_cpu_attention(&mut self, _model_input: &ModelInput) -> Result<Vec<f32>> {
        Err(LLMError::GpuError(
            "CPU attention path removed -- use GPU runner (f16)".into(),
        ))
    }

    /// Sample tokens from the flat logits buffer.
    fn sample_tokens(
        &mut self,
        logits: &[f32],
        metadata: &[SequenceGroupMetadata],
    ) -> Result<Vec<GpuSamplerResult>> {
        let vocab_size = self.vocab_size;
        let mut results = Vec::new();
        let mut offset = 0;

        for group in metadata {
            for (seq_id, seq_data) in &group.seq_data {
                let num_tokens = if group.is_prompt {
                    seq_data.prompt_token_ids.len()
                } else {
                    1
                };

                let last_logit_start = offset + (num_tokens - 1) * vocab_size;
                let last_logit_end = last_logit_start + vocab_size;

                let (token_id, logprob, top_logprobs) = if last_logit_end <= logits.len() {
                    let seq_logits = &logits[last_logit_start..last_logit_end];

                    let past: Vec<TokenId> = seq_data
                        .prompt_token_ids
                        .iter()
                        .chain(seq_data.output_token_ids.iter())
                        .copied()
                        .collect();

                    // Apply guided decoding mask if response_format is set
                    let mut guided_logits_buf;
                    let final_logits =
                        if !matches!(group.sampling_params.response_format, ResponseFormat::Text) {
                            guided_logits_buf = seq_logits.to_vec();
                            let state = self.guided_states.entry(seq_id.0).or_insert_with(|| {
                                GuidedDecodingState::new(&group.sampling_params.response_format)
                                    .unwrap_or_else(|_| {
                                        GuidedDecodingState::new(&ResponseFormat::Text).unwrap()
                                    })
                            });
                            if let Some(ref vocab) = self.vocab_table {
                                state.apply_mask(&mut guided_logits_buf, vocab);
                            }
                            &guided_logits_buf[..]
                        } else {
                            seq_logits
                        };

                    let mut rng = make_rng(group.sampling_params.seed);

                    let sampler_out = self.sampler.sample(
                        final_logits,
                        vocab_size,
                        &group.sampling_params,
                        &past,
                        &mut rng,
                    )?;

                    (
                        sampler_out.token_id,
                        sampler_out.logprob,
                        sampler_out.top_logprobs,
                    )
                } else {
                    warn!(
                        seq_id = seq_id.0,
                        logits_len = logits.len(),
                        expected_end = last_logit_end,
                        "logits too short, producing dummy token"
                    );
                    (0, 0.0, Vec::new())
                };

                results.push(GpuSamplerResult {
                    seq_id: seq_id.0,
                    token_id,
                    logprob,
                    top_logprobs,
                });

                offset += num_tokens * vocab_size;
            }
        }

        Ok(results)
    }

    /// Fast path: convert GPU-side argmax token IDs to sampling results.
    /// Synchronize the runner stream and read token IDs from the pinned buffer.
    /// Called after an async DtoH was enqueued via `ForwardOutput::TokenIdsPending`.
    #[cfg(feature = "cuda")]
    fn collect_pending_tokens(&mut self, actual_batch: usize) -> Result<Vec<i32>> {
        let runner = self
            .gpu_model_runner
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;
        runner.sync_stream()?;
        let pinned = self
            .pinned_output
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("pinned output buffer not allocated".into()))?;
        let slice = pinned
            .as_slice()
            .map_err(|e| LLMError::GpuError(format!("pinned buf read: {e}")))?;
        Ok(slice[..actual_batch].to_vec())
    }

    #[cfg(feature = "cuda")]
    fn sample_pending_tokens_from_pinned(
        &mut self,
        actual_batch: usize,
        metadata: &[SequenceGroupMetadata],
    ) -> Result<Vec<GpuSamplerResult>> {
        let runner = self
            .gpu_model_runner
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;
        runner.sync_stream()?;
        let pinned = self
            .pinned_output
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("pinned output buffer not allocated".into()))?;
        let slice = pinned
            .as_slice()
            .map_err(|e| LLMError::GpuError(format!("pinned buf read: {e}")))?;
        self.sample_tokens_from_gpu_argmax(&slice[..actual_batch], metadata)
    }

    #[cfg(not(feature = "cuda"))]
    fn collect_pending_tokens(&mut self, _actual_batch: usize) -> Result<Vec<i32>> {
        Err(LLMError::GpuError(
            "async DtoH requires cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn sample_pending_tokens_from_pinned(
        &mut self,
        _actual_batch: usize,
        _metadata: &[SequenceGroupMetadata],
    ) -> Result<Vec<GpuSamplerResult>> {
        Err(LLMError::GpuError(
            "async DtoH requires cuda feature".into(),
        ))
    }

    /// Used when all requests in the batch use temperature=0 (greedy).
    /// The argmax kernel produces one token ID per input token row;
    /// we pick the last token per sequence (same logic as sample_tokens).
    fn sample_tokens_from_gpu_argmax(
        &self,
        token_ids: &[i32],
        metadata: &[SequenceGroupMetadata],
    ) -> Result<Vec<GpuSamplerResult>> {
        let mut results = Vec::new();
        let mut offset = 0usize;

        for group in metadata {
            for (seq_id, seq_data) in &group.seq_data {
                let num_tokens = if group.is_prompt {
                    seq_data.prompt_token_ids.len()
                } else {
                    1
                };

                let last_idx = offset + num_tokens - 1;
                let token_id = if last_idx < token_ids.len() {
                    token_ids[last_idx] as u32
                } else {
                    warn!(
                        seq_id = seq_id.0,
                        token_ids_len = token_ids.len(),
                        expected_idx = last_idx,
                        "argmax token_ids too short, producing dummy token"
                    );
                    0
                };

                results.push(GpuSamplerResult {
                    seq_id: seq_id.0,
                    token_id,
                    logprob: 0.0, // no logprob available in greedy fast path
                    top_logprobs: Vec::new(),
                });

                offset += num_tokens;
            }
        }

        Ok(results)
    }

    pub fn warm_up(&self) -> Result<()> {
        info!(device_id = self.device_id, "GpuWorker warm-up complete");
        Ok(())
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    pub fn forward_logits_partial(
        &mut self,
        metadata: &[SequenceGroupMetadata],
        max_layers: usize,
    ) -> Result<Vec<f32>> {
        if metadata.is_empty() {
            return Ok(Vec::new());
        }
        let model_input = self.prepare_model_input(metadata)?;
        #[cfg(feature = "cuda")]
        {
            let runner = self
                .gpu_model_runner
                .as_ref()
                .ok_or_else(|| LLMError::GpuError("GPU model runner not initialized".into()))?;
            return runner.forward_partial(
                &model_input.token_ids,
                &model_input.position_ids,
                &model_input.attention_metadata,
                model_input.is_prefill,
                max_layers,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = max_layers;
            Err(LLMError::GpuError("requires --features cuda".into()))
        }
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
    pub fn cache_engine(&self) -> Option<&CacheEngine> {
        self.cache_engine.as_ref()
    }
    pub fn blas(&self) -> &CublasHandle {
        &self.cublas
    }
    pub fn gpu(&self) -> &Arc<CudaContext> {
        &self.context
    }
    pub fn gpu_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Build a WorkerConfig from EngineConfig.
fn worker_config_from_engine(
    model_path: &str,
    config: &rvllm_config::EngineConfig,
) -> WorkerConfig {
    WorkerConfig {
        device_id: 0,
        num_layers: 32,
        num_kv_heads: 32,
        head_dim: 128,
        hidden_size: 4096,
        num_attention_heads: 32,
        intermediate_size: 11008,
        vocab_size: 32000,
        max_model_len: config.model.max_model_len,
        block_size: config.cache.block_size,
        gpu_memory_utilization: config.cache.gpu_memory_utilization,
        rank: 0,
        tensor_parallel_size: config.parallel.tensor_parallel_size,
        pipeline_parallel_size: config.parallel.pipeline_parallel_size,
        architecture: "llama".into(),
        dtype: config.model.dtype.clone(),
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        kv_cache_dtype: config.cache.kv_cache_dtype.clone(),
        enable_prefix_caching: config.cache.enable_prefix_caching,
        partial_rotary_factor: 1.0,
        attn_logit_softcapping: 0.0,
        num_local_experts: 0,
        num_experts_per_tok: 0,
    }
}

fn gpu_err(e: impl std::fmt::Display) -> LLMError {
    LLMError::GpuError(format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::prelude::{RequestId, SequenceId};

    fn make_seq_data(prompt: Vec<TokenId>, output: Vec<TokenId>) -> SequenceData {
        SequenceData {
            prompt_token_ids: prompt,
            output_token_ids: output,
            cumulative_logprob: 0.0,
            seq_len: 0,
            last_token_id: 0,
        }
    }

    #[test]
    fn gpu_worker_output_construction() {
        let output = GpuWorkerOutput {
            outputs: vec![
                GpuSamplerResult {
                    seq_id: 1,
                    token_id: 42,
                    logprob: -0.5,
                    top_logprobs: vec![],
                },
                GpuSamplerResult {
                    seq_id: 2,
                    token_id: 99,
                    logprob: -1.2,
                    top_logprobs: vec![(99, -1.2), (50, -2.0)],
                },
            ],
        };
        assert_eq!(output.outputs.len(), 2);
        assert_eq!(output.outputs[0].token_id, 42);
    }

    #[test]
    fn worker_config_from_engine_defaults() {
        let engine = rvllm_config::EngineConfig::default();
        let wc = worker_config_from_engine("test/model", &engine);
        assert_eq!(wc.device_id, 0);
        assert_eq!(wc.block_size, engine.cache.block_size);
    }

    #[test]
    fn rope_table_basic() {
        let rope = RopeTable::new(4, 16, 10000.0);
        assert_eq!(rope.half_dim, 2);
        assert_eq!(rope.max_seq_len, 16);
        // pos=0: cos(0)=1, sin(0)=0
        assert!((rope.cos_table[0] - 1.0).abs() < 1e-6);
        assert!(rope.sin_table[0].abs() < 1e-6);
    }

    #[test]
    fn rope_apply_identity_at_pos_zero() {
        let rope = RopeTable::new(4, 16, 10000.0);
        let mut data = vec![1.0, 2.0, 3.0, 4.0]; // 1 token, 1 head, head_dim=4
        let positions = vec![0u32];
        rope.apply(&mut data, &positions, 1, 1, 4);
        // At pos 0, cos=1 sin=0, so values unchanged
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn kv_cache_append_and_retrieve() {
        let mut cache = KVCache::new(2, 2, 4); // 2 layers, 2 kv heads, head_dim 4
        assert_eq!(cache.len(), 0);

        // Simulate prefill: 3 tokens, kv_dim=8
        let k_data = vec![1.0f32; 3 * 8];
        let v_data = vec![2.0f32; 3 * 8];
        cache.append(0, &k_data, &v_data);
        cache.advance(3);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.keys(0).len(), 24); // 3 * 8

        // Simulate decode: 1 token
        let k_new = vec![3.0f32; 8];
        let v_new = vec![4.0f32; 8];
        cache.append(0, &k_new, &v_new);
        cache.advance(1);
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.keys(0).len(), 32); // 4 * 8

        // Layer 1 should still be empty
        assert_eq!(cache.keys(1).len(), 0);
    }
}
