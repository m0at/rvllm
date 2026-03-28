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
use tracing::{debug, info, warn};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use rvllm_core::prelude::{BlockId, LLMError, Result, SamplingParams, TokenId};
use rvllm_gpu::prelude::{CublasHandle, CudaGpuAllocator, GpuStream};

use rvllm_kv_cache::CacheEngine;
use rvllm_model_runner::gpu_runner::ForwardOutput;
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

/// Per-layer weights stored as CudaSlice<f32> on GPU.
struct LayerWeights {
    input_layernorm: CudaSlice<f32>,
    post_attention_layernorm: CudaSlice<f32>,
    q_proj_weight: CudaSlice<f32>,
    q_proj_bias: Option<CudaSlice<f32>>,
    k_proj_weight: CudaSlice<f32>,
    k_proj_bias: Option<CudaSlice<f32>>,
    v_proj_weight: CudaSlice<f32>,
    v_proj_bias: Option<CudaSlice<f32>>,
    o_proj_weight: CudaSlice<f32>,
    gate_proj_weight: CudaSlice<f32>,
    up_proj_weight: CudaSlice<f32>,
    down_proj_weight: CudaSlice<f32>,
}

/// All model weights on GPU.
struct GpuModelWeights {
    embed_tokens: CudaSlice<f32>,
    layers: Vec<LayerWeights>,
    norm_weight: CudaSlice<f32>,
    lm_head_weight: Option<CudaSlice<f32>>, // None if tie_word_embeddings
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
    guided_states: HashMap<u64, GuidedDecodingState>,
    vocab_table: Option<VocabTable>,
    /// Raw weight map preserved for deferred GpuModelRunner construction.
    #[cfg(feature = "cuda")]
    raw_weight_map: Option<HashMap<String, CudaSlice<f32>>>,
    #[cfg(feature = "cuda")]
    raw_weight_map_f16: Option<HashMap<String, CudaSlice<half::f16>>>,
    /// GPU-resident model runner (full forward pass on GPU, no CPU attention fallback).
    /// Constructed in init_cache() once cache geometry is known.
    #[cfg(feature = "cuda")]
    gpu_model_runner: Option<rvllm_model_runner::gpu_runner::GpuModelRunner>,
    /// CUDA graph runner for decode step capture/replay.
    graph_runner: GraphRunner,
    /// Number of forward calls so far (for warmup before graph capture).
    forward_count: usize,
}

impl GpuWorker {
    pub fn new(config: WorkerConfig) -> Result<Self> {
        let device_id = config.device_id;
        info!(device_id, "creating GpuWorker");

        let context = CudaContext::new(device_id).map_err(|e| {
            LLMError::GpuError(format!("failed to init CUDA device {}: {}", device_id, e))
        })?;

        let stream = context.default_stream();

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
        );
        if use_fp8_kv {
            info!("FP8 KV cache enabled");
        }

        let graph_runner = GraphRunner::new(GraphRunnerConfig {
            max_batch_size: 32,
            enabled: true,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
        });

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
            guided_states: HashMap::new(),
            vocab_table: None,
            #[cfg(feature = "cuda")]
            raw_weight_map: None,
            #[cfg(feature = "cuda")]
            raw_weight_map_f16: None,
            #[cfg(feature = "cuda")]
            gpu_model_runner: None,
            graph_runner,
            forward_count: 0,
        })
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

        let all_weights_full =
            rvllm_model_loader::gpu_loader::load_weights_to_gpu(model_path, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("weight loading failed: {e}")))?;

        info!("loaded {} weight tensors to GPU", all_weights_full.len());

        // Preserve a clone of the raw weight map for deferred GpuModelRunner construction
        #[cfg(feature = "cuda")]
        {
            self.raw_weight_map = Some(all_weights_full.clone());

            // Also load f16 weights for hgemm path when dtype is half
            if self.config.dtype.is_half() {
                info!("loading f16 weights for hgemm path");
                let f16_weights = rvllm_model_loader::gpu_loader::load_weights_to_gpu_f16(
                    model_path,
                    &self.stream,
                )
                .map_err(|e| LLMError::GpuError(format!("f16 weight loading failed: {e}")))?;
                info!("loaded {} f16 weight tensors", f16_weights.len());
                self.raw_weight_map_f16 = Some(f16_weights);
            }
        }

        let mut all_weights = all_weights_full;

        let tie = !all_weights.contains_key("lm_head.weight");
        info!(tie_word_embeddings = tie, "building model weight structure");

        let embed_tokens = all_weights
            .remove("model.embed_tokens.weight")
            .ok_or_else(|| LLMError::ModelError("missing model.embed_tokens.weight".into()))?;

        let norm_weight = all_weights
            .remove("model.norm.weight")
            .ok_or_else(|| LLMError::ModelError("missing model.norm.weight".into()))?;

        let lm_head_weight = all_weights.remove("lm_head.weight");

        let num_layers = self.config.num_layers;
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(LayerWeights {
                input_layernorm: all_weights
                    .remove(&format!("{p}.input_layernorm.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                post_attention_layernorm: all_weights
                    .remove(&format!("{p}.post_attention_layernorm.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                q_proj_weight: all_weights
                    .remove(&format!("{p}.self_attn.q_proj.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                q_proj_bias: all_weights.remove(&format!("{p}.self_attn.q_proj.bias")),
                k_proj_weight: all_weights
                    .remove(&format!("{p}.self_attn.k_proj.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                k_proj_bias: all_weights.remove(&format!("{p}.self_attn.k_proj.bias")),
                v_proj_weight: all_weights
                    .remove(&format!("{p}.self_attn.v_proj.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                v_proj_bias: all_weights.remove(&format!("{p}.self_attn.v_proj.bias")),
                o_proj_weight: all_weights
                    .remove(&format!("{p}.self_attn.o_proj.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                gate_proj_weight: all_weights
                    .remove(&format!("{p}.mlp.gate_proj.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                up_proj_weight: all_weights
                    .remove(&format!("{p}.mlp.up_proj.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
                down_proj_weight: all_weights
                    .remove(&format!("{p}.mlp.down_proj.weight"))
                    .ok_or_else(|| LLMError::ModelError(format!("missing weight")))?,
            });

            debug!(layer = i, "loaded layer weights");
        }

        self.model_weights = Some(GpuModelWeights {
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
            tie_word_embeddings: tie,
        });

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

            let raw_map = self.raw_weight_map.take().ok_or_else(|| {
                LLMError::GpuError("raw weight map not available -- call load_weights first".into())
            })?;
            let mut loader_weights = LoaderWeights::new(raw_map, HashMap::new());

            // Insert f16 weights for hgemm path
            if let Some(f16_map) = self.raw_weight_map_f16.take() {
                for (name, slice) in f16_map {
                    loader_weights.insert_f16(name, slice, vec![]);
                }
                info!("inserted f16 weights into model weight container");
            }

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

            let runner_blas = CublasHandle::new(self.stream.clone())
                .map_err(|e| LLMError::GpuError(format!("runner cublas: {e}")))?;

            // KernelLoader: try build output dir, fall back to empty (PTX loaded at runtime)
            let ptx_dir = Self::find_ptx_dir();
            let loader = rvllm_gpu::kernel_loader::KernelLoader::new(
                self.context.clone(),
                self.stream.clone(),
                &ptx_dir.unwrap_or_else(|| std::path::PathBuf::from("/nonexistent")),
            )
            .map_err(|e| LLMError::GpuError(format!("kernel loader: {e}")))?;

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

            if self.config.dtype.is_half() {
                runner.enable_fp16();
                info!("FP16 inference enabled (hgemm path)");
            }

            self.gpu_model_runner = Some(runner);
            info!(
                "GPU model runner initialized with {} GPU blocks (block_size={})",
                num_gpu_blocks, block_size
            );
        }

        Ok(())
    }

    /// Search for compiled PTX directory from build output.
    #[cfg(feature = "cuda")]
    fn find_ptx_dir() -> Option<std::path::PathBuf> {
        for base in &[
            "/root/vllm-rs/target/release/build",
            "/root/vllm-rs/target/debug/build",
            "./target/release/build",
            "./target/debug/build",
        ] {
            if let Ok(entries) = std::fs::read_dir(base) {
                for entry in entries.flatten() {
                    let ptx_path = entry.path().join("out/ptx");
                    if ptx_path.join("rotary_embedding.ptx").exists() {
                        return Some(ptx_path);
                    }
                }
            }
        }
        if let Ok(dir) = std::env::var("RVLLM_PTX_DIR") {
            let p = std::path::PathBuf::from(dir);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    /// Profile available GPU memory and return (num_gpu_blocks, num_cpu_blocks).
    pub fn profile_num_available_blocks(
        &self,
        gpu_memory_utilization: f32,
    ) -> Result<(usize, usize)> {
        let (free, _total) = self.context.mem_get_info()
            .map_err(|e| LLMError::GpuError(format!("mem_get_info failed: {e}")))?;
        let available = (free as f32 * gpu_memory_utilization) as usize;

        let cache_cfg = self.config.cache_config();
        let total_block_bytes = cache_cfg.total_block_bytes();

        // CudaCacheEngine now stores f16, matching CacheConfig::block_bytes.
        let num_gpu_blocks = if total_block_bytes > 0 {
            available / total_block_bytes
        } else {
            0
        };
        let num_cpu_blocks = 128;

        info!(
            free,
            available, num_gpu_blocks, num_cpu_blocks, "profiled available blocks"
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

    /// Execute one inference step with real GPU matmuls.
    pub fn execute(&mut self, metadata: &[SequenceGroupMetadata]) -> Result<GpuWorkerOutput> {
        if metadata.is_empty() {
            return Ok(GpuWorkerOutput {
                outputs: Vec::new(),
            });
        }

        debug!(
            device_id = self.device_id,
            num_groups = metadata.len(),
            "GpuWorker execute"
        );

        let model_input = input::prepare_input(metadata, self.config.block_size)?;
        let total_tokens = model_input.num_tokens();

        debug!(
            num_tokens = total_tokens,
            is_prefill = model_input.is_prefill,
            "input prepared"
        );

        let greedy_only = Self::all_greedy(metadata);

        // Run real GPU forward pass with RoPE and KV cache
        info!(greedy_only, "gpu_worker: entering gpu_forward");
        let fwd_output = self.gpu_forward_ex(&model_input, greedy_only)?;

        let outputs = match fwd_output {
            ForwardOutput::TokenIds(ref token_ids) => {
                info!(
                    num_ids = token_ids.len(),
                    "gpu_worker: gpu argmax fast path"
                );
                self.sample_tokens_from_gpu_argmax(token_ids, metadata)?
            }
            ForwardOutput::Logits(ref logits) => {
                info!(
                    logits_len = logits.len(),
                    "gpu_worker: full logits path"
                );
                self.sample_tokens(logits, metadata)?
            }
        };

        debug!(
            device_id = self.device_id,
            num_outputs = outputs.len(),
            "execute done"
        );
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
        self.forward_count += 1;

        // Try graph replay for eligible decode steps.
        if self.graph_runner.can_use_graph(model_input) {
            let actual_batch = model_input.num_tokens();
            if let Some(logits) =
                self.graph_runner
                    .try_replay_decode(&self.compute_stream, model_input, actual_batch)?
            {
                debug!(actual_batch, "CUDA graph replayed for decode");
                return Ok(logits);
            }
        }

        // Normal forward pass.
        let logits = self.raw_gpu_forward(model_input)?;

        // After warmup, try to capture a graph for this decode batch size.
        self.maybe_capture_graph(model_input, Some(&logits))?;

        Ok(logits)
    }

    /// Extended GPU forward with greedy fast path. When `greedy_only` is true and
    /// CUDA graphs are not replaying, runs argmax on GPU and returns only token IDs.
    /// Falls back to full logits when graph replay is active (graph captured with
    /// full logits path).
    fn gpu_forward_ex(
        &mut self,
        model_input: &ModelInput,
        greedy_only: bool,
    ) -> Result<ForwardOutput> {
        self.forward_count += 1;

        // Try graph replay for eligible decode steps.
        // Graph path always returns full logits since it was captured with the
        // standard (non-greedy) forward.
        if self.graph_runner.can_use_graph(model_input) {
            let actual_batch = model_input.num_tokens();
            if let Some(logits) =
                self.graph_runner
                    .try_replay_decode(&self.compute_stream, model_input, actual_batch)?
            {
                debug!(actual_batch, "CUDA graph replayed for decode (ex path)");
                return Ok(ForwardOutput::Logits(logits));
            }
        }

        // Normal forward pass with optional greedy argmax on GPU.
        let output = self.raw_gpu_forward_ex(model_input, greedy_only)?;

        // After warmup, try to capture a graph for this decode batch size.
        // Extract logits for caching (only available from Logits variant).
        let logits_for_cache = match &output {
            ForwardOutput::Logits(l) => Some(l.as_slice()),
            ForwardOutput::TokenIds(_) => None,
        };
        self.maybe_capture_graph(model_input, logits_for_cache)?;

        Ok(output)
    }

    /// Attempt to capture a CUDA graph for the current decode batch size.
    ///
    /// Called after a normal forward pass once the warmup period has elapsed.
    /// Stores a persistent copy of the padded input so the graph's internal
    /// memcpy nodes reference stable host addresses on replay.
    fn maybe_capture_graph(
        &mut self,
        model_input: &ModelInput,
        logits: Option<&[f32]>,
    ) -> Result<()> {
        if self.forward_count <= Self::GRAPH_WARMUP_CALLS
            || model_input.is_prefill
            || !self.graph_runner.can_use_graph(model_input)
        {
            return Ok(());
        }

        let (padded_input, _actual_batch) = self.graph_runner.pad_input(model_input)?;
        let padded_bs = padded_input.num_tokens();

        if self.graph_runner.was_capture_attempted(padded_bs) {
            return Ok(());
        }

        info!(padded_bs, "capturing CUDA graph after warmup");

        // Store the padded input as persistent buffer before capture.
        // The graph's memcpy nodes will reference these Vec heap addresses.
        self.graph_runner
            .store_input_buffer(padded_input.clone(), padded_bs);

        #[cfg(feature = "cuda")]
        {
            // Use the persistent padded_input for the capture closure.
            // The closure references the stored buffer's Vec heap addresses
            // so CUDA graph memcpy nodes point to stable memory.
            let gpu_runner = self.gpu_model_runner.as_ref();
            // We stored padded_input above; now retrieve the persistent copy's
            // address for the capture run. Split borrow: take the input ref
            // before the &mut call by extracting what we need.
            let pi_tokens: &[u32] = &padded_input.token_ids;
            let pi_positions: &[u32] = &padded_input.position_ids;
            let pi_attn = &padded_input.attention_metadata;
            let pi_prefill = padded_input.is_prefill;

            let capture_result =
                self.graph_runner
                    .capture_decode_graph(&self.compute_stream, padded_bs, || {
                        let runner = gpu_runner.ok_or_else(|| {
                            LLMError::GpuError("GPU model runner not initialized".into())
                        })?;
                        runner.forward(pi_tokens, pi_positions, pi_attn, pi_prefill)
                    });

            match capture_result {
                Ok(capture_logits) => {
                    // Cache the logits from the capture forward pass.
                    self.graph_runner.cache_output(padded_bs, capture_logits);
                }
                Err(e) => {
                    warn!(%e, "CUDA graph capture failed, continuing without graph");
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Mock path: mark as captured but do NOT cache output.
            // Without real CUDA graph replay, each decode step must run the
            // real forward pass. The graph pool's no-op replay + empty output
            // cache causes try_replay_decode to return None, falling through
            // to the normal forward path.
            let _ = logits;
            self.graph_runner.mark_captured(padded_bs);
        }

        Ok(())
    }

    /// Legacy CPU forward pass (kept for comparison benchmarks only, not used in production).
    #[allow(dead_code)]
    fn gpu_forward_cpu_attention(&mut self, model_input: &ModelInput) -> Result<Vec<f32>> {
        if self.model_weights.is_none() {
            return Err(LLMError::GpuError("model not loaded".into()));
        }

        let token_ids = &model_input.token_ids;
        let position_ids = &model_input.position_ids;
        let is_prefill = model_input.is_prefill;

        let num_tokens = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let intermediate_size = self.config.intermediate_size;

        // On prefill, reset the KV cache (new sequence)
        if is_prefill {
            if let Some(ref mut cache) = self.kv_cache {
                cache.clear();
            }
            if let Some(ref mut cache) = self.fp8_kv_cache {
                cache.clear();
            }
        }

        let cache_len = self
            .kv_cache
            .as_ref()
            .map(|c| c.len())
            .or_else(|| self.fp8_kv_cache.as_ref().map(|c| c.len()))
            .unwrap_or(0);
        debug!(
            num_tokens,
            is_prefill, "gpu_forward_cpu_attention: cache_len={}", cache_len
        );

        // 1. Embedding lookup (GPU gather when kernel available, else CPU fallback)
        let mut hidden = {
            let w = self.model_weights.as_ref().unwrap();
            self.embedding_lookup_gpu(token_ids, &w.embed_tokens, hidden_size)?
        };

        // 2. Transformer layers
        let num_layers = self.model_weights.as_ref().unwrap().layers.len();
        for layer_idx in 0..num_layers {
            hidden = self.transformer_layer_cached(
                &hidden,
                layer_idx,
                position_ids,
                is_prefill,
                num_tokens,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
            )?;
        }

        // Advance KV cache position after all layers processed
        if let Some(ref mut cache) = self.kv_cache {
            cache.advance(num_tokens);
        }
        if let Some(ref mut cache) = self.fp8_kv_cache {
            cache.advance(num_tokens);
        }

        // 3. Final RMS norm
        {
            let w = self.model_weights.as_ref().unwrap();
            hidden = self.rms_norm_gpu(&hidden, &w.norm_weight, num_tokens, hidden_size)?;
        }

        // 4. LM head
        let vocab_size = self.vocab_size;
        let mut logits_gpu = self
            .stream
            .alloc_zeros::<f32>(num_tokens * vocab_size)
            .map_err(gpu_err)?;

        {
            let w = self.model_weights.as_ref().unwrap();
            let lm_head = if w.tie_word_embeddings {
                &w.embed_tokens
            } else {
                w.lm_head_weight.as_ref().unwrap()
            };

            self.cublas
                .sgemm(
                    num_tokens,
                    vocab_size,
                    hidden_size,
                    1.0,
                    &hidden,
                    lm_head,
                    0.0,
                    &mut logits_gpu,
                )
                .map_err(|e| LLMError::GpuError(format!("lm_head sgemm: {e}")))?;
        }

        let logits = self.stream.clone_dtoh(&logits_gpu).map_err(gpu_err)?;

        Ok(logits)
    }

    /// Embedding lookup via GPU gather kernel when available.
    ///
    /// With the kernel: uploads only token_ids (~batch*4 bytes), gathers on GPU.
    /// Without: falls back to CPU gather (DtoH full embed table).
    fn embedding_lookup_gpu(
        &self,
        token_ids: &[TokenId],
        embed_weight: &CudaSlice<f32>,
        hidden_size: usize,
    ) -> Result<CudaSlice<f32>> {
        let num_tokens = token_ids.len();
        let output_len = num_tokens * hidden_size;

        // CPU gather (legacy path -- production uses GpuModelRunner)
        let embed_host = self.stream.clone_dtoh(embed_weight).map_err(gpu_err)?;
        let mut result = vec![0.0f32; output_len];
        for (i, &tid) in token_ids.iter().enumerate() {
            let tid = tid as usize;
            if tid * hidden_size + hidden_size <= embed_host.len() {
                let src = &embed_host[tid * hidden_size..(tid + 1) * hidden_size];
                result[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(src);
            }
        }
        self.stream.clone_htod(&result).map_err(gpu_err)
    }

    /// Transformer layer with RoPE and KV cache.
    fn transformer_layer_cached(
        &mut self,
        hidden: &CudaSlice<f32>,
        layer_idx: usize,
        position_ids: &[u32],
        is_prefill: bool,
        num_tokens: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Result<CudaSlice<f32>> {
        // We need to borrow weights immutably but also use &mut self for cache.
        // Work around by extracting what we need from weights first.
        let weights = self
            .model_weights
            .as_ref()
            .ok_or_else(|| LLMError::GpuError("model not loaded".into()))?;
        let layer = &weights.layers[layer_idx];

        // 1. RMS Norm
        let normed = self.rms_norm_gpu(hidden, &layer.input_layernorm, num_tokens, hidden_size)?;

        // 2. QKV projections on GPU
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut q_gpu = self
            .stream
            .alloc_zeros::<f32>(num_tokens * q_dim)
            .map_err(gpu_err)?;
        let mut k_gpu = self
            .stream
            .alloc_zeros::<f32>(num_tokens * kv_dim)
            .map_err(gpu_err)?;
        let mut v_gpu = self
            .stream
            .alloc_zeros::<f32>(num_tokens * kv_dim)
            .map_err(gpu_err)?;

        self.cublas.sgemm(
            num_tokens,
            q_dim,
            hidden_size,
            1.0,
            &normed,
            &layer.q_proj_weight,
            0.0,
            &mut q_gpu,
        )?;
        self.cublas.sgemm(
            num_tokens,
            kv_dim,
            hidden_size,
            1.0,
            &normed,
            &layer.k_proj_weight,
            0.0,
            &mut k_gpu,
        )?;
        self.cublas.sgemm(
            num_tokens,
            kv_dim,
            hidden_size,
            1.0,
            &normed,
            &layer.v_proj_weight,
            0.0,
            &mut v_gpu,
        )?;

        // Add biases if present
        if let Some(ref qb) = layer.q_proj_bias {
            self.add_bias_gpu(&mut q_gpu, qb, num_tokens, q_dim)?;
        }
        if let Some(ref kb) = layer.k_proj_bias {
            self.add_bias_gpu(&mut k_gpu, kb, num_tokens, kv_dim)?;
        }
        if let Some(ref vb) = layer.v_proj_bias {
            self.add_bias_gpu(&mut v_gpu, vb, num_tokens, kv_dim)?;
        }

        // 3. Download Q, K, V to CPU for RoPE + attention
        let mut q_host = self.stream.clone_dtoh(&q_gpu).map_err(gpu_err)?;
        let mut k_host = self.stream.clone_dtoh(&k_gpu).map_err(gpu_err)?;
        let v_host = self.stream.clone_dtoh(&v_gpu).map_err(gpu_err)?;

        // 4. Apply RoPE to Q and K
        if let Some(ref rope) = self.rope_table {
            rope.apply(&mut q_host, position_ids, num_tokens, num_heads, head_dim);
            rope.apply(
                &mut k_host,
                position_ids,
                num_tokens,
                num_kv_heads,
                head_dim,
            );
        }

        // 5. KV Cache: append new K,V, then use full cache for attention
        let (full_k, full_v, total_kv_len) = if let Some(ref mut cache) = self.fp8_kv_cache {
            cache.append_quantized(layer_idx, &k_host, &v_host);
            let total_len = cache.len() + num_tokens;
            let full_k = cache.keys_dequantized(layer_idx);
            let full_v = cache.values_dequantized(layer_idx);
            (full_k, full_v, total_len)
        } else if let Some(ref mut cache) = self.kv_cache {
            cache.append(layer_idx, &k_host, &v_host);
            let total_len = cache.len() + num_tokens;
            let full_k = cache.keys(layer_idx).to_vec();
            let full_v = cache.values(layer_idx).to_vec();
            (full_k, full_v, total_len)
        } else {
            (k_host, v_host, num_tokens)
        };

        // 6. Attention (CPU)
        let attn_output = self.cached_attention(
            &q_host,
            &full_k,
            &full_v,
            num_tokens,
            total_kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            is_prefill,
        );

        let attn_out_gpu = self.stream.clone_htod(&attn_output).map_err(gpu_err)?;

        // Need to re-borrow weights after mutable kv_cache borrow is done
        let weights = self.model_weights.as_ref().unwrap();
        let layer = &weights.layers[layer_idx];

        // 7. Output projection
        let mut attn_proj = self
            .stream
            .alloc_zeros::<f32>(num_tokens * hidden_size)
            .map_err(gpu_err)?;
        self.cublas.sgemm(
            num_tokens,
            hidden_size,
            q_dim,
            1.0,
            &attn_out_gpu,
            &layer.o_proj_weight,
            0.0,
            &mut attn_proj,
        )?;

        // 8+9. Fused residual add + post-attention RMS Norm (one kernel when available)
        let (normed2, residual1) = self.fused_residual_rmsnorm_gpu(
            hidden,
            &attn_proj,
            &layer.post_attention_layernorm,
            num_tokens,
            hidden_size,
        )?;

        // 10. MLP: SiLU-gated
        let mut gate = self
            .stream
            .alloc_zeros::<f32>(num_tokens * intermediate_size)
            .map_err(gpu_err)?;
        let mut up = self
            .stream
            .alloc_zeros::<f32>(num_tokens * intermediate_size)
            .map_err(gpu_err)?;

        self.cublas.sgemm(
            num_tokens,
            intermediate_size,
            hidden_size,
            1.0,
            &normed2,
            &layer.gate_proj_weight,
            0.0,
            &mut gate,
        )?;
        self.cublas.sgemm(
            num_tokens,
            intermediate_size,
            hidden_size,
            1.0,
            &normed2,
            &layer.up_proj_weight,
            0.0,
            &mut up,
        )?;

        let mlp_activated = self.fused_silu_mul_gpu(&gate, &up, num_tokens * intermediate_size)?;

        let mut down = self
            .stream
            .alloc_zeros::<f32>(num_tokens * hidden_size)
            .map_err(gpu_err)?;
        self.cublas.sgemm(
            num_tokens,
            hidden_size,
            intermediate_size,
            1.0,
            &mlp_activated,
            &layer.down_proj_weight,
            0.0,
            &mut down,
        )?;

        // 11. Residual add
        let result = self.add_tensors_gpu(&residual1, &down, num_tokens * hidden_size)?;

        Ok(result)
    }

    /// Multi-head attention with GQA support, operating on CPU host buffers.
    /// Handles both prefill (causal mask over all Q positions) and decode (Q is just new tokens,
    /// K/V includes full cache).
    fn cached_attention(
        &self,
        q: &[f32], // [num_q_tokens, num_heads * head_dim]
        k: &[f32], // [total_kv_len, num_kv_heads * head_dim]
        v: &[f32], // [total_kv_len, num_kv_heads * head_dim]
        num_q_tokens: usize,
        total_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        is_prefill: bool,
    ) -> Vec<f32> {
        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = vec![0.0f32; num_q_tokens * q_stride];

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;

            for qi in 0..num_q_tokens {
                // Causal masking:
                // During prefill, Q position qi can attend to K positions 0..=qi
                // During decode, Q is the new token(s), can attend to all KV (full cache)
                let max_ki = if is_prefill {
                    // qi-th query token corresponds to the qi-th position in the sequence
                    qi
                } else {
                    // decode: Q tokens can attend to everything in cache
                    total_kv_len - 1
                };

                let attend_len = max_ki + 1;

                // Compute attention scores
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(attend_len);

                for ki in 0..attend_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q_val = q[qi * q_stride + h * head_dim + d];
                        let k_val = k[ki * kv_stride + kv_h * head_dim + d];
                        dot += q_val * k_val;
                    }
                    let score = dot * scale;
                    if score > max_score {
                        max_score = score;
                    }
                    scores.push(score);
                }

                // Softmax
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for s in scores.iter_mut() {
                        *s *= inv_sum;
                    }
                }

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for (ki, &weight) in scores.iter().enumerate() {
                        val += weight * v[ki * kv_stride + kv_h * head_dim + d];
                    }
                    output[qi * q_stride + h * head_dim + d] = val;
                }
            }
        }

        output
    }

    /// RMS Norm via GPU kernel (no CPU round-trip when kernel is loaded).
    fn rms_norm_gpu(
        &self,
        hidden: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        num_tokens: usize,
        hidden_size: usize,
    ) -> Result<CudaSlice<f32>> {
        let n = num_tokens * hidden_size;
        let eps = self.rms_norm_eps;

        // CPU RMS norm (legacy path -- production uses GpuModelRunner)
        let h = self.stream.clone_dtoh(hidden).map_err(gpu_err)?;
        let w = self.stream.clone_dtoh(weight).map_err(gpu_err)?;
        let mut out = vec![0.0f32; n];
        for t in 0..num_tokens {
            let row = &h[t * hidden_size..(t + 1) * hidden_size];
            let ss: f32 = row.iter().map(|x| x * x).sum();
            let inv_rms = (ss / hidden_size as f32 + eps).sqrt().recip();
            for d in 0..hidden_size {
                out[t * hidden_size + d] = row[d] * inv_rms * w.get(d).copied().unwrap_or(1.0);
            }
        }
        self.stream.clone_htod(&out).map_err(gpu_err)
    }

    /// Add bias to a tensor via GPU kernel (no CPU round-trip when kernel is loaded).
    fn add_bias_gpu(
        &self,
        tensor: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        num_tokens: usize,
        dim: usize,
    ) -> Result<()> {
        // CPU bias add (legacy path -- production uses GpuModelRunner)
        let mut data = self.stream.clone_dtoh(tensor).map_err(gpu_err)?;
        let b = self.stream.clone_dtoh(bias).map_err(gpu_err)?;
        for t in 0..num_tokens {
            for d in 0..dim.min(b.len()) {
                data[t * dim + d] += b[d];
            }
        }
        *tensor = self.stream.clone_htod(&data).map_err(gpu_err)?;
        Ok(())
    }

    /// Element-wise add two tensors via GPU kernel (no CPU round-trip when kernel is loaded).
    fn add_tensors_gpu(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<CudaSlice<f32>> {
        // CPU add (legacy path -- production uses GpuModelRunner)
        let a_h = self.stream.clone_dtoh(a).map_err(gpu_err)?;
        let b_h = self.stream.clone_dtoh(b).map_err(gpu_err)?;
        let out: Vec<f32> = a_h.iter().zip(b_h.iter()).map(|(x, y)| x + y).collect();
        self.stream.clone_htod(&out).map_err(gpu_err)
    }

    /// Fused SiLU(gate) * up via GPU kernel (no CPU round-trip when kernel is loaded).
    fn fused_silu_mul_gpu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        len: usize,
    ) -> Result<CudaSlice<f32>> {
        // CPU fused SiLU * mul (legacy path -- production uses GpuModelRunner)
        let g = self.stream.clone_dtoh(gate).map_err(gpu_err)?;
        let u = self.stream.clone_dtoh(up).map_err(gpu_err)?;
        let out: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(&gv, &uv)| (gv / (1.0 + (-gv).exp())) * uv)
            .collect();
        self.stream.clone_htod(&out).map_err(gpu_err)
    }

    /// Fused residual add + RMS norm. When the fused kernel is loaded, this
    /// executes in a single kernel launch. Otherwise falls back to separate ops.
    fn fused_residual_rmsnorm_gpu(
        &self,
        input: &CudaSlice<f32>,
        add: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        num_tokens: usize,
        hidden_size: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let n = num_tokens * hidden_size;

        // Separate add + norm (legacy path -- production uses GpuModelRunner)
        let residual = self.add_tensors_gpu(input, add, n)?;
        let normed = self.rms_norm_gpu(&residual, weight, num_tokens, hidden_size)?;
        Ok((normed, residual))
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
