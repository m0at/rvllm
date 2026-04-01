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

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync};

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
    device: Arc<CudaDevice>,
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
    /// FP8 (u8) weight map for FP8 inference mode.
    #[cfg(feature = "cuda")]
    raw_weight_map_fp8: Option<HashMap<String, CudaSlice<u8>>>,
    /// Per-tensor scale factors for FP8/GPTQ weights.
    #[cfg(feature = "cuda")]
    raw_weight_scales: Option<HashMap<String, CudaSlice<f32>>>,
    /// Per-group zero points for GPTQ INT4 weights.
    #[cfg(feature = "cuda")]
    raw_weight_zeros: Option<HashMap<String, CudaSlice<f32>>>,
    /// GPTQ group size (0 = not GPTQ).
    gptq_group_size: usize,
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

        let device = CudaDevice::new(device_id).map_err(|e| {
            LLMError::GpuError(format!("failed to init CUDA device {}: {}", device_id, e))
        })?;

        let cublas = CublasHandle::new(device.clone())
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
            device,
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
            raw_weight_map_fp8: None,
            #[cfg(feature = "cuda")]
            raw_weight_scales: None,
            #[cfg(feature = "cuda")]
            raw_weight_zeros: None,
            gptq_group_size: 0,
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
        info!(device_id = self.device_id, path = %model_path.display(), dtype = %self.config.dtype, "loading weights to GPU");

        // GPTQ INT4 path: load packed int4 weights with per-group scales/zeros.
        if self.config.dtype.is_gptq() {
            return self.load_weights_gptq(model_path);
        }

        // FP8 path: load F8_E4M3 as u8, non-FP8 as f32. Never materializes full f32/f16 model.
        if self.config.dtype.is_fp8() {
            return self.load_weights_fp8(model_path);
        }

        let all_weights_full =
            rvllm_model_loader::gpu_loader::load_weights_to_gpu(model_path, &self.device)
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
                    &self.device,
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

    /// Load FP8 model weights: F8_E4M3 tensors as u8, non-FP8 as f32.
    /// Never materializes the full f32/f16 model, preventing OOM on large models.
    fn load_weights_fp8(&mut self, model_path: &Path) -> Result<()> {
        info!(device_id = self.device_id, "loading FP8 weights (native u8, no f32 intermediate)");

        let result =
            rvllm_model_loader::gpu_loader::load_weights_to_gpu_fp8(model_path, &self.device)
                .map_err(|e| LLMError::GpuError(format!("fp8 weight loading failed: {e}")))?;

        info!(
            "loaded {} fp8 + {} f32 + {} scales to GPU",
            result.fp8_weights.len(),
            result.f32_weights.len(),
            result.weight_scales.len()
        );

        #[cfg(feature = "cuda")]
        {
            // Store the f32 weights (norms, biases, embeddings) for GpuModelRunner
            self.raw_weight_map = Some(result.f32_weights.clone());
            // Store FP8 weights and scales for GpuModelRunner
            self.raw_weight_map_fp8 = Some(result.fp8_weights);
            self.raw_weight_scales = Some(result.weight_scales);
        }

        // Build the local weight structure from f32 weights only (for fallback path).
        // For FP8, the GPU model runner handles everything - the local weights
        // just need norms and embeddings for the KV cache / fallback path.
        let mut f32_weights = result.f32_weights;

        let tie = !f32_weights.contains_key("lm_head.weight");
        info!(tie_word_embeddings = tie, "building model weight structure (fp8)");

        let embed_tokens = f32_weights
            .remove("model.embed_tokens.weight")
            .ok_or_else(|| LLMError::ModelError("missing model.embed_tokens.weight".into()))?;

        let norm_weight = f32_weights
            .remove("model.norm.weight")
            .ok_or_else(|| LLMError::ModelError("missing model.norm.weight".into()))?;

        let lm_head_weight = f32_weights.remove("lm_head.weight");

        // For FP8, we need dummy layer weights for the local fallback struct.
        // The real inference goes through GpuModelRunner which uses the fp8 maps.
        let num_layers = self.config.num_layers;
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let p = format!("model.layers.{}", i);
            // Norm weights are f32, projection weights may not be in f32_weights (they're in fp8).
            // For the local struct, use norms only. Projection weight fields need dummies.
            let input_ln = f32_weights
                .remove(&format!("{p}.input_layernorm.weight"))
                .ok_or_else(|| LLMError::ModelError(format!("missing {p}.input_layernorm.weight")))?;
            let post_ln = f32_weights
                .remove(&format!("{p}.post_attention_layernorm.weight"))
                .ok_or_else(|| LLMError::ModelError(format!("missing {p}.post_attention_layernorm.weight")))?;

            // Create zero-length dummy slices for projection weights (not used in GPU runner path)
            let dummy = self.device
                .alloc_zeros::<f32>(1)
                .map_err(|e| LLMError::GpuError(format!("dummy alloc: {e}")))?;

            layers.push(LayerWeights {
                input_layernorm: input_ln,
                post_attention_layernorm: post_ln,
                q_proj_weight: dummy.clone(),
                q_proj_bias: f32_weights.remove(&format!("{p}.self_attn.q_proj.bias")),
                k_proj_weight: dummy.clone(),
                k_proj_bias: f32_weights.remove(&format!("{p}.self_attn.k_proj.bias")),
                v_proj_weight: dummy.clone(),
                v_proj_bias: f32_weights.remove(&format!("{p}.self_attn.v_proj.bias")),
                o_proj_weight: dummy.clone(),
                gate_proj_weight: dummy.clone(),
                up_proj_weight: dummy.clone(),
                down_proj_weight: dummy.clone(),
            });

            debug!(layer = i, "loaded fp8 layer weights");
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

        // Init KV cache
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
            "FP8 model weights loaded to GPU successfully"
        );
        Ok(())
    }

    /// Load GPTQ INT4 model weights: packed int4 qweights + scales + zeros.
    fn load_weights_gptq(&mut self, model_path: &Path) -> Result<()> {
        info!(device_id = self.device_id, "loading GPTQ INT4 weights");

        let result =
            rvllm_model_loader::gpu_loader::load_weights_to_gpu_gptq(model_path, &self.device)
                .map_err(|e| LLMError::GpuError(format!("gptq weight loading failed: {e}")))?;

        info!(
            "loaded {} qweights + {} scales + {} zeros + {} f32 (group_size={})",
            result.qweights.len(), result.scales.len(), result.zeros.len(),
            result.f32_weights.len(), result.group_size
        );

        self.gptq_group_size = result.group_size;

        #[cfg(feature = "cuda")]
        {
            self.raw_weight_map = Some(result.f32_weights.clone());
            // Store GPTQ qweights in fp8 map (both are CudaSlice<u8>)
            self.raw_weight_map_fp8 = Some(result.qweights);
            // Store GPTQ scales in weight_scales map (both are CudaSlice<f32>)
            self.raw_weight_scales = Some(result.scales);
            // Store GPTQ zeros
            self.raw_weight_zeros = Some(result.zeros);
        }

        // Build the local weight structure from f32 weights
        let mut f32_weights = result.f32_weights;

        let tie = !f32_weights.contains_key("lm_head.weight");
        info!(tie_word_embeddings = tie, "building model weight structure (gptq)");

        let embed_tokens = f32_weights
            .remove("model.embed_tokens.weight")
            .ok_or_else(|| LLMError::ModelError("missing model.embed_tokens.weight".into()))?;

        let norm_weight = f32_weights
            .remove("model.norm.weight")
            .ok_or_else(|| LLMError::ModelError("missing model.norm.weight".into()))?;

        let lm_head_weight = f32_weights.remove("lm_head.weight");

        let num_layers = self.config.num_layers;
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let p = format!("model.layers.{}", i);
            let input_ln = f32_weights
                .remove(&format!("{p}.input_layernorm.weight"))
                .ok_or_else(|| LLMError::ModelError(format!("missing {p}.input_layernorm.weight")))?;
            let post_ln = f32_weights
                .remove(&format!("{p}.post_attention_layernorm.weight"))
                .ok_or_else(|| LLMError::ModelError(format!("missing {p}.post_attention_layernorm.weight")))?;

            let dummy = self.device
                .alloc_zeros::<f32>(1)
                .map_err(|e| LLMError::GpuError(format!("dummy alloc: {e}")))?;

            layers.push(LayerWeights {
                input_layernorm: input_ln,
                post_attention_layernorm: post_ln,
                q_proj_weight: dummy.clone(),
                q_proj_bias: f32_weights.remove(&format!("{p}.self_attn.q_proj.bias")),
                k_proj_weight: dummy.clone(),
                k_proj_bias: f32_weights.remove(&format!("{p}.self_attn.k_proj.bias")),
                v_proj_weight: dummy.clone(),
                v_proj_bias: f32_weights.remove(&format!("{p}.self_attn.v_proj.bias")),
                o_proj_weight: dummy.clone(),
                gate_proj_weight: dummy.clone(),
                up_proj_weight: dummy.clone(),
                down_proj_weight: dummy.clone(),
            });

            debug!(layer = i, "loaded gptq layer weights");
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
        info!(head_dim = self.config.head_dim, max_seq_len, rope_theta, "RoPE tables initialized");

        // Init KV cache
        self.kv_cache = Some(KVCache::new(
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
        ));
        info!("KV cache initialized");

        info!(device_id = self.device_id, "GPTQ model weights loaded to GPU successfully");
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

            // Insert FP8 weights and scales for FP8 inference path
            if let Some(fp8_map) = self.raw_weight_map_fp8.take() {
                for (name, slice) in fp8_map {
                    loader_weights.insert_fp8(name, slice, vec![]);
                }
                info!("inserted fp8 weights into model weight container");
            }
            if let Some(scale_map) = self.raw_weight_scales.take() {
                for (name, slice) in scale_map {
                    loader_weights.insert_scale(name, slice);
                }
                info!("inserted weight scales into model weight container");
            }
            if let Some(zeros_map) = self.raw_weight_zeros.take() {
                for (name, slice) in zeros_map {
                    loader_weights.insert_zeros(name, slice);
                }
                info!("inserted GPTQ zero points into model weight container");
            }

            let block_size = self.config.block_size;
            let cache = rvllm_kv_cache::engine_cuda::CudaCacheEngine::new(
                self.config.num_layers,
                self.config.num_kv_heads,
                self.config.head_dim,
                block_size,
                num_gpu_blocks,
                num_cpu_blocks,
                self.device.clone(),
            )
            .map_err(|e| LLMError::GpuError(format!("CudaCacheEngine init: {e}")))?;

            let runner_blas = CublasHandle::new(self.device.clone())
                .map_err(|e| LLMError::GpuError(format!("runner cublas: {e}")))?;

            // KernelLoader: try build output dir, fall back to empty (PTX loaded at runtime)
            let ptx_dir = Self::find_ptx_dir();
            if ptx_dir.is_none() {
                warn!(
                    "no PTX directory found. Set RVLLM_PTX_DIR or RVLLM_KERNEL_DIR to the directory containing compiled .ptx files. \
                     Compile with: cd kernels && bash build.sh <arch>"
                );
            }
            let loader = rvllm_gpu::kernel_loader::KernelLoader::new(
                self.device.clone(),
                &ptx_dir.unwrap_or_else(|| std::path::PathBuf::from("/nonexistent")),
            )
            .map_err(|e| LLMError::GpuError(format!("kernel loader: {e}")))?;

<<<<<<< Updated upstream
            // JIT compile fused CuTE kernels for this model's specific dimensions.
            // Uses nvcc + CUTLASS headers. Cached to ~/.cache/rvllm/fusion/.
            Self::jit_compile_fused_kernels(&mut loader, &self.config)?;

            // Hard validation: all required kernels must be loaded. No silent fallbacks.
            loader.validate_required_kernels();

=======
>>>>>>> Stashed changes
            let mr_config = self.config.model_runner_config();
            let mut runner = rvllm_model_runner::gpu_runner::GpuModelRunner::new(
                loader_weights,
                cache,
                runner_blas,
                loader,
                mr_config,
                self.device.clone(),
            )?;

            if self.config.dtype.is_gptq() {
                runner.enable_gptq(self.gptq_group_size);
                info!(group_size = self.gptq_group_size, "GPTQ INT4 inference enabled");
            } else if self.config.dtype.is_fp8() {
                runner.enable_fp8();
                info!("FP8 inference enabled (dequant u8->f16 + hgemm path)");
            } else if self.config.dtype.is_half() {
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

    /// Search for compiled PTX directory from build output or environment.
    #[cfg(feature = "cuda")]
    fn find_ptx_dir() -> Option<std::path::PathBuf> {
        // 1. Explicit env vars (highest priority)
        for var in &["RVLLM_PTX_DIR", "RVLLM_KERNEL_DIR"] {
            if let Ok(dir) = std::env::var(var) {
                let p = std::path::PathBuf::from(&dir);
                if p.exists() && p.is_dir() {
                    tracing::info!(env = var, dir = %p.display(), "found PTX directory via env var");
                    return Some(p);
                }
            }
        }

        // 2. Cargo build output directories
        // Also check CARGO_TARGET_DIR if set
        let mut search_dirs = vec![
            "./target/release/build".to_string(),
            "./target/debug/build".to_string(),
            "/root/vllm-rs/target/release/build".to_string(),
            "/root/vllm-rs/target/debug/build".to_string(),
        ];
        if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
            search_dirs.push(format!("{}/release/build", target_dir));
            search_dirs.push(format!("{}/debug/build", target_dir));
        }
        // Also check common user target dirs
        if let Ok(home) = std::env::var("HOME") {
            search_dirs.push(format!("{}/rvllm-target/release/build", home));
        }
        for base in &search_dirs {
            if let Ok(entries) = std::fs::read_dir(base) {
                for entry in entries.flatten() {
                    let ptx_path = entry.path().join("out/ptx");
                    if ptx_path.join("rotary_embedding.ptx").exists() {
                        return Some(ptx_path);
                    }
                }
            }
        }

        // 3. Common kernel directory locations relative to binary
        if let Ok(exe) = std::env::current_exe() {
            if let Some(bin_dir) = exe.parent() {
                // Check sibling `kernels/` directory (e.g., target/release/kernels/)
                let sibling = bin_dir.join("kernels");
                if sibling.join("rotary_embedding.ptx").exists() {
                    return Some(sibling);
                }
            }
        }

        None
    }

    /// Profile available GPU memory and return (num_gpu_blocks, num_cpu_blocks).
    pub fn profile_num_available_blocks(
        &self,
        gpu_memory_utilization: f32,
    ) -> Result<(usize, usize)> {
        let (free, _total) = cudarc::driver::result::mem_get_info()
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
        trace!(greedy_only, "gpu_worker: entering gpu_forward");
        let fwd_output = self.gpu_forward_ex(&model_input, greedy_only)?;

        let outputs = match fwd_output {
            ForwardOutput::TokenIds(ref token_ids) => {
                trace!(
                    num_ids = token_ids.len(),
                    "gpu_worker: gpu argmax fast path"
                );
                self.sample_tokens_from_gpu_argmax(token_ids, metadata)?
            }
            ForwardOutput::Logits(ref logits) => {
                trace!(
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
            let (padded_input, actual_batch) = self.graph_runner.pad_input(model_input)?;
            let replayed = self
                .graph_runner
                .try_replay(&self.compute_stream, actual_batch)?;
            if replayed {
                debug!(actual_batch, "CUDA graph replayed, unpadding logits");
                let padded_logits = self.raw_gpu_forward(&padded_input)?;
                return Ok(self.graph_runner.unpad_logits(&padded_logits, actual_batch));
            }
        }

        // Normal forward pass.
        let logits = self.raw_gpu_forward(model_input)?;

        // After warmup, capture a graph for this decode batch size if not yet captured.
        if self.forward_count > Self::GRAPH_WARMUP_CALLS
            && !model_input.is_prefill
            && self.graph_runner.can_use_graph(model_input)
        {
            let (padded_input, _actual_batch) = self.graph_runner.pad_input(model_input)?;
            let padded_bs = padded_input.num_tokens();
            if !self.graph_runner.was_capture_attempted(padded_bs) {
                info!(padded_bs, "capturing CUDA graph after warmup");
                // Split borrow: take references to disjoint fields so the closure
                // doesn't conflict with &mut graph_runner.
                #[cfg(feature = "cuda")]
                {
                    let gpu_runner = self.gpu_model_runner.as_ref();
                    let capture_result =
                        self.graph_runner
                            .capture_graph(&self.compute_stream, padded_bs, || {
                                let runner = gpu_runner.ok_or_else(|| {
                                    LLMError::GpuError("GPU model runner not initialized".into())
                                })?;
                                let _ = runner.forward(
                                    &padded_input.token_ids,
                                    &padded_input.position_ids,
                                    &padded_input.attention_metadata,
                                    padded_input.is_prefill,
                                )?;
                                Ok(())
                            });
                    if let Err(e) = capture_result {
                        warn!(%e, "CUDA graph capture failed, continuing without graph");
                    }
                }
            }
        }

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

        // Try graph replay for eligible decode steps -- graph path always
        // returns full logits since it was captured with the standard forward.
        if self.graph_runner.can_use_graph(model_input) {
            let (padded_input, actual_batch) = self.graph_runner.pad_input(model_input)?;
            let replayed = self
                .graph_runner
                .try_replay(&self.compute_stream, actual_batch)?;
            if replayed {
                debug!(actual_batch, "CUDA graph replayed, unpadding logits");
                let padded_logits = self.raw_gpu_forward(&padded_input)?;
                let logits = self.graph_runner.unpad_logits(&padded_logits, actual_batch);
                return Ok(ForwardOutput::Logits(logits));
            }
        }

        // Normal forward pass with optional greedy argmax on GPU.
        let output = self.raw_gpu_forward_ex(model_input, greedy_only)?;

        // After warmup, capture a graph for this decode batch size if not yet captured.
        // Graph capture always uses the standard (non-greedy) forward.
        if self.forward_count > Self::GRAPH_WARMUP_CALLS
            && !model_input.is_prefill
            && self.graph_runner.can_use_graph(model_input)
        {
            let (padded_input, _actual_batch) = self.graph_runner.pad_input(model_input)?;
            let padded_bs = padded_input.num_tokens();
            if !self.graph_runner.was_capture_attempted(padded_bs) {
                info!(padded_bs, "capturing CUDA graph after warmup");
                #[cfg(feature = "cuda")]
                {
                    let gpu_runner = self.gpu_model_runner.as_ref();
                    let capture_result =
                        self.graph_runner
                            .capture_graph(&self.compute_stream, padded_bs, || {
                                let runner = gpu_runner.ok_or_else(|| {
                                    LLMError::GpuError("GPU model runner not initialized".into())
                                })?;
                                let _ = runner.forward(
                                    &padded_input.token_ids,
                                    &padded_input.position_ids,
                                    &padded_input.attention_metadata,
                                    padded_input.is_prefill,
                                )?;
                                Ok(())
                            });
                    if let Err(e) = capture_result {
                        warn!(%e, "CUDA graph capture failed, continuing without graph");
                    }
                }
            }
        }

        Ok(output)
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
            .device
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

        let logits = self.device.dtoh_sync_copy(&logits_gpu).map_err(gpu_err)?;

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

        // Try GPU gather kernel first
        if let Some(kernel) = self
            .device
            .get_func("embedding_gather", "embedding_gather_kernel")
        {
            let output = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(gpu_err)?;
            let ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
            let ids_gpu = self.device.htod_sync_copy(&ids_i32).map_err(gpu_err)?;

            let block_dim = hidden_size.min(1024) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let vocab_size = self.vocab_size as i32;
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            &output,
                            embed_weight,
                            &ids_gpu,
                            hidden_size as i32,
                            vocab_size,
                        ),
                    )
                    .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
            }
            return Ok(output);
        }

        // Fallback: CPU gather
        let embed_host = self.device.dtoh_sync_copy(embed_weight).map_err(gpu_err)?;
        let mut result = vec![0.0f32; output_len];
        for (i, &tid) in token_ids.iter().enumerate() {
            let tid = tid as usize;
            if tid * hidden_size + hidden_size <= embed_host.len() {
                let src = &embed_host[tid * hidden_size..(tid + 1) * hidden_size];
                result[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(src);
            }
        }
        self.device.htod_sync_copy(&result).map_err(gpu_err)
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
            .device
            .alloc_zeros::<f32>(num_tokens * q_dim)
            .map_err(gpu_err)?;
        let mut k_gpu = self
            .device
            .alloc_zeros::<f32>(num_tokens * kv_dim)
            .map_err(gpu_err)?;
        let mut v_gpu = self
            .device
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
        let mut q_host = self.device.dtoh_sync_copy(&q_gpu).map_err(gpu_err)?;
        let mut k_host = self.device.dtoh_sync_copy(&k_gpu).map_err(gpu_err)?;
        let v_host = self.device.dtoh_sync_copy(&v_gpu).map_err(gpu_err)?;

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

        let attn_out_gpu = self.device.htod_sync_copy(&attn_output).map_err(gpu_err)?;

        // Need to re-borrow weights after mutable kv_cache borrow is done
        let weights = self.model_weights.as_ref().unwrap();
        let layer = &weights.layers[layer_idx];

        // 7. Output projection
        let mut attn_proj = self
            .device
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
            .device
            .alloc_zeros::<f32>(num_tokens * intermediate_size)
            .map_err(gpu_err)?;
        let mut up = self
            .device
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
            .device
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

        if let Some(kernel) = self.device.get_func("rms_norm", "rms_norm_kernel") {
            let output = self.device.alloc_zeros::<f32>(n).map_err(gpu_err)?;
            let block_dim = hidden_size.min(1024) as u32;
            let shared_mem = block_dim * std::mem::size_of::<f32>() as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared_mem,
            };
            unsafe {
                kernel
                    .launch(cfg, (&output, hidden, weight, eps, hidden_size as i32))
                    .map_err(|e| LLMError::GpuError(format!("rms_norm launch: {e}")))?;
            }
            return Ok(output);
        }

        // Fallback: CPU RMS norm
        let h = self.device.dtoh_sync_copy(hidden).map_err(gpu_err)?;
        let w = self.device.dtoh_sync_copy(weight).map_err(gpu_err)?;
        let mut out = vec![0.0f32; n];
        for t in 0..num_tokens {
            let row = &h[t * hidden_size..(t + 1) * hidden_size];
            let ss: f32 = row.iter().map(|x| x * x).sum();
            let inv_rms = (ss / hidden_size as f32 + eps).sqrt().recip();
            for d in 0..hidden_size {
                out[t * hidden_size + d] = row[d] * inv_rms * w.get(d).copied().unwrap_or(1.0);
            }
        }
        self.device.htod_sync_copy(&out).map_err(gpu_err)
    }

    /// Add bias to a tensor via GPU kernel (no CPU round-trip when kernel is loaded).
    fn add_bias_gpu(
        &self,
        tensor: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        num_tokens: usize,
        dim: usize,
    ) -> Result<()> {
        if let Some(kernel) = self.device.get_func("add_bias", "add_bias_kernel") {
            let block_dim = dim.min(1024) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernel
                    .launch(cfg, (tensor, bias, dim as i32))
                    .map_err(|e| LLMError::GpuError(format!("add_bias launch: {e}")))?;
            }
            return Ok(());
        }

        // Fallback: CPU bias add
        let mut data = self.device.dtoh_sync_copy(tensor).map_err(gpu_err)?;
        let b = self.device.dtoh_sync_copy(bias).map_err(gpu_err)?;
        for t in 0..num_tokens {
            for d in 0..dim.min(b.len()) {
                data[t * dim + d] += b[d];
            }
        }
        *tensor = self.device.htod_sync_copy(&data).map_err(gpu_err)?;
        Ok(())
    }

    /// Element-wise add two tensors via GPU kernel (no CPU round-trip when kernel is loaded).
    fn add_tensors_gpu(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<CudaSlice<f32>> {
        if let Some(kernel) = self.device.get_func("add_bias", "add_kernel") {
            let output = self.device.alloc_zeros::<f32>(len).map_err(gpu_err)?;
            let threads = 256u32;
            let blocks = (len as u32 + threads - 1) / threads;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernel
                    .launch(cfg, (&output, a, b, len as i32))
                    .map_err(|e| LLMError::GpuError(format!("add_kernel launch: {e}")))?;
            }
            return Ok(output);
        }

        // Fallback: CPU add
        let a_h = self.device.dtoh_sync_copy(a).map_err(gpu_err)?;
        let b_h = self.device.dtoh_sync_copy(b).map_err(gpu_err)?;
        let out: Vec<f32> = a_h.iter().zip(b_h.iter()).map(|(x, y)| x + y).collect();
        self.device.htod_sync_copy(&out).map_err(gpu_err)
    }

    /// Fused SiLU(gate) * up via GPU kernel (no CPU round-trip when kernel is loaded).
    fn fused_silu_mul_gpu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        len: usize,
    ) -> Result<CudaSlice<f32>> {
        if let Some(kernel) = self.device.get_func("activation", "fused_silu_mul_kernel") {
            let output = self.device.alloc_zeros::<f32>(len).map_err(gpu_err)?;
            let threads = 256u32;
            let blocks = (len as u32 + threads - 1) / threads;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernel
                    .launch(cfg, (&output, gate, up, len as i32))
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul launch: {e}")))?;
            }
            return Ok(output);
        }

        // Fallback: CPU fused SiLU * mul
        let g = self.device.dtoh_sync_copy(gate).map_err(gpu_err)?;
        let u = self.device.dtoh_sync_copy(up).map_err(gpu_err)?;
        let out: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(&gv, &uv)| (gv / (1.0 + (-gv).exp())) * uv)
            .collect();
        self.device.htod_sync_copy(&out).map_err(gpu_err)
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

        if let Some(kernel) = self
            .device
            .get_func("fused_residual_rmsnorm", "fused_residual_rmsnorm_kernel")
        {
            let output = self.device.alloc_zeros::<f32>(n).map_err(gpu_err)?;
            let residual = self.device.alloc_zeros::<f32>(n).map_err(gpu_err)?;
            let block_dim = hidden_size.min(1024) as u32;
            let shared_mem = block_dim * std::mem::size_of::<f32>() as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared_mem,
            };
            let eps = self.rms_norm_eps;
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            &output,
                            &residual,
                            input,
                            add,
                            weight,
                            eps,
                            hidden_size as i32,
                        ),
                    )
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_residual_rmsnorm launch: {e}"))
                    })?;
            }
            return Ok((output, residual));
        }

        // Fallback: separate add + norm
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
    pub fn gpu(&self) -> &Arc<CudaDevice> {
        &self.device
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
        layer_types: vec![],
        has_attn_output_gate: false,
        rms_norm_eps: 1e-5,
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
