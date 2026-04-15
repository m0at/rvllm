use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, LaunchConfig, PushKernelArg};
use half::f16;

use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;
use rvllm_gpu::cublaslt_ops::CublasLtOps;
use rvllm_gpu::cutlass_autotune::{CutlassAutotuneCache, max_workspace_size};
use rvllm_gpu::cutlass_ffi::CutlassKernels;
use rvllm_gpu::fa3_ffi::Fa3Kernels;
use rvllm_gpu::kernel_loader::KernelLoader;
use rvllm_gpu::pinned_memory::PinnedBuffer;

use crate::kv_cache::CudaKVCache;
use crate::layer::{AttentionMeta, F16LayerScratch, Fp8LayerWeightRefs, GemmStrategy, GpuTransformerLayer, LayerWeights};
use crate::types::GpuBatchInput;

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub block_size: usize,
    pub max_seq_len: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
}

struct ModelWeightsStore {
    #[allow(dead_code)]
    fused_qkv: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    fused_gate_up: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    o_proj: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    down_proj: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    input_layernorm: Vec<CudaSlice<f16>>,
    #[allow(dead_code)]
    post_attn_layernorm: Vec<CudaSlice<f16>>,
    // FP8 quantized weights (empty if FP8 disabled)
    fp8_qkv: Vec<CudaSlice<u8>>,
    fp8_qkv_scales: Vec<CudaSlice<f32>>,
    fp8_o_proj: Vec<CudaSlice<u8>>,
    fp8_o_proj_scales: Vec<CudaSlice<f32>>,
    fp8_gate_up: Vec<CudaSlice<u8>>,
    fp8_gate_up_scales: Vec<CudaSlice<f32>>,
    fp8_down: Vec<CudaSlice<u8>>,
    fp8_down_scales: Vec<CudaSlice<f32>>,
    fp8_enabled: bool,
}

/// Offsets into the packed metadata GPU buffer (i32 elements).
#[derive(Clone, Copy)]
struct PackedMetaOffsets {
    token_ids: usize,
    num_token_ids: usize,
    positions: usize,
    num_positions: usize,
    context_lens: usize,
    num_context_lens: usize,
    block_tables: usize,
    num_block_tables: usize,
    slot_mapping: usize,
    num_slot_mapping: usize,
    seq_start_pos: usize,
    num_seq_start_pos: usize,
}

pub struct GpuModelRunner {
    config: RunnerConfig,
    layers: Vec<GpuTransformerLayer>,
    gemm_strategy: GemmStrategy,
    cutlass: Option<Arc<CutlassKernels>>,
    fa3: Option<Arc<Fa3Kernels>>,
    autotune: Option<CutlassAutotuneCache>,
    cublas: CublasHandle,
    lt_ops: Option<CublasLtOps>,
    stream: Arc<CudaStream>,
    loader: Arc<KernelLoader>,
    weights: ModelWeightsStore,
    embed_tokens: CudaSlice<f16>,
    lm_head_weight: CudaSlice<f16>,
    final_norm_weight: CudaSlice<f16>,
    rope_cos: CudaSlice<f32>,
    rope_sin: CudaSlice<f32>,
    // Reusable CPU scratch for packing metadata
    cpu_scratch: Vec<i32>,
    // Reusable GPU packed metadata buffer
    meta_packed: CudaSlice<i32>,
    // Max blocks per seq for block table padding
    graph_max_blocks: usize,
    // Pre-allocated reusable GPU scratch (sized for max_num_seqs tokens)
    scratch: F16LayerScratch,
    residual_a: CudaSlice<f16>,
    residual_b: CudaSlice<f16>,
    down_a: CudaSlice<f16>,
    down_b: CudaSlice<f16>,
    final_normed: CudaSlice<f16>,
    residual_tmp: CudaSlice<f16>,
    logits_gpu: CudaSlice<f32>,
    embed_output: CudaSlice<f16>,
    argmax_output: CudaSlice<i32>,
    // Pinned host buffers for truly async DtoH/HtoD (pageable memory degrades to sync)
    pinned_argmax: PinnedBuffer<i32>,
    pinned_meta: PinnedBuffer<i32>,
    // Stored metadata offsets from the last upload (for forward_gpu_only)
    last_meta_offsets: Option<PackedMetaOffsets>,
}

impl GpuModelRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: RunnerConfig,
        layers: Vec<GpuTransformerLayer>,
        cutlass: Option<Arc<CutlassKernels>>,
        fa3: Option<Arc<Fa3Kernels>>,
        autotune: Option<CutlassAutotuneCache>,
        cublas: CublasHandle,
        lt_ops: Option<CublasLtOps>,
        stream: Arc<CudaStream>,
        loader: Arc<KernelLoader>,
        embed_tokens: CudaSlice<f16>,
        lm_head_weight: CudaSlice<f16>,
        final_norm_weight: CudaSlice<f16>,
        fused_qkv_weights: Vec<CudaSlice<f16>>,
        fused_gate_up_weights: Vec<CudaSlice<f16>>,
        o_proj_weights: Vec<CudaSlice<f16>>,
        down_proj_weights: Vec<CudaSlice<f16>>,
        input_layernorm_weights: Vec<CudaSlice<f16>>,
        post_attn_layernorm_weights: Vec<CudaSlice<f16>>,
        rope_cos: CudaSlice<f32>,
        rope_sin: CudaSlice<f32>,
    ) -> Result<Self> {
        // cuBLAS auto-tunes tile selection per problem size, beating our
        // fixed CUTLASS tiles at small M (decode).  Hybrid's fused gateup+silu
        // CUTLASS kernel uses 136us vs cuBLAS's 119us at M=32 on H100.
        let gemm_strategy = GemmStrategy::Cublas;

        let graph_max_blocks = config.max_seq_len / config.block_size + 1;

        // Pre-allocate packed metadata buffer for max capacity
        // Worst case: 256 seqs * (max_seq_len tokens + metadata)
        let max_meta_elems = 256 * (config.max_seq_len + graph_max_blocks + 10);
        let meta_packed = stream
            .alloc_zeros::<i32>(max_meta_elems.max(4096))
            .map_err(|e| LLMError::GpuError(format!("meta_packed alloc: {e}")))?;

        // Pre-allocate reusable GPU scratch for max tokens per step
        let max_t = config.max_num_batched_tokens.max(config.max_num_seqs);
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let layer_config = layers[0].config_ref();
        let cutlass_ws_bytes = if let Some(ref ck) = cutlass {
            let h = config.hidden_size;
            let qkv = (config.num_heads + 2 * config.num_kv_heads) * config.head_dim;
            let gate_up = config.intermediate_size * 2;
            let inter = config.intermediate_size;
            let hgemm_shapes = [(max_t, qkv, h), (max_t, h, qkv), (max_t, gate_up, h), (max_t, h, inter), (max_t, config.vocab_size, h)];
            let oproj_shapes = [(max_t, h, h), (max_t, h, inter)];
            let gateup_shapes = [(max_t, gate_up, h)];
            let fp8_shapes = [(max_t, qkv, h), (max_t, h, qkv), (max_t, gate_up, h), (max_t, h, inter)];
            max_workspace_size(ck, &hgemm_shapes, &oproj_shapes, &gateup_shapes, &fp8_shapes)
        } else {
            4 * 1024 * 1024
        };
        let scratch = F16LayerScratch::alloc(&stream, layer_config, max_t, cutlass_ws_bytes)?;
        let max_n = max_t * hidden_size;
        let alloc_f16 = |label: &str, count: usize| -> Result<CudaSlice<f16>> {
            stream.alloc_zeros::<f16>(count)
                .map_err(|e| LLMError::GpuError(format!("{label} alloc: {e}")))
        };
        let residual_a = alloc_f16("residual_a", max_n)?;
        let residual_b = alloc_f16("residual_b", max_n)?;
        let down_a = alloc_f16("down_a", max_n)?;
        let down_b = alloc_f16("down_b", max_n)?;
        let final_normed = alloc_f16("final_normed", max_n)?;
        let residual_tmp = alloc_f16("residual_tmp", max_n)?;
        let logits_gpu = stream.alloc_zeros::<f32>(max_t * vocab_size)
            .map_err(|e| LLMError::GpuError(format!("logits alloc: {e}")))?;
        let embed_output = alloc_f16("embed_output", max_n)?;
        let argmax_output = stream.alloc_zeros::<i32>(max_t)
            .map_err(|e| LLMError::GpuError(format!("argmax_output alloc: {e}")))?;

        let pinned_argmax = PinnedBuffer::<i32>::new(max_t)?;
        let pinned_meta = PinnedBuffer::<i32>::new(max_meta_elems.max(4096))?;

        Ok(Self {
            config,
            layers,
            gemm_strategy,
            cutlass,
            fa3,
            autotune,
            cublas,
            lt_ops,
            stream,
            loader,
            weights: ModelWeightsStore {
                fused_qkv: fused_qkv_weights,
                fused_gate_up: fused_gate_up_weights,
                o_proj: o_proj_weights,
                down_proj: down_proj_weights,
                input_layernorm: input_layernorm_weights,
                post_attn_layernorm: post_attn_layernorm_weights,
                fp8_qkv: Vec::new(),
                fp8_qkv_scales: Vec::new(),
                fp8_o_proj: Vec::new(),
                fp8_o_proj_scales: Vec::new(),
                fp8_gate_up: Vec::new(),
                fp8_gate_up_scales: Vec::new(),
                fp8_down: Vec::new(),
                fp8_down_scales: Vec::new(),
                fp8_enabled: false,
            },
            embed_tokens,
            lm_head_weight,
            final_norm_weight,
            rope_cos,
            rope_sin,
            cpu_scratch: Vec::with_capacity(65536),
            meta_packed,
            graph_max_blocks,
            scratch,
            residual_a,
            residual_b,
            down_a,
            down_b,
            final_normed,
            residual_tmp,
            logits_gpu,
            embed_output,
            argmax_output,
            pinned_argmax,
            pinned_meta,
            last_meta_offsets: None,
        })
    }

    pub fn gemm_strategy(&self) -> GemmStrategy {
        self.gemm_strategy
    }

    /// Quantize all layer weights to FP8 E4M3 per-tensor and upload to GPU.
    /// Halves weight bandwidth for decode GEMMs. Call once after model load.
    pub fn enable_fp8_weights(&mut self) -> Result<()> {
        use rvllm_gpu::fp8_quantize::quantize_weight_fp8_per_tensor;

        let num_layers = self.config.num_layers;
        tracing::info!(num_layers, "quantizing weights to FP8 E4M3 per-tensor");

        let mut fp8_qkv = Vec::with_capacity(num_layers);
        let mut fp8_qkv_scales = Vec::with_capacity(num_layers);
        let mut fp8_o_proj = Vec::with_capacity(num_layers);
        let mut fp8_o_proj_scales = Vec::with_capacity(num_layers);
        let mut fp8_gate_up = Vec::with_capacity(num_layers);
        let mut fp8_gate_up_scales = Vec::with_capacity(num_layers);
        let mut fp8_down = Vec::with_capacity(num_layers);
        let mut fp8_down_scales = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            // Read f16 weights from GPU to CPU for quantization
            let quantize_and_upload = |weight: &CudaSlice<f16>, out_dim: usize, in_dim: usize, name: &str|
                -> Result<(CudaSlice<u8>, CudaSlice<f32>)>
            {
                let cpu_f16: Vec<f16> = self.stream.clone_dtoh(weight)
                    .map_err(|e| LLMError::GpuError(format!("{name} DtoH: {e}")))?;
                let q = quantize_weight_fp8_per_tensor(&cpu_f16, out_dim, in_dim);
                let gpu_fp8 = self.stream.clone_htod(&q.data)
                    .map_err(|e| LLMError::GpuError(format!("{name} fp8 HtoD: {e}")))?;
                let gpu_scale = self.stream.clone_htod(&[q.scale])
                    .map_err(|e| LLMError::GpuError(format!("{name} scale HtoD: {e}")))?;
                tracing::debug!(layer_idx, name, out_dim, in_dim, scale = q.scale, "FP8 quantized");
                Ok((gpu_fp8, gpu_scale))
            };

            let cfg = self.layers[layer_idx].config_ref();
            let hidden = cfg.hidden_size;
            let qkv_dim = cfg.qkv_dim();
            let q_dim = cfg.q_dim();
            let intermediate = cfg.intermediate_size;
            let gate_up_dim = cfg.gate_up_dim();

            let (qkv_fp8, qkv_s) = quantize_and_upload(
                &self.weights.fused_qkv[layer_idx], qkv_dim, hidden, "qkv")?;
            let (oproj_fp8, oproj_s) = quantize_and_upload(
                &self.weights.o_proj[layer_idx], hidden, q_dim, "o_proj")?;
            let (gateup_fp8, gateup_s) = quantize_and_upload(
                &self.weights.fused_gate_up[layer_idx], gate_up_dim, hidden, "gate_up")?;
            let (down_fp8, down_s) = quantize_and_upload(
                &self.weights.down_proj[layer_idx], hidden, intermediate, "down_proj")?;

            fp8_qkv.push(qkv_fp8);
            fp8_qkv_scales.push(qkv_s);
            fp8_o_proj.push(oproj_fp8);
            fp8_o_proj_scales.push(oproj_s);
            fp8_gate_up.push(gateup_fp8);
            fp8_gate_up_scales.push(gateup_s);
            fp8_down.push(down_fp8);
            fp8_down_scales.push(down_s);
        }

        self.weights.fp8_qkv = fp8_qkv;
        self.weights.fp8_qkv_scales = fp8_qkv_scales;
        self.weights.fp8_o_proj = fp8_o_proj;
        self.weights.fp8_o_proj_scales = fp8_o_proj_scales;
        self.weights.fp8_gate_up = fp8_gate_up;
        self.weights.fp8_gate_up_scales = fp8_gate_up_scales;
        self.weights.fp8_down = fp8_down;
        self.weights.fp8_down_scales = fp8_down_scales;
        self.weights.fp8_enabled = true;

        tracing::info!(num_layers, "FP8 weight quantization complete");
        Ok(())
    }

    pub fn forward(&mut self, input: &GpuBatchInput, kv_cache: &CudaKVCache) -> Result<Vec<f32>> {
        let num_tokens = total_tokens(input);
        if num_tokens == 0 {
            return Ok(Vec::new());
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let num_seqs = input.num_seqs;
        let num_layers = self.config.num_layers;
        let gemm_strategy = self.gemm_strategy;

        // 1. Pack and upload metadata (single HtoD memcpy)
        let offsets = self.upload_metadata(input)?;

        // 2. Embedding lookup (into pre-allocated buffer)
        self.embedding_lookup(num_tokens, &offsets)?;

        // Destructure self to allow borrowing disjoint fields in the layer loop
        let Self {
            ref layers,
            ref cutlass,
            ref fa3,
            ref autotune,
            ref cublas,
            ref lt_ops,
            ref stream,
            ref meta_packed,
            ref rope_cos,
            ref rope_sin,
            ref weights,
            ref final_norm_weight,
            ref lm_head_weight,
            ref mut scratch,
            ref mut residual_a,
            ref mut residual_b,
            ref mut down_a,
            ref mut down_b,
            ref mut final_normed,
            ref mut residual_tmp,
            ref mut logits_gpu,
            ref embed_output,
            ..
        } = *self;

        let cutlass_ref: Option<&CutlassKernels> = cutlass.as_deref();
        let fa3_ref: Option<&Fa3Kernels> = fa3.as_deref();
        let autotune_ref: Option<&CutlassAutotuneCache> = autotune.as_ref();

        // 4. Layer loop -- layers write directly to double-buffer targets (zero copies)
        let mut prev_fused = false;
        for layer_idx in 0..num_layers {
            let (key_cache, value_cache) = &kv_cache.gpu_cache[layer_idx];

            let attn = AttentionMeta {
                positions: meta_packed
                    .slice(offsets.positions..offsets.positions + offsets.num_positions),
                key_cache,
                value_cache,
                block_tables: meta_packed
                    .slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                context_lens: meta_packed
                    .slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                slot_mapping: meta_packed
                    .slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                seq_start_pos: meta_packed
                    .slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                num_tokens,
                num_seqs,
                max_context_len: input.max_context_len,
                is_prefill: !input.is_all_decode,
                rope_cos,
                rope_sin,
            };

            let fp8_refs = if weights.fp8_enabled {
                Some(Fp8LayerWeightRefs {
                    qkv_fp8: &weights.fp8_qkv[layer_idx],
                    qkv_scale: &weights.fp8_qkv_scales[layer_idx],
                    o_proj_fp8: &weights.fp8_o_proj[layer_idx],
                    o_proj_scale: &weights.fp8_o_proj_scales[layer_idx],
                    gate_up_fp8: &weights.fp8_gate_up[layer_idx],
                    gate_up_scale: &weights.fp8_gate_up_scales[layer_idx],
                    down_proj_fp8: &weights.fp8_down[layer_idx],
                    down_proj_scale: &weights.fp8_down_scales[layer_idx],
                })
            } else {
                None
            };

            let layer_weights = LayerWeights {
                qkv_weight: &weights.fused_qkv[layer_idx],
                o_proj_weight: &weights.o_proj[layer_idx],
                gate_up_weight: &weights.fused_gate_up[layer_idx],
                down_proj_weight: &weights.down_proj[layer_idx],
                input_layernorm_weight: &weights.input_layernorm[layer_idx],
                post_attention_layernorm_weight: &weights.post_attn_layernorm[layer_idx],
                fp8: fp8_refs,
            };

            let fused = if layer_idx == 0 {
                layers[layer_idx].forward_batched_v2(
                    embed_output, &attn, &layer_weights, scratch, None,
                    residual_a, down_a,
                    gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                )?
            } else if layer_idx % 2 == 1 {
                if prev_fused {
                    layers[layer_idx].forward_batched_v2(
                        &*down_a, &attn, &layer_weights, scratch, None,
                        residual_b, down_b,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                } else {
                    layers[layer_idx].forward_batched_v2(
                        &*residual_a, &attn, &layer_weights, scratch, Some(&*down_a),
                        residual_b, down_b,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                }
            } else {
                if prev_fused {
                    layers[layer_idx].forward_batched_v2(
                        &*down_b, &attn, &layer_weights, scratch, None,
                        residual_a, down_a,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                } else {
                    layers[layer_idx].forward_batched_v2(
                        &*residual_b, &attn, &layer_weights, scratch, Some(&*down_b),
                        residual_a, down_a,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                }
            };
            prev_fused = fused;
        }

        // Final output: last odd layer wrote to b, last even (>0) layer wrote to a
        let last_wrote_to_b = num_layers > 1 && (num_layers - 1) % 2 == 1;
        let (final_residual, final_down): (&CudaSlice<f16>, &CudaSlice<f16>) =
            if last_wrote_to_b {
                (residual_b, down_b)
            } else {
                (residual_a, down_a)
            };

        // 5. Final RMSNorm
        if prev_fused {
            layers[0].rms_norm_pub(
                final_down, final_norm_weight,
                num_tokens, hidden_size, final_normed,
            )?;
        } else {
            layers[0].fused_residual_rmsnorm_pub(
                final_residual, final_down, final_norm_weight,
                num_tokens, hidden_size, final_normed, residual_tmp,
            )?;
        }

        // 6. LM head: f16 hidden x f16 lm_head -> f32 logits
        cublas.hgemm_f32_output(
            num_tokens, vocab_size, hidden_size,
            1.0, &*final_normed, lm_head_weight, 0.0, logits_gpu,
        )?;

        // 7. DtoH logits
        let logits_cpu = stream
            .clone_dtoh(&logits_gpu.slice(..num_tokens * vocab_size))
            .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;

        Ok(logits_cpu)
    }

    /// Launch greedy forward (GPU-only, no DtoH, no sync).
    /// Runs embedding -> layers -> norm -> LM head -> GPU argmax into argmax_output.
    /// Call `read_graph_output()` afterward to collect results.
    pub fn forward_greedy_launch(&mut self, input: &GpuBatchInput, kv_cache: &CudaKVCache) -> Result<usize> {
        let num_tokens = total_tokens(input);
        if num_tokens == 0 {
            return Ok(0);
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let num_seqs = input.num_seqs;
        let num_layers = self.config.num_layers;
        let gemm_strategy = self.gemm_strategy;

        let offsets = self.upload_metadata(input)?;
        self.embedding_lookup(num_tokens, &offsets)?;

        let Self {
            ref layers,
            ref cutlass,
            ref fa3,
            ref autotune,
            ref cublas,
            ref lt_ops,
            ref stream,
            ref meta_packed,
            ref rope_cos,
            ref rope_sin,
            ref weights,
            ref final_norm_weight,
            ref lm_head_weight,
            ref mut scratch,
            ref mut residual_a,
            ref mut residual_b,
            ref mut down_a,
            ref mut down_b,
            ref mut final_normed,
            ref mut residual_tmp,
            ref mut logits_gpu,
            ref embed_output,
            ref loader,
            ref mut argmax_output,
            ..
        } = *self;

        let cutlass_ref: Option<&CutlassKernels> = cutlass.as_deref();
        let fa3_ref: Option<&Fa3Kernels> = fa3.as_deref();
        let autotune_ref: Option<&CutlassAutotuneCache> = autotune.as_ref();

        let mut prev_fused = false;
        for layer_idx in 0..num_layers {
            let (key_cache, value_cache) = &kv_cache.gpu_cache[layer_idx];
            let attn = AttentionMeta {
                positions: meta_packed
                    .slice(offsets.positions..offsets.positions + offsets.num_positions),
                key_cache,
                value_cache,
                block_tables: meta_packed
                    .slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                context_lens: meta_packed
                    .slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                slot_mapping: meta_packed
                    .slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                seq_start_pos: meta_packed
                    .slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                num_tokens,
                num_seqs,
                max_context_len: input.max_context_len,
                is_prefill: !input.is_all_decode,
                rope_cos,
                rope_sin,
            };
            let fp8_refs = if weights.fp8_enabled {
                Some(Fp8LayerWeightRefs {
                    qkv_fp8: &weights.fp8_qkv[layer_idx],
                    qkv_scale: &weights.fp8_qkv_scales[layer_idx],
                    o_proj_fp8: &weights.fp8_o_proj[layer_idx],
                    o_proj_scale: &weights.fp8_o_proj_scales[layer_idx],
                    gate_up_fp8: &weights.fp8_gate_up[layer_idx],
                    gate_up_scale: &weights.fp8_gate_up_scales[layer_idx],
                    down_proj_fp8: &weights.fp8_down[layer_idx],
                    down_proj_scale: &weights.fp8_down_scales[layer_idx],
                })
            } else {
                None
            };
            let layer_weights = LayerWeights {
                qkv_weight: &weights.fused_qkv[layer_idx],
                o_proj_weight: &weights.o_proj[layer_idx],
                gate_up_weight: &weights.fused_gate_up[layer_idx],
                down_proj_weight: &weights.down_proj[layer_idx],
                input_layernorm_weight: &weights.input_layernorm[layer_idx],
                post_attention_layernorm_weight: &weights.post_attn_layernorm[layer_idx],
                fp8: fp8_refs,
            };
            let fused = if layer_idx == 0 {
                layers[layer_idx].forward_batched_v2(
                    embed_output, &attn, &layer_weights, scratch, None,
                    residual_a, down_a,
                    gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                )?
            } else if layer_idx % 2 == 1 {
                if prev_fused {
                    layers[layer_idx].forward_batched_v2(
                        &*down_a, &attn, &layer_weights, scratch, None,
                        residual_b, down_b,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                } else {
                    layers[layer_idx].forward_batched_v2(
                        &*residual_a, &attn, &layer_weights, scratch, Some(&*down_a),
                        residual_b, down_b,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                }
            } else {
                if prev_fused {
                    layers[layer_idx].forward_batched_v2(
                        &*down_b, &attn, &layer_weights, scratch, None,
                        residual_a, down_a,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                } else {
                    layers[layer_idx].forward_batched_v2(
                        &*residual_b, &attn, &layer_weights, scratch, Some(&*down_b),
                        residual_a, down_a,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                }
            };
            prev_fused = fused;
        }

        let last_wrote_to_b = num_layers > 1 && (num_layers - 1) % 2 == 1;
        let (final_residual, final_down): (&CudaSlice<f16>, &CudaSlice<f16>) =
            if last_wrote_to_b {
                (residual_b, down_b)
            } else {
                (residual_a, down_a)
            };

        if prev_fused {
            layers[0].rms_norm_pub(
                final_down, final_norm_weight,
                num_tokens, hidden_size, final_normed,
            )?;
        } else {
            layers[0].fused_residual_rmsnorm_pub(
                final_residual, final_down, final_norm_weight,
                num_tokens, hidden_size, final_normed, residual_tmp,
            )?;
        }

        // LM head GEMM
        cublas.hgemm_f32_output(
            num_tokens, vocab_size, hidden_size,
            1.0, &*final_normed, lm_head_weight, 0.0, logits_gpu,
        )?;

        // GPU-side argmax (no full logits DtoH)
        let argmax_kernel = loader.get_func("argmax", "argmax_kernel")?;
        let block_dim = vocab_size.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&argmax_kernel)
                .arg(&*logits_gpu)
                .arg(&*argmax_output)
                .arg(&(vocab_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("argmax launch: {e}")))?;
        }

        Ok(num_tokens)
    }

    /// Fast greedy decode path: runs full forward, GPU argmax, returns only token IDs.
    /// Avoids the massive DtoH of full logits (N * vocab_size * 4 bytes).
    pub fn forward_greedy(&mut self, input: &GpuBatchInput, kv_cache: &CudaKVCache) -> Result<Vec<i32>> {
        let n = self.forward_greedy_launch(input, kv_cache)?;
        if n == 0 { return Ok(Vec::new()); }
        Ok(self.read_graph_output(n)?.to_vec())
    }

    /// Pack all metadata fields into one contiguous GPU buffer (1 memcpy).
    pub fn upload_metadata(&mut self, input: &GpuBatchInput) -> Result<PackedMetaOffsets> {
        let num_tokens = total_tokens(input);
        let num_seqs = input.num_seqs;
        let max_blocks = self.graph_max_blocks;

        self.cpu_scratch.clear();

        // token_ids
        let token_ids_off = self.cpu_scratch.len();
        if input.is_all_decode {
            for req_idx in 0..num_seqs {
                self.cpu_scratch.push(input.token_ids[req_idx] as i32);
            }
        } else {
            for &t in &input.prefill_tokens {
                self.cpu_scratch.push(t as i32);
            }
            for req_idx in input.num_prefill_seqs..num_seqs {
                self.cpu_scratch.push(input.token_ids[req_idx] as i32);
            }
        }
        let num_token_ids = self.cpu_scratch.len() - token_ids_off;

        // positions
        let positions_off = self.cpu_scratch.len();
        if input.is_all_decode {
            for &p in &input.position_ids {
                self.cpu_scratch.push(p as i32);
            }
        } else {
            for &p in &input.prefill_positions {
                self.cpu_scratch.push(p as i32);
            }
            for idx in input.num_prefill_seqs..num_seqs {
                self.cpu_scratch.push(input.position_ids[idx] as i32);
            }
        }
        let num_positions = self.cpu_scratch.len() - positions_off;

        // context_lens
        let context_lens_off = self.cpu_scratch.len();
        for &c in &input.context_lens {
            self.cpu_scratch.push(c as i32);
        }
        let num_context_lens = self.cpu_scratch.len() - context_lens_off;

        // block_tables: [num_seqs, max_blocks], zero-padded
        let block_tables_off = self.cpu_scratch.len();
        let bt_len = num_seqs * max_blocks;
        let old_len = self.cpu_scratch.len();
        self.cpu_scratch.resize(old_len + bt_len, 0i32);
        // Unpack the flat block_tables from GpuBatchInput
        let max_blocks_input = input.max_blocks_per_seq;
        for s in 0..num_seqs {
            let src_start = s * max_blocks_input;
            let copy_len = max_blocks_input.min(max_blocks);
            for b in 0..copy_len {
                if src_start + b < input.block_tables_flat.len() {
                    self.cpu_scratch[block_tables_off + s * max_blocks + b] =
                        input.block_tables_flat[src_start + b] as i32;
                }
            }
        }
        let num_block_tables = bt_len;

        // slot_mapping
        let slot_mapping_off = self.cpu_scratch.len();
        if input.is_all_decode {
            for &s in &input.slot_mapping {
                self.cpu_scratch.push(s as i32);
            }
        } else {
            for &s in &input.prefill_slot_mapping {
                self.cpu_scratch.push(s as i32);
            }
            for idx in input.num_prefill_seqs..num_seqs {
                self.cpu_scratch.push(input.slot_mapping[idx] as i32);
            }
        }
        let num_slot_mapping = self.cpu_scratch.len() - slot_mapping_off;

        // seq_start_pos: prefix sums of query_lens + total
        let seq_start_pos_off = self.cpu_scratch.len();
        let mut pos = 0i32;
        for &ql in &input.query_lens {
            self.cpu_scratch.push(pos);
            pos += ql as i32;
        }
        self.cpu_scratch.push(num_tokens as i32);
        let num_seq_start_pos = self.cpu_scratch.len() - seq_start_pos_off;

        // Single packed upload via pinned staging buffer (pageable HtoD is synchronous)
        let total_elems = self.cpu_scratch.len();
        if total_elems > self.meta_packed.len() {
            self.meta_packed = self.stream
                .alloc_zeros::<i32>(total_elems * 2)
                .map_err(|e| LLMError::GpuError(format!("meta_packed realloc: {e}")))?;
        }
        if total_elems > self.pinned_meta.len() {
            self.pinned_meta = PinnedBuffer::<i32>::new(total_elems * 2)?;
        }

        self.pinned_meta.as_mut_slice()[..total_elems]
            .copy_from_slice(&self.cpu_scratch[..total_elems]);
        let dst_dev_ptr = {
            let (ptr, _) = self.meta_packed.device_ptr(&self.stream);
            ptr
        };
        let bytes = total_elems * std::mem::size_of::<i32>();
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst_dev_ptr,
                self.pinned_meta.as_ptr() as *const std::ffi::c_void,
                bytes,
                self.stream.cu_stream(),
            )
            .result()
            .map_err(|e| LLMError::GpuError(format!("pinned metadata HtoD: {e}")))?;
        }

        let offsets = PackedMetaOffsets {
            token_ids: token_ids_off,
            num_token_ids,
            positions: positions_off,
            num_positions,
            context_lens: context_lens_off,
            num_context_lens,
            block_tables: block_tables_off,
            num_block_tables,
            slot_mapping: slot_mapping_off,
            num_slot_mapping,
            seq_start_pos: seq_start_pos_off,
            num_seq_start_pos,
        };
        self.last_meta_offsets = Some(offsets);
        Ok(offsets)
    }

    /// Embedding gather: token_ids -> f16 hidden states via custom kernel.
    fn embedding_lookup(
        &mut self,
        num_tokens: usize,
        offsets: &PackedMetaOffsets,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;

        let kernel = self
            .loader
            .get_func("embedding_gather_f16", "embedding_gather_f16_kernel")?;

        let token_ids_view: CudaView<'_, i32> = self
            .meta_packed
            .slice(offsets.token_ids..offsets.token_ids + offsets.num_token_ids);

        let block_dim = hidden_size.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&self.embed_output)
                .arg(&self.embed_tokens)
                .arg(&token_ids_view)
                .arg(&(hidden_size as i32))
                .arg(&(self.config.vocab_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
        }

        Ok(())
    }

    // =================================================================
    // CUDA graph support
    // =================================================================

    /// Accessor for the runner's CUDA stream.
    pub fn cuda_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Pre-allocate cuBLAS workspace for CUDA graph capture.
    /// Must be called BEFORE any graph capture begins.
    pub fn prepare_for_graph_capture(&mut self) -> Result<()> {
        self.cublas.prepare_for_graph_capture()
            .map_err(|e| LLMError::GpuError(format!("cublas graph prep: {e}")))?;
        Ok(())
    }

    /// Upload metadata for a decode batch, padded to `padded_batch` sequences.
    /// Extra slots get token_id=0, position=0, context_len=0, slot_mapping=-1.
    /// Stores offsets for subsequent `forward_gpu_only` calls.
    pub fn upload_metadata_padded(
        &mut self,
        input: &GpuBatchInput,
        padded_batch: usize,
    ) -> Result<()> {
        let actual = input.num_seqs;
        let max_blocks = self.graph_max_blocks;

        self.cpu_scratch.clear();
        let pad = padded_batch - actual;

        // token_ids: bulk extend + zero-pad
        let token_ids_off = 0;
        self.cpu_scratch.extend(input.token_ids[..actual].iter().map(|&t| t as i32));
        self.cpu_scratch.resize(self.cpu_scratch.len() + pad, 0i32);

        // positions: bulk extend + zero-pad
        let positions_off = self.cpu_scratch.len();
        self.cpu_scratch.extend(input.position_ids[..actual].iter().map(|&p| p as i32));
        self.cpu_scratch.resize(self.cpu_scratch.len() + pad, 0i32);

        // context_lens: bulk extend + zero-pad
        let context_lens_off = self.cpu_scratch.len();
        self.cpu_scratch.extend(input.context_lens[..actual].iter().map(|&c| c as i32));
        self.cpu_scratch.resize(self.cpu_scratch.len() + pad, 0i32);

        // block_tables: padded_batch * max_blocks, zero-padded
        let block_tables_off = self.cpu_scratch.len();
        let bt_len = padded_batch * max_blocks;
        self.cpu_scratch.resize(self.cpu_scratch.len() + bt_len, 0i32);
        let max_blocks_input = input.max_blocks_per_seq;
        let copy_len = max_blocks_input.min(max_blocks);
        for s in 0..actual {
            let src_start = s * max_blocks_input;
            let dst_start = block_tables_off + s * max_blocks;
            let src_end = (src_start + copy_len).min(input.block_tables_flat.len());
            for (i, &v) in input.block_tables_flat[src_start..src_end].iter().enumerate() {
                self.cpu_scratch[dst_start + i] = v as i32;
            }
        }

        // slot_mapping: actual + padding with -1
        let slot_mapping_off = self.cpu_scratch.len();
        self.cpu_scratch.extend(input.slot_mapping[..actual].iter().map(|&s| s as i32));
        self.cpu_scratch.resize(self.cpu_scratch.len() + pad, -1i32);

        // seq_start_pos: each decode seq has 1 token -> [0, 1, 2, ..., padded_batch]
        let seq_start_pos_off = self.cpu_scratch.len();
        self.cpu_scratch.extend((0..=padded_batch as i32).map(|i| i));

        // Upload via pinned staging (pageable HtoD is synchronous)
        let total = self.cpu_scratch.len();
        if total > self.meta_packed.len() {
            self.meta_packed = self.stream
                .alloc_zeros::<i32>(total * 2)
                .map_err(|e| LLMError::GpuError(format!("meta_packed realloc: {e}")))?;
        }
        if total > self.pinned_meta.len() {
            self.pinned_meta = PinnedBuffer::<i32>::new(total * 2)?;
        }

        self.pinned_meta.as_mut_slice()[..total]
            .copy_from_slice(&self.cpu_scratch[..total]);
        let dst_dev_ptr = {
            let (ptr, _) = self.meta_packed.device_ptr(&self.stream);
            ptr
        };
        let bytes = total * std::mem::size_of::<i32>();
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst_dev_ptr,
                self.pinned_meta.as_ptr() as *const std::ffi::c_void,
                bytes,
                self.stream.cu_stream(),
            )
            .result()
            .map_err(|e| LLMError::GpuError(format!("pinned padded meta HtoD: {e}")))?;
        }

        self.last_meta_offsets = Some(PackedMetaOffsets {
            token_ids: token_ids_off,
            num_token_ids: padded_batch,
            positions: positions_off,
            num_positions: padded_batch,
            context_lens: context_lens_off,
            num_context_lens: padded_batch,
            block_tables: block_tables_off,
            num_block_tables: bt_len,
            slot_mapping: slot_mapping_off,
            num_slot_mapping: padded_batch,
            seq_start_pos: seq_start_pos_off,
            num_seq_start_pos: padded_batch + 1,
        });

        Ok(())
    }

    /// GPU-only forward pass for CUDA graph capture/replay.
    ///
    /// Runs embedding -> layer loop -> final RMSNorm -> LM head -> argmax.
    /// All GPU kernels -- no HtoD, no DtoH. Uses stored metadata offsets.
    /// Writes argmax token IDs into `self.argmax_output`.
    ///
    /// Call `upload_metadata` or `upload_metadata_padded` first.
    pub fn forward_gpu_only(
        &mut self,
        num_tokens: usize,
        num_seqs: usize,
        max_context_len: u32,
        kv_cache: &CudaKVCache,
    ) -> Result<()> {
        let offsets = self.last_meta_offsets
            .ok_or_else(|| LLMError::GpuError("no metadata uploaded before forward_gpu_only".into()))?;

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let num_layers = self.config.num_layers;
        let gemm_strategy = self.gemm_strategy;

        // 1. Embedding lookup
        self.embedding_lookup(num_tokens, &offsets)?;

        // 2. Layer loop + norm + LM head + argmax
        let Self {
            ref layers,
            ref cutlass,
            ref fa3,
            ref autotune,
            ref cublas,
            ref lt_ops,
            ref stream,
            ref meta_packed,
            ref rope_cos,
            ref rope_sin,
            ref weights,
            ref final_norm_weight,
            ref lm_head_weight,
            ref mut scratch,
            ref mut residual_a,
            ref mut residual_b,
            ref mut down_a,
            ref mut down_b,
            ref mut final_normed,
            ref mut residual_tmp,
            ref mut logits_gpu,
            ref embed_output,
            ref loader,
            ref mut argmax_output,
            ..
        } = *self;

        let cutlass_ref: Option<&CutlassKernels> = cutlass.as_deref();
        let fa3_ref: Option<&Fa3Kernels> = fa3.as_deref();
        let autotune_ref: Option<&CutlassAutotuneCache> = autotune.as_ref();

        let mut prev_fused = false;
        for layer_idx in 0..num_layers {
            let (key_cache, value_cache) = &kv_cache.gpu_cache[layer_idx];
            let attn = AttentionMeta {
                positions: meta_packed
                    .slice(offsets.positions..offsets.positions + offsets.num_positions),
                key_cache,
                value_cache,
                block_tables: meta_packed
                    .slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                context_lens: meta_packed
                    .slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                slot_mapping: meta_packed
                    .slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                seq_start_pos: meta_packed
                    .slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                num_tokens,
                num_seqs,
                max_context_len,
                is_prefill: false,
                rope_cos,
                rope_sin,
            };
            let fp8_refs = if weights.fp8_enabled {
                Some(Fp8LayerWeightRefs {
                    qkv_fp8: &weights.fp8_qkv[layer_idx],
                    qkv_scale: &weights.fp8_qkv_scales[layer_idx],
                    o_proj_fp8: &weights.fp8_o_proj[layer_idx],
                    o_proj_scale: &weights.fp8_o_proj_scales[layer_idx],
                    gate_up_fp8: &weights.fp8_gate_up[layer_idx],
                    gate_up_scale: &weights.fp8_gate_up_scales[layer_idx],
                    down_proj_fp8: &weights.fp8_down[layer_idx],
                    down_proj_scale: &weights.fp8_down_scales[layer_idx],
                })
            } else {
                None
            };
            let layer_weights = LayerWeights {
                qkv_weight: &weights.fused_qkv[layer_idx],
                o_proj_weight: &weights.o_proj[layer_idx],
                gate_up_weight: &weights.fused_gate_up[layer_idx],
                down_proj_weight: &weights.down_proj[layer_idx],
                input_layernorm_weight: &weights.input_layernorm[layer_idx],
                post_attention_layernorm_weight: &weights.post_attn_layernorm[layer_idx],
                fp8: fp8_refs,
            };
            let fused = if layer_idx == 0 {
                layers[layer_idx].forward_batched_v2(
                    embed_output, &attn, &layer_weights, scratch, None,
                    residual_a, down_a,
                    gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                )?
            } else if layer_idx % 2 == 1 {
                if prev_fused {
                    layers[layer_idx].forward_batched_v2(
                        &*down_a, &attn, &layer_weights, scratch, None,
                        residual_b, down_b,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                } else {
                    layers[layer_idx].forward_batched_v2(
                        &*residual_a, &attn, &layer_weights, scratch, Some(&*down_a),
                        residual_b, down_b,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                }
            } else {
                if prev_fused {
                    layers[layer_idx].forward_batched_v2(
                        &*down_b, &attn, &layer_weights, scratch, None,
                        residual_a, down_a,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                } else {
                    layers[layer_idx].forward_batched_v2(
                        &*residual_b, &attn, &layer_weights, scratch, Some(&*down_b),
                        residual_a, down_a,
                        gemm_strategy, cutlass_ref, fa3_ref, autotune_ref, cublas, lt_ops.as_ref(),
                    )?
                }
            };
            prev_fused = fused;
        }

        let last_wrote_to_b = num_layers > 1 && (num_layers - 1) % 2 == 1;
        let (final_residual, final_down): (&CudaSlice<f16>, &CudaSlice<f16>) =
            if last_wrote_to_b {
                (residual_b, down_b)
            } else {
                (residual_a, down_a)
            };

        if prev_fused {
            layers[0].rms_norm_pub(
                final_down, final_norm_weight,
                num_tokens, hidden_size, final_normed,
            )?;
        } else {
            layers[0].fused_residual_rmsnorm_pub(
                final_residual, final_down, final_norm_weight,
                num_tokens, hidden_size, final_normed, residual_tmp,
            )?;
        }

        // LM head GEMM
        cublas.hgemm_f32_output(
            num_tokens, vocab_size, hidden_size,
            1.0, &*final_normed, lm_head_weight, 0.0, logits_gpu,
        )?;

        // GPU-side argmax into pre-allocated buffer
        let argmax_kernel = loader.get_func("argmax", "argmax_kernel")?;
        let block_dim = vocab_size.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&argmax_kernel)
                .arg(&*logits_gpu)
                .arg(&*argmax_output)
                .arg(&(vocab_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("argmax launch: {e}")))?;
        }

        Ok(())
    }

    /// Read the argmax token IDs from the GPU after forward_gpu_only or graph replay.
    /// Uses pinned host memory for truly async DtoH (pageable Vec is synchronous).
    /// Returns a slice into the pinned buffer -- valid until next read_graph_output call.
    pub fn read_graph_output(&mut self, num_tokens: usize) -> Result<&[i32]> {
        let bytes = num_tokens * std::mem::size_of::<i32>();
        let src_dev_ptr = {
            let (ptr, _) = self.argmax_output.device_ptr(&self.stream);
            ptr
        };
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                self.pinned_argmax.as_mut_ptr() as *mut std::ffi::c_void,
                src_dev_ptr,
                bytes,
                self.stream.cu_stream(),
            )
            .result()
            .map_err(|e| LLMError::GpuError(format!("pinned graph DtoH: {e}")))?;
        }
        self.stream.synchronize()
            .map_err(|e| LLMError::GpuError(format!("stream sync: {e}")))?;
        Ok(&self.pinned_argmax.as_slice()[..num_tokens])
    }

    /// Returns the max_seq_len from config (for graph capture max_context_len).
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }
}

fn total_tokens(input: &GpuBatchInput) -> usize {
    if input.is_all_decode {
        input.num_seqs
    } else {
        input.prefill_tokens.len() + input.num_decode_seqs
    }
}
