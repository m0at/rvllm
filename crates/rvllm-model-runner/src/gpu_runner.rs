//! GPU forward pass orchestrator (Agent 13).
//!
//! `GpuModelRunner` drives the full Llama-family forward pass on CUDA:
//! token embedding lookup -> N transformer layers -> final RMSNorm -> LM head -> logits.
//!
//! All CUDA code is gated behind `#[cfg(feature = "cuda")]`. Under `mock-gpu`
//! (the default), this module provides a compile-compatible stub that returns
//! an error at runtime so existing Mac-side tests keep working.

// =========================================================================
//  CUDA implementation
// =========================================================================
/// Output of a forward pass -- either full logits or just argmax token IDs.
#[derive(Debug, Clone)]
pub enum ForwardOutput {
    /// Full logits buffer: [num_tokens * vocab_size] f32.
    Logits(Vec<f32>),
    /// GPU-side argmax token IDs: [num_tokens] i32 (greedy fast path).
    TokenIds(Vec<i32>),
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync};
    use half::f16;
    use tracing::{debug, info, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    use crate::gpu_layer::{
<<<<<<< Updated upstream
        ForwardPath, GemmStrategy, GpuLayerConfig, GpuLayerInput, GpuLayerWeights,
        GpuTransformerLayer, LayerScratchRef,
=======
        GpuLayerConfig, GpuLayerInput, GpuLayerWeights, GpuLayerWeightsF16,
        GpuLayerWeightsFp8, GpuLayerWeightsGptq, GpuLinearAttnWeights,
        GpuLinearAttnWeightsFp8, GpuLinearAttnWeightsGptq,
        GpuTransformerLayer, SsmWorkspace, AttnWorkspace,
>>>>>>> Stashed changes
    };
    use crate::layers::linear_cuda::CudaLinearLayer;
    use crate::layers::norm_cuda::CudaRMSNorm;
    use rvllm_gpu::kernel_loader::KernelLoader;
    use rvllm_gpu::prelude::CublasHandle;
    use rvllm_kv_cache::engine_cuda::CudaCacheEngine;
    use rvllm_model_loader::gpu_weights::GpuModelWeights;

    use super::ForwardOutput;

    /// Convert a float to FP8 E4M3 (1 sign, 4 exp bias=7, 3 mantissa). Range: ±448.
    fn f32_to_fp8_e4m3(val: f32) -> u8 {
        if val == 0.0 { return 0; }
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127; // unbias from f32
        let mantissa = bits & 0x7FFFFF;

        // Rebias to E4M3 (bias=7)
        let fp8_exp = exp + 7;

        if fp8_exp <= 0 {
            // Underflow to zero (subnormals not worth the complexity)
            return (sign << 7) as u8;
        }
        if fp8_exp >= 15 {
            // Overflow to max (±448): exp=1110, mantissa=111 → 0x7E or 0xFE
            return ((sign << 7) | 0x7E) as u8;
        }

        // Round mantissa from 23 bits to 3 bits (round-to-nearest-even)
        let shifted = mantissa >> 20; // top 3 bits
        let remainder = mantissa & 0xFFFFF;
        let half = 0x80000; // midpoint for rounding
        let round_up = remainder > half || (remainder == half && (shifted & 1) == 1);
        let mut mant3 = shifted + if round_up { 1 } else { 0 };

        let mut e = fp8_exp as u32;
        if mant3 >= 8 {
            mant3 = 0;
            e += 1;
            if e >= 15 {
                return ((sign << 7) | 0x7E) as u8;
            }
        }

<<<<<<< Updated upstream
        fn slice(&self) -> &CudaSlice<i32> {
            self.buf.as_ref().expect("upload() must be called first")
        }
    }

    /// Pre-allocated f16 scratch buffers for the forward pass.
    /// Sized for max_batch_tokens. Reused across all layers (sequential execution).
    pub struct F16LayerScratch {
        pub qkv: CudaSlice<f16>,          // [max_tokens * qkv_dim]
        pub attn_out: CudaSlice<f16>,     // [max_tokens * q_dim]
        pub o_proj: CudaSlice<f16>,       // [max_tokens * hidden]
        pub normed: CudaSlice<f16>,       // [max_tokens * hidden]
        pub gate_up: CudaSlice<f16>,      // [max_tokens * intermediate * 2]
        pub silu_out: CudaSlice<f16>,     // [max_tokens * intermediate]
        // Double-buffered: layer N writes to pair A, reads from pair B.
        // Layer N+1 writes to pair B, reads from pair A. Zero alloc, zero copy.
        pub residual_a: CudaSlice<f16>,   // [max_tokens * hidden]
        pub down_a: CudaSlice<f16>,       // [max_tokens * hidden]
        pub residual_b: CudaSlice<f16>,   // [max_tokens * hidden]
        pub down_b: CudaSlice<f16>,       // [max_tokens * hidden]
    }

    /// Element offsets into the packed metadata GPU buffer.
    #[derive(Clone, Copy, Default)]
    struct PackedMetaOffsets {
        token_ids: usize,
        positions: usize,
        context_lens: usize,
        block_tables: usize,
        slot_mapping: usize,
        seq_start_pos: usize,
        num_token_ids: usize,
        num_positions: usize,
        num_context_lens: usize,
        num_block_tables: usize,
        num_slot_mapping: usize,
        num_seq_start_pos: usize,
=======
        ((sign << 7) | (e << 3) | mant3) as u8
>>>>>>> Stashed changes
    }

    pub struct GpuModelRunner {
        weights: GpuModelWeights,
        cache: CudaCacheEngine,
        blas: CublasHandle,
        loader: KernelLoader,
        config: ModelRunnerConfig,
<<<<<<< Updated upstream
        device: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        cutlass: Option<Arc<rvllm_gpu::cutlass_ffi::CutlassKernels>>,
        gemm_strategy: GemmStrategy,
=======
        device: Arc<CudaDevice>,
>>>>>>> Stashed changes
        layers: Vec<GpuTransformerLayer>,
        embed_tokens: CudaSlice<f32>,
        final_norm_weight: CudaSlice<f32>,
        lm_head_weight: CudaSlice<f32>,
        rms_norm_eps: f32,
        /// Weight offset for RMSNorm: 0.0 (standard) or 1.0 (Qwen3.5).
        rms_norm_weight_offset: f32,
        /// Precomputed RoPE cos table on GPU: [max_position, head_dim/2]
        rope_cos: CudaSlice<f32>,
        /// Precomputed RoPE sin table on GPU: [max_position, head_dim/2]
        rope_sin: CudaSlice<f32>,
        /// When true, use hgemm with f16 projection weights instead of sgemm.
        use_fp16: bool,
        /// When true, use FP8 weights: dequant u8->f16 on-the-fly before hgemm.
        use_fp8: bool,
        /// When true, use GPTQ INT4 weights: dequant int4->f32 fused in GEMV.
        use_gptq: bool,
        /// GPTQ group size (128 typical). Only valid when use_gptq is true.
        gptq_group_size: usize,
        /// Gated DeltaNet recurrent states per linear attention layer.
        /// Each entry: CudaSlice<f32> of shape [num_value_heads, key_head_dim, value_head_dim].
        /// Indexed by layer index (only populated for linear_attention layers).
        /// UnsafeCell allows mutation through &self (safe: single-threaded CUDA dispatch).
        ssm_states: std::cell::UnsafeCell<Vec<Option<CudaSlice<f32>>>>,
        /// Conv1d rolling buffer states per linear attention layer.
        /// Each entry: CudaSlice<f32> of shape [conv_dim, kernel_size].
        /// Indexed by layer index (only populated for linear_attention layers).
        conv_states: std::cell::UnsafeCell<Vec<Option<CudaSlice<f32>>>>,
        /// Pre-allocated workspace for SSM per-token loop (eliminates per-layer mallocs).
        ssm_workspace: std::cell::UnsafeCell<Option<SsmWorkspace>>,
        /// Pre-allocated workspace for attention layers (eliminates ~19 cudaMalloc per layer).
        attn_workspace: std::cell::UnsafeCell<Option<AttnWorkspace>>,
        /// LM head quantized to FP8 at load time (saves ~4× bandwidth vs f32 sgemm).
        lm_head_fp8: Option<CudaSlice<u8>>,
        lm_head_fp8_scale: Option<CudaSlice<f32>>,
    }

    impl GpuModelRunner {
        pub fn new(
            weights: GpuModelWeights,
            cache: CudaCacheEngine,
            blas: CublasHandle,
            loader: KernelLoader,
            config: ModelRunnerConfig,
            device: Arc<CudaDevice>,
        ) -> Result<Self> {
            debug!(
                num_layers = config.num_layers,
                hidden = config.hidden_size,
                vocab = config.vocab_size,
                "GpuModelRunner::new"
            );

            let embed_tokens = weights
                .get("model.embed_tokens.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.embed_tokens.weight".into()))?
                .clone();

            let final_norm_weight = weights
                .get("model.norm.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.norm.weight".into()))?
                .clone();

            let has_lm_head = weights.get("lm_head.weight").is_some();
            let has_embed = weights.get("model.embed_tokens.weight").is_some();
            let lm_head_weight = weights
                .get("lm_head.weight")
                .or_else(|| weights.get("model.embed_tokens.weight"))
                .ok_or_else(|| {
                    LLMError::GpuError(
                        "missing lm_head.weight and model.embed_tokens.weight".into(),
                    )
                })?
                .clone();
            info!(
                lm_head_len = lm_head_weight.len(),
                expected = config.vocab_size * config.hidden_size,
                has_lm_head, has_embed,
                source = if has_lm_head { "lm_head.weight" } else { "embed_tokens" },
                "LM head weight loaded"
            );

            let rms_norm_eps = config.rms_norm_eps;
            // Qwen3.5 uses (1 + weight) in RMSNorm; detect via architecture name
            let is_qwen35 = config.architecture.contains("Qwen3_5");
            let rotary_dim = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
            let mut layers = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let layer_cfg = GpuLayerConfig {
                    hidden_size: config.hidden_size,
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                    intermediate_size: config.intermediate_size,
                    rms_norm_eps: config.rms_norm_eps,
                    layer_idx: i,
                    rotary_dim,
                    has_attn_output_gate: config.has_attn_output_gate,
                    rms_norm_weight_offset: if is_qwen35 { 1.0 } else { 0.0 },
                };
                layers.push(GpuTransformerLayer::new(layer_cfg, Arc::clone(&device)));
            }

            // Precompute RoPE cos/sin tables
            let head_dim = config.head_dim;
            let rotary_dim = (head_dim as f32 * config.partial_rotary_factor) as usize;
            let max_pos = config.max_position.min(8192);
            let half_rotary = rotary_dim / 2;
            let rope_theta = config.rope_theta;
            let mut cos_table = vec![0.0f32; max_pos * half_rotary];
            let mut sin_table = vec![0.0f32; max_pos * half_rotary];
            for pos in 0..max_pos {
                for i in 0..half_rotary {
                    let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / rotary_dim as f32);
                    let theta = pos as f32 * freq;
                    cos_table[pos * half_rotary + i] = theta.cos();
                    sin_table[pos * half_rotary + i] = theta.sin();
                }
            }
            let rope_cos = device
                .htod_sync_copy(&cos_table)
                .map_err(|e| LLMError::GpuError(format!("rope cos HtoD: {e}")))?;
            let rope_sin = device
                .htod_sync_copy(&sin_table)
                .map_err(|e| LLMError::GpuError(format!("rope sin HtoD: {e}")))?;
            info!(max_pos, half_rotary, rotary_dim, head_dim, partial_rotary_factor = config.partial_rotary_factor, "RoPE tables uploaded to GPU");

            // Allocate SSM and conv states for linear attention layers
            let num_layers = config.num_layers;
            let mut ssm_states: Vec<Option<CudaSlice<f32>>> = vec![None; num_layers];
            let mut conv_states: Vec<Option<CudaSlice<f32>>> = vec![None; num_layers];
            for i in 0..num_layers {
                let is_linear_attn = !config.layer_types.is_empty()
                    && config.layer_types[i] == "linear_attention";
                if is_linear_attn {
                    // Recurrent state: [num_value_heads=48, key_head_dim=128, value_head_dim=128]
                    let num_value_heads = 48;  // TODO: from config
                    let key_head_dim = 128;
                    let value_head_dim = 128;
                    let ssm_size = num_value_heads * key_head_dim * value_head_dim;
                    ssm_states[i] = Some(device.alloc_zeros::<f32>(ssm_size).map_err(|e| {
                        LLMError::GpuError(format!("ssm_state alloc layer {i}: {e}"))
                    })?);
                    // Conv state: [conv_dim=10240, kernel_size=4]
                    let conv_dim = 10240;  // TODO: from config
                    let kernel_size = 4;
                    conv_states[i] = Some(device.alloc_zeros::<f32>(conv_dim * kernel_size).map_err(|e| {
                        LLMError::GpuError(format!("conv_state alloc layer {i}: {e}"))
                    })?);
                }
            }
            let has_linear_attn = ssm_states.iter().any(|s| s.is_some());
            let ssm_workspace = if has_linear_attn {
                Some(SsmWorkspace::new(&device, config.hidden_size, config.intermediate_size)?)
            } else {
                None
            };
            // Create attention workspace for pre-allocated scratch buffers
            let has_attn = ssm_states.iter().any(|s| s.is_none()); // non-SSM layers are attention
            let attn_workspace = if has_attn {
                Some(AttnWorkspace::new(
                    &device,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    is_qwen35, // has_attn_output_gate
                )?)
            } else {
                None
            };
            info!(
                num_linear = ssm_states.iter().filter(|s| s.is_some()).count(),
                has_ssm_workspace = ssm_workspace.is_some(),
                has_attn_workspace = attn_workspace.is_some(),
                "allocated SSM + conv states for linear attention layers"
            );

            Ok(Self {
                weights,
                cache,
                blas,
                loader,
                config,
                device,
<<<<<<< Updated upstream
                stream,
                gemm_strategy: if cutlass.is_some() { GemmStrategy::Cutlass } else { GemmStrategy::Cublas },
                cutlass,
=======
>>>>>>> Stashed changes
                layers,
                embed_tokens,
                final_norm_weight,
                lm_head_weight,
                rms_norm_eps,
                rms_norm_weight_offset: if is_qwen35 { 1.0 } else { 0.0 },
                rope_cos,
                rope_sin,
                use_fp16: false,
                use_fp8: false,
                use_gptq: false,
                gptq_group_size: 0,
                ssm_states: std::cell::UnsafeCell::new(ssm_states),
                conv_states: std::cell::UnsafeCell::new(conv_states),
                ssm_workspace: std::cell::UnsafeCell::new(ssm_workspace),
                attn_workspace: std::cell::UnsafeCell::new(attn_workspace),
                lm_head_fp8: None,
                lm_head_fp8_scale: None,
            })
        }

        /// Quantize the f32 LM head weight to FP8 E4M3 with blockwise scales.
        /// Called once after set_use_fp8(true) to eliminate the 5.09 GB f32 sgemm.
        fn quantize_lm_head_to_fp8(&mut self) -> Result<()> {
            if self.lm_head_fp8.is_some() {
                return Ok(()); // already quantized
            }
            // Check if model already has FP8 lm_head
            if self.weights.get_fp8("lm_head.weight").is_some()
                || self.weights.get_fp8("model.embed_tokens.weight").is_some()
            {
                return Ok(()); // model already provides FP8 lm_head
            }

            let vocab_size = self.config.vocab_size;
            let hidden_size = self.config.hidden_size;
            let n = vocab_size;
            let k = hidden_size;

            info!(n, k, total = n * k, "quantizing lm_head to FP8 E4M3 (one-time)");
            let start = std::time::Instant::now();

            // Download f32 weight to CPU
            let weight_f32: Vec<f32> = self.device.dtoh_sync_copy(&self.lm_head_weight)
                .map_err(|e| LLMError::GpuError(format!("lm_head dtoh: {e}")))?;

            // Blockwise quantization: 128×128 blocks
            let block_size = 128usize;
            let num_row_blocks = (n + block_size - 1) / block_size;
            let num_col_blocks = (k + block_size - 1) / block_size;
            let mut fp8_bytes = vec![0u8; n * k];
            let mut scales = vec![0.0f32; num_row_blocks * num_col_blocks];

            let fp8_max: f32 = 448.0; // E4M3 max representable value

            for rb in 0..num_row_blocks {
                for cb in 0..num_col_blocks {
                    let row_start = rb * block_size;
                    let row_end = (row_start + block_size).min(n);
                    let col_start = cb * block_size;
                    let col_end = (col_start + block_size).min(k);

                    // Find absmax in this block
                    let mut absmax: f32 = 0.0;
                    for r in row_start..row_end {
                        for c in col_start..col_end {
                            let v = weight_f32[r * k + c].abs();
                            if v > absmax { absmax = v; }
                        }
                    }

                    let scale = if absmax > 0.0 { absmax / fp8_max } else { 1.0 };
                    scales[rb * num_col_blocks + cb] = scale;

                    // Quantize each element
                    for r in row_start..row_end {
                        for c in col_start..col_end {
                            let val = weight_f32[r * k + c] / scale;
                            let clamped = val.clamp(-fp8_max, fp8_max);
                            fp8_bytes[r * k + c] = f32_to_fp8_e4m3(clamped);
                        }
                    }
                }
            }

            let ms = start.elapsed().as_secs_f64() * 1000.0;
            info!(ms = format!("{ms:.0}"), scales_len = scales.len(), "lm_head FP8 quantization done");

<<<<<<< Updated upstream
            // FP8 weight quantization for ALL projection weights (when RVLLM_FP8_WEIGHTS=1)
            // Note: FP8 only helps for M=1 decode (bandwidth-bound GEMV). For batched
            // decode (M>=8), f16 tensor cores already saturate compute and FP8 adds
            // cast overhead with no throughput gain. Use for latency-sensitive single-
            // stream workloads, not high-throughput batch serving.
            if std::env::var("RVLLM_FP8_WEIGHTS").map_or(false, |v| v == "1") {
                use rvllm_gpu::fp8_quantize::quantize_weight_fp8;
                info!("quantizing ALL weights to FP8 E4M3 (per-row scales)...");
                tracing::warn!("FP8 weights: improves single-stream decode latency but does NOT improve batched throughput. For high-concurrency serving, f16 is equivalent or faster.");

                let q_dim = self.config.num_heads * self.config.head_dim;
                let intermediate = self.config.intermediate_size;
                let gate_up_dim = intermediate * 2;

                for i in 0..num_layers {
                    // QKV: [qkv_dim, hidden]
                    let mut host = vec![half::f16::ZERO; self.fused_qkv_weights[i].len()];
                    self.stream.memcpy_dtoh(&self.fused_qkv_weights[i], &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh qkv: {e}")))?;
                    let q = quantize_weight_fp8(&host, qkv_dim, hidden);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc qkv: {e}")))?;
                    self.stream.memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod qkv: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale qkv: {e}")))?;
                    self.stream.memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod qkv: {e}")))?;
                    self.fp8_fused_qkv.push(fp8);
                    self.fp8_fused_qkv_scale.push(sc);

                    // O-proj: [hidden, q_dim]
                    let o_name = format!("model.layers.{i}.self_attn.o_proj.weight");
                    let o_w = self.weights.get(&o_name)
                        .ok_or_else(|| LLMError::GpuError(format!("missing {o_name}")))?;
                    let mut host = vec![half::f16::ZERO; o_w.len()];
                    self.stream.memcpy_dtoh(o_w, &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh o: {e}")))?;
                    let q = quantize_weight_fp8(&host, hidden, q_dim);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc o: {e}")))?;
                    self.stream.memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod o: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale o: {e}")))?;
                    self.stream.memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod o: {e}")))?;
                    self.fp8_o_proj.push(fp8);
                    self.fp8_o_proj_scale.push(sc);

                    // Gate+up: [gate_up_dim, hidden]
                    let mut host = vec![half::f16::ZERO; self.fused_gate_up_weights[i].len()];
                    self.stream.memcpy_dtoh(&self.fused_gate_up_weights[i], &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh gu: {e}")))?;
                    let q = quantize_weight_fp8(&host, gate_up_dim, hidden);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc gu: {e}")))?;
                    self.stream.memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod gu: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale gu: {e}")))?;
                    self.stream.memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod gu: {e}")))?;
                    self.fp8_fused_gate_up.push(fp8);
                    self.fp8_fused_gate_up_scale.push(sc);

                    // Down: [hidden, intermediate]
                    let down_name = format!("model.layers.{i}.mlp.down_proj.weight");
                    let down_w = self.weights.get(&down_name)
                        .ok_or_else(|| LLMError::GpuError(format!("missing {down_name}")))?;
                    let mut host = vec![half::f16::ZERO; down_w.len()];
                    self.stream.memcpy_dtoh(down_w, &mut host)
                        .map_err(|e| LLMError::GpuError(format!("fp8 dtoh down: {e}")))?;
                    let q = quantize_weight_fp8(&host, hidden, intermediate);
                    let mut fp8 = unsafe { self.stream.alloc::<u8>(q.data.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 alloc down: {e}")))?;
                    self.stream.memcpy_htod(&q.data, &mut fp8)
                        .map_err(|e| LLMError::GpuError(format!("fp8 htod down: {e}")))?;
                    let mut sc = unsafe { self.stream.alloc::<half::f16>(q.scales.len()) }
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale down: {e}")))?;
                    self.stream.memcpy_htod(&q.scales, &mut sc)
                        .map_err(|e| LLMError::GpuError(format!("fp8 scale htod down: {e}")))?;
                    self.fp8_down_proj.push(fp8);
                    self.fp8_down_proj_scale.push(sc);
                }
                // Allocate FP8 input scratch (max of all input dimensions)
                let max_k = *[hidden, q_dim, intermediate].iter().max().unwrap();
                self.fp8_input_scratch = Some(unsafe { self.stream.alloc::<u8>(max_k) }
                    .map_err(|e| LLMError::GpuError(format!("fp8 input scratch: {e}")))?);
                info!(num_layers, "FP8 weight quantization complete (all projections)");
            }

            self.alloc_scratch()?;
            Ok(())
        }

        /// Pre-allocate a reusable set of f16 scratch buffers for the forward pass.
        /// Sized for max padded batch (256). Since all layers are processed
        /// sequentially, one set of buffers covers every layer.
        fn alloc_scratch(&mut self) -> Result<()> {
            let max_tokens: usize = 512; // support up to N=512 batch decode
            let hidden = self.config.hidden_size;
            let q_dim = self.config.num_heads * self.config.head_dim;
            let kv_dim = self.config.num_kv_heads * self.config.head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;
            let intermediate = self.config.intermediate_size;

            let alloc = |n: usize| -> Result<CudaSlice<f16>> {
                // Safety: scratch buffers are immediately overwritten by kernels each layer
                unsafe { self.stream.alloc::<f16>(n) }
                    .map_err(|e| LLMError::GpuError(format!("f16 scratch alloc ({n} elems): {e}")))
            };

            let scratch = F16LayerScratch {
                qkv: alloc(max_tokens * qkv_dim)?,
                attn_out: alloc(max_tokens * q_dim)?,
                o_proj: alloc(max_tokens * hidden)?,
                normed: alloc(max_tokens * hidden)?,
                gate_up: alloc(max_tokens * intermediate * 2)?,
                silu_out: alloc(max_tokens * intermediate)?,
                residual_a: alloc(max_tokens * hidden)?,
                down_a: alloc(max_tokens * hidden)?,
                residual_b: alloc(max_tokens * hidden)?,
                down_b: alloc(max_tokens * hidden)?,
            };

            let total_bytes = (max_tokens * (qkv_dim + q_dim + hidden * 3 + intermediate * 3)) * 2;
            info!(max_tokens, total_bytes, "f16 layer scratch allocated");
            *self.f16_scratch.borrow_mut() = Some(scratch);
=======
            // Upload to GPU
            let fp8_gpu = self.device.htod_sync_copy(&fp8_bytes)
                .map_err(|e| LLMError::GpuError(format!("lm_head fp8 htod: {e}")))?;
            let scale_gpu = self.device.htod_sync_copy(&scales)
                .map_err(|e| LLMError::GpuError(format!("lm_head scale htod: {e}")))?;

            self.lm_head_fp8 = Some(fp8_gpu);
            self.lm_head_fp8_scale = Some(scale_gpu);
>>>>>>> Stashed changes
            Ok(())
        }

        pub fn forward(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
        ) -> Result<Vec<f32>> {
            match self.forward_ex(token_ids, positions, attn_meta, is_prefill, false)? {
                ForwardOutput::Logits(logits) => Ok(logits),
                ForwardOutput::TokenIds(_) => unreachable!("greedy_only=false must return Logits"),
            }
        }

        /// Extended forward: when `greedy_only` is true, runs argmax on GPU and
        /// returns only token IDs (num_tokens * 4 bytes DtoH instead of
        /// num_tokens * vocab_size * 4 bytes).
        pub fn forward_ex(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
            greedy_only: bool,
        ) -> Result<ForwardOutput> {
            let num_tokens = token_ids.len();
            let num_seqs = attn_meta.context_lens.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            debug!(num_tokens, num_seqs, is_prefill, greedy_only, "GpuModelRunner::forward_ex");

<<<<<<< Updated upstream
            let max_context_len = attn_meta.max_context_len;

            // Single packed upload: all 6 metadata fields in one memcpy_htod.
            self.upload_metadata(token_ids, positions, attn_meta)?;

            // Step 1: token embedding lookup from packed buffer
            info!("gpu_runner: embedding lookup");

            // === f16 forward path ===
            let debug_fwd = std::env::var("RVLLM_DEBUG").is_ok();
            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            if debug_fwd {
                let vals: Vec<f16> = self.stream.clone_dtoh(&hidden_f16)
                    .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                let first10: Vec<f32> = vals.iter().take(10).map(|v| v.to_f32()).collect();
                let has_nan = vals.iter().any(|v| v.to_f32().is_nan());
                let has_inf = vals.iter().any(|v| v.to_f32().is_infinite());
                let max_abs = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                info!("DEBUG embed: first10={first10:?} nan={has_nan} inf={has_inf} max={max_abs}");
            }

            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            let use_scratch = num_tokens > 1 || is_prefill;
            let mut scratch_borrow = self.f16_scratch.borrow_mut();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;

            // For scratch double-buffering: copy embedding into residual_a so the
            // first layer can read from it while writing to residual_b.
            if use_scratch {
                if let Some(ref mut s) = *scratch_borrow {
                    self.stream.memcpy_dtod(&hidden_f16, &mut s.residual_a.slice_mut(..num_tokens * hidden_size))
                        .map_err(|e| LLMError::GpuError(format!("embed->scratch: {e}")))?;
                }
            }

            let path = self.resolve_forward_path(num_tokens, is_prefill);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];

                // Double-buffer: even layers write A read B, odd write B read A.
                let use_double_buf = use_scratch && scratch_borrow.is_some();
                let (scratch_ref_opt, hs_ref, pmo_ref): (Option<LayerScratchRef<'_>>, &CudaSlice<f16>, Option<&CudaSlice<f16>>) = if use_double_buf {
                    let s = scratch_borrow.as_mut().unwrap();
                    let even = layer_idx % 2 == 0;
                    let (write_res, write_down, read_res, read_down) = if even {
                        (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                    } else {
                        (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                    };
                    let pmo = if layer_idx > 0 { Some(read_down as &CudaSlice<f16>) } else { None };
                    (Some(LayerScratchRef {
                        normed: &mut s.normed,
                        residual: write_res,
                        qkv: &mut s.qkv,
                        attn_out: &mut s.attn_out,
                        o_proj: &mut s.o_proj,
                        gate_up: &mut s.gate_up,
                        silu_out: &mut s.silu_out,
                        down: write_down,
                    }), read_res as &CudaSlice<f16>, pmo)
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
                    hidden_states: hs_ref,
                    positions: packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                    context_lens: packed_buf.slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                    slot_mapping: packed_buf.slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                    fp8_input_scratch_ptr: self.fp8_input_scratch.as_ref().map_or(0u64, |s| {
                        let (p, _) = DevicePtr::device_ptr(s, &self.stream);
                        p
                    }),
                    fp8_input_scratch_len: self.fp8_input_scratch.as_ref().map_or(0, |s| s.len()),
                };
                let weights = self.layer_weights(layer_idx)?;
                let mut scratch_ref = scratch_ref_opt;
                let result = layer.forward(path, &input, &weights, &self.blas,
                    if use_double_buf { pmo_ref } else { prev_mlp_out.as_ref() },
                    self.cublaslt_ref(), scratch_ref.as_mut(), self.gemm_strategy, self.cutlass.as_deref())?;
                if let Some((residual, mlp_out)) = result {
                    // Non-scratch path: take ownership
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
                // Scratch path (None): results are in s.residual/s.down,
                // next iteration reads them via the double-buffer swap.

                if debug_fwd && (layer_idx < 3 || layer_idx == num_layers - 1) {
                    let vals: Vec<f16> = self.stream.clone_dtoh(&hidden_f16)
                        .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                    let first5: Vec<f32> = vals.iter().take(5).map(|v| v.to_f32()).collect();
                    let has_nan = vals.iter().any(|v| v.to_f32().is_nan());
                    let has_inf = vals.iter().any(|v| v.to_f32().is_infinite());
                    let max_abs = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    info!("DEBUG layer {layer_idx} residual: first5={first5:?} nan={has_nan} inf={has_inf} max={max_abs}");

                    let mvals: Vec<f16> = self.stream.clone_dtoh(prev_mlp_out.as_ref().unwrap())
                        .map_err(|e| LLMError::GpuError(format!("debug dtoh: {e}")))?;
                    let mfirst5: Vec<f32> = mvals.iter().take(5).map(|v| v.to_f32()).collect();
                    let mnan = mvals.iter().any(|v| v.to_f32().is_nan());
                    let mmax = mvals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    info!("DEBUG layer {layer_idx} mlp_out: first5={mfirst5:?} nan={mnan} max={mmax}");
                }
            }

            // Double-buffer: extract final layer results from scratch (1 copy at end, not 28)
            if use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (num_layers - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let n = num_tokens * hidden_size;
                    let mut res_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final res: {e}")))?;
                    self.stream.memcpy_dtod(&res_src.slice(..n), &mut res_out)
                        .map_err(|e| LLMError::GpuError(format!("final res dtod: {e}")))?;
                    hidden_f16 = res_out;
                    let mut down_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final mlp: {e}")))?;
                    self.stream.memcpy_dtod(&down_src.slice(..n), &mut down_out)
                        .map_err(|e| LLMError::GpuError(format!("final mlp dtod: {e}")))?;
                    prev_mlp_out = Some(down_out);
                }
            }
            drop(scratch_borrow);
=======
            // DEBUG: dump input metadata
            {
                let tids: Vec<u32> = token_ids.iter().copied().take(20).collect();
                let pids: Vec<u32> = positions.iter().copied().take(20).collect();
                trace!(
                    ?tids, ?pids,
                    num_tokens, num_seqs, is_prefill,
                    context_lens = ?&attn_meta.context_lens,
                    query_lens = ?&attn_meta.query_lens,
                    slot_mapping_len = attn_meta.slot_mapping.len(),
                    block_tables_len = attn_meta.block_tables.len(),
                    max_context_len = attn_meta.max_context_len,
                    "DEBUG forward_ex input"
                );
                if !attn_meta.block_tables.is_empty() {
                    let bt0: Vec<u32> = attn_meta.block_tables[0].iter().copied().take(10).collect();
                    let sm0: Vec<u32> = attn_meta.slot_mapping.iter().copied().take(20).collect();
                    trace!(?bt0, ?sm0, "DEBUG block_tables[0] & slot_mapping");
                }
            }
>>>>>>> Stashed changes

            // Upload positions to GPU as i32 (CUDA kernels expect int*)
            let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
            let positions_gpu: CudaSlice<i32> = self
                .device
                .htod_sync_copy(&pos_i32)
                .map_err(|e| LLMError::GpuError(format!("positions HtoD: {e}")))?;

            // Upload context_lens as i32
            let cl_i32: Vec<i32> = attn_meta.context_lens.iter().map(|&c| c as i32).collect();
            let context_lens_gpu: CudaSlice<i32> = self
                .device
                .htod_sync_copy(&cl_i32)
                .map_err(|e| LLMError::GpuError(format!("context_lens HtoD: {e}")))?;

<<<<<<< Updated upstream
                // Compute full logits and find top-5
                let logits_dbg = CudaLinearLayer::forward_f16_in(
                    &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                    &self.blas,
                )?;
                let logits_cpu: Vec<f32> = self.stream.clone_dtoh(&logits_dbg)
                    .map_err(|e| LLMError::GpuError(format!("debug logits dtoh: {e}")))?;
                // Find top-5 for last token
                let last_start = (num_tokens - 1) * vocab_size;
                let last_logits = &logits_cpu[last_start..last_start + vocab_size];
                let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<(usize, f32)> = indexed[..5.min(indexed.len())].to_vec();
                info!("DEBUG top5_logits: {:?}", top5);
                info!("DEBUG logits range: min={:.2} max={:.2} mean={:.4}",
                    last_logits.iter().cloned().fold(f32::INFINITY, f32::min),
                    last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    last_logits.iter().sum::<f32>() / last_logits.len() as f32,
                );
            }

            // LM head + argmax: f16 hidden -> fused argmax
            if num_tokens == 1 && greedy_only {
                let token_ids_gpu = self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16, &self.lm_head_weight, vocab_size, hidden_size)?;
                let token_ids_cpu = self.stream.clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head token DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Full logits path: hgemm f16 hidden x f16 lm_head -> f32 logits
            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                &self.blas,
            )?;

            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self.stream.clone_dtoh(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax DtoH: {e}")))?;
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            let logits_cpu = self.stream.clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;
            Ok(ForwardOutput::Logits(logits_cpu))
        }

        /// Partial forward: run only the first `max_layers` transformer layers,
        /// then apply final RMSNorm + LM head to produce logits.
        /// Used by self-draft speculative decoding to get approximate predictions
        /// from a fraction of the model's depth.
        pub fn forward_partial(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
            max_layers: usize,
        ) -> Result<Vec<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            self.upload_metadata(token_ids, positions, attn_meta)?;

            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            let gpu_cache = self.cache.gpu_cache();
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            let num_seqs = attn_meta.context_lens.len();
            let max_context_len = attn_meta.max_context_len;
            let layers_to_run = max_layers.min(self.layers.len());
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;
            let path = self.resolve_forward_path(num_tokens, is_prefill);
            let use_scratch = num_tokens > 1 || is_prefill;
            let mut scratch_borrow = self.f16_scratch.borrow_mut();

            // For Batched path: copy embedding into residual_a for double-buffer read
            if use_scratch {
                if let Some(ref mut s) = *scratch_borrow {
                    self.stream.memcpy_dtod(&hidden_f16, &mut s.residual_a.slice_mut(..num_tokens * hidden_size))
                        .map_err(|e| LLMError::GpuError(format!("partial embed->scratch: {e}")))?;
                }
            }

            for (layer_idx, layer) in self.layers.iter().take(layers_to_run).enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];

                let (mut scratch_ref_opt, hs_ref, pmo_ref): (Option<LayerScratchRef<'_>>, &CudaSlice<f16>, Option<&CudaSlice<f16>>) = if use_scratch {
                    if let Some(ref mut s) = *scratch_borrow {
                        let even = layer_idx % 2 == 0;
                        let (write_res, write_down, read_res, read_down) = if even {
                            (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                        } else {
                            (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                        };
                        let pmo = if layer_idx > 0 { Some(read_down as &CudaSlice<f16>) } else { None };
                        let hs: &CudaSlice<f16> = if layer_idx > 0 { read_res } else { &hidden_f16 };
                        (Some(LayerScratchRef {
                            normed: &mut s.normed,
                            residual: write_res,
                            qkv: &mut s.qkv,
                            attn_out: &mut s.attn_out,
                            o_proj: &mut s.o_proj,
                            gate_up: &mut s.gate_up,
                            silu_out: &mut s.silu_out,
                            down: write_down,
                        }), hs, pmo)
                    } else {
                        return Err(LLMError::GpuError("Batched path requires scratch buffers".into()));
                    }
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
                    hidden_states: hs_ref,
                    positions: packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                    context_lens: packed_buf.slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                    slot_mapping: packed_buf.slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                    fp8_input_scratch_ptr: self.fp8_input_scratch.as_ref().map_or(0u64, |s| {
                        let (p, _) = DevicePtr::device_ptr(s, &self.stream);
                        p
                    }),
                    fp8_input_scratch_len: self.fp8_input_scratch.as_ref().map_or(0, |s| s.len()),
                };
                let weights = self.layer_weights(layer_idx)?;
                let result = layer.forward(path, &input, &weights, &self.blas,
                    if use_scratch { pmo_ref } else { prev_mlp_out.as_ref() },
                    self.cublaslt_ref(), scratch_ref_opt.as_mut(), self.gemm_strategy, self.cutlass.as_deref())?;
                if let Some((residual, mlp_out)) = result {
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            }

            // Extract final results from scratch double-buffer
            if use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (layers_to_run - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let n = num_tokens * hidden_size;
                    let mut res_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("partial final res: {e}")))?;
                    self.stream.memcpy_dtod(&res_src.slice(..n), &mut res_out)
                        .map_err(|e| LLMError::GpuError(format!("partial final res dtod: {e}")))?;
                    hidden_f16 = res_out;
                    let mut down_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("partial final mlp: {e}")))?;
                    self.stream.memcpy_dtod(&down_src.slice(..n), &mut down_out)
                        .map_err(|e| LLMError::GpuError(format!("partial final mlp dtod: {e}")))?;
                    prev_mlp_out = Some(down_out);
                }
            }

            // Final norm + LM head (same as full forward)
            let normed_f16 = if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream, &self.loader,
                    &hidden_f16, last_mlp, &self.final_norm_weight,
                    self.rms_norm_eps, num_tokens, hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };

            let logits_gpu = CudaLinearLayer::forward_f16_in(
                &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                &self.blas,
            )?;

            let logits_cpu = self.stream.clone_dtoh(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("forward_partial logits DtoH: {e}")))?;
            Ok(logits_cpu)
        }

        /// Upload all per-step metadata into persistent GPU buffers.
        ///
        /// This MUST be called before `forward_graph_body` (or before replaying
        /// a captured CUDA graph). The memcpy_htod calls update the data at
        /// stable GPU pointers that the graph's kernels will read.
        pub fn upload_metadata(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
        ) -> Result<()> {
            let num_tokens = token_ids.len();
            let num_seqs = attn_meta.context_lens.len();
            let max_blocks = self.graph_max_blocks;

            let mut scratch = self.cpu_scratch.borrow_mut();
            scratch.clear();

            // Pack all 6 metadata fields contiguously, recording offsets.
            let token_ids_off = scratch.len();
            scratch.extend(token_ids.iter().map(|&t| t as i32));
            let num_token_ids = scratch.len() - token_ids_off;

            let positions_off = scratch.len();
            scratch.extend(positions.iter().map(|&p| p as i32));
            let num_positions = scratch.len() - positions_off;

            let context_lens_off = scratch.len();
            scratch.extend(attn_meta.context_lens.iter().map(|&c| c as i32));
            let num_context_lens = scratch.len() - context_lens_off;

            // block_tables: [num_seqs, graph_max_blocks], zero-padded.
            let block_tables_off = scratch.len();
            let bt_len = num_seqs * max_blocks;
            let new_len = scratch.len() + bt_len;
            scratch.resize(new_len, 0i32);
=======
            // Flatten block_tables to [num_seqs, max_blocks_per_seq] row-major as i32
            let max_blocks = attn_meta
                .block_tables
                .iter()
                .map(|r| r.len())
                .max()
                .unwrap_or(1)
                .max(1);
            let mut flat_bt = vec![0i32; num_seqs * max_blocks];
>>>>>>> Stashed changes
            for (s, row) in attn_meta.block_tables.iter().enumerate() {
                for (b, &blk) in row.iter().enumerate() {
                    flat_bt[s * max_blocks + b] = blk as i32;
                }
            }
            let block_tables_gpu: CudaSlice<i32> = self
                .device
                .htod_sync_copy(&flat_bt)
                .map_err(|e| LLMError::GpuError(format!("block_tables HtoD: {e}")))?;

            // Upload slot_mapping as i32
            let sm_i32: Vec<i32> = attn_meta.slot_mapping.iter().map(|&s| s as i32).collect();
            let slot_mapping_gpu: CudaSlice<i32> = self
                .device
                .htod_sync_copy(&sm_i32)
                .map_err(|e| LLMError::GpuError(format!("slot_mapping HtoD: {e}")))?;

            let max_context_len = attn_meta.max_context_len;

            // Build seq_start_pos from query_lens (not context_lens).
            // query_lens[i] = prompt_len for prefill, 1 for decode. Sum == num_tokens.
            let mut seq_starts_host = Vec::with_capacity(num_seqs + 1);
            let mut pos = 0i32;
            for &ql in &attn_meta.query_lens {
                seq_starts_host.push(pos);
                pos += ql as i32;
            }
            seq_starts_host.push(num_tokens as i32); // sentinel
            let seq_start_pos_gpu: CudaSlice<i32> = self
                .device
                .htod_sync_copy(&seq_starts_host)
                .map_err(|e| LLMError::GpuError(format!("seq_start_pos HtoD: {e}")))?;

            // Step 1: token embedding lookup
            trace!("gpu_runner: embedding lookup");
            // DIAG: dump first few token IDs for Python reference comparison
            if std::env::var("RVLLM_DUMP_FP8").is_ok() && is_prefill {
                info!("DIAG token_ids[0..min(5,len)]: {:?}", &token_ids[..token_ids.len().min(5)]);
            }
            let mut hidden_states = self.embedding_lookup(token_ids)?;

            // Diagnostic: embedding output stats (gated behind RVLLM_DUMP_FP8)
            if is_prefill && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                let hs_cpu: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states)
                    .unwrap_or_default();
                if !hs_cpu.is_empty() {
                    let n = hs_cpu.len() as f32;
                    let mean = hs_cpu.iter().sum::<f32>() / n;
                    let rms = (hs_cpu.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
                    let min = hs_cpu.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = hs_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    info!(mean, rms, min, max, total_elements = hs_cpu.len(),
                        "DIAG embedding output");
                }
            }

            // Reset SSM/conv recurrent states on prefill (new sequence).
            // Without this, states from the previous request leak into the next.
            // Uses htod_copy to zero in-place (no alloc/free cycle).
            if is_prefill {
                let ssm_states = unsafe { &mut *self.ssm_states.get() };
                let conv_states = unsafe { &mut *self.conv_states.get() };
                for state in ssm_states.iter_mut() {
                    if let Some(s) = state {
                        let zeros = vec![0.0f32; s.len()];
                        self.device.htod_sync_copy_into(&zeros, s).map_err(|e| {
                            LLMError::GpuError(format!("ssm_state reset: {e}"))
                        })?;
                    }
                }
                for state in conv_states.iter_mut() {
                    if let Some(s) = state {
                        let zeros = vec![0.0f32; s.len()];
                        self.device.htod_sync_copy_into(&zeros, s).map_err(|e| {
                            LLMError::GpuError(format!("conv_state reset: {e}"))
                        })?;
                    }
                }
                debug!("reset SSM/conv states for new sequence");
            }

            // Step 2: transformer layers
            let step2_start = std::time::Instant::now();
            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
<<<<<<< Updated upstream
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            // Double-buffered scratch for T>1 decode: zero per-layer allocations.
            let use_scratch = num_tokens > 1 || is_prefill;
            let mut scratch_borrow = self.f16_scratch.borrow_mut();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;
            let path = self.resolve_forward_path(num_tokens, is_prefill);
=======
            // DIAG: optionally limit number of layers processed
            let max_layers = std::env::var("RVLLM_MAX_LAYERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(num_layers);
            // Per-layer profiling: enabled for first 3 decode steps via counter
            let profile_layers = !is_prefill && std::env::var("RVLLM_PROFILE").is_ok();
            let mut ssm_total_ms = 0.0f64;
            let mut attn_total_ms = 0.0f64;
            let mut ssm_count = 0u32;
            let mut attn_count = 0u32;

>>>>>>> Stashed changes
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if layer_idx >= max_layers {
                    break;
                }
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let even = layer_idx % 2 == 0;

                let (mut scratch_ref_opt, hs_ref, pmo_ref): (Option<LayerScratchRef<'_>>, &CudaSlice<f16>, Option<&CudaSlice<f16>>) = if use_scratch {
                    if let Some(ref mut s) = *scratch_borrow {
                        let (write_res, write_down, read_res, read_down) = if even {
                            (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                        } else {
                            (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                        };
                        let pmo = if layer_idx > 0 { Some(read_down as &CudaSlice<f16>) } else { None };
                        let hs: &CudaSlice<f16> = if layer_idx > 0 { read_res } else { &hidden_f16 };
                        (Some(LayerScratchRef {
                            normed: &mut s.normed,
                            residual: write_res,
                            qkv: &mut s.qkv,
                            attn_out: &mut s.attn_out,
                            o_proj: &mut s.o_proj,
                            gate_up: &mut s.gate_up,
                            silu_out: &mut s.silu_out,
                            down: write_down,
                        }), hs, pmo)
                    } else {
                        (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                    }
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
<<<<<<< Updated upstream
                    hidden_states: hs_ref,
                    positions: packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions),
=======
                    hidden_states: &hidden_states,
                    positions: &positions_gpu,
>>>>>>> Stashed changes
                    key_cache,
                    value_cache,
                    block_tables: &block_tables_gpu,
                    context_lens: &context_lens_gpu,
                    slot_mapping: &slot_mapping_gpu,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: &seq_start_pos_gpu,
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
<<<<<<< Updated upstream
                let weights = self.layer_weights(layer_idx)?;
                let result = layer.forward(path, &input, &weights, &self.blas,
                    if use_scratch { pmo_ref } else { prev_mlp_out.as_ref() },
                    self.cublaslt_ref(), scratch_ref_opt.as_mut(), self.gemm_strategy, self.cutlass.as_deref())?;
                if let Some((residual, mlp_out)) = result {
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            }

            // Double-buffer: extract final layer results (1 copy at end, not per-layer)
            if use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (num_layers - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let n = num_tokens * hidden_size;
                    let mut res_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final res: {e}")))?;
                    self.stream.memcpy_dtod(&res_src.slice(..n), &mut res_out)
                        .map_err(|e| LLMError::GpuError(format!("final res dtod: {e}")))?;
                    hidden_f16 = res_out;
                    let mut down_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final mlp: {e}")))?;
                    self.stream.memcpy_dtod(&down_src.slice(..n), &mut down_out)
                        .map_err(|e| LLMError::GpuError(format!("final mlp dtod: {e}")))?;
                    prev_mlp_out = Some(down_out);
                }
            }
            drop(scratch_borrow);
=======
                // Check if this is a linear attention (Mamba-2) layer
                let is_linear_attn = !self.config.layer_types.is_empty()
                    && self.config.layer_types[layer_idx] == "linear_attention";

                let layer_start = if profile_layers {
                    self.device.synchronize().ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
>>>>>>> Stashed changes

                hidden_states = if is_linear_attn && self.use_gptq {
                    let weights = self.layer_weights_linear_attn_gptq(layer_idx)?;
                    let ssm_states = unsafe { &mut *self.ssm_states.get() };
                    let conv_states = unsafe { &mut *self.conv_states.get() };
                    let ws = unsafe { &mut *self.ssm_workspace.get() };
                    let ssm_state = ssm_states[layer_idx].as_mut()
                        .ok_or_else(|| LLMError::GpuError(format!("missing ssm_state layer {layer_idx}")))?;
                    let conv_state = conv_states[layer_idx].as_mut()
                        .ok_or_else(|| LLMError::GpuError(format!("missing conv_state layer {layer_idx}")))?;
                    let workspace = ws.as_mut()
                        .ok_or_else(|| LLMError::GpuError("missing ssm_workspace".into()))?;
                    layer.forward_linear_attn_gptq(
                        &input, &weights, &self.blas, self.gptq_group_size,
                        ssm_state, conv_state, workspace,
                    )?
                } else if is_linear_attn && self.use_fp8 {
                    let weights = self.layer_weights_linear_attn_fp8(layer_idx)?;
                    let ssm_states = unsafe { &mut *self.ssm_states.get() };
                    let conv_states = unsafe { &mut *self.conv_states.get() };
                    let ws = unsafe { &mut *self.ssm_workspace.get() };
                    let ssm_state = ssm_states[layer_idx].as_mut()
                        .ok_or_else(|| LLMError::GpuError(format!("missing ssm_state layer {layer_idx}")))?;
                    let conv_state = conv_states[layer_idx].as_mut()
                        .ok_or_else(|| LLMError::GpuError(format!("missing conv_state layer {layer_idx}")))?;
                    let workspace = ws.as_mut()
                        .ok_or_else(|| LLMError::GpuError("missing ssm_workspace".into()))?;
                    layer.forward_linear_attn_fp8(
                        &input, &weights, &self.blas,
                        ssm_state, conv_state, workspace,
                    )?
                } else if is_linear_attn {
                    let weights = self.layer_weights_linear_attn(layer_idx)?;
                    let ssm_states = unsafe { &mut *self.ssm_states.get() };
                    let conv_states = unsafe { &mut *self.conv_states.get() };
                    let ws = unsafe { &mut *self.ssm_workspace.get() };
                    let ssm_state = ssm_states[layer_idx].as_mut()
                        .ok_or_else(|| LLMError::GpuError(format!("missing ssm_state layer {layer_idx}")))?;
                    let conv_state = conv_states[layer_idx].as_mut()
                        .ok_or_else(|| LLMError::GpuError(format!("missing conv_state layer {layer_idx}")))?;
                    let workspace = ws.as_mut()
                        .ok_or_else(|| LLMError::GpuError("missing ssm_workspace".into()))?;
                    layer.forward_linear_attn(
                        &input, &weights, &self.blas,
                        ssm_state, conv_state, workspace,
                    )?
                } else if self.use_gptq {
                    let weights = self.layer_weights_gptq(layer_idx)?;
                    layer.forward_gptq(&input, &weights, &self.blas, self.gptq_group_size)?
                } else if self.use_fp8 {
                    let weights = self.layer_weights_fp8(layer_idx)?;
                    let aws = unsafe { &mut *self.attn_workspace.get() };
                    layer.forward_fp8(&input, &weights, &self.blas, aws.as_mut())?
                } else if self.use_fp16 {
                    let weights = self.layer_weights_f16(layer_idx)?;
                    layer.forward_f16(&input, &weights, &self.blas)?
                } else {
                    let weights = self.layer_weights(layer_idx)?;
                    layer.forward(&input, &weights, &self.blas)?
                };

                if let Some(t0) = layer_start {
                    self.device.synchronize().ok();
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if is_linear_attn {
                        ssm_total_ms += ms;
                        ssm_count += 1;
                    } else {
                        attn_total_ms += ms;
                        attn_count += 1;
                    }
                }
            }

            // Sync GPU and log layer loop timing
            self.device.synchronize().ok();
            let layers_ms = step2_start.elapsed().as_secs_f64() * 1000.0;
            if !is_prefill {
                info!(layers_ms = format!("{layers_ms:.1}"), num_tokens, "PERF layers");
                if profile_layers && ssm_count > 0 {
                    let ssm_avg = ssm_total_ms / ssm_count as f64;
                    let attn_avg = if attn_count > 0 { attn_total_ms / attn_count as f64 } else { 0.0 };
                    info!(
                        ssm_ms = format!("{ssm_total_ms:.1}"),
                        ssm_avg_ms = format!("{ssm_avg:.2}"),
                        ssm_layers = ssm_count,
                        attn_ms = format!("{attn_total_ms:.1}"),
                        attn_avg_ms = format!("{attn_avg:.2}"),
                        attn_layers = attn_count,
                        "PERF layer_breakdown"
                    );
                }
            }

            // Step 3: final RMSNorm (all on stream 0, no sync needed)
            let mut normed = CudaRMSNorm::forward_with_offset(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                self.rms_norm_weight_offset,
            )?;
            // Round to BF16 precision to match HF's BF16 trajectory
            if let Some(kernel) = self.device.get_func("cast_fp", "round_f32_to_bf16_kernel") {
                let n = num_tokens * hidden_size;
                let threads = 256u32;
                let blocks = ((n as u32) + threads - 1) / threads;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    let _ = kernel.launch(cfg, (&mut normed, n as i32));
                }
            }

            // DIAG: dump full hidden state and normed state to /tmp for Python verification
            if is_prefill && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                let hs_cpu: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states).unwrap_or_default();
                let normed_cpu: Vec<f32> = self.device.dtoh_sync_copy(&normed).unwrap_or_default();
                let lm_cpu: Vec<f32> = self.device.dtoh_sync_copy(&self.lm_head_weight).unwrap_or_default();
                // Dump normed[0:20]
                if normed_cpu.len() >= 20 {
                    info!("DIAG final_normed[0:20]: {:?}", &normed_cpu[..20]);
                    let rms = (normed_cpu.iter().take(hidden_size).map(|x| x*x).sum::<f32>() / hidden_size as f32).sqrt();
                    info!("DIAG final_normed rms={rms:.6}");
                }
                // Dump final_norm_weight[0:10]
                let fnw: Vec<f32> = self.device.dtoh_sync_copy(&self.final_norm_weight).unwrap_or_default();
                if fnw.len() >= 10 {
                    info!("DIAG final_norm_weight[0:10]: {:?}", &fnw[..10]);
                }
                // Dump lm_head[0:10] (first row) and lm_head size
                info!("DIAG lm_head total len={}", lm_cpu.len());
                if lm_cpu.len() >= 10 {
                    info!("DIAG lm_head[0:10]: {:?}", &lm_cpu[..10]);
                }
                // Manual dot product: normed @ lm_head[0]
                if normed_cpu.len() >= hidden_size && lm_cpu.len() >= hidden_size {
                    let dot: f32 = normed_cpu.iter().take(hidden_size)
                        .zip(lm_cpu.iter().take(hidden_size))
                        .map(|(a, b)| a * b).sum();
                    info!("DIAG manual dot(normed, lm_head[0]) = {dot:.6}");
                }
                // Save full normed and hidden state to files for Python verification
                if let Ok(mut f) = std::fs::File::create("/tmp/rvllm_normed.bin") {
                    use std::io::Write;
                    let bytes: Vec<u8> = normed_cpu.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let _ = f.write_all(&bytes);
                    info!("DIAG saved normed ({} floats) to /tmp/rvllm_normed.bin", normed_cpu.len());
                }
                if let Ok(mut f) = std::fs::File::create("/tmp/rvllm_hidden.bin") {
                    use std::io::Write;
                    let bytes: Vec<u8> = hs_cpu.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let _ = f.write_all(&bytes);
                    info!("DIAG saved hidden ({} floats) to /tmp/rvllm_hidden.bin", hs_cpu.len());
                }
            }

            // Step 4: LM head  normed [num_tokens, hidden] @ lm_head^T [hidden, vocab]
            let step4_start = std::time::Instant::now();
            let logits_gpu = if self.use_gptq {
                // GPTQ LM head: check if quantized, otherwise use f32
                let lm_gptq = self.weights.get_fp8("lm_head.weight")
                    .or_else(|| self.weights.get_fp8("model.embed_tokens.weight"));
                let lm_scale = self.weights.get_scale("lm_head.weight")
                    .or_else(|| self.weights.get_scale("model.embed_tokens.weight"));
                if let (Some(lm_w), Some(sc)) = (lm_gptq, lm_scale) {
                    // lm_head is GPTQ quantized
                    let zeros = self.weights.get_zeros("lm_head.weight")
                        .or_else(|| self.weights.get_zeros("model.embed_tokens.weight"))
                        .ok_or_else(|| LLMError::GpuError("missing lm_head gptq zeros".into()))?;
                    CudaLinearLayer::forward_once_gptq(
                        &normed, lm_w, sc, zeros,
                        num_tokens, vocab_size, hidden_size, self.gptq_group_size, &self.blas,
                    )?
                } else {
                    // lm_head is f32 (tied to embed_tokens, not quantized)
                    CudaLinearLayer::forward_once(
                        &normed, &self.lm_head_weight, None,
                        num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                }
            } else if self.use_fp8 {
                // FP8 LM head: try pre-quantized first, then model's FP8, then f32 fallback
                if let (Some(lm_fp8), Some(lm_scale)) = (&self.lm_head_fp8, &self.lm_head_fp8_scale) {
                    CudaLinearLayer::forward_once_fp8(
                        &normed, lm_fp8, Some(lm_scale), num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                } else {
                    let lm_fp8 = self.weights.get_fp8("lm_head.weight")
                        .or_else(|| self.weights.get_fp8("model.embed_tokens.weight"));
                    if let Some(lm_w) = lm_fp8 {
                        let scale = self.weights.get_scale("lm_head.weight")
                            .or_else(|| self.weights.get_scale("model.embed_tokens.weight"));
                        CudaLinearLayer::forward_once_fp8(
                            &normed, lm_w, scale, num_tokens, vocab_size, hidden_size, &self.blas,
                        )?
                    } else {
                        // Final fallback: lm_head is f32
                        CudaLinearLayer::forward_once(
                            &normed, &self.lm_head_weight, None,
                            num_tokens, vocab_size, hidden_size, &self.blas,
                        )?
                    }
                }
            } else if self.use_fp16 {
                let lm_f16 = self
                    .weights
                    .get_f16("lm_head.weight")
                    .or_else(|| self.weights.get_f16("model.embed_tokens.weight"));
                if let Some(lm_w) = lm_f16 {
                    CudaLinearLayer::forward_once_f16(
                        &normed, lm_w, num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                } else {
                    // Fallback: lm_head not available as f16, use f32
                    CudaLinearLayer::forward_once(
                        &normed, &self.lm_head_weight, None,
                        num_tokens, vocab_size, hidden_size, &self.blas,
                    )?
                }
            } else {
                CudaLinearLayer::forward_once(
                    &normed, &self.lm_head_weight, None,
                    num_tokens, vocab_size, hidden_size, &self.blas,
                )?
            };

            self.device.synchronize().ok();
            let lmhead_ms = step4_start.elapsed().as_secs_f64() * 1000.0;
            if !is_prefill {
                info!(lmhead_ms = format!("{lmhead_ms:.1}"), "PERF lm_head");
            }

            // Log top-5 final logits (gated behind RVLLM_DUMP_FP8)
            if is_prefill && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                let logits_cpu = self.device.dtoh_sync_copy(&logits_gpu).unwrap_or_default();
                let lo = (num_tokens - 1) * vocab_size;
                if logits_cpu.len() >= lo + vocab_size {
                    let ll = &logits_cpu[lo..lo + vocab_size];
                    let mut idxd: Vec<(usize, f32)> = ll.iter().enumerate().map(|(i,&v)| (i,v)).collect();
                    idxd.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let top5: Vec<(usize,f32)> = idxd.iter().take(5).cloned().collect();
                    info!(?top5, "FINAL logits top-5");
                }
            }

            // Step 5: greedy fast path -- argmax on GPU, copy only token IDs
            if greedy_only {
                let token_ids_gpu = self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?;
                let token_ids_cpu = self
                    .device
                    .dtoh_sync_copy(&token_ids_gpu)
                    .map_err(|e| LLMError::GpuError(format!("argmax token_ids DtoH: {e}")))?;
                debug!(
                    num_tokens,
                    "forward_ex complete (greedy, {} bytes DtoH)",
                    num_tokens * 4
                );
                return Ok(ForwardOutput::TokenIds(token_ids_cpu));
            }

            // Step 5 (fallback): full logits DtoH for temperature>0 sampling
            let logits_cpu = self
                .device
                .dtoh_sync_copy(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;

            debug!(
                logits_len = logits_cpu.len(),
                expected = num_tokens * vocab_size,
                "forward_ex complete (full logits)"
            );
            Ok(ForwardOutput::Logits(logits_cpu))
        }

<<<<<<< Updated upstream
        /// Get a reference to the CUDA stream used by this runner.
        pub fn cuda_stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }

        /// Get cublasLt reference (None when feature is off).
        fn cublaslt_ref(&self) -> Option<&crate::CublasLtRef> {
            #[cfg(feature = "cublaslt")]
            { self.blas_lt.as_ref() }
            #[cfg(not(feature = "cublaslt"))]
            { None }
        }

        fn resolve_forward_path(&self, num_tokens: usize, is_prefill: bool) -> ForwardPath {
            if num_tokens == 1 && !is_prefill {
                // Megakernel: all 28 layers in one kernel launch
                // Enable with RVLLM_MEGAKERNEL=1
                if std::env::var("RVLLM_MEGAKERNEL").map_or(false, |v| v == "1") {
                    return ForwardPath::MegakernelDecode;
                }
                // DAG persistent decode: single kernel per layer
                // Enable with RVLLM_PERSISTENT=1
                if std::env::var("RVLLM_PERSISTENT").map_or(false, |v| v == "1") {
                    return ForwardPath::PersistentDecode;
                }
                #[cfg(feature = "cublaslt")]
                if self.blas_lt.is_some()
                    && self.weights.has_fp8_weights()
                    && self.fp8_input_scratch.is_some()
                {
                    return ForwardPath::Fp8Decode;
                }
                ForwardPath::FusedDecode
            } else {
                ForwardPath::Batched
            }
        }

        /// Prepare the runner for CUDA graph capture.
        ///
        /// Pre-allocates the cuBLAS workspace (required by cuBLAS for graph
        /// capture) and ensures the stream has async alloc support.
        pub fn prepare_for_graph_capture(&mut self) -> rvllm_core::error::Result<()> {
            self.blas.prepare_for_graph_capture()?;
            if !self.stream.context().has_async_alloc() {
                tracing::warn!(
                    "GPU does not support async memory allocation (cuMemAllocAsync). \
                     CUDA graph capture may fail. Consider upgrading the CUDA driver."
                );
            }
            // Pre-allocate the packed metadata buffer large enough for max batch size
            // so it never reallocates (which would invalidate graph-captured pointers).
            // Layout per step: token_ids(N) + positions(N) + context_lens(N) +
            //   block_tables(N * graph_max_blocks) + slot_mapping(N) + seq_start_pos(N+1)
            let max_seqs = 256usize; // must match max_num_seqs
            let max_meta = max_seqs * (1 + 1 + 1 + self.graph_max_blocks + 1 + 1) + 1;
            let mut meta = self.meta_packed.borrow_mut();
            let dummy = vec![0i32; max_meta];
            meta.upload(&dummy, &self.stream)
                .map_err(|e| LLMError::GpuError(format!("pre-alloc meta: {e}")))?;
            info!(max_meta, "pre-allocated packed metadata buffer for graph stability");
            // Pre-allocate graph output buffer too (same reason -- stable pointer).
            {
                let mut out = self.graph_output.borrow_mut();
                *out = Some(self.stream.alloc_zeros::<i32>(max_seqs)
                    .map_err(|e| LLMError::GpuError(format!("pre-alloc graph_output: {e}")))?);
            }
            Ok(())
        }

        /// GPU-only forward pass for CUDA graph capture.
        ///
        /// Runs the full forward pass (embedding -> layers -> norm -> argmax)
        /// writing the argmax result into the persistent `graph_output` buffer.
        /// Does NOT do any DtoH copy, so this is safe to capture in a CUDA graph.
        ///
        /// Call `upload_metadata()` first. After this, call `read_graph_output()`
        /// to get the host-side token IDs.
        pub fn forward_gpu_only(
            &self,
            num_tokens: usize,
            num_seqs: usize,
            max_context_len: u32,
            is_prefill: bool,
        ) -> Result<()> {
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            let mut hidden_f16 = self.embedding_lookup_from_meta(num_tokens)?;

            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            let meta_packed = self.meta_packed.borrow();
            let packed_buf = meta_packed.slice();
            let offsets = self.meta_packed_offsets.get();
            // Double-buffered scratch for T>1 decode: zero per-layer allocations.
            let use_scratch = num_tokens > 1 || is_prefill;
            let mut scratch_borrow = self.f16_scratch.borrow_mut();
            let mut prev_mlp_out: Option<CudaSlice<f16>> = None;
            let path = self.resolve_forward_path(num_tokens, is_prefill);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let even = layer_idx % 2 == 0;

                let (mut scratch_ref_opt, hs_ref, pmo_ref): (Option<LayerScratchRef<'_>>, &CudaSlice<f16>, Option<&CudaSlice<f16>>) = if use_scratch {
                    if let Some(ref mut s) = *scratch_borrow {
                        let (write_res, write_down, read_res, read_down) = if even {
                            (&mut s.residual_b, &mut s.down_b, &s.residual_a, &s.down_a)
                        } else {
                            (&mut s.residual_a, &mut s.down_a, &s.residual_b, &s.down_b)
                        };
                        let pmo = if layer_idx > 0 { Some(read_down as &CudaSlice<f16>) } else { None };
                        let hs: &CudaSlice<f16> = if layer_idx > 0 { read_res } else { &hidden_f16 };
                        (Some(LayerScratchRef {
                            normed: &mut s.normed,
                            residual: write_res,
                            qkv: &mut s.qkv,
                            attn_out: &mut s.attn_out,
                            o_proj: &mut s.o_proj,
                            gate_up: &mut s.gate_up,
                            silu_out: &mut s.silu_out,
                            down: write_down,
                        }), hs, pmo)
                    } else {
                        (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                    }
                } else {
                    (None, &hidden_f16 as &CudaSlice<f16>, prev_mlp_out.as_ref())
                };

                let input = GpuLayerInput {
                    hidden_states: hs_ref,
                    positions: packed_buf.slice(offsets.positions..offsets.positions + offsets.num_positions),
                    key_cache,
                    value_cache,
                    block_tables: packed_buf.slice(offsets.block_tables..offsets.block_tables + offsets.num_block_tables),
                    context_lens: packed_buf.slice(offsets.context_lens..offsets.context_lens + offsets.num_context_lens),
                    slot_mapping: packed_buf.slice(offsets.slot_mapping..offsets.slot_mapping + offsets.num_slot_mapping),
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    seq_start_pos: packed_buf.slice(offsets.seq_start_pos..offsets.seq_start_pos + offsets.num_seq_start_pos),
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                    fp8_input_scratch_ptr: self.fp8_input_scratch.as_ref().map_or(0u64, |s| {
                        let (p, _) = DevicePtr::device_ptr(s, &self.stream);
                        p
                    }),
                    fp8_input_scratch_len: self.fp8_input_scratch.as_ref().map_or(0, |s| s.len()),
                };
                let weights = self.layer_weights(layer_idx)?;
                let result = layer.forward(path, &input, &weights, &self.blas,
                    if use_scratch { pmo_ref } else { prev_mlp_out.as_ref() },
                    self.cublaslt_ref(), scratch_ref_opt.as_mut(), self.gemm_strategy, self.cutlass.as_deref())?;
                if let Some((residual, mlp_out)) = result {
                    hidden_f16 = residual;
                    prev_mlp_out = Some(mlp_out);
                }
            }

            // Double-buffer: extract final layer results (1 copy at end, not per-layer)
            if use_scratch {
                if let Some(ref s) = *scratch_borrow {
                    let last_even = (num_layers - 1) % 2 == 0;
                    let (res_src, down_src) = if last_even {
                        (&s.residual_b, &s.down_b)
                    } else {
                        (&s.residual_a, &s.down_a)
                    };
                    let n = num_tokens * hidden_size;
                    let mut res_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final res: {e}")))?;
                    self.stream.memcpy_dtod(&res_src.slice(..n), &mut res_out)
                        .map_err(|e| LLMError::GpuError(format!("final res dtod: {e}")))?;
                    hidden_f16 = res_out;
                    let mut down_out = unsafe { self.stream.alloc::<f16>(n) }
                        .map_err(|e| LLMError::GpuError(format!("final mlp: {e}")))?;
                    self.stream.memcpy_dtod(&down_src.slice(..n), &mut down_out)
                        .map_err(|e| LLMError::GpuError(format!("final mlp dtod: {e}")))?;
                    prev_mlp_out = Some(down_out);
                }
            }
            drop(scratch_borrow);

            // Final: fuse last layer's residual add with final RMSNorm
            let normed_f16 = if let Some(ref last_mlp) = prev_mlp_out {
                let (n, _) = GpuTransformerLayer::fused_residual_rmsnorm_f16(
                    &self.stream, &self.loader,
                    &hidden_f16, last_mlp, &self.final_norm_weight,
                    self.rms_norm_eps, num_tokens, hidden_size,
                )?;
                n
            } else {
                self.rms_norm_f16_runner(&hidden_f16, &self.final_norm_weight, hidden_size)?
            };

            let token_ids_gpu = if num_tokens == 1 {
                self.gpu_fused_lm_head_argmax_f16_hidden(
                    &normed_f16, &self.lm_head_weight, vocab_size, hidden_size)?
            } else {
                // Multi-token: hgemm f16 hidden x f16 lm_head -> f32 logits -> argmax
                let logits_gpu = CudaLinearLayer::forward_f16_in(
                    &normed_f16, &self.lm_head_weight, num_tokens, vocab_size, hidden_size,
                    &self.blas,
                )?;
                self.gpu_argmax(&logits_gpu, num_tokens, vocab_size)?
            };

            // Copy argmax result into the persistent output buffer.
            // On first call this allocates; on subsequent calls with the same
            // num_tokens it reuses the same GPU pointer (crucial for graph replay).
            let mut out = self.graph_output.borrow_mut();
            let need = num_tokens;
            let have = out.as_ref().map_or(0, |b| b.len());
            if have < need {
                *out = Some(self.stream.alloc_zeros::<i32>(need)
                    .map_err(|e| LLMError::GpuError(format!("graph_output alloc: {e}")))?);
            }
            let dst = out.as_mut().unwrap();
            self.stream.memcpy_dtod(&token_ids_gpu, dst)
                .map_err(|e| LLMError::GpuError(format!("graph_output dtod: {e}")))?;

            Ok(())
        }

        /// Read the argmax token IDs from the persistent graph output buffer.
        ///
        /// Call after `forward_gpu_only()` or after replaying a CUDA graph.
        /// This performs a DtoH copy (outside the graph).
        pub fn read_graph_output(&self, num_tokens: usize) -> Result<Vec<i32>> {
            let out = self.graph_output.borrow();
            let buf = out.as_ref().ok_or_else(|| {
                LLMError::GpuError("graph_output not populated -- call forward_gpu_only first".into())
            })?;
            // Copy only the needed elements
            let full = self.stream.clone_dtoh(buf)
                .map_err(|e| LLMError::GpuError(format!("graph_output DtoH: {e}")))?;
            Ok(full[..num_tokens].to_vec())
        }

        /// Enqueue an async DtoH copy of graph output into a pinned host buffer.
        ///
        /// Unlike `read_graph_output`, this does NOT synchronize the stream.
        /// The caller must call `sync_stream()` before reading from `dst`.
        /// `dst` MUST be pinned host memory for truly async behavior; with
        /// pageable memory cuMemcpyDtoHAsync degrades to synchronous.
        pub fn read_graph_output_async(
            &self,
            num_tokens: usize,
            dst: &mut [i32],
        ) -> Result<()> {
            let out = self.graph_output.borrow();
            let buf = out.as_ref().ok_or_else(|| {
                LLMError::GpuError("graph_output not populated -- call forward_gpu_only first".into())
            })?;
            // Only copy num_tokens elements (not the full padded buffer)
            let src_view = buf.slice(..num_tokens);
            self.stream.memcpy_dtoh(&src_view, &mut dst[..num_tokens])
                .map_err(|e| LLMError::GpuError(format!("graph_output async DtoH: {e}")))?;
            Ok(())
        }

        /// Synchronize the runner's CUDA stream, blocking until all enqueued
        /// work (graph replay + async DtoH) completes.
        pub fn sync_stream(&self) -> Result<()> {
            self.stream.synchronize()
                .map_err(|e| LLMError::GpuError(format!("stream sync: {e}")))?;
            Ok(())
        }

=======
>>>>>>> Stashed changes
        /// Launch argmax kernel on GPU, returning [num_tokens] i32 token IDs.
        fn gpu_argmax(
            &self,
            logits_gpu: &CudaSlice<f32>,
            num_tokens: usize,
            vocab_size: usize,
        ) -> Result<CudaSlice<i32>> {
            let kernel = self
                .device
                .get_func("argmax", "argmax_kernel")
                .ok_or_else(|| LLMError::GpuError("argmax_kernel not loaded".into()))?;

            let output: CudaSlice<i32> = self
                .device
                .alloc_zeros::<i32>(num_tokens)
                .map_err(|e| LLMError::GpuError(format!("argmax alloc: {e}")))?;

            let block_dim = vocab_size.min(1024) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            logits_gpu,
                            &output,
                            vocab_size as i32,
                        ),
                    )
                    .map_err(|e| LLMError::GpuError(format!("argmax_kernel launch: {e}")))?;
            }

            Ok(output)
        }

<<<<<<< Updated upstream
        /// Fused LM-head matvec + argmax for single-token greedy decode (f16 weights).
        fn gpu_fused_lm_head_argmax_f16(
            &self,
            hidden_state: &CudaSlice<f32>,
            weight: &CudaSlice<f16>,
            vocab_size: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<i32>> {
            let pass1 = self
                .loader
                .get_func("fused_lm_head_argmax_f16", "fused_lm_head_argmax_f16_kernel")?;
            let pass2 = self
                .loader
                .get_func("fused_lm_head_argmax", "fused_lm_head_argmax_reduce_kernel")?;

            let num_blocks = (vocab_size + 255) / 256;

            let partial_val: CudaSlice<f32> = self
                .stream
                .alloc_zeros::<f32>(num_blocks)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head_f16 partial_val alloc: {e}")))?;
            let partial_idx: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(num_blocks)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head_f16 partial_idx alloc: {e}")))?;
            let output: CudaSlice<i32> = self
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| LLMError::GpuError(format!("fused_lm_head_f16 output alloc: {e}")))?;

            // Pass 1: per-block dot + local argmax (f16 weight, f32 hidden)
            let cfg1 = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (hidden_size * std::mem::size_of::<f32>()) as u32,
            };
            unsafe {
                self.stream
                    .launch_builder(&pass1)
                    .arg(weight)
                    .arg(hidden_state)
                    .arg(&partial_val)
                    .arg(&partial_idx)
                    .arg(&(vocab_size as i32))
                    .arg(&(hidden_size as i32))
                    .launch(cfg1)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head_argmax_f16_kernel launch: {e}")))?;
            }

            // Pass 2: reduce partials to single token ID
            let reduce_threads = num_blocks.min(1024) as u32;
            let cfg2 = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (reduce_threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&pass2)
                    .arg(&partial_val)
                    .arg(&partial_idx)
                    .arg(&output)
                    .arg(&(num_blocks as i32))
                    .launch(cfg2)
                    .map_err(|e| LLMError::GpuError(format!("fused_lm_head_argmax_f16_reduce launch: {e}")))?;
            }

            Ok(output)
        }

=======
>>>>>>> Stashed changes
        /// Per-layer weight references into the GPU weight map.
        fn layer_weights(&self, i: usize) -> Result<GpuLayerWeights<'_>> {
            let g = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeights {
                input_layernorm: g(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                q_norm: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_norm.weight")),
                k_norm: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_norm.weight")),
                post_attention_layernorm: g(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: g(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }

        fn embedding_lookup(&self, token_ids: &[u32]) -> Result<CudaSlice<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;

            let kernel = self
                .device
                .get_func("embedding_gather", "embedding_gather_kernel")
                .ok_or_else(|| LLMError::GpuError("embedding_gather_kernel not loaded".into()))?;

            let output = self
                .device
                .alloc_zeros::<f32>(num_tokens * hidden_size)
                .map_err(|e| LLMError::GpuError(format!("embed alloc: {e}")))?;

            let ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
            let ids_gpu = self
                .device
                .htod_sync_copy(&ids_i32)
                .map_err(|e| LLMError::GpuError(format!("token_ids HtoD: {e}")))?;

            let block_dim = hidden_size.min(1024) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            &output,
                            &self.embed_tokens,
                            &ids_gpu,
                            hidden_size as i32,
                            self.config.vocab_size as i32,
                        ),
                    )
                    .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
            }

            Ok(output)
        }

<<<<<<< Updated upstream
        /// RMSNorm f16: f16 input, f16 weight, f16 output.
        fn rms_norm_f16_runner(
            &self,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            hidden_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let num_tokens = input.len() / hidden_size;
            // Safety: rms_norm kernel writes all elements
            let mut output = unsafe { self.stream.alloc::<f16>(input.len()) }
                .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = self.loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                self.stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(weight)
                    .arg(&self.rms_norm_eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Fused LM-head matvec + argmax for single-token greedy decode (f16 weights, f16 hidden).
        /// Casts f16 hidden -> f32 internally since the kernel expects f32 hidden.
        fn gpu_fused_lm_head_argmax_f16_hidden(
            &self,
            hidden_state_f16: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            vocab_size: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<i32>> {
            // Cast f16 hidden -> f32 for the LM head kernel
            let cast_kernel = self.loader.get_func("cast_fp", "cast_f16_to_f32_kernel")?;
            // Safety: cast kernel writes all hidden_size elements
            let mut hidden_f32 = unsafe { self.stream.alloc::<f32>(hidden_size) }
                .map_err(|e| LLMError::GpuError(format!("lm_head f16->f32 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((hidden_size as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&cast_kernel)
                    .arg(&mut hidden_f32)
                    .arg(hidden_state_f16)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("lm_head f16->f32 launch: {e}")))?;
            }
            self.gpu_fused_lm_head_argmax_f16(&hidden_f32, weight, vocab_size, hidden_size)
        }

=======
>>>>>>> Stashed changes
        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn cache(&self) -> &CudaCacheEngine {
            &self.cache
        }

        pub fn cache_mut(&mut self) -> &mut CudaCacheEngine {
            &mut self.cache
        }

        /// Enable f16 inference mode.
        ///
        /// After calling this, `forward_ex` will use hgemm with f16 projection
        /// weights for all linear layers. The `GpuModelWeights` must already
        /// have f16 weights populated via `insert_f16` / the f16 loader.
        pub fn enable_fp16(&mut self) {
            self.use_fp16 = true;
            info!(use_fp16 = true, "GpuModelRunner: fp16 mode enabled");
        }

        pub fn use_fp16(&self) -> bool {
            self.use_fp16
        }

        /// Enable FP8 inference mode.
        ///
        /// After calling this, `forward_ex` will dequant FP8 u8 weights to f16
        /// on-the-fly before hgemm for all linear layers.
        /// Also quantizes the LM head to FP8 if it's currently f32.
        pub fn enable_fp8(&mut self) {
            self.use_fp8 = true;
            // Quantize LM head to FP8 to avoid 5 GB f32 sgemm during decode
            if let Err(e) = self.quantize_lm_head_to_fp8() {
                info!(error = %e, "lm_head FP8 quantization failed, will use f32 fallback");
            }
            info!(use_fp8 = true, has_lm_head_fp8 = self.lm_head_fp8.is_some(),
                  "GpuModelRunner: fp8 mode enabled");
        }

        pub fn use_fp8(&self) -> bool {
            self.use_fp8
        }

        /// Enable GPTQ INT4 inference mode.
        ///
        /// After calling this, `forward_ex` will use the INT4 GEMV kernel for
        /// decode and dequant INT4->F16 + HGEMM for prefill.
        pub fn enable_gptq(&mut self, group_size: usize) {
            self.use_gptq = true;
            // Also enable fp8 flag to reuse the FP8 weight accessor infrastructure
            // (GPTQ qweights are stored in the same fp8 map as CudaSlice<u8>)
            self.use_fp8 = true;
            self.gptq_group_size = group_size;
            info!(use_gptq = true, group_size, "GpuModelRunner: gptq int4 mode enabled");
        }

        pub fn use_gptq(&self) -> bool {
            self.use_gptq
        }

        /// Per-layer f16 weight references into the GPU weight map.
        fn layer_weights_f16(&self, i: usize) -> Result<GpuLayerWeightsF16<'_>> {
            let g_f16 = |name: &str| -> Result<&CudaSlice<f16>> {
                self.weights
                    .get_f16(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing f16 weight: {name}")))
            };
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeightsF16 {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g_f16(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g_f16(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g_f16(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g_f16(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                post_attention_layernorm: g_f32(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: g_f16(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g_f16(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g_f16(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }

        /// Per-layer FP8 weight references into the GPU weight map.
        fn layer_weights_fp8(&self, i: usize) -> Result<GpuLayerWeightsFp8<'_>> {
            let g_fp8 = |name: &str| -> Result<&CudaSlice<u8>> {
                self.weights
                    .get_fp8(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing fp8 weight: {name}")))
            };
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeightsFp8 {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g_fp8(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g_fp8(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g_fp8(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g_fp8(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.self_attn.q_proj.weight")),
                k_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.self_attn.k_proj.weight")),
                v_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.self_attn.v_proj.weight")),
                o_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.self_attn.o_proj.weight")),
                q_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self
                    .weights
                    .get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                q_norm: self.weights.get(&format!("model.layers.{i}.self_attn.q_norm.weight")),
                k_norm: self.weights.get(&format!("model.layers.{i}.self_attn.k_norm.weight")),
                post_attention_layernorm: g_f32(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: g_fp8(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g_fp8(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g_fp8(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
                gate_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.mlp.gate_proj.weight")),
                up_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.mlp.up_proj.weight")),
                down_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.mlp.down_proj.weight")),
            })
        }

        /// Per-layer FP8 weight references for linear attention (Mamba-2) layers.
        ///
        /// Mamba-2 layers use `model.layers.{i}.linear_attn.*` weight names instead of
        /// `model.layers.{i}.self_attn.*`. The projection dimensions are derived
        /// from the FP8 weight buffer sizes.
        fn layer_weights_linear_attn_fp8(&self, i: usize) -> Result<GpuLinearAttnWeightsFp8<'_>> {
            let g_fp8 = |name: &str| -> Result<&CudaSlice<u8>> {
                self.weights
                    .get_fp8(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing fp8 weight: {name}")))
            };
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };

            let in_proj_qkv = g_fp8(&format!("model.layers.{i}.linear_attn.in_proj_qkv.weight"))?;
            let in_proj_z = g_fp8(&format!("model.layers.{i}.linear_attn.in_proj_z.weight"))?;
            let hidden = self.config.hidden_size;
            let qkv_dim = in_proj_qkv.len() / hidden;
            let z_dim = in_proj_z.len() / hidden;

            // Mamba-2 SSM weights (loaded as f32 by FP8 loader since they're BF16/F32)
            let in_proj_a = g_f32(&format!("model.layers.{i}.linear_attn.in_proj_a.weight"))?;
            let in_proj_b = g_f32(&format!("model.layers.{i}.linear_attn.in_proj_b.weight"))?;
            let conv1d_weight = g_f32(&format!("model.layers.{i}.linear_attn.conv1d.weight"))?;
            let a_log = g_f32(&format!("model.layers.{i}.linear_attn.A_log"))?;
            let dt_bias = g_f32(&format!("model.layers.{i}.linear_attn.dt_bias"))?;
            let ssm_norm = g_f32(&format!("model.layers.{i}.linear_attn.norm.weight"))?;

            // Derive SSM config from model config or weight shapes
            // in_proj_a: [num_value_heads, hidden] → num_value_heads = in_proj_a.len() / hidden
            let num_value_heads = in_proj_a.len() / hidden; // 48
            // Q dim from qkv: total = q_dim + k_dim + v_dim
            // v_dim = z_dim (gate matches value dimension) = 6144
            // k_dim = q_dim (key and query have same dim)
            // qkv_dim = 2*k_dim + v_dim → k_dim = (qkv_dim - z_dim) / 2
            let k_total_dim = (qkv_dim - z_dim) / 2; // 2048
            let num_key_heads = 16; // from config: linear_num_key_heads
            let key_head_dim = k_total_dim / num_key_heads; // 128
            let value_head_dim = z_dim / num_value_heads; // 128
            let conv_kernel_size = 4; // from config: linear_conv_kernel_dim

            Ok(GpuLinearAttnWeightsFp8 {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                in_proj_qkv,
                in_proj_z,
                out_proj: g_fp8(&format!("model.layers.{i}.linear_attn.out_proj.weight"))?,
                in_proj_qkv_scale: self.weights.get_scale(&format!("model.layers.{i}.linear_attn.in_proj_qkv.weight")),
                in_proj_z_scale: self.weights.get_scale(&format!("model.layers.{i}.linear_attn.in_proj_z.weight")),
                out_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.linear_attn.out_proj.weight")),
                qkv_dim,
                z_dim,
                // Mamba-2 SSM weights
                in_proj_a,
                in_proj_b,
                conv1d_weight,
                a_log,
                dt_bias,
                ssm_norm,
                num_key_heads,
                num_value_heads,
                key_head_dim,
                value_head_dim,
                conv_kernel_size,
                // Post-attn and MLP
                post_attention_layernorm: g_f32(&format!(
                    "model.layers.{i}.post_attention_layernorm.weight"
                ))?,
                gate_proj: g_fp8(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g_fp8(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g_fp8(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
                gate_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.mlp.gate_proj.weight")),
                up_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.mlp.up_proj.weight")),
                down_proj_scale: self.weights.get_scale(&format!("model.layers.{i}.mlp.down_proj.weight")),
            })
        }

        /// Per-layer GPTQ INT4 weight references for full attention layers.
        fn layer_weights_gptq(&self, i: usize) -> Result<GpuLayerWeightsGptq<'_>> {
            let g_fp8 = |name: &str| -> Result<&CudaSlice<u8>> {
                self.weights
                    .get_fp8(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing gptq weight: {name}")))
            };
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            let g_scale = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get_scale(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing gptq scale: {name}")))
            };
            let g_zeros = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get_zeros(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing gptq zeros: {name}")))
            };
            Ok(GpuLayerWeightsGptq {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj: g_fp8(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj: g_fp8(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj: g_fp8(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj: g_fp8(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_scales: g_scale(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj_scales: g_scale(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj_scales: g_scale(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj_scales: g_scale(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_zeros: g_zeros(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj_zeros: g_zeros(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj_zeros: g_zeros(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj_zeros: g_zeros(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                q_proj_bias: self.weights.get(&format!("model.layers.{i}.self_attn.q_proj.bias")),
                k_proj_bias: self.weights.get(&format!("model.layers.{i}.self_attn.k_proj.bias")),
                v_proj_bias: self.weights.get(&format!("model.layers.{i}.self_attn.v_proj.bias")),
                q_norm: self.weights.get(&format!("model.layers.{i}.self_attn.q_norm.weight")),
                k_norm: self.weights.get(&format!("model.layers.{i}.self_attn.k_norm.weight")),
                post_attention_layernorm: g_f32(&format!("model.layers.{i}.post_attention_layernorm.weight"))?,
                gate_proj: g_fp8(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g_fp8(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g_fp8(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
                gate_proj_scales: g_scale(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj_scales: g_scale(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj_scales: g_scale(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
                gate_proj_zeros: g_zeros(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj_zeros: g_zeros(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj_zeros: g_zeros(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }

        /// Per-layer GPTQ INT4 weight references for linear attention (Mamba-2) layers.
        fn layer_weights_linear_attn_gptq(&self, i: usize) -> Result<GpuLinearAttnWeightsGptq<'_>> {
            let g_fp8 = |name: &str| -> Result<&CudaSlice<u8>> {
                self.weights
                    .get_fp8(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing gptq weight: {name}")))
            };
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            let g_scale = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get_scale(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing gptq scale: {name}")))
            };
            let g_zeros = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get_zeros(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing gptq zeros: {name}")))
            };

            let in_proj_qkv = g_fp8(&format!("model.layers.{i}.linear_attn.in_proj_qkv.weight"))?;
            let in_proj_z = g_fp8(&format!("model.layers.{i}.linear_attn.in_proj_z.weight"))?;
            let hidden = self.config.hidden_size;
            // GPTQ: CudaSlice<u8> stores [N, K/8] int32 as bytes → N = len * 2 / K
            let qkv_dim = in_proj_qkv.len() * 2 / hidden;
            let z_dim = in_proj_z.len() * 2 / hidden;

            let in_proj_a = g_f32(&format!("model.layers.{i}.linear_attn.in_proj_a.weight"))?;
            let num_value_heads = in_proj_a.len() / hidden;
            let k_total_dim = (qkv_dim - z_dim) / 2;
            let num_key_heads = 16;
            let key_head_dim = k_total_dim / num_key_heads;
            let value_head_dim = z_dim / num_value_heads;
            let conv_kernel_size = 4;

            Ok(GpuLinearAttnWeightsGptq {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                in_proj_qkv,
                in_proj_z,
                out_proj: g_fp8(&format!("model.layers.{i}.linear_attn.out_proj.weight"))?,
                in_proj_qkv_scales: g_scale(&format!("model.layers.{i}.linear_attn.in_proj_qkv.weight"))?,
                in_proj_z_scales: g_scale(&format!("model.layers.{i}.linear_attn.in_proj_z.weight"))?,
                out_proj_scales: g_scale(&format!("model.layers.{i}.linear_attn.out_proj.weight"))?,
                in_proj_qkv_zeros: g_zeros(&format!("model.layers.{i}.linear_attn.in_proj_qkv.weight"))?,
                in_proj_z_zeros: g_zeros(&format!("model.layers.{i}.linear_attn.in_proj_z.weight"))?,
                out_proj_zeros: g_zeros(&format!("model.layers.{i}.linear_attn.out_proj.weight"))?,
                qkv_dim,
                z_dim,
                in_proj_a,
                in_proj_b: g_f32(&format!("model.layers.{i}.linear_attn.in_proj_b.weight"))?,
                conv1d_weight: g_f32(&format!("model.layers.{i}.linear_attn.conv1d.weight"))?,
                a_log: g_f32(&format!("model.layers.{i}.linear_attn.A_log"))?,
                dt_bias: g_f32(&format!("model.layers.{i}.linear_attn.dt_bias"))?,
                ssm_norm: g_f32(&format!("model.layers.{i}.linear_attn.norm.weight"))?,
                num_key_heads,
                num_value_heads,
                key_head_dim,
                value_head_dim,
                conv_kernel_size,
                post_attention_layernorm: g_f32(&format!("model.layers.{i}.post_attention_layernorm.weight"))?,
                gate_proj: g_fp8(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g_fp8(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g_fp8(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
                gate_proj_scales: g_scale(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj_scales: g_scale(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj_scales: g_scale(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
                gate_proj_zeros: g_zeros(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj_zeros: g_zeros(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj_zeros: g_zeros(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }

        /// Per-layer f32 weight references for linear attention (Mamba-2) layers.
        /// Used for BF16 models loaded as f32.
        fn layer_weights_linear_attn(&self, i: usize) -> Result<GpuLinearAttnWeights<'_>> {
            let g_f32 = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights
                    .get(name)
                    .ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };

            let in_proj_qkv = g_f32(&format!("model.layers.{i}.linear_attn.in_proj_qkv.weight"))?;
            let in_proj_z = g_f32(&format!("model.layers.{i}.linear_attn.in_proj_z.weight"))?;
            let hidden = self.config.hidden_size;
            let qkv_dim = in_proj_qkv.len() / hidden;
            let z_dim = in_proj_z.len() / hidden;

            let in_proj_a = g_f32(&format!("model.layers.{i}.linear_attn.in_proj_a.weight"))?;
            let num_value_heads = in_proj_a.len() / hidden;
            let k_total_dim = (qkv_dim - z_dim) / 2;
            let num_key_heads = 16;
            let key_head_dim = k_total_dim / num_key_heads;
            let value_head_dim = z_dim / num_value_heads;
            let conv_kernel_size = 4;

            Ok(GpuLinearAttnWeights {
                input_layernorm: g_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                in_proj_qkv,
                in_proj_z,
                out_proj: g_f32(&format!("model.layers.{i}.linear_attn.out_proj.weight"))?,
                qkv_dim,
                z_dim,
                in_proj_a,
                in_proj_b: g_f32(&format!("model.layers.{i}.linear_attn.in_proj_b.weight"))?,
                conv1d_weight: g_f32(&format!("model.layers.{i}.linear_attn.conv1d.weight"))?,
                a_log: g_f32(&format!("model.layers.{i}.linear_attn.A_log"))?,
                dt_bias: g_f32(&format!("model.layers.{i}.linear_attn.dt_bias"))?,
                ssm_norm: g_f32(&format!("model.layers.{i}.linear_attn.norm.weight"))?,
                num_key_heads,
                num_value_heads,
                key_head_dim,
                value_head_dim,
                conv_kernel_size,
                post_attention_layernorm: g_f32(&format!("model.layers.{i}.post_attention_layernorm.weight"))?,
                gate_proj: g_f32(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj: g_f32(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj: g_f32(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }
    }
}

// Re-export under cuda feature gate.
#[cfg(feature = "cuda")]
pub use cuda_impl::GpuModelRunner;

// =========================================================================
//  Mock-GPU stub (default feature)
// =========================================================================
#[cfg(not(feature = "cuda"))]
mod mock_impl {
    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;
    use super::ForwardOutput;

    /// Stub GpuModelRunner for non-CUDA builds.
    ///
    /// Allows downstream code to reference the type without conditional
    /// compilation everywhere. All methods return an error at runtime.
    pub struct GpuModelRunner {
        config: ModelRunnerConfig,
    }

    impl GpuModelRunner {
        /// Returns an error -- real CUDA is required.
        pub fn forward(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
        ) -> Result<Vec<f32>> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn forward_ex(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
            _greedy_only: bool,
        ) -> Result<ForwardOutput> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn enable_fp16(&mut self) {}

        pub fn use_fp16(&self) -> bool {
            false
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use mock_impl::GpuModelRunner;

// =========================================================================
//  Tests (run under mock-gpu / default features)
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_runner_returns_error() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 2,
                hidden_size: 64,
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 16,
                intermediate_size: 128,
                vocab_size: 100,
                max_position: 512,
                rope_theta: 10000.0,
                dtype: rvllm_core::types::Dtype::Float32,
                architecture: "LlamaForCausalLM".to_string(),
                layer_types: vec![],
            };
            let runner = GpuModelRunner { config };
            let result = runner.forward(&[1, 2, 3], &[0, 1, 2], &[], &[]);
            assert!(result.is_err());
            let err_msg = format!("{}", result.unwrap_err());
            assert!(err_msg.contains("cuda"));
        }
    }

    #[test]
    fn config_accessible() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 4,
                hidden_size: 256,
                num_heads: 8,
                num_kv_heads: 8,
                head_dim: 32,
                intermediate_size: 512,
                vocab_size: 32000,
                max_position: 2048,
                rope_theta: 10000.0,
                dtype: rvllm_core::types::Dtype::Float16,
                architecture: "LlamaForCausalLM".to_string(),
                layer_types: vec![],
            };
            let runner = GpuModelRunner { config };
            assert_eq!(runner.config().num_layers, 4);
            assert_eq!(runner.config().vocab_size, 32000);
        }
    }
}
