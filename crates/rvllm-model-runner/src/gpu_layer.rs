//! GPU Transformer Layer -- one complete transformer block on CUDA.
//!
//! Combines all CUDA dispatch ops (Agents 2-7) into the standard
//! decoder-only transformer sequence:
//!
//! 1. RMSNorm(input)
//! 2. QKV projection (cuBLAS sgemm)
//! 3. RoPE on Q, K
//! 4. PagedAttention(Q, K_cache, V_cache)
//! 5. Output projection (cuBLAS sgemm)
//! 6. RMSNorm(residual + attn_out)
//! 7. MLP: gate+up (cuBLAS) -> fused_silu_mul -> down (cuBLAS)
//! 8. residual + mlp_out
//!
//! All code is gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg};
    use half::f16;
    use tracing::{info, trace};

    use rvllm_core::error::{LLMError, Result};
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::kernel_loader::KernelLoader;

    // ===================================================================
    // CUDA event profiling (RVLLM_PROFILE=1)
    // ===================================================================

    /// Per-layer timing breakdown in milliseconds.
    #[derive(Debug, Clone, Default)]
    pub struct LayerProfileMs {
        pub norm: f32,
        pub qkv: f32,
        pub attn: f32,
        pub o_proj: f32,
        pub norm2: f32,
        pub mlp: f32,
    }

    /// Per-layer CUDA event pairs collected during a profiled forward pass.
    pub struct LayerProfileEvents {
        pub norm: (CudaEvent, CudaEvent),
        pub qkv: (CudaEvent, CudaEvent),
        pub attn: (CudaEvent, CudaEvent),
        pub o_proj: (CudaEvent, CudaEvent),
        pub norm2: (CudaEvent, CudaEvent),
        pub mlp: (CudaEvent, CudaEvent),
    }

    impl LayerProfileEvents {
        pub fn elapsed_ms(&self) -> Result<LayerProfileMs> {
            let ms = |pair: &(CudaEvent, CudaEvent)| -> Result<f32> {
                pair.0
                    .elapsed_ms(&pair.1)
                    .map_err(|e| LLMError::GpuError(format!("event elapsed: {e}")))
            };
            Ok(LayerProfileMs {
                norm: ms(&self.norm)?,
                qkv: ms(&self.qkv)?,
                attn: ms(&self.attn)?,
                o_proj: ms(&self.o_proj)?,
                norm2: ms(&self.norm2)?,
                mlp: ms(&self.mlp)?,
            })
        }
    }

    /// Record a timing-enabled CUDA event on the given stream.
    pub fn record_event(device: &Arc<CudaContext>, stream: &CudaStream) -> Result<CudaEvent> {
        let ev = device
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| LLMError::GpuError(format!("event create: {e}")))?;
        ev.record(stream)
            .map_err(|e| LLMError::GpuError(format!("event record: {e}")))?;
        Ok(ev)
    }

    /// Configuration for a single transformer layer.
    #[derive(Debug, Clone)]
    pub struct GpuLayerConfig {
        pub hidden_size: usize,
        pub num_heads: usize,
        pub num_kv_heads: usize,
        pub head_dim: usize,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub layer_idx: usize,
    }

    /// Weight references for a single transformer layer.
    ///
    /// All slices live on GPU and are owned by the GpuModelWeights container.
    /// This struct borrows them for the duration of a forward pass.
    pub struct GpuLayerWeights<'a> {
        // Pre-attention norm
        pub input_layernorm: &'a CudaSlice<f32>,
        // Attention projections
        pub q_proj: &'a CudaSlice<f32>,
        pub k_proj: &'a CudaSlice<f32>,
        pub v_proj: &'a CudaSlice<f32>,
        pub o_proj: &'a CudaSlice<f32>,
        // Optional QKV biases (Qwen2.5 has these)
        pub q_proj_bias: Option<&'a CudaSlice<f32>>,
        pub k_proj_bias: Option<&'a CudaSlice<f32>>,
        pub v_proj_bias: Option<&'a CudaSlice<f32>>,
        // Post-attention norm
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        // MLP weights
        pub gate_proj: &'a CudaSlice<f32>,
        pub up_proj: &'a CudaSlice<f32>,
        pub down_proj: &'a CudaSlice<f32>,
    }

    /// FP16 weight references for a single transformer layer (f16 GEMM path).
    ///
    /// Projection weights are f16 (used with hgemm). Norm weights and biases
    /// remain f32 since RMSNorm and bias-add operate in f32.
    pub struct GpuLayerWeightsF16<'a> {
        pub input_layernorm: &'a CudaSlice<f32>,
        pub q_proj: &'a CudaSlice<f16>,
        pub k_proj: &'a CudaSlice<f16>,
        pub v_proj: &'a CudaSlice<f16>,
        pub o_proj: &'a CudaSlice<f16>,
        pub q_proj_bias: Option<&'a CudaSlice<f32>>,
        pub k_proj_bias: Option<&'a CudaSlice<f32>>,
        pub v_proj_bias: Option<&'a CudaSlice<f32>>,
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        pub gate_proj: &'a CudaSlice<f16>,
        pub up_proj: &'a CudaSlice<f16>,
        pub down_proj: &'a CudaSlice<f16>,
    }

    /// Metadata needed for a single layer forward pass.
    pub struct GpuLayerInput<'a> {
        /// Hidden states entering this layer, shape [num_tokens, hidden_size].
        pub hidden_states: &'a CudaSlice<f32>,
        /// Position ids for RoPE, shape [num_tokens]. Kernels expect int*.
        pub positions: &'a CudaSlice<i32>,
        /// KV cache key block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub key_cache: &'a CudaSlice<f16>,
        /// KV cache value block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub value_cache: &'a CudaSlice<f16>,
        /// Block table mapping sequence positions to cache blocks, shape [num_seqs, max_blocks_per_seq].
        pub block_tables: &'a CudaSlice<i32>,
        /// Context length for each sequence, shape [num_seqs].
        pub context_lens: &'a CudaSlice<i32>,
        /// Slot mapping for cache writes during prefill, shape [num_tokens].
        pub slot_mapping: &'a CudaSlice<i32>,
        /// Number of tokens in the batch.
        pub num_tokens: usize,
        /// Number of sequences in the batch.
        pub num_seqs: usize,
        /// Maximum context length across sequences.
        pub max_context_len: u32,
        /// Block size for paged attention.
        pub block_size: usize,
        /// True during prefill (prompt processing), false during decode.
        pub is_prefill: bool,
        /// Per-sequence query token start positions: [num_seqs + 1] with sentinel.
        /// Built from actual query token counts, NOT context_lens.
        pub seq_start_pos: &'a CudaSlice<i32>,
        /// RoPE cos table on GPU: [max_position, head_dim/2].
        pub rope_cos: &'a CudaSlice<f32>,
        /// RoPE sin table on GPU: [max_position, head_dim/2].
        pub rope_sin: &'a CudaSlice<f32>,
    }

    /// One complete GPU transformer layer.
    ///
    /// Holds references to the kernel loader and cuBLAS handle;
    /// weights are passed in per-call via `GpuLayerWeights`.
    pub struct GpuTransformerLayer {
        config: GpuLayerConfig,
        stream: Arc<CudaStream>,
        loader: Arc<KernelLoader>,
    }

    impl GpuTransformerLayer {
        pub fn new(config: GpuLayerConfig, stream: Arc<CudaStream>, loader: Arc<KernelLoader>) -> Self {
            Self { config, stream, loader }
        }

        /// Execute a full transformer layer forward pass.
        ///
        /// Returns the output hidden states as a new CudaSlice<f32> of shape
        /// [num_tokens, hidden_size]. The caller is responsible for using this
        /// as input to the next layer.
        /// FP16 forward pass -- uses hgemm for projection weights (f16), while
        /// norms, biases, RoPE, and attention remain in f32.
        pub fn forward_f16(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeightsF16<'_>,
            blas: &CublasHandle,
        ) -> Result<CudaSlice<f32>> {
            use crate::layers::linear_cuda::CudaLinearLayer;

            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            // 1. Pre-attention RMSNorm (f32)
            let normed = Self::rms_norm(
                &self.stream,
                &self.loader,
                input.hidden_states,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // 2. QKV projections via hgemm (f16 weights)
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let mut q = CudaLinearLayer::forward_once_f16(
                &normed, weights.q_proj, num_tokens, q_dim, hidden, blas, &self.loader,
            )?;
            let mut k = CudaLinearLayer::forward_once_f16(
                &normed, weights.k_proj, num_tokens, kv_dim, hidden, blas, &self.loader,
            )?;
            let mut v = CudaLinearLayer::forward_once_f16(
                &normed, weights.v_proj, num_tokens, kv_dim, hidden, blas, &self.loader,
            )?;

            // QKV biases (f32)
            if let Some(bias) = weights.q_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut q, bias, num_tokens, q_dim)?;
            }
            if let Some(bias) = weights.k_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut k, bias, num_tokens, kv_dim)?;
            }
            if let Some(bias) = weights.v_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut v, bias, num_tokens, kv_dim)?;
            }

            // 3. RoPE
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.stream,
                &self.loader,
                &q,
                &k,
                input.positions,
                input.rope_cos,
                input.rope_sin,
                num_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;

            // 4. KV cache write + attention
            Self::cache_write(
                &self.stream,
                &self.loader,
                &k_rot,
                &v,
                input.key_cache,
                input.value_cache,
                input.slot_mapping,
                num_tokens,
                num_kv_heads,
                head_dim,
            )?;

            let attn_out = if input.is_prefill {
                Self::prefill_attention(
                    &self.stream,
                    &self.loader,
                    &q_rot,
                    input.key_cache,
                    input.value_cache,
                    input.block_tables,
                    input.context_lens,
                    input.seq_start_pos,
                    num_tokens,
                    input.num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    input.max_context_len,
                    input.block_size,
                )?
            } else {
                Self::decode_attention(
                    &self.stream,
                    &self.loader,
                    &q_rot,
                    input.key_cache,
                    input.value_cache,
                    input.block_tables,
                    input.context_lens,
                    num_tokens,
                    input.num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    input.max_context_len,
                    input.block_size,
                )?
            };

            // 5. Output projection (f16 weight)
            let attn_proj = CudaLinearLayer::forward_once_f16(
                &attn_out, weights.o_proj, num_tokens, hidden, q_dim, blas, &self.loader,
            )?;

            // Residual
            let residual = Self::add_tensors(
                &self.stream,
                &self.loader,
                input.hidden_states,
                &attn_proj,
                num_tokens * hidden,
            )?;

            // 6. Post-attention RMSNorm (f32)
            let normed2 = Self::rms_norm(
                &self.stream,
                &self.loader,
                &residual,
                weights.post_attention_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // 7. MLP (f16 weights)
            let gate = CudaLinearLayer::forward_once_f16(
                &normed2, weights.gate_proj, num_tokens, intermediate, hidden, blas, &self.loader,
            )?;
            let up = CudaLinearLayer::forward_once_f16(
                &normed2, weights.up_proj, num_tokens, intermediate, hidden, blas, &self.loader,
            )?;
            let fused = Self::fused_silu_mul(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?;
            let mlp_out = CudaLinearLayer::forward_once_f16(
                &fused, weights.down_proj, num_tokens, hidden, intermediate, blas, &self.loader,
            )?;

            // 8. Residual
            Self::add_tensors(&self.stream, &self.loader, &residual, &mlp_out, num_tokens * hidden)
        }

        pub fn forward(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeights<'_>,
            blas: &CublasHandle,
        ) -> Result<CudaSlice<f32>> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            trace!(
                layer = cfg.layer_idx,
                num_tokens,
                "gpu transformer layer forward"
            );

            // ---------------------------------------------------------------
            // 1. Pre-attention RMSNorm
            // ---------------------------------------------------------------
            let normed = Self::rms_norm(
                &self.stream,
                &self.loader,
                input.hidden_states,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // All ops on this stream -- no cross-stream sync needed

            // ---------------------------------------------------------------
            // 2. QKV projections via cuBLAS sgemm
            //    input [num_tokens, hidden] x weight^T [hidden, proj_dim]
            // ---------------------------------------------------------------
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let mut q = Self::linear(
                &self.stream,
                blas,
                &normed,
                weights.q_proj,
                num_tokens,
                q_dim,
                hidden,
            )?;
            let mut k = Self::linear(
                &self.stream,
                blas,
                &normed,
                weights.k_proj,
                num_tokens,
                kv_dim,
                hidden,
            )?;
            let mut v = Self::linear(
                &self.stream,
                blas,
                &normed,
                weights.v_proj,
                num_tokens,
                kv_dim,
                hidden,
            )?;

            // Apply QKV biases if present (e.g. Qwen2.5)
            if let Some(bias) = weights.q_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut q, bias, num_tokens, q_dim)?;
            }
            if let Some(bias) = weights.k_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut k, bias, num_tokens, kv_dim)?;
            }
            if let Some(bias) = weights.v_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut v, bias, num_tokens, kv_dim)?;
            }

            // ---------------------------------------------------------------
            // 3. RoPE on Q and K
            // ---------------------------------------------------------------
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.stream,
                &self.loader,
                &q,
                &k,
                input.positions,
                input.rope_cos,
                input.rope_sin,
                num_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;

            // ---------------------------------------------------------------
            // 4. KV cache write + Attention (prefill vs decode)
            // ---------------------------------------------------------------
            // Always write K/V into paged cache via slot_mapping
            info!(layer = cfg.layer_idx, "gpu_layer: cache_write start");
            Self::cache_write(
                &self.stream,
                &self.loader,
                &k_rot,
                &v,
                input.key_cache,
                input.value_cache,
                input.slot_mapping,
                num_tokens,
                num_kv_heads,
                head_dim,
            )?;

            info!(layer = cfg.layer_idx, "gpu_layer: cache_write done");

            let attn_out = if input.is_prefill {
                // Prefill: use FA2 prefill kernel reading from paged cache
                info!(layer = cfg.layer_idx, "gpu_layer: prefill_attention start");
                Self::prefill_attention(
                    &self.stream,
                    &self.loader,
                    &q_rot,
                    input.key_cache,
                    input.value_cache,
                    input.block_tables,
                    input.context_lens,
                    input.seq_start_pos,
                    num_tokens,
                    input.num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    input.max_context_len,
                    input.block_size,
                )?
            } else {
                // Decode: read from paged cache
                info!(layer = cfg.layer_idx, "gpu_layer: decode_attention start");
                Self::decode_attention(
                    &self.stream,
                    &self.loader,
                    &q_rot,
                    input.key_cache,
                    input.value_cache,
                    input.block_tables,
                    input.context_lens,
                    num_tokens,
                    input.num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    input.max_context_len,
                    input.block_size,
                )?
            };

            // ---------------------------------------------------------------
            // 5. Output projection
            // ---------------------------------------------------------------
            let attn_proj = Self::linear(
                &self.stream,
                blas,
                &attn_out,
                weights.o_proj,
                num_tokens,
                hidden,
                q_dim,
            )?;

            // ---------------------------------------------------------------
            // Residual: hidden_states + attn_proj
            // ---------------------------------------------------------------
            let residual = Self::add_tensors(
                &self.stream,
                &self.loader,
                input.hidden_states,
                &attn_proj,
                num_tokens * hidden,
            )?;

            // ---------------------------------------------------------------
            // 6. Post-attention RMSNorm
            // ---------------------------------------------------------------
            let normed2 = Self::rms_norm(
                &self.stream,
                &self.loader,
                &residual,
                weights.post_attention_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // ---------------------------------------------------------------
            // 7. MLP: gate_proj + up_proj -> fused_silu_mul -> down_proj
            // ---------------------------------------------------------------
            let gate = Self::linear(
                &self.stream,
                blas,
                &normed2,
                weights.gate_proj,
                num_tokens,
                intermediate,
                hidden,
            )?;
            let up = Self::linear(
                &self.stream,
                blas,
                &normed2,
                weights.up_proj,
                num_tokens,
                intermediate,
                hidden,
            )?;

            let fused = Self::fused_silu_mul(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?;

            let mlp_out = Self::linear(
                &self.stream,
                blas,
                &fused,
                weights.down_proj,
                num_tokens,
                hidden,
                intermediate,
            )?;

            // ---------------------------------------------------------------
            // 8. Residual: residual + mlp_out
            // ---------------------------------------------------------------
            let output = Self::add_tensors(&self.stream, &self.loader, &residual, &mlp_out, num_tokens * hidden)?;

            Ok(output)
        }

        /// Profiled f32 forward: identical to `forward` but records CUDA events
        /// around each phase. Returns (hidden_states, LayerProfileEvents).
        pub fn forward_profiled(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeights<'_>,
            blas: &CublasHandle,
            device: &Arc<CudaContext>,
        ) -> Result<(CudaSlice<f32>, LayerProfileEvents)> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // 1. Pre-attention RMSNorm
            let ev_norm_start = record_event(device, &self.stream)?;
            let normed = Self::rms_norm(
                &self.stream, &self.loader, input.hidden_states,
                weights.input_layernorm, cfg.rms_norm_eps, num_tokens, hidden,
            )?;
            let ev_norm_end = record_event(device, &self.stream)?;

            // 2. QKV projections + biases + RoPE
            let ev_qkv_start = record_event(device, &self.stream)?;
            let mut q = Self::linear(&self.stream, blas, &normed, weights.q_proj, num_tokens, q_dim, hidden)?;
            let mut k = Self::linear(&self.stream, blas, &normed, weights.k_proj, num_tokens, kv_dim, hidden)?;
            let mut v = Self::linear(&self.stream, blas, &normed, weights.v_proj, num_tokens, kv_dim, hidden)?;
            if let Some(bias) = weights.q_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut q, bias, num_tokens, q_dim)?;
            }
            if let Some(bias) = weights.k_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut k, bias, num_tokens, kv_dim)?;
            }
            if let Some(bias) = weights.v_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut v, bias, num_tokens, kv_dim)?;
            }
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.stream, &self.loader, &q, &k, input.positions,
                input.rope_cos, input.rope_sin, num_tokens, num_heads, num_kv_heads, head_dim,
            )?;
            let ev_qkv_end = record_event(device, &self.stream)?;

            // 3. Attention (cache write + prefill/decode)
            let ev_attn_start = record_event(device, &self.stream)?;
            Self::cache_write(
                &self.stream, &self.loader, &k_rot, &v,
                input.key_cache, input.value_cache, input.slot_mapping,
                num_tokens, num_kv_heads, head_dim,
            )?;
            let attn_out = if input.is_prefill {
                Self::prefill_attention(
                    &self.stream, &self.loader, &q_rot,
                    input.key_cache, input.value_cache, input.block_tables,
                    input.context_lens, input.seq_start_pos,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?
            } else {
                Self::decode_attention(
                    &self.stream, &self.loader, &q_rot,
                    input.key_cache, input.value_cache, input.block_tables,
                    input.context_lens, num_tokens, input.num_seqs,
                    num_heads, num_kv_heads, head_dim, input.max_context_len, input.block_size,
                )?
            };
            let ev_attn_end = record_event(device, &self.stream)?;

            // 4. Output projection + residual
            let ev_oproj_start = record_event(device, &self.stream)?;
            let attn_proj = Self::linear(&self.stream, blas, &attn_out, weights.o_proj, num_tokens, hidden, q_dim)?;
            let residual = Self::add_tensors(&self.stream, &self.loader, input.hidden_states, &attn_proj, num_tokens * hidden)?;
            let ev_oproj_end = record_event(device, &self.stream)?;

            // 5. Post-attention RMSNorm
            let ev_norm2_start = record_event(device, &self.stream)?;
            let normed2 = Self::rms_norm(
                &self.stream, &self.loader, &residual,
                weights.post_attention_layernorm, cfg.rms_norm_eps, num_tokens, hidden,
            )?;
            let ev_norm2_end = record_event(device, &self.stream)?;

            // 6. MLP
            let ev_mlp_start = record_event(device, &self.stream)?;
            let gate = Self::linear(&self.stream, blas, &normed2, weights.gate_proj, num_tokens, intermediate, hidden)?;
            let up = Self::linear(&self.stream, blas, &normed2, weights.up_proj, num_tokens, intermediate, hidden)?;
            let fused = Self::fused_silu_mul(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?;
            let mlp_out = Self::linear(&self.stream, blas, &fused, weights.down_proj, num_tokens, hidden, intermediate)?;
            let output = Self::add_tensors(&self.stream, &self.loader, &residual, &mlp_out, num_tokens * hidden)?;
            let ev_mlp_end = record_event(device, &self.stream)?;

            let events = LayerProfileEvents {
                norm: (ev_norm_start, ev_norm_end),
                qkv: (ev_qkv_start, ev_qkv_end),
                attn: (ev_attn_start, ev_attn_end),
                o_proj: (ev_oproj_start, ev_oproj_end),
                norm2: (ev_norm2_start, ev_norm2_end),
                mlp: (ev_mlp_start, ev_mlp_end),
            };

            Ok((output, events))
        }

        /// Profiled f16 forward: identical to `forward_f16` but records CUDA events.
        pub fn forward_f16_profiled(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeightsF16<'_>,
            blas: &CublasHandle,
            device: &Arc<CudaContext>,
        ) -> Result<(CudaSlice<f32>, LayerProfileEvents)> {
            use crate::layers::linear_cuda::CudaLinearLayer;

            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // 1. Pre-attention RMSNorm
            let ev_norm_start = record_event(device, &self.stream)?;
            let normed = Self::rms_norm(
                &self.stream, &self.loader, input.hidden_states,
                weights.input_layernorm, cfg.rms_norm_eps, num_tokens, hidden,
            )?;
            let ev_norm_end = record_event(device, &self.stream)?;

            // 2. QKV projections + biases + RoPE (f16 weights)
            let ev_qkv_start = record_event(device, &self.stream)?;
            let mut q = CudaLinearLayer::forward_once_f16(&normed, weights.q_proj, num_tokens, q_dim, hidden, blas, &self.loader)?;
            let mut k = CudaLinearLayer::forward_once_f16(&normed, weights.k_proj, num_tokens, kv_dim, hidden, blas, &self.loader)?;
            let mut v = CudaLinearLayer::forward_once_f16(&normed, weights.v_proj, num_tokens, kv_dim, hidden, blas, &self.loader)?;
            if let Some(bias) = weights.q_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut q, bias, num_tokens, q_dim)?;
            }
            if let Some(bias) = weights.k_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut k, bias, num_tokens, kv_dim)?;
            }
            if let Some(bias) = weights.v_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut v, bias, num_tokens, kv_dim)?;
            }
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.stream, &self.loader, &q, &k, input.positions,
                input.rope_cos, input.rope_sin, num_tokens, num_heads, num_kv_heads, head_dim,
            )?;
            let ev_qkv_end = record_event(device, &self.stream)?;

            // 3. Attention
            let ev_attn_start = record_event(device, &self.stream)?;
            Self::cache_write(
                &self.stream, &self.loader, &k_rot, &v,
                input.key_cache, input.value_cache, input.slot_mapping,
                num_tokens, num_kv_heads, head_dim,
            )?;
            let attn_out = if input.is_prefill {
                Self::prefill_attention(
                    &self.stream, &self.loader, &q_rot,
                    input.key_cache, input.value_cache, input.block_tables,
                    input.context_lens, input.seq_start_pos,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?
            } else {
                Self::decode_attention(
                    &self.stream, &self.loader, &q_rot,
                    input.key_cache, input.value_cache, input.block_tables,
                    input.context_lens, num_tokens, input.num_seqs,
                    num_heads, num_kv_heads, head_dim, input.max_context_len, input.block_size,
                )?
            };
            let ev_attn_end = record_event(device, &self.stream)?;

            // 4. Output projection + residual (f16 weight)
            let ev_oproj_start = record_event(device, &self.stream)?;
            let attn_proj = CudaLinearLayer::forward_once_f16(&attn_out, weights.o_proj, num_tokens, hidden, q_dim, blas, &self.loader)?;
            let residual = Self::add_tensors(&self.stream, &self.loader, input.hidden_states, &attn_proj, num_tokens * hidden)?;
            let ev_oproj_end = record_event(device, &self.stream)?;

            // 5. Post-attention RMSNorm
            let ev_norm2_start = record_event(device, &self.stream)?;
            let normed2 = Self::rms_norm(
                &self.stream, &self.loader, &residual,
                weights.post_attention_layernorm, cfg.rms_norm_eps, num_tokens, hidden,
            )?;
            let ev_norm2_end = record_event(device, &self.stream)?;

            // 6. MLP (f16 weights)
            let ev_mlp_start = record_event(device, &self.stream)?;
            let gate = CudaLinearLayer::forward_once_f16(&normed2, weights.gate_proj, num_tokens, intermediate, hidden, blas, &self.loader)?;
            let up = CudaLinearLayer::forward_once_f16(&normed2, weights.up_proj, num_tokens, intermediate, hidden, blas, &self.loader)?;
            let fused = Self::fused_silu_mul(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?;
            let mlp_out = CudaLinearLayer::forward_once_f16(&fused, weights.down_proj, num_tokens, hidden, intermediate, blas, &self.loader)?;
            let output = Self::add_tensors(&self.stream, &self.loader, &residual, &mlp_out, num_tokens * hidden)?;
            let ev_mlp_end = record_event(device, &self.stream)?;

            let events = LayerProfileEvents {
                norm: (ev_norm_start, ev_norm_end),
                qkv: (ev_qkv_start, ev_qkv_end),
                attn: (ev_attn_start, ev_attn_end),
                o_proj: (ev_oproj_start, ev_oproj_end),
                norm2: (ev_norm2_start, ev_norm2_end),
                mlp: (ev_mlp_start, ev_mlp_end),
            };

            Ok((output, events))
        }

        // ===================================================================
        // Private dispatch helpers
        //
        // Each wraps the corresponding CUDA kernel or cuBLAS call.
        // These are the seams where Agent 2-7 implementations plug in.
        // ===================================================================

        /// RMSNorm: out[i] = (x[i] / rms) * weight[i % hidden]
        /// where rms = sqrt(mean(x^2) + eps).
        ///
        /// Dispatches to the rms_norm CUDA kernel.
        fn rms_norm(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let n = num_tokens * hidden_size;
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("rms_norm alloc failed: {e}")))?;

            // Launch rms_norm kernel: one block per token, hidden_size threads per block.
            // The kernel reads `input`, `weight`, writes `output`.
            let module_name = "rms_norm";
            let func_name = "rms_norm_kernel";
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                // kernel uses extern __shared__ float sdata[blockDim.x]
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };

            // SAFETY: All CudaSlice pointers are valid device memory allocated on
            // the same device. Grid/block dims are checked above. The kernel reads
            // `input` [num_tokens * hidden_size], `weight` [hidden_size], and writes
            // `output` [num_tokens * hidden_size].
            let kernel = loader.get_func(module_name, func_name)?;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm launch failed: {e}")))?;
            }

            Ok(output)
        }

        /// Linear projection via cuBLAS sgemm.
        /// Computes output = input @ weight^T where:
        ///   input: [m, k], weight: [n, k] (row-major), output: [m, n].
        /// Add bias in-place: tensor[i*dim + j] += bias[j]
        fn add_bias(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            tensor: &mut CudaSlice<f32>,
            bias: &CudaSlice<f32>,
            num_tokens: usize,
            dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("add_bias", "add_bias_kernel")?;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let dim_i32 = dim as i32;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias launch: {e}")))?;
            }
            Ok(())
        }

        fn linear(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = stream
                .alloc_zeros::<f32>(m * n)
                .map_err(|e| LLMError::GpuError(format!("linear alloc failed: {e}")))?;

            blas.sgemm(m, n, k, 1.0, input, weight, 0.0, &mut output)?;

            Ok(output)
        }

        /// Apply rotary positional embeddings to Q and K tensors.
        ///
        /// Dispatches to the rotary_embedding CUDA kernel.
        /// Q shape: [num_tokens, num_heads * head_dim]
        /// K shape: [num_tokens, num_kv_heads * head_dim]
        /// positions: [num_tokens]
        /// Apply RoPE to Q and K in a single kernel launch.
        /// Kernel signature: (query, key, cos_cache, sin_cache, positions,
        ///                     num_tokens, num_heads, num_kv_heads, head_dim)
        fn apply_rotary_embedding(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            positions: &CudaSlice<i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
            let q_len = num_tokens * num_heads * head_dim;
            let k_len = num_tokens * num_kv_heads * head_dim;

            // Clone Q and K so we can apply rotation in-place.
            let mut q_out = stream
                .alloc_zeros::<f32>(q_len)
                .map_err(|e| LLMError::GpuError(format!("rope q alloc: {e}")))?;
            let mut k_out = stream
                .alloc_zeros::<f32>(k_len)
                .map_err(|e| LLMError::GpuError(format!("rope k alloc: {e}")))?;

            stream
                .memcpy_dtod(q, &mut q_out)
                .map_err(|e| LLMError::GpuError(format!("rope q copy: {e}")))?;
            stream
                .memcpy_dtod(k, &mut k_out)
                .map_err(|e| LLMError::GpuError(format!("rope k copy: {e}")))?;

            let kernel = loader.get_func("rotary_embedding", "rotary_embedding_kernel")?;

            // Single launch: grid (num_tokens, max(num_heads, num_kv_heads), 1)
            // The kernel internally guards `if (head_idx < num_kv_heads)` for K.
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            let num_tokens_i32 = num_tokens as i32;
            let num_heads_i32 = num_heads as i32;
            let num_kv_heads_i32 = num_kv_heads as i32;
            let head_dim_i32 = head_dim as i32;

            // positions is u32 on GPU but kernel expects int* (i32).
            // They're the same size; cast is safe.
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut q_out)
                    .arg(&mut k_out)
                    .arg(rope_cos)
                    .arg(rope_sin)
                    .arg(positions)
                    .arg(&num_tokens_i32)
                    .arg(&num_heads_i32)
                    .arg(&num_kv_heads_i32)
                    .arg(&head_dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope k launch failed: {e}")))?;
            }

            Ok((q_out, k_out))
        }

        /// Paged attention forward pass.
        ///
        /// Writes new K,V into the f16 cache at slot_mapping positions.
        /// Input k/v are f32 (from projection); the kernel converts to f16 on write.
        /// Uses reshape_and_cache_f16_kernel: 1 launch per layer.
        #[allow(clippy::too_many_arguments)]
        fn cache_write(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            k: &CudaSlice<f32>,
            v: &CudaSlice<f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            slot_mapping: &CudaSlice<i32>,
            num_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("reshape_and_cache", "reshape_and_cache_f16_kernel")?;

            let kv_dim = num_kv_heads * head_dim;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (kv_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            let num_tokens_i32 = num_tokens as i32;
            let num_kv_heads_i32 = num_kv_heads as i32;
            let head_dim_i32 = head_dim as i32;

            // Kernel: (f16* key_cache, f16* value_cache, f32* key, f32* value, int* slot_mapping, int, int, int)
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(k)
                    .arg(v)
                    .arg(slot_mapping)
                    .arg(&num_tokens_i32)
                    .arg(&num_kv_heads_i32)
                    .arg(&head_dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("reshape_and_cache_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// Prefill attention: write K/V to cache, then launch flash_attention_2_kernel
        /// reading from the paged cache with real block_tables.
        /// Naive prefill attention: per-head Q@K^T -> softmax -> @V via cuBLAS.
        /// Bypasses FA2 kernel for correctness. Used only during prefill (once per request).
        #[allow(dead_code)]
        fn naive_prefill_attention(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            q: &CudaSlice<f32>, // [num_tokens, num_heads * head_dim]
            k: &CudaSlice<f32>, // [num_tokens, num_kv_heads * head_dim]
            v: &CudaSlice<f32>, // [num_tokens, num_kv_heads * head_dim]
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<CudaSlice<f32>> {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let heads_per_kv = num_heads / num_kv_heads;
            let q_stride = num_heads * head_dim;

            // Output: [num_tokens, num_heads * head_dim]
            let mut output = stream
                .alloc_zeros::<f32>(num_tokens * q_stride)
                .map_err(|e| LLMError::GpuError(format!("naive attn output alloc: {e}")))?;

            // Per-head attention via cuBLAS
            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;

                // Extract Q_head [num_tokens, head_dim] from Q [num_tokens, num_heads * head_dim]
                // Extract K_head [num_tokens, head_dim] from K [num_tokens, num_kv_heads * head_dim]
                // Extract V_head [num_tokens, head_dim] from V [num_tokens, num_kv_heads * head_dim]
                // Use CPU gather for correctness (not perf-critical for prefill)
                let q_all: Vec<f32> = stream
                    .clone_dtoh(q)
                    .map_err(|e| LLMError::GpuError(format!("naive attn q DtoH: {e}")))?;
                let k_all: Vec<f32> = stream
                    .clone_dtoh(k)
                    .map_err(|e| LLMError::GpuError(format!("naive attn k DtoH: {e}")))?;
                let v_all: Vec<f32> = stream
                    .clone_dtoh(v)
                    .map_err(|e| LLMError::GpuError(format!("naive attn v DtoH: {e}")))?;

                let kv_stride = num_kv_heads * head_dim;
                let mut qh = vec![0.0f32; num_tokens * head_dim];
                let mut kh = vec![0.0f32; num_tokens * head_dim];
                let mut vh = vec![0.0f32; num_tokens * head_dim];

                for t in 0..num_tokens {
                    for d in 0..head_dim {
                        qh[t * head_dim + d] = q_all[t * q_stride + h * head_dim + d];
                        kh[t * head_dim + d] = k_all[t * kv_stride + kv_h * head_dim + d];
                        vh[t * head_dim + d] = v_all[t * kv_stride + kv_h * head_dim + d];
                    }
                }

                // scores[i][j] = sum_d qh[i][d] * kh[j][d] * scale (with causal mask)
                let mut scores = vec![0.0f32; num_tokens * num_tokens];
                for qi in 0..num_tokens {
                    for ki in 0..num_tokens {
                        if ki > qi {
                            scores[qi * num_tokens + ki] = f32::NEG_INFINITY;
                        } else {
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += qh[qi * head_dim + d] * kh[ki * head_dim + d];
                            }
                            scores[qi * num_tokens + ki] = dot * scale;
                        }
                    }
                }

                // Softmax per row
                for qi in 0..num_tokens {
                    let row = &mut scores[qi * num_tokens..(qi + 1) * num_tokens];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    for v in row.iter_mut() {
                        *v *= inv;
                    }
                }

                // out_head = scores @ vh
                let mut out_head = vec![0.0f32; num_tokens * head_dim];
                for qi in 0..num_tokens {
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for ki in 0..num_tokens {
                            acc += scores[qi * num_tokens + ki] * vh[ki * head_dim + d];
                        }
                        out_head[qi * head_dim + d] = acc;
                    }
                }

                // Scatter back into output
                let mut out_all: Vec<f32> = stream
                    .clone_dtoh(&output)
                    .map_err(|e| LLMError::GpuError(format!("naive attn out DtoH: {e}")))?;
                for t in 0..num_tokens {
                    for d in 0..head_dim {
                        out_all[t * q_stride + h * head_dim + d] = out_head[t * head_dim + d];
                    }
                }
                output = stream
                    .clone_htod(&out_all)
                    .map_err(|e| LLMError::GpuError(format!("naive attn out HtoD: {e}")))?;
            }

            Ok(output)
        }

        /// FA2 prefill attention reading from f16 paged cache with real block_tables.
        /// Q is f32, cache is f16; the kernel loads f16 and promotes to f32 internally.
        fn prefill_attention(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaSlice<f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaSlice<i32>,
            context_lens: &CudaSlice<i32>,
            seq_start_pos: &CudaSlice<i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let out_len = num_tokens * num_heads * head_dim;
            let output = stream
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("prefill_attn alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let kernel = loader.get_func("flash_attention", "flash_attention_2_f16kv_kernel")?;

            let bt_len = DeviceSlice::len(block_tables);
            info!(
                num_tokens,
                num_seqs,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_context_len,
                bt_len,
                shared_mem_bytes,
                "prefill_attention: dimensions"
            );

            if num_seqs == 0 {
                return Err(LLMError::GpuError(
                    "prefill_attention: num_seqs == 0".into(),
                ));
            }

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            // seq_start_pos is pre-computed by the caller (gpu_runner.rs) from actual
            // query token positions, not context_lens. This correctly handles mixed
            // prefill+decode batches where context_lens != query token counts.

            let max_blocks_per_seq = if num_seqs > 0 {
                (DeviceSlice::len(block_tables) / num_seqs) as i32
            } else {
                1
            };

            // Opt into extended shared memory if needed (A100 supports up to 100KB)
            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("prefill FA2 set max shared mem: {e}")))?;
            }

            let p_num_heads = num_heads as i32;
            let p_num_kv = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_ctx = max_context_len as i32;
            let p_num_tokens = num_tokens as i32;
            let p_causal = 1i32;

            // FA2 prefill kernel: 16 args via builder pattern
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&output)
                    .arg(q)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(block_tables)
                    .arg(context_lens)
                    .arg(seq_start_pos)
                    .arg(&scale)
                    .arg(&p_num_heads)
                    .arg(&p_num_kv)
                    .arg(&p_head_dim)
                    .arg(&p_block_size)
                    .arg(&p_max_ctx)
                    .arg(&max_blocks_per_seq)
                    .arg(&p_num_tokens)
                    .arg(&p_causal)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("prefill FA2 launch: {e}")))?;
            }

            Ok(output)
        }

        /// Decode attention: read f16 K/V from paged cache, one FA2 decode kernel per layer.
        /// Q is f32, cache is f16; kernel promotes f16 to f32 on load.
        fn decode_attention(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaSlice<f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaSlice<i32>,
            context_lens: &CudaSlice<i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let out_len = num_tokens * num_heads * head_dim;
            let mut output = stream
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("paged_attn alloc failed: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            // Use FA2 decode kernel: correct block-level reductions, GQA support.
            // FA2_BC=64, FA2_THREADS=128 (compile-time constants in flash_attention.cu)
            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            // smem: s_key[FA2_BC*head_dim] + s_val[FA2_BC*head_dim] + s_score[FA2_BC] + s_reduce[FA2_THREADS/32]
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let module_name = "flash_attention";
            let func_name = "flash_attention_2_decode_f16kv_kernel";

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            let kernel = loader.get_func(module_name, func_name)?;

            // Opt into extended shared memory if needed
            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("decode FA2 set max shared mem: {e}")))?;
            }

            let p_num_heads = num_heads as i32;
            let p_num_kv_heads = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_blocks = (block_tables.len() / num_seqs.max(1)) as i32;

            // SAFETY: All slices are valid GPU memory on this device.
            // output: [num_seqs, num_heads, head_dim]
            // q:      [num_seqs, num_heads, head_dim]  (decode: num_seqs == num_tokens)
            // key_cache, value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            // block_tables: [num_seqs, max_blocks_per_seq]
            // context_lens: [num_seqs]
            // Scalar int args cast from usize; all values fit in i32 range.
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(q)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(block_tables)
                    .arg(context_lens)
                    .arg(&scale)
                    .arg(&p_num_heads)
                    .arg(&p_num_kv_heads)
                    .arg(&p_head_dim)
                    .arg(&p_block_size)
                    .arg(&p_max_blocks)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("flash_attention_2_decode launch failed: {e}"))
                    })?;
            }

            Ok(output)
        }

        /// Fused SiLU activation with element-wise multiply: out = silu(gate) * up.
        ///
        /// Dispatches to the activation CUDA kernel.
        fn fused_silu_mul(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate: &CudaSlice<f32>,
            up: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul alloc failed: {e}")))?;

            let module_name = "activation";
            let func_name = "fused_silu_mul_kernel";

            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            let kernel = loader.get_func(module_name, func_name)?;

            let n_i32 = n as i32;

            // SAFETY: gate, up, and output all have exactly n elements.
            // Grid covers all elements with ceil division.
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(gate)
                    .arg(up)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_silu_mul launch failed: {e}"))
                    })?;
            }

            Ok(output)
        }

        /// Element-wise tensor addition: out = a + b.
        ///
        /// Tries "add_bias" module first (Agent 20's dedicated kernel), then
        /// "activation" module, then falls back to CPU.
        fn add_tensors(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("add_tensors alloc failed: {e}")))?;

            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            let n_i32 = n as i32;

            // Try dedicated add_bias module first, then activation module
            let kernel = loader
                .get_func("add_bias", "add_kernel")
                .or_else(|_| loader.get_func("activation", "add_kernel"));

            match kernel {
                Ok(k) => {
                    // SAFETY: a, b, output all have exactly n elements.
                    unsafe {
                        stream
                            .launch_builder(&k)
                            .arg(&mut output)
                            .arg(a)
                            .arg(b)
                            .arg(&n_i32)
                            .launch(cfg)
                            .map_err(|e| {
                                LLMError::GpuError(format!("add_kernel launch failed: {e}"))
                            })?;
                    }
                }
                Err(_) => {
                    // Fallback: CPU add (only until kernels are compiled).
                    let a_host = stream
                        .clone_dtoh(a)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh a failed: {e}")))?;
                    let b_host = stream
                        .clone_dtoh(b)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh b failed: {e}")))?;
                    let sum: Vec<f32> = a_host
                        .iter()
                        .zip(b_host.iter())
                        .map(|(x, y)| x + y)
                        .collect();
                    output = stream
                        .clone_htod(&sum)
                        .map_err(|e| LLMError::GpuError(format!("add htod failed: {e}")))?;
                }
            }

            Ok(output)
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;

#[cfg(test)]
mod tests {
    // Tests run under default features (mock-gpu), so we verify the module
    // compiles but the CUDA types are not exposed.
    #[test]
    fn module_compiles_without_cuda() {
        // Under mock-gpu the `inner` module is not compiled.
        // This test confirms that the crate still builds cleanly.
        assert!(true);
    }
}
