//! GPU Transformer Layer -- one complete transformer block on CUDA (f16 only).
//!
//! All code is gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtrMut, DeviceSlice, LaunchConfig, PushKernelArg};
    use half::f16;
    use tracing::{info, trace};

    use rvllm_core::error::{LLMError, Result};
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::kernel_loader::KernelLoader;

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

    /// Weight references for a single transformer layer (all f16).
    ///
    /// All slices live on GPU and are owned by the GpuModelWeights container.
    /// This struct borrows them for the duration of a forward pass.
    pub struct GpuLayerWeights<'a> {
        // Pre-attention norm
        pub input_layernorm: &'a CudaSlice<f16>,
        // Attention projections
        pub q_proj: &'a CudaSlice<f16>,
        pub k_proj: &'a CudaSlice<f16>,
        pub v_proj: &'a CudaSlice<f16>,
        pub o_proj: &'a CudaSlice<f16>,
        // Post-attention norm
        pub post_attention_layernorm: &'a CudaSlice<f16>,
        // MLP weights
        pub gate_proj: &'a CudaSlice<f16>,
        pub up_proj: &'a CudaSlice<f16>,
        pub down_proj: &'a CudaSlice<f16>,
        /// Fused QKV weight: [q_dim + kv_dim + kv_dim, hidden]. None if not fused.
        pub fused_qkv: Option<&'a CudaSlice<f16>>,
        /// Fused gate+up weight: [intermediate*2, hidden]. None if not fused.
        pub fused_gate_up: Option<&'a CudaSlice<f16>>,
        /// Fused QKV bias (f16). None if model has no QKV bias.
        pub qkv_bias: Option<&'a CudaSlice<f16>>,
        // FP8 quantized weights (u8 data + f16 per-row scales).
        // Present when RVLLM_FP8_WEIGHTS=1.
        pub fused_qkv_fp8: Option<&'a CudaSlice<u8>>,
        pub fused_qkv_fp8_scale: Option<&'a CudaSlice<f16>>,
    }

    /// Metadata needed for a single layer forward pass.
    pub struct GpuLayerInput<'a> {
        /// Hidden states entering this layer, shape [num_tokens, hidden_size] (f16).
        pub hidden_states: &'a CudaSlice<f16>,
        /// Position ids for RoPE, shape [num_tokens]. Kernels expect int*.
        pub positions: CudaView<'a, i32>,
        /// KV cache key block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub key_cache: &'a CudaSlice<f16>,
        /// KV cache value block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub value_cache: &'a CudaSlice<f16>,
        /// Block table mapping sequence positions to cache blocks, shape [num_seqs, max_blocks_per_seq].
        pub block_tables: CudaView<'a, i32>,
        /// Context length for each sequence, shape [num_seqs].
        pub context_lens: CudaView<'a, i32>,
        /// Slot mapping for cache writes during prefill, shape [num_tokens].
        pub slot_mapping: CudaView<'a, i32>,
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
        pub seq_start_pos: CudaView<'a, i32>,
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
        /// Fully f16 forward pass -- ZERO type casts within the layer.
        /// All inputs, intermediates, and outputs are f16.
        //   B) qkv [T*(Q+2K)] dead at step 5 -> reuse for gate_up [T*2I] at step 9
        //      (qkv_dim ~ H+2*H/GQA_ratio, gate_up = 2*I; for llama 2I > qkv, but
        //       qkv buf could be a prefix if gate_up scratch is sized to max(qkv,2I))
        //   C) attn_out [T*Q] dead at step 7 -> reuse for fused/silu [T*I] at step 9
        //      (I > Q typically, so need max(Q,I) sizing)
        //   D) attn_proj [T*H] dead at step 8 -> reuse for mlp_out [T*H] at step 10
        //      (exact same size)
        //   E) normed2 [T*H] dead at step 9 -> reuse for mlp_out [T*H] at step 10
        //      (exact same size, alternative to D)
        //
        // LOW-HANGING FRUIT (same size, zero waste):
        //   1. attn_proj -> mlp_out  [both T*H, saves 1 alloc]
        //   2. normed -> attn_out    [T*H vs T*Q, same when Q=H, saves 1 alloc]
        //   3. normed2 is an alternative to attn_proj for mlp_out reuse
        //
        // With a single scratch buffer sized to max(T*qkv_dim, T*2I, T*H):
        //   Could eliminate 4-5 allocs, reducing 10 -> 5-6 per layer.
        // ==================================================================
        /// Full f16 forward with cross-layer fusion support.
        /// When `prev_mlp_out` is Some, fuses the previous layer's residual add
        /// with this layer's pre-attention RMSNorm (1 kernel instead of 2).
        /// Returns (residual, mlp_out) for the NEXT layer to fuse.
        pub fn forward(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeights<'_>,
            blas: &CublasHandle,
            prev_mlp_out: Option<&CudaSlice<f16>>,
            lt: Option<&crate::CublasLtRef>,
        ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            let hidden_f16 = input.hidden_states;
            let norm_w = weights.input_layernorm;
            let post_norm_w = weights.post_attention_layernorm;
            let dbg = cfg.layer_idx < 1 && std::env::var("RVLLM_DEBUG").is_ok();
            let dbg_dump = |label: &str, buf: &CudaSlice<f16>, stream: &Arc<CudaStream>| {
                if let Ok(vals) = stream.clone_dtoh(buf) {
                    let first5: Vec<f32> = vals.iter().take(5).map(|v| v.to_f32()).collect();
                    let last5: Vec<f32> = vals.iter().rev().take(5).rev().map(|v| v.to_f32()).collect();
                    let nan = vals.iter().any(|v| v.to_f32().is_nan());
                    let max = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    let mean = vals.iter().map(|v| v.to_f32()).sum::<f32>() / vals.len() as f32;
                    info!("DEBUG L0 {label}: first5={first5:?} last5={last5:?} max={max:.4} mean={mean:.6} nan={nan} len={}", vals.len());
                }
            };

            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;
            let q_end = num_tokens * q_dim;
            let k_end = q_end + num_tokens * kv_dim;

            // ================================================================
            // T=1 DECODE: maximally fused path
            // Fuses: add+norm+QKV GEMV, bias, RoPE+cache_write, attn, add+norm+gateup, silu+down
            // ================================================================
            if num_tokens == 1 && !input.is_prefill {
                // --- Step 1+2: Fused add+norm+QKV GEMV or norm+QKV GEMV ---
                let (mut qkv, residual_ref, bias_fused) = if let Some(prev_mlp) = prev_mlp_out {
                    // Try fused add+norm+QKV GEMV (3-way: add + norm + projection)
                    let fused_qkv_w = weights.fused_qkv.unwrap_or(weights.q_proj);
                    // FP8 fused variant (halves weight bandwidth)
                    if let (Some(fp8_w), Some(fp8_s), Ok(ref fk)) = (
                        weights.fused_qkv_fp8,
                        weights.fused_qkv_fp8_scale,
                        self.loader.get_func("gemv_fp8", "fused_add_norm_fp8_gemv_kernel"),
                    ) {
                        let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                            .map_err(|e| LLMError::GpuError(format!("fp8 qkv alloc: {e}")))?;
                        let mut residual_out = unsafe { self.stream.alloc::<f16>(hidden) }
                            .map_err(|e| LLMError::GpuError(format!("fp8 residual alloc: {e}")))?;
                        let smem = (hidden * 4 + 8 * 4) as u32;
                        let rpb = 8u32;
                        if smem > 49152 {
                            fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                                .map_err(|e| LLMError::GpuError(format!("fp8 smem attr: {e}")))?;
                        }
                        unsafe {
                            self.stream.launch_builder(fk)
                                .arg(&mut qkv_out).arg(&mut residual_out)
                                .arg(hidden_f16).arg(prev_mlp).arg(norm_w)
                                .arg(fp8_w).arg(fp8_s)
                                .arg(&cfg.rms_norm_eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                                .launch(LaunchConfig { grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                                .map_err(|e| LLMError::GpuError(format!("fp8 add_norm_qkv: {e}")))?;
                        }
                        (qkv_out, residual_out, false)
                    // f16 bias-fused variant
                    } else if let (Some(qkv_bias), Ok(ref fk)) = (weights.qkv_bias, self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_add_norm_qkv_bias_gemv")) {
                        let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                            .map_err(|e| LLMError::GpuError(format!("fused qkv alloc: {e}")))?;
                        let mut residual_out = unsafe { self.stream.alloc::<f16>(hidden) }
                            .map_err(|e| LLMError::GpuError(format!("fused residual alloc: {e}")))?;
                        let smem = (hidden * 4 + 8 * 4) as u32;
                        let rpb = 8u32;
                        if smem > 49152 {
                            fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                                .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
                        }
                        unsafe {
                            self.stream.launch_builder(fk)
                                .arg(&mut qkv_out).arg(&mut residual_out)
                                .arg(hidden_f16).arg(prev_mlp).arg(norm_w).arg(fused_qkv_w)
                                .arg(qkv_bias)
                                .arg(&cfg.rms_norm_eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                                .launch(LaunchConfig { grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                                .map_err(|e| LLMError::GpuError(format!("fused add_norm_qkv_bias: {e}")))?;
                        }
                        (qkv_out, residual_out, true)
                    } else if let Ok(ref fk) = self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_add_norm_qkv_gemv") {
                        let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                            .map_err(|e| LLMError::GpuError(format!("fused qkv alloc: {e}")))?;
                        let mut residual_out = unsafe { self.stream.alloc::<f16>(hidden) }
                            .map_err(|e| LLMError::GpuError(format!("fused residual alloc: {e}")))?;
                        let smem = (hidden * 4 + 8 * 4) as u32;
                        let rpb = 8u32;
                        if smem > 49152 {
                            fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                                .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
                        }
                        unsafe {
                            self.stream.launch_builder(fk)
                                .arg(&mut qkv_out).arg(&mut residual_out)
                                .arg(hidden_f16).arg(prev_mlp).arg(norm_w).arg(fused_qkv_w)
                                .arg(&cfg.rms_norm_eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                                .launch(LaunchConfig { grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                                .map_err(|e| LLMError::GpuError(format!("fused add_norm_qkv: {e}")))?;
                        }
                        (qkv_out, residual_out, false)
                    } else {
                        // Fallback: separate add+norm then QKV
                        let (normed, residual) = Self::fused_residual_rmsnorm_f16(
                            &self.stream, &self.loader, hidden_f16, prev_mlp, norm_w,
                            cfg.rms_norm_eps, num_tokens, hidden)?;
                        let qkv = Self::hgemm_dispatch(&self.stream, blas, lt, &normed, fused_qkv_w, 1, qkv_dim, hidden, &self.loader)?;
                        (qkv, residual, false)
                    }
                } else {
                    // First layer: norm+QKV GEMV
                    let fused_qkv_w = weights.fused_qkv.unwrap_or(weights.q_proj);
                    // Try bias-fused variant first if model has QKV bias
                    if let (Some(qkv_bias), Ok(ref fk)) = (weights.qkv_bias, self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_norm_qkv_bias_gemv")) {
                        let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                            .map_err(|e| LLMError::GpuError(format!("fused qkv alloc: {e}")))?;
                        let smem = (hidden * 4 + 8 * 4) as u32;
                        let rpb = 8u32;
                        if smem > 49152 {
                            fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                                .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
                        }
                        unsafe {
                            self.stream.launch_builder(fk)
                                .arg(&mut qkv_out).arg(hidden_f16).arg(norm_w).arg(fused_qkv_w)
                                .arg(qkv_bias)
                                .arg(&cfg.rms_norm_eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                                .launch(LaunchConfig { grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                                .map_err(|e| LLMError::GpuError(format!("fused norm_qkv_bias: {e}")))?;
                        }
                        let residual = hidden_f16.clone();
                        (qkv_out, residual, true)
                    } else if let Ok(ref fk) = self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_norm_qkv_gemv") {
                        let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                            .map_err(|e| LLMError::GpuError(format!("fused qkv alloc: {e}")))?;
                        let smem = (hidden * 4 + 8 * 4) as u32;
                        let rpb = 8u32;
                        if smem > 49152 {
                            fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                                .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
                        }
                        unsafe {
                            self.stream.launch_builder(fk)
                                .arg(&mut qkv_out).arg(hidden_f16).arg(norm_w).arg(fused_qkv_w)
                                .arg(&cfg.rms_norm_eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                                .launch(LaunchConfig { grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                                .map_err(|e| LLMError::GpuError(format!("fused norm_qkv: {e}")))?;
                        }
                        let residual = hidden_f16.clone();
                        (qkv_out, residual, false)
                    } else {
                        let normed = Self::rms_norm_f16(&self.stream, &self.loader, hidden_f16, norm_w, cfg.rms_norm_eps, 1, hidden)?;
                        let qkv = Self::hgemm_dispatch(&self.stream, blas, lt, &normed, fused_qkv_w, 1, qkv_dim, hidden, &self.loader)?;
                        let residual = hidden_f16.clone();
                        (qkv, residual, false)
                    }
                };

                // --- Step 3: QKV bias (if model has it and not already fused) ---
                if !bias_fused {
                    if let Some(bias) = weights.qkv_bias {
                        let mut qkv_view = qkv.slice_mut(..qkv_dim);
                        Self::add_bias_f16_view(&self.stream, &self.loader, &mut qkv_view, bias, 1, qkv_dim)?;
                    }
                }

                // --- Step 4+5: Fused RoPE + KV cache write ---
                if let Ok(ref fk) = self.loader.get_func("fused_rope_cache", "fused_rope_cache_f16_kernel") {
                    let (mut q_part, mut kv_rest) = qkv.split_at_mut(q_dim);
                    let (mut k_part, v_part) = kv_rest.split_at_mut(kv_dim);
                    let v_view = v_part.slice(..kv_dim);
                    let half_dim = head_dim / 2;
                    let grid_y = num_heads.max(num_kv_heads) as u32;
                    unsafe {
                        self.stream.launch_builder(fk)
                            .arg(&mut q_part).arg(&mut k_part).arg(&v_view)
                            .arg(input.key_cache).arg(input.value_cache)
                            .arg(input.rope_cos).arg(input.rope_sin)
                            .arg(&input.positions).arg(&input.slot_mapping)
                            .arg(&(num_tokens as i32)).arg(&(num_heads as i32))
                            .arg(&(num_kv_heads as i32)).arg(&(head_dim as i32))
                            .launch(LaunchConfig { grid_dim: (num_tokens as u32, grid_y, 1), block_dim: (half_dim.min(1024) as u32, 1, 1), shared_mem_bytes: 0 })
                            .map_err(|e| LLMError::GpuError(format!("fused rope+cache: {e}")))?;
                    }
                } else {
                    // Fallback: separate RoPE then cache write
                    {
                        let (mut q_part, mut kv_part) = qkv.split_at_mut(q_dim);
                        let mut k_view = kv_part.slice_mut(..kv_dim);
                        Self::apply_rotary_embedding_f16_views(
                            &self.stream, &self.loader, &mut q_part, &mut k_view,
                            &input.positions, input.rope_cos, input.rope_sin,
                            num_tokens, num_heads, num_kv_heads, head_dim)?;
                    }
                    {
                        let k_view = qkv.slice(q_dim..q_dim + kv_dim);
                        let v_view = qkv.slice(q_dim + kv_dim..);
                        Self::cache_write_f16_views(
                            &self.stream, &self.loader, &k_view, &v_view,
                            input.key_cache, input.value_cache, &input.slot_mapping,
                            num_tokens, num_kv_heads, head_dim)?;
                    }
                }

                // --- Step 6: Attention (FA3) ---
                let attn_out = Self::decode_attention_f16io(
                    &self.stream, &self.loader,
                    &qkv.slice(..q_dim),
                    input.key_cache, input.value_cache,
                    &input.block_tables, &input.context_lens,
                    1, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size)?;

                // --- Steps 7-10: Try mega-fused O-proj+add+norm+gateup, fall back to separate ---
                let (residual, mlp_out) = {
                    let mega_ok = self.loader.get_func("fused_oproj_add_norm_gateup_gemv", "fused_cute_oproj_add_norm_gateup_gemv").ok();
                    let fused_gate_up_w = weights.fused_gate_up.unwrap_or(weights.gate_proj);
                    if let Some(ref mk) = mega_ok {
                        // Mega kernel: O-proj + add + norm + gateup in one launch
                        let gate_up_dim = intermediate * 2;
                        let mut gate_up_out = unsafe { self.stream.alloc::<f16>(gate_up_dim) }
                            .map_err(|e| LLMError::GpuError(format!("mega gateup alloc: {e}")))?;
                        let mut residual_out2 = unsafe { self.stream.alloc::<f16>(hidden) }
                            .map_err(|e| LLMError::GpuError(format!("mega residual alloc: {e}")))?;
                        let smem = (hidden * 4 + 8 * 4) as u32;
                        let rpb = 8u32;
                        if smem > 49152 {
                            mk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                                .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
                        }
                        unsafe {
                            self.stream.launch_builder(mk)
                                .arg(&mut gate_up_out).arg(&mut residual_out2)
                                .arg(&attn_out).arg(weights.o_proj).arg(&residual_ref)
                                .arg(post_norm_w).arg(fused_gate_up_w)
                                .arg(&cfg.rms_norm_eps).arg(&(q_dim as i32)).arg(&(hidden as i32)).arg(&(gate_up_dim as i32))
                                .launch(LaunchConfig { grid_dim: ((gate_up_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                                .map_err(|e| LLMError::GpuError(format!("mega oproj_add_norm_gateup: {e}")))?;
                        }
                        let gate = gate_up_out.slice(..intermediate);
                        let up = gate_up_out.slice(intermediate..gate_up_dim);
                        let mlp = if let Ok(ref sk) = self.loader.get_func("fused_silu_down_gemv", "fused_cute_silu_down_gemv") {
                            let mut down_out = unsafe { self.stream.alloc::<f16>(hidden) }
                                .map_err(|e| LLMError::GpuError(format!("fused silu_down alloc: {e}")))?;
                            let sk_smem = (8 * 4) as u32;
                            unsafe {
                                self.stream.launch_builder(sk)
                                    .arg(&mut down_out).arg(&gate).arg(&up).arg(weights.down_proj)
                                    .arg(&(hidden as i32)).arg(&(intermediate as i32))
                                    .launch(LaunchConfig { grid_dim: (((hidden as u32) + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: sk_smem })
                                    .map_err(|e| LLMError::GpuError(format!("fused silu_down: {e}")))?;
                            }
                            down_out
                        } else {
                            let fused_act = Self::fused_silu_mul_f16_split(&self.stream, &self.loader, &gate_up_out, intermediate)?;
                            Self::hgemm_dispatch(&self.stream, blas, lt, &fused_act, weights.down_proj, 1, hidden, intermediate, &self.loader)?
                        };
                        (residual_out2, mlp)
                    } else {
                        // Fallback: separate O-proj + fused add+norm+gateup
                        let attn_proj = Self::hgemm_dispatch(&self.stream, blas, lt, &attn_out, weights.o_proj, 1, hidden, q_dim, &self.loader)?;
                        let fused_gateup_ok = self.loader.get_func("fused_add_norm_gateup_gemv", "fused_cute_add_norm_gateup_gemv").ok();
                        if let Some(ref fk) = fused_gateup_ok {
                            let gate_up_dim = intermediate * 2;
                            let mut gate_up_out = unsafe { self.stream.alloc::<f16>(gate_up_dim) }
                                .map_err(|e| LLMError::GpuError(format!("fused gateup alloc: {e}")))?;
                            let mut residual_out2 = unsafe { self.stream.alloc::<f16>(hidden) }
                                .map_err(|e| LLMError::GpuError(format!("fused residual alloc: {e}")))?;
                            let smem = (hidden * 4 + 8 * 4) as u32;
                            let rpb = 8u32;
                            if smem > 49152 {
                                fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                                    .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
                            }
                            unsafe {
                                self.stream.launch_builder(fk)
                                    .arg(&mut gate_up_out).arg(&mut residual_out2)
                                    .arg(&residual_ref).arg(&attn_proj).arg(post_norm_w).arg(fused_gate_up_w)
                                    .arg(&cfg.rms_norm_eps).arg(&(hidden as i32)).arg(&(gate_up_dim as i32))
                                    .launch(LaunchConfig { grid_dim: ((gate_up_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                                    .map_err(|e| LLMError::GpuError(format!("fused add_norm_gateup: {e}")))?;
                            }
                            let gate = gate_up_out.slice(..intermediate);
                            let up = gate_up_out.slice(intermediate..gate_up_dim);
                            let mlp = if let Ok(ref sk) = self.loader.get_func("fused_silu_down_gemv", "fused_cute_silu_down_gemv") {
                                let mut down_out = unsafe { self.stream.alloc::<f16>(hidden) }
                                    .map_err(|e| LLMError::GpuError(format!("fused silu_down alloc: {e}")))?;
                                let sk_smem = (8 * 4) as u32;
                                unsafe {
                                    self.stream.launch_builder(sk)
                                        .arg(&mut down_out).arg(&gate).arg(&up).arg(weights.down_proj)
                                        .arg(&(hidden as i32)).arg(&(intermediate as i32))
                                        .launch(LaunchConfig { grid_dim: (((hidden as u32) + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: sk_smem })
                                        .map_err(|e| LLMError::GpuError(format!("fused silu_down: {e}")))?;
                                }
                                down_out
                            } else {
                                let fused_act = Self::fused_silu_mul_f16_split(&self.stream, &self.loader, &gate_up_out, intermediate)?;
                                Self::hgemm_dispatch(&self.stream, blas, lt, &fused_act, weights.down_proj, 1, hidden, intermediate, &self.loader)?
                            };
                            (residual_out2, mlp)
                        } else {
                            let (normed2, residual2) = Self::fused_residual_rmsnorm_f16(
                                &self.stream, &self.loader, &residual_ref, &attn_proj, post_norm_w,
                                cfg.rms_norm_eps, 1, hidden)?;
                            let gate_up = if let Some(fguw) = weights.fused_gate_up {
                                Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, fguw, 1, intermediate * 2, hidden, &self.loader)?
                            } else {
                                let g = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.gate_proj, 1, intermediate, hidden, &self.loader)?;
                                let u = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.up_proj, 1, intermediate, hidden, &self.loader)?;
                                let mut buf = unsafe { self.stream.alloc::<f16>(intermediate * 2) }
                                    .map_err(|e| LLMError::GpuError(format!("concat: {e}")))?;
                                self.stream.memcpy_dtod(&g, &mut buf.slice_mut(..intermediate)).map_err(|e| LLMError::GpuError(format!("g: {e}")))?;
                                self.stream.memcpy_dtod(&u, &mut buf.slice_mut(intermediate..)).map_err(|e| LLMError::GpuError(format!("u: {e}")))?;
                                buf
                            };
                            let fused_act = Self::fused_silu_mul_f16_split(&self.stream, &self.loader, &gate_up, intermediate)?;
                            let mlp = Self::hgemm_dispatch(&self.stream, blas, lt, &fused_act, weights.down_proj, 1, hidden, intermediate, &self.loader)?;
                            (residual2, mlp)
                        }
                    }
                };
                return Ok((residual, mlp_out));
            }

            // ================================================================
            // T>1 PREFILL: unfused path (cuBLAS GEMMs, separate kernels)
            // ================================================================

            // 1. Pre-attention RMSNorm
            let (normed, fused_residual) = if let Some(prev_mlp) = prev_mlp_out {
                let (n, r) = Self::fused_residual_rmsnorm_f16(
                    &self.stream, &self.loader, hidden_f16, prev_mlp, norm_w,
                    cfg.rms_norm_eps, num_tokens, hidden)?;
                (n, Some(r))
            } else {
                let n = Self::rms_norm_f16(&self.stream, &self.loader, hidden_f16, norm_w,
                    cfg.rms_norm_eps, num_tokens, hidden)?;
                (n, None)
            };
            let residual_ref = fused_residual.as_ref().unwrap_or(hidden_f16);

            // 2. QKV projections
            // Fused GEMM [N, hidden] x [hidden, qkv_dim]^T + transpose for all N
            // Fallback: 3 separate GEMMs when fused weight unavailable
            let mut qkv = if let Some(fused_qkv) = weights.fused_qkv {
                let qkv_interleaved = Self::hgemm_dispatch(&self.stream, blas, lt, &normed, fused_qkv, num_tokens, qkv_dim, hidden, &self.loader)?;
                if let Ok(transpose_fn) = self.loader.get_func("qkv_transpose", "qkv_transpose_f16_kernel") {
                    let mut qkv_transposed = unsafe { self.stream.alloc::<f16>(num_tokens * qkv_dim) }
                        .map_err(|e| LLMError::GpuError(format!("qkv transpose alloc: {e}")))?;
                    let total = (num_tokens * qkv_dim) as u32;
                    let cfg = LaunchConfig { grid_dim: ((total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        self.stream.launch_builder(&transpose_fn)
                            .arg(&mut qkv_transposed).arg(&qkv_interleaved)
                            .arg(&(num_tokens as i32)).arg(&(q_dim as i32)).arg(&(kv_dim as i32))
                            .launch(cfg)
                            .map_err(|e| LLMError::GpuError(format!("qkv transpose: {e}")))?;
                    }
                    qkv_transposed
                } else {
                    let mut qkv_buf = unsafe { self.stream.alloc::<f16>(num_tokens * qkv_dim) }
                        .map_err(|e| LLMError::GpuError(format!("qkv alloc: {e}")))?;
                    let q_end_t = num_tokens * q_dim;
                    let k_end_t = q_end_t + num_tokens * kv_dim;
                    { let mut d = qkv_buf.slice_mut(..q_end_t); Self::hgemm_dispatch_into(blas, lt, &normed, weights.q_proj, num_tokens, q_dim, hidden, &mut d)?; }
                    { let mut d = qkv_buf.slice_mut(q_end_t..k_end_t); Self::hgemm_dispatch_into(blas, lt, &normed, weights.k_proj, num_tokens, kv_dim, hidden, &mut d)?; }
                    { let mut d = qkv_buf.slice_mut(k_end_t..); Self::hgemm_dispatch_into(blas, lt, &normed, weights.v_proj, num_tokens, kv_dim, hidden, &mut d)?; }
                    qkv_buf
                }
            } else {
                let mut qkv_buf = unsafe { self.stream.alloc::<f16>(num_tokens * qkv_dim) }
                    .map_err(|e| LLMError::GpuError(format!("qkv alloc: {e}")))?;
                let q_end_t = num_tokens * q_dim;
                let k_end_t = q_end_t + num_tokens * kv_dim;
                { let mut d = qkv_buf.slice_mut(..q_end_t); Self::hgemm_dispatch_into(blas, lt, &normed, weights.q_proj, num_tokens, q_dim, hidden, &mut d)?; }
                { let mut d = qkv_buf.slice_mut(q_end_t..k_end_t); Self::hgemm_dispatch_into(blas, lt, &normed, weights.k_proj, num_tokens, kv_dim, hidden, &mut d)?; }
                { let mut d = qkv_buf.slice_mut(k_end_t..); Self::hgemm_dispatch_into(blas, lt, &normed, weights.v_proj, num_tokens, kv_dim, hidden, &mut d)?; }
                qkv_buf
            };

            // 3. QKV bias -- broadcast bias[qkv_dim] across N tokens in one kernel per section
            if let Some(bias) = weights.qkv_bias {
                if let Ok(ref bk) = self.loader.get_func("add_bias_broadcast", "add_bias_broadcast_f16_kernel") {
                    // Q bias
                    let mut q_view = qkv.slice_mut(..q_end);
                    let q_bias = bias.slice(..q_dim);
                    let q_total = (num_tokens * q_dim) as u32;
                    unsafe {
                        self.stream.launch_builder(bk)
                            .arg(&mut q_view).arg(&q_bias).arg(&(num_tokens as i32)).arg(&(q_dim as i32))
                            .launch(LaunchConfig { grid_dim: ((q_total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                            .map_err(|e| LLMError::GpuError(format!("q bias: {e}")))?;
                    }
                    // K bias
                    let mut k_view = qkv.slice_mut(q_end..k_end);
                    let k_bias = bias.slice(q_dim..q_dim + kv_dim);
                    let k_total = (num_tokens * kv_dim) as u32;
                    unsafe {
                        self.stream.launch_builder(bk)
                            .arg(&mut k_view).arg(&k_bias).arg(&(num_tokens as i32)).arg(&(kv_dim as i32))
                            .launch(LaunchConfig { grid_dim: ((k_total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                            .map_err(|e| LLMError::GpuError(format!("k bias: {e}")))?;
                    }
                    // V bias
                    let mut v_view = qkv.slice_mut(k_end..);
                    let v_bias = bias.slice(q_dim + kv_dim..qkv_dim);
                    unsafe {
                        self.stream.launch_builder(bk)
                            .arg(&mut v_view).arg(&v_bias).arg(&(num_tokens as i32)).arg(&(kv_dim as i32))
                            .launch(LaunchConfig { grid_dim: ((k_total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                            .map_err(|e| LLMError::GpuError(format!("v bias: {e}")))?;
                    }
                } else {
                    // Fallback: old path
                    { let mut v = qkv.slice_mut(..q_end); Self::add_bias_f16_view(&self.stream, &self.loader, &mut v, bias, num_tokens, q_dim)?; }
                }
            }

            // 4. RoPE
            {
                let (mut q_part, mut kv_part) = qkv.split_at_mut(q_end);
                let mut k_view = kv_part.slice_mut(..num_tokens * kv_dim);
                Self::apply_rotary_embedding_f16_views(&self.stream, &self.loader,
                    &mut q_part, &mut k_view, &input.positions, input.rope_cos, input.rope_sin,
                    num_tokens, num_heads, num_kv_heads, head_dim)?;
            }

            // 5. KV cache write
            {
                let k_view = qkv.slice(q_end..k_end);
                let v_view = qkv.slice(k_end..);
                Self::cache_write_f16_views(&self.stream, &self.loader, &k_view, &v_view,
                    input.key_cache, input.value_cache, &input.slot_mapping,
                    num_tokens, num_kv_heads, head_dim)?;
            }

            // 6. Attention
            let attn_out = if input.is_prefill {
                Self::prefill_attention_f16io(
                    &self.stream, &self.loader,
                    &qkv.slice(..q_end),
                    input.key_cache, input.value_cache,
                    &input.block_tables, &input.context_lens, &input.seq_start_pos,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?
            } else {
                Self::decode_attention_f16io(
                    &self.stream, &self.loader,
                    &qkv.slice(..q_end),
                    input.key_cache, input.value_cache,
                    &input.block_tables, &input.context_lens,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?
            };

            if dbg { dbg_dump("attn_out", &attn_out, &self.stream); }

            // 7. O projection (T>1 prefill path)
            let attn_proj = Self::hgemm_dispatch(&self.stream, blas, lt, &attn_out, weights.o_proj, num_tokens, hidden, q_dim, &self.loader)?;

            // 8-10. Post-attention MLP (T>1 prefill only, separate kernels)
            let (normed2, residual) = Self::fused_residual_rmsnorm_f16(
                &self.stream, &self.loader, residual_ref, &attn_proj, post_norm_w,
                cfg.rms_norm_eps, num_tokens, hidden)?;
            // Gate+up: fused GEMM + interleaved silu_mul for all N when available
            let fused_act = if let Some(fused_gate_up) = weights.fused_gate_up {
                let gate_up_dim = intermediate * 2;
                let gu_interleaved = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, fused_gate_up, num_tokens, gate_up_dim, hidden, &self.loader)?;
                // Direct silu_mul on interleaved layout: 1 kernel, no transpose/copy
                if let Ok(ref silu_fn) = self.loader.get_func("silu_mul_interleaved", "silu_mul_interleaved_f16_kernel") {
                    let n_elems = num_tokens * intermediate;
                    let mut fused_out = unsafe { self.stream.alloc::<f16>(n_elems) }
                        .map_err(|e| LLMError::GpuError(format!("silu_interleaved alloc: {e}")))?;
                    let total = n_elems as u32;
                    unsafe {
                        self.stream.launch_builder(silu_fn)
                            .arg(&mut fused_out).arg(&gu_interleaved)
                            .arg(&(num_tokens as i32)).arg(&(intermediate as i32))
                            .launch(LaunchConfig { grid_dim: ((total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                            .map_err(|e| LLMError::GpuError(format!("silu_interleaved: {e}")))?;
                    }
                    fused_out
                } else {
                    // Fallback: 2 separate GEMMs
                    let gate = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.gate_proj, num_tokens, intermediate, hidden, &self.loader)?;
                    let up = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.up_proj, num_tokens, intermediate, hidden, &self.loader)?;
                    Self::fused_silu_mul_f16(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?
                }
            } else {
                let gate = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.gate_proj, num_tokens, intermediate, hidden, &self.loader)?;
                let up = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.up_proj, num_tokens, intermediate, hidden, &self.loader)?;
                Self::fused_silu_mul_f16(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?
            };
            let mlp_out = Self::hgemm_dispatch(&self.stream, blas, lt, &fused_act, weights.down_proj, num_tokens, hidden, intermediate, &self.loader)?;
            Ok((residual, mlp_out))
        }

        // ===================================================================
        // f16 dispatch helpers
        // ===================================================================

        /// RMSNorm f16: f16 input, f16 weight, f16 output.
        fn rms_norm_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let n = num_tokens * hidden_size;
            // Safety: kernel writes all num_tokens * hidden_size elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// In-place RMSNorm f16: normalizes `input` directly, no output allocation.
        /// Safe because the kernel uses __syncthreads() between the read pass and
        /// write pass within each block (single-token-per-block launch).
        fn rms_norm_f16_inplace(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &mut CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<()> {
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                let (raw_ptr, _guard) = DevicePtrMut::device_ptr_mut(input, stream);
                stream.launch_builder(&kernel)
                    .arg(&raw_ptr)  // output (same ptr)
                    .arg(&raw_ptr)  // input  (same ptr)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16_inplace launch: {e}")))?;
            }
            Ok(())
        }

        /// hgemm with output allocation: f16 x f16 -> f16.
        fn hgemm_alloc(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: hgemm with beta=0 writes all m*n elements
            let mut output = unsafe { stream.alloc::<f16>(m * n) }
                .map_err(|e| LLMError::GpuError(format!("hgemm_alloc: {e}")))?;
            blas.hgemm(m, n, k, f16::ONE, input, weight, f16::ZERO, &mut output)?;
            Ok(output)
        }

        /// hgemm dispatch: uses cublasLt for M<=32 (split-K), falls back to standard cuBLAS.
        /// GEMM dispatch: custom GEMV for M=1, cublasLt for M<=32, cuBLAS for larger.
        fn hgemm_dispatch(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            lt: Option<&crate::CublasLtRef>,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
            loader: &KernelLoader,
        ) -> Result<CudaSlice<f16>> {
            let mut output = unsafe { stream.alloc::<f16>(m * n) }
                .map_err(|e| LLMError::GpuError(format!("hgemm_dispatch: {e}")))?;

            // M=1: custom GEMV kernel (vectorized half2, warp shuffle reduction)
            // TMA async-prefetch GEMV available but only benefits when weight > L2 cache.
            // For standalone O-proj GEMV (~25MB), L2 handles it. TMA will be wired into
            // the fused kernels where it matters (gateup 135MB weight reads).
            if m == 1 {
                if let Ok(kernel) = loader.get_func("gemv_f16", "gemv_f16_kernel") {
                    let cfg = LaunchConfig {
                        grid_dim: (n as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        stream.launch_builder(&kernel)
                            .arg(&mut output)
                            .arg(weight)
                            .arg(input)
                            .arg(&(n as i32))
                            .arg(&(k as i32))
                            .launch(cfg)
                            .map_err(|e| LLMError::GpuError(format!("gemv_f16 launch: {e}")))?;
                    }
                    return Ok(output);
                }
                // Fallthrough to cuBLAS if kernel not loaded
            }

            #[cfg(feature = "cublaslt")]
            if let Some(lt_ops) = lt {
                if m <= rvllm_gpu::cublaslt_ops::CUBLASLT_M_THRESHOLD {
                    lt_ops.hgemm_a_bt(m, n, k, 1.0, input, weight, 0.0, &mut output)?;
                    return Ok(output);
                }
            }
            blas.hgemm(m, n, k, f16::ONE, input, weight, f16::ZERO, &mut output)?;
            Ok(output)
        }

        /// hgemm dispatch into a pre-allocated view (sub-slice of a larger buffer).
        /// Uses cublasLt for M<=32, falls back to standard cuBLAS.
        fn hgemm_dispatch_into(
            blas: &CublasHandle,
            lt: Option<&crate::CublasLtRef>,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
            out: &mut CudaViewMut<'_, f16>,
        ) -> Result<()> {
            #[cfg(feature = "cublaslt")]
            if let Some(lt_ops) = lt {
                if m <= rvllm_gpu::cublaslt_ops::CUBLASLT_M_THRESHOLD {
                    return lt_ops.hgemm_a_bt_into(m, n, k, 1.0, input, weight, 0.0, out);
                }
            }
            blas.hgemm_into(m, n, k, 1.0, input, weight, 0.0, out)
        }

        /// Add bias f16 in-place on a CudaViewMut<f16>.
        fn add_bias_f16_view(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            tensor: &mut CudaViewMut<'_, f16>,
            bias: &CudaSlice<f16>,
            num_tokens: usize,
            dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("add_bias_f16", "add_bias_f16_kernel")?;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&(dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// RoPE f16 in-place on CudaViewMut<f16> Q/K slices.
        fn apply_rotary_embedding_f16_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &mut CudaViewMut<'_, f16>,
            k: &mut CudaViewMut<'_, f16>,
            positions: &CudaView<'_, i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            if num_tokens == 0 { return Ok(()); }
            let kernel = loader.get_func("rotary_embedding_f16", "rotary_embedding_f16_kernel")?;
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(q).arg(k)
                    .arg(rope_cos).arg(rope_sin).arg(positions)
                    .arg(&(num_tokens as i32)).arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32)).arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// Cache write f16: f16 K/V input -> f16 paged cache (pure copy).
        fn cache_write_f16_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            k: &CudaView<'_, f16>,
            v: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            slot_mapping: &CudaView<'_, i32>,
            num_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kv_dim = num_kv_heads * head_dim;
            let kernel = loader.get_func("reshape_and_cache_f16", "reshape_and_cache_f16io_kernel")?;
            let threads = kv_dim.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(key_cache).arg(value_cache)
                    .arg(k).arg(v)
                    .arg(slot_mapping)
                    .arg(&(num_tokens as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cache_write_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// FA3 prefill attention with f16 I/O: f16 Q, f16 KV cache, f16 output.
        /// Eliminates the f32 cast round-trip from the prefill path.
        #[allow(clippy::too_many_arguments)]
        fn prefill_attention_f16io(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            seq_start_pos: &CudaView<'_, i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let kernel = loader.get_func("flash_attention_3_prefill", "flash_attention_3_prefill_f16io_kernel")
                .map_err(|e| LLMError::GpuError(format!("load prefill f16io kernel: {e}")))?;

            let out_len = num_tokens * num_heads * head_dim;
            let mut output = unsafe { stream.alloc::<f16>(out_len) }
                .map_err(|e| LLMError::GpuError(format!("prefill_attn_f16io alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA3_BC: usize = 64;
            const FA3_THREADS: u32 = 256;
            let smem = (FA3_BC * head_dim + FA3_BC + 8) * std::mem::size_of::<f32>();
            let shared_mem_bytes = smem as u32;

            if num_seqs == 0 {
                return Err(LLMError::GpuError("prefill_attention_f16io: num_seqs == 0".into()));
            }

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA3_THREADS, 1, 1),
                shared_mem_bytes,
            };

            let max_blocks_per_seq = (DeviceSlice::len(block_tables) / num_seqs) as i32;

            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("prefill FA3 f16io set max shared mem: {e}")))?;
            }

            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(q)
                    .arg(key_cache).arg(value_cache)
                    .arg(block_tables).arg(context_lens)
                    .arg(seq_start_pos)
                    .arg(&scale)
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .arg(&(block_size as i32))
                    .arg(&(max_context_len as i32))
                    .arg(&max_blocks_per_seq)
                    .arg(&(num_tokens as i32))
                    .arg(&1i32) // causal
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("prefill FA3 f16io launch: {e}")))?;
            }

            Ok(output)
        }

        /// FA2 decode attention with f16 I/O: f16 Q, f16 KV cache, f16 output.
        #[allow(clippy::too_many_arguments)]
        fn decode_attention_f16io(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let out_len = num_tokens * num_heads * head_dim;
            let mut output = unsafe { stream.alloc::<f16>(out_len) }
                .map_err(|e| LLMError::GpuError(format!("decode_attn_f16io alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let p_num_heads = num_heads as i32;
            let p_num_kv_heads = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_blocks = (block_tables.len() / num_seqs.max(1)) as i32;

            // GQA-optimized FA3 kernel: one block per KV head, processes all query heads
            // in the group. Reduces KV cache loads by heads_per_group (e.g. 6x for Qwen2.5-1.5B).
            // Only used when GQA is active (num_heads != num_kv_heads) and ratio <= 8.
            let heads_per_group = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };
            if num_heads != num_kv_heads && heads_per_group <= 8 {
                if let Ok(gqa_kernel) = loader.get_func("flash_attention_3", "flash_attention_3_decode_gqa_f16io_kernel") {
                    const GQA_BC: usize = 64;
                    const GQA_THREADS: u32 = 256;
                    const GQA_MAX_HPG: usize = 8;
                    // s_kv[BC*head_dim] + s_scores[MAX_HPG*BC] + s_warp[8]
                    let smem = (GQA_BC * head_dim + GQA_MAX_HPG * GQA_BC + 8) * std::mem::size_of::<f32>();
                    let shared_mem_bytes = smem as u32;

                    let p_max_context = max_context_len as i32;

                    let cfg = LaunchConfig {
                        grid_dim: (num_seqs as u32, num_kv_heads as u32, 1),
                        block_dim: (GQA_THREADS, 1, 1),
                        shared_mem_bytes,
                    };

                    if shared_mem_bytes > 49152 {
                        gqa_kernel.set_attribute(
                            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            shared_mem_bytes as i32,
                        ).map_err(|e| LLMError::GpuError(format!("FA3 GQA set max shared mem: {e}")))?;
                    }

                    unsafe {
                        stream.launch_builder(&gqa_kernel)
                            .arg(&mut output)
                            .arg(q)
                            .arg(key_cache).arg(value_cache)
                            .arg(block_tables).arg(context_lens)
                            .arg(&scale)
                            .arg(&p_num_heads).arg(&p_num_kv_heads)
                            .arg(&p_head_dim).arg(&p_block_size)
                            .arg(&p_max_context)
                            .arg(&p_max_blocks)
                            .launch(cfg)
                            .map_err(|e| LLMError::GpuError(format!("FA3 GQA decode launch: {e}")))?;
                    }
                    return Ok(output);
                }
            }

            // FA3 kernel: 256 threads, vectorized half2 loads, warp-parallel reductions.
            // Shared memory: 33KB (fits in default 48KB, no set_attribute needed).
            if let Ok(fa3_kernel) = loader.get_func("flash_attention_3", "flash_attention_3_decode_f16io_kernel") {
                const FA3_BC: usize = 64;
                const FA3_THREADS: u32 = 256;
                // Single KV buffer (reused for K then V) + scores + warp_reduce
                let smem = (FA3_BC * head_dim + FA3_BC + 8) * std::mem::size_of::<f32>();
                let shared_mem_bytes = smem as u32;

                let cfg = LaunchConfig {
                    grid_dim: (num_seqs as u32, num_heads as u32, 1),
                    block_dim: (FA3_THREADS, 1, 1),
                    shared_mem_bytes,
                };

                if shared_mem_bytes > 49152 {
                    fa3_kernel.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared_mem_bytes as i32,
                    ).map_err(|e| LLMError::GpuError(format!("FA3 set max shared mem: {e}")))?;
                }

                unsafe {
                    stream.launch_builder(&fa3_kernel)
                        .arg(&mut output)
                        .arg(q)
                        .arg(key_cache).arg(value_cache)
                        .arg(block_tables).arg(context_lens)
                        .arg(&scale)
                        .arg(&p_num_heads).arg(&p_num_kv_heads)
                        .arg(&p_head_dim).arg(&p_block_size)
                        .arg(&p_max_blocks)
                        .launch(cfg)
                        .map_err(|e| LLMError::GpuError(format!("FA3 decode launch: {e}")))?;
                }
                return Ok(output);
            }
            // Fallback: FA2 kernel
            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let kernel = loader.get_func("flash_attention", "flash_attention_2_decode_f16io_kernel")?;

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("FA2 set max shared mem: {e}")))?;
            }

            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(q)
                    .arg(key_cache).arg(value_cache)
                    .arg(block_tables).arg(context_lens)
                    .arg(&scale)
                    .arg(&p_num_heads).arg(&p_num_kv_heads)
                    .arg(&p_head_dim).arg(&p_block_size)
                    .arg(&p_max_blocks)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("FA2 decode f16io launch: {e}")))?;
            }
            Ok(output)
        }

        /// Fused residual add + RMSNorm, all f16.
        /// Returns (normed_output, residual) both f16.
        pub fn fused_residual_rmsnorm_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            add: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
            let n = num_tokens * hidden_size;
            // Safety: kernel writes all n elements to both output and residual
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 output alloc: {e}")))?;
            let mut residual = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 residual alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("fused_residual_rmsnorm_f16", "fused_residual_rmsnorm_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&mut residual)
                    .arg(input)
                    .arg(add)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 launch: {e}")))?;
            }
            Ok((output, residual))
        }

        /// Fused SiLU * mul, f16 variant.
        fn fused_silu_mul_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate: &CudaSlice<f16>,
            up: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("activation_f16", "fused_silu_mul_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(gate).arg(up)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Fused SiLU*mul on a contiguous [gate || up] buffer, f16 variant.
        fn fused_silu_mul_f16_split(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate_up: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            let gate_view = gate_up.slice(..n);
            let up_view = gate_up.slice(n..n * 2);
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16_split alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("activation_f16", "fused_silu_mul_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&gate_view).arg(&up_view)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16_split launch: {e}")))?;
            }
            Ok(output)
        }

        /// Element-wise tensor addition f16: out = a + b.
        fn add_tensors_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            a: &CudaSlice<f16>,
            b: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("add_tensors_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("add_bias_f16", "add_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(a).arg(b)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_f16 launch: {e}")))?;
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
