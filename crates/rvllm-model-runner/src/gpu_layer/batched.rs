//! T>=1 batched decode and prefill forward path.
//!
//! Always uses pre-allocated scratch buffers. Results written to scratch.
//! Per-op CUTLASS vs cuBLAS routing determined by GemmStrategy (set at init time).

use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, DeviceSlice,
    LaunchConfig, PushKernelArg,
};
use half::f16;
use tracing::info;

use rvllm_core::error::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;

use super::{BatchedLayerPhaseTimings, GemmStrategy, GpuLayerInput, GpuLayerWeights, GpuTransformerLayer, LayerScratchRef};

#[derive(Clone, Copy)]
struct BatchedPipelinePolicy {
    use_cutlass_qkv: bool,
    use_cutlass_oproj: bool,
    use_cutlass_gateup: bool,
    use_cutlass_gate_aux: bool,
}

impl GpuTransformerLayer {
    #[inline]
    fn use_cutlass_qkv(strategy: GemmStrategy) -> bool {
        matches!(strategy, GemmStrategy::Cutlass)
    }

    #[inline]
    fn use_cutlass_oproj(strategy: GemmStrategy) -> bool {
        matches!(strategy, GemmStrategy::Cutlass)
    }

    #[inline]
    fn use_cutlass_gateup(strategy: GemmStrategy) -> bool {
        matches!(strategy, GemmStrategy::Hybrid | GemmStrategy::Cutlass)
    }

    /// Batched forward pass into pre-allocated scratch buffers.
    ///
    /// Results are written to `scratch.residual` and `scratch.down`.
    /// Caller reads from scratch after this returns.
    pub(crate) fn forward_batched(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        prev_mlp_out: Option<&CudaSlice<f16>>,
        scratch: &mut LayerScratchRef<'_>,
        gemm_strategy: GemmStrategy,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
    ) -> Result<()> {
        self.forward_batched_profiled(
            input,
            weights,
            blas,
            lt,
            prev_mlp_out,
            scratch,
            None,
            gemm_strategy,
            cutlass,
        )
    }

    pub(crate) fn forward_batched_v2(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        prev_mlp_out: Option<&CudaSlice<f16>>,
        scratch: &mut LayerScratchRef<'_>,
        gemm_strategy: GemmStrategy,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
    ) -> Result<()> {
        self.forward_batched_v2_profiled(
            input,
            weights,
            blas,
            lt,
            prev_mlp_out,
            scratch,
            None,
            gemm_strategy,
            cutlass,
        )
    }

    pub(crate) fn forward_batched_profiled(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        prev_mlp_out: Option<&CudaSlice<f16>>,
        scratch: &mut LayerScratchRef<'_>,
        mut phase_timings: Option<&mut BatchedLayerPhaseTimings>,
        gemm_strategy: GemmStrategy,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
    ) -> Result<()> {
        let policy = Self::legacy_batched_policy(gemm_strategy, input, weights, cutlass);
        self.forward_batched_with_policy(
            input,
            weights,
            blas,
            lt,
            prev_mlp_out,
            scratch,
            phase_timings,
            gemm_strategy,
            cutlass,
            policy,
        )
    }

    pub(crate) fn forward_batched_v2_profiled(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        prev_mlp_out: Option<&CudaSlice<f16>>,
        scratch: &mut LayerScratchRef<'_>,
        phase_timings: Option<&mut BatchedLayerPhaseTimings>,
        gemm_strategy: GemmStrategy,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
    ) -> Result<()> {
        let policy = self.batched_v2_policy(input, weights, cutlass);
        self.forward_batched_with_policy(
            input,
            weights,
            blas,
            lt,
            prev_mlp_out,
            scratch,
            phase_timings,
            gemm_strategy,
            cutlass,
            policy,
        )
    }

    fn legacy_batched_policy(
        gemm_strategy: GemmStrategy,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
    ) -> BatchedPipelinePolicy {
        BatchedPipelinePolicy {
            use_cutlass_qkv: Self::use_cutlass_qkv(gemm_strategy) && cutlass.is_some(),
            use_cutlass_oproj: Self::use_cutlass_oproj(gemm_strategy) && cutlass.is_some(),
            use_cutlass_gateup: Self::use_cutlass_gateup(gemm_strategy) && cutlass.is_some(),
            use_cutlass_gate_aux: Self::use_cutlass_gateup(gemm_strategy)
                && cutlass.is_some()
                && !input.is_prefill
                && weights.fused_gate_up_fp8.is_none()
                && weights.down_proj_fp8.is_none()
                && std::env::var("RVLLM_CUTLASS_GATE_AUX").map_or(true, |v| v != "0"),
        }
    }

    fn batched_v2_policy(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
    ) -> BatchedPipelinePolicy {
        let cutlass_loaded = cutlass.is_some();
        let gateup = cutlass_loaded && self.batched_v2_policy.use_cutlass_gateup;
        let gate_aux = gateup
            && !input.is_prefill
            && weights.fused_gate_up_fp8.is_none()
            && weights.down_proj_fp8.is_none()
            && self.batched_v2_policy.use_cutlass_gate_aux;
        BatchedPipelinePolicy {
            use_cutlass_qkv: cutlass_loaded && self.batched_v2_policy.use_cutlass_qkv,
            use_cutlass_oproj: cutlass_loaded && self.batched_v2_policy.use_cutlass_oproj,
            use_cutlass_gateup: gateup,
            use_cutlass_gate_aux: gate_aux,
        }
    }

    fn forward_batched_with_policy(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        prev_mlp_out: Option<&CudaSlice<f16>>,
        scratch: &mut LayerScratchRef<'_>,
        mut phase_timings: Option<&mut BatchedLayerPhaseTimings>,
        gemm_strategy: GemmStrategy,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
        policy: BatchedPipelinePolicy,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_tokens = input.num_tokens;
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + kv_dim + kv_dim;
        let q_end = num_tokens * q_dim;
        let k_end = q_end + num_tokens * kv_dim;

        let dbg = cfg.layer_idx < 1 && std::env::var("RVLLM_DEBUG").is_ok();
        let dbg_dump = |label: &str, buf: &CudaSlice<f16>, stream: &Arc<CudaStream>| {
            if let Ok(vals) = stream.clone_dtoh(buf) {
                let first5: Vec<f32> = vals.iter().take(5).map(|v| v.to_f32()).collect();
                let last5: Vec<f32> = vals
                    .iter()
                    .rev()
                    .take(5)
                    .rev()
                    .map(|v| v.to_f32())
                    .collect();
                let nan = vals.iter().any(|v| v.to_f32().is_nan());
                let max = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                let mean = vals.iter().map(|v| v.to_f32()).sum::<f32>() / vals.len() as f32;
                info!("DEBUG L0 {label}: first5={first5:?} last5={last5:?} max={max:.4} mean={mean:.6} nan={nan} len={}", vals.len());
            }
        };
        let profile_enabled = phase_timings.is_some();
        let mut phase_start = std::time::Instant::now();
        let mut mark_phase =
            |slot: fn(&mut BatchedLayerPhaseTimings) -> &mut std::time::Duration| -> Result<()> {
                if let Some(t) = phase_timings.as_deref_mut() {
                    self.stream
                        .synchronize()
                        .map_err(|e| LLMError::GpuError(format!("phase profile sync: {e}")))?;
                    *slot(t) = phase_start.elapsed();
                    phase_start = std::time::Instant::now();
                }
                Ok(())
            };

        // 1. Pre-attention RMSNorm
        let residual_from_fused: bool;
        if let Some(prev_mlp) = prev_mlp_out {
            Self::fused_residual_rmsnorm_f16_into(
                &self.stream,
                &self.loader,
                input.hidden_states,
                prev_mlp,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                scratch.normed,
                scratch.down,
            )?;
            residual_from_fused = true;
        } else {
            Self::rms_norm_f16_into(
                &self.stream,
                &self.loader,
                input.hidden_states,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                scratch.normed,
            )?;
            residual_from_fused = false;
        };
        if profile_enabled {
            mark_phase(|t| &mut t.pre_attn_norm)?;
        }

        // 2-3. QKV GEMM + bias (before taking residual_ref to avoid borrow conflict)
        let used_fused_cutlass = if policy.use_cutlass_qkv {
            self.batched_qkv_cutlass(
                weights,
                scratch,
                cutlass,
                gemm_strategy,
                num_tokens,
                qkv_dim,
                hidden,
                q_dim,
                kv_dim,
            )?
        } else {
            false
        };

        if !used_fused_cutlass {
            self.batched_qkv_gemm(
                weights, blas, lt, scratch, num_tokens, qkv_dim, hidden, q_dim, kv_dim,
            )?;
        }
        if profile_enabled {
            mark_phase(|t| &mut t.qkv)?;
        }

        let residual_ref: &CudaSlice<f16> = if residual_from_fused {
            &*scratch.down
        } else {
            input.hidden_states
        };

        // 4-5. RoPE + KV cache write
        if num_tokens == 1 {
            // Fused single kernel: RoPE + cache write in one launch
            Self::fused_rope_cache_write(
                &self.stream,
                &self.loader,
                &mut *scratch.qkv,
                input,
                q_dim,
                kv_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                num_tokens,
            )?;
        } else {
            // Separate kernels for T>1
            {
                let (mut q_part, mut kv_part) = scratch.qkv.split_at_mut(q_end);
                let mut k_view = kv_part.slice_mut(..num_tokens * kv_dim);
                Self::apply_rotary_embedding_f16_views(
                    &self.stream,
                    &self.loader,
                    &mut q_part,
                    &mut k_view,
                    &input.positions,
                    input.rope_cos,
                    input.rope_sin,
                    num_tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )?;
            }
            {
                let k_view = scratch.qkv.slice(q_end..k_end);
                let v_view = scratch.qkv.slice(k_end..k_end + num_tokens * kv_dim);
                Self::cache_write_f16_views(
                    &self.stream,
                    &self.loader,
                    &k_view,
                    &v_view,
                    input.key_cache,
                    input.value_cache,
                    &input.slot_mapping,
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                )?;
            }
        }
        if profile_enabled {
            mark_phase(|t| &mut t.rope_cache)?;
        }

        // 6. Attention
        let attn_out = if input.is_prefill {
            Self::prefill_attention_f16io(
                &self.stream,
                &self.loader,
                &scratch.qkv.slice(..q_end),
                input.key_cache,
                input.value_cache,
                &input.block_tables,
                &input.context_lens,
                &input.seq_start_pos,
                num_tokens,
                input.num_seqs,
                num_heads,
                num_kv_heads,
                head_dim,
                input.max_context_len,
                input.block_size,
            )?
        } else {
            Self::decode_attention_f16io(
                &self.stream,
                &self.loader,
                &scratch.qkv.slice(..q_end),
                input.key_cache,
                input.value_cache,
                &input.block_tables,
                &input.context_lens,
                num_tokens,
                input.num_seqs,
                num_heads,
                num_kv_heads,
                head_dim,
                input.max_context_len,
                input.block_size,
            )?
        };

        if dbg {
            dbg_dump("attn_out", &attn_out, &self.stream);
        }
        if profile_enabled {
            mark_phase(|t| &mut t.attn)?;
        }

        // 7-8. O projection + residual + post-norm
        if policy.use_cutlass_oproj {
            let ck = cutlass.expect("GemmStrategy::Cutlass requires loaded CUTLASS kernels");
            let m = num_tokens as i32;
            let n = hidden as i32;
            let k = q_dim as i32;
            let ws_bytes = ck.oproj_residual_workspace_size(m, n, k);
            let mut ws = unsafe { self.stream.alloc::<u8>(ws_bytes.max(1)) }
                .map_err(|e| LLMError::GpuError(format!("cutlass oproj ws alloc: {e}")))?;
            let out_ptr = {
                let (p, _g) = DevicePtrMut::device_ptr_mut(&mut *scratch.o_proj, &self.stream);
                p
            };
            let in_ptr = {
                let (p, _g) = DevicePtr::device_ptr(&attn_out, &self.stream);
                p
            };
            let (w_ptr, _g2) = DevicePtr::device_ptr(weights.o_proj, &self.stream);
            let (r_ptr, _g3) = DevicePtr::device_ptr(residual_ref, &self.stream);
            let ws_ptr = {
                let (p, _g) = DevicePtrMut::device_ptr_mut(&mut ws, &self.stream);
                p
            };
            let stream_ptr = self.stream.cu_stream() as u64;
            ck.oproj_residual_gemm(
                out_ptr, in_ptr, w_ptr, r_ptr, m, n, k, ws_ptr, ws_bytes, stream_ptr,
            )
            .map_err(|e| LLMError::GpuError(e))?;
            self.stream
                .memcpy_dtod(
                    &scratch.o_proj.slice(..num_tokens * hidden),
                    &mut scratch.residual.slice_mut(..num_tokens * hidden),
                )
                .map_err(|e| LLMError::GpuError(format!("oproj residual copy: {e}")))?;
            Self::rms_norm_f16_into(
                &self.stream,
                &self.loader,
                &*scratch.residual,
                weights.post_attention_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                scratch.normed,
            )?;
        } else {
            Self::hgemm_dispatch_fp8_into(
                &self.stream,
                blas,
                lt,
                &attn_out,
                weights.o_proj,
                num_tokens,
                hidden,
                q_dim,
                &self.loader,
                weights.o_proj_fp8,
                scratch.o_proj,
            )?;
            Self::fused_residual_rmsnorm_f16_into(
                &self.stream,
                &self.loader,
                residual_ref,
                &*scratch.o_proj,
                weights.post_attention_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                scratch.normed,
                scratch.residual,
            )?;
        }
        if profile_enabled {
            mark_phase(|t| &mut t.oproj_norm)?;
        }

        // 9. Gate+up GEMM + SiLU
        let fused_gate_up = weights.fused_gate_up.ok_or_else(|| {
            LLMError::GpuError("Batched path requires fused_gate_up weight".into())
        })?;
        let gate_up_dim = intermediate * 2;
        let use_cutlass_gate_aux = policy.use_cutlass_gate_aux;

        if use_cutlass_gate_aux {
            let ck = cutlass.expect("CUTLASS gate-aux path requires loaded CUTLASS kernels");
            Self::hgemm_dispatch_fp8_into(
                &self.stream,
                blas,
                lt,
                &*scratch.normed,
                weights.up_proj,
                num_tokens,
                intermediate,
                hidden,
                &self.loader,
                None,
                scratch.gate_up,
            )?;

            let m = num_tokens as i32;
            let n = intermediate as i32;
            let k = hidden as i32;
            let ws_bytes = ck.gate_silu_mul_workspace_size(m, n, k);
            if scratch.gateup_ws.len() < ws_bytes.max(1) {
                return Err(LLMError::GpuError(format!(
                    "cutlass gate_silu_mul workspace too small: need {} bytes, have {}",
                    ws_bytes.max(1),
                    scratch.gateup_ws.len()
                )));
            }
            let out_ptr = {
                let (p, _g) =
                    DevicePtrMut::device_ptr_mut(&mut *scratch.silu_out, &self.stream);
                p
            };
            let in_ptr = {
                let (p, _g) = DevicePtr::device_ptr(&*scratch.normed, &self.stream);
                p
            };
            let (gate_ptr, _g2) = DevicePtr::device_ptr(weights.gate_proj, &self.stream);
            let (aux_ptr, _g3) = DevicePtr::device_ptr(&*scratch.gate_up, &self.stream);
            let ws_ptr = {
                let mut ws = scratch.gateup_ws.slice_mut(..ws_bytes.max(1));
                let (p, _g) = DevicePtrMut::device_ptr_mut(&mut ws, &self.stream);
                p
            };
            let stream_ptr = self.stream.cu_stream() as u64;
            ck.gate_silu_mul(
                out_ptr, in_ptr, gate_ptr, aux_ptr, m, n, k, ws_ptr, ws_bytes, stream_ptr,
            )
            .map_err(LLMError::GpuError)?;
        } else {
            if policy.use_cutlass_gateup {
                let ck = cutlass.expect("GemmStrategy::Cutlass requires loaded CUTLASS kernels");
                let m = num_tokens as i32;
                let n = gate_up_dim as i32;
                let k = hidden as i32;
                let ws_bytes = ck.gateup_silu_workspace_size(m, n, k);
                if scratch.gateup_ws.len() < ws_bytes.max(1) {
                    return Err(LLMError::GpuError(format!(
                        "cutlass gateup workspace too small: need {} bytes, have {}",
                        ws_bytes.max(1),
                        scratch.gateup_ws.len()
                    )));
                }
                let out_ptr = {
                    let (p, _g) =
                        DevicePtrMut::device_ptr_mut(&mut *scratch.silu_out, &self.stream);
                    p
                };
                let in_ptr = {
                    let (p, _g) = DevicePtr::device_ptr(&*scratch.normed, &self.stream);
                    p
                };
                let (w_ptr, _g2) = DevicePtr::device_ptr(fused_gate_up, &self.stream);
                let ws_ptr = {
                    let mut ws = scratch.gateup_ws.slice_mut(..ws_bytes.max(1));
                    let (p, _g) = DevicePtrMut::device_ptr_mut(&mut ws, &self.stream);
                    p
                };
                let stream_ptr = self.stream.cu_stream() as u64;
                ck.gateup_silu(
                    out_ptr, in_ptr, w_ptr, m, n, k, ws_ptr, ws_bytes, stream_ptr,
                )
                .map_err(|e| LLMError::GpuError(e))?;
            } else {
                Self::hgemm_dispatch_fp8_into(
                    &self.stream,
                    blas,
                    lt,
                    &*scratch.normed,
                    fused_gate_up,
                    num_tokens,
                    gate_up_dim,
                    hidden,
                    &self.loader,
                    weights.fused_gate_up_fp8,
                    scratch.gate_up,
                )?;
                let silu_fn = self
                    .loader
                    .get_func("silu_mul_interleaved", "silu_mul_interleaved_f16_kernel")
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "Required silu_mul_interleaved kernel missing: {e}"
                        ))
                    })?;
                let n_elems = num_tokens * intermediate;
                let total = n_elems as u32;
                unsafe {
                    self.stream
                        .launch_builder(&silu_fn)
                        .arg(&mut *scratch.silu_out)
                        .arg(&*scratch.gate_up)
                        .arg(&(num_tokens as i32))
                        .arg(&(intermediate as i32))
                        .launch(LaunchConfig {
                            grid_dim: ((total + 255) / 256, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|e| LLMError::GpuError(format!("silu_interleaved: {e}")))?;
                }
            }
            if profile_enabled {
                mark_phase(|t| &mut t.gateup_silu)?;
            }

            // 10. Down projection into scratch.down
            Self::hgemm_dispatch_fp8_into(
                &self.stream,
                blas,
                lt,
                &*scratch.silu_out,
                weights.down_proj,
                num_tokens,
                hidden,
                intermediate,
                &self.loader,
                weights.down_proj_fp8,
                scratch.down,
            )?;
            if profile_enabled {
                mark_phase(|t| &mut t.down)?;
            }
            return Ok(());
        }

        if profile_enabled {
            mark_phase(|t| &mut t.gateup_silu)?;
        }
        Self::hgemm_dispatch_fp8_into(
            &self.stream,
            blas,
            lt,
            &*scratch.silu_out,
            weights.down_proj,
            num_tokens,
            hidden,
            intermediate,
            &self.loader,
            weights.down_proj_fp8,
            scratch.down,
        )?;
        if profile_enabled {
            mark_phase(|t| &mut t.down)?;
        }

        Ok(())
    }

    // ----------------------------------------------------------------
    // Batched-path QKV helpers
    // ----------------------------------------------------------------

    /// Try CUTLASS fused QKV+bias GEMM (requires CUTLASS + fused QKV + bias).
    /// Returns true if CUTLASS was used and QKV is written to scratch.qkv.
    fn batched_qkv_cutlass(
        &self,
        weights: &GpuLayerWeights<'_>,
        scratch: &mut LayerScratchRef<'_>,
        cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
        gemm_strategy: GemmStrategy,
        num_tokens: usize,
        qkv_dim: usize,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<bool> {
        if !Self::use_cutlass_qkv(gemm_strategy) {
            return Ok(false);
        }
        let (fused_qkv, bias, ck) = match (weights.fused_qkv, weights.qkv_bias, cutlass) {
            (Some(q), Some(b), Some(c)) => (q, b, c),
            _ => return Ok(false),
        };
        let m = num_tokens as i32;
        let n = qkv_dim as i32;
        let k = hidden as i32;
        let ws_bytes = ck.qkv_bias_workspace_size(m, n, k);
        let mut ws = unsafe { self.stream.alloc::<u8>(ws_bytes.max(1)) }
            .map_err(|e| LLMError::GpuError(format!("cutlass qkv ws alloc: {e}")))?;
        let mut qkv_view = scratch.qkv.slice_mut(..num_tokens * qkv_dim);
        let out_ptr = {
            let (p, _g) = DevicePtrMut::device_ptr_mut(&mut qkv_view, &self.stream);
            p
        };
        let in_ptr = {
            let (p, _g) = DevicePtr::device_ptr(&*scratch.normed, &self.stream);
            p
        };
        let (w_ptr, _g2) = DevicePtr::device_ptr(fused_qkv, &self.stream);
        let (b_ptr, _g3) = DevicePtr::device_ptr(bias, &self.stream);
        let ws_ptr = {
            let (p, _g) = DevicePtrMut::device_ptr_mut(&mut ws, &self.stream);
            p
        };
        let stream_ptr = self.stream.cu_stream() as u64;
        ck.qkv_bias_gemm(
            out_ptr, in_ptr, w_ptr, b_ptr, m, n, k, ws_ptr, ws_bytes, stream_ptr,
        )
        .map_err(|e| LLMError::GpuError(e))?;
        Ok(true)
    }

    /// GEMM + bias via cublasLt BIAS epilogue. Caller should skip separate bias add.
    #[cfg(feature = "cublaslt")]
    fn hgemm_dispatch_fp8_bias_into(
        stream: &Arc<CudaStream>,
        lt: &crate::CublasLtRef,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        c: &mut CudaSlice<f16>,
    ) -> Result<()> {
        let (bias_ptr, _bg) = DevicePtr::device_ptr(bias, stream);
        lt.hgemm_a_bt_bias_into(m, n, k, 1.0, a, b, bias_ptr, c)
    }

    /// cuBLAS/cublasLt QKV GEMM + bias into scratch.qkv.
    fn batched_qkv_gemm(
        &self,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        scratch: &mut LayerScratchRef<'_>,
        num_tokens: usize,
        qkv_dim: usize,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<()> {
        #[allow(unused_mut, unused_assignments)]
        let mut bias_fused = false;

        if let Some(fused_qkv) = weights.fused_qkv {
            // Try cublasLt bias epilogue (eliminates separate add_bias kernel launches)
            #[cfg(feature = "cublaslt")]
            if let (Some(lt_ops), Some(bias)) = (lt, weights.qkv_bias) {
                if num_tokens == 1 {
                    // N=1: write directly to qkv with bias
                    Self::hgemm_dispatch_fp8_bias_into(
                        &self.stream,
                        lt_ops,
                        &*scratch.normed,
                        fused_qkv,
                        bias,
                        num_tokens,
                        qkv_dim,
                        hidden,
                        scratch.qkv,
                    )?;
                    bias_fused = true;
                } else {
                    // N>1: write to gate_up (temp) with bias, then deinterleave
                    Self::hgemm_dispatch_fp8_bias_into(
                        &self.stream,
                        lt_ops,
                        &*scratch.normed,
                        fused_qkv,
                        bias,
                        num_tokens,
                        qkv_dim,
                        hidden,
                        scratch.gate_up,
                    )?;
                    bias_fused = true;
                    let interleaved_len = num_tokens * qkv_dim;
                    let dk = self
                        .loader
                        .get_func("deinterleave_qkv", "deinterleave_qkv_f16_kernel")
                        .map_err(|e| {
                            LLMError::GpuError(format!(
                                "Required deinterleave_qkv kernel missing: {e}"
                            ))
                        })?;
                    let total = interleaved_len as u32;
                    let tmp_view = scratch.gate_up.slice(..interleaved_len);
                    let mut qkv_view = scratch.qkv.slice_mut(..interleaved_len);
                    unsafe {
                        self.stream
                            .launch_builder(&dk)
                            .arg(&mut qkv_view)
                            .arg(&tmp_view)
                            .arg(&(num_tokens as i32))
                            .arg(&(q_dim as i32))
                            .arg(&(kv_dim as i32))
                            .launch(LaunchConfig {
                                grid_dim: ((total + 255) / 256, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| LLMError::GpuError(format!("deinterleave qkv: {e}")))?;
                    }
                }
            }
            // Fallback: GEMM without bias epilogue
            if !bias_fused {
                if num_tokens == 1 {
                    Self::hgemm_dispatch_fp8_into(
                        &self.stream,
                        blas,
                        lt,
                        &*scratch.normed,
                        fused_qkv,
                        num_tokens,
                        qkv_dim,
                        hidden,
                        &self.loader,
                        weights.fused_qkv_fp8,
                        scratch.qkv,
                    )?;
                } else {
                    Self::hgemm_dispatch_fp8_into(
                        &self.stream,
                        blas,
                        lt,
                        &*scratch.normed,
                        fused_qkv,
                        num_tokens,
                        qkv_dim,
                        hidden,
                        &self.loader,
                        weights.fused_qkv_fp8,
                        scratch.gate_up,
                    )?;
                    let interleaved_len = num_tokens * qkv_dim;
                    let dk = self
                        .loader
                        .get_func("deinterleave_qkv", "deinterleave_qkv_f16_kernel")
                        .map_err(|e| {
                            LLMError::GpuError(format!(
                                "Required deinterleave_qkv kernel missing: {e}"
                            ))
                        })?;
                    let total = interleaved_len as u32;
                    let tmp_view = scratch.gate_up.slice(..interleaved_len);
                    let mut qkv_view = scratch.qkv.slice_mut(..interleaved_len);
                    unsafe {
                        self.stream
                            .launch_builder(&dk)
                            .arg(&mut qkv_view)
                            .arg(&tmp_view)
                            .arg(&(num_tokens as i32))
                            .arg(&(q_dim as i32))
                            .arg(&(kv_dim as i32))
                            .launch(LaunchConfig {
                                grid_dim: ((total + 255) / 256, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| LLMError::GpuError(format!("deinterleave qkv: {e}")))?;
                    }
                }
            }
        } else {
            let q_end_t = num_tokens * q_dim;
            let k_end_t = q_end_t + num_tokens * kv_dim;
            {
                let mut d = scratch.qkv.slice_mut(..q_end_t);
                Self::hgemm_dispatch_into(
                    blas,
                    lt,
                    &*scratch.normed,
                    weights.q_proj,
                    num_tokens,
                    q_dim,
                    hidden,
                    &mut d,
                )?;
            }
            {
                let mut d = scratch.qkv.slice_mut(q_end_t..k_end_t);
                Self::hgemm_dispatch_into(
                    blas,
                    lt,
                    &*scratch.normed,
                    weights.k_proj,
                    num_tokens,
                    kv_dim,
                    hidden,
                    &mut d,
                )?;
            }
            {
                let mut d = scratch
                    .qkv
                    .slice_mut(k_end_t..k_end_t + num_tokens * kv_dim);
                Self::hgemm_dispatch_into(
                    blas,
                    lt,
                    &*scratch.normed,
                    weights.v_proj,
                    num_tokens,
                    kv_dim,
                    hidden,
                    &mut d,
                )?;
            }
        }

        // Bias add (skip if already fused into GEMM)
        if !bias_fused {
            if let Some(bias) = weights.qkv_bias {
                if num_tokens == 1 {
                    let bk = self
                        .loader
                        .get_func("add_bias_broadcast", "add_bias_broadcast_f16_kernel")
                        .map_err(|e| {
                            LLMError::GpuError(format!(
                                "Required add_bias_broadcast kernel missing: {e}"
                            ))
                        })?;
                    let total = (num_tokens * qkv_dim) as u32;
                    let mut qkv_view = scratch.qkv.slice_mut(..num_tokens * qkv_dim);
                    unsafe {
                        self.stream
                            .launch_builder(&bk)
                            .arg(&mut qkv_view)
                            .arg(bias)
                            .arg(&(num_tokens as i32))
                            .arg(&(qkv_dim as i32))
                            .launch(LaunchConfig {
                                grid_dim: ((total + 255) / 256, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| LLMError::GpuError(format!("qkv bias: {e}")))?;
                    }
                } else {
                    let q_end_t = num_tokens * q_dim;
                    let k_end_t = q_end_t + num_tokens * kv_dim;
                    let q_bias = bias.slice(0..q_dim);
                    let k_bias = bias.slice(q_dim..q_dim + kv_dim);
                    let v_bias = bias.slice(q_dim + kv_dim..qkv_dim);
                    {
                        let mut d = scratch.qkv.slice_mut(..q_end_t);
                        Self::add_bias_f16_view_from_slice(
                            &self.stream,
                            &self.loader,
                            &mut d,
                            &q_bias,
                            num_tokens,
                            q_dim,
                        )?;
                    }
                    {
                        let mut d = scratch.qkv.slice_mut(q_end_t..k_end_t);
                        Self::add_bias_f16_view_from_slice(
                            &self.stream,
                            &self.loader,
                            &mut d,
                            &k_bias,
                            num_tokens,
                            kv_dim,
                        )?;
                    }
                    {
                        let mut d = scratch
                            .qkv
                            .slice_mut(k_end_t..k_end_t + num_tokens * kv_dim);
                        Self::add_bias_f16_view_from_slice(
                            &self.stream,
                            &self.loader,
                            &mut d,
                            &v_bias,
                            num_tokens,
                            kv_dim,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }
}
