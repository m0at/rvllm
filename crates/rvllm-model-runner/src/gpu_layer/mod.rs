//! GPU Transformer Layer -- one complete transformer block on CUDA (f16 only).
//!
//! Split into 3 files:
//! - `mod.rs`: types, dispatch, shared kernel helpers
//! - `decode.rs`: T=1 single-token decode paths (fused GEMV, FP8 cublasLt)
//! - `batched.rs`: T>=1 batched decode / prefill path (cuBLAS/CUTLASS GEMMs)

#[cfg(feature = "cuda")]
mod batched;
#[cfg(feature = "cuda")]
mod decode;

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, DeviceSlice,
        LaunchConfig, PushKernelArg,
    };
    use half::f16;
    use tracing::info;

    use rvllm_core::error::{LLMError, Result};
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::kernel_loader::KernelLoader;

    // ===================================================================
    // Types
    // ===================================================================

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
    pub struct GpuLayerWeights<'a> {
        pub input_layernorm: &'a CudaSlice<f16>,
        pub q_proj: &'a CudaSlice<f16>,
        pub k_proj: &'a CudaSlice<f16>,
        pub v_proj: &'a CudaSlice<f16>,
        pub o_proj: &'a CudaSlice<f16>,
        pub post_attention_layernorm: &'a CudaSlice<f16>,
        pub gate_proj: &'a CudaSlice<f16>,
        pub up_proj: &'a CudaSlice<f16>,
        pub down_proj: &'a CudaSlice<f16>,
        pub fused_qkv: Option<&'a CudaSlice<f16>>,
        pub fused_gate_up: Option<&'a CudaSlice<f16>>,
        pub qkv_bias: Option<&'a CudaSlice<f16>>,
        pub fused_qkv_fp8: Option<&'a CudaSlice<u8>>,
        pub fused_qkv_fp8_scale: Option<&'a CudaSlice<f16>>,
        pub o_proj_fp8: Option<&'a CudaSlice<u8>>,
        pub o_proj_fp8_scale: Option<&'a CudaSlice<f16>>,
        pub fused_gate_up_fp8: Option<&'a CudaSlice<u8>>,
        pub fused_gate_up_fp8_scale: Option<&'a CudaSlice<f16>>,
        pub down_proj_fp8: Option<&'a CudaSlice<u8>>,
        pub down_proj_fp8_scale: Option<&'a CudaSlice<f16>>,
    }

    /// Metadata needed for a single layer forward pass.
    pub struct GpuLayerInput<'a> {
        pub hidden_states: &'a CudaSlice<f16>,
        pub positions: CudaView<'a, i32>,
        pub key_cache: &'a CudaSlice<f16>,
        pub value_cache: &'a CudaSlice<f16>,
        pub block_tables: CudaView<'a, i32>,
        pub context_lens: CudaView<'a, i32>,
        pub slot_mapping: CudaView<'a, i32>,
        pub num_tokens: usize,
        pub num_seqs: usize,
        pub max_context_len: u32,
        pub block_size: usize,
        pub is_prefill: bool,
        pub seq_start_pos: CudaView<'a, i32>,
        pub rope_cos: &'a CudaSlice<f32>,
        pub rope_sin: &'a CudaSlice<f32>,
        pub fp8_input_scratch_ptr: u64,
        pub fp8_input_scratch_len: usize,
    }

    /// Pre-allocated scratch buffers for the batched forward path.
    /// Required (not optional) for ForwardPath::Batched.
    pub struct LayerScratchRef<'a> {
        pub normed: &'a mut CudaSlice<f16>,
        pub residual: &'a mut CudaSlice<f16>,
        pub qkv: &'a mut CudaSlice<f16>,
        pub attn_out: &'a mut CudaSlice<f16>,
        pub o_proj: &'a mut CudaSlice<f16>,
        pub gate_up: &'a mut CudaSlice<f16>,
        pub silu_out: &'a mut CudaSlice<f16>,
        pub down: &'a mut CudaSlice<f16>,
    }

    // ===================================================================
    // FSM: deterministic forward path selection
    // ===================================================================

    /// Which forward path to execute. Determined by the caller (GpuModelRunner)
    /// based on (num_tokens, is_prefill, has_fp8_weights, has_cublaslt).
    /// Each variant has exactly one code path -- no fallbacks.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ForwardPath {
        /// T=1 decode with FP8 cublasLt GEMMs.
        Fp8Decode,
        /// T=1 decode with fused f16 GEMV kernels.
        FusedDecode,
        /// T=1 decode: single cooperative kernel executes the entire transformer layer.
        /// Requires persistent_layer_decode cubin. Eliminates ~6 kernel launches per layer.
        PersistentDecode,
        /// T=1 decode with the persistent_v2 runner-level path.
        PersistentV2Decode,
        /// T=1 decode with the persistent_v3 runner-level path.
        PersistentV3Decode,
        /// T=1 decode with separate norm + cuBLAS GEMM (better BW than fused GEMV).
        /// Enable with RVLLM_CUBLAS_DECODE=1.
        CublasGemvDecode,
        /// All 28 layers + LM head in ONE kernel launch via interpreter.
        /// Enable with RVLLM_MEGAKERNEL=1.
        MegakernelDecode,
        /// All 28 layers + LM head in ONE kernel launch via the v2 interpreter.
        MegakernelV2Decode,
        /// T>=1 batched decode or prefill with cuBLAS/CUTLASS GEMMs.
        /// Always requires scratch buffers.
        Batched,
    }

    /// GEMM implementation strategy for the batched path.
    /// Determined once at model load time based on CUTLASS .so availability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum GemmStrategy {
        /// cuBLAS/cublasLt for QKV, O-proj, and down-proj; CUTLASS only for gateup+silu.
        Hybrid,
        /// CUTLASS FFI: fused oproj+residual, fused gateup+silu.
        Cutlass,
        /// cuBLAS (+ cublasLt for small M): separate kernels.
        Cublas,
    }

    // ===================================================================
    // GpuTransformerLayer
    // ===================================================================

    pub struct GpuTransformerLayer {
        pub(crate) config: GpuLayerConfig,
        pub(crate) stream: Arc<CudaStream>,
        pub(crate) loader: Arc<KernelLoader>,
    }

    impl GpuTransformerLayer {
        pub fn new(
            config: GpuLayerConfig,
            stream: Arc<CudaStream>,
            loader: Arc<KernelLoader>,
        ) -> Self {
            Self {
                config,
                stream,
                loader,
            }
        }

        /// Dispatch to the correct forward path based on `path` enum.
        ///
        /// Returns `Some((residual, mlp_out))` for decode paths,
        /// `None` for batched path (results written to scratch buffers).
        pub fn forward(
            &self,
            path: ForwardPath,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeights<'_>,
            blas: &CublasHandle,
            prev_mlp_out: Option<&CudaSlice<f16>>,
            lt: Option<&crate::CublasLtRef>,
            scratch: Option<&mut LayerScratchRef<'_>>,
            gemm_strategy: GemmStrategy,
            cutlass: Option<&rvllm_gpu::cutlass_ffi::CutlassKernels>,
        ) -> Result<Option<(CudaSlice<f16>, CudaSlice<f16>)>> {
            match path {
                ForwardPath::Fp8Decode => {
                    #[cfg(feature = "cublaslt")]
                    {
                        let lt = lt.expect("Fp8Decode requires cublasLt");
                        Ok(Some(self.forward_fp8_decode(
                            input,
                            weights,
                            blas,
                            lt,
                            prev_mlp_out,
                        )?))
                    }
                    #[cfg(not(feature = "cublaslt"))]
                    {
                        Err(LLMError::GpuError(
                            "Fp8Decode requires cublaslt feature".into(),
                        ))
                    }
                }
                ForwardPath::FusedDecode => Ok(Some(self.forward_fused_decode(
                    input,
                    weights,
                    blas,
                    lt,
                    prev_mlp_out,
                )?)),
                ForwardPath::PersistentDecode => Ok(Some(self.forward_persistent_decode(
                    input,
                    weights,
                    prev_mlp_out,
                )?)),
                ForwardPath::PersistentV2Decode => Err(LLMError::GpuError(
                    "PersistentV2Decode is handled at the runner level, not per-layer".into(),
                )),
                ForwardPath::PersistentV3Decode => Err(LLMError::GpuError(
                    "PersistentV3Decode is handled at the runner level, not per-layer".into(),
                )),
                ForwardPath::CublasGemvDecode => Ok(Some(self.forward_cublas_decode(
                    input,
                    weights,
                    blas,
                    lt,
                    prev_mlp_out,
                )?)),
                ForwardPath::MegakernelDecode => Err(LLMError::GpuError(
                    "MegakernelDecode is handled at the runner level, not per-layer".into(),
                )),
                ForwardPath::MegakernelV2Decode => Err(LLMError::GpuError(
                    "MegakernelV2Decode is handled at the runner level, not per-layer".into(),
                )),
                ForwardPath::Batched => {
                    let scratch = scratch.expect("Batched path requires scratch buffers");
                    self.forward_batched(
                        input,
                        weights,
                        blas,
                        lt,
                        prev_mlp_out,
                        scratch,
                        gemm_strategy,
                        cutlass,
                    )?;
                    Ok(None)
                }
            }
        }

        // ===================================================================
        // Shared kernel helpers (used by decode.rs and batched.rs)
        // ===================================================================

        /// RMSNorm f16 (allocating).
        pub(crate) fn rms_norm_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let n = num_tokens * hidden_size;
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
                stream
                    .launch_builder(&kernel)
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

        /// RMSNorm f16 into pre-allocated output.
        pub(crate) fn rms_norm_f16_into(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
            output: &mut CudaSlice<f16>,
        ) -> Result<()> {
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(output)
                    .arg(input)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16_into launch: {e}")))?;
            }
            Ok(())
        }

        /// Fused residual add + RMSNorm (allocating). Returns (normed, residual).
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
            let kernel = loader.get_func(
                "fused_residual_rmsnorm_f16",
                "fused_residual_rmsnorm_f16_kernel",
            )?;
            unsafe {
                stream
                    .launch_builder(&kernel)
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

        /// Fused residual add + RMSNorm into pre-allocated buffers.
        pub(crate) fn fused_residual_rmsnorm_f16_into(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            add: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
            output: &mut CudaSlice<f16>,
            residual: &mut CudaSlice<f16>,
        ) -> Result<()> {
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func(
                "fused_residual_rmsnorm_f16",
                "fused_residual_rmsnorm_f16_kernel",
            )?;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(output)
                    .arg(residual)
                    .arg(input)
                    .arg(add)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_rn_f16_into launch: {e}")))?;
            }
            Ok(())
        }

        /// hgemm dispatch: GEMV for M=1, cublasLt for small M, cuBLAS for large M.
        pub(crate) fn hgemm_dispatch(
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
            Self::hgemm_dispatch_fp8(stream, blas, lt, input, weight, m, n, k, loader, None, None)
        }

        /// hgemm dispatch with optional FP8 weights (allocating).
        pub(crate) fn hgemm_dispatch_fp8(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            lt: Option<&crate::CublasLtRef>,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
            loader: &KernelLoader,
            fp8_weight: Option<&CudaSlice<u8>>,
            _fp8_input_scratch: Option<&mut CudaSlice<u8>>,
        ) -> Result<CudaSlice<f16>> {
            let mut output = unsafe { stream.alloc::<f16>(m * n) }
                .map_err(|e| LLMError::GpuError(format!("hgemm_dispatch: {e}")))?;

            #[cfg(feature = "cublaslt")]
            if let (Some(w_fp8), Some(lt_ops)) = (fp8_weight, lt) {
                let cast_kernel = loader
                    .get_func("cast_f16_to_fp8", "cast_f16_to_fp8_kernel")
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "Required cast_f16_to_fp8 kernel missing (FP8 weights present): {e}"
                        ))
                    })?;
                let total_elems = m * k;
                let mut fp8_input_buf = unsafe { stream.alloc::<u8>(total_elems) }
                    .map_err(|e| LLMError::GpuError(format!("fp8 input alloc: {e}")))?;
                let cast_cfg = LaunchConfig {
                    grid_dim: (((total_elems + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (fp8_in_ptr, _ig) = DevicePtrMut::device_ptr_mut(&mut fp8_input_buf, stream);
                let (input_ptr, _ipg) = DevicePtr::device_ptr(input, stream);
                unsafe {
                    stream
                        .launch_builder(&cast_kernel)
                        .arg(&fp8_in_ptr)
                        .arg(&input_ptr)
                        .arg(&(total_elems as i32))
                        .launch(cast_cfg)
                        .map_err(|e| LLMError::GpuError(format!("cast f16->fp8: {e}")))?;
                }
                let (w_ptr, _wg) = DevicePtr::device_ptr(w_fp8, stream);
                let (out_ptr, _og) = DevicePtrMut::device_ptr_mut(&mut output, stream);
                lt_ops.fp8_gemm_a_bt_raw(m, n, k, fp8_in_ptr, w_ptr, out_ptr)?;
                drop((_wg, _og));
                return Ok(output);
            }

            if m == 1 {
                let kernel = loader
                    .get_func("gemv_f16", "gemv_f16_kernel")
                    .map_err(|e| {
                        LLMError::GpuError(format!("Required gemv_f16 kernel missing for M=1: {e}"))
                    })?;
                let cfg = LaunchConfig {
                    grid_dim: (n as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&kernel)
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

            #[cfg(feature = "cublaslt")]
            if let Some(lt_ops) = lt {
                if lt_ops.should_use_f16_for_shape(m, n, k) {
                    lt_ops.hgemm_a_bt(m, n, k, 1.0, input, weight, 0.0, &mut output)?;
                    return Ok(output);
                }
            }
            blas.hgemm(m, n, k, f16::ONE, input, weight, f16::ZERO, &mut output)?;
            Ok(output)
        }

        /// hgemm dispatch with optional FP8 into pre-allocated output.
        pub(crate) fn hgemm_dispatch_fp8_into(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            lt: Option<&crate::CublasLtRef>,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
            loader: &KernelLoader,
            fp8_weight: Option<&CudaSlice<u8>>,
            output: &mut CudaSlice<f16>,
        ) -> Result<()> {
            #[cfg(feature = "cublaslt")]
            if let (Some(w_fp8), Some(lt_ops)) = (fp8_weight, lt) {
                let cast_kernel = loader
                    .get_func("cast_f16_to_fp8", "cast_f16_to_fp8_kernel")
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "Required cast_f16_to_fp8 kernel missing (FP8 weights present): {e}"
                        ))
                    })?;
                let total_elems = m * k;
                let mut fp8_input_buf = unsafe { stream.alloc::<u8>(total_elems) }
                    .map_err(|e| LLMError::GpuError(format!("fp8 input alloc: {e}")))?;
                let cast_cfg = LaunchConfig {
                    grid_dim: (((total_elems + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (fp8_in_ptr, _ig) = DevicePtrMut::device_ptr_mut(&mut fp8_input_buf, stream);
                let (input_ptr, _ipg) = DevicePtr::device_ptr(input, stream);
                unsafe {
                    stream
                        .launch_builder(&cast_kernel)
                        .arg(&fp8_in_ptr)
                        .arg(&input_ptr)
                        .arg(&(total_elems as i32))
                        .launch(cast_cfg)
                        .map_err(|e| LLMError::GpuError(format!("cast f16->fp8: {e}")))?;
                }
                let (w_ptr, _wg) = DevicePtr::device_ptr(w_fp8, stream);
                let (out_ptr, _og) = DevicePtrMut::device_ptr_mut(output, stream);
                lt_ops.fp8_gemm_a_bt_raw(m, n, k, fp8_in_ptr, w_ptr, out_ptr)?;
                return Ok(());
            }

            #[cfg(feature = "cublaslt")]
            if let Some(lt_ops) = lt {
                if lt_ops.should_use_f16_for_shape(m, n, k) {
                    lt_ops.hgemm_a_bt(m, n, k, 1.0, input, weight, 0.0, output)?;
                    return Ok(());
                }
            }
            blas.hgemm(m, n, k, f16::ONE, input, weight, f16::ZERO, output)?;
            Ok(())
        }

        /// hgemm dispatch into a CudaViewMut (sub-slice of a larger buffer).
        pub(crate) fn hgemm_dispatch_into(
            blas: &CublasHandle,
            lt: Option<&crate::CublasLtRef>,
            input: &CudaSlice<f16>,
            weight: &(impl DevicePtr<f16> + DeviceSlice<f16>),
            m: usize,
            n: usize,
            k: usize,
            out: &mut CudaViewMut<'_, f16>,
        ) -> Result<()> {
            #[cfg(feature = "cublaslt")]
            if let Some(lt_ops) = lt {
                if lt_ops.should_use_f16_for_shape(m, n, k) {
                    return lt_ops.hgemm_a_bt_into(m, n, k, 1.0, input, weight, 0.0, out);
                }
            }
            blas.hgemm_into(m, n, k, 1.0, input, weight, 0.0, out)
        }

        /// Add bias f16 in-place on a CudaViewMut.
        pub(crate) fn add_bias_f16_view(
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
                stream
                    .launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&(dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// Add bias f16 using a bias view (sliced from fused bias).
        pub(crate) fn add_bias_f16_view_from_slice(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            tensor: &mut CudaViewMut<'_, f16>,
            bias: &CudaView<'_, f16>,
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
                stream
                    .launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&(dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// RoPE f16 in-place on Q/K views.
        pub(crate) fn apply_rotary_embedding_f16_views(
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
            if num_tokens == 0 {
                return Ok(());
            }
            let kernel = loader.get_func("rotary_embedding_f16", "rotary_embedding_f16_kernel")?;
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(q)
                    .arg(k)
                    .arg(rope_cos)
                    .arg(rope_sin)
                    .arg(positions)
                    .arg(&(num_tokens as i32))
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// KV cache write: f16 K/V -> f16 paged cache.
        pub(crate) fn cache_write_f16_views(
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
            let kernel =
                loader.get_func("reshape_and_cache_f16", "reshape_and_cache_f16io_kernel")?;
            let threads = kv_dim.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(k)
                    .arg(v)
                    .arg(slot_mapping)
                    .arg(&(num_tokens as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cache_write_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// Fused RoPE + KV cache write (single kernel, used by T=1 decode paths).
        pub(crate) fn fused_rope_cache_write(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            qkv: &mut CudaSlice<f16>,
            input: &GpuLayerInput<'_>,
            q_dim: usize,
            kv_dim: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            num_tokens: usize,
        ) -> Result<()> {
            let fk = loader
                .get_func("fused_rope_cache", "fused_rope_cache_f16_kernel")
                .map_err(|e| {
                    LLMError::GpuError(format!("Required fused_rope_cache kernel missing: {e}"))
                })?;
            let (mut q_part, mut kv_rest) = qkv.split_at_mut(q_dim);
            let (mut k_part, v_part) = kv_rest.split_at_mut(kv_dim);
            let v_view = v_part.slice(..kv_dim);
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            unsafe {
                stream
                    .launch_builder(&fk)
                    .arg(&mut q_part)
                    .arg(&mut k_part)
                    .arg(&v_view)
                    .arg(input.key_cache)
                    .arg(input.value_cache)
                    .arg(input.rope_cos)
                    .arg(input.rope_sin)
                    .arg(&input.positions)
                    .arg(&input.slot_mapping)
                    .arg(&(num_tokens as i32))
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .launch(LaunchConfig {
                        grid_dim: (num_tokens as u32, grid_y, 1),
                        block_dim: (half_dim.min(1024) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| LLMError::GpuError(format!("fused rope+cache: {e}")))?;
            }
            Ok(())
        }

        /// FA3 prefill attention with f16 I/O.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn prefill_attention_f16io(
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
            let kernel = loader
                .get_func(
                    "flash_attention_3_prefill",
                    "flash_attention_3_prefill_f16io_kernel",
                )
                .map_err(|e| LLMError::GpuError(format!("load prefill f16io kernel: {e}")))?;

            let out_len = num_tokens * num_heads * head_dim;
            let mut output = unsafe { stream.alloc::<f16>(out_len) }
                .map_err(|e| LLMError::GpuError(format!("prefill_attn_f16io alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();
            const FA3_BC: usize = 64;
            const FA3_THREADS: u32 = 256;
            let smem = FA3_BC * head_dim * std::mem::size_of::<u16>()
                + (FA3_BC + 8) * std::mem::size_of::<f32>();
            let shared_mem_bytes = smem as u32;

            if num_seqs == 0 {
                return Err(LLMError::GpuError(
                    "prefill_attention_f16io: num_seqs == 0".into(),
                ));
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
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(q)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(block_tables)
                    .arg(context_lens)
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

        /// FA3 v3 decode attention with f16 I/O.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn decode_attention_f16io(
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
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let p_num_heads = num_heads as i32;
            let p_num_kv_heads = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_blocks = (block_tables.len() / num_seqs.max(1)) as i32;
            let heads_per_group = if num_kv_heads > 0 {
                num_heads / num_kv_heads
            } else {
                1
            };

            let num_splits: i32 = if max_context_len <= 512 {
                1
            } else if max_context_len <= 2048 {
                2
            } else if max_context_len <= 8192 {
                4
            } else {
                8
            };

            // v3 GQA kernel
            if num_heads != num_kv_heads && heads_per_group <= 8 && head_dim % 8 == 0 {
                let v3_gqa = loader.get_func("flash_attention_3_v3", "fa3_v3_decode_gqa_kernel")?;
                const V3_BC: usize = 64;
                const V3_THREADS: u32 = 256;
                const V3_MAX_HPG: usize = 8;
                const V3_SCORE_STRIDE: usize = V3_BC + 1;

                // Double-buffered: 2x KV tile buffers (K + V)
                let smem = 2 * V3_BC * head_dim * std::mem::size_of::<u16>()
                    + V3_MAX_HPG * V3_SCORE_STRIDE * std::mem::size_of::<f32>()
                    + 8 * std::mem::size_of::<f32>();
                let shared_mem_bytes = smem as u32;
                let p_max_context = max_context_len as i32;

                if shared_mem_bytes > 49152 {
                    v3_gqa.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared_mem_bytes as i32,
                    ).map_err(|e| LLMError::GpuError(format!("v3 GQA set smem: {e}")))?;
                }

                return Self::launch_v3_attention(
                    stream,
                    loader,
                    &v3_gqa,
                    q,
                    key_cache,
                    value_cache,
                    block_tables,
                    context_lens,
                    scale,
                    p_num_heads,
                    p_num_kv_heads,
                    p_head_dim,
                    p_block_size,
                    p_max_context,
                    p_max_blocks,
                    num_splits,
                    num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    out_len,
                    shared_mem_bytes,
                    V3_THREADS,
                    true,
                );
            }

            // v3 non-GQA kernel (MHA)
            if head_dim % 8 == 0 {
                let v3_kernel = loader.get_func("flash_attention_3_v3", "fa3_v3_decode_kernel")?;
                const V3_BC: usize = 64;
                const V3_THREADS: u32 = 256;
                // Double-buffered: 2x KV tile buffers (K + V)
                let smem = 2 * V3_BC * head_dim * std::mem::size_of::<u16>()
                    + (V3_BC + 8) * std::mem::size_of::<f32>();
                let shared_mem_bytes = smem as u32;

                if shared_mem_bytes > 49152 {
                    v3_kernel.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared_mem_bytes as i32,
                    ).map_err(|e| LLMError::GpuError(format!("v3 set smem: {e}")))?;
                }

                return Self::launch_v3_attention(
                    stream,
                    loader,
                    &v3_kernel,
                    q,
                    key_cache,
                    value_cache,
                    block_tables,
                    context_lens,
                    scale,
                    p_num_heads,
                    p_num_kv_heads,
                    p_head_dim,
                    p_block_size,
                    max_context_len as i32,
                    p_max_blocks,
                    num_splits,
                    num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    out_len,
                    shared_mem_bytes,
                    V3_THREADS,
                    false,
                );
            }

            Err(LLMError::GpuError(format!(
                "No attention kernel for num_heads={num_heads} num_kv_heads={num_kv_heads} \
                 head_dim={head_dim}. Ensure flash_attention_3_v3.ptx is compiled."
            )))
        }

        /// Launch v3 decode attention (shared by GQA and MHA variants).
        #[allow(clippy::too_many_arguments)]
        fn launch_v3_attention(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            kernel: &cudarc::driver::CudaFunction,
            q: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            scale: f32,
            p_num_heads: i32,
            p_num_kv_heads: i32,
            p_head_dim: i32,
            p_block_size: i32,
            p_max_context: i32,
            p_max_blocks: i32,
            num_splits: i32,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            out_len: usize,
            shared_mem_bytes: u32,
            threads: u32,
            is_gqa: bool,
        ) -> Result<CudaSlice<f16>> {
            let grid_y = if is_gqa { num_kv_heads } else { num_heads };
            let mut output = unsafe { stream.alloc::<f16>(out_len) }
                .map_err(|e| LLMError::GpuError(format!("v3 output alloc: {e}")))?;

            if num_splits == 1 {
                let dummy = unsafe { stream.alloc::<f32>(1) }
                    .map_err(|e| LLMError::GpuError(format!("v3 dummy: {e}")))?;
                let cfg = LaunchConfig {
                    grid_dim: (num_seqs as u32, grid_y as u32, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes,
                };
                unsafe {
                    stream
                        .launch_builder(kernel)
                        .arg(&mut output)
                        .arg(&dummy)
                        .arg(&dummy)
                        .arg(&dummy)
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
                        .arg(&p_max_context)
                        .arg(&p_max_blocks)
                        .arg(&num_splits)
                        .launch(cfg)
                        .map_err(|e| LLMError::GpuError(format!("v3 decode: {e}")))?;
                }
                return Ok(output);
            }

            // Multi-split: workspace + combine kernel
            let ws_size = (num_splits as usize) * num_seqs * num_heads;
            let mut p_out = unsafe { stream.alloc::<f32>(ws_size * head_dim) }
                .map_err(|e| LLMError::GpuError(format!("v3 p_out: {e}")))?;
            let mut p_max = unsafe { stream.alloc::<f32>(ws_size) }
                .map_err(|e| LLMError::GpuError(format!("v3 p_max: {e}")))?;
            let mut p_sum = unsafe { stream.alloc::<f32>(ws_size) }
                .map_err(|e| LLMError::GpuError(format!("v3 p_sum: {e}")))?;

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, grid_y as u32, num_splits as u32),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            };
            unsafe {
                stream
                    .launch_builder(kernel)
                    .arg(&mut output)
                    .arg(&mut p_out)
                    .arg(&mut p_max)
                    .arg(&mut p_sum)
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
                    .arg(&p_max_context)
                    .arg(&p_max_blocks)
                    .arg(&num_splits)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("v3 decode: {e}")))?;
            }

            let combine = loader
                .get_func("flash_attention_3_v3", "fa3_v3_combine_f16_kernel")
                .map_err(|e| LLMError::GpuError(format!("v3 combine missing: {e}")))?;
            let p_num_seqs = num_seqs as i32;
            unsafe {
                stream
                    .launch_builder(&combine)
                    .arg(&mut output)
                    .arg(&p_out)
                    .arg(&p_max)
                    .arg(&p_sum)
                    .arg(context_lens)
                    .arg(&p_num_seqs)
                    .arg(&p_num_heads)
                    .arg(&p_head_dim)
                    .arg(&num_splits)
                    .launch(LaunchConfig {
                        grid_dim: (num_seqs as u32, num_heads as u32, 1),
                        block_dim: (head_dim as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| LLMError::GpuError(format!("v3 combine: {e}")))?;
            }

            Ok(output)
        }

        /// Fused SiLU*mul on a contiguous [gate || up] buffer.
        pub(crate) fn fused_silu_mul_f16_split(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate_up: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            let gate_view = gate_up.slice(..n);
            let up_view = gate_up.slice(n..n * 2);
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
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&gate_view)
                    .arg(&up_view)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_silu_mul_f16_split launch: {e}"))
                    })?;
            }
            Ok(output)
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles_without_cuda() {
        // Under mock-gpu the `inner` module is not compiled.
        // This test confirms that the crate still builds cleanly.
        assert!(true);
    }
}
