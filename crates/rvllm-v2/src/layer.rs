use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use rvllm_attention::choose_num_splits;
use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;
use rvllm_gpu::cublaslt_ops::CublasLtOps;
use rvllm_gpu::cutlass_ffi::CutlassKernels;
use rvllm_gpu::kernel_loader::KernelLoader;

// ===================================================================
// cuBLASLt-first GEMM dispatch: autotuned cuBLASLt -> cuBLAS fallback
// ===================================================================

fn hgemm_dispatch(
    lt_ops: Option<&CublasLtOps>,
    cublas: &CublasHandle,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &impl cudarc::driver::DevicePtr<f16>,
    b: &impl cudarc::driver::DevicePtr<f16>,
    beta: f32,
    c: &mut impl cudarc::driver::DevicePtrMut<f16>,
) -> Result<()> {
    if let Some(lt) = lt_ops {
        if lt.should_use_f16_for_shape(m, n, k) {
            return lt.hgemm_a_bt_into(m, n, k, alpha, a, b, beta, c)
                .map_err(|e| LLMError::GpuError(format!("cublasLt hgemm: {e}")));
        }
    }
    cublas.hgemm_into(m, n, k, alpha, a, b, beta, c)
}

// ===================================================================
// FP8 GEMM dispatch: quantize activation on GPU, then cuBLASLt FP8
// ===================================================================

fn fp8_gemm_dispatch(
    lt_ops: &CublasLtOps,
    loader: &KernelLoader,
    stream: &CudaStream,
    m: usize, n: usize, k: usize,
    act_f16: &CudaSlice<f16>,              // [m, k] f16 activation
    weight_fp8: &CudaSlice<u8>,            // [n, k] FP8 weight
    weight_scale: &CudaSlice<f32>,         // [1] per-tensor weight scale
    output_f16: &mut CudaSlice<f16>,       // [m, n] f16 output
    act_fp8_scratch: &mut CudaSlice<u8>,   // [m * k] scratch for FP8 activation
    act_scale_scratch: &mut CudaSlice<f32>, // [1] scratch for activation scale
    absmax_scratch: &mut CudaSlice<f32>,    // [1] scratch for absmax reduction
) -> Result<()> {
    let num_elements = m * k;

    // Step 1: Quantize activation to FP8 on GPU (multi-block: find absmax, then quantize)
    let absmax_kernel = loader
        .get_func("quantize_activation_fp8", "find_absmax_fp8_kernel")
        .map_err(|e| LLMError::GpuError(format!("find_absmax_fp8 kernel: {e}")))?;
    let quant_kernel = loader
        .get_func("quantize_activation_fp8", "apply_fp8_quantize_kernel")
        .map_err(|e| LLMError::GpuError(format!("apply_fp8_quantize kernel: {e}")))?;

    // Grid size: enough blocks to saturate SMs, each block handles a chunk
    let num_blocks = ((num_elements + 1023) / 1024).min(128) as u32;

    unsafe {
        // Zero the absmax buffer (scoped borrow for memset)
        {
            let (abs_ptr, _g) = absmax_scratch.device_ptr_mut(stream);
            cudarc::driver::result::memset_d8_async(
                abs_ptr, 0, std::mem::size_of::<f32>(), stream.cu_stream(),
            ).map_err(|e| LLMError::GpuError(format!("memset absmax: {e}")))?;
        }

        // Phase 1: find global absmax via atomicMax across blocks
        stream
            .launch_builder(&absmax_kernel)
            .arg(&mut *absmax_scratch)     // global_absmax[0]
            .arg(act_f16)
            .arg(&(num_elements as i32))
            .launch(LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| LLMError::GpuError(format!("find_absmax_fp8 launch: {e}")))?;

        // Phase 2: quantize using computed absmax (writes scale to act_scale_scratch)
        stream
            .launch_builder(&quant_kernel)
            .arg(&mut *act_fp8_scratch)    // output_fp8
            .arg(&mut *act_scale_scratch)  // output_scale
            .arg(act_f16)
            .arg(&*absmax_scratch)         // global_absmax (read-only)
            .arg(&(num_elements as i32))
            .launch(LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| LLMError::GpuError(format!("apply_fp8_quantize launch: {e}")))?;
    }

    // Step 2: FP8 GEMM via cuBLASLt with scale pointers
    let (w_ptr, _gw) = weight_fp8.device_ptr(stream);
    let (ws_ptr, _gs) = weight_scale.device_ptr(stream);
    let (afp8_ptr, _gafp8) = act_fp8_scratch.device_ptr(stream);
    let (as_ptr, _gas) = act_scale_scratch.device_ptr(stream);
    let (out_ptr, _gout) = output_f16.device_ptr_mut(stream);

    lt_ops.fp8_gemm_scaled_a_bt(
        m, n, k,
        w_ptr as u64,
        ws_ptr as u64,
        afp8_ptr as u64,
        as_ptr as u64,
        out_ptr as u64,
    ).map_err(|e| LLMError::GpuError(format!("fp8_gemm_scaled: {e}")))
}

// ===================================================================
// GemmStrategy (defined in v2/runner.rs at runtime, mirrored here)
// ===================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmStrategy {
    /// cuBLAS for QKV, O-proj, down-proj; CUTLASS fused for GateUp+SiLU.
    Hybrid,
    /// CUTLASS for oproj+residual, gateup+silu; cuBLAS for remainder.
    Cutlass,
    /// cuBLAS for everything.
    Cublas,
}

// ===================================================================
// LayerConfig
// ===================================================================

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position: usize,
    pub block_size: usize,
}

impl LayerConfig {
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn qkv_dim(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }

    pub fn gate_up_dim(&self) -> usize {
        self.intermediate_size * 2
    }
}

// ===================================================================
// LayerWeights -- references to per-layer weight slices (all f16)
// ===================================================================

pub struct LayerWeights<'a> {
    pub qkv_weight: &'a CudaSlice<f16>,
    pub o_proj_weight: &'a CudaSlice<f16>,
    pub gate_up_weight: &'a CudaSlice<f16>,
    pub down_proj_weight: &'a CudaSlice<f16>,
    pub input_layernorm_weight: &'a CudaSlice<f16>,
    pub post_attention_layernorm_weight: &'a CudaSlice<f16>,
    // FP8 quantized weights (None = use f16 path)
    pub fp8: Option<Fp8LayerWeightRefs<'a>>,
}

pub struct Fp8LayerWeightRefs<'a> {
    pub qkv_fp8: &'a CudaSlice<u8>,
    pub qkv_scale: &'a CudaSlice<f32>,
    pub o_proj_fp8: &'a CudaSlice<u8>,
    pub o_proj_scale: &'a CudaSlice<f32>,
    pub gate_up_fp8: &'a CudaSlice<u8>,
    pub gate_up_scale: &'a CudaSlice<f32>,
    pub down_proj_fp8: &'a CudaSlice<u8>,
    pub down_proj_scale: &'a CudaSlice<f32>,
}

// ===================================================================
// F16LayerScratch -- pre-allocated scratch buffers for one layer pass
// ===================================================================

pub struct F16LayerScratch {
    pub normed: CudaSlice<f16>,
    pub qkv_buf: CudaSlice<f16>,
    pub attn_out: CudaSlice<f16>,
    pub o_proj_out: CudaSlice<f16>,
    pub gate_up_out: CudaSlice<f16>,
    pub silu_out: CudaSlice<f16>,
    pub gateup_workspace: CudaSlice<u8>,
    pub attn_split_out: CudaSlice<f32>,
    pub attn_split_max: CudaSlice<f32>,
    pub attn_split_sum: CudaSlice<f32>,
    // Intermediate buffer for residual sum within a layer (input + prev_mlp)
    pub residual_tmp: CudaSlice<f16>,
    // FP8 activation quantization scratch (shared across all GEMMs per step)
    pub fp8_act_scratch: CudaSlice<u8>,
    pub fp8_act_scale: CudaSlice<f32>,
    pub fp8_absmax: CudaSlice<f32>,
}

impl F16LayerScratch {
    pub fn alloc(
        stream: &Arc<CudaStream>,
        config: &LayerConfig,
        max_batch_tokens: usize,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let q_dim = config.q_dim();
        let qkv_dim = config.qkv_dim();
        let intermediate = config.intermediate_size;
        let gate_up_dim = config.gate_up_dim();
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let t = max_batch_tokens;
        let max_splits = 16;

        let alloc_f16 = |n: usize| -> Result<CudaSlice<f16>> {
            stream
                .alloc_zeros::<f16>(n)
                .map_err(|e| LLMError::GpuError(format!("scratch alloc f16 ({n}): {e}")))
        };
        let alloc_f32 = |n: usize| -> Result<CudaSlice<f32>> {
            stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("scratch alloc f32 ({n}): {e}")))
        };
        let alloc_u8 = |n: usize| -> Result<CudaSlice<u8>> {
            stream
                .alloc_zeros::<u8>(n.max(1))
                .map_err(|e| LLMError::GpuError(format!("scratch alloc u8 ({n}): {e}")))
        };

        Ok(Self {
            normed: alloc_f16(t * hidden)?,
            qkv_buf: alloc_f16(t * qkv_dim)?,
            attn_out: alloc_f16(t * q_dim)?,
            o_proj_out: alloc_f16(t * hidden)?,
            gate_up_out: alloc_f16(t * gate_up_dim)?,
            silu_out: alloc_f16(t * intermediate)?,
            gateup_workspace: alloc_u8(t * gate_up_dim * std::mem::size_of::<f16>() * 4)?,
            attn_split_out: alloc_f32(max_splits * t * num_heads * head_dim)?,
            attn_split_max: alloc_f32(max_splits * t * num_heads)?,
            attn_split_sum: alloc_f32(max_splits * t * num_heads)?,
            residual_tmp: alloc_f16(t * hidden)?,
            // FP8 scratch: sized for largest activation (down_proj input = t * intermediate)
            fp8_act_scratch: alloc_u8(t * intermediate * 2)?,  // *2 for gate_up_dim
            fp8_act_scale: alloc_f32(1)?,
            fp8_absmax: alloc_f32(1)?,
        })
    }
}

// ===================================================================
// Attention metadata passed into the layer
// ===================================================================

pub struct AttentionMeta<'a> {
    pub positions: CudaView<'a, i32>,
    pub key_cache: &'a CudaSlice<f16>,
    pub value_cache: &'a CudaSlice<f16>,
    pub block_tables: CudaView<'a, i32>,
    pub context_lens: CudaView<'a, i32>,
    pub slot_mapping: CudaView<'a, i32>,
    pub seq_start_pos: CudaView<'a, i32>,
    pub num_tokens: usize,
    pub num_seqs: usize,
    pub max_context_len: u32,
    pub is_prefill: bool,
    pub rope_cos: &'a CudaSlice<f32>,
    pub rope_sin: &'a CudaSlice<f32>,
}

// ===================================================================
// GpuTransformerLayer
// ===================================================================

pub struct GpuTransformerLayer {
    config: LayerConfig,
    stream: Arc<CudaStream>,
    loader: Arc<KernelLoader>,
    // Persistent dummy buffer for single-pass attention (avoids per-step alloc)
    attn_dummy: CudaSlice<f32>,
}

impl GpuTransformerLayer {
    pub fn new(config: LayerConfig, stream: Arc<CudaStream>, loader: Arc<KernelLoader>) -> Self {
        let attn_dummy = stream.alloc_zeros::<f32>(1).expect("attn_dummy alloc");
        Self {
            config,
            stream,
            loader,
            attn_dummy,
        }
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn loader(&self) -> &KernelLoader {
        &self.loader
    }

    pub fn config_ref(&self) -> &LayerConfig {
        &self.config
    }

    pub fn rms_norm_pub(
        &self,
        input: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        output: &mut CudaSlice<f16>,
    ) -> Result<()> {
        self.rms_norm(input, weight, num_tokens, hidden_size, output)
    }

    pub fn fused_residual_rmsnorm_pub(
        &self,
        input: &CudaSlice<f16>,
        add: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        normed_out: &mut CudaSlice<f16>,
        residual_out: &mut CudaSlice<f16>,
    ) -> Result<()> {
        self.fused_residual_rmsnorm(
            input,
            add,
            weight,
            num_tokens,
            hidden_size,
            normed_out,
            residual_out,
        )
    }

    // =================================================================
    // THE SINGLE FORWARD PATH
    // =================================================================

    pub fn forward_batched_v2(
        &self,
        hidden: &CudaSlice<f16>,
        attn: &AttentionMeta<'_>,
        weights: &LayerWeights<'_>,
        scratch: &mut F16LayerScratch,
        prev_mlp_out: Option<&CudaSlice<f16>>,
        residual_write: &mut CudaSlice<f16>,
        down_write: &mut CudaSlice<f16>,
        gemm_strategy: GemmStrategy,
        cutlass: Option<&CutlassKernels>,
        cublas: &CublasHandle,
        lt_ops: Option<&CublasLtOps>,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_tokens = attn.num_tokens;
        let hidden_size = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let qkv_dim = cfg.qkv_dim();
        let intermediate = cfg.intermediate_size;
        let q_end = num_tokens * q_dim;
        let k_end = q_end + num_tokens * kv_dim;

        // Step 1: Input layernorm (+ residual add if prev_mlp exists)
        // When fused, the residual sum goes into scratch.residual_tmp as intermediate
        let residual_from_fused = if let Some(prev_mlp) = prev_mlp_out {
            self.fused_residual_rmsnorm(
                hidden,
                prev_mlp,
                weights.input_layernorm_weight,
                num_tokens,
                hidden_size,
                &mut scratch.normed,
                &mut scratch.residual_tmp,
            )?;
            true
        } else {
            self.rms_norm(
                hidden,
                weights.input_layernorm_weight,
                num_tokens,
                hidden_size,
                &mut scratch.normed,
            )?;
            false
        };

        // FP8 GEMM for QKV: quantize normed activation, FP8 matmul
        if let (Some(lt), Some(fp8w)) = (lt_ops, &weights.fp8) {
            fp8_gemm_dispatch(
                lt, &self.loader, &self.stream,
                num_tokens, qkv_dim, hidden_size,
                &scratch.normed, fp8w.qkv_fp8, fp8w.qkv_scale,
                &mut scratch.qkv_buf,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                &mut scratch.fp8_absmax,
            )?;
        } else {
            hgemm_dispatch(lt_ops, cublas, num_tokens, qkv_dim, hidden_size,
                1.0, &scratch.normed, weights.qkv_weight, 0.0, &mut scratch.qkv_buf)?;
        }

        self.apply_rope(&mut scratch.qkv_buf, attn, q_dim, kv_dim, q_end, num_tokens)?;
        self.kv_cache_write(&mut scratch.qkv_buf, attn, q_end, k_end, kv_dim, num_tokens)?;

        self.attention(
            &scratch.qkv_buf,
            attn,
            q_end,
            num_tokens,
            &mut scratch.attn_out,
            &mut scratch.attn_split_out,
            &mut scratch.attn_split_max,
            &mut scratch.attn_split_sum,
        )?;

        // O-proj GEMM (FP8 or F16)
        if let (Some(lt), Some(fp8w)) = (lt_ops, &weights.fp8) {
            fp8_gemm_dispatch(
                lt, &self.loader, &self.stream,
                num_tokens, hidden_size, q_dim,
                &scratch.attn_out, fp8w.o_proj_fp8, fp8w.o_proj_scale,
                &mut scratch.o_proj_out,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                &mut scratch.fp8_absmax,
            )?;
        } else {
            hgemm_dispatch(lt_ops, cublas, num_tokens, hidden_size, q_dim,
                1.0, &scratch.attn_out, weights.o_proj_weight, 0.0, &mut scratch.o_proj_out)?;
        }

        // Residual add + post-attention layernorm
        let residual_src: &CudaSlice<f16> = if residual_from_fused {
            &scratch.residual_tmp
        } else {
            hidden
        };
        self.fused_residual_rmsnorm(
            residual_src, &scratch.o_proj_out,
            weights.post_attention_layernorm_weight,
            num_tokens, hidden_size,
            &mut scratch.normed, residual_write,
        )?;

        // GateUp GEMM (FP8 or F16)
        let gate_up_dim = intermediate * 2;
        if let (Some(lt), Some(fp8w)) = (lt_ops, &weights.fp8) {
            fp8_gemm_dispatch(
                lt, &self.loader, &self.stream,
                num_tokens, gate_up_dim, hidden_size,
                &scratch.normed, fp8w.gate_up_fp8, fp8w.gate_up_scale,
                &mut scratch.gate_up_out,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                &mut scratch.fp8_absmax,
            )?;
        } else {
            hgemm_dispatch(lt_ops, cublas, num_tokens, gate_up_dim, hidden_size,
                1.0, &scratch.normed, weights.gate_up_weight, 0.0, &mut scratch.gate_up_out)?;
        }

        // SiLU activation
        let silu_fn = self.loader
            .get_func("silu_mul_interleaved", "silu_mul_interleaved_f16_kernel")
            .map_err(|e| LLMError::GpuError(format!("silu_mul_interleaved kernel: {e}")))?;
        let total_silu = (num_tokens * intermediate) as u32;
        unsafe {
            self.stream
                .launch_builder(&silu_fn)
                .arg(&mut scratch.silu_out)
                .arg(&scratch.gate_up_out)
                .arg(&(num_tokens as i32))
                .arg(&(intermediate as i32))
                .launch(LaunchConfig {
                    grid_dim: ((total_silu + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| LLMError::GpuError(format!("silu_mul: {e}")))?;
        }

        // Down projection GEMM (FP8 or F16) -> writes to caller's target
        if let (Some(lt), Some(fp8w)) = (lt_ops, &weights.fp8) {
            fp8_gemm_dispatch(
                lt, &self.loader, &self.stream,
                num_tokens, hidden_size, intermediate,
                &scratch.silu_out, fp8w.down_proj_fp8, fp8w.down_proj_scale,
                down_write,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                &mut scratch.fp8_absmax,
            )?;
        } else {
            hgemm_dispatch(lt_ops, cublas, num_tokens, hidden_size, intermediate,
                1.0, &scratch.silu_out, weights.down_proj_weight, 0.0, down_write)?;
        }

        Ok(())
    }

    // =================================================================
    // Step 1: RMSNorm kernels
    // =================================================================

    fn rms_norm(
        &self,
        input: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        output: &mut CudaSlice<f16>,
    ) -> Result<()> {
        let block_threads = hidden_size.min(1024) as u32;
        let kernel = self
            .loader
            .get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_threads, 1, 1),
            shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(output)
                .arg(input)
                .arg(weight)
                .arg(&self.config.rms_norm_eps)
                .arg(&(hidden_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("rms_norm: {e}")))?;
        }
        Ok(())
    }

    fn fused_residual_rmsnorm(
        &self,
        input: &CudaSlice<f16>,
        add: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        normed_out: &mut CudaSlice<f16>,
        residual_out: &mut CudaSlice<f16>,
    ) -> Result<()> {
        let block_threads = hidden_size.min(1024) as u32;
        let kernel = self.loader.get_func(
            "fused_residual_rmsnorm_f16",
            "fused_residual_rmsnorm_f16_kernel",
        )?;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_threads, 1, 1),
            shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(normed_out)
                .arg(residual_out)
                .arg(input)
                .arg(add)
                .arg(weight)
                .arg(&self.config.rms_norm_eps)
                .arg(&(hidden_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("fused_residual_rmsnorm: {e}")))?;
        }
        Ok(())
    }

    // =================================================================
    // Step 2: QKV projection -- GEMM dispatch by strategy
    // =================================================================

    fn qkv_projection(
        &self,
        normed: &CudaSlice<f16>,
        qkv_weight: &CudaSlice<f16>,
        num_tokens: usize,
        qkv_dim: usize,
        hidden_size: usize,
        qkv_out: &mut CudaSlice<f16>,
        _gemm_strategy: GemmStrategy,
        _cutlass: Option<&CutlassKernels>,
        cublas: &CublasHandle,
        lt_ops: Option<&CublasLtOps>,
    ) -> Result<()> {
        hgemm_dispatch(lt_ops, cublas, num_tokens, qkv_dim, hidden_size, 1.0, normed, qkv_weight, 0.0, qkv_out)
    }

    // =================================================================
    // Step 3: RoPE rotation
    // =================================================================

    fn apply_rope(
        &self,
        qkv: &mut CudaSlice<f16>,
        attn: &AttentionMeta<'_>,
        _q_dim: usize,
        kv_dim: usize,
        q_end: usize,
        num_tokens: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;

        if num_tokens == 1 {
            return Ok(());
        }

        let kernel = self
            .loader
            .get_func("rotary_embedding_f16", "rotary_embedding_f16_kernel")?;
        let half_dim = head_dim / 2;
        let grid_y = num_heads.max(num_kv_heads) as u32;
        let launch_cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, grid_y, 1),
            block_dim: (half_dim.min(1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (mut q_part, mut kv_part) = qkv.split_at_mut(q_end);
        let mut k_view = kv_part.slice_mut(..num_tokens * kv_dim);
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&mut q_part)
                .arg(&mut k_view)
                .arg(attn.rope_cos)
                .arg(attn.rope_sin)
                .arg(&attn.positions)
                .arg(&(num_tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(launch_cfg)
                .map_err(|e| LLMError::GpuError(format!("rope: {e}")))?;
        }
        Ok(())
    }

    // =================================================================
    // Step 4: KV cache write
    // =================================================================

    fn kv_cache_write(
        &self,
        qkv: &mut CudaSlice<f16>,
        attn: &AttentionMeta<'_>,
        q_end: usize,
        k_end: usize,
        kv_dim: usize,
        num_tokens: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;

        if num_tokens == 1 {
            return self.fused_rope_cache_write_single(qkv, attn);
        }

        let k_view = qkv.slice(q_end..k_end);
        let v_view = qkv.slice(k_end..k_end + num_tokens * kv_dim);
        let kernel = self
            .loader
            .get_func("reshape_and_cache_f16", "reshape_and_cache_f16io_kernel")?;
        let threads = kv_dim.min(1024) as u32;
        let launch_cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(attn.key_cache)
                .arg(attn.value_cache)
                .arg(&k_view)
                .arg(&v_view)
                .arg(&attn.slot_mapping)
                .arg(&(num_tokens as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(launch_cfg)
                .map_err(|e| LLMError::GpuError(format!("kv_cache_write: {e}")))?;
        }
        Ok(())
    }

    fn fused_rope_cache_write_single(
        &self,
        qkv: &mut CudaSlice<f16>,
        attn: &AttentionMeta<'_>,
    ) -> Result<()> {
        let cfg = &self.config;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let half_dim = head_dim / 2;
        let grid_y = num_heads.max(num_kv_heads) as u32;

        let kernel = self
            .loader
            .get_func("fused_rope_cache", "fused_rope_cache_f16_kernel")
            .map_err(|e| LLMError::GpuError(format!("fused_rope_cache kernel missing: {e}")))?;

        let (mut q_part, mut kv_rest) = qkv.split_at_mut(q_dim);
        let (mut k_part, v_rest) = kv_rest.split_at_mut(kv_dim);
        let v_view = v_rest.slice(..kv_dim);

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&mut q_part)
                .arg(&mut k_part)
                .arg(&v_view)
                .arg(attn.key_cache)
                .arg(attn.value_cache)
                .arg(attn.rope_cos)
                .arg(attn.rope_sin)
                .arg(&attn.positions)
                .arg(&attn.slot_mapping)
                .arg(&1i32)
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(LaunchConfig {
                    grid_dim: (1, grid_y, 1),
                    block_dim: (half_dim.min(1024) as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| LLMError::GpuError(format!("fused_rope_cache_write: {e}")))?;
        }
        Ok(())
    }

    // =================================================================
    // Step 5: Attention (Flash Attention)
    // =================================================================

    fn attention(
        &self,
        qkv: &CudaSlice<f16>,
        attn: &AttentionMeta<'_>,
        q_end: usize,
        num_tokens: usize,
        attn_out: &mut CudaSlice<f16>,
        split_out: &mut CudaSlice<f32>,
        split_max: &mut CudaSlice<f32>,
        split_sum: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let q_view = qkv.slice(..q_end);
        if attn.is_prefill {
            self.prefill_attention(&q_view, attn, num_tokens, attn_out)
        } else {
            self.decode_attention(
                &q_view, attn, num_tokens, attn_out, split_out, split_max, split_sum,
            )
        }
    }

    fn prefill_attention(
        &self,
        q: &CudaView<'_, f16>,
        attn: &AttentionMeta<'_>,
        num_tokens: usize,
        output: &mut CudaSlice<f16>,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let block_size = cfg.block_size;
        let num_seqs = attn.num_seqs;

        if num_seqs == 0 {
            return Err(LLMError::GpuError(
                "prefill_attention: num_seqs == 0".into(),
            ));
        }

        let kernel = self
            .loader
            .get_func(
                "flash_attention_3_prefill",
                "flash_attention_3_prefill_f16io_kernel",
            )
            .map_err(|e| LLMError::GpuError(format!("prefill FA3 kernel: {e}")))?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        const FA3_BC: usize = 64;
        const FA3_THREADS: u32 = 256;
        let smem = FA3_BC * head_dim * std::mem::size_of::<u16>()
            + (FA3_BC + 8) * std::mem::size_of::<f32>();
        let shared_mem_bytes = smem as u32;
        let max_blocks_per_seq = (DeviceSlice::len(&attn.block_tables) / num_seqs) as i32;

        if shared_mem_bytes > 49152 {
            kernel
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                )
                .map_err(|e| LLMError::GpuError(format!("prefill FA3 set smem: {e}")))?;
        }

        let launch_cfg = LaunchConfig {
            grid_dim: (num_seqs as u32, num_heads as u32, 1),
            block_dim: (FA3_THREADS, 1, 1),
            shared_mem_bytes,
        };

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(output)
                .arg(q)
                .arg(attn.key_cache)
                .arg(attn.value_cache)
                .arg(&attn.block_tables)
                .arg(&attn.context_lens)
                .arg(&attn.seq_start_pos)
                .arg(&scale)
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(block_size as i32))
                .arg(&(attn.max_context_len as i32))
                .arg(&max_blocks_per_seq)
                .arg(&(num_tokens as i32))
                .arg(&1i32)
                .launch(launch_cfg)
                .map_err(|e| LLMError::GpuError(format!("prefill FA3 launch: {e}")))?;
        }
        Ok(())
    }

    fn decode_attention(
        &self,
        q: &CudaView<'_, f16>,
        attn: &AttentionMeta<'_>,
        num_tokens: usize,
        output: &mut CudaSlice<f16>,
        split_out: &mut CudaSlice<f32>,
        split_max: &mut CudaSlice<f32>,
        split_sum: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let num_seqs = attn.num_seqs;

        let out_len = num_tokens * num_heads * head_dim;
        if output.len() < out_len {
            return Err(LLMError::GpuError(format!(
                "decode_attention output too small: have {}, need {}",
                output.len(),
                out_len
            )));
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let max_blocks = (attn.block_tables.len() / num_seqs.max(1)) as i32;
        let heads_per_group = if num_kv_heads > 0 {
            num_heads / num_kv_heads
        } else {
            1
        };

        const DECODE_TILE_TOKENS: usize = 64;
        let max_tiles =
            ((attn.max_context_len as usize) + DECODE_TILE_TOKENS - 1) / DECODE_TILE_TOKENS;
        let num_splits = choose_num_splits(attn.max_context_len as usize)
            .max(1)
            .min(max_tiles.max(1)) as i32;

        if num_heads != num_kv_heads && heads_per_group <= 8 && head_dim % 8 == 0 {
            return self.decode_attention_gqa_v3(
                q, attn, num_tokens, num_seqs, output, split_out, split_max, split_sum, scale,
                max_blocks, num_splits,
            );
        }

        self.decode_attention_standard(
            q, attn, num_tokens, num_seqs, output, split_out, split_max, split_sum, scale,
            max_blocks, num_splits,
        )
    }

    fn decode_attention_gqa_v3(
        &self,
        q: &CudaView<'_, f16>,
        attn: &AttentionMeta<'_>,
        num_tokens: usize,
        num_seqs: usize,
        output: &mut CudaSlice<f16>,
        split_out: &mut CudaSlice<f32>,
        split_max: &mut CudaSlice<f32>,
        split_sum: &mut CudaSlice<f32>,
        scale: f32,
        max_blocks: i32,
        num_splits: i32,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let block_size = cfg.block_size;

        let v3_gqa = self
            .loader
            .get_func("flash_attention_3_v3", "fa3_v3_decode_gqa_kernel")?;

        const V3_BC: usize = 64;
        const V3_THREADS: u32 = 256;
        const V3_MAX_HPG: usize = 8;
        const V3_SCORE_STRIDE: usize = V3_BC + 1;

        let smem = 2 * V3_BC * head_dim * std::mem::size_of::<u16>()
            + V3_MAX_HPG * V3_SCORE_STRIDE * std::mem::size_of::<f32>()
            + 8 * std::mem::size_of::<f32>();
        let shared_mem_bytes = smem as u32;

        if shared_mem_bytes > 49152 {
            v3_gqa
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                )
                .map_err(|e| LLMError::GpuError(format!("v3 GQA set smem: {e}")))?;
        }

        let p_max_context = attn.max_context_len as i32;

        if num_splits > 1 {
            let launch_cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_kv_heads as u32, num_splits as u32),
                block_dim: (V3_THREADS, 1, 1),
                shared_mem_bytes,
            };
            unsafe {
                self.stream
                    .launch_builder(&v3_gqa)
                    .arg(&mut *output)
                    .arg(&mut *split_out)
                    .arg(&mut *split_max)
                    .arg(&mut *split_sum)
                    .arg(q)
                    .arg(attn.key_cache)
                    .arg(attn.value_cache)
                    .arg(&attn.block_tables)
                    .arg(&attn.context_lens)
                    .arg(&scale)
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .arg(&(block_size as i32))
                    .arg(&p_max_context)
                    .arg(&max_blocks)
                    .arg(&num_splits)
                    .launch(launch_cfg)
                    .map_err(|e| LLMError::GpuError(format!("v3 GQA split-K launch: {e}")))?;
            }
            self.reduce_split_k(
                output, split_out, split_max, split_sum, &attn.context_lens,
                num_seqs, num_heads, head_dim, num_splits,
            )?;
        } else {
            let launch_cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_kv_heads as u32, 1),
                block_dim: (V3_THREADS, 1, 1),
                shared_mem_bytes,
            };
            unsafe {
                self.stream
                    .launch_builder(&v3_gqa)
                    .arg(output)
                    .arg(&self.attn_dummy)
                    .arg(&self.attn_dummy)
                    .arg(&self.attn_dummy)
                    .arg(q)
                    .arg(attn.key_cache)
                    .arg(attn.value_cache)
                    .arg(&attn.block_tables)
                    .arg(&attn.context_lens)
                    .arg(&scale)
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .arg(&(block_size as i32))
                    .arg(&p_max_context)
                    .arg(&max_blocks)
                    .arg(&1i32)
                    .launch(launch_cfg)
                    .map_err(|e| LLMError::GpuError(format!("v3 GQA single-pass launch: {e}")))?;
            }
        }
        Ok(())
    }

    fn decode_attention_standard(
        &self,
        q: &CudaView<'_, f16>,
        attn: &AttentionMeta<'_>,
        num_tokens: usize,
        num_seqs: usize,
        output: &mut CudaSlice<f16>,
        split_out: &mut CudaSlice<f32>,
        split_max: &mut CudaSlice<f32>,
        split_sum: &mut CudaSlice<f32>,
        scale: f32,
        max_blocks: i32,
        num_splits: i32,
    ) -> Result<()> {
        let cfg = &self.config;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let block_size = cfg.block_size;

        let kernel = self
            .loader
            .get_func("flash_attention_3_v3", "fa3_v3_decode_kernel")?;

        const V3_BC: usize = 64;
        const V3_THREADS: u32 = 256;
        let smem = 2 * V3_BC * head_dim * std::mem::size_of::<u16>()
            + (V3_BC + 8) * std::mem::size_of::<f32>();
        let shared_mem_bytes = smem as u32;
        let p_max_context = attn.max_context_len as i32;

        if shared_mem_bytes > 49152 {
            kernel
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                )
                .map_err(|e| LLMError::GpuError(format!("v3 decode set smem: {e}")))?;
        }

        if num_splits > 1 {
            let launch_cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, num_splits as u32),
                block_dim: (V3_THREADS, 1, 1),
                shared_mem_bytes,
            };
            unsafe {
                self.stream
                    .launch_builder(&kernel)
                    .arg(&mut *output)
                    .arg(&mut *split_out)
                    .arg(&mut *split_max)
                    .arg(&mut *split_sum)
                    .arg(q)
                    .arg(attn.key_cache)
                    .arg(attn.value_cache)
                    .arg(&attn.block_tables)
                    .arg(&attn.context_lens)
                    .arg(&scale)
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .arg(&(block_size as i32))
                    .arg(&p_max_context)
                    .arg(&max_blocks)
                    .arg(&num_splits)
                    .launch(launch_cfg)
                    .map_err(|e| LLMError::GpuError(format!("v3 decode split-K launch: {e}")))?;
            }
            self.reduce_split_k(
                output, split_out, split_max, split_sum, &attn.context_lens,
                num_seqs, num_heads, head_dim, num_splits,
            )?;
        } else {
            let launch_cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (V3_THREADS, 1, 1),
                shared_mem_bytes,
            };
            unsafe {
                self.stream
                    .launch_builder(&kernel)
                    .arg(output)
                    .arg(&self.attn_dummy)
                    .arg(&self.attn_dummy)
                    .arg(&self.attn_dummy)
                    .arg(q)
                    .arg(attn.key_cache)
                    .arg(attn.value_cache)
                    .arg(&attn.block_tables)
                    .arg(&attn.context_lens)
                    .arg(&scale)
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .arg(&(block_size as i32))
                    .arg(&p_max_context)
                    .arg(&max_blocks)
                    .arg(&1i32)
                    .launch(launch_cfg)
                    .map_err(|e| LLMError::GpuError(format!("v3 decode single launch: {e}")))?;
            }
        }
        Ok(())
    }

    fn reduce_split_k(
        &self,
        output: &mut CudaSlice<f16>,
        split_out: &CudaSlice<f32>,
        split_max: &CudaSlice<f32>,
        split_sum: &CudaSlice<f32>,
        context_lens: &CudaView<'_, i32>,
        num_seqs: usize,
        num_heads: usize,
        head_dim: usize,
        num_splits: i32,
    ) -> Result<()> {
        let kernel = self
            .loader
            .get_func("flash_attention_3_v3", "fa3_v3_combine_f16_kernel")?;
        let launch_cfg = LaunchConfig {
            grid_dim: (num_seqs as u32, num_heads as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let p_num_seqs = num_seqs as i32;
        let p_num_heads = num_heads as i32;
        let p_head_dim = head_dim as i32;
        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(output)
                .arg(split_out)
                .arg(split_max)
                .arg(split_sum)
                .arg(context_lens)
                .arg(&p_num_seqs)
                .arg(&p_num_heads)
                .arg(&p_head_dim)
                .arg(&num_splits)
                .launch(launch_cfg)
                .map_err(|e| LLMError::GpuError(format!("split-K reduce: {e}")))?;
        }
        Ok(())
    }

    // =================================================================
    // Steps 6+7: O-projection + residual + post-attention RMSNorm
    // =================================================================

    fn oproj_residual_postnorm_dispatch(
        &self,
        hidden: &CudaSlice<f16>,
        residual_from_fused: bool,
        weights: &LayerWeights<'_>,
        _attn: &AttentionMeta<'_>,
        scratch: &mut F16LayerScratch,
        residual_write: &mut CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        q_dim: usize,
        _gemm_strategy: GemmStrategy,
        _cutlass: Option<&CutlassKernels>,
        cublas: &CublasHandle,
        lt_ops: Option<&CublasLtOps>,
    ) -> Result<()> {
        hgemm_dispatch(
            lt_ops, cublas,
            num_tokens, hidden_size, q_dim,
            1.0, &scratch.attn_out, weights.o_proj_weight, 0.0,
            &mut scratch.o_proj_out,
        )?;

        let residual_src: &CudaSlice<f16> = if residual_from_fused {
            &scratch.residual_tmp
        } else {
            hidden
        };
        self.fused_residual_rmsnorm(
            residual_src,
            &scratch.o_proj_out,
            weights.post_attention_layernorm_weight,
            num_tokens,
            hidden_size,
            &mut scratch.normed,
            residual_write,
        )?;
        Ok(())
    }

    // =================================================================
    // Step 8: GateUp + SiLU
    // =================================================================

    fn gateup_silu(
        &self,
        normed: &CudaSlice<f16>,
        gate_up_weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        gate_up_out: &mut CudaSlice<f16>,
        silu_out: &mut CudaSlice<f16>,
        _gateup_workspace: &mut CudaSlice<u8>,
        _gemm_strategy: GemmStrategy,
        _cutlass: Option<&CutlassKernels>,
        cublas: &CublasHandle,
        lt_ops: Option<&CublasLtOps>,
    ) -> Result<()> {
        let gate_up_dim = intermediate_size * 2;

        hgemm_dispatch(
            lt_ops, cublas,
            num_tokens, gate_up_dim, hidden_size,
            1.0, normed, gate_up_weight, 0.0, gate_up_out,
        )?;

        let silu_fn = self
            .loader
            .get_func("silu_mul_interleaved", "silu_mul_interleaved_f16_kernel")
            .map_err(|e| {
                LLMError::GpuError(format!("silu_mul_interleaved kernel missing: {e}"))
            })?;

        let total = (num_tokens * intermediate_size) as u32;
        unsafe {
            self.stream
                .launch_builder(&silu_fn)
                .arg(silu_out)
                .arg(gate_up_out)
                .arg(&(num_tokens as i32))
                .arg(&(intermediate_size as i32))
                .launch(LaunchConfig {
                    grid_dim: ((total + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| LLMError::GpuError(format!("silu_mul_interleaved: {e}")))?;
        }
        Ok(())
    }

    // =================================================================
    // Step 9: Down projection
    // =================================================================

    fn down_projection(
        &self,
        silu_out: &CudaSlice<f16>,
        down_proj_weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        down_out: &mut CudaSlice<f16>,
        cublas: &CublasHandle,
        lt_ops: Option<&CublasLtOps>,
    ) -> Result<()> {
        hgemm_dispatch(lt_ops, cublas, num_tokens, hidden_size, intermediate_size, 1.0, silu_out, down_proj_weight, 0.0, down_out)
    }
}
