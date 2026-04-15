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
use rvllm_gpu::cutlass_autotune::CutlassAutotuneCache;
use rvllm_gpu::cutlass_ffi::CutlassKernels;
use rvllm_gpu::fa3_ffi::Fa3Kernels;
use rvllm_gpu::kernel_loader::KernelLoader;

// ===================================================================
// cuBLASLt-first GEMM dispatch: autotuned cuBLASLt -> cuBLAS
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
// CUTLASS FP8 GEMM dispatch: pre-quantized activation + CUTLASS SM90
// ===================================================================

fn cutlass_fp8_gemm_dispatch(
    cutlass: &CutlassKernels,
    autotune: Option<&CutlassAutotuneCache>,
    stream: &CudaStream,
    m: usize, n: usize, k: usize,
    act_fp8: &CudaSlice<u8>,
    act_scales: &CudaSlice<f32>,
    weight_fp8: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    output_f16: &mut CudaSlice<f16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    let stream_ptr = stream.cu_stream() as u64;
    let (act_ptr, _) = act_fp8.device_ptr(stream);
    let (as_ptr, _) = act_scales.device_ptr(stream);
    let (w_ptr, _) = weight_fp8.device_ptr(stream);
    let (ws_ptr, _) = weight_scale.device_ptr(stream);
    let (out_ptr, _) = output_f16.device_ptr_mut(stream);
    let (wk_ptr, _) = workspace.device_ptr_mut(stream);

    if let Some(at) = autotune {
        if let Some(variant) = at.best_fp8_gemm(m, n, k) {
            return cutlass.fp8_gemm_variant(
                variant,
                out_ptr as u64, act_ptr as u64, w_ptr as u64,
                as_ptr as u64, ws_ptr as u64,
                m as i32, n as i32, k as i32,
                wk_ptr as u64, workspace.len(),
                stream_ptr,
            ).map_err(|e| LLMError::GpuError(e));
        }
    }

    // Small tile (64x128x128) for decode: better tile utilization when M <= 64
    if m <= 64 && cutlass.has_fp8_gemm_small() {
        return cutlass.fp8_gemm_small(
            out_ptr as u64, act_ptr as u64, w_ptr as u64,
            as_ptr as u64, ws_ptr as u64,
            m as i32, n as i32, k as i32,
            wk_ptr as u64, workspace.len(),
            stream_ptr,
        ).map_err(|e| LLMError::GpuError(e));
    }

    // Default CUTLASS FP8 kernel
    cutlass.fp8_gemm(
        out_ptr as u64, act_ptr as u64, w_ptr as u64,
        as_ptr as u64, ws_ptr as u64,
        m as i32, n as i32, k as i32,
        wk_ptr as u64, workspace.len(),
        stream_ptr,
    ).map_err(|e| LLMError::GpuError(e))
}

/// FP8 GEMM + residual add (fused in CUTLASS epilogue).
/// D = cast<f16>(a_scales * b_scale * GEMM(act_fp8, weight_fp8) + residual)
/// Eliminates the intermediate o_proj_out / down_out buffer.
fn cutlass_fp8_gemm_residual_dispatch(
    cutlass: &CutlassKernels,
    stream: &CudaStream,
    m: usize, n: usize, k: usize,
    act_fp8: &CudaSlice<u8>,
    act_scales: &CudaSlice<f32>,
    weight_fp8: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    residual: &CudaSlice<f16>,
    output_f16: &mut CudaSlice<f16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    let stream_ptr = stream.cu_stream() as u64;
    let (act_ptr, _) = act_fp8.device_ptr(stream);
    let (as_ptr, _) = act_scales.device_ptr(stream);
    let (w_ptr, _) = weight_fp8.device_ptr(stream);
    let (ws_ptr, _) = weight_scale.device_ptr(stream);
    let (res_ptr, _) = residual.device_ptr(stream);
    let (out_ptr, _) = output_f16.device_ptr_mut(stream);
    let (wk_ptr, _) = workspace.device_ptr_mut(stream);

    // Variant 0 (64x128x128 WS) is verified neutral for O-proj [M, 3584, 3584].
    // FP8FastAccum variants (v4+) regress with the residual EVT epilogue.
    let variant = if m <= 64 { 0 } else { 1 };

    cutlass.fp8_gemm_residual(
        variant,
        out_ptr as u64, act_ptr as u64, w_ptr as u64,
        as_ptr as u64, ws_ptr as u64,
        res_ptr as u64,
        m as i32, n as i32, k as i32,
        wk_ptr as u64, workspace.len(),
        stream_ptr,
    ).map_err(|e| LLMError::GpuError(e))
}

// ===================================================================
// Autotuned CUTLASS SM90 WGMMA dispatch for F16
// On SM90: always use CUTLASS WGMMA (cuBLAS may pick SM80 mma.sync).
// No CUTLASS loaded: cuBLASLt/cuBLAS path (non-SM90 builds).
// ===================================================================

fn f16_gemm_autotuned(
    cutlass: Option<&CutlassKernels>,
    autotune: Option<&CutlassAutotuneCache>,
    lt_ops: Option<&CublasLtOps>,
    cublas: &CublasHandle,
    stream: &CudaStream,
    m: usize, n: usize, k: usize,
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    if let (Some(ck), Some(at)) = (cutlass, autotune) {
        let variant = match at.best_hgemm(m, n, k) {
            Some(v) => v,
            None => {
                // No CUTLASS entry -- cuBLAS wins for this shape (bandwidth-bound)
                return hgemm_dispatch(lt_ops, cublas, m, n, k, 1.0, input, weight, 0.0, output);
            }
        };
        let stream_ptr = stream.cu_stream() as u64;
        let ws_len = workspace.len();
        let (in_ptr, _g1) = input.device_ptr(stream);
        let (w_ptr, _g2) = weight.device_ptr(stream);
        let (out_ptr, _g3) = output.device_ptr_mut(stream);
        let (wk_ptr, _g4) = workspace.device_ptr_mut(stream);
        return ck.hgemm_variant(
            variant,
            out_ptr as u64, in_ptr as u64, w_ptr as u64,
            m as i32, n as i32, k as i32,
            wk_ptr as u64, ws_len, stream_ptr,
        ).map_err(|e| LLMError::GpuError(e));
    }
    // No CUTLASS loaded -- cuBLAS path (non-SM90 build)
    hgemm_dispatch(lt_ops, cublas, m, n, k, 1.0, input, weight, 0.0, output)
}

// ===================================================================
// Fused O-proj GEMM + residual: output = GEMM(input, weight) + residual
// ===================================================================

fn f16_oproj_residual_autotuned(
    cutlass: &CutlassKernels,
    stream: &CudaStream,
    variant: usize,
    m: usize, n: usize, k: usize,
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    residual: &CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    let stream_ptr = stream.cu_stream() as u64;
    let ws_len = workspace.len();
    let (in_ptr, _g1) = input.device_ptr(stream);
    let (w_ptr, _g2) = weight.device_ptr(stream);
    let (r_ptr, _g3) = residual.device_ptr(stream);
    let (out_ptr, _g4) = output.device_ptr_mut(stream);
    let (wk_ptr, _g5) = workspace.device_ptr_mut(stream);
    cutlass.oproj_residual_variant(
        variant,
        out_ptr as u64, in_ptr as u64, w_ptr as u64, r_ptr as u64,
        m as i32, n as i32, k as i32,
        wk_ptr as u64, ws_len, stream_ptr,
    ).map_err(|e| LLMError::GpuError(e))
}

// ===================================================================
// Fused GateUp GEMM + SiLU: output[M, N/2] = SiLU(GEMM(input, weight))
// ===================================================================

fn f16_gateup_silu_autotuned(
    cutlass: &CutlassKernels,
    stream: &CudaStream,
    variant: usize,
    m: usize, n: usize, k: usize,
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    let stream_ptr = stream.cu_stream() as u64;
    let ws_len = workspace.len();
    let (in_ptr, _g1) = input.device_ptr(stream);
    let (w_ptr, _g2) = weight.device_ptr(stream);
    let (out_ptr, _g3) = output.device_ptr_mut(stream);
    let (wk_ptr, _g4) = workspace.device_ptr_mut(stream);
    cutlass.gateup_silu_variant(
        variant,
        out_ptr as u64, in_ptr as u64, w_ptr as u64,
        m as i32, n as i32, k as i32,
        wk_ptr as u64, ws_len, stream_ptr,
    ).map_err(|e| LLMError::GpuError(e))
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
// LayerPlan -- pre-computed dispatch FSM, built once at init
// ===================================================================

#[derive(Debug, Clone, Copy)]
pub enum GemmOp {
    CuBLAS,
    CutlassHgemm(usize),
}

#[derive(Debug, Clone, Copy)]
pub enum OprojOp {
    CuBLAS,
    Fused(usize),  // oproj_residual variant -- output = GEMM + residual
}

#[derive(Debug, Clone, Copy)]
pub enum GateUpOp {
    CuBLAS,
    FusedSilu(usize),  // gateup_silu variant -- output = SiLU(GEMM)
}

#[derive(Debug, Clone, Copy)]
pub enum DownOp {
    CuBLAS,
    FusedResidual(usize),  // oproj_residual variant -- output = GEMM + residual
}

#[derive(Debug, Clone, Copy)]
pub enum Fp8Path {
    CutlassFp8,
    CuBLASLtFp8,
}

#[derive(Debug, Clone, Copy)]
pub enum LayerPlan {
    Fp8 { path: Fp8Path },
    F16 {
        qkv: GemmOp,
        oproj: OprojOp,
        gateup: GateUpOp,
        down: DownOp,
    },
}

impl LayerPlan {
    /// Is down_proj fused with residual? (next layer skips residual add in input norm)
    pub fn down_fused(&self) -> bool {
        matches!(self, LayerPlan::F16 { down: DownOp::FusedResidual(_), .. })
    }

    /// Build optimal dispatch plan from autotune cache and model config.
    /// Called once at model load. No runtime fallbacks -- every step is locked in.
    pub fn build(
        cfg: &LayerConfig,
        max_tokens: usize,
        has_fp8: bool,
        has_cutlass: bool,
        autotune: Option<&CutlassAutotuneCache>,
    ) -> Self {
        // FP8 weights -> FP8 path (CUTLASS FP8 if available, else cuBLASLt)
        if has_fp8 {
            return LayerPlan::Fp8 {
                path: if has_cutlass { Fp8Path::CutlassFp8 } else { Fp8Path::CuBLASLtFp8 },
            };
        }

        let h = cfg.hidden_size;
        let qkv = cfg.qkv_dim();
        let q = cfg.q_dim();
        let inter = cfg.intermediate_size;
        let gate_up = cfg.gate_up_dim();

        // Use representative M for dispatch decision (decode = small M)
        // Autotune nearest-M lookup handles prefill automatically
        let m = max_tokens.min(32);

        let at = autotune;

        // QKV: check if CUTLASS hgemm has an entry (hybrid policy: only stored when CUTLASS wins)
        let qkv_op = match at.and_then(|a| a.best_hgemm(m, qkv, h)) {
            Some(v) if has_cutlass => GemmOp::CutlassHgemm(v),
            _ => GemmOp::CuBLAS,
        };

        // O-proj: check oproj_residual entry (only stored when CUTLASS beats cuBLAS)
        let oproj_op = match at.and_then(|a| a.best_oproj_residual(m, h, q)) {
            Some(v) if has_cutlass => OprojOp::Fused(v),
            _ => OprojOp::CuBLAS,
        };

        // GateUp+SiLU: check gateup_silu entry
        let gateup_op = match at.and_then(|a| a.best_gateup_silu(m, gate_up, h)) {
            Some(v) if has_cutlass => GateUpOp::FusedSilu(v),
            _ => GateUpOp::CuBLAS,
        };

        // Down proj: check oproj_residual entry for down shape (only if CUTLASS wins)
        let down_op = match at.and_then(|a| a.best_oproj_residual(m, h, inter)) {
            Some(v) if has_cutlass => DownOp::FusedResidual(v),
            _ => DownOp::CuBLAS,
        };

        LayerPlan::F16 {
            qkv: qkv_op,
            oproj: oproj_op,
            gateup: gateup_op,
            down: down_op,
        }
    }
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
    // CUTLASS workspace (shared by FP8 and F16 SM90 GEMM paths)
    pub cutlass_workspace: CudaSlice<u8>,
}

impl F16LayerScratch {
    pub fn alloc(
        stream: &Arc<CudaStream>,
        config: &LayerConfig,
        max_batch_tokens: usize,
        cutlass_workspace_bytes: usize,
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
            fp8_act_scale: alloc_f32(t)?,  // per-token scales for CUTLASS FP8 path
            fp8_absmax: alloc_f32(1)?,
            cutlass_workspace: alloc_u8(cutlass_workspace_bytes)?,
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
    // Fused norm+quant and activation+quant helpers for CUTLASS FP8
    // =================================================================

    fn fused_rmsnorm_fp8_quant(
        &self,
        input: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        output_fp8: &mut CudaSlice<u8>,
        output_scales: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let kernel = self.loader.get_func("fused_rmsnorm_fp8_quant", "fused_rmsnorm_fp8_quant_kernel")?;
        let block_threads = hidden_size.min(1024) as u32;
        unsafe {
            self.stream.launch_builder(&kernel)
                .arg(output_fp8)
                .arg(output_scales)
                .arg(input)
                .arg(weight)
                .arg(&self.config.rms_norm_eps)
                .arg(&(hidden_size as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_tokens as u32, 1, 1),
                    block_dim: (block_threads, 1, 1),
                    shared_mem_bytes: (block_threads as usize * std::mem::size_of::<f32>()) as u32,
                })
                .map_err(|e| LLMError::GpuError(format!("fused_rmsnorm_fp8_quant: {e}")))?;
        }
        Ok(())
    }

    fn fused_add_rmsnorm_fp8_quant(
        &self,
        input: &CudaSlice<f16>,
        residual: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        num_tokens: usize,
        hidden_size: usize,
        output_fp8: &mut CudaSlice<u8>,
        output_scales: &mut CudaSlice<f32>,
        residual_out: &mut CudaSlice<f16>,
    ) -> Result<()> {
        let kernel = self.loader.get_func("fused_rmsnorm_fp8_quant", "fused_add_rmsnorm_fp8_quant_kernel")?;
        let block_threads = hidden_size.min(1024) as u32;
        unsafe {
            self.stream.launch_builder(&kernel)
                .arg(output_fp8)
                .arg(output_scales)
                .arg(residual_out)
                .arg(input)
                .arg(residual)
                .arg(weight)
                .arg(&self.config.rms_norm_eps)
                .arg(&(hidden_size as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_tokens as u32, 1, 1),
                    block_dim: (block_threads, 1, 1),
                    shared_mem_bytes: (block_threads as usize * std::mem::size_of::<f32>()) as u32,
                })
                .map_err(|e| LLMError::GpuError(format!("fused_add_rmsnorm_fp8_quant: {e}")))?;
        }
        Ok(())
    }

    fn quantize_fp8_per_token(
        &self,
        input: &CudaSlice<f16>,
        num_tokens: usize,
        dim: usize,
        output_fp8: &mut CudaSlice<u8>,
        output_scales: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let kernel = self.loader.get_func("fused_rmsnorm_fp8_quant", "quantize_fp8_per_token_kernel")?;
        let block_threads = dim.min(1024) as u32;
        unsafe {
            self.stream.launch_builder(&kernel)
                .arg(output_fp8)
                .arg(output_scales)
                .arg(input)
                .arg(&(dim as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_tokens as u32, 1, 1),
                    block_dim: (block_threads, 1, 1),
                    shared_mem_bytes: (block_threads as usize * std::mem::size_of::<f32>()) as u32,
                })
                .map_err(|e| LLMError::GpuError(format!("quantize_fp8_per_token: {e}")))?;
        }
        Ok(())
    }

    fn fused_silu_mul_fp8_quant(
        &self,
        gate_up: &CudaSlice<f16>,
        num_tokens: usize,
        intermediate_size: usize,
        output_fp8: &mut CudaSlice<u8>,
        output_scales: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let kernel = self.loader.get_func("fused_silu_fp8_quant", "fused_silu_mul_fp8_quant_kernel")?;
        let block_threads = intermediate_size.min(1024) as u32;
        unsafe {
            self.stream.launch_builder(&kernel)
                .arg(output_fp8)
                .arg(output_scales)
                .arg(gate_up)
                .arg(&(intermediate_size as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_tokens as u32, 1, 1),
                    block_dim: (block_threads, 1, 1),
                    shared_mem_bytes: (block_threads as usize * std::mem::size_of::<f32>()) as u32,
                })
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_fp8_quant: {e}")))?;
        }
        Ok(())
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
        fa3: Option<&Fa3Kernels>,
        autotune: Option<&CutlassAutotuneCache>,
        cublas: &CublasHandle,
        lt_ops: Option<&CublasLtOps>,
    ) -> Result<bool> {
        let cfg = &self.config;
        let num_tokens = attn.num_tokens;
        let hidden_size = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let qkv_dim = cfg.qkv_dim();
        let intermediate = cfg.intermediate_size;
        let q_end = num_tokens * q_dim;
        let k_end = q_end + num_tokens * kv_dim;

        // Step 1: Input layernorm + QKV GEMM
        // CUTLASS FP8 path: fused norm+quant -> CUTLASS FP8 GEMM (2 kernels)
        // Old FP8 path: norm -> 2-pass quantize -> cuBLASLt FP8 GEMM (4 kernels)
        // F16 path: norm -> hgemm
        let residual_from_fused;
        if let (Some(ck), Some(fp8w)) = (cutlass, &weights.fp8) {
            // CUTLASS FP8: fused norm+quant, then CUTLASS GEMM
            if let Some(prev_mlp) = prev_mlp_out {
                self.fused_add_rmsnorm_fp8_quant(
                    hidden, prev_mlp, weights.input_layernorm_weight,
                    num_tokens, hidden_size,
                    &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                    &mut scratch.residual_tmp,
                )?;
                residual_from_fused = true;
            } else {
                self.fused_rmsnorm_fp8_quant(
                    hidden, weights.input_layernorm_weight,
                    num_tokens, hidden_size,
                    &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                )?;
                residual_from_fused = false;
            }
            cutlass_fp8_gemm_dispatch(
                ck, autotune, &self.stream, num_tokens, qkv_dim, hidden_size,
                &scratch.fp8_act_scratch, &scratch.fp8_act_scale,
                fp8w.qkv_fp8, fp8w.qkv_scale,
                &mut scratch.qkv_buf, &mut scratch.cutlass_workspace,
            )?;
        } else if let (Some(lt), Some(fp8w)) = (lt_ops, &weights.fp8) {
            // Old cuBLASLt FP8 fallback: separate norm, then 2-pass quantize + cuBLASLt
            residual_from_fused = if let Some(prev_mlp) = prev_mlp_out {
                self.fused_residual_rmsnorm(
                    hidden, prev_mlp, weights.input_layernorm_weight,
                    num_tokens, hidden_size,
                    &mut scratch.normed, &mut scratch.residual_tmp,
                )?;
                true
            } else {
                self.rms_norm(
                    hidden, weights.input_layernorm_weight,
                    num_tokens, hidden_size, &mut scratch.normed,
                )?;
                false
            };
            fp8_gemm_dispatch(
                lt, &self.loader, &self.stream,
                num_tokens, qkv_dim, hidden_size,
                &scratch.normed, fp8w.qkv_fp8, fp8w.qkv_scale,
                &mut scratch.qkv_buf,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                &mut scratch.fp8_absmax,
            )?;
        } else {
            // F16 path
            residual_from_fused = if let Some(prev_mlp) = prev_mlp_out {
                self.fused_residual_rmsnorm(
                    hidden, prev_mlp, weights.input_layernorm_weight,
                    num_tokens, hidden_size,
                    &mut scratch.normed, &mut scratch.residual_tmp,
                )?;
                true
            } else {
                self.rms_norm(
                    hidden, weights.input_layernorm_weight,
                    num_tokens, hidden_size, &mut scratch.normed,
                )?;
                false
            };
            f16_gemm_autotuned(cutlass, autotune, lt_ops, cublas, &self.stream,
                num_tokens, qkv_dim, hidden_size, &scratch.normed, weights.qkv_weight,
                &mut scratch.qkv_buf, &mut scratch.cutlass_workspace)?;
        }

        if !attn.is_prefill && num_tokens > 1 {
            // All-decode batch: fused RoPE + KV cache write (1 kernel instead of 2 per layer)
            self.fused_rope_cache_write_batch(&mut scratch.qkv_buf, attn, num_tokens)?;
        } else {
            self.apply_rope(&mut scratch.qkv_buf, attn, q_dim, kv_dim, q_end, num_tokens)?;
            self.kv_cache_write(&mut scratch.qkv_buf, attn, q_end, k_end, kv_dim, num_tokens)?;
        }

        self.attention(
            &scratch.qkv_buf,
            attn,
            q_end,
            num_tokens,
            &mut scratch.attn_out,
            &mut scratch.attn_split_out,
            &mut scratch.attn_split_max,
            &mut scratch.attn_split_sum,
            fa3,
        )?;

        // Compute residual source for post-attention residual connection
        let residual_src: &CudaSlice<f16> = if residual_from_fused {
            &scratch.residual_tmp
        } else {
            hidden
        };

        let mut used_fused_oproj = false;
        let mut used_fused_gateup = false;

        // O-proj GEMM (CUTLASS FP8 / cuBLASLt FP8 / F16)
        if let (Some(ck), Some(fp8w)) = (cutlass, &weights.fp8) {
            self.quantize_fp8_per_token(
                &scratch.attn_out, num_tokens, q_dim,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
            )?;
            // Try fused FP8 GEMM + residual add (eliminates o_proj_out buffer)
            if ck.fp8_gemm_residual_variant_count() > 0 {
                cutlass_fp8_gemm_residual_dispatch(
                    ck, &self.stream, num_tokens, hidden_size, q_dim,
                    &scratch.fp8_act_scratch, &scratch.fp8_act_scale,
                    fp8w.o_proj_fp8, fp8w.o_proj_scale,
                    residual_src, residual_write,
                    &mut scratch.cutlass_workspace,
                )?;
                used_fused_oproj = true;
            } else {
                cutlass_fp8_gemm_dispatch(
                    ck, autotune, &self.stream, num_tokens, hidden_size, q_dim,
                    &scratch.fp8_act_scratch, &scratch.fp8_act_scale,
                    fp8w.o_proj_fp8, fp8w.o_proj_scale,
                    &mut scratch.o_proj_out, &mut scratch.cutlass_workspace,
                )?;
            }
        } else if let (Some(lt), Some(fp8w)) = (lt_ops, &weights.fp8) {
            fp8_gemm_dispatch(
                lt, &self.loader, &self.stream,
                num_tokens, hidden_size, q_dim,
                &scratch.attn_out, fp8w.o_proj_fp8, fp8w.o_proj_scale,
                &mut scratch.o_proj_out,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                &mut scratch.fp8_absmax,
            )?;
        } else {
            // F16: try fused oproj+residual (GEMM + residual in one kernel)
            if let (Some(ck), Some(at)) = (cutlass, autotune) {
                if let Some(variant) = at.best_oproj_residual(num_tokens, hidden_size, q_dim) {
                    f16_oproj_residual_autotuned(
                        ck, &self.stream, variant,
                        num_tokens, hidden_size, q_dim,
                        &scratch.attn_out, weights.o_proj_weight,
                        residual_src, residual_write,
                        &mut scratch.cutlass_workspace,
                    )?;
                    used_fused_oproj = true;
                }
            }
            if !used_fused_oproj {
                f16_gemm_autotuned(cutlass, autotune, lt_ops, cublas, &self.stream,
                    num_tokens, hidden_size, q_dim, &scratch.attn_out, weights.o_proj_weight,
                    &mut scratch.o_proj_out, &mut scratch.cutlass_workspace)?;
            }
        }

        // Residual add + post-attention layernorm + GateUp GEMM
        let gate_up_dim = intermediate * 2;
        if let (Some(ck), Some(fp8w)) = (cutlass, &weights.fp8) {
            if used_fused_oproj {
                // Fused O-proj already wrote GEMM+residual to residual_write; just norm+quant
                self.fused_rmsnorm_fp8_quant(
                    &*residual_write,
                    weights.post_attention_layernorm_weight,
                    num_tokens, hidden_size,
                    &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                )?;
            } else {
                // Separate O-proj: need residual add + norm + quant
                self.fused_add_rmsnorm_fp8_quant(
                    residual_src, &scratch.o_proj_out,
                    weights.post_attention_layernorm_weight,
                    num_tokens, hidden_size,
                    &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                    residual_write,
                )?;
            }
            cutlass_fp8_gemm_dispatch(
                ck, autotune, &self.stream, num_tokens, gate_up_dim, hidden_size,
                &scratch.fp8_act_scratch, &scratch.fp8_act_scale,
                fp8w.gate_up_fp8, fp8w.gate_up_scale,
                &mut scratch.gate_up_out, &mut scratch.cutlass_workspace,
            )?;
        } else if let (Some(lt), Some(fp8w)) = (lt_ops, &weights.fp8) {
            // Old cuBLASLt FP8 fallback
            self.fused_residual_rmsnorm(
                residual_src, &scratch.o_proj_out,
                weights.post_attention_layernorm_weight,
                num_tokens, hidden_size,
                &mut scratch.normed, residual_write,
            )?;
            fp8_gemm_dispatch(
                lt, &self.loader, &self.stream,
                num_tokens, gate_up_dim, hidden_size,
                &scratch.normed, fp8w.gate_up_fp8, fp8w.gate_up_scale,
                &mut scratch.gate_up_out,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
                &mut scratch.fp8_absmax,
            )?;
        } else {
            // F16 path: norm depends on whether fused oproj already wrote residual
            if used_fused_oproj {
                // Fused oproj already wrote GEMM+residual to residual_write, just norm
                self.rms_norm(
                    &*residual_write, weights.post_attention_layernorm_weight,
                    num_tokens, hidden_size, &mut scratch.normed,
                )?;
            } else {
                self.fused_residual_rmsnorm(
                    residual_src, &scratch.o_proj_out,
                    weights.post_attention_layernorm_weight,
                    num_tokens, hidden_size,
                    &mut scratch.normed, residual_write,
                )?;
            }
            // Try fused gateup+silu (GEMM -> workspace temp -> SiLU -> silu_out)
            if let (Some(ck), Some(at)) = (cutlass, autotune) {
                if let Some(variant) = at.best_gateup_silu(num_tokens, gate_up_dim, hidden_size) {
                    f16_gateup_silu_autotuned(
                        ck, &self.stream, variant,
                        num_tokens, gate_up_dim, hidden_size,
                        &scratch.normed, weights.gate_up_weight,
                        &mut scratch.silu_out,
                        &mut scratch.gateup_workspace,
                    )?;
                    used_fused_gateup = true;
                }
            }
            if !used_fused_gateup {
                f16_gemm_autotuned(cutlass, autotune, lt_ops, cublas, &self.stream,
                    num_tokens, gate_up_dim, hidden_size, &scratch.normed, weights.gate_up_weight,
                    &mut scratch.gate_up_out, &mut scratch.cutlass_workspace)?;
            }
        }

        // SiLU activation + Down projection GEMM
        let mut used_fused_downproj = false;
        if let (Some(ck), Some(fp8w)) = (cutlass, &weights.fp8) {
            // CUTLASS FP8: fused silu+quant -> CUTLASS GEMM (skips separate silu kernel)
            self.fused_silu_mul_fp8_quant(
                &scratch.gate_up_out, num_tokens, intermediate,
                &mut scratch.fp8_act_scratch, &mut scratch.fp8_act_scale,
            )?;
            // Down-proj uses autotuned non-residual FP8 GEMM (residual kernel lacks
            // autotune for K=intermediate shapes and regresses ~24% vs autotuned path)
            cutlass_fp8_gemm_dispatch(
                ck, autotune, &self.stream, num_tokens, hidden_size, intermediate,
                &scratch.fp8_act_scratch, &scratch.fp8_act_scale,
                fp8w.down_proj_fp8, fp8w.down_proj_scale,
                down_write, &mut scratch.cutlass_workspace,
            )?;
        } else {
            // SiLU activation (skip if fused gateup+silu already wrote to silu_out)
            if !used_fused_gateup {
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
            }

            // Down projection GEMM (old FP8 or F16)
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
                // F16: try fused down_proj + residual (GEMM + residual in epilogue)
                // Uses same oproj_residual CUTLASS kernel: D = GEMM(A,B) + C
                // down_write = GEMM(silu_out, down_weight) + residual_write
                // Saves one HBM round-trip: next layer reads only down_write (full residual)
                if let (Some(ck), Some(at)) = (cutlass, autotune) {
                    if let Some(variant) = at.best_oproj_residual(num_tokens, hidden_size, intermediate) {
                        f16_oproj_residual_autotuned(
                            ck, &self.stream, variant,
                            num_tokens, hidden_size, intermediate,
                            &scratch.silu_out, weights.down_proj_weight,
                            &*residual_write, down_write,
                            &mut scratch.cutlass_workspace,
                        )?;
                        used_fused_downproj = true;
                    }
                }
                if !used_fused_downproj {
                    f16_gemm_autotuned(cutlass, autotune, lt_ops, cublas, &self.stream,
                        num_tokens, hidden_size, intermediate, &scratch.silu_out, weights.down_proj_weight,
                        &mut *down_write, &mut scratch.cutlass_workspace)?;
                }
            }
        }

        Ok(used_fused_downproj)
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

    /// Fused RoPE + KV cache write for batched decode (num_tokens > 1, all-decode).
    /// Same kernel as single-token fused path but with grid.x = num_tokens.
    fn fused_rope_cache_write_batch(
        &self,
        qkv: &mut CudaSlice<f16>,
        attn: &AttentionMeta<'_>,
        num_tokens: usize,
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
            .map_err(|e| LLMError::GpuError(format!("fused_rope_cache kernel: {e}")))?;

        let q_end = num_tokens * q_dim;
        let (mut q_part, mut kv_rest) = qkv.split_at_mut(q_end);
        let (mut k_part, v_rest) = kv_rest.split_at_mut(num_tokens * kv_dim);
        let v_view = v_rest.slice(..num_tokens * kv_dim);

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
                .arg(&(num_tokens as i32))
                .arg(&(num_heads as i32))
                .arg(&(num_kv_heads as i32))
                .arg(&(head_dim as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_tokens as u32, grid_y, 1),
                    block_dim: (half_dim.min(1024) as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| LLMError::GpuError(format!("fused_rope_cache_write_batch: {e}")))?;
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
        fa3: Option<&Fa3Kernels>,
    ) -> Result<()> {
        let q_view = qkv.slice(..q_end);
        if attn.is_prefill {
            self.prefill_attention(&q_view, attn, num_tokens, attn_out)
        } else {
            self.decode_attention(
                &q_view, attn, num_tokens, attn_out, split_out, split_max, split_sum, fa3,
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
        fa3: Option<&Fa3Kernels>,
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

        // FA3 SM90 path: WGMMA/TMA-accelerated paged KV decode
        if let Some(fa3k) = fa3 {
            if head_dim == 128 {
                return self.decode_attention_fa3_sm90(
                    q, attn, num_seqs, output, split_out, fa3k,
                );
            }
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

    fn decode_attention_fa3_sm90(
        &self,
        q: &CudaView<'_, f16>,
        attn: &AttentionMeta<'_>,
        num_seqs: usize,
        output: &mut CudaSlice<f16>,
        workspace: &mut CudaSlice<f32>,  // reuse split_out as workspace
        fa3: &Fa3Kernels,
    ) -> Result<()> {
        let cfg = &self.config;
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let max_blocks_per_seq = (attn.block_tables.len() / num_seqs.max(1)) as i32;
        // num_blocks_total = total physical blocks in KV cache
        let kv_stride = cfg.block_size * cfg.num_kv_heads * cfg.head_dim;
        let num_blocks_total = (attn.key_cache.len() / kv_stride.max(1)) as i32;

        // Get raw device pointers
        let stream = &self.stream;
        let (q_ptr, _gq) = q.device_ptr(stream);
        let (kc_ptr, _gkc) = attn.key_cache.device_ptr(stream);
        let (vc_ptr, _gvc) = attn.value_cache.device_ptr(stream);
        let (o_ptr, _go) = output.device_ptr_mut(stream);
        let (bt_ptr, _gbt) = attn.block_tables.device_ptr(stream);
        let (cl_ptr, _gcl) = attn.context_lens.device_ptr(stream);
        let (ws_ptr, _gws) = workspace.device_ptr_mut(stream);
        let stream_ptr = stream.cu_stream() as u64;

        fa3.paged_decode(
            q_ptr as u64,
            kc_ptr as u64,
            vc_ptr as u64,
            o_ptr as u64,
            bt_ptr as u64,
            cl_ptr as u64,
            ws_ptr as u64,
            scale,
            num_seqs as i32,
            cfg.num_heads as i32,
            cfg.num_kv_heads as i32,
            cfg.head_dim as i32,
            cfg.block_size as i32,
            max_blocks_per_seq,
            num_blocks_total,
            stream_ptr,
        ).map_err(|e| LLMError::GpuError(e))
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
