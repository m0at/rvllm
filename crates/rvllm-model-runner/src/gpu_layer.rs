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

    use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
    use half::f16;
    use tracing::{info, trace};

    use rvllm_core::error::{LLMError, Result};
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::stream::GpuStream;

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
        /// Number of dimensions that get RoPE (head_dim * partial_rotary_factor).
        pub rotary_dim: usize,
        /// Qwen3.5: q_proj produces 2x head_dim (query + output gate).
        pub has_attn_output_gate: bool,
        /// Weight offset for RMSNorm: 0.0 (standard) or 1.0 (Qwen3.5 "(1+w)" style).
        pub rms_norm_weight_offset: f32,
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
        /// Per-head Q RMSNorm weight [head_dim]. None for models without QK norm.
        pub q_norm: Option<&'a CudaSlice<f32>>,
        /// Per-head K RMSNorm weight [head_dim]. None for models without QK norm.
        pub k_norm: Option<&'a CudaSlice<f32>>,
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

    /// FP8 weight references for a single transformer layer.
    ///
    /// Projection weights are u8 (FP8 E4M3) with optional per-tensor f32 scales.
    /// Norm weights and biases remain f32. At compute time, FP8 weights are
    /// dequantized to f16 on-the-fly before hgemm.
    pub struct GpuLayerWeightsFp8<'a> {
        pub input_layernorm: &'a CudaSlice<f32>,
        pub q_proj: &'a CudaSlice<u8>,
        pub k_proj: &'a CudaSlice<u8>,
        pub v_proj: &'a CudaSlice<u8>,
        pub o_proj: &'a CudaSlice<u8>,
        pub q_proj_scale: Option<&'a CudaSlice<f32>>,
        pub k_proj_scale: Option<&'a CudaSlice<f32>>,
        pub v_proj_scale: Option<&'a CudaSlice<f32>>,
        pub o_proj_scale: Option<&'a CudaSlice<f32>>,
        pub q_proj_bias: Option<&'a CudaSlice<f32>>,
        pub k_proj_bias: Option<&'a CudaSlice<f32>>,
        pub v_proj_bias: Option<&'a CudaSlice<f32>>,
        /// Per-head Q RMSNorm weight [head_dim]. None for models without QK norm.
        pub q_norm: Option<&'a CudaSlice<f32>>,
        /// Per-head K RMSNorm weight [head_dim]. None for models without QK norm.
        pub k_norm: Option<&'a CudaSlice<f32>>,
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        pub gate_proj: &'a CudaSlice<u8>,
        pub up_proj: &'a CudaSlice<u8>,
        pub down_proj: &'a CudaSlice<u8>,
        pub gate_proj_scale: Option<&'a CudaSlice<f32>>,
        pub up_proj_scale: Option<&'a CudaSlice<f32>>,
        pub down_proj_scale: Option<&'a CudaSlice<f32>>,
    }

    /// GPTQ INT4 weight references for a single transformer layer.
    ///
    /// Projection weights are packed INT4 (u8 raw bytes) with per-group f32 scales
    /// and zero points. Norm weights and biases remain f32.
    pub struct GpuLayerWeightsGptq<'a> {
        pub input_layernorm: &'a CudaSlice<f32>,
        pub q_proj: &'a CudaSlice<u8>,            // [N, K/8] repacked int32
        pub k_proj: &'a CudaSlice<u8>,
        pub v_proj: &'a CudaSlice<u8>,
        pub o_proj: &'a CudaSlice<u8>,
        pub q_proj_scales: &'a CudaSlice<f32>,     // [N, num_groups]
        pub k_proj_scales: &'a CudaSlice<f32>,
        pub v_proj_scales: &'a CudaSlice<f32>,
        pub o_proj_scales: &'a CudaSlice<f32>,
        pub q_proj_zeros: &'a CudaSlice<f32>,      // [N, num_groups]
        pub k_proj_zeros: &'a CudaSlice<f32>,
        pub v_proj_zeros: &'a CudaSlice<f32>,
        pub o_proj_zeros: &'a CudaSlice<f32>,
        pub q_proj_bias: Option<&'a CudaSlice<f32>>,
        pub k_proj_bias: Option<&'a CudaSlice<f32>>,
        pub v_proj_bias: Option<&'a CudaSlice<f32>>,
        pub q_norm: Option<&'a CudaSlice<f32>>,
        pub k_norm: Option<&'a CudaSlice<f32>>,
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        pub gate_proj: &'a CudaSlice<u8>,
        pub up_proj: &'a CudaSlice<u8>,
        pub down_proj: &'a CudaSlice<u8>,
        pub gate_proj_scales: &'a CudaSlice<f32>,
        pub up_proj_scales: &'a CudaSlice<f32>,
        pub down_proj_scales: &'a CudaSlice<f32>,
        pub gate_proj_zeros: &'a CudaSlice<f32>,
        pub up_proj_zeros: &'a CudaSlice<f32>,
        pub down_proj_zeros: &'a CudaSlice<f32>,
    }

    /// GPTQ INT4 weight references for a linear attention (Mamba-2 SSM) layer.
    pub struct GpuLinearAttnWeightsGptq<'a> {
        pub input_layernorm: &'a CudaSlice<f32>,
        pub in_proj_qkv: &'a CudaSlice<u8>,
        pub in_proj_z: &'a CudaSlice<u8>,
        pub out_proj: &'a CudaSlice<u8>,
        pub in_proj_qkv_scales: &'a CudaSlice<f32>,
        pub in_proj_z_scales: &'a CudaSlice<f32>,
        pub out_proj_scales: &'a CudaSlice<f32>,
        pub in_proj_qkv_zeros: &'a CudaSlice<f32>,
        pub in_proj_z_zeros: &'a CudaSlice<f32>,
        pub out_proj_zeros: &'a CudaSlice<f32>,
        pub qkv_dim: usize,
        pub z_dim: usize,
        pub in_proj_a: &'a CudaSlice<f32>,
        pub in_proj_b: &'a CudaSlice<f32>,
        pub conv1d_weight: &'a CudaSlice<f32>,
        pub a_log: &'a CudaSlice<f32>,
        pub dt_bias: &'a CudaSlice<f32>,
        pub ssm_norm: &'a CudaSlice<f32>,
        pub num_key_heads: usize,
        pub num_value_heads: usize,
        pub key_head_dim: usize,
        pub value_head_dim: usize,
        pub conv_kernel_size: usize,
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        pub gate_proj: &'a CudaSlice<u8>,
        pub up_proj: &'a CudaSlice<u8>,
        pub down_proj: &'a CudaSlice<u8>,
        pub gate_proj_scales: &'a CudaSlice<f32>,
        pub up_proj_scales: &'a CudaSlice<f32>,
        pub down_proj_scales: &'a CudaSlice<f32>,
        pub gate_proj_zeros: &'a CudaSlice<f32>,
        pub up_proj_zeros: &'a CudaSlice<f32>,
        pub down_proj_zeros: &'a CudaSlice<f32>,
    }

    /// FP8 weight references for a linear attention (Mamba-2 SSM) layer.
    ///
    /// Used by Qwen3.5 hybrid models. The large linear projections (in_proj_qkv,
    /// in_proj_z, out_proj) are FP8 E4M3 quantized. Small projections and norms
    /// remain in f32. The SSM core (conv1d, selective scan) is simplified for
    /// benchmarking -- GEMMs dominate ~90% of inference time.
    pub struct GpuLinearAttnWeightsFp8<'a> {
        pub input_layernorm: &'a CudaSlice<f32>,
        // Mamba-2 projections (FP8)
        pub in_proj_qkv: &'a CudaSlice<u8>,      // [qkv_dim, hidden] FP8
        pub in_proj_z: &'a CudaSlice<u8>,         // [z_dim, hidden] FP8
        pub out_proj: &'a CudaSlice<u8>,          // [hidden, z_dim] FP8
        pub in_proj_qkv_scale: Option<&'a CudaSlice<f32>>,
        pub in_proj_z_scale: Option<&'a CudaSlice<f32>>,
        pub out_proj_scale: Option<&'a CudaSlice<f32>>,
        // Projection output dimensions (derived from weight shapes)
        pub qkv_dim: usize,    // 10240 = Q(2048) + K(2048) + V(6144)
        pub z_dim: usize,      // 6144 = 48 value heads * 128 dim

        // Mamba-2 SSM weights (f32, non-quantized)
        pub in_proj_a: &'a CudaSlice<f32>,        // [num_value_heads=48, hidden=5120] → dt projection
        pub in_proj_b: &'a CudaSlice<f32>,        // [num_value_heads=48, hidden=5120] → gate/scale
        pub conv1d_weight: &'a CudaSlice<f32>,    // [conv_dim=10240, 1, kernel_size=4] → depthwise conv
        pub a_log: &'a CudaSlice<f32>,            // [num_value_heads=48] → log SSM decay
        pub dt_bias: &'a CudaSlice<f32>,          // [num_value_heads=48] → timestep bias
        pub ssm_norm: &'a CudaSlice<f32>,         // [value_head_dim=128] → per-head RMSNorm

        // Mamba-2 SSM config
        pub num_key_heads: usize,      // 16
        pub num_value_heads: usize,    // 48
        pub key_head_dim: usize,       // 128
        pub value_head_dim: usize,     // 128
        pub conv_kernel_size: usize,   // 4

        // Post-attention norm
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        // MLP weights (FP8) -- same as transformer layers
        pub gate_proj: &'a CudaSlice<u8>,
        pub up_proj: &'a CudaSlice<u8>,
        pub down_proj: &'a CudaSlice<u8>,
        pub gate_proj_scale: Option<&'a CudaSlice<f32>>,
        pub up_proj_scale: Option<&'a CudaSlice<f32>>,
        pub down_proj_scale: Option<&'a CudaSlice<f32>>,
    }

    /// F32 weight references for a linear attention (Mamba-2 SSM) layer.
    ///
    /// Used by Qwen3.5 hybrid models with BF16/F32 weights. All projections are f32.
    pub struct GpuLinearAttnWeights<'a> {
        pub input_layernorm: &'a CudaSlice<f32>,
        pub in_proj_qkv: &'a CudaSlice<f32>,      // [qkv_dim, hidden]
        pub in_proj_z: &'a CudaSlice<f32>,         // [z_dim, hidden]
        pub out_proj: &'a CudaSlice<f32>,          // [hidden, z_dim]
        pub qkv_dim: usize,
        pub z_dim: usize,
        pub in_proj_a: &'a CudaSlice<f32>,
        pub in_proj_b: &'a CudaSlice<f32>,
        pub conv1d_weight: &'a CudaSlice<f32>,
        pub a_log: &'a CudaSlice<f32>,
        pub dt_bias: &'a CudaSlice<f32>,
        pub ssm_norm: &'a CudaSlice<f32>,
        pub num_key_heads: usize,
        pub num_value_heads: usize,
        pub key_head_dim: usize,
        pub value_head_dim: usize,
        pub conv_kernel_size: usize,
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        pub gate_proj: &'a CudaSlice<f32>,
        pub up_proj: &'a CudaSlice<f32>,
        pub down_proj: &'a CudaSlice<f32>,
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

    /// Pre-allocated GPU workspace for SSM (Gated DeltaNet) per-token loop
    /// and projection/MLP outputs. Eliminates ~30 cudaMalloc+cudaMemset calls
    /// per SSM layer per decode step.
    pub struct SsmWorkspace {
        // --- SSM per-token loop buffers ---
        pub qkv_t: CudaSlice<f32>,     // [conv_dim] = 10240
        pub conv_out: CudaSlice<f32>,   // [conv_dim] = 10240
        pub q_t: CudaSlice<f32>,        // [key_dim] = 2048
        pub k_t: CudaSlice<f32>,        // [key_dim] = 2048
        pub v_t: CudaSlice<f32>,        // [value_dim] = 6144
        pub a_t: CudaSlice<f32>,        // [num_value_heads] = 48
        pub b_t: CudaSlice<f32>,        // [num_value_heads] = 48
        pub beta_t: CudaSlice<f32>,     // [num_value_heads] = 48
        pub g_t: CudaSlice<f32>,        // [num_value_heads] = 48
        pub q_exp: CudaSlice<f32>,      // [num_value_heads * key_head_dim] = 6144
        pub k_exp: CudaSlice<f32>,      // [num_value_heads * key_head_dim] = 6144
        pub y_t: CudaSlice<f32>,        // [num_value_heads * value_head_dim] = 6144
        pub z_t: CudaSlice<f32>,        // [z_dim] = 6144
        pub norm_t: CudaSlice<f32>,     // [value_dim] = 6144
        pub zero_bias: CudaSlice<f32>,  // [conv_dim] = 10240
        // --- Projection/MLP output buffers (decode only, num_tokens=1) ---
        pub proj_qkv: CudaSlice<f32>,   // [qkv_dim] = 10240
        pub proj_z: CudaSlice<f32>,     // [z_dim] = 6144
        pub proj_a: CudaSlice<f32>,     // [num_value_heads] = 48
        pub proj_b: CudaSlice<f32>,     // [num_value_heads] = 48
        pub proj_out: CudaSlice<f32>,   // [hidden] = 5120
        pub proj_gate: CudaSlice<f32>,  // [intermediate] = 17408
        pub proj_up: CudaSlice<f32>,    // [intermediate] = 17408
        pub proj_down: CudaSlice<f32>,  // [hidden] = 5120
        // --- Scratch buffers to eliminate cudaMalloc in decode hot path ---
        pub scratch_normed: CudaSlice<f32>,    // [hidden] = 5120
        pub scratch_a_raw: CudaSlice<f32>,     // [num_value_heads] = 48
        pub scratch_b_raw: CudaSlice<f32>,     // [num_value_heads] = 48
        pub scratch_norm_out: CudaSlice<f32>,  // [value_dim] = 6144
        pub scratch_residual: CudaSlice<f32>,  // [hidden] = 5120
        pub scratch_normed2: CudaSlice<f32>,   // [hidden] = 5120
        pub scratch_fused: CudaSlice<f32>,     // [intermediate] = 17408
        pub scratch_output: CudaSlice<f32>,    // [hidden] = 5120
    }

    impl SsmWorkspace {
        pub fn new(device: &Arc<CudaDevice>, hidden: usize, intermediate: usize) -> std::result::Result<Self, LLMError> {
            // Qwen3.5-27B SSM dimensions
            let num_key_heads = 16;
            let num_value_heads = 48;
            let key_head_dim = 128;
            let value_head_dim = 128;
            let conv_dim = 10240;
            let key_dim = num_key_heads * key_head_dim;       // 2048
            let value_dim = num_value_heads * value_head_dim;  // 6144
            let z_dim = value_dim;                             // 6144
            let gqa_dim = num_value_heads * key_head_dim;      // 6144
            let y_size = num_value_heads * value_head_dim;     // 6144

            let e = |msg: String| LLMError::GpuError(msg);
            Ok(Self {
                qkv_t: device.alloc_zeros::<f32>(conv_dim).map_err(|x| e(format!("ws qkv_t: {x}")))?,
                conv_out: device.alloc_zeros::<f32>(conv_dim).map_err(|x| e(format!("ws conv_out: {x}")))?,
                q_t: device.alloc_zeros::<f32>(key_dim).map_err(|x| e(format!("ws q_t: {x}")))?,
                k_t: device.alloc_zeros::<f32>(key_dim).map_err(|x| e(format!("ws k_t: {x}")))?,
                v_t: device.alloc_zeros::<f32>(value_dim).map_err(|x| e(format!("ws v_t: {x}")))?,
                a_t: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws a_t: {x}")))?,
                b_t: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws b_t: {x}")))?,
                beta_t: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws beta_t: {x}")))?,
                g_t: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws g_t: {x}")))?,
                q_exp: device.alloc_zeros::<f32>(gqa_dim).map_err(|x| e(format!("ws q_exp: {x}")))?,
                k_exp: device.alloc_zeros::<f32>(gqa_dim).map_err(|x| e(format!("ws k_exp: {x}")))?,
                y_t: device.alloc_zeros::<f32>(y_size).map_err(|x| e(format!("ws y_t: {x}")))?,
                z_t: device.alloc_zeros::<f32>(z_dim).map_err(|x| e(format!("ws z_t: {x}")))?,
                norm_t: device.alloc_zeros::<f32>(value_dim).map_err(|x| e(format!("ws norm_t: {x}")))?,
                zero_bias: device.alloc_zeros::<f32>(conv_dim).map_err(|x| e(format!("ws zero_bias: {x}")))?,
                proj_qkv: device.alloc_zeros::<f32>(conv_dim).map_err(|x| e(format!("ws proj_qkv: {x}")))?,
                proj_z: device.alloc_zeros::<f32>(z_dim).map_err(|x| e(format!("ws proj_z: {x}")))?,
                proj_a: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws proj_a: {x}")))?,
                proj_b: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws proj_b: {x}")))?,
                proj_out: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("ws proj_out: {x}")))?,
                proj_gate: device.alloc_zeros::<f32>(intermediate).map_err(|x| e(format!("ws proj_gate: {x}")))?,
                proj_up: device.alloc_zeros::<f32>(intermediate).map_err(|x| e(format!("ws proj_up: {x}")))?,
                proj_down: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("ws proj_down: {x}")))?,
                scratch_normed: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("ws scratch_normed: {x}")))?,
                scratch_a_raw: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws scratch_a_raw: {x}")))?,
                scratch_b_raw: device.alloc_zeros::<f32>(num_value_heads).map_err(|x| e(format!("ws scratch_b_raw: {x}")))?,
                scratch_norm_out: device.alloc_zeros::<f32>(value_dim).map_err(|x| e(format!("ws scratch_norm_out: {x}")))?,
                scratch_residual: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("ws scratch_residual: {x}")))?,
                scratch_normed2: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("ws scratch_normed2: {x}")))?,
                scratch_fused: device.alloc_zeros::<f32>(intermediate).map_err(|x| e(format!("ws scratch_fused: {x}")))?,
                scratch_output: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("ws scratch_output: {x}")))?,
            })
        }
    }

    /// Pre-allocated GPU workspace for attention layers.
    /// Eliminates ~19 cudaMalloc calls per attention layer per decode step
    /// (16 attention layers × 19 = ~304 allocations per decode).
    pub struct AttnWorkspace {
        pub scratch_normed: CudaSlice<f32>,     // [hidden]
        pub scratch_q_full: CudaSlice<f32>,     // [q_proj_dim] (with gate: q_dim*2)
        pub scratch_q: CudaSlice<f32>,          // [q_dim]
        pub scratch_gate_out: CudaSlice<f32>,   // [q_dim]
        pub scratch_k: CudaSlice<f32>,          // [kv_dim]
        pub scratch_v: CudaSlice<f32>,          // [kv_dim]
        pub scratch_q_rot: CudaSlice<f32>,      // [q_dim]
        pub scratch_k_rot: CudaSlice<f32>,      // [kv_dim]
        pub scratch_attn_out: CudaSlice<f32>,   // [q_dim]
        pub scratch_gated: CudaSlice<f32>,      // [q_dim]
        pub scratch_attn_proj: CudaSlice<f32>,  // [hidden]
        pub scratch_residual: CudaSlice<f32>,   // [hidden]
        pub scratch_normed2: CudaSlice<f32>,    // [hidden]
        pub proj_gate: CudaSlice<f32>,          // [intermediate]
        pub proj_up: CudaSlice<f32>,            // [intermediate]
        pub scratch_fused: CudaSlice<f32>,      // [intermediate]
        pub proj_down: CudaSlice<f32>,          // [hidden]
    }

    impl AttnWorkspace {
        pub fn new(
            device: &Arc<CudaDevice>,
            hidden: usize,
            intermediate: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            has_attn_output_gate: bool,
        ) -> std::result::Result<Self, LLMError> {
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let q_proj_dim = if has_attn_output_gate { q_dim * 2 } else { q_dim };
            let e = |msg: String| LLMError::GpuError(msg);
            Ok(Self {
                scratch_normed: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("attn_ws normed: {x}")))?,
                scratch_q_full: device.alloc_zeros::<f32>(q_proj_dim).map_err(|x| e(format!("attn_ws q_full: {x}")))?,
                scratch_q: device.alloc_zeros::<f32>(q_dim).map_err(|x| e(format!("attn_ws q: {x}")))?,
                scratch_gate_out: device.alloc_zeros::<f32>(q_dim).map_err(|x| e(format!("attn_ws gate_out: {x}")))?,
                scratch_k: device.alloc_zeros::<f32>(kv_dim).map_err(|x| e(format!("attn_ws k: {x}")))?,
                scratch_v: device.alloc_zeros::<f32>(kv_dim).map_err(|x| e(format!("attn_ws v: {x}")))?,
                scratch_q_rot: device.alloc_zeros::<f32>(q_dim).map_err(|x| e(format!("attn_ws q_rot: {x}")))?,
                scratch_k_rot: device.alloc_zeros::<f32>(kv_dim).map_err(|x| e(format!("attn_ws k_rot: {x}")))?,
                scratch_attn_out: device.alloc_zeros::<f32>(q_dim).map_err(|x| e(format!("attn_ws attn_out: {x}")))?,
                scratch_gated: device.alloc_zeros::<f32>(q_dim).map_err(|x| e(format!("attn_ws gated: {x}")))?,
                scratch_attn_proj: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("attn_ws attn_proj: {x}")))?,
                scratch_residual: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("attn_ws residual: {x}")))?,
                scratch_normed2: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("attn_ws normed2: {x}")))?,
                proj_gate: device.alloc_zeros::<f32>(intermediate).map_err(|x| e(format!("attn_ws gate: {x}")))?,
                proj_up: device.alloc_zeros::<f32>(intermediate).map_err(|x| e(format!("attn_ws up: {x}")))?,
                scratch_fused: device.alloc_zeros::<f32>(intermediate).map_err(|x| e(format!("attn_ws fused: {x}")))?,
                proj_down: device.alloc_zeros::<f32>(hidden).map_err(|x| e(format!("attn_ws down: {x}")))?,
            })
        }
    }

    /// One complete GPU transformer layer.
    ///
    /// Holds references to the kernel loader and cuBLAS handle;
    /// weights are passed in per-call via `GpuLayerWeights`.
    pub struct GpuTransformerLayer {
        config: GpuLayerConfig,
        device: Arc<CudaDevice>,
    }

    impl GpuTransformerLayer {
        pub fn new(config: GpuLayerConfig, device: Arc<CudaDevice>) -> Self {
            Self { config, device }
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
            let normed = Self::rms_norm_with_offset(
                &self.device,
                input.hidden_states,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                cfg.rms_norm_weight_offset,
            )?;

            // 2. QKV projections via hgemm (f16 weights)
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let mut q = CudaLinearLayer::forward_once_f16(
                &normed, weights.q_proj, num_tokens, q_dim, hidden, blas,
            )?;
            let mut k = CudaLinearLayer::forward_once_f16(
                &normed, weights.k_proj, num_tokens, kv_dim, hidden, blas,
            )?;
            let mut v = CudaLinearLayer::forward_once_f16(
                &normed, weights.v_proj, num_tokens, kv_dim, hidden, blas,
            )?;

            // QKV biases (f32)
            if let Some(bias) = weights.q_proj_bias {
                Self::add_bias(&self.device, &mut q, bias, num_tokens, q_dim)?;
            }
            if let Some(bias) = weights.k_proj_bias {
                Self::add_bias(&self.device, &mut k, bias, num_tokens, kv_dim)?;
            }
            if let Some(bias) = weights.v_proj_bias {
                Self::add_bias(&self.device, &mut v, bias, num_tokens, kv_dim)?;
            }

            // 3. RoPE
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.device,
                &q,
                &k,
                input.positions,
                input.rope_cos,
                input.rope_sin,
                num_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
                cfg.rotary_dim,
            )?;

            // 4. KV cache write + attention
            Self::cache_write(
                &self.device,
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
                    &self.device,
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
                    &self.device,
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
                &attn_out, weights.o_proj, num_tokens, hidden, q_dim, blas,
            )?;

            // Residual
            let residual = Self::add_tensors(
                &self.device,
                input.hidden_states,
                &attn_proj,
                num_tokens * hidden,
            )?;

            // 6. Post-attention RMSNorm (f32)
            let normed2 = Self::rms_norm_with_offset(
                &self.device,
                &residual,
                weights.post_attention_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                cfg.rms_norm_weight_offset,
            )?;

            // 7. MLP (f16 weights)
            let gate = CudaLinearLayer::forward_once_f16(
                &normed2, weights.gate_proj, num_tokens, intermediate, hidden, blas,
            )?;
            let up = CudaLinearLayer::forward_once_f16(
                &normed2, weights.up_proj, num_tokens, intermediate, hidden, blas,
            )?;
            let fused = Self::fused_silu_mul(&self.device, &gate, &up, num_tokens * intermediate)?;
            let mlp_out = CudaLinearLayer::forward_once_f16(
                &fused, weights.down_proj, num_tokens, hidden, intermediate, blas,
            )?;

            // 8. Residual
            Self::add_tensors(&self.device, &residual, &mlp_out, num_tokens * hidden)
        }

        /// FP8 forward pass -- dequants FP8 u8 weights to f16 on-the-fly, then hgemm.
        /// Norms, biases, RoPE, and attention remain in f32.
        /// When `ws` is Some and num_tokens==1, uses pre-allocated scratch buffers
        /// to eliminate ~19 cudaMalloc calls per layer.
        pub fn forward_fp8(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeightsFp8<'_>,
            blas: &CublasHandle,
            ws: Option<&mut AttnWorkspace>,
        ) -> Result<CudaSlice<f32>> {
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
            let q_proj_dim = if cfg.has_attn_output_gate { q_dim * 2 } else { q_dim };

            // Use scratch buffers for decode (num_tokens==1) to eliminate ~19 cudaMalloc per layer.
            // Convert to raw pointer to avoid borrow-checker issues with multiple mutable accesses.
            // SAFETY: ws is exclusively owned by this function call; single-threaded CUDA dispatch.
            let ws_ptr: Option<*mut AttnWorkspace> = ws.map(|w| w as *mut AttnWorkspace);
            let use_scratch = num_tokens == 1 && ws_ptr.is_some();

            // 1. Pre-attention RMSNorm
            let mut _normed_alloc: CudaSlice<f32>;
            let normed: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                Self::rms_norm_into(
                    &self.device, &mut ws.scratch_normed,
                    input.hidden_states, weights.input_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_normed, num_tokens * hidden)?;
                &ws.scratch_normed
            } else {
                _normed_alloc = Self::rms_norm_with_offset(
                    &self.device, input.hidden_states, weights.input_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut _normed_alloc, num_tokens * hidden)?;
                &_normed_alloc
            };

            // 2. QKV projections via fp8 dequant + hgemm
            // Q projection
            let mut _q_full_alloc: CudaSlice<f32>;
            let q_full_ref: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.scratch_q_full, normed, weights.q_proj, weights.q_proj_scale,
                    num_tokens, q_proj_dim, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_q_full, num_tokens * q_proj_dim)?;
                &ws.scratch_q_full
            } else {
                _q_full_alloc = CudaLinearLayer::forward_once_fp8(
                    normed, weights.q_proj, weights.q_proj_scale, num_tokens, q_proj_dim, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _q_full_alloc, num_tokens * q_proj_dim)?;
                &_q_full_alloc
            };

            // QKV projections, bias, split, and per-head norm — combined to satisfy borrow checker.
            // Scratch path: all writes go to workspace buffers.
            // Alloc path: allocates owned buffers.
            let mut _q_alloc: CudaSlice<f32>;
            let mut _gate_out_alloc: CudaSlice<f32>;
            let mut _k_alloc: CudaSlice<f32>;
            let mut _v_alloc: CudaSlice<f32>;
            let mut _q_normed_alloc: CudaSlice<f32>;
            let mut _k_normed_alloc: CudaSlice<f32>;

            let attn_gate_ref: Option<&CudaSlice<f32>>;

            // Use scratch path: all projections → workspace buffers
            let (q_normed, k_normed, v_ref): (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>) = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };

                // Q split
                if cfg.has_attn_output_gate {
                    let threads = 256u32;
                    let total = (num_tokens * q_dim) as u32;
                    let blocks = (total + threads - 1) / threads;
                    let launch_cfg = LaunchConfig {
                        grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0,
                    };
                    let trunc_kernel = self.device.get_func("attn_gate", "truncate_q_kernel")
                        .ok_or_else(|| LLMError::GpuError("truncate_q_kernel not loaded".into()))?;
                    let split_kernel = self.device.get_func("attn_gate", "split_gate_kernel")
                        .ok_or_else(|| LLMError::GpuError("split_gate_kernel not loaded".into()))?;
                    unsafe {
                        trunc_kernel.launch(launch_cfg, (
                            &mut ws.scratch_q, q_full_ref, num_tokens as i32, q_dim as i32, head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("truncate_q launch: {e}")))?;
                        split_kernel.launch(launch_cfg, (
                            &mut ws.scratch_gate_out, q_full_ref, num_tokens as i32, q_dim as i32, head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("split_gate launch: {e}")))?;
                    }
                    attn_gate_ref = Some(&ws.scratch_gate_out);
                } else {
                    // No split needed — copy q_full into scratch_q
                    self.device.dtod_copy(q_full_ref, &mut ws.scratch_q)
                        .map_err(|e| LLMError::GpuError(format!("q copy: {e}")))?;
                    attn_gate_ref = None;
                }

                // K, V projections
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.scratch_k, normed, weights.k_proj, weights.k_proj_scale,
                    num_tokens, kv_dim, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_k, num_tokens * kv_dim)?;
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.scratch_v, normed, weights.v_proj, weights.v_proj_scale,
                    num_tokens, kv_dim, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_v, num_tokens * kv_dim)?;

                // Biases
                if let Some(bias) = weights.q_proj_bias {
                    Self::add_bias(&self.device, &mut ws.scratch_q, bias, num_tokens, q_dim)?;
                }
                if let Some(bias) = weights.k_proj_bias {
                    Self::add_bias(&self.device, &mut ws.scratch_k, bias, num_tokens, kv_dim)?;
                }
                if let Some(bias) = weights.v_proj_bias {
                    Self::add_bias(&self.device, &mut ws.scratch_v, bias, num_tokens, kv_dim)?;
                }

                // Q/K per-head RMSNorm
                if let Some(q_norm_w) = weights.q_norm {
                    // Use scratch_q_rot as temp for in-place norm
                    self.device.dtod_copy(&ws.scratch_q, &mut ws.scratch_q_rot)
                        .map_err(|e| LLMError::GpuError(format!("q norm temp: {e}")))?;
                    Self::rms_norm_into(
                        &self.device, &mut ws.scratch_q, &ws.scratch_q_rot, q_norm_w,
                        cfg.rms_norm_eps, num_tokens * num_heads, head_dim, 0.0,
                    )?;
                    Self::round_to_bf16(&self.device, &mut ws.scratch_q, num_tokens * q_dim)?;
                }
                if let Some(k_norm_w) = weights.k_norm {
                    self.device.dtod_copy(&ws.scratch_k, &mut ws.scratch_k_rot)
                        .map_err(|e| LLMError::GpuError(format!("k norm temp: {e}")))?;
                    Self::rms_norm_into(
                        &self.device, &mut ws.scratch_k, &ws.scratch_k_rot, k_norm_w,
                        cfg.rms_norm_eps, num_tokens * num_kv_heads, head_dim, 0.0,
                    )?;
                    Self::round_to_bf16(&self.device, &mut ws.scratch_k, num_tokens * kv_dim)?;
                }

                (&ws.scratch_q, &ws.scratch_k, &ws.scratch_v)
            } else {
                // Allocating path
                if cfg.has_attn_output_gate {
                    let threads = 256u32;
                    let total = (num_tokens * q_dim) as u32;
                    let blocks = (total + threads - 1) / threads;
                    let launch_cfg = LaunchConfig {
                        grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0,
                    };
                    let trunc_kernel = self.device.get_func("attn_gate", "truncate_q_kernel")
                        .ok_or_else(|| LLMError::GpuError("truncate_q_kernel not loaded".into()))?;
                    let split_kernel = self.device.get_func("attn_gate", "split_gate_kernel")
                        .ok_or_else(|| LLMError::GpuError("split_gate_kernel not loaded".into()))?;
                    _q_alloc = self.device.alloc_zeros::<f32>(num_tokens * q_dim)
                        .map_err(|e| LLMError::GpuError(format!("q split alloc: {e}")))?;
                    _gate_out_alloc = self.device.alloc_zeros::<f32>(num_tokens * q_dim)
                        .map_err(|e| LLMError::GpuError(format!("gate split alloc: {e}")))?;
                    unsafe {
                        trunc_kernel.launch(launch_cfg, (
                            &mut _q_alloc, q_full_ref, num_tokens as i32, q_dim as i32, head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("truncate_q launch: {e}")))?;
                        split_kernel.launch(launch_cfg, (
                            &mut _gate_out_alloc, q_full_ref, num_tokens as i32, q_dim as i32, head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("split_gate launch: {e}")))?;
                    }
                    attn_gate_ref = Some(&_gate_out_alloc);
                } else {
                    _q_alloc = self.device.alloc_zeros::<f32>(num_tokens * q_proj_dim)
                        .map_err(|e| LLMError::GpuError(format!("q alloc: {e}")))?;
                    self.device.dtod_copy(q_full_ref, &mut _q_alloc)
                        .map_err(|e| LLMError::GpuError(format!("q copy: {e}")))?;
                    attn_gate_ref = None;
                }

                _k_alloc = CudaLinearLayer::forward_once_fp8(
                    normed, weights.k_proj, weights.k_proj_scale, num_tokens, kv_dim, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _k_alloc, num_tokens * kv_dim)?;
                _v_alloc = CudaLinearLayer::forward_once_fp8(
                    normed, weights.v_proj, weights.v_proj_scale, num_tokens, kv_dim, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _v_alloc, num_tokens * kv_dim)?;

                // Biases
                if let Some(bias) = weights.q_proj_bias {
                    Self::add_bias(&self.device, &mut _q_alloc, bias, num_tokens, q_dim)?;
                }
                if let Some(bias) = weights.k_proj_bias {
                    Self::add_bias(&self.device, &mut _k_alloc, bias, num_tokens, kv_dim)?;
                }
                if let Some(bias) = weights.v_proj_bias {
                    Self::add_bias(&self.device, &mut _v_alloc, bias, num_tokens, kv_dim)?;
                }

                // Q/K per-head RMSNorm
                if let Some(q_norm_w) = weights.q_norm {
                    _q_normed_alloc = Self::rms_norm(
                        &self.device, &_q_alloc, q_norm_w, cfg.rms_norm_eps,
                        num_tokens * num_heads, head_dim,
                    )?;
                    Self::round_to_bf16(&self.device, &mut _q_normed_alloc, num_tokens * q_dim)?;
                    // Swap to use normed version
                    _q_alloc = _q_normed_alloc;
                }
                if let Some(k_norm_w) = weights.k_norm {
                    _k_normed_alloc = Self::rms_norm(
                        &self.device, &_k_alloc, k_norm_w, cfg.rms_norm_eps,
                        num_tokens * num_kv_heads, head_dim,
                    )?;
                    Self::round_to_bf16(&self.device, &mut _k_normed_alloc, num_tokens * kv_dim)?;
                    _k_alloc = _k_normed_alloc;
                }

                (&_q_alloc, &_k_alloc, &_v_alloc)
            };

            // 3. RoPE
            let mut _q_rot_alloc: CudaSlice<f32>;
            let mut _k_rot_alloc: CudaSlice<f32>;
            let (q_rot_ref, k_rot_ref): (&CudaSlice<f32>, &CudaSlice<f32>) = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                Self::apply_rotary_embedding_into(
                    &self.device, &mut ws.scratch_q_rot, &mut ws.scratch_k_rot,
                    q_normed, k_normed,
                    input.positions, input.rope_cos, input.rope_sin,
                    num_tokens, num_heads, num_kv_heads, head_dim, cfg.rotary_dim,
                )?;
                (&ws.scratch_q_rot, &ws.scratch_k_rot)
            } else {
                let (qr, kr) = Self::apply_rotary_embedding(
                    &self.device, q_normed, k_normed,
                    input.positions, input.rope_cos, input.rope_sin,
                    num_tokens, num_heads, num_kv_heads, head_dim, cfg.rotary_dim,
                )?;
                _q_rot_alloc = qr;
                _k_rot_alloc = kr;
                (&_q_rot_alloc, &_k_rot_alloc)
            };

            // 4. KV cache write + attention
            Self::cache_write(
                &self.device,
                k_rot_ref,
                v_ref,
                input.key_cache,
                input.value_cache,
                input.slot_mapping,
                num_tokens,
                num_kv_heads,
                head_dim,
            )?;

            let mut _attn_out_alloc: CudaSlice<f32>;
            let attn_out_ref: &CudaSlice<f32> = if input.is_prefill {
                _attn_out_alloc = Self::prefill_attention(
                    &self.device, q_rot_ref,
                    input.key_cache, input.value_cache,
                    input.block_tables, input.context_lens, input.seq_start_pos,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?;
                &_attn_out_alloc
            } else if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                Self::decode_attention_into(
                    &self.device, &mut ws.scratch_attn_out, q_rot_ref,
                    input.key_cache, input.value_cache,
                    input.block_tables, input.context_lens,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?;
                &ws.scratch_attn_out
            } else {
                _attn_out_alloc = Self::decode_attention(
                    &self.device, q_rot_ref,
                    input.key_cache, input.value_cache,
                    input.block_tables, input.context_lens,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?;
                &_attn_out_alloc
            };

            // 4b. Apply output gate: attn_out = attn_out * sigmoid(gate)
            let mut _gated_alloc: CudaSlice<f32>;
            let gated_attn_out: &CudaSlice<f32> = if let Some(gate) = attn_gate_ref {
                let n = num_tokens * q_dim;
                let threads = 256u32;
                let blocks = ((n as u32) + threads - 1) / threads;
                let launch_cfg = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let kernel = self.device
                    .get_func("attn_gate", "sigmoid_gate_kernel")
                    .ok_or_else(|| LLMError::GpuError("sigmoid_gate_kernel not loaded".into()))?;

                if use_scratch {
                    let ws = unsafe { &mut *ws_ptr.unwrap() };
                    unsafe {
                        kernel.launch(launch_cfg, (
                            &mut ws.scratch_gated, attn_out_ref, gate, n as i32,
                        )).map_err(|e| LLMError::GpuError(format!("sigmoid_gate launch: {e}")))?;
                    }
                    &ws.scratch_gated
                } else {
                    _gated_alloc = self.device
                        .alloc_zeros::<f32>(n)
                        .map_err(|e| LLMError::GpuError(format!("sigmoid_gate alloc: {e}")))?;
                    unsafe {
                        kernel.launch(launch_cfg, (
                            &mut _gated_alloc, attn_out_ref, gate, n as i32,
                        )).map_err(|e| LLMError::GpuError(format!("sigmoid_gate launch: {e}")))?;
                    }
                    &_gated_alloc
                }
            } else {
                attn_out_ref
            };

            // 5. Output projection
            let mut _attn_proj_alloc: CudaSlice<f32>;
            let attn_proj: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.scratch_attn_proj, gated_attn_out, weights.o_proj, weights.o_proj_scale,
                    num_tokens, hidden, q_dim, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_attn_proj, num_tokens * hidden)?;
                &ws.scratch_attn_proj
            } else {
                _attn_proj_alloc = CudaLinearLayer::forward_once_fp8(
                    gated_attn_out, weights.o_proj, weights.o_proj_scale, num_tokens, hidden, q_dim, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _attn_proj_alloc, num_tokens * hidden)?;
                &_attn_proj_alloc
            };

            // Residual
            let mut _residual_alloc: CudaSlice<f32>;
            let residual: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                Self::add_tensors_into(
                    &self.device, &mut ws.scratch_residual,
                    input.hidden_states, attn_proj, num_tokens * hidden,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_residual, num_tokens * hidden)?;
                &ws.scratch_residual
            } else {
                _residual_alloc = Self::add_tensors(
                    &self.device, input.hidden_states, attn_proj, num_tokens * hidden,
                )?;
                Self::round_to_bf16(&self.device, &mut _residual_alloc, num_tokens * hidden)?;
                &_residual_alloc
            };

            // 6. Post-attention RMSNorm
            let mut _normed2_alloc: CudaSlice<f32>;
            let normed2: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                Self::rms_norm_into(
                    &self.device, &mut ws.scratch_normed2,
                    residual, weights.post_attention_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_normed2, num_tokens * hidden)?;
                &ws.scratch_normed2
            } else {
                _normed2_alloc = Self::rms_norm_with_offset(
                    &self.device, residual, weights.post_attention_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut _normed2_alloc, num_tokens * hidden)?;
                &_normed2_alloc
            };

            // 7. MLP (fp8 weights)
            let mut _gate_mlp_alloc: CudaSlice<f32>;
            let gate_mlp: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_gate, normed2, weights.gate_proj, weights.gate_proj_scale,
                    num_tokens, intermediate, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_gate, num_tokens * intermediate)?;
                &ws.proj_gate
            } else {
                _gate_mlp_alloc = CudaLinearLayer::forward_once_fp8(
                    normed2, weights.gate_proj, weights.gate_proj_scale, num_tokens, intermediate, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _gate_mlp_alloc, num_tokens * intermediate)?;
                &_gate_mlp_alloc
            };
            let mut _up_alloc: CudaSlice<f32>;
            let up: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_up, normed2, weights.up_proj, weights.up_proj_scale,
                    num_tokens, intermediate, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_up, num_tokens * intermediate)?;
                &ws.proj_up
            } else {
                _up_alloc = CudaLinearLayer::forward_once_fp8(
                    normed2, weights.up_proj, weights.up_proj_scale, num_tokens, intermediate, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _up_alloc, num_tokens * intermediate)?;
                &_up_alloc
            };
            let mut _fused_alloc: CudaSlice<f32>;
            let fused: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                Self::fused_silu_mul_into(&self.device, &mut ws.scratch_fused, gate_mlp, up, num_tokens * intermediate)?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_fused, num_tokens * intermediate)?;
                &ws.scratch_fused
            } else {
                _fused_alloc = Self::fused_silu_mul(&self.device, gate_mlp, up, num_tokens * intermediate)?;
                Self::round_to_bf16(&self.device, &mut _fused_alloc, num_tokens * intermediate)?;
                &_fused_alloc
            };
            let mut _mlp_out_alloc: CudaSlice<f32>;
            let mlp_out: &CudaSlice<f32> = if use_scratch {
                let ws = unsafe { &mut *ws_ptr.unwrap() };
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_down, fused, weights.down_proj, weights.down_proj_scale,
                    num_tokens, hidden, intermediate, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_down, num_tokens * hidden)?;
                &ws.proj_down
            } else {
                _mlp_out_alloc = CudaLinearLayer::forward_once_fp8(
                    fused, weights.down_proj, weights.down_proj_scale, num_tokens, hidden, intermediate, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _mlp_out_alloc, num_tokens * hidden)?;
                &_mlp_out_alloc
            };

            // 8. Final residual (always allocates — return value must be owned)
            let mut output = Self::add_tensors(&self.device, residual, mlp_out, num_tokens * hidden)?;
            Self::round_to_bf16(&self.device, &mut output, num_tokens * hidden)?;
            Ok(output)
        }

        /// GPTQ INT4 attention layer forward pass.
        ///
        /// Identical to forward_fp8 but uses INT4 GEMV kernels for linear projections.
        /// Halves weight bandwidth vs FP8 for ~2x decode throughput.
        pub fn forward_gptq(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeightsGptq<'_>,
            blas: &CublasHandle,
            group_size: usize,
        ) -> Result<CudaSlice<f32>> {
            use crate::layers::linear_cuda::CudaLinearLayer;

            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            // 1. Pre-attention RMSNorm
            let mut normed = Self::rms_norm_with_offset(
                &self.device, input.hidden_states, weights.input_layernorm,
                cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
            )?;
            Self::round_to_bf16(&self.device, &mut normed, num_tokens * hidden)?;

            // 2. QKV projections via INT4 GEMV
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let q_proj_dim = if cfg.has_attn_output_gate { q_dim * 2 } else { q_dim };

            let mut q_full = CudaLinearLayer::forward_once_gptq(
                &normed, weights.q_proj, weights.q_proj_scales, weights.q_proj_zeros,
                num_tokens, q_proj_dim, hidden, group_size, blas,
            )?;
            Self::round_to_bf16(&self.device, &mut q_full, num_tokens * q_proj_dim)?;

            let (mut q, attn_gate) = if cfg.has_attn_output_gate {
                let mut q_out = self.device.alloc_zeros::<f32>(num_tokens * q_dim)
                    .map_err(|e| LLMError::GpuError(format!("q split alloc: {e}")))?;
                let mut gate_out = self.device.alloc_zeros::<f32>(num_tokens * q_dim)
                    .map_err(|e| LLMError::GpuError(format!("gate split alloc: {e}")))?;
                let threads = 256u32;
                let total = (num_tokens * q_dim) as u32;
                let blocks = (total + threads - 1) / threads;
                let launch_cfg = LaunchConfig {
                    grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0,
                };
                let trunc_kernel = self.device.get_func("attn_gate", "truncate_q_kernel")
                    .ok_or_else(|| LLMError::GpuError("truncate_q_kernel not loaded".into()))?;
                unsafe {
                    trunc_kernel.launch(launch_cfg, (
                        &mut q_out, &q_full, num_tokens as i32, q_dim as i32, head_dim as i32,
                    )).map_err(|e| LLMError::GpuError(format!("truncate_q launch: {e}")))?;
                }
                let split_kernel = self.device.get_func("attn_gate", "split_gate_kernel")
                    .ok_or_else(|| LLMError::GpuError("split_gate_kernel not loaded".into()))?;
                unsafe {
                    split_kernel.launch(launch_cfg, (
                        &mut gate_out, &q_full, num_tokens as i32, q_dim as i32, head_dim as i32,
                    )).map_err(|e| LLMError::GpuError(format!("split_gate launch: {e}")))?;
                }
                (q_out, Some(gate_out))
            } else {
                (q_full, None)
            };

            let mut k = CudaLinearLayer::forward_once_gptq(
                &normed, weights.k_proj, weights.k_proj_scales, weights.k_proj_zeros,
                num_tokens, kv_dim, hidden, group_size, blas,
            )?;
            Self::round_to_bf16(&self.device, &mut k, num_tokens * kv_dim)?;
            let mut v = CudaLinearLayer::forward_once_gptq(
                &normed, weights.v_proj, weights.v_proj_scales, weights.v_proj_zeros,
                num_tokens, kv_dim, hidden, group_size, blas,
            )?;
            Self::round_to_bf16(&self.device, &mut v, num_tokens * kv_dim)?;

            if let Some(bias) = weights.q_proj_bias { Self::add_bias(&self.device, &mut q, bias, num_tokens, q_dim)?; }
            if let Some(bias) = weights.k_proj_bias { Self::add_bias(&self.device, &mut k, bias, num_tokens, kv_dim)?; }
            if let Some(bias) = weights.v_proj_bias { Self::add_bias(&self.device, &mut v, bias, num_tokens, kv_dim)?; }

            if let Some(q_norm_w) = weights.q_norm {
                q = Self::rms_norm(&self.device, &q, q_norm_w, cfg.rms_norm_eps, num_tokens * num_heads, head_dim)?;
                Self::round_to_bf16(&self.device, &mut q, num_tokens * q_dim)?;
            }
            if let Some(k_norm_w) = weights.k_norm {
                k = Self::rms_norm(&self.device, &k, k_norm_w, cfg.rms_norm_eps, num_tokens * num_kv_heads, head_dim)?;
                Self::round_to_bf16(&self.device, &mut k, num_tokens * kv_dim)?;
            }

            // 3. RoPE
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.device, &q, &k, input.positions, input.rope_cos, input.rope_sin,
                num_tokens, num_heads, num_kv_heads, head_dim, cfg.rotary_dim,
            )?;

            // 4. KV cache + attention
            Self::cache_write(
                &self.device, &k_rot, &v, input.key_cache, input.value_cache,
                input.slot_mapping, num_tokens, num_kv_heads, head_dim,
            )?;

            let attn_out = if input.is_prefill {
                Self::prefill_attention(
                    &self.device, &q_rot, input.key_cache, input.value_cache,
                    input.block_tables, input.context_lens, input.seq_start_pos,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?
            } else {
                Self::decode_attention(
                    &self.device, &q_rot, input.key_cache, input.value_cache,
                    input.block_tables, input.context_lens, num_tokens, input.num_seqs,
                    num_heads, num_kv_heads, head_dim, input.max_context_len, input.block_size,
                )?
            };

            // Output gate
            let gated_attn_out = if let Some(ref gate) = attn_gate {
                let n = num_tokens * q_dim;
                let mut gated = self.device.alloc_zeros::<f32>(n)
                    .map_err(|e| LLMError::GpuError(format!("sigmoid_gate alloc: {e}")))?;
                let threads = 256u32;
                let blocks = ((n as u32) + threads - 1) / threads;
                let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                let kernel = self.device.get_func("attn_gate", "sigmoid_gate_kernel")
                    .ok_or_else(|| LLMError::GpuError("sigmoid_gate_kernel not loaded".into()))?;
                unsafe {
                    kernel.launch(cfg, (&mut gated, &attn_out, gate, n as i32))
                        .map_err(|e| LLMError::GpuError(format!("sigmoid_gate launch: {e}")))?;
                }
                gated
            } else { attn_out };

            // 5. Output projection
            let mut attn_proj = CudaLinearLayer::forward_once_gptq(
                &gated_attn_out, weights.o_proj, weights.o_proj_scales, weights.o_proj_zeros,
                num_tokens, hidden, q_dim, group_size, blas,
            )?;
            Self::round_to_bf16(&self.device, &mut attn_proj, num_tokens * hidden)?;

            let mut residual = Self::add_tensors(&self.device, input.hidden_states, &attn_proj, num_tokens * hidden)?;
            Self::round_to_bf16(&self.device, &mut residual, num_tokens * hidden)?;

            // 6. Post-attention RMSNorm
            let mut normed2 = Self::rms_norm_with_offset(
                &self.device, &residual, weights.post_attention_layernorm,
                cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
            )?;
            Self::round_to_bf16(&self.device, &mut normed2, num_tokens * hidden)?;

            // 7. MLP
            let mut gate = CudaLinearLayer::forward_once_gptq(
                &normed2, weights.gate_proj, weights.gate_proj_scales, weights.gate_proj_zeros,
                num_tokens, intermediate, hidden, group_size, blas,
            )?;
            Self::round_to_bf16(&self.device, &mut gate, num_tokens * intermediate)?;
            let mut up = CudaLinearLayer::forward_once_gptq(
                &normed2, weights.up_proj, weights.up_proj_scales, weights.up_proj_zeros,
                num_tokens, intermediate, hidden, group_size, blas,
            )?;
            Self::round_to_bf16(&self.device, &mut up, num_tokens * intermediate)?;
            let mut fused = Self::fused_silu_mul(&self.device, &gate, &up, num_tokens * intermediate)?;
            Self::round_to_bf16(&self.device, &mut fused, num_tokens * intermediate)?;
            let mut mlp_out = CudaLinearLayer::forward_once_gptq(
                &fused, weights.down_proj, weights.down_proj_scales, weights.down_proj_zeros,
                num_tokens, hidden, intermediate, group_size, blas,
            )?;
            Self::round_to_bf16(&self.device, &mut mlp_out, num_tokens * hidden)?;

            // 8. Residual
            let mut output = Self::add_tensors(&self.device, &residual, &mlp_out, num_tokens * hidden)?;
            Self::round_to_bf16(&self.device, &mut output, num_tokens * hidden)?;
            Ok(output)
        }

        /// Linear attention (Gated DeltaNet) forward pass with GPTQ INT4 weights.
        /// Mirrors forward_linear_attn_fp8 but uses INT4 GEMV kernel for projections.
        pub fn forward_linear_attn_gptq(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLinearAttnWeightsGptq<'_>,
            blas: &CublasHandle,
            group_size: usize,
            ssm_state: &mut CudaSlice<f32>,
            conv_state: &mut CudaSlice<f32>,
            ws: &mut SsmWorkspace,
        ) -> Result<CudaSlice<f32>> {
            use crate::layers::linear_cuda::CudaLinearLayer;

            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let intermediate = cfg.intermediate_size;
            let qkv_dim = weights.qkv_dim;
            let z_dim = weights.z_dim;
            let num_key_heads = weights.num_key_heads;
            let num_value_heads = weights.num_value_heads;
            let key_head_dim = weights.key_head_dim;
            let value_head_dim = weights.value_head_dim;
            let key_dim = num_key_heads * key_head_dim;
            let value_dim = num_value_heads * value_head_dim;
            let conv_dim = qkv_dim;
            let kernel_size = weights.conv_kernel_size;

            // 1. Pre-attention RMSNorm
            let mut normed = Self::rms_norm_with_offset(
                &self.device, input.hidden_states, weights.input_layernorm,
                cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
            )?;
            Self::round_to_bf16(&self.device, &mut normed, num_tokens * hidden)?;

            // 2. GPTQ INT4 projections
            // in_proj_qkv: [T, hidden] -> [T, qkv_dim=10240]
            let mut _qkv_alloc: CudaSlice<f32>;
            let mixed_qkv_pre_conv: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_gptq_into(
                    &mut ws.proj_qkv, &normed, weights.in_proj_qkv,
                    weights.in_proj_qkv_scales, weights.in_proj_qkv_zeros,
                    num_tokens, qkv_dim, hidden, group_size, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_qkv, num_tokens * qkv_dim)?;
                &ws.proj_qkv
            } else {
                _qkv_alloc = CudaLinearLayer::forward_once_gptq(
                    &normed, weights.in_proj_qkv, weights.in_proj_qkv_scales,
                    weights.in_proj_qkv_zeros,
                    num_tokens, qkv_dim, hidden, group_size, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _qkv_alloc, num_tokens * qkv_dim)?;
                &_qkv_alloc
            };

            // in_proj_z: [T, hidden] -> [T, z_dim=6144]
            let mut _z_alloc: CudaSlice<f32>;
            let z: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_gptq_into(
                    &mut ws.proj_z, &normed, weights.in_proj_z,
                    weights.in_proj_z_scales, weights.in_proj_z_zeros,
                    num_tokens, z_dim, hidden, group_size, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_z, num_tokens * z_dim)?;
                &ws.proj_z
            } else {
                _z_alloc = CudaLinearLayer::forward_once_gptq(
                    &normed, weights.in_proj_z, weights.in_proj_z_scales,
                    weights.in_proj_z_zeros,
                    num_tokens, z_dim, hidden, group_size, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _z_alloc, num_tokens * z_dim)?;
                &_z_alloc
            };

            // 3. in_proj_a and in_proj_b via cuBLAS sgemm (f32, small matrices)
            let mut a_raw = Self::linear(
                &self.device, blas, &normed, weights.in_proj_a,
                num_tokens, num_value_heads, hidden,
            )?;
            Self::round_to_bf16(&self.device, &mut a_raw, num_tokens * num_value_heads)?;
            let mut b_raw = Self::linear(
                &self.device, blas, &normed, weights.in_proj_b,
                num_tokens, num_value_heads, hidden,
            )?;
            Self::round_to_bf16(&self.device, &mut b_raw, num_tokens * num_value_heads)?;

            // 4-10. Sequential per-token SSM processing
            let f32_bytes = std::mem::size_of::<f32>();
            let y_token_size = num_value_heads * value_head_dim;
            let mut norm_out = self.device.alloc_zeros::<f32>(num_tokens * value_dim)
                .map_err(|e| LLMError::GpuError(format!("norm_out alloc: {e}")))?;
            let scale = 1.0f32 / (key_head_dim as f32).sqrt();

            for t in 0..num_tokens {
                // Extract mixed_qkv_pre_conv[t]
                {
                    use cudarc::driver::DevicePtr;
                    let src = *DevicePtr::device_ptr(mixed_qkv_pre_conv) + (t * conv_dim * f32_bytes) as u64;
                    unsafe { cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.qkv_t), src, conv_dim * f32_bytes); }
                }

                // Conv1d step
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_conv1d_step")
                        .ok_or_else(|| LLMError::GpuError("mamba2_conv1d_step not loaded".into()))?;
                    let threads = 256u32;
                    let blocks = ((conv_dim as u32) + threads - 1) / threads;
                    let conv_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        kernel.launch(conv_cfg, (
                            conv_state as &mut CudaSlice<f32>, weights.conv1d_weight, &ws.zero_bias, &ws.qkv_t, &mut ws.conv_out,
                            conv_dim as i32, kernel_size as i32,
                        )).map_err(|e| LLMError::GpuError(format!("conv1d t={t}: {e}")))?;
                    }
                }

                // Split conv_out -> Q, K, V
                {
                    use cudarc::driver::DevicePtr;
                    let src_base = *DevicePtr::device_ptr(&ws.conv_out);
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.q_t), src_base, key_dim * f32_bytes);
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.k_t), src_base + (key_dim * f32_bytes) as u64, key_dim * f32_bytes);
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.v_t), src_base + (2 * key_dim * f32_bytes) as u64, value_dim * f32_bytes);
                    }
                }

                // Extract a[t], b[t]
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.a_t), *DevicePtr::device_ptr(&a_raw) + (t * num_value_heads * f32_bytes) as u64, num_value_heads * f32_bytes);
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.b_t), *DevicePtr::device_ptr(&b_raw) + (t * num_value_heads * f32_bytes) as u64, num_value_heads * f32_bytes);
                    }
                }

                // Compute gates
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_compute_gates")
                        .ok_or_else(|| LLMError::GpuError("mamba2_compute_gates not loaded".into()))?;
                    let threads = 64u32;
                    let blocks = ((num_value_heads as u32) + threads - 1) / threads;
                    let gate_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        kernel.launch(gate_cfg, (
                            &ws.a_t, &ws.b_t, weights.a_log, weights.dt_bias, &mut ws.beta_t, &mut ws.g_t, num_value_heads as i32,
                        )).map_err(|e| LLMError::GpuError(format!("gates t={t}: {e}")))?;
                    }
                }

                // L2-normalize Q and K
                {
                    let threads = key_head_dim.min(128) as u32;
                    let l2_cfg = LaunchConfig { grid_dim: (num_key_heads as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: threads * 4 };
                    let kq = self.device.get_func("mamba2_ssm", "mamba2_l2_normalize")
                        .ok_or_else(|| LLMError::GpuError("mamba2_l2_normalize not loaded".into()))?;
                    unsafe {
                        kq.launch(l2_cfg, (&mut ws.q_t, num_key_heads as i32, key_head_dim as i32, 1e-6f32))
                            .map_err(|e| LLMError::GpuError(format!("l2norm Q t={t}: {e}")))?;
                    }
                    let kk = self.device.get_func("mamba2_ssm", "mamba2_l2_normalize")
                        .ok_or_else(|| LLMError::GpuError("mamba2_l2_normalize not loaded".into()))?;
                    unsafe {
                        kk.launch(l2_cfg, (&mut ws.k_t, num_key_heads as i32, key_head_dim as i32, 1e-6f32))
                            .map_err(|e| LLMError::GpuError(format!("l2norm K t={t}: {e}")))?;
                    }
                }

                // GQA expand
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_gqa_expand")
                        .ok_or_else(|| LLMError::GpuError("mamba2_gqa_expand not loaded".into()))?;
                    let total = (num_value_heads * key_head_dim) as u32;
                    let threads = 256u32;
                    let blocks = (total + threads - 1) / threads;
                    let gqa_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        kernel.launch(gqa_cfg, (
                            &ws.q_t, &mut ws.q_exp, num_key_heads as i32, num_value_heads as i32, key_head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("gqa Q t={t}: {e}")))?;
                    }
                    // Re-fetch kernel — cudarc 0.12 launch() consumes self
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_gqa_expand")
                        .ok_or_else(|| LLMError::GpuError("mamba2_gqa_expand not loaded".into()))?;
                    unsafe {
                        kernel.launch(gqa_cfg, (
                            &ws.k_t, &mut ws.k_exp, num_key_heads as i32, num_value_heads as i32, key_head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("gqa K t={t}: {e}")))?;
                    }
                }

                // Gated delta rule SSM step
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_ssm_step")
                        .ok_or_else(|| LLMError::GpuError("mamba2_ssm_step not loaded".into()))?;
                    let threads = key_head_dim.min(128) as u32;
                    let ssm_cfg = LaunchConfig {
                        grid_dim: (num_value_heads as u32, value_head_dim as u32, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        kernel.launch(ssm_cfg, (
                            ssm_state as &mut CudaSlice<f32>,
                            &ws.q_exp, &ws.k_exp, &ws.v_t, &ws.beta_t, &ws.g_t,
                            &mut ws.y_t,
                            num_value_heads as i32, key_head_dim as i32, value_head_dim as i32, scale,
                        )).map_err(|e| LLMError::GpuError(format!("ssm_step t={t}: {e}")))?;
                    }
                }
                Self::round_to_bf16(&self.device, &mut ws.y_t, y_token_size)?;
                let ssm_size = num_value_heads * key_head_dim * value_head_dim;
                Self::round_to_bf16(&self.device, ssm_state, ssm_size)?;

                // Gated RMSNorm: silu(z[t]) * rms_norm(y_t)
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.z_t), *DevicePtr::device_ptr(z) + (t * z_dim * f32_bytes) as u64, z_dim * f32_bytes);
                    }
                }
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_norm_gate")
                        .ok_or_else(|| LLMError::GpuError("mamba2_norm_gate not loaded".into()))?;
                    let threads = value_head_dim.min(128) as u32;
                    let norm_cfg = LaunchConfig {
                        grid_dim: (num_value_heads as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 4 * 4 + 4,
                    };
                    unsafe {
                        kernel.launch(norm_cfg, (
                            &ws.y_t, &ws.z_t, weights.ssm_norm, &mut ws.norm_t,
                            num_value_heads as i32, value_head_dim as i32, 1e-6f32,
                        )).map_err(|e| LLMError::GpuError(format!("norm_gate t={t}: {e}")))?;
                    }
                }
                Self::round_to_bf16(&self.device, &mut ws.norm_t, value_dim)?;

                // Store norm_t into norm_out[t]
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&norm_out) + (t * value_dim * f32_bytes) as u64,
                            *DevicePtr::device_ptr(&ws.norm_t), value_dim * f32_bytes);
                    }
                }
            } // end per-token loop

            // 11. out_proj: [T, value_dim=6144] -> [T, hidden=5120]
            let mut _out_alloc: CudaSlice<f32>;
            let attn_proj: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_gptq_into(
                    &mut ws.proj_out, &norm_out, weights.out_proj,
                    weights.out_proj_scales, weights.out_proj_zeros,
                    num_tokens, hidden, z_dim, group_size, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_out, num_tokens * hidden)?;
                &ws.proj_out
            } else {
                _out_alloc = CudaLinearLayer::forward_once_gptq(
                    &norm_out, weights.out_proj, weights.out_proj_scales,
                    weights.out_proj_zeros,
                    num_tokens, hidden, z_dim, group_size, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _out_alloc, num_tokens * hidden)?;
                &_out_alloc
            };

            // 12. Residual
            let mut residual = Self::add_tensors(&self.device, input.hidden_states, attn_proj, num_tokens * hidden)?;
            Self::round_to_bf16(&self.device, &mut residual, num_tokens * hidden)?;

            // 13. Post-attention RMSNorm
            let mut normed2 = Self::rms_norm_with_offset(
                &self.device, &residual, weights.post_attention_layernorm,
                cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
            )?;
            Self::round_to_bf16(&self.device, &mut normed2, num_tokens * hidden)?;

            // 14. MLP via GPTQ INT4
            let mut _gate_alloc: CudaSlice<f32>;
            let gate: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_gptq_into(
                    &mut ws.proj_gate, &normed2, weights.gate_proj,
                    weights.gate_proj_scales, weights.gate_proj_zeros,
                    num_tokens, intermediate, hidden, group_size, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_gate, num_tokens * intermediate)?;
                &ws.proj_gate
            } else {
                _gate_alloc = CudaLinearLayer::forward_once_gptq(
                    &normed2, weights.gate_proj, weights.gate_proj_scales,
                    weights.gate_proj_zeros,
                    num_tokens, intermediate, hidden, group_size, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _gate_alloc, num_tokens * intermediate)?;
                &_gate_alloc
            };
            let mut _up_alloc: CudaSlice<f32>;
            let up: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_gptq_into(
                    &mut ws.proj_up, &normed2, weights.up_proj,
                    weights.up_proj_scales, weights.up_proj_zeros,
                    num_tokens, intermediate, hidden, group_size, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_up, num_tokens * intermediate)?;
                &ws.proj_up
            } else {
                _up_alloc = CudaLinearLayer::forward_once_gptq(
                    &normed2, weights.up_proj, weights.up_proj_scales,
                    weights.up_proj_zeros,
                    num_tokens, intermediate, hidden, group_size, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _up_alloc, num_tokens * intermediate)?;
                &_up_alloc
            };
            let mut fused = Self::fused_silu_mul(&self.device, gate, up, num_tokens * intermediate)?;
            Self::round_to_bf16(&self.device, &mut fused, num_tokens * intermediate)?;
            let mut _down_alloc: CudaSlice<f32>;
            let mlp_out: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_gptq_into(
                    &mut ws.proj_down, &fused, weights.down_proj,
                    weights.down_proj_scales, weights.down_proj_zeros,
                    num_tokens, hidden, intermediate, group_size, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_down, num_tokens * hidden)?;
                &ws.proj_down
            } else {
                _down_alloc = CudaLinearLayer::forward_once_gptq(
                    &fused, weights.down_proj, weights.down_proj_scales,
                    weights.down_proj_zeros,
                    num_tokens, hidden, intermediate, group_size, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _down_alloc, num_tokens * hidden)?;
                &_down_alloc
            };

            // 15. Residual
            let mut output = Self::add_tensors(&self.device, &residual, mlp_out, num_tokens * hidden)?;
            Self::round_to_bf16(&self.device, &mut output, num_tokens * hidden)?;
            Ok(output)
        }

        /// Linear attention (Mamba-2 SSM) forward pass for FP8 quantized weights.
        ///
        /// Runs all linear projections at correct sizes for accurate GEMM throughput
        /// measurement. The SSM core (conv1d, selective scan) is simplified: we run
        /// the projections and use silu-gated output as a stand-in. This is valid
        /// for benchmarking since GEMMs are ~90% of layer compute time.
        ///
        /// No KV cache interaction -- linear attention layers maintain internal SSM
        /// state instead of KV caches.
        /// Full Gated DeltaNet forward pass for Qwen3.5 linear attention layers.
        ///
        /// Decode-step pipeline (single token, seq_len=1):
        /// 1. RMSNorm(hidden) -> normed
        /// 2. in_proj_qkv(normed) -> mixed_qkv [qkv_dim]
        /// 3. in_proj_z(normed) -> z [z_dim]
        /// 4. in_proj_a(normed) -> a [num_value_heads]; in_proj_b(normed) -> b [num_value_heads]
        /// 5. Conv1d update: shift state, depthwise conv, SiLU -> mixed_qkv [qkv_dim]
        /// 6. Split mixed_qkv -> Q [key_dim], K [key_dim], V [value_dim]
        /// 7. Compute gates: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
        /// 8. L2-normalize Q and K
        /// 9. GQA expand Q, K from num_key_heads to num_value_heads
        /// 10. Gated delta rule SSM step (updates recurrent state in-place)
        /// 11. Gated RMSNorm: output = silu(z) * rms_norm(ssm_out)
        /// 12. out_proj(output) -> attn_out [hidden]
        /// 13. Residual + MLP
        pub fn forward_linear_attn_fp8(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLinearAttnWeightsFp8<'_>,
            blas: &CublasHandle,
            ssm_state: &mut CudaSlice<f32>,
            conv_state: &mut CudaSlice<f32>,
            ws: &mut SsmWorkspace,
        ) -> Result<CudaSlice<f32>> {
            use crate::layers::linear_cuda::CudaLinearLayer;

            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let intermediate = cfg.intermediate_size;
            let qkv_dim = weights.qkv_dim;
            let z_dim = weights.z_dim;
            let num_key_heads = weights.num_key_heads;
            let num_value_heads = weights.num_value_heads;
            let key_head_dim = weights.key_head_dim;
            let value_head_dim = weights.value_head_dim;
            let key_dim = num_key_heads * key_head_dim;     // 2048
            let value_dim = num_value_heads * value_head_dim; // 6144
            let conv_dim = qkv_dim;                          // 10240
            let kernel_size = weights.conv_kernel_size;      // 4

            trace!(
                layer = cfg.layer_idx,
                num_tokens,
                qkv_dim,
                z_dim,
                num_key_heads,
                num_value_heads,
                "gated deltanet linear attention forward_fp8"
            );

            // 1. Pre-attention RMSNorm
            // Decode path: write into pre-allocated scratch to avoid cudaMalloc
            let mut _normed_alloc: CudaSlice<f32>;
            let normed: &CudaSlice<f32> = if num_tokens == 1 {
                Self::rms_norm_into(
                    &self.device, &mut ws.scratch_normed,
                    input.hidden_states, weights.input_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_normed, num_tokens * hidden)?;
                &ws.scratch_normed
            } else {
                _normed_alloc = Self::rms_norm_with_offset(
                    &self.device, input.hidden_states, weights.input_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut _normed_alloc, num_tokens * hidden)?;
                &_normed_alloc
            };

            // DIAG: dump layer 0 normed input and FP8 projection output
            if cfg.layer_idx == 0 && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                let normed_cpu: Vec<f32> = self.device.dtoh_sync_copy(normed).unwrap_or_default();
                if normed_cpu.len() >= 20 {
                    info!("DIAG L0 normed[0..20]: {:?}", &normed_cpu[..20]);
                    let rms = (normed_cpu.iter().take(hidden).map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
                    info!("DIAG L0 normed token0 rms={rms:.6}");
                }
                // Also dump the raw FP8 weight bytes and scale
                let fp8_bytes: Vec<u8> = self.device.dtoh_sync_copy(weights.in_proj_qkv).unwrap_or_default();
                if fp8_bytes.len() >= 20 {
                    info!("DIAG L0 in_proj_qkv FP8 bytes[0..20]: {:?}", &fp8_bytes[..20]);
                }
                if let Some(scale) = weights.in_proj_qkv_scale {
                    let scale_cpu: Vec<f32> = self.device.dtoh_sync_copy(scale).unwrap_or_default();
                    if scale_cpu.len() >= 10 {
                        info!("DIAG L0 in_proj_qkv scale[0..10]: {:?}", &scale_cpu[..10]);
                        info!("DIAG L0 in_proj_qkv scale len={}", scale_cpu.len());
                    }
                }
            }

            // Intra-layer profiling (gated by RVLLM_PROFILE env var)
            let intra_profile = cfg.layer_idx == 0 && std::env::var("RVLLM_PROFILE").is_ok();
            let _proj_start = if intra_profile { self.device.synchronize().ok(); Some(std::time::Instant::now()) } else { None };

            // 2. FP8 projections (round outputs to BF16 precision)
            // in_proj_qkv: [T, hidden] -> [T, qkv_dim=10240]
            // Decode path: use pre-allocated workspace to avoid cudaMalloc+cudaMemset
            let mut _qkv_alloc: CudaSlice<f32>;
            let mixed_qkv_pre_conv: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_qkv, normed, weights.in_proj_qkv, weights.in_proj_qkv_scale,
                    num_tokens, qkv_dim, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_qkv, num_tokens * qkv_dim)?;
                &ws.proj_qkv
            } else {
                _qkv_alloc = CudaLinearLayer::forward_once_fp8(
                    normed, weights.in_proj_qkv, weights.in_proj_qkv_scale,
                    num_tokens, qkv_dim, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _qkv_alloc, num_tokens * qkv_dim)?;
                &_qkv_alloc
            };

            // DIAG: dump FP8 projection output
            if cfg.layer_idx == 0 && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                let out_cpu: Vec<f32> = self.device.dtoh_sync_copy(mixed_qkv_pre_conv).unwrap_or_default();
                if out_cpu.len() >= 20 {
                    info!("DIAG L0 in_proj_qkv out[0..20]: {:?}", &out_cpu[..20]);
                    let rms = (out_cpu.iter().take(qkv_dim).map(|x| x * x).sum::<f32>() / qkv_dim as f32).sqrt();
                    info!("DIAG L0 in_proj_qkv token0 output rms={rms:.6}");
                }
            }

            // in_proj_z: [T, hidden] -> [T, z_dim=6144]
            let mut _z_alloc: CudaSlice<f32>;
            let z: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_z, normed, weights.in_proj_z, weights.in_proj_z_scale,
                    num_tokens, z_dim, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_z, num_tokens * z_dim)?;
                &ws.proj_z
            } else {
                _z_alloc = CudaLinearLayer::forward_once_fp8(
                    normed, weights.in_proj_z, weights.in_proj_z_scale,
                    num_tokens, z_dim, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _z_alloc, num_tokens * z_dim)?;
                &_z_alloc
            };

            // 3. in_proj_a and in_proj_b via cuBLAS sgemm (batched across all tokens)
            // in_proj_a: [48, 5120], normed: [T, 5120] -> a_raw: [T, 48]
            // in_proj_b: [48, 5120], normed: [T, 5120] -> b_raw: [T, 48]
            let mut _a_raw_alloc: CudaSlice<f32>;
            let a_raw: &CudaSlice<f32> = if num_tokens == 1 {
                Self::linear_into(
                    &self.device, blas, &mut ws.scratch_a_raw, normed, weights.in_proj_a,
                    num_tokens, num_value_heads, hidden,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_a_raw, num_tokens * num_value_heads)?;
                &ws.scratch_a_raw
            } else {
                _a_raw_alloc = Self::linear(
                    &self.device, blas, normed, weights.in_proj_a,
                    num_tokens, num_value_heads, hidden,
                )?;
                Self::round_to_bf16(&self.device, &mut _a_raw_alloc, num_tokens * num_value_heads)?;
                &_a_raw_alloc
            };
            let mut _b_raw_alloc: CudaSlice<f32>;
            let b_raw: &CudaSlice<f32> = if num_tokens == 1 {
                Self::linear_into(
                    &self.device, blas, &mut ws.scratch_b_raw, normed, weights.in_proj_b,
                    num_tokens, num_value_heads, hidden,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_b_raw, num_tokens * num_value_heads)?;
                &ws.scratch_b_raw
            } else {
                _b_raw_alloc = Self::linear(
                    &self.device, blas, normed, weights.in_proj_b,
                    num_tokens, num_value_heads, hidden,
                )?;
                Self::round_to_bf16(&self.device, &mut _b_raw_alloc, num_tokens * num_value_heads)?;
                &_b_raw_alloc
            };

            if let Some(t0) = _proj_start {
                self.device.synchronize().ok();
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                info!(proj_ms = format!("{ms:.2}"), "PERF SSM_L0 projections");
            }
            let _ssm_start = if intra_profile { Some(std::time::Instant::now()) } else { None };

            // ================================================================
            // 4-10. SEQUENTIAL per-token SSM processing
            // The Gated DeltaNet recurrence is causal: state[t] depends on
            // state[t-1]. We MUST process tokens one at a time through the
            // conv1d -> split -> gates -> normalize -> GQA -> SSM pipeline.
            // Projections above are batched; this loop is sequential.
            // ================================================================
            let f32_bytes = std::mem::size_of::<f32>();
            let y_token_size = num_value_heads * value_head_dim; // 6144
            // Decode: reuse scratch buffer; prefill: allocate
            let mut _norm_out_alloc: CudaSlice<f32>;
            let norm_out: &mut CudaSlice<f32> = if num_tokens == 1 {
                &mut ws.scratch_norm_out
            } else {
                _norm_out_alloc = self.device.alloc_zeros::<f32>(num_tokens * value_dim)
                    .map_err(|e| LLMError::GpuError(format!("norm_out alloc: {e}")))?;
                &mut _norm_out_alloc
            };

            let scale = 1.0f32 / (key_head_dim as f32).sqrt();

            for t in 0..num_tokens {
                // --- 4a. Extract mixed_qkv_pre_conv[t] --- (workspace: no alloc)
                {
                    use cudarc::driver::DevicePtr;
                    let src = *DevicePtr::device_ptr(mixed_qkv_pre_conv)
                        + (t * conv_dim * f32_bytes) as u64;
                    let dst = *DevicePtr::device_ptr(&ws.qkv_t);
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            dst, src, conv_dim * f32_bytes,
                        );
                    }
                }

                // --- 4b. Conv1d step: shift state, depthwise conv, SiLU ---
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_conv1d_step")
                        .ok_or_else(|| LLMError::GpuError("mamba2_conv1d_step not loaded".into()))?;
                    let threads = 256u32;
                    let blocks = ((conv_dim as u32) + threads - 1) / threads;
                    let conv_cfg = LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        kernel.launch(conv_cfg, (
                            conv_state as &mut CudaSlice<f32>,
                            weights.conv1d_weight,
                            &ws.zero_bias,
                            &ws.qkv_t,
                            &mut ws.conv_out,
                            conv_dim as i32,
                            kernel_size as i32,
                        )).map_err(|e| LLMError::GpuError(format!("conv1d t={t}: {e}")))?;
                    }
                }
                Self::round_to_bf16(&self.device, &mut ws.conv_out, conv_dim)?;

                // DIAG: dump conv1d output and SSM intermediates for layer 0 token 0
                if cfg.layer_idx == 0 && t == 0 && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                    let co: Vec<f32> = self.device.dtoh_sync_copy(&ws.conv_out).unwrap_or_default();
                    if co.len() >= 20 {
                        let rms = (co.iter().map(|x| x * x).sum::<f32>() / co.len() as f32).sqrt();
                        info!("DIAG L0t0 conv_out[0..20]: {:?}, rms={rms:.6}", &co[..20]);
                    }
                    let cw: Vec<f32> = self.device.dtoh_sync_copy(weights.conv1d_weight).unwrap_or_default();
                    if cw.len() >= 40 {
                        let w3: Vec<f32> = (0..10).map(|ch| cw[ch * kernel_size + kernel_size - 1]).collect();
                        info!("DIAG L0t0 conv_weight[ch*4+3] first10: {:?}", w3);
                    }
                    let qt: Vec<f32> = self.device.dtoh_sync_copy(&ws.qkv_t).unwrap_or_default();
                    if qt.len() >= 10 {
                        info!("DIAG L0t0 qkv_t[0..10]: {:?}", &qt[..10]);
                    }
                }

                // --- 5. Split conv_out -> Q[key_dim], K[key_dim], V[value_dim] ---
                {
                    use cudarc::driver::DevicePtr;
                    let src_base = *DevicePtr::device_ptr(&ws.conv_out);
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.q_t),
                            src_base,
                            key_dim * f32_bytes,
                        );
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.k_t),
                            src_base + (key_dim * f32_bytes) as u64,
                            key_dim * f32_bytes,
                        );
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.v_t),
                            src_base + (2 * key_dim * f32_bytes) as u64,
                            value_dim * f32_bytes,
                        );
                    }
                }

                // --- 6. Extract a[t], b[t] from batched projections ---
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.a_t),
                            *DevicePtr::device_ptr(a_raw) + (t * num_value_heads * f32_bytes) as u64,
                            num_value_heads * f32_bytes,
                        );
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.b_t),
                            *DevicePtr::device_ptr(b_raw) + (t * num_value_heads * f32_bytes) as u64,
                            num_value_heads * f32_bytes,
                        );
                    }
                }

                // --- 7. Compute gates ---
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_compute_gates")
                        .ok_or_else(|| LLMError::GpuError("mamba2_compute_gates not loaded".into()))?;
                    let threads = 64u32;
                    let blocks = ((num_value_heads as u32) + threads - 1) / threads;
                    let gate_cfg = LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        kernel.launch(gate_cfg, (
                            &ws.a_t, &ws.b_t,
                            weights.a_log, weights.dt_bias,
                            &mut ws.beta_t, &mut ws.g_t,
                            num_value_heads as i32,
                        )).map_err(|e| LLMError::GpuError(format!("gates t={t}: {e}")))?;
                    }
                }
                Self::round_to_bf16(&self.device, &mut ws.beta_t, num_value_heads)?;
                Self::round_to_bf16(&self.device, &mut ws.g_t, num_value_heads)?;

                // DIAG: dump gates, a, b for layer 0 token 0
                if cfg.layer_idx == 0 && t == 0 && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                    let at: Vec<f32> = self.device.dtoh_sync_copy(&ws.a_t).unwrap_or_default();
                    let bt_v: Vec<f32> = self.device.dtoh_sync_copy(&ws.b_t).unwrap_or_default();
                    let beta_v: Vec<f32> = self.device.dtoh_sync_copy(&ws.beta_t).unwrap_or_default();
                    let gv: Vec<f32> = self.device.dtoh_sync_copy(&ws.g_t).unwrap_or_default();
                    info!("DIAG L0t0 a[0..10]: {:?}", &at[..at.len().min(10)]);
                    info!("DIAG L0t0 b[0..10]: {:?}", &bt_v[..bt_v.len().min(10)]);
                    info!("DIAG L0t0 beta[0..10]: {:?}", &beta_v[..beta_v.len().min(10)]);
                    info!("DIAG L0t0 g[0..10]: {:?}", &gv[..gv.len().min(10)]);
                    let q_v: Vec<f32> = self.device.dtoh_sync_copy(&ws.q_t).unwrap_or_default();
                    let k_v: Vec<f32> = self.device.dtoh_sync_copy(&ws.k_t).unwrap_or_default();
                    let v_v: Vec<f32> = self.device.dtoh_sync_copy(&ws.v_t).unwrap_or_default();
                    info!("DIAG L0t0 Q[0..10]: {:?}", &q_v[..q_v.len().min(10)]);
                    info!("DIAG L0t0 K[0..10]: {:?}", &k_v[..k_v.len().min(10)]);
                    info!("DIAG L0t0 V[0..10]: {:?}", &v_v[..v_v.len().min(10)]);
                }

                // --- 8. L2-normalize Q and K per-head ---
                {
                    let threads = key_head_dim.min(128) as u32;
                    let l2_cfg = LaunchConfig {
                        grid_dim: (num_key_heads as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    };
                    let kq = self.device.get_func("mamba2_ssm", "mamba2_l2_normalize")
                        .ok_or_else(|| LLMError::GpuError("mamba2_l2_normalize not loaded".into()))?;
                    unsafe {
                        kq.launch(l2_cfg, (
                            &mut ws.q_t, num_key_heads as i32, key_head_dim as i32, 1e-6f32,
                        )).map_err(|e| LLMError::GpuError(format!("l2norm Q t={t}: {e}")))?;
                    }
                    let kk = self.device.get_func("mamba2_ssm", "mamba2_l2_normalize")
                        .ok_or_else(|| LLMError::GpuError("mamba2_l2_normalize not loaded".into()))?;
                    unsafe {
                        kk.launch(l2_cfg, (
                            &mut ws.k_t, num_key_heads as i32, key_head_dim as i32, 1e-6f32,
                        )).map_err(|e| LLMError::GpuError(format!("l2norm K t={t}: {e}")))?;
                    }
                }
                Self::round_to_bf16(&self.device, &mut ws.q_t, key_dim)?;
                Self::round_to_bf16(&self.device, &mut ws.k_t, key_dim)?;

                // --- 9a. GQA expand Q, K: num_key_heads -> num_value_heads ---
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_gqa_expand")
                        .ok_or_else(|| LLMError::GpuError("mamba2_gqa_expand not loaded".into()))?;
                    let total = (num_value_heads * key_head_dim) as u32;
                    let threads = 256u32;
                    let blocks = (total + threads - 1) / threads;
                    let gqa_cfg = LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        kernel.launch(gqa_cfg, (
                            &ws.q_t, &mut ws.q_exp,
                            num_key_heads as i32, num_value_heads as i32, key_head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("gqa Q t={t}: {e}")))?;
                    }
                }
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_gqa_expand")
                        .ok_or_else(|| LLMError::GpuError("mamba2_gqa_expand not loaded".into()))?;
                    let total = (num_value_heads * key_head_dim) as u32;
                    let threads = 256u32;
                    let blocks = (total + threads - 1) / threads;
                    let gqa_cfg = LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        kernel.launch(gqa_cfg, (
                            &ws.k_t, &mut ws.k_exp,
                            num_key_heads as i32, num_value_heads as i32, key_head_dim as i32,
                        )).map_err(|e| LLMError::GpuError(format!("gqa K t={t}: {e}")))?;
                    }
                }

                // --- 9b. Gated DeltaNet SSM step (updates ssm_state in-place) ---
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_ssm_step")
                        .ok_or_else(|| LLMError::GpuError("mamba2_ssm_step not loaded".into()))?;
                    let threads = value_head_dim as u32;
                    let ssm_cfg = LaunchConfig {
                        grid_dim: (num_value_heads as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: (2 * value_head_dim * 4) as u32,
                    };
                    unsafe {
                        kernel.launch(ssm_cfg, (
                            ssm_state as &mut CudaSlice<f32>,
                            &ws.q_exp,
                            &ws.k_exp,
                            &ws.v_t,
                            &ws.beta_t,
                            &ws.g_t,
                            &mut ws.y_t,
                            num_value_heads as i32,
                            key_head_dim as i32,
                            value_head_dim as i32,
                            scale,
                        )).map_err(|e| LLMError::GpuError(format!("ssm_step t={t}: {e}")))?;
                    }
                }
                Self::round_to_bf16(&self.device, &mut ws.y_t, y_token_size)?;
                let ssm_size = num_value_heads * key_head_dim * value_head_dim;
                Self::round_to_bf16(&self.device, ssm_state, ssm_size)?;

                // DIAG: dump SSM output for layer 0 token 0
                if cfg.layer_idx == 0 && t == 0 && std::env::var("RVLLM_DUMP_FP8").is_ok() {
                    let yt: Vec<f32> = self.device.dtoh_sync_copy(&ws.y_t).unwrap_or_default();
                    let rms = (yt.iter().map(|x| x * x).sum::<f32>() / yt.len() as f32).sqrt();
                    info!("DIAG L0t0 y_t[0..10]: {:?}, rms={rms:.6}", &yt[..yt.len().min(10)]);
                }

                // --- 10. Gated RMSNorm for this token: silu(z[t]) * rms_norm(y_t) ---
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.z_t),
                            *DevicePtr::device_ptr(z) + (t * z_dim * f32_bytes) as u64,
                            z_dim * f32_bytes,
                        );
                    }
                }
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_norm_gate")
                        .ok_or_else(|| LLMError::GpuError("mamba2_norm_gate not loaded".into()))?;
                    let threads = value_head_dim.min(128) as u32;
                    let norm_cfg = LaunchConfig {
                        grid_dim: (num_value_heads as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 4 * 4 + 4,
                    };
                    unsafe {
                        kernel.launch(norm_cfg, (
                            &ws.y_t, &ws.z_t, weights.ssm_norm, &mut ws.norm_t,
                            num_value_heads as i32, value_head_dim as i32, 1e-6f32,
                        )).map_err(|e| LLMError::GpuError(format!("norm_gate t={t}: {e}")))?;
                    }
                }
                Self::round_to_bf16(&self.device, &mut ws.norm_t, value_dim)?;

                // Store norm_t into norm_out[t]
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&*norm_out) + (t * value_dim * f32_bytes) as u64,
                            *DevicePtr::device_ptr(&ws.norm_t),
                            value_dim * f32_bytes,
                        );
                    }
                }
            } // end per-token loop

            if let Some(t0) = _ssm_start {
                self.device.synchronize().ok();
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                info!(ssm_loop_ms = format!("{ms:.2}"), "PERF SSM_L0 sequential_loop");
            }
            let _post_start = if intra_profile { Some(std::time::Instant::now()) } else { None };
            // Detail profiling helper: sync + time since last checkpoint
            let detail_profile = intra_profile && std::env::var("RVLLM_PROFILE_DETAIL").is_ok();
            let mut _detail_t = if detail_profile { self.device.synchronize().ok(); Some(std::time::Instant::now()) } else { None };

            // 11. out_proj: [T, value_dim=6144] -> [T, hidden=5120]
            let mut _out_alloc: CudaSlice<f32>;
            let attn_proj: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_out, &*norm_out, weights.out_proj, weights.out_proj_scale,
                    num_tokens, hidden, z_dim, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_out, num_tokens * hidden)?;
                &ws.proj_out
            } else {
                _out_alloc = CudaLinearLayer::forward_once_fp8(
                    &*norm_out, weights.out_proj, weights.out_proj_scale,
                    num_tokens, hidden, z_dim, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _out_alloc, num_tokens * hidden)?;
                &_out_alloc
            };
            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL out_proj");
                *t = std::time::Instant::now();
            }

            // 12. Residual (decode: scratch, prefill: allocate)
            let mut _residual_alloc: CudaSlice<f32>;
            let residual: &CudaSlice<f32> = if num_tokens == 1 {
                Self::add_tensors_into(
                    &self.device, &mut ws.scratch_residual,
                    input.hidden_states, attn_proj, num_tokens * hidden,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_residual, num_tokens * hidden)?;
                &ws.scratch_residual
            } else {
                _residual_alloc = Self::add_tensors(&self.device, input.hidden_states, attn_proj, num_tokens * hidden)?;
                Self::round_to_bf16(&self.device, &mut _residual_alloc, num_tokens * hidden)?;
                &_residual_alloc
            };
            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL add_res1");
                *t = std::time::Instant::now();
            }

            // 13. Post-attention RMSNorm (decode: scratch, prefill: allocate)
            let mut _normed2_alloc: CudaSlice<f32>;
            let normed2: &CudaSlice<f32> = if num_tokens == 1 {
                Self::rms_norm_into(
                    &self.device, &mut ws.scratch_normed2,
                    residual, weights.post_attention_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_normed2, num_tokens * hidden)?;
                &ws.scratch_normed2
            } else {
                _normed2_alloc = Self::rms_norm_with_offset(
                    &self.device, residual, weights.post_attention_layernorm,
                    cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
                )?;
                Self::round_to_bf16(&self.device, &mut _normed2_alloc, num_tokens * hidden)?;
                &_normed2_alloc
            };
            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL rms_norm");
                *t = std::time::Instant::now();
            }

            // 14. MLP (same as transformer layers)
            let mut _gate_alloc: CudaSlice<f32>;
            let gate: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_gate, normed2, weights.gate_proj, weights.gate_proj_scale,
                    num_tokens, intermediate, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_gate, num_tokens * intermediate)?;
                &ws.proj_gate
            } else {
                _gate_alloc = CudaLinearLayer::forward_once_fp8(
                    normed2, weights.gate_proj, weights.gate_proj_scale,
                    num_tokens, intermediate, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _gate_alloc, num_tokens * intermediate)?;
                &_gate_alloc
            };
            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL gate_gemv");
                *t = std::time::Instant::now();
            }
            let mut _up_alloc: CudaSlice<f32>;
            let up: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_up, normed2, weights.up_proj, weights.up_proj_scale,
                    num_tokens, intermediate, hidden, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_up, num_tokens * intermediate)?;
                &ws.proj_up
            } else {
                _up_alloc = CudaLinearLayer::forward_once_fp8(
                    normed2, weights.up_proj, weights.up_proj_scale,
                    num_tokens, intermediate, hidden, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _up_alloc, num_tokens * intermediate)?;
                &_up_alloc
            };
            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL up_gemv");
                *t = std::time::Instant::now();
            }
            // Fused SiLU*mul (decode: scratch, prefill: allocate)
            let mut _fused_alloc: CudaSlice<f32>;
            let fused: &CudaSlice<f32> = if num_tokens == 1 {
                Self::fused_silu_mul_into(&self.device, &mut ws.scratch_fused, gate, up, num_tokens * intermediate)?;
                Self::round_to_bf16(&self.device, &mut ws.scratch_fused, num_tokens * intermediate)?;
                &ws.scratch_fused
            } else {
                _fused_alloc = Self::fused_silu_mul(&self.device, gate, up, num_tokens * intermediate)?;
                Self::round_to_bf16(&self.device, &mut _fused_alloc, num_tokens * intermediate)?;
                &_fused_alloc
            };
            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL silu_mul");
                *t = std::time::Instant::now();
            }
            let mut _down_alloc: CudaSlice<f32>;
            let mlp_out: &CudaSlice<f32> = if num_tokens == 1 {
                CudaLinearLayer::forward_once_fp8_into(
                    &mut ws.proj_down, fused, weights.down_proj, weights.down_proj_scale,
                    num_tokens, hidden, intermediate, &self.device,
                )?;
                Self::round_to_bf16(&self.device, &mut ws.proj_down, num_tokens * hidden)?;
                &ws.proj_down
            } else {
                _down_alloc = CudaLinearLayer::forward_once_fp8(
                    fused, weights.down_proj, weights.down_proj_scale,
                    num_tokens, hidden, intermediate, blas,
                )?;
                Self::round_to_bf16(&self.device, &mut _down_alloc, num_tokens * hidden)?;
                &_down_alloc
            };
            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL down_gemv");
                *t = std::time::Instant::now();
            }

            // 15. Final residual — this one still allocates (return value must be owned)
            let mut output = Self::add_tensors(&self.device, residual, mlp_out, num_tokens * hidden)?;
            Self::round_to_bf16(&self.device, &mut output, num_tokens * hidden)?;

            if let Some(ref mut t) = _detail_t {
                self.device.synchronize().ok();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                info!(ms = format!("{ms:.3}"), "DETAIL add_res2+alloc");
            }
            if let Some(t0) = _post_start {
                if !detail_profile { self.device.synchronize().ok(); }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                info!(post_ms = format!("{ms:.2}"), "PERF SSM_L0 post_ssm_mlp");
            }

            Ok(output)
        }

        /// Linear attention (Gated DeltaNet) forward pass with f32 weights.
        /// Identical to forward_linear_attn_fp8 but uses sgemm for projections.
        pub fn forward_linear_attn(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLinearAttnWeights<'_>,
            blas: &CublasHandle,
            ssm_state: &mut CudaSlice<f32>,
            conv_state: &mut CudaSlice<f32>,
            ws: &mut SsmWorkspace,
        ) -> Result<CudaSlice<f32>> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let intermediate = cfg.intermediate_size;
            let qkv_dim = weights.qkv_dim;
            let z_dim = weights.z_dim;
            let num_key_heads = weights.num_key_heads;
            let num_value_heads = weights.num_value_heads;
            let key_head_dim = weights.key_head_dim;
            let value_head_dim = weights.value_head_dim;
            let key_dim = num_key_heads * key_head_dim;
            let value_dim = num_value_heads * value_head_dim;
            let conv_dim = qkv_dim;
            let kernel_size = weights.conv_kernel_size;

            // 1. Pre-attention RMSNorm
            let normed = Self::rms_norm_with_offset(
                &self.device, input.hidden_states, weights.input_layernorm,
                cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset,
            )?;

            // 2. Projections via sgemm (f32)
            let mixed_qkv_pre_conv = Self::linear(
                &self.device, blas, &normed, weights.in_proj_qkv,
                num_tokens, qkv_dim, hidden,
            )?;
            let z = Self::linear(
                &self.device, blas, &normed, weights.in_proj_z,
                num_tokens, z_dim, hidden,
            )?;
            let a_raw = Self::linear(
                &self.device, blas, &normed, weights.in_proj_a,
                num_tokens, num_value_heads, hidden,
            )?;
            let b_raw = Self::linear(
                &self.device, blas, &normed, weights.in_proj_b,
                num_tokens, num_value_heads, hidden,
            )?;

            // 3-10. Sequential per-token SSM processing (identical to FP8 path)
            let f32_bytes = std::mem::size_of::<f32>();
            let y_token_size = num_value_heads * value_head_dim;
            let mut norm_out = self.device.alloc_zeros::<f32>(num_tokens * value_dim)
                .map_err(|e| LLMError::GpuError(format!("norm_out alloc: {e}")))?;
            let scale = 1.0f32 / (key_head_dim as f32).sqrt();

            for t in 0..num_tokens {
                // Extract mixed_qkv_pre_conv[t]
                {
                    use cudarc::driver::DevicePtr;
                    let src = *DevicePtr::device_ptr(&mixed_qkv_pre_conv) + (t * conv_dim * f32_bytes) as u64;
                    unsafe { cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.qkv_t), src, conv_dim * f32_bytes); }
                }

                // Conv1d step
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_conv1d_step")
                        .ok_or_else(|| LLMError::GpuError("mamba2_conv1d_step not loaded".into()))?;
                    let threads = 256u32;
                    let blocks = ((conv_dim as u32) + threads - 1) / threads;
                    let conv_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        kernel.launch(conv_cfg, (
                            conv_state as &mut CudaSlice<f32>, weights.conv1d_weight, &ws.zero_bias, &ws.qkv_t, &mut ws.conv_out,
                            conv_dim as i32, kernel_size as i32,
                        )).map_err(|e| LLMError::GpuError(format!("conv1d t={t}: {e}")))?;
                    }
                }

                // Split conv_out -> Q, K, V
                {
                    use cudarc::driver::DevicePtr;
                    let src_base = *DevicePtr::device_ptr(&ws.conv_out);
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.q_t), src_base, key_dim * f32_bytes);
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.k_t), src_base + (key_dim * f32_bytes) as u64, key_dim * f32_bytes);
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*DevicePtr::device_ptr(&ws.v_t), src_base + (2 * key_dim * f32_bytes) as u64, value_dim * f32_bytes);
                    }
                }

                // Extract a[t], b[t]
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.a_t), *DevicePtr::device_ptr(&a_raw) + (t * num_value_heads * f32_bytes) as u64, num_value_heads * f32_bytes);
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.b_t), *DevicePtr::device_ptr(&b_raw) + (t * num_value_heads * f32_bytes) as u64, num_value_heads * f32_bytes);
                    }
                }

                // Compute gates
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_compute_gates")
                        .ok_or_else(|| LLMError::GpuError("mamba2_compute_gates not loaded".into()))?;
                    let threads = 64u32;
                    let blocks = ((num_value_heads as u32) + threads - 1) / threads;
                    let gate_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        kernel.launch(gate_cfg, (
                            &ws.a_t, &ws.b_t, weights.a_log, weights.dt_bias, &mut ws.beta_t, &mut ws.g_t, num_value_heads as i32,
                        )).map_err(|e| LLMError::GpuError(format!("gates t={t}: {e}")))?;
                    }
                }

                // L2-normalize Q and K
                {
                    let threads = key_head_dim.min(128) as u32;
                    let l2_cfg = LaunchConfig { grid_dim: (num_key_heads as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: threads * 4 };
                    let kq = self.device.get_func("mamba2_ssm", "mamba2_l2_normalize")
                        .ok_or_else(|| LLMError::GpuError("mamba2_l2_normalize not loaded".into()))?;
                    unsafe {
                        kq.launch(l2_cfg, (&mut ws.q_t, num_key_heads as i32, key_head_dim as i32, 1e-6f32))
                            .map_err(|e| LLMError::GpuError(format!("l2norm Q t={t}: {e}")))?;
                    }
                    let kk = self.device.get_func("mamba2_ssm", "mamba2_l2_normalize")
                        .ok_or_else(|| LLMError::GpuError("mamba2_l2_normalize not loaded".into()))?;
                    unsafe {
                        kk.launch(l2_cfg, (&mut ws.k_t, num_key_heads as i32, key_head_dim as i32, 1e-6f32))
                            .map_err(|e| LLMError::GpuError(format!("l2norm K t={t}: {e}")))?;
                    }
                }

                // GQA expand
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_gqa_expand")
                        .ok_or_else(|| LLMError::GpuError("mamba2_gqa_expand not loaded".into()))?;
                    let total = (num_value_heads * key_head_dim) as u32;
                    let threads = 256u32;
                    let blocks = (total + threads - 1) / threads;
                    let gqa_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        kernel.launch(gqa_cfg, (&ws.q_t, &mut ws.q_exp, num_key_heads as i32, num_value_heads as i32, key_head_dim as i32))
                            .map_err(|e| LLMError::GpuError(format!("gqa Q t={t}: {e}")))?;
                    }
                }
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_gqa_expand")
                        .ok_or_else(|| LLMError::GpuError("mamba2_gqa_expand not loaded".into()))?;
                    let total = (num_value_heads * key_head_dim) as u32;
                    let threads = 256u32;
                    let blocks = (total + threads - 1) / threads;
                    let gqa_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        kernel.launch(gqa_cfg, (&ws.k_t, &mut ws.k_exp, num_key_heads as i32, num_value_heads as i32, key_head_dim as i32))
                            .map_err(|e| LLMError::GpuError(format!("gqa K t={t}: {e}")))?;
                    }
                }

                // SSM step
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_ssm_step")
                        .ok_or_else(|| LLMError::GpuError("mamba2_ssm_step not loaded".into()))?;
                    let threads = value_head_dim as u32;
                    let ssm_cfg = LaunchConfig {
                        grid_dim: (num_value_heads as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: (2 * value_head_dim * 4) as u32,
                    };
                    unsafe {
                        kernel.launch(ssm_cfg, (
                            ssm_state as &mut CudaSlice<f32>, &ws.q_exp, &ws.k_exp, &ws.v_t, &ws.beta_t, &ws.g_t, &mut ws.y_t,
                            num_value_heads as i32, key_head_dim as i32, value_head_dim as i32, scale,
                        )).map_err(|e| LLMError::GpuError(format!("ssm_step t={t}: {e}")))?;
                    }
                }

                // Gated RMSNorm
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&ws.z_t), *DevicePtr::device_ptr(&z) + (t * z_dim * f32_bytes) as u64, z_dim * f32_bytes);
                    }
                }
                {
                    let kernel = self.device.get_func("mamba2_ssm", "mamba2_norm_gate")
                        .ok_or_else(|| LLMError::GpuError("mamba2_norm_gate not loaded".into()))?;
                    let threads = value_head_dim.min(128) as u32;
                    let norm_cfg = LaunchConfig {
                        grid_dim: (num_value_heads as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 4 * 4 + 4,
                    };
                    unsafe {
                        kernel.launch(norm_cfg, (
                            &ws.y_t, &ws.z_t, weights.ssm_norm, &mut ws.norm_t,
                            num_value_heads as i32, value_head_dim as i32, 1e-6f32,
                        )).map_err(|e| LLMError::GpuError(format!("norm_gate t={t}: {e}")))?;
                    }
                }

                // Store norm_t into norm_out[t]
                {
                    use cudarc::driver::DevicePtr;
                    unsafe {
                        cudarc::driver::sys::lib().cuMemcpyDtoD_v2(
                            *DevicePtr::device_ptr(&norm_out) + (t * value_dim * f32_bytes) as u64,
                            *DevicePtr::device_ptr(&ws.norm_t), value_dim * f32_bytes);
                    }
                }
            }

            // 11. out_proj via sgemm
            let attn_proj = Self::linear(
                &self.device, blas, &norm_out, weights.out_proj,
                num_tokens, hidden, z_dim,
            )?;

            // 12. Residual
            let residual = Self::add_tensors(&self.device, input.hidden_states, &attn_proj, num_tokens * hidden)?;

            // 13. Post-attention RMSNorm
            let normed2 = Self::rms_norm_with_offset(&self.device, &residual, weights.post_attention_layernorm, cfg.rms_norm_eps, num_tokens, hidden, cfg.rms_norm_weight_offset)?;

            // 14. MLP via sgemm
            let gate = Self::linear(&self.device, blas, &normed2, weights.gate_proj, num_tokens, intermediate, hidden)?;
            let up = Self::linear(&self.device, blas, &normed2, weights.up_proj, num_tokens, intermediate, hidden)?;
            let fused = Self::fused_silu_mul(&self.device, &gate, &up, num_tokens * intermediate)?;
            let mlp_out = Self::linear(&self.device, blas, &fused, weights.down_proj, num_tokens, hidden, intermediate)?;

            // 15. Residual
            Self::add_tensors(&self.device, &residual, &mlp_out, num_tokens * hidden)
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
            let normed = Self::rms_norm_with_offset(
                &self.device,
                input.hidden_states,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                cfg.rms_norm_weight_offset,
            )?;

            // All ops on stream 0 -- no cross-stream sync needed

            // ---------------------------------------------------------------
            // 2. QKV projections via cuBLAS sgemm
            //    input [num_tokens, hidden] x weight^T [hidden, proj_dim]
            // ---------------------------------------------------------------
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // Qwen3.5: q_proj produces 2x q_dim when attn_output_gate is enabled
            let q_proj_dim = if cfg.has_attn_output_gate { q_dim * 2 } else { q_dim };

            let q_full = Self::linear(
                &self.device, blas, &normed, weights.q_proj,
                num_tokens, q_proj_dim, hidden,
            )?;

            // Split Q into query and gate if attn_output_gate is enabled
            let (mut q, attn_gate) = if cfg.has_attn_output_gate {
                let mut q_out = self.device.alloc_zeros::<f32>(num_tokens * q_dim)
                    .map_err(|e| LLMError::GpuError(format!("q split alloc: {e}")))?;
                let mut gate_out = self.device.alloc_zeros::<f32>(num_tokens * q_dim)
                    .map_err(|e| LLMError::GpuError(format!("gate split alloc: {e}")))?;

                let threads = 256u32;
                let total = (num_tokens * q_dim) as u32;
                let blocks = (total + threads - 1) / threads;
                let launch_cfg = LaunchConfig {
                    grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0,
                };

                let trunc_kernel = self.device.get_func("attn_gate", "truncate_q_kernel")
                    .ok_or_else(|| LLMError::GpuError("truncate_q_kernel not loaded".into()))?;
                unsafe {
                    trunc_kernel.launch(launch_cfg, (
                        &mut q_out, &q_full, num_tokens as i32, q_dim as i32, head_dim as i32,
                    )).map_err(|e| LLMError::GpuError(format!("truncate_q launch: {e}")))?;
                }

                let split_kernel = self.device.get_func("attn_gate", "split_gate_kernel")
                    .ok_or_else(|| LLMError::GpuError("split_gate_kernel not loaded".into()))?;
                unsafe {
                    split_kernel.launch(launch_cfg, (
                        &mut gate_out, &q_full, num_tokens as i32, q_dim as i32, head_dim as i32,
                    )).map_err(|e| LLMError::GpuError(format!("split_gate launch: {e}")))?;
                }

                (q_out, Some(gate_out))
            } else {
                (q_full, None)
            };

            let mut k = Self::linear(
                &self.device, blas, &normed, weights.k_proj,
                num_tokens, kv_dim, hidden,
            )?;
            let mut v = Self::linear(
                &self.device, blas, &normed, weights.v_proj,
                num_tokens, kv_dim, hidden,
            )?;

            // Apply QKV biases if present (e.g. Qwen2.5)
            if let Some(bias) = weights.q_proj_bias {
                Self::add_bias(&self.device, &mut q, bias, num_tokens, q_dim)?;
            }
            if let Some(bias) = weights.k_proj_bias {
                Self::add_bias(&self.device, &mut k, bias, num_tokens, kv_dim)?;
            }
            if let Some(bias) = weights.v_proj_bias {
                Self::add_bias(&self.device, &mut v, bias, num_tokens, kv_dim)?;
            }

            // QK per-head RMSNorm (Qwen3.5)
            if let Some(q_norm_w) = weights.q_norm {
                q = Self::rms_norm(
                    &self.device, &q, q_norm_w, cfg.rms_norm_eps,
                    num_tokens * num_heads, head_dim,
                )?;
            }
            if let Some(k_norm_w) = weights.k_norm {
                k = Self::rms_norm(
                    &self.device, &k, k_norm_w, cfg.rms_norm_eps,
                    num_tokens * num_kv_heads, head_dim,
                )?;
            }

            // ---------------------------------------------------------------
            // 3. RoPE on Q and K
            // ---------------------------------------------------------------
            let (q_rot, k_rot) = Self::apply_rotary_embedding(
                &self.device,
                &q,
                &k,
                input.positions,
                input.rope_cos,
                input.rope_sin,
                num_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
                cfg.rotary_dim,
            )?;

            // ---------------------------------------------------------------
            // 4. KV cache write + Attention (prefill vs decode)
            // ---------------------------------------------------------------
            // Always write K/V into paged cache via slot_mapping
            trace!(layer = cfg.layer_idx, "gpu_layer: cache_write start");
            Self::cache_write(
                &self.device,
                &k_rot,
                &v,
                input.key_cache,
                input.value_cache,
                input.slot_mapping,
                num_tokens,
                num_kv_heads,
                head_dim,
            )?;

            trace!(layer = cfg.layer_idx, "gpu_layer: cache_write done");

            let attn_out = if input.is_prefill {
                // Prefill: use FA2 prefill kernel reading from paged cache
                trace!(layer = cfg.layer_idx, "gpu_layer: prefill_attention start");
                Self::prefill_attention(
                    &self.device,
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
                trace!(layer = cfg.layer_idx, "gpu_layer: decode_attention start");
                Self::decode_attention(
                    &self.device,
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

            // 4b. Apply output gate: attn_out = attn_out * sigmoid(gate)
            let gated_attn_out = if let Some(ref gate) = attn_gate {
                let n = num_tokens * q_dim;
                let mut gated = self.device.alloc_zeros::<f32>(n)
                    .map_err(|e| LLMError::GpuError(format!("sigmoid_gate alloc: {e}")))?;
                let threads = 256u32;
                let blocks = ((n as u32) + threads - 1) / threads;
                let launch_cfg = LaunchConfig {
                    grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0,
                };
                let kernel = self.device.get_func("attn_gate", "sigmoid_gate_kernel")
                    .ok_or_else(|| LLMError::GpuError("sigmoid_gate_kernel not loaded".into()))?;
                unsafe {
                    kernel.launch(launch_cfg, (
                        &mut gated, &attn_out, gate, n as i32,
                    )).map_err(|e| LLMError::GpuError(format!("sigmoid_gate launch: {e}")))?;
                }
                gated
            } else {
                attn_out
            };

            // ---------------------------------------------------------------
            // 5. Output projection
            // ---------------------------------------------------------------
            let attn_proj = Self::linear(
                &self.device,
                blas,
                &gated_attn_out,
                weights.o_proj,
                num_tokens,
                hidden,
                q_dim,
            )?;

            // ---------------------------------------------------------------
            // Residual: hidden_states + attn_proj
            // ---------------------------------------------------------------
            let residual = Self::add_tensors(
                &self.device,
                input.hidden_states,
                &attn_proj,
                num_tokens * hidden,
            )?;

            // ---------------------------------------------------------------
            // 6. Post-attention RMSNorm
            // ---------------------------------------------------------------
            let normed2 = Self::rms_norm_with_offset(
                &self.device,
                &residual,
                weights.post_attention_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
                cfg.rms_norm_weight_offset,
            )?;

            // ---------------------------------------------------------------
            // 7. MLP: gate_proj + up_proj -> fused_silu_mul -> down_proj
            // ---------------------------------------------------------------
            let gate = Self::linear(
                &self.device,
                blas,
                &normed2,
                weights.gate_proj,
                num_tokens,
                intermediate,
                hidden,
            )?;
            let up = Self::linear(
                &self.device,
                blas,
                &normed2,
                weights.up_proj,
                num_tokens,
                intermediate,
                hidden,
            )?;

            let fused = Self::fused_silu_mul(&self.device, &gate, &up, num_tokens * intermediate)?;

            let mlp_out = Self::linear(
                &self.device,
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
            let output = Self::add_tensors(&self.device, &residual, &mlp_out, num_tokens * hidden)?;

            Ok(output)
        }

        // ===================================================================
        // Private dispatch helpers
        //
        // Each wraps the corresponding CUDA kernel or cuBLAS call.
        // These are the seams where Agent 2-7 implementations plug in.
        // ===================================================================

        /// RMSNorm: out[i] = (x[i] / rms) * (weight[i] + weight_offset)
        /// where rms = sqrt(mean(x^2) + eps).
        ///
        /// weight_offset = 0.0 for standard RMSNorm (Llama, etc.)
        /// weight_offset = 1.0 for Qwen3.5 "(1 + weight)" style RMSNorm
        ///
        /// Dispatches to the rms_norm CUDA kernel.
        pub(crate) fn rms_norm(
            device: &Arc<CudaDevice>,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f32>> {
            Self::rms_norm_with_offset(device, input, weight, eps, num_tokens, hidden_size, 0.0)
        }

        /// RMSNorm with configurable weight offset.
        pub(crate) fn rms_norm_with_offset(
            device: &Arc<CudaDevice>,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
            weight_offset: f32,
        ) -> Result<CudaSlice<f32>> {
            let n = num_tokens * hidden_size;
            let mut output = device
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
            let kernel = device.get_func(module_name, func_name).ok_or_else(|| {
                LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded"))
            })?;
            unsafe {
                kernel
                    .launch(cfg, (&mut output, input, weight, eps, hidden_size as i32, weight_offset))
                    .map_err(|e| LLMError::GpuError(format!("rms_norm launch failed: {e}")))?;
            }

            Ok(output)
        }

        /// RMSNorm writing into a pre-allocated output buffer (no cudaMalloc).
        fn rms_norm_into(
            device: &Arc<CudaDevice>,
            output: &mut CudaSlice<f32>,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
            weight_offset: f32,
        ) -> Result<()> {
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = device.get_func("rms_norm", "rms_norm_kernel").ok_or_else(|| {
                LLMError::GpuError("kernel rms_norm::rms_norm_kernel not loaded".into())
            })?;
            unsafe {
                kernel
                    .launch(cfg, (output as &mut CudaSlice<f32>, input, weight, eps, hidden_size as i32, weight_offset))
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_into launch: {e}")))?;
            }
            Ok(())
        }

        /// Element-wise add writing into a pre-allocated output buffer (no cudaMalloc).
        fn add_tensors_into(
            device: &Arc<CudaDevice>,
            output: &mut CudaSlice<f32>,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
            n: usize,
        ) -> Result<()> {
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = device
                .get_func("add_bias", "add_kernel")
                .or_else(|| device.get_func("activation", "add_kernel"))
                .ok_or_else(|| LLMError::GpuError("add_kernel not loaded".into()))?;
            unsafe {
                kernel.launch(cfg, (output as &mut CudaSlice<f32>, a, b, n as i32))
                    .map_err(|e| LLMError::GpuError(format!("add_tensors_into launch: {e}")))?;
            }
            Ok(())
        }

        /// Fused SiLU*mul writing into a pre-allocated output buffer (no cudaMalloc).
        fn fused_silu_mul_into(
            device: &Arc<CudaDevice>,
            output: &mut CudaSlice<f32>,
            gate: &CudaSlice<f32>,
            up: &CudaSlice<f32>,
            n: usize,
        ) -> Result<()> {
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = device.get_func("activation", "fused_silu_mul_kernel").ok_or_else(|| {
                LLMError::GpuError("fused_silu_mul_kernel not loaded".into())
            })?;
            unsafe {
                kernel
                    .launch(cfg, (output as &mut CudaSlice<f32>, gate, up, n as i32))
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_into launch: {e}")))?;
            }
            Ok(())
        }

        /// Linear projection writing into a pre-allocated output buffer (no cudaMalloc).
        fn linear_into(
            _device: &Arc<CudaDevice>,
            blas: &CublasHandle,
            output: &mut CudaSlice<f32>,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<()> {
            blas.sgemm(m, n, k, 1.0, input, weight, 0.0, output)?;
            Ok(())
        }

        /// Round f32 values to BF16 precision in-place.
        ///
        /// This keeps f32 computation on the same numerical trajectory as
        /// BF16 inference (matching HuggingFace / PyTorch BF16 behavior).
        /// Without this, f32/f16 compute diverges from BF16 over 64 layers,
        /// producing garbled output.
        fn round_to_bf16(
            _device: &Arc<CudaDevice>,
            _data: &mut CudaSlice<f32>,
            _n: usize,
        ) -> Result<()> {
            // DISABLED: round_to_bf16 adds unnecessary quantization noise.
            // BF16 precision is already applied at GEMM boundaries (forward_once_bf16).
            // Intermediate f32 computation matches how PyTorch BF16 autocast works.
            Ok(())
        }

        /// Linear projection via cuBLAS sgemm.
        /// Computes output = input @ weight^T where:
        ///   input: [m, k], weight: [n, k] (row-major), output: [m, n].
        /// Add bias in-place: tensor[i*dim + j] += bias[j]
        fn add_bias(
            device: &Arc<CudaDevice>,
            tensor: &mut CudaSlice<f32>,
            bias: &CudaSlice<f32>,
            num_tokens: usize,
            dim: usize,
        ) -> Result<()> {
            let kernel = device
                .get_func("add_bias", "add_bias_kernel")
                .ok_or_else(|| LLMError::GpuError("add_bias_kernel not loaded".into()))?;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernel
                    .launch(cfg, (tensor as &mut CudaSlice<f32>, bias, dim as i32))
                    .map_err(|e| LLMError::GpuError(format!("add_bias launch: {e}")))?;
            }
            Ok(())
        }

        fn linear(
            device: &Arc<CudaDevice>,
            blas: &CublasHandle,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = device
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
            device: &Arc<CudaDevice>,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            positions: &CudaSlice<i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            rotary_dim: usize,
        ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
            let q_len = num_tokens * num_heads * head_dim;
            let k_len = num_tokens * num_kv_heads * head_dim;

            // Clone Q and K so we can apply rotation in-place.
            // Non-rotated dimensions pass through unchanged from this copy.
            let mut q_out = device
                .alloc_zeros::<f32>(q_len)
                .map_err(|e| LLMError::GpuError(format!("rope q alloc: {e}")))?;
            let mut k_out = device
                .alloc_zeros::<f32>(k_len)
                .map_err(|e| LLMError::GpuError(format!("rope k alloc: {e}")))?;

            device
                .dtod_copy(q, &mut q_out)
                .map_err(|e| LLMError::GpuError(format!("rope q copy: {e}")))?;
            device
                .dtod_copy(k, &mut k_out)
                .map_err(|e| LLMError::GpuError(format!("rope k copy: {e}")))?;

            let kernel = device
                .get_func("rotary_embedding", "rotary_embedding_kernel")
                .ok_or_else(|| LLMError::GpuError("rotary_embedding_kernel not loaded".into()))?;

            // Only rotate `rotary_dim` dimensions; rest pass through from copy above.
            let half_rotary = rotary_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_rotary.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            &mut q_out,
                            &mut k_out,
                            rope_cos,
                            rope_sin,
                            positions,
                            num_tokens as i32,
                            num_heads as i32,
                            num_kv_heads as i32,
                            head_dim as i32,
                            rotary_dim as i32,
                        ),
                    )
                    .map_err(|e| LLMError::GpuError(format!("rope launch failed: {e}")))?;
            }

            Ok((q_out, k_out))
        }

        /// Apply RoPE into pre-allocated output buffers (no cudaMalloc).
        #[allow(clippy::too_many_arguments)]
        fn apply_rotary_embedding_into(
            device: &Arc<CudaDevice>,
            q_out: &mut CudaSlice<f32>,
            k_out: &mut CudaSlice<f32>,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            positions: &CudaSlice<i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            rotary_dim: usize,
        ) -> Result<()> {
            device
                .dtod_copy(q, q_out)
                .map_err(|e| LLMError::GpuError(format!("rope q copy: {e}")))?;
            device
                .dtod_copy(k, k_out)
                .map_err(|e| LLMError::GpuError(format!("rope k copy: {e}")))?;

            let kernel = device
                .get_func("rotary_embedding", "rotary_embedding_kernel")
                .ok_or_else(|| LLMError::GpuError("rotary_embedding_kernel not loaded".into()))?;

            let half_rotary = rotary_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_rotary.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            q_out as &mut CudaSlice<f32>,
                            k_out as &mut CudaSlice<f32>,
                            rope_cos,
                            rope_sin,
                            positions,
                            num_tokens as i32,
                            num_heads as i32,
                            num_kv_heads as i32,
                            head_dim as i32,
                            rotary_dim as i32,
                        ),
                    )
                    .map_err(|e| LLMError::GpuError(format!("rope launch failed: {e}")))?;
            }

            Ok(())
        }

        /// Paged attention forward pass.
        ///
        /// Writes new K,V into the f16 cache at slot_mapping positions.
        /// Input k/v are f32 (from projection); the kernel converts to f16 on write.
        /// Uses reshape_and_cache_f16_kernel: 1 launch per layer.
        #[allow(clippy::too_many_arguments)]
        fn cache_write(
            device: &Arc<CudaDevice>,
            k: &CudaSlice<f32>,
            v: &CudaSlice<f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            slot_mapping: &CudaSlice<i32>,
            num_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kernel = device
                .get_func("reshape_and_cache", "reshape_and_cache_f16_kernel")
                .ok_or_else(|| LLMError::GpuError("reshape_and_cache_f16_kernel not loaded".into()))?;

            let kv_dim = num_kv_heads * head_dim;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (kv_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // Kernel: (f16* key_cache, f16* value_cache, f32* key, f32* value, int* slot_mapping, int, int, int)
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            key_cache,
                            value_cache,
                            k,
                            v,
                            slot_mapping,
                            num_tokens as i32,
                            num_kv_heads as i32,
                            head_dim as i32,
                        ),
                    )
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
            device: &Arc<CudaDevice>,
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
            let mut output = device
                .alloc_zeros::<f32>(num_tokens * q_stride)
                .map_err(|e| LLMError::GpuError(format!("naive attn output alloc: {e}")))?;

            // Per-head attention via cuBLAS
            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;

                // Extract Q_head [num_tokens, head_dim] from Q [num_tokens, num_heads * head_dim]
                // Extract K_head [num_tokens, head_dim] from K [num_tokens, num_kv_heads * head_dim]
                // Extract V_head [num_tokens, head_dim] from V [num_tokens, num_kv_heads * head_dim]
                // Use CPU gather for correctness (not perf-critical for prefill)
                let q_all: Vec<f32> = device
                    .dtoh_sync_copy(q)
                    .map_err(|e| LLMError::GpuError(format!("naive attn q DtoH: {e}")))?;
                let k_all: Vec<f32> = device
                    .dtoh_sync_copy(k)
                    .map_err(|e| LLMError::GpuError(format!("naive attn k DtoH: {e}")))?;
                let v_all: Vec<f32> = device
                    .dtoh_sync_copy(v)
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
                let mut out_all: Vec<f32> = device
                    .dtoh_sync_copy(&output)
                    .map_err(|e| LLMError::GpuError(format!("naive attn out DtoH: {e}")))?;
                for t in 0..num_tokens {
                    for d in 0..head_dim {
                        out_all[t * q_stride + h * head_dim + d] = out_head[t * head_dim + d];
                    }
                }
                output = device
                    .htod_sync_copy(&out_all)
                    .map_err(|e| LLMError::GpuError(format!("naive attn out HtoD: {e}")))?;
            }

            Ok(output)
        }

        /// FA2 prefill attention reading from f16 paged cache with real block_tables.
        /// Q is f32, cache is f16; the kernel loads f16 and promotes to f32 internally.
        fn prefill_attention(
            device: &Arc<CudaDevice>,
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
            let output = device
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("prefill_attn alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA2_BC: usize = 32;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let kernel = device
                .get_func("flash_attention", "flash_attention_2_f16kv_kernel")
                .ok_or_else(|| LLMError::GpuError("flash_attention_2_f16kv_kernel not loaded".into()))?;

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

            // FA2 prefill kernel: 16 args, use raw void** launch (exceeds tuple limit)
            unsafe {
                use cudarc::driver::DevicePtr;
                let mut out_ptr = *DevicePtr::device_ptr(&output);
                let mut q_ptr = *DevicePtr::device_ptr(q);
                let mut kc_ptr = *DevicePtr::device_ptr(key_cache);
                let mut vc_ptr = *DevicePtr::device_ptr(value_cache);
                let mut bt_ptr = *DevicePtr::device_ptr(block_tables);
                let mut cl_ptr = *DevicePtr::device_ptr(context_lens);
                let mut ss_ptr = *DevicePtr::device_ptr(seq_start_pos);
                let mut p_scale = scale;
                let mut p_num_heads = num_heads as i32;
                let mut p_num_kv = num_kv_heads as i32;
                let mut p_head_dim = head_dim as i32;
                let mut p_block_size = block_size as i32;
                let mut p_max_ctx = max_context_len as i32;
                let mut p_max_blocks = max_blocks_per_seq;
                let mut p_num_tokens = num_tokens as i32;
                let mut p_causal = 1i32;
                let params: &mut [*mut std::ffi::c_void] = &mut [
                    &mut out_ptr as *mut _ as *mut _,
                    &mut q_ptr as *mut _ as *mut _,
                    &mut kc_ptr as *mut _ as *mut _,
                    &mut vc_ptr as *mut _ as *mut _,
                    &mut bt_ptr as *mut _ as *mut _,
                    &mut cl_ptr as *mut _ as *mut _,
                    &mut ss_ptr as *mut _ as *mut _,
                    &mut p_scale as *mut _ as *mut _,
                    &mut p_num_heads as *mut _ as *mut _,
                    &mut p_num_kv as *mut _ as *mut _,
                    &mut p_head_dim as *mut _ as *mut _,
                    &mut p_block_size as *mut _ as *mut _,
                    &mut p_max_ctx as *mut _ as *mut _,
                    &mut p_max_blocks as *mut _ as *mut _,
                    &mut p_num_tokens as *mut _ as *mut _,
                    &mut p_causal as *mut _ as *mut _,
                ];
                kernel
                    .launch(cfg, params)
                    .map_err(|e| LLMError::GpuError(format!("prefill FA2 launch: {e}")))?;
            }

            Ok(output)
        }

        /// Decode attention: read f16 K/V from paged cache, one FA2 decode kernel per layer.
        /// Q is f32, cache is f16; kernel promotes f16 to f32 on load.
        fn decode_attention(
            device: &Arc<CudaDevice>,
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
            let mut output = device
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("paged_attn alloc failed: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            // Use FA2 decode kernel: correct block-level reductions, GQA support.
            // FA2_BC=64, FA2_THREADS=128 (compile-time constants in flash_attention.cu)
            const FA2_BC: usize = 32;
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

            let kernel = device.get_func(module_name, func_name).ok_or_else(|| {
                LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded"))
            })?;

            // Opt into extended shared memory if needed
            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("decode FA2 set max shared mem: {e}")))?;
            }

            // SAFETY: All slices are valid GPU memory on this device.
            // output: [num_seqs, num_heads, head_dim]
            // q:      [num_seqs, num_heads, head_dim]  (decode: num_seqs == num_tokens)
            // key_cache, value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            // block_tables: [num_seqs, max_blocks_per_seq]
            // context_lens: [num_seqs]
            // Scalar int args cast from usize; all values fit in i32 range.
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            &mut output,
                            q,
                            key_cache,
                            value_cache,
                            block_tables,
                            context_lens,
                            scale,
                            num_heads as i32,
                            num_kv_heads as i32,
                            head_dim as i32,
                            block_size as i32,
                            // max_blocks_per_seq: block_tables row width
                            (block_tables.len() / num_seqs.max(1)) as i32,
                        ),
                    )
                    .map_err(|e| {
                        LLMError::GpuError(format!("flash_attention_2_decode launch failed: {e}"))
                    })?;
            }

            Ok(output)
        }

        /// Decode attention into pre-allocated output buffer (no cudaMalloc).
        #[allow(clippy::too_many_arguments)]
        fn decode_attention_into(
            device: &Arc<CudaDevice>,
            output: &mut CudaSlice<f32>,
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
        ) -> Result<()> {
            // Zero the output buffer
            device.memset_zeros(output)
                .map_err(|e| LLMError::GpuError(format!("decode_attn_into memset: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA2_BC: usize = 32;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let module_name = "flash_attention";
            let func_name = "flash_attention_2_decode_f16kv_kernel";

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            let kernel = device.get_func(module_name, func_name).ok_or_else(|| {
                LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded"))
            })?;

            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("decode FA2 set max shared mem: {e}")))?;
            }

            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            output as &mut CudaSlice<f32>,
                            q,
                            key_cache,
                            value_cache,
                            block_tables,
                            context_lens,
                            scale,
                            num_heads as i32,
                            num_kv_heads as i32,
                            head_dim as i32,
                            block_size as i32,
                            (block_tables.len() / num_seqs.max(1)) as i32,
                        ),
                    )
                    .map_err(|e| {
                        LLMError::GpuError(format!("flash_attention_2_decode launch failed: {e}"))
                    })?;
            }

            Ok(())
        }

        /// Fused SiLU activation with element-wise multiply: out = silu(gate) * up.
        ///
        /// Dispatches to the activation CUDA kernel.
        pub(crate) fn fused_silu_mul(
            device: &Arc<CudaDevice>,
            gate: &CudaSlice<f32>,
            up: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = device
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

            let kernel = device.get_func(module_name, func_name).ok_or_else(|| {
                LLMError::GpuError(format!("kernel {module_name}::{func_name} not loaded"))
            })?;

            // SAFETY: gate, up, and output all have exactly n elements.
            // Grid covers all elements with ceil division.
            unsafe {
                kernel
                    .launch(cfg, (&mut output, gate, up, n as i32))
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
        pub(crate) fn add_tensors(
            device: &Arc<CudaDevice>,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = device
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("add_tensors alloc failed: {e}")))?;

            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            // Try dedicated add_bias module first, then activation module
            let kernel = device
                .get_func("add_bias", "add_kernel")
                .or_else(|| device.get_func("activation", "add_kernel"));

            match kernel {
                Some(k) => {
                    // SAFETY: a, b, output all have exactly n elements.
                    unsafe {
                        k.launch(cfg, (&mut output, a, b, n as i32)).map_err(|e| {
                            LLMError::GpuError(format!("add_kernel launch failed: {e}"))
                        })?;
                    }
                }
                None => {
                    // Fallback: CPU add (only until kernels are compiled).
                    let a_host = device
                        .dtoh_sync_copy(a)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh a failed: {e}")))?;
                    let b_host = device
                        .dtoh_sync_copy(b)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh b failed: {e}")))?;
                    let sum: Vec<f32> = a_host
                        .iter()
                        .zip(b_host.iter())
                        .map(|(x, y)| x + y)
                        .collect();
                    output = device
                        .htod_sync_copy(&sum)
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
