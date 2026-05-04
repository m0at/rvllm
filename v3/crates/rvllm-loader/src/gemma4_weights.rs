//! Gemma 4 weight structures.
//!
//! Sliding and global layers do NOT share identical attention weight shapes.
//! Sliding layers use `(q, k, v, o) = (8192, 4096, 4096, 8192)` over the
//! head axis, while global layers use `(16384, 2048, no v_proj, 16384)`.
//! Global attention has `attention_k_eq_v=true`, so the K projection is
//! reused for V when building the fused QKV weight.
//!
//! Per-layer extras vs Llama/Qwen:
//!   - 4 norms (input, post_attn, pre_ff, post_ff)
//!   - QK-norm gammas (q_norm [256], k_norm [256])
//!   - layer_scalar [1] (per-layer residual multiplier)
//!
//! Sliding layer shapes:
//!   q_proj:        [8192, 5376]
//!   k_proj:        [4096, 5376]
//!   v_proj:        [4096, 5376]
//!   o_proj:        [5376, 8192]
//!
//! Global layer shapes:
//!   q_proj:        [16384, 5376]
//!   k_proj:        [2048, 5376]
//!   v_proj:        absent, reuse `k_proj`
//!   o_proj:        [5376, 16384]
//!
//! Shared MLP / norm shapes:
//!   gate_proj:     [21504, 5376]
//!   up_proj:       [21504, 5376]
//!   down_proj:     [5376, 21504]
//!   q_norm:        [256]
//!   k_norm:        [256]
//!   layer_scalar:  [1]
//!   *_layernorm:   [5376]

use crate::weights::{AwqLayerWeights, F16Weight, Fp8Weight};

#[derive(Debug)]
pub struct Gemma4LayerWeights {
    /// FP8 attention QKV weights. `None` when this layer is AWQ-quantized
    /// (the `awq` field below carries Q/K/V instead). At least one of
    /// `qkv` / `qkv_f16` / `awq.q_proj` must be `Some` for the layer
    /// to execute; bring-up reflects that as a non-zero pointer in
    /// the corresponding `Gemma4LayerWeightPtrs` slot.
    ///
    /// Cycle 48 step 7a: made Optional to support AWQ-only layers
    /// without forcing dummy FP8 weights to occupy ~half the GPU
    /// arena.
    pub qkv: Option<Fp8Weight>,
    pub o_proj: Option<Fp8Weight>,
    pub gate_up: Option<Fp8Weight>,
    pub down_proj: Option<Fp8Weight>,
    pub qkv_f16: Option<F16Weight>,
    pub o_proj_f16: Option<F16Weight>,
    pub gate_up_f16: Option<F16Weight>,
    pub down_proj_f16: Option<F16Weight>,
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub pre_feedforward_layernorm: F16Weight,
    pub post_feedforward_layernorm: F16Weight,
    pub q_norm: F16Weight,
    pub k_norm: F16Weight,
    pub layer_scalar: F16Weight,
    /// Cycle 46 step 5c: optional AWQ INT4 W4A16 weights for this layer.
    /// `None` = FP8 path stays in charge for every linear in the layer
    /// (no behavior change for non-AWQ checkpoints). `Some` = the seven
    /// projections (q/k/v/o + gate/up/down) have AWQ tensors uploaded
    /// via `compressed_tensors::upload_gemma4_awq_layer`; bring-up
    /// reads this and populates `Gemma4AwqLayerPtrs` so exec_layer
    /// dispatches through `awq_int4_gemv_f16_kernel`.
    ///
    /// Populated by an AWQ-aware load path that is wired in cycle 47;
    /// the field is added here so the runtime crate can already
    /// thread it through bring-up without a follow-up data-shape
    /// change.
    pub awq: Option<AwqLayerWeights>,
}

#[derive(Debug)]
pub struct Gemma4LoadedModel {
    pub embedding: F16Weight,
    pub lm_head_fp8: Fp8Weight,
    pub lm_head_f16: F16Weight,
    pub final_norm: F16Weight,
    /// Sliding layers: theta=10000, full rotation (rotary_dim=256)
    pub rope_cos_sliding: F16Weight,
    pub rope_sin_sliding: F16Weight,
    /// Global layers: theta=1M, partial rotation (rotary_dim=128 of head_dim=512)
    pub rope_cos_global: F16Weight,
    pub rope_sin_global: F16Weight,
    pub layers: Vec<Gemma4LayerWeights>,
    /// Vision tower (SigLIP-style ViT) for Gemma 4 multimodal.
    /// `None` for text-only checkpoints.
    pub vision: Option<Gemma4Vision>,
}

// ─── Vision (Gemma 4 SigLIP-style ViT) ────────────────────────────────

/// Geometry (verified via state-dict):
/// - patch_embedder.input_proj.weight: [1152, 768] (16² * 3 → hidden=1152)
/// - patch_embedder.position_embedding_table: [2, 10240, 1152] f16
///   (axis 0 indexes row vs col, axis 1 picks position 0..10239)
/// - 27 encoder layers, each carries:
///     input_layernorm.weight (RMSNorm, Gemma-style +1 shift)
///     post_attention_layernorm.weight
///     pre_feedforward_layernorm.weight
///     post_feedforward_layernorm.weight
///     self_attn.q_proj.linear.weight [1152, 1152] (NO bias; attention_bias=False)
///     self_attn.k_proj.linear.weight [1152, 1152]
///     self_attn.v_proj.linear.weight [1152, 1152]
///     self_attn.o_proj.linear.weight [1152, 1152]
///     self_attn.q_norm.weight [72] (per-head RMSNorm, Gemma-style +1)
///     self_attn.k_norm.weight [72]
///     mlp.gate_proj.linear.weight [4304, 1152]
///     mlp.up_proj.linear.weight   [4304, 1152]
///     mlp.down_proj.linear.weight [1152, 4304]
/// - std_bias, std_scale: [1152] applied after the encoder output,
///   before the embed_vision projection (Gemma vision_config
///   `standardize=True`).
/// - embed_vision.embedding_projection.weight: [5376, 1152]
///   post-pool linear from vision-hidden to text-hidden.
#[derive(Debug)]
pub struct Gemma4Vision {
    pub patch_embedder_input_proj: F16Weight, // [1152, 768]
    pub patch_embedder_pos_table: F16Weight,  // [2, 10240, 1152]
    pub blocks: Vec<Gemma4VisionBlock>,
    pub std_bias: F16Weight,                   // [1152]
    pub std_scale: F16Weight,                  // [1152]
    pub embed_vision_projection: F16Weight,    // [5376, 1152]
}

#[derive(Debug)]
pub struct Gemma4VisionBlock {
    pub input_layernorm_w: F16Weight,
    pub post_attention_layernorm_w: F16Weight,
    pub pre_feedforward_layernorm_w: F16Weight,
    pub post_feedforward_layernorm_w: F16Weight,
    pub q_proj_w: F16Weight,
    pub k_proj_w: F16Weight,
    pub v_proj_w: F16Weight,
    pub o_proj_w: F16Weight,
    pub q_norm_w: F16Weight, // [72]
    pub k_norm_w: F16Weight, // [72]
    pub gate_proj_w: F16Weight,
    pub up_proj_w: F16Weight,
    pub down_proj_w: F16Weight,
}
