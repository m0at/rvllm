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

use crate::weights::{F16Weight, Fp8Weight};

#[derive(Debug)]
pub struct Gemma4LayerWeights {
    pub qkv: Fp8Weight,
    pub o_proj: Fp8Weight,
    pub gate_up: Fp8Weight,
    pub down_proj: Fp8Weight,
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
    pub pli_gate: Option<F16Weight>,
    pub pli_projection: Option<F16Weight>,
    pub post_pli_norm: Option<F16Weight>,
    pub kv_shared: bool,
    pub kv_source_layer: usize,
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
    pub pli_embed: Option<F16Weight>,
    pub pli_model_projection: Option<F16Weight>,
    pub pli_projection_norm: Option<F16Weight>,
}
