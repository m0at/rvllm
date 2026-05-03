//! Qwen 3.6 35B-A3B weight structures (Phase 1: outside-layers only).
//!
//! Phase 1 scope is the three "outside-the-stack" tensors:
//!   - `model.language_model.embed_tokens.weight`
//!   - `model.language_model.norm.weight`
//!   - `lm_head.weight` (NOT tied — Qwen 3.6 ships a separate lm_head)
//!
//! Per-layer weights (Q/K/V/O + Q-Norm + K-Norm + 256 expert G/U/D +
//! shared-expert + router + linear-attn state matrices) land in
//! Phase 2/3 alongside the corresponding kernels.
//!
//! Storage convention: all three outside tensors are bf16 → uploaded
//! as f16 (existing arena infra is f16-native; bf16→f16 conversion in
//! `tensor_to_f16_bytes` lives in `load.rs`).

use crate::weights::{F16Weight, Fp8Weight};

#[derive(Debug)]
pub struct Qwen36LoadedOutside {
    pub embed_tokens: F16Weight,
    pub final_norm: F16Weight,
    pub lm_head: F16Weight,
    pub embed_tokens_bytes: u64,
    pub final_norm_bytes: u64,
    pub lm_head_bytes: u64,
}

/// Full-attention layer weights (Phase 2a).
///
/// 10 of the 40 Qwen 3.6 35B-A3B layers carry standard self-attention
/// (the `full_attention` entries in `layer_types`). Each holds:
///   - 2× layer RMSNorm (input, post_attention)
///   - 2× per-head RMSNorm (q_norm, k_norm) — distinct from Gemma 4's
///     fused QK-norm
///   - 4× FP8 blockwise-scaled projections (Q/K/V/O) with their
///     `weight_scale_inv` 2-D tables ([N/128, K/128])
///
/// MoE block (256-expert + shared) lives on EVERY layer — both
/// linear and full — and is loaded by a separate Phase-2b loader.
#[derive(Debug)]
pub struct Qwen36FullAttnLayer {
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub q_norm: F16Weight,
    pub k_norm: F16Weight,
    pub q_proj: Fp8Weight,
    pub k_proj: Fp8Weight,
    pub v_proj: Fp8Weight,
    pub o_proj: Fp8Weight,
}

/// Qwen 3.6 loaded model, growing per-phase.
///
/// Phase 1: `outside` populated.
/// Phase 2a: `full_attn_layers` populated for the 10 full-attention
/// layer indices; other slots are `None`.
/// Phase 2b/3: linear-attention + MoE expansions land in additional
/// `Option<...>` fields rather than reshuffling existing ones.
#[derive(Debug)]
pub struct Qwen36LoadedModel {
    pub outside: Qwen36LoadedOutside,
    /// Indexed by absolute layer index (0..num_hidden_layers). `Some`
    /// for full-attention layers, `None` for linear-attention layers
    /// (which carry no `self_attn.*` tensors).
    pub full_attn_layers: Vec<Option<Qwen36FullAttnLayer>>,
}
