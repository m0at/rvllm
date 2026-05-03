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

use crate::weights::F16Weight;

#[derive(Debug)]
pub struct Qwen36LoadedOutside {
    pub embed_tokens: F16Weight,
    pub final_norm: F16Weight,
    pub lm_head: F16Weight,
    pub embed_tokens_bytes: u64,
    pub final_norm_bytes: u64,
    pub lm_head_bytes: u64,
}
