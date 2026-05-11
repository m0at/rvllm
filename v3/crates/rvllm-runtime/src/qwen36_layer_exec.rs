//! Qwen 3.6 per-layer forward pass (Phase 2/3 placeholder).
//!
//! Phase 0: empty. Phase 2 adds `forward_full_attention_layer` (10 of
//! 40 layers, with Q/K-Norm + RoPE + paged-attn + attn_output_gate +
//! MoE-block). Phase 3 adds `forward_linear_attention_layer` (30 of 40
//! layers, gated-deltanet / linear-recurrent attention).

#![allow(dead_code)]
