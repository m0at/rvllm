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
    /// f16 mirror of lm_head (Phase 1) — kept for any future bf16
    /// matmul fallback. The fp8_gemv kernel path consumes
    /// `lm_head_fp8` instead.
    pub lm_head: F16Weight,
    /// CPU-quantized FP8 mirror of lm_head (Phase 3d). Per-tensor
    /// scale; consumed by the `fp8_gemv` kernel in the outside-only
    /// forward path.
    pub lm_head_fp8: Fp8Weight,
    pub embed_tokens_bytes: u64,
    pub final_norm_bytes: u64,
    pub lm_head_bytes: u64,
    pub lm_head_fp8_bytes: u64,
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

/// Linear-attention layer weights (Phase 2b).
///
/// 30 of the 40 Qwen 3.6 35B-A3B layers are `linear_attention` (the
/// `qwen3_next` Gated-DeltaNet style state-space block). Per-layer:
///   - 2× layer RMSNorm (input, post_attention)
///   - SSM scalars: A_log [num_ssm_heads], dt_bias [num_ssm_heads]
///   - 1D causal conv: conv1d.weight [conv_channels, 1, kernel]
///   - small bf16 projections in_proj_a / in_proj_b [num_ssm_heads, H]
///   - FP8 blockwise projections: in_proj_qkv [N_qkv, H],
///     in_proj_z [N_z, H], out_proj [H, N_out]
///   - SSM head-wise RMSNorm: norm.weight [head_dim_ssm]
///
/// MoE block lives on a separate struct (every layer carries one).
#[derive(Debug)]
pub struct Qwen36LinearAttnLayer {
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub a_log: F16Weight,
    pub dt_bias: F16Weight,
    pub conv1d: F16Weight,
    pub in_proj_a: F16Weight,
    pub in_proj_b: F16Weight,
    pub in_proj_qkv: Fp8Weight,
    pub in_proj_z: Fp8Weight,
    pub norm: F16Weight,
    pub out_proj: Fp8Weight,
}

/// Per-layer MoE block (Phase 2b). Present on EVERY Qwen 3.6 layer
/// (both linear and full attention variants share the same FFN block
/// structure).
///
/// Storage layout (fused-by-projection):
///   - Per-layer 256 experts are concatenated into one FP8 region per
///     projection role (gate / up / down). For experts[e][role], the
///     slice begins at `e * expert_role_bytes` within the fused
///     region. The blockscale tensor likewise stacks 256 expert
///     scales contiguously. This keeps 30 720 expert tensors collapsed
///     to 9 device pointers per layer — friendly for grouped-FP8
///     MoE-GEMM kernels in Phase 3.
///   - Shared expert is a single (gate, up, down) triplet — same
///     shape as one routed expert.
///   - Router (`mlp.gate`) is a small bf16 [num_experts, hidden]
///     classifier; no FP8.
///   - `shared_expert_gate` is a bf16 [1, hidden] sigmoid gate that
///     scales the shared expert's contribution per token.
#[derive(Debug)]
pub struct Qwen36MoeBlock {
    pub router: F16Weight,
    pub shared_expert_gate_logit: F16Weight,
    pub experts_gate_proj_fused: Fp8Weight,
    pub experts_up_proj_fused: Fp8Weight,
    pub experts_down_proj_fused: Fp8Weight,
    pub shared_expert_gate_proj: Fp8Weight,
    pub shared_expert_up_proj: Fp8Weight,
    pub shared_expert_down_proj: Fp8Weight,
}

/// Per-layer attention block — exactly one of the two variants is
/// populated per layer, matching the `layer_types` array.
#[derive(Debug)]
pub enum Qwen36LayerAttn {
    Linear(Qwen36LinearAttnLayer),
    Full(Qwen36FullAttnLayer),
}

/// Full per-layer weights (attention block + MoE block).
#[derive(Debug)]
pub struct Qwen36Layer {
    pub attn: Qwen36LayerAttn,
    pub moe: Qwen36MoeBlock,
}

/// Qwen 3.6 loaded model — fully populated after Phase 2b.
///
/// Phase 1: `outside` populated, `layers` empty.
/// Phase 2b: `outside` + `layers` (all 40 slots, attn variant matches
///   `layer_types`, MoE block + 256 experts uploaded).
#[derive(Debug)]
pub struct Qwen36LoadedModel {
    pub outside: Qwen36LoadedOutside,
    /// All 40 layers, fully populated in Phase 2b. Linear/full attn
    /// distinction lives inside the per-layer `attn` enum.
    pub layers: Vec<Qwen36Layer>,
}
