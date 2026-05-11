//! Qwen 3.6 35B-A3B architecture summary (Phase 0 scaffolding).
//!
//! The generic `rvllm_loader::ModelArch::from_dir` already parses
//! `layer_types` (sliding/linear/full) and the standard transformer
//! fields. This module adds the Qwen-specific markers that distinguish
//! a Qwen-3.6 MoE config from Gemma 4 / Llama / Qwen 3.5 dense:
//!
//!   - hybrid attention pattern (3:1 linear:full, 30 + 10 = 40 layers
//!     in the 35B-A3B variant)
//!   - 256-expert MoE with top-k=8 + shared expert
//!   - `attn_output_gate=true` (sigmoid gate after o_proj)
//!   - SiLU activation, no logit softcap, no tied embeddings
//!
//! Phase 0 only loads + validates these markers and logs a summary.
//! No tensor loading, no forward pass — `Qwen36Bringup` returns
//! `unimplemented!` for all forward methods until Phase 1+ lands.

use std::path::Path;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::{LayerAttnType, ModelArch};

/// Qwen 3.6 specific markers read from `config.json`. The base
/// transformer fields (layers, hidden_size, head_dim, etc.) live on
/// the embedded `ModelArch`.
#[derive(Debug, Clone)]
pub struct Qwen36Arch {
    pub base: ModelArch,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub attn_output_gate: bool,
    pub mtp_present: bool,
    pub n_linear: usize,
    pub n_full: usize,
}

impl Qwen36Arch {
    /// Probe a model directory and decide whether it's a Qwen-3.6
    /// MoE checkpoint. Returns `Ok(None)` if the markers don't match
    /// (caller should fall through to the Gemma 4 path).
    pub fn from_dir(model_dir: &Path) -> Result<Option<Self>> {
        let cfg_path = model_dir.join("config.json");
        let bytes = match std::fs::read(&cfg_path) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };
        let v: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let tc = if v["text_config"]["hidden_size"].is_u64() {
            &v["text_config"]
        } else {
            &v
        };
        let num_experts = tc["num_experts"].as_u64().unwrap_or(0) as usize;
        let attn_output_gate = tc["attn_output_gate"].as_bool().unwrap_or(false);

        // Qwen-3.6 detection: needs MoE + attn_output_gate + at least
        // one linear_attention layer in layer_types.
        if num_experts < 16 || !attn_output_gate {
            return Ok(None);
        }

        let mut base = ModelArch::from_dir(model_dir)?;
        // Qwen 3.6 config nests rope under `rope_parameters` at the top
        // level (not `rope_parameters.sliding_attention.*` like Gemma 4),
        // so the generic `ModelArch::from_dir` can't find it and falls
        // back to 10_000. Re-read the correct value here so downstream
        // code (RoPE table precompute, MRoPE section math) gets the
        // real Qwen `rope_theta` (10_000_000 for the 35B-A3B variant).
        if let Some(t) = tc["rope_parameters"]["rope_theta"].as_f64() {
            base.rope_theta = t as f32;
        }
        let n_linear = base
            .layer_types
            .iter()
            .filter(|t| **t == LayerAttnType::Linear)
            .count();
        let n_full = base
            .layer_types
            .iter()
            .filter(|t| **t == LayerAttnType::Full)
            .count();
        if n_linear == 0 {
            return Ok(None);
        }

        let num_experts_per_tok = tc["num_experts_per_tok"].as_u64().unwrap_or(0) as usize;
        let moe_intermediate_size = tc["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
        let shared_expert_intermediate_size = tc["shared_expert_intermediate_size"]
            .as_u64()
            .unwrap_or(0) as usize;

        let mtp_present = model_dir.join("mtp.safetensors").exists();

        if num_experts_per_tok == 0 || moe_intermediate_size == 0 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "qwen36 config.json missing MoE fields: num_experts_per_tok={num_experts_per_tok}, moe_intermediate_size={moe_intermediate_size}"
                    ),
                },
                ctx: LoaderCtx {
                    path: cfg_path,
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        Ok(Some(Self {
            base,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            attn_output_gate,
            mtp_present,
            n_linear,
            n_full,
        }))
    }

    pub fn log_summary(&self) {
        eprintln!(
            "[loader] Qwen 3.6: {} layers ({} linear + {} full), \
             hidden={}, heads={}/{} kvh, hd={}, vocab={}, \
             experts={} top_k={} shared_expert_int={} moe_int={}, \
             attn_output_gate={}, tied_emb={}, softcap={:?}, mtp={}",
            self.base.num_hidden_layers,
            self.n_linear,
            self.n_full,
            self.base.hidden_size,
            self.base.num_attention_heads,
            self.base.num_key_value_heads,
            self.base.head_dim,
            self.base.vocab_size,
            self.num_experts,
            self.num_experts_per_tok,
            self.shared_expert_intermediate_size,
            self.moe_intermediate_size,
            self.attn_output_gate,
            self.base.tie_word_embeddings,
            self.base.final_logit_softcapping,
            self.mtp_present,
        );
    }
}
