//! MiniMax-M2 specific configuration parsed from HF `config.json`.
//!
//! The M2 architecture extends the base `ModelConfig` with MoE routing metadata,
//! partial-RoPE knobs, QK-norm, multi-token-prediction heads, and the NVFP4
//! quantization manifest produced by ModelOpt. This module exposes the
//! `MiniMaxM2Extras` + `NvFp4Config` structs and a parser callable from the
//! base registry when the architecture is `MiniMaxM2`.

use std::path::Path;

use crate::error::{ConfigError, Result, RvllmError};

use super::hf;

#[derive(Clone, Debug)]
pub struct NvFp4Config {
    pub weight_num_bits: u8,
    pub activation_num_bits: u8,
    pub group_size: usize,
    pub scale_dtype: String,
    pub ignore_patterns: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct MiniMaxM2Extras {
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub rotary_dim: usize,
    pub partial_rotary_factor: f32,
    pub use_qk_norm: bool,
    pub use_routing_bias: bool,
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub use_mtp: bool,
    pub num_mtp_modules: usize,
    pub mtp_transformer_layers: usize,
    pub nvfp4: Option<NvFp4Config>,
}

impl MiniMaxM2Extras {
    /// Parse the M2-specific fields from the top-level HF config JSON.
    pub fn from_hf_value(v: &serde_json::Value, file: &Path) -> Result<Self> {
        let num_local_experts = hf::usize_field(v, "num_local_experts", file)?;
        let num_experts_per_tok = hf::usize_field(v, "num_experts_per_tok", file)?;
        let moe_intermediate_size = hf::usize_field(v, "intermediate_size", file)?;
        let shared_intermediate_size = hf::usize_field(v, "shared_intermediate_size", file)?;
        let rotary_dim = hf::usize_field(v, "rotary_dim", file)?;
        let partial_rotary_factor = f32_field_required(v, "partial_rotary_factor", file)?;
        let use_qk_norm = bool_field_required(v, "use_qk_norm", file)?;
        let use_routing_bias = bool_field_required(v, "use_routing_bias", file)?;
        let scoring_func = hf::str_field(v, "scoring_func", file)?;
        let norm_topk_prob = bool_field_required(v, "norm_topk_prob", file)?;
        let use_mtp = bool_field_required(v, "use_mtp", file)?;
        let num_mtp_modules = hf::usize_field(v, "num_mtp_modules", file)?;
        let mtp_transformer_layers = hf::usize_field(v, "mtp_transformer_layers", file)?;

        if num_experts_per_tok == 0 || num_experts_per_tok > num_local_experts {
            return Err(RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "num_experts_per_tok {num_experts_per_tok} out of range for num_local_experts {num_local_experts}"
                    )],
                },
                "num_experts_per_tok",
            ));
        }
        if rotary_dim == 0 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "rotary_dim",
                    reason: "must be > 0".into(),
                },
                "rotary_dim",
            ));
        }
        if !(0.0..=1.0).contains(&partial_rotary_factor) {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "partial_rotary_factor",
                    reason: format!("must be in [0,1], got {partial_rotary_factor}"),
                },
                "partial_rotary_factor",
            ));
        }
        match scoring_func.as_str() {
            "sigmoid" | "softmax" => {}
            other => {
                return Err(RvllmError::config(
                    ConfigError::InvalidField {
                        name: "scoring_func",
                        reason: format!("unsupported scoring_func: {other}"),
                    },
                    "scoring_func",
                ));
            }
        }

        let nvfp4 = match v.get("quantization_config") {
            Some(qc) if !qc.is_null() => Some(NvFp4Config::from_hf_value(qc, file)?),
            _ => None,
        };

        Ok(Self {
            num_local_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_intermediate_size,
            rotary_dim,
            partial_rotary_factor,
            use_qk_norm,
            use_routing_bias,
            scoring_func,
            norm_topk_prob,
            use_mtp,
            num_mtp_modules,
            mtp_transformer_layers,
            nvfp4,
        })
    }
}

impl NvFp4Config {
    fn from_hf_value(qc: &serde_json::Value, file: &Path) -> Result<Self> {
        let algo = hf::str_field(qc, "quant_algo", file)?;
        if algo != "NVFP4" {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "quant_algo",
                    reason: format!("unsupported quant_algo: {algo}"),
                },
                "quant_algo",
            ));
        }

        let weights = qc.get("weights").ok_or_else(|| {
            RvllmError::config(
                ConfigError::MissingHfField {
                    name: "quantization_config.weights",
                    file: file.to_path_buf(),
                },
                "quantization_config.weights",
            )
        })?;
        let weight_num_bits = hf::usize_field(weights, "num_bits", file)? as u8;
        let group_size = hf::usize_field(weights, "group_size", file)?;
        let weight_type = hf::str_field(weights, "type", file)?;
        if weight_type != "float" {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "quantization_config.weights.type",
                    reason: format!("expected float, got {weight_type}"),
                },
                "quantization_config.weights.type",
            ));
        }
        if weight_num_bits != 4 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "quantization_config.weights.num_bits",
                    reason: format!("expected 4, got {weight_num_bits}"),
                },
                "quantization_config.weights.num_bits",
            ));
        }
        if group_size != 16 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "quantization_config.weights.group_size",
                    reason: format!("expected 16, got {group_size}"),
                },
                "quantization_config.weights.group_size",
            ));
        }

        let acts = qc.get("input_activations").ok_or_else(|| {
            RvllmError::config(
                ConfigError::MissingHfField {
                    name: "quantization_config.input_activations",
                    file: file.to_path_buf(),
                },
                "quantization_config.input_activations",
            )
        })?;
        let activation_num_bits = hf::usize_field(acts, "num_bits", file)? as u8;

        let scale_dtype = qc
            .get("scale_dtype")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "e4m3".to_string());

        let ignore_patterns = match qc.get("ignore") {
            Some(serde_json::Value::Array(items)) => items
                .iter()
                .map(|x| match x.as_str() {
                    Some(s) => Ok(s.to_string()),
                    None => Err(RvllmError::config(
                        ConfigError::HfTypeMismatch {
                            name: "quantization_config.ignore[*]",
                            expected: "string",
                        },
                        "quantization_config.ignore",
                    )),
                })
                .collect::<Result<Vec<_>>>()?,
            Some(_) => {
                return Err(RvllmError::config(
                    ConfigError::HfTypeMismatch {
                        name: "quantization_config.ignore",
                        expected: "array of strings",
                    },
                    "quantization_config.ignore",
                ));
            }
            None => Vec::new(),
        };

        Ok(Self {
            weight_num_bits,
            activation_num_bits,
            group_size,
            scale_dtype,
            ignore_patterns,
        })
    }
}

fn f32_field_required(v: &serde_json::Value, field: &'static str, file: &Path) -> Result<f32> {
    match v.get(field).and_then(|x| x.as_f64()) {
        Some(x) => Ok(x as f32),
        None => Err(RvllmError::config(
            ConfigError::MissingHfField {
                name: field,
                file: file.to_path_buf(),
            },
            field,
        )),
    }
}

fn bool_field_required(v: &serde_json::Value, field: &'static str, file: &Path) -> Result<bool> {
    match v.get(field).and_then(|x| x.as_bool()) {
        Some(x) => Ok(x),
        None => Err(RvllmError::config(
            ConfigError::MissingHfField {
                name: field,
                file: file.to_path_buf(),
            },
            field,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m2_config_json() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["MiniMaxM2ForCausalLM"],
            "model_type": "minimax_m2",
            "torch_dtype": "bfloat16",
            "hidden_size": 3072,
            "num_hidden_layers": 62,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 1536,
            "vocab_size": 200064,
            "max_position_embeddings": 196608,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000.0,
            "rotary_dim": 64,
            "partial_rotary_factor": 0.5,
            "use_qk_norm": true,
            "qk_norm_type": "per_layer",
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "shared_intermediate_size": 0,
            "scoring_func": "sigmoid",
            "use_routing_bias": true,
            "norm_topk_prob": true,
            "router_jitter_noise": 0.0,
            "use_mtp": true,
            "num_mtp_modules": 3,
            "mtp_transformer_layers": 1,
            "tie_word_embeddings": false,
            "sliding_window": serde_json::Value::Null,
            "hidden_act": "silu",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "quantization_config": {
                "quant_algo": "NVFP4",
                "quant_method": "modelopt",
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16
                },
                "ignore": [
                    "lm_head",
                    "model.layers.*.block_sparse_moe.gate",
                    "model.layers.*.self_attn*"
                ]
            }
        })
    }

    #[test]
    fn parses_minimax_m2_extras_and_nvfp4() {
        let v = m2_config_json();
        let extras = MiniMaxM2Extras::from_hf_value(&v, Path::new("config.json"))
            .expect("m2 extras should parse");
        assert_eq!(extras.num_local_experts, 256);
        assert_eq!(extras.num_experts_per_tok, 8);
        assert_eq!(extras.moe_intermediate_size, 1536);
        assert_eq!(extras.shared_intermediate_size, 0);
        assert_eq!(extras.rotary_dim, 64);
        assert!((extras.partial_rotary_factor - 0.5).abs() < 1e-6);
        assert!(extras.use_qk_norm);
        assert!(extras.use_routing_bias);
        assert_eq!(extras.scoring_func, "sigmoid");
        assert!(extras.norm_topk_prob);
        assert!(extras.use_mtp);
        assert_eq!(extras.num_mtp_modules, 3);
        assert_eq!(extras.mtp_transformer_layers, 1);

        let nvfp4 = extras.nvfp4.expect("nvfp4 block must be present");
        assert_eq!(nvfp4.weight_num_bits, 4);
        assert_eq!(nvfp4.activation_num_bits, 4);
        assert_eq!(nvfp4.group_size, 16);
        assert_eq!(nvfp4.ignore_patterns.len(), 3);
        assert_eq!(nvfp4.ignore_patterns[0], "lm_head");
    }

    #[test]
    fn rejects_unsupported_scoring_func() {
        let mut v = m2_config_json();
        v["scoring_func"] = serde_json::Value::String("relu".into());
        let err = MiniMaxM2Extras::from_hf_value(&v, Path::new("config.json"))
            .expect_err("relu scoring_func must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("scoring_func"));
    }

    #[test]
    fn rejects_bad_experts_per_tok() {
        let mut v = m2_config_json();
        v["num_experts_per_tok"] = serde_json::json!(512);
        let err = MiniMaxM2Extras::from_hf_value(&v, Path::new("config.json"))
            .expect_err("oversized top-k must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("num_experts_per_tok"));
    }

    #[test]
    fn rejects_non_nvfp4_quant_algo() {
        let mut v = m2_config_json();
        v["quantization_config"]["quant_algo"] = serde_json::Value::String("AWQ".into());
        let err = MiniMaxM2Extras::from_hf_value(&v, Path::new("config.json"))
            .expect_err("non-NVFP4 algo must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("quant_algo"));
    }
}
