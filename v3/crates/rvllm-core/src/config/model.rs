//! Model-architecture config, parsed from HF `config.json`.

use std::path::Path;

use crate::dtype::DType;
use crate::error::{ConfigError, Result, RvllmError};

use super::hf;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ModelArch {
    Qwen2,
    Llama,
    Mistral,
    Gemma2,
    Gemma4,
}

impl ModelArch {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "Qwen2ForCausalLM" => Some(ModelArch::Qwen2),
            "LlamaForCausalLM" => Some(ModelArch::Llama),
            "MistralForCausalLM" => Some(ModelArch::Mistral),
            "Gemma2ForCausalLM" => Some(ModelArch::Gemma2),
            "Gemma4ForConditionalGeneration" => Some(ModelArch::Gemma4),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architecture: ModelArch,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: DType,
}

impl ModelConfig {
    /// Parse an HF `config.json`. Every referenced field is required.
    pub fn load_hf(dir: &Path) -> Result<Self> {
        let file = dir.join("config.json");
        let body = std::fs::read_to_string(&file).map_err(|source| RvllmError::Io {
            err: crate::error::IoError::from(&source),
            path: file.clone(),
            source,
        })?;
        let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("config.json is not valid JSON: {e}")],
                },
                "config.json",
            )
        })?;
        Self::from_hf_value(&v, &file)
    }

    fn from_hf_value(v: &serde_json::Value, file: &Path) -> Result<Self> {
        let arch_name = hf::str_field(v, "architectures.0", file)?;
        let architecture = ModelArch::parse(&arch_name).ok_or_else(|| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "architectures[0]",
                    reason: format!("unsupported architecture: {arch_name}"),
                },
                "architectures[0]",
            )
        })?;

        // Gemma 3/4: text model fields nested under text_config.
        let tc = if v["text_config"]["hidden_size"].is_u64() {
            &v["text_config"]
        } else {
            v
        };

        let hidden_size = hf::usize_field(tc, "hidden_size", file)?;
        let num_layers = hf::usize_field(tc, "num_hidden_layers", file)?;
        let num_attention_heads = hf::usize_field(tc, "num_attention_heads", file)?;
        let num_kv_heads = hf::usize_field(tc, "num_key_value_heads", file)?;
        let intermediate_size = hf::usize_field(tc, "intermediate_size", file)?;
        let vocab_size = hf::usize_field(tc, "vocab_size", file)?;
        let max_position_embeddings = hf::usize_field(tc, "max_position_embeddings", file)?;
        let rms_norm_eps = hf::f32_field(tc, "rms_norm_eps", file)?;
        let rope_theta = tc["rope_parameters"]["sliding_attention"]["rope_theta"]
            .as_f64()
            .map(|t| t as f32)
            .map(Ok)
            .unwrap_or_else(|| hf::f32_field(tc, "rope_theta", file))?;
        let tie_word_embeddings = hf::bool_field_opt(tc, "tie_word_embeddings")
            .or_else(|| hf::bool_field_opt(v, "tie_word_embeddings"))
            .unwrap_or(false);
        let torch_dtype = match hf::str_field(v, "torch_dtype", file)
            .or_else(|_| hf::str_field(tc, "dtype", file))?
            .as_str()
        {
            "float16" => DType::F16,
            "bfloat16" => DType::Bf16,
            other => {
                return Err(RvllmError::config(
                    ConfigError::InvalidField {
                        name: "torch_dtype",
                        reason: format!("unsupported torch_dtype: {other}"),
                    },
                    "torch_dtype",
                ));
            }
        };

        if num_attention_heads == 0 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "num_attention_heads",
                    reason: "must be > 0".into(),
                },
                "num_attention_heads",
            ));
        }
        // Gemma 4 has explicit head_dim (256) that doesn't equal hidden_size/num_heads.
        let head_dim = tc["head_dim"]
            .as_u64()
            .map(|d| d as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads);
        if tc["head_dim"].as_u64().is_none() && head_dim * num_attention_heads != hidden_size {
            return Err(RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "hidden_size {hidden_size} not divisible by num_attention_heads {num_attention_heads}"
                    )],
                },
                "hidden_size",
            ));
        }

        Ok(Self {
            architecture,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            torch_dtype,
        })
    }
}
