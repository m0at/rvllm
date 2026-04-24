//! Gemma 4 model architecture parser.
//!
//! Parsed from the real google/gemma-4-31B-it config.json:
//! - `architectures: ["Gemma4ForConditionalGeneration"]`
//! - `text_config` sub-object for the language model params
//! - `head_dim: 256` (sliding), `global_head_dim: 512` (global)
//! - `num_key_value_heads: 16` (sliding), `num_global_key_value_heads: 4`
//! - `rope_parameters` is a nested object with per-type sub-configs
//! - `layer_types` array: 5 sliding + 1 global, repeating
//! - `tie_word_embeddings: true` (no separate lm_head.weight)
//! - `hidden_activation: "gelu_pytorch_tanh"`
//! - `final_logit_softcapping: 30.0`
//!
//! Actual Gemma 4 31B dimensions:
//!   hidden=5376, heads=32, layers=60, intermediate=21504, vocab=262144
//!   sliding: head_dim=256, kv_heads=16, theta=10000, full rotation
//!   global:  head_dim=512, kv_heads=4,  theta=1M, partial_rotary=0.25

use std::path::Path;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Gemma4LayerType {
    SlidingAttention,
    GlobalAttention,
}

#[derive(Clone, Debug)]
pub struct Gemma4Arch {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim_sliding: usize,
    pub head_dim_global: usize,
    pub num_kv_heads_sliding: usize,
    pub num_kv_heads_global: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub sliding_window_size: usize,
    pub rope_theta_sliding: f32,
    pub rope_theta_global: f32,
    pub partial_rotary_factor_global: f32,
    pub attn_scale: f32,
    pub logit_softcap: f32,
    pub layer_types: Vec<Gemma4LayerType>,
    pub weight_prefix: String,
    pub tie_word_embeddings: bool,
    pub pli_dim: usize,
    pub num_kv_shared_layers: usize,
    pub kv_shared_map: std::collections::BTreeMap<usize, usize>,
    pub num_experts: usize,
    pub top_k_experts: usize,
}

impl Gemma4Arch {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let bytes = std::fs::read(&p).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: p.clone(),
            source,
        })?;
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("config.json: {e}"),
                },
                ctx: LoaderCtx {
                    path: p.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;

        let tc = if v.get("text_config").is_some() {
            &v["text_config"]
        } else {
            &v
        };

        let num_hidden_layers = tc["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
        let hidden_size = tc["hidden_size"].as_u64().unwrap_or(0) as usize;
        let num_attention_heads = tc["num_attention_heads"].as_u64().unwrap_or(0) as usize;

        let head_dim_sliding = tc["head_dim"].as_u64().unwrap_or(256) as usize;
        let head_dim_global = tc["global_head_dim"]
            .as_u64()
            .unwrap_or(head_dim_sliding as u64) as usize;

        let intermediate_size = tc["intermediate_size"].as_u64().unwrap_or(0) as usize;
        let vocab_size = tc["vocab_size"]
            .as_u64()
            .or_else(|| v["vocab_size"].as_u64())
            .unwrap_or(0) as usize;
        let rms_norm_eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let max_position_embeddings =
            tc["max_position_embeddings"].as_u64().unwrap_or(262144) as usize;
        let sliding_window_size = tc["sliding_window"]
            .as_u64()
            .or_else(|| tc["sliding_window_size"].as_u64())
            .unwrap_or(1024) as usize;

        let num_kv_heads_sliding = tc["num_key_value_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize;
        let num_kv_heads_global = tc["num_global_key_value_heads"]
            .as_u64()
            .or_else(|| tc["global_num_key_value_heads"].as_u64())
            .or_else(|| tc["num_key_value_heads_global"].as_u64())
            .unwrap_or(num_kv_heads_sliding as u64) as usize;

        // RoPE parameters -- nested per-type in Gemma 4
        let rope = &tc["rope_parameters"];
        let rope_theta_sliding = rope["sliding_attention"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta"].as_f64())
            .unwrap_or(10000.0) as f32;
        let rope_theta_global = rope["full_attention"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta_global"].as_f64())
            .unwrap_or(1_000_000.0) as f32;
        let partial_rotary_factor_global = rope["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .or_else(|| tc["partial_rotary_factor"].as_f64())
            .unwrap_or(0.25) as f32;
        let _query_pre_attn_scalar = tc["query_pre_attn_scalar"]
            .as_f64()
            .unwrap_or(head_dim_sliding as f64) as f32;
        let attn_scale = 1.0;

        let logit_softcap = tc["final_logit_softcapping"]
            .as_f64()
            .or_else(|| tc["logit_softcapping"].as_f64())
            .unwrap_or(30.0) as f32;

        let tie_word_embeddings = tc["tie_word_embeddings"]
            .as_bool()
            .or_else(|| v["tie_word_embeddings"].as_bool())
            .unwrap_or(true);

        let pli_dim = tc["hidden_size_per_layer_input"].as_u64().unwrap_or(0) as usize;
        let num_kv_shared_layers = tc["num_kv_shared_layers"].as_u64().unwrap_or(0) as usize;
        let num_experts = tc["num_local_experts"]
            .as_u64()
            .or_else(|| tc["num_experts"].as_u64())
            .unwrap_or(0) as usize;
        let top_k_experts = tc["num_experts_per_tok"]
            .as_u64()
            .or_else(|| tc["top_k"].as_u64())
            .unwrap_or(0) as usize;

        let layer_types = Self::parse_layer_types(tc, num_hidden_layers);
        let weight_prefix = Self::detect_weight_prefix(dir);
        let kv_shared_map =
            Self::build_kv_shared_map(&layer_types, num_hidden_layers, num_kv_shared_layers);

        if num_attention_heads == 0 || hidden_size == 0 || num_hidden_layers == 0 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: "Gemma4 config has zero-valued required fields".into(),
                },
                ctx: LoaderCtx {
                    path: p,
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        Ok(Self {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            head_dim_sliding,
            head_dim_global,
            num_kv_heads_sliding,
            num_kv_heads_global,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            max_position_embeddings,
            sliding_window_size,
            rope_theta_sliding,
            rope_theta_global,
            partial_rotary_factor_global,
            attn_scale,
            logit_softcap,
            layer_types,
            weight_prefix,
            tie_word_embeddings,
            pli_dim,
            num_kv_shared_layers,
            kv_shared_map,
            num_experts,
            top_k_experts,
        })
    }

    fn parse_layer_types(tc: &serde_json::Value, n: usize) -> Vec<Gemma4LayerType> {
        if let Some(arr) = tc["layer_types"].as_array() {
            return arr
                .iter()
                .map(|t| match t.as_str().unwrap_or("sliding_attention") {
                    "global_attention" | "full_attention" => Gemma4LayerType::GlobalAttention,
                    _ => Gemma4LayerType::SlidingAttention,
                })
                .collect();
        }
        // Default: 5 sliding + 1 global repeating
        (0..n)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    Gemma4LayerType::GlobalAttention
                } else {
                    Gemma4LayerType::SlidingAttention
                }
            })
            .collect()
    }

    fn detect_weight_prefix(dir: &Path) -> String {
        fn detect_from_keys(keys: impl Iterator<Item = String>) -> Option<String> {
            for key in keys {
                if key.starts_with("model.language_model.") {
                    return Some("model.language_model".to_string());
                }
                if key.starts_with("language_model.") {
                    return Some("language_model".to_string());
                }
            }
            None
        }

        let idx_path = dir.join("model.safetensors.index.json");
        if let Ok(bytes) = std::fs::read(&idx_path) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(map) = v["weight_map"].as_object() {
                    if let Some(prefix) = detect_from_keys(map.keys().cloned()) {
                        return prefix;
                    }
                }
            }
        }

        let single_path = dir.join("model.safetensors");
        if let Ok(mut f) = std::fs::File::open(&single_path) {
            use std::io::Read;

            let mut len_bytes = [0u8; 8];
            if f.read_exact(&mut len_bytes).is_ok() {
                let header_len = u64::from_le_bytes(len_bytes) as usize;
                let mut header_bytes = vec![0u8; header_len];
                if f.read_exact(&mut header_bytes).is_ok() {
                    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&header_bytes) {
                        if let Some(map) = v.as_object() {
                            if let Some(prefix) = detect_from_keys(map.keys().cloned()) {
                                return prefix;
                            }
                        }
                    }
                }
            }
        }

        "model".to_string()
    }

    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.head_dim_sliding,
            Gemma4LayerType::GlobalAttention => self.head_dim_global,
        }
    }

    pub fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.num_kv_heads_sliding,
            Gemma4LayerType::GlobalAttention => self.num_kv_heads_global,
        }
    }

    pub fn rotary_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            // Sliding: full rotation of head_dim_sliding (256)
            Gemma4LayerType::SlidingAttention => self.head_dim_sliding,
            // Global: partial rotation of head_dim_global (512 * 0.25 = 128)
            Gemma4LayerType::GlobalAttention => {
                let rd = (self.head_dim_global as f32 * self.partial_rotary_factor_global) as usize;
                (rd / 2) * 2 // ensure even
            }
        }
    }

    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> f32 {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.rope_theta_sliding,
            Gemma4LayerType::GlobalAttention => self.rope_theta_global,
        }
    }

    pub fn q_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.num_attention_heads * self.head_dim_for_layer(layer_idx)
    }

    pub fn kv_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.num_kv_heads_for_layer(layer_idx) * self.head_dim_for_layer(layer_idx)
    }

    pub fn max_head_dim(&self) -> usize {
        self.head_dim_sliding.max(self.head_dim_global)
    }

    pub fn max_kv_heads(&self) -> usize {
        self.num_kv_heads_sliding.max(self.num_kv_heads_global)
    }

    pub fn max_q_dim(&self) -> usize {
        self.num_attention_heads * self.max_head_dim()
    }

    pub fn is_pli_enabled(&self) -> bool {
        self.pli_dim > 0
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    pub fn is_kv_shared(&self, layer_idx: usize) -> bool {
        self.kv_shared_map.contains_key(&layer_idx)
    }

    pub fn kv_source_layer(&self, layer_idx: usize) -> usize {
        self.kv_shared_map
            .get(&layer_idx)
            .copied()
            .unwrap_or(layer_idx)
    }

    fn build_kv_shared_map(
        layer_types: &[Gemma4LayerType],
        num_layers: usize,
        num_kv_shared: usize,
    ) -> std::collections::BTreeMap<usize, usize> {
        let mut map = std::collections::BTreeMap::new();
        if num_kv_shared == 0 || num_kv_shared >= num_layers {
            return map;
        }
        let first_shared = num_layers - num_kv_shared;
        for i in first_shared..num_layers {
            let Some(current_type) = layer_types.get(i) else {
                continue;
            };
            if let Some(src) = (0..first_shared)
                .rev()
                .find(|&src| layer_types.get(src) == Some(current_type))
            {
                map.insert(i, src);
            }
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layer_pattern_every_6th_global() {
        let types = Gemma4Arch::parse_layer_types(&serde_json::Value::Null, 12);
        // 0:s 1:s 2:s 3:s 4:s 5:g 6:s 7:s 8:s 9:s 10:s 11:g
        assert_eq!(types[0], Gemma4LayerType::SlidingAttention);
        assert_eq!(types[4], Gemma4LayerType::SlidingAttention);
        assert_eq!(types[5], Gemma4LayerType::GlobalAttention);
        assert_eq!(types[11], Gemma4LayerType::GlobalAttention);
    }

    #[test]
    fn parses_real_layer_types() {
        let v: serde_json::Value = serde_json::json!({
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ]
        });
        let types = Gemma4Arch::parse_layer_types(&v, 6);
        assert_eq!(types[5], Gemma4LayerType::GlobalAttention);
        assert_eq!(types[0], Gemma4LayerType::SlidingAttention);
    }

    #[test]
    fn rotary_dim_sliding_is_full() {
        let arch = Gemma4Arch {
            num_hidden_layers: 6,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21504,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            sliding_window_size: 1024,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1000000.0,
            partial_rotary_factor_global: 0.25,
            attn_scale: 1.0,
            logit_softcap: 30.0,
            layer_types: vec![Gemma4LayerType::SlidingAttention; 6],
            weight_prefix: "model".into(),
            tie_word_embeddings: true,
            pli_dim: 0,
            num_kv_shared_layers: 0,
            kv_shared_map: std::collections::BTreeMap::new(),
            num_experts: 0,
            top_k_experts: 0,
        };
        assert_eq!(arch.rotary_dim_for_layer(0), 256);
    }

    #[test]
    fn rotary_dim_global_is_partial() {
        let arch = Gemma4Arch {
            num_hidden_layers: 6,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21504,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            sliding_window_size: 1024,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1000000.0,
            partial_rotary_factor_global: 0.25,
            attn_scale: 1.0,
            logit_softcap: 30.0,
            layer_types: vec![Gemma4LayerType::GlobalAttention; 6],
            weight_prefix: "model".into(),
            tie_word_embeddings: true,
            pli_dim: 0,
            num_kv_shared_layers: 0,
            kv_shared_map: std::collections::BTreeMap::new(),
            num_experts: 0,
            top_k_experts: 0,
        };
        // 512 * 0.25 = 128
        assert_eq!(arch.rotary_dim_for_layer(0), 128);
    }

    #[test]
    fn kv_shared_map_e4b_pattern() {
        let layer_types = Gemma4Arch::parse_layer_types(&serde_json::Value::Null, 42);
        let map = Gemma4Arch::build_kv_shared_map(&layer_types, 42, 18);
        assert!(map.contains_key(&24));
        assert!(!map.contains_key(&23));
        assert_eq!(map[&24], 22);
        assert_eq!(map[&25], 22);
        assert_eq!(map[&29], 23);
        assert_eq!(map[&41], 23);
        assert_eq!(layer_types[24], Gemma4LayerType::SlidingAttention);
        assert_eq!(layer_types[29], Gemma4LayerType::GlobalAttention);
    }
}
