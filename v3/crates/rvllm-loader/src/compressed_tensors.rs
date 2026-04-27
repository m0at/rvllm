//! Compressed-tensors AWQ checkpoint metadata.
//!
//! Cycle 41 step 3b: parse `quantization_config` from the model's
//! `config.json` so the loader knows which linears are AWQ-quantized
//! and what their group structure is.
//!
//! Format reference: ebircak/gemma-4-31B-it-4bit-W4A16-AWQ ships
//! `config.json` with:
//!
//! ```json
//! "quantization_config": {
//!   "config_groups": {
//!     "group_0": {
//!       "format": "pack-quantized",
//!       "targets": ["Linear"],
//!       "weights": {
//!         "num_bits": 4,
//!         "group_size": 128,
//!         "strategy": "group",
//!         "symmetric": false,
//!         "type": "int",
//!         "zp_dtype": "torch.int8"
//!       }
//!     }
//!   },
//!   "format": "pack-quantized",
//!   "ignore": ["lm_head", "re:model.vision_tower.*"],
//!   "quantization_status": "compressed"
//! }
//! ```
//!
//! Per-tensor layout (verified on the Gemma 4 31B AWQ checkpoint):
//!
//! ```text
//! <linear>.weight_packed     : I32 [N, K/8]   (8 INT4 per int32 along K)
//! <linear>.weight_scale      : BF16 [N, K/g]
//! <linear>.weight_zero_point : I32 [N/8, K/g] (8 INT4 per int32 along N)
//! <linear>.weight_shape      : I64 [2]         (original [N, K] dense shape)
//! ```

// No `serde` dep on this crate — parse the JSON via serde_json::Value
// directly to avoid pulling a new dep just for one struct.

/// AWQ / compressed-tensors weight quantization scheme. Captured from
/// `quantization_config.config_groups[*].weights` in `config.json`.
#[derive(Clone, Debug)]
pub struct AwqWeightScheme {
    /// Bits per weight element. Always 4 for the AWQ W4A16 path.
    pub num_bits: u32,
    /// Block-scale group size along K. Typical AWQ value: 128.
    pub group_size: u32,
    /// `true` = symmetric quant (zero_point fixed at 2^(num_bits-1) = 8 for 4-bit);
    /// `false` = asymmetric (per-group, per-row INT4 zero stored separately).
    pub symmetric: bool,
}

/// AWQ pack format names we accept. compressed-tensors uses
/// `"pack-quantized"`. Other strings are rejected so the loader fails
/// loudly on unknown variants instead of silently mis-loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AwqPackFormat {
    /// 4-bit nibbles packed 8-per-int32 along K, zero-points 8-per-int32
    /// along N (the format ebircak / cyankiwi / QuantTrio Gemma 4
    /// checkpoints use).
    PackQuantized,
}

/// Top-level AWQ config plumbed in from `config.json`.
#[derive(Clone, Debug)]
pub struct AwqConfig {
    /// Format of weight packing.
    pub format: AwqPackFormat,
    /// Quant scheme — single shared scheme across all targeted linears
    /// (compressed-tensors supports multiple `config_groups` but every
    /// real AWQ Gemma 4 checkpoint uses one).
    pub scheme: AwqWeightScheme,
    /// Linear-name patterns to skip (kept dense / unquantized).
    /// Compressed-tensors `ignore` lists raw names AND `re:<regex>`
    /// patterns. We store the raw strings; runtime matching uses
    /// [`Self::is_ignored`].
    pub ignore: Vec<String>,
}

impl AwqConfig {
    /// Parse `quantization_config` JSON value. Returns `Ok(None)` when
    /// the model is unquantized (no `quantization_config` field), an
    /// `Err` when the field is present but malformed.
    pub fn from_json(qc: Option<&serde_json::Value>) -> Result<Option<Self>, String> {
        let Some(qc) = qc else { return Ok(None) };

        let format = match qc.get("format").and_then(|v| v.as_str()) {
            Some("pack-quantized") => AwqPackFormat::PackQuantized,
            Some(other) => {
                return Err(format!(
                    "quantization_config.format = {other:?} unsupported; \
                     only \"pack-quantized\" recognized"
                ));
            }
            None => return Err("quantization_config.format missing".into()),
        };

        let groups = qc
            .get("config_groups")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "quantization_config.config_groups missing".to_string())?;

        // Find the (one) group that targets Linear.
        let mut linear_groups = groups.values().filter(|g| {
            g.get("targets")
                .and_then(|t| t.as_array())
                .map(|arr| arr.iter().any(|s| s.as_str() == Some("Linear")))
                .unwrap_or(false)
        });
        let group = linear_groups
            .next()
            .ok_or_else(|| "no config_group targets Linear".to_string())?;
        if linear_groups.next().is_some() {
            return Err(
                "multiple config_groups target Linear — heterogeneous quant \
                 schemes not yet supported"
                    .into(),
            );
        }

        let weights = group
            .get("weights")
            .ok_or_else(|| "config_group.weights missing".to_string())?;

        let r#type = weights.get("type").and_then(|v| v.as_str()).unwrap_or("int");
        if r#type != "int" {
            return Err(format!(
                "weights.type = {type:?} not supported; expected \"int\"",
                type = r#type
            ));
        }
        let strategy = weights.get("strategy").and_then(|v| v.as_str()).unwrap_or("group");
        if strategy != "group" {
            return Err(format!(
                "weights.strategy = {strategy:?} not supported; expected \"group\""
            ));
        }
        let num_bits = weights
            .get("num_bits")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "weights.num_bits missing or not a number".to_string())?
            as u32;
        let group_size = weights
            .get("group_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "weights.group_size missing or not a number".to_string())?
            as u32;
        let symmetric = weights
            .get("symmetric")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let scheme = AwqWeightScheme {
            num_bits,
            group_size,
            symmetric,
        };

        let ignore = qc
            .get("ignore")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(Self {
            format,
            scheme,
            ignore,
        }))
    }

    /// True when `linear_name` matches one of the `ignore` entries
    /// (literal name OR `re:<regex>`). For now the regex side does a
    /// simple substring fallback if the entry starts with `re:` —
    /// proper regex would require pulling in `regex` as a loader dep,
    /// which we defer until a real ignore pattern bites us.
    pub fn is_ignored(&self, linear_name: &str) -> bool {
        for entry in &self.ignore {
            if let Some(pat) = entry.strip_prefix("re:") {
                // Crude regex fallback (avoids pulling `regex` as a
                // loader dep). Strip a trailing `.*` (the dominant
                // pattern in compressed-tensors `ignore` lists) and
                // substring-match the prefix. If anchored or more
                // complex metacharacters appear, conservatively skip.
                let pat = pat.strip_suffix(".*").unwrap_or(pat);
                if !pat.contains(['^', '$', '|', '+', '?', '(', ')', '[', ']'])
                    && !pat.contains("\\")
                    && linear_name.contains(pat)
                {
                    return true;
                }
            } else if linear_name == entry {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> serde_json::Value {
        // Mirrors the real ebircak/gemma-4-31B-it-4bit-W4A16-AWQ
        // quantization_config block.
        json!({
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "group_size": 128,
                        "strategy": "group",
                        "symmetric": false,
                        "type": "int",
                        "zp_dtype": "torch.int8"
                    }
                }
            },
            "format": "pack-quantized",
            "ignore": [
                "lm_head",
                "re:model.vision_tower.*",
                "re:model.multi_modal_projector.*"
            ]
        })
    }

    #[test]
    fn parses_real_gemma4_awq_config() {
        let parsed = AwqConfig::from_json(Some(&cfg())).expect("parse").expect("Some");
        assert_eq!(parsed.format, AwqPackFormat::PackQuantized);
        assert_eq!(parsed.scheme.num_bits, 4);
        assert_eq!(parsed.scheme.group_size, 128);
        assert!(!parsed.scheme.symmetric);
        assert_eq!(parsed.ignore.len(), 3);
    }

    #[test]
    fn ignores_lm_head_literal() {
        let parsed = AwqConfig::from_json(Some(&cfg())).unwrap().unwrap();
        assert!(parsed.is_ignored("lm_head"));
        assert!(!parsed.is_ignored("model.language_model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn ignores_vision_tower_substring() {
        let parsed = AwqConfig::from_json(Some(&cfg())).unwrap().unwrap();
        assert!(parsed.is_ignored("model.vision_tower.layer_0.weight"));
        // multi_modal_projector has metacharacter-free regex too.
        assert!(parsed.is_ignored("foo.model.multi_modal_projector.bar"));
    }

    #[test]
    fn unquantized_returns_none() {
        let r = AwqConfig::from_json(None).expect("ok");
        assert!(r.is_none());
    }

    #[test]
    fn rejects_unknown_format() {
        let bad = json!({
            "config_groups": {
                "g": {"targets": ["Linear"], "weights": {"num_bits": 4, "group_size": 128, "strategy": "group", "type": "int"}}
            },
            "format": "marlin"
        });
        let err = AwqConfig::from_json(Some(&bad)).expect_err("must reject");
        assert!(err.contains("marlin") || err.contains("unsupported"));
    }

    #[test]
    fn rejects_non_int_type() {
        let bad = json!({
            "config_groups": {
                "g": {"targets": ["Linear"], "weights": {"num_bits": 4, "group_size": 128, "strategy": "group", "type": "float"}}
            },
            "format": "pack-quantized"
        });
        assert!(AwqConfig::from_json(Some(&bad)).is_err());
    }
}
