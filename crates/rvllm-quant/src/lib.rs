//! Quantized weight storage and dequantization kernels for vllm-rs.

pub mod config;
pub mod dequant;
pub mod gemm;
pub mod linear;
pub mod method;
pub mod weight;

pub use config::QuantConfig;
pub use linear::QuantizedLinear;
pub use method::QuantMethod;
pub use weight::QuantizedWeight;

use std::path::Path;
use rvllm_core::prelude::*;

/// Inspect a model directory/config to detect the quantization method used.
/// Looks for `config.json` or `quantize_config.json` in the model path.
pub fn detect_quant_method(model_path: &Path) -> Result<QuantMethod> {
    // Try quantize_config.json first (GPTQ/AWQ convention)
    let quant_config_path = model_path.join("quantize_config.json");
    if quant_config_path.exists() {
        let content = std::fs::read_to_string(&quant_config_path)?;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(method) = val.get("quant_method").and_then(|v| v.as_str()) {
                return match method.to_lowercase().as_str() {
                    "gptq" => Ok(QuantMethod::GPTQ),
                    "awq" => Ok(QuantMethod::AWQ),
                    "squeezellm" => Ok(QuantMethod::SqueezeLLM),
                    "fp8" => Ok(QuantMethod::FP8),
                    other => {
                        tracing::warn!(method = other, "unknown quant_method in quantize_config.json");
                        Ok(QuantMethod::None)
                    }
                };
            }
        }
    }

    // Try config.json (HuggingFace convention)
    let config_path = model_path.join("config.json");
    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            // Check quantization_config.quant_method
            if let Some(qcfg) = val.get("quantization_config") {
                if let Some(method) = qcfg.get("quant_method").and_then(|v| v.as_str()) {
                    return match method.to_lowercase().as_str() {
                        "gptq" => Ok(QuantMethod::GPTQ),
                        "awq" => Ok(QuantMethod::AWQ),
                        "squeezellm" => Ok(QuantMethod::SqueezeLLM),
                        "fp8" => Ok(QuantMethod::FP8),
                        other => {
                            tracing::warn!(method = other, "unknown quant_method in config.json");
                            Ok(QuantMethod::None)
                        }
                    };
                }
            }
        }
    }

    // Check for GGUF files
    let gguf_path = if model_path.is_file() {
        Some(model_path.to_path_buf())
    } else {
        // Look for any .gguf file in the directory
        std::fs::read_dir(model_path)
            .ok()
            .and_then(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .find(|e| {
                        e.path()
                            .extension()
                            .map_or(false, |ext| ext == "gguf")
                    })
                    .map(|e| e.path())
            })
    };

    if let Some(path) = gguf_path {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        if name.contains("q4_0") {
            return Ok(QuantMethod::GgufQ4_0);
        } else if name.contains("q4_k_m") {
            return Ok(QuantMethod::GgufQ4KM);
        } else if name.contains("q5_0") {
            return Ok(QuantMethod::GgufQ5_0);
        } else if name.contains("q5_k_m") {
            return Ok(QuantMethod::GgufQ5KM);
        } else if name.contains("q8_0") {
            return Ok(QuantMethod::GgufQ8_0);
        }
    }

    tracing::info!(path = %model_path.display(), "no quantization detected");
    Ok(QuantMethod::None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn detect_gptq_from_quantize_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{"quant_method": "gptq", "bits": 4, "group_size": 128}"#;
        fs::write(dir.path().join("quantize_config.json"), config).unwrap();
        let method = detect_quant_method(dir.path()).unwrap();
        assert_eq!(method, QuantMethod::GPTQ);
    }

    #[test]
    fn detect_awq_from_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{"quantization_config": {"quant_method": "awq"}}"#;
        fs::write(dir.path().join("config.json"), config).unwrap();
        let method = detect_quant_method(dir.path()).unwrap();
        assert_eq!(method, QuantMethod::AWQ);
    }

    #[test]
    fn detect_gguf_from_filename() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model-q4_k_m.gguf"), b"fake").unwrap();
        let method = detect_quant_method(dir.path()).unwrap();
        assert_eq!(method, QuantMethod::GgufQ4KM);
    }

    #[test]
    fn detect_none_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let method = detect_quant_method(dir.path()).unwrap();
        assert_eq!(method, QuantMethod::None);
    }

    #[test]
    fn detect_fp8() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{"quant_method": "fp8"}"#;
        fs::write(dir.path().join("quantize_config.json"), config).unwrap();
        let method = detect_quant_method(dir.path()).unwrap();
        assert_eq!(method, QuantMethod::FP8);
    }
}
