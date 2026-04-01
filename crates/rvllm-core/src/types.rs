//! Newtype wrappers, vocabulary types, sampling parameters, and marker traits.

use std::fmt;
use std::str::FromStr;

use derive_more::{Display, From, Into};
use serde::{Deserialize, Serialize};

/// Data type for model weights and compute precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// Automatic selection: float16 on SM >= 7.0 GPUs, float32 otherwise.
    Auto,
    /// 32-bit floating point.
    Float32,
    /// 16-bit floating point (IEEE 754 half).
    Float16,
    /// 16-bit brain floating point.
    BFloat16,
}

impl Dtype {
    /// Size in bytes of a single element of this dtype.
    /// Returns 4 for Auto (conservative default; caller should resolve first).
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::Auto | Dtype::Float32 => 4,
            Dtype::Float16 | Dtype::BFloat16 => 2,
        }
    }

    /// True if this is a half-precision type (f16 or bf16).
    pub fn is_half(self) -> bool {
        matches!(self, Dtype::Float16 | Dtype::BFloat16)
    }

    /// True if this is Auto (needs resolution based on GPU capability).
    pub fn is_auto(self) -> bool {
        matches!(self, Dtype::Auto)
    }

    /// Resolve Auto to a concrete dtype based on GPU compute capability.
    /// SM >= 7.0 (Volta+) -> Float16, otherwise Float32.
    pub fn resolve(self, sm_major: u32) -> Dtype {
        match self {
            Dtype::Auto => {
                if sm_major >= 7 {
                    Dtype::Float16
                } else {
                    Dtype::Float32
                }
            }
            other => other,
        }
    }

    /// String suitable for kernel dispatch ("float32", "float16", "bfloat16").
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::Auto => "auto",
            Dtype::Float32 => "float32",
            Dtype::Float16 => "float16",
            Dtype::BFloat16 => "bfloat16",
        }
    }

    /// Whether this dtype should use hgemm (half GEMM) vs sgemm (single GEMM).
    pub fn use_hgemm(self) -> bool {
        matches!(self, Dtype::Float16 | Dtype::BFloat16)
    }
}

impl Default for Dtype {
    fn default() -> Self {
        Dtype::Auto
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Dtype {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Dtype::Auto),
            "float32" | "fp32" | "f32" => Ok(Dtype::Float32),
            "float16" | "fp16" | "f16" | "half" => Ok(Dtype::Float16),
            "bfloat16" | "bf16" => Ok(Dtype::BFloat16),
            _ => Err(format!(
                "unknown dtype '{}': expected auto, float32, float16, or bfloat16",
                s
            )),
        }
    }
}

/// Unique identifier for a sequence within a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, From, Into, Serialize, Deserialize)]
#[display("{_0}")]
pub struct SequenceId(pub u64);

/// Unique identifier for an incoming request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, From, Into, Serialize, Deserialize)]
#[display("{_0}")]
pub struct RequestId(pub u64);

/// Unique identifier for a KV-cache block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, From, Into, Serialize, Deserialize)]
#[display("{_0}")]
pub struct BlockId(pub u32);

/// A single token identifier in the model vocabulary.
pub type TokenId = u32;

/// Log-probability value.
pub type LogProb = f32;

/// Format constraint for guided decoding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// No constraint -- free-form text.
    #[serde(rename = "text")]
    Text,
    /// Force output to be valid JSON.
    #[serde(rename = "json_object")]
    JsonObject,
    /// Force output to conform to a JSON schema.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The JSON schema that output must satisfy.
        json_schema: serde_json::Value,
    },
    /// Force output to match a regex pattern.
    #[serde(rename = "regex")]
    Regex {
        /// The regex pattern.
        pattern: String,
    },
}

impl Default for ResponseFormat {
    fn default() -> Self {
        Self::Text
    }
}

/// Sampling parameters that control generation behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Controls randomness. 0.0 = greedy, higher = more random.
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k sampling cutoff.
    pub top_k: u32,
    /// Minimum probability threshold.
    pub min_p: f32,
    /// Penalise repeated tokens.
    pub repetition_penalty: f32,
    /// Penalise tokens proportional to their frequency so far.
    pub frequency_penalty: f32,
    /// Penalise tokens that have appeared at all.
    pub presence_penalty: f32,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Stop generation when any of these strings are produced.
    pub stop_strings: Vec<String>,
    /// If true, do not stop when the EOS token is generated.
    #[serde(default)]
    pub ignore_eos: bool,
    /// If set, return this many top log-probabilities per position.
    pub logprobs: Option<usize>,
    /// Deterministic sampling seed.
    pub seed: Option<u64>,
    /// Number of independent completions to generate; best is returned.
    pub best_of: usize,
    /// Enable beam search decoding with `best_of` beams.
    pub use_beam_search: bool,
    /// Guided decoding response format constraint.
    #[serde(default)]
    pub response_format: ResponseFormat,
    /// If true, return logprobs for prompt tokens as well as generated tokens.
    #[serde(default)]
    pub echo: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 256,
            stop_strings: Vec::new(),
            ignore_eos: false,
            logprobs: None,
            seed: None,
            best_of: 1,
            use_beam_search: false,
            response_format: ResponseFormat::default(),
            echo: false,
        }
    }
}

/// Reason a generation finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FinishReason {
    /// Hit the max-tokens limit.
    Length,
    /// Produced a stop string or EOS.
    Stop,
    /// Aborted by the scheduler or user.
    Abort,
}

/// Marker trait for types that can be configured from external sources.
pub trait Configurable: Send + Sync {}

/// Marker trait for types that support resetting to a clean state.
pub trait Resetable: Send + Sync {}

/// Marker trait for types bound to a GPU device.
pub trait GpuBound: Send + Sync {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtype_copy_and_display() {
        let sid = SequenceId(42);
        let sid2 = sid;
        assert_eq!(sid, sid2);
        assert_eq!(sid.to_string(), "42");

        let rid = RequestId(7);
        assert_eq!(rid.to_string(), "7");

        let bid = BlockId(99);
        assert_eq!(bid.to_string(), "99");
    }

    #[test]
    fn newtype_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SequenceId(1));
        set.insert(SequenceId(1));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn sampling_params_defaults() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.max_tokens, 256);
        assert_eq!(p.best_of, 1);
    }

    #[test]
    fn sampling_params_serde_roundtrip() {
        let p = SamplingParams::default();
        let json = serde_json::to_string(&p).unwrap();
        let p2: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p2.temperature, p.temperature);
    }

    #[test]
    fn finish_reason_variants() {
        assert_ne!(FinishReason::Length, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::Abort);
    }

    #[test]
    fn send_sync_assertions() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SequenceId>();
        assert_send_sync::<RequestId>();
        assert_send_sync::<BlockId>();
        assert_send_sync::<SamplingParams>();
        assert_send_sync::<FinishReason>();
        assert_send_sync::<Dtype>();
    }

    #[test]
    fn dtype_from_str() {
        assert_eq!("auto".parse::<Dtype>().unwrap(), Dtype::Auto);
        assert_eq!("float32".parse::<Dtype>().unwrap(), Dtype::Float32);
        assert_eq!("fp16".parse::<Dtype>().unwrap(), Dtype::Float16);
        assert_eq!("bfloat16".parse::<Dtype>().unwrap(), Dtype::BFloat16);
        assert_eq!("bf16".parse::<Dtype>().unwrap(), Dtype::BFloat16);
        assert!("invalid".parse::<Dtype>().is_err());
    }

    #[test]
    fn dtype_resolve() {
        assert_eq!(Dtype::Auto.resolve(7), Dtype::Float16);
        assert_eq!(Dtype::Auto.resolve(8), Dtype::Float16);
        assert_eq!(Dtype::Auto.resolve(9), Dtype::Float16);
        assert_eq!(Dtype::Auto.resolve(12), Dtype::Float16); // Blackwell sm_120/sm_122
        assert_eq!(Dtype::Auto.resolve(6), Dtype::Float32);
        assert_eq!(Dtype::Float32.resolve(8), Dtype::Float32);
    }

    #[test]
    fn dtype_size_and_half() {
        assert_eq!(Dtype::Float32.size_bytes(), 4);
        assert_eq!(Dtype::Float16.size_bytes(), 2);
        assert_eq!(Dtype::BFloat16.size_bytes(), 2);
        assert!(Dtype::Float16.is_half());
        assert!(Dtype::BFloat16.is_half());
        assert!(!Dtype::Float32.is_half());
    }

    #[test]
    fn dtype_display() {
        assert_eq!(Dtype::Auto.to_string(), "auto");
        assert_eq!(Dtype::Float16.to_string(), "float16");
    }

    #[test]
    fn dtype_serde_roundtrip() {
        let d = Dtype::Float16;
        let json = serde_json::to_string(&d).unwrap();
        let d2: Dtype = serde_json::from_str(&json).unwrap();
        assert_eq!(d, d2);
    }

    #[test]
    fn dtype_hgemm() {
        assert!(Dtype::Float16.use_hgemm());
        assert!(Dtype::BFloat16.use_hgemm());
        assert!(!Dtype::Float32.use_hgemm());
        assert!(!Dtype::Auto.use_hgemm());
    }
}
