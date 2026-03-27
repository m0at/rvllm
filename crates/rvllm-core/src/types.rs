//! Newtype wrappers, vocabulary types, sampling parameters, and marker traits.

use derive_more::{Display, From, Into};
use serde::{Deserialize, Serialize};

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
    }
}
