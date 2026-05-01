//! `POST /v1/completions` — legacy text-completion endpoint.
//!
//! Kept for OpenAI-SDK clients that still call it. Internally this
//! skips the chat template and feeds `prompt` directly through the
//! tokenizer.

use serde::{Deserialize, Serialize};

use crate::openai::chat::StopField;
use crate::openai::types::{FinishReason, Usage};
use crate::sampling::SamplingParams;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct CompletionRequest {
    pub model: String,
    /// Single string or array of strings (batch — v1 accepts only one).
    pub prompt: PromptField,
    pub max_tokens: Option<u32>,

    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,

    pub stop: Option<StopField>,
    pub stream: bool,

    pub n: Option<u32>,
    pub logit_bias: Option<serde_json::Value>,
    pub logprobs: Option<u32>,
    pub echo: Option<bool>,
    pub suffix: Option<String>,

    // OpenAI sampling-shaping params we don't honour. Captured here
    // so the completions handler rejects them with a clear 400
    // instead of silently dropping (mirrors the chat handler's
    // reject_v1_unsupported_chat — Codex4-4).
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub response_format: Option<serde_json::Value>,
    pub stream_options: Option<serde_json::Value>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            prompt: PromptField::One(String::new()),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            stop: None,
            stream: false,
            n: None,
            logit_bias: None,
            logprobs: None,
            echo: None,
            suffix: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_format: None,
            stream_options: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptField {
    One(String),
    Many(Vec<String>),
    /// Token-ID form: `[1, 2, 3]` or `[[1, 2, 3]]`. v1 rejects it to
    /// keep the happy path string-only.
    Tokens(Vec<u32>),
    TokensBatched(Vec<Vec<u32>>),
}

impl CompletionRequest {
    pub fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k,
            seed: self.seed,
        }
    }
}

// ─── Responses ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    /// Always null for v1 (we don't emit logprobs).
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_string_prompt() {
        let b = r#"{"model":"m","prompt":"hello"}"#;
        let r: CompletionRequest = serde_json::from_str(b).expect("parse");
        assert!(matches!(r.prompt, PromptField::One(s) if s == "hello"));
    }

    #[test]
    fn parses_array_prompt() {
        let b = r#"{"model":"m","prompt":["a","b"]}"#;
        let r: CompletionRequest = serde_json::from_str(b).expect("parse");
        assert!(matches!(r.prompt, PromptField::Many(v) if v == ["a","b"]));
    }
}
