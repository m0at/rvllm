//! `POST /v1/chat/completions`.
//!
//! Schemas track the OpenAI spec as of November 2025. Unknown request
//! fields are ignored via `#[serde(default)]` on the top-level struct
//! (tolerant parse). Response bodies emit exactly the spec'd fields.
//!
//! Streaming vs non-streaming is selected by `stream: true` in the
//! body. The handler dispatches to the right builder; both share the
//! same worker channel plumbing in [`crate::worker`].

use serde::{Deserialize, Serialize};

use crate::openai::types::{FinishReason, Role, ToolCall, Usage};
use crate::sampling::SamplingParams;

/// Request body. Tolerant of extra fields — the OpenAI SDK sometimes
/// injects newer ones (e.g. `stream_options`, `response_format`).
///
/// Fields deliberately missing for v1: `tools`, `tool_choice`,
/// `function_call`, `logit_bias`, `logprobs`, `top_logprobs`,
/// `n`. The handler rejects with `invalid_request_error` if any of
/// those show up as non-default.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    // Sampling (see [`SamplingParams::ensure_supported`]).
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,

    pub max_tokens: Option<u32>,
    pub stop: Option<StopField>,
    pub stream: bool,

    // Accepted-and-rejected fields — kept so the server complains
    // instead of silently downgrading.
    pub n: Option<u32>,
    pub logit_bias: Option<serde_json::Value>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            max_tokens: None,
            stop: None,
            stream: false,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
        }
    }
}

impl ChatCompletionRequest {
    /// Extract validated sampling params. Reject v1-unsupported knobs.
    pub fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k,
            seed: self.seed,
        }
    }
}

/// `stop` can be a string, array of strings, or null.
#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    /// Flatten to a vec, skipping empty strings (OpenAI ignores them).
    pub fn into_vec(self) -> Vec<String> {
        let v = match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        };
        v.into_iter().filter(|s| !s.is_empty()).collect()
    }
}

/// A single chat message in the request.
///
/// OpenAI's Chat Completions spec allows four shapes here:
///   * user / system: `content` is required (string or parts array).
///   * assistant with prose: `content` is a string.
///   * assistant with tool calls: `content` is `null` and `tool_calls`
///     carries the call payload (id + function + arguments).
///   * tool result: `content` carries the result, `tool_call_id` links
///     back to the assistant's call id.
///
/// We keep every field optional at the Rust level and let the chat
/// template enforce shape — Gemma 4's template already handles the
/// full matrix. The previous struct required `content: String` and
/// 422'd the moment zeroclaw replayed an assistant-with-tool-calls
/// message on a follow-up turn.
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

// ─── Response bodies ─────────────────────────────────────────────────

/// Non-streaming response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatAssistantMessage,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Serialize)]
pub struct ChatAssistantMessage {
    pub role: Role,
    /// OpenAI spec: `null` when the assistant emitted tool calls
    /// instead of prose. Clients discriminate on presence.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ─── Streaming (SSE) chunks ──────────────────────────────────────────

/// One SSE chunk. OpenAI streams `data: {..}\n\n` per token, ending
/// with `data: [DONE]\n\n`.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub finish_reason: Option<FinishReason>,
}

/// Incremental content. First chunk carries `role: assistant`, later
/// chunks only carry `content`. The final chunk carries
/// `finish_reason` and an empty delta.
#[derive(Debug, Default, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatDelta {
    pub fn role(role: Role) -> Self {
        Self { role: Some(role), content: None, tool_calls: None }
    }
    pub fn content<S: Into<String>>(c: S) -> Self {
        Self { role: None, content: Some(c.into()), tool_calls: None }
    }
    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Self { role: None, content: None, tool_calls: Some(calls) }
    }
    pub fn done() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_request() {
        let body = r#"{
            "model": "gemma-4-31b",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("parse");
        assert_eq!(req.model, "gemma-4-31b");
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
    }

    #[test]
    fn parses_stop_string() {
        let body = r####"{"model":"m","messages":[],"stop":"###"}"####;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("parse");
        let stop = req.stop.expect("stop present").into_vec();
        assert_eq!(stop, vec!["###"]);
    }

    #[test]
    fn parses_stop_array() {
        let body = r#"{"model":"m","messages":[],"stop":["a","","b"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("parse");
        let stop = req.stop.expect("stop present").into_vec();
        assert_eq!(stop, vec!["a", "b"]); // empties filtered
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let body = r#"{
            "model": "m",
            "messages": [],
            "response_format": {"type": "json"},
            "future_feature": 42
        }"#;
        let r: Result<ChatCompletionRequest, _> = serde_json::from_str(body);
        assert!(r.is_ok());
    }

    #[test]
    fn stream_delta_omits_null_fields() {
        let d = ChatDelta::content("hello");
        let j = serde_json::to_string(&d).expect("ok");
        assert_eq!(j, r#"{"content":"hello"}"#);
        let done = serde_json::to_string(&ChatDelta::done()).expect("ok");
        assert_eq!(done, "{}");
    }

    #[test]
    fn stream_delta_tool_calls_shape() {
        use crate::openai::types::{new_tool_call_id, ToolCall, ToolCallFunction};
        let d = ChatDelta::tool_calls(vec![ToolCall {
            id: new_tool_call_id(),
            kind: "function",
            function: ToolCallFunction { name: "foo".into(), arguments: "{}".into() },
        }]);
        let j = serde_json::to_string(&d).expect("ok");
        assert!(j.contains(r#""tool_calls""#));
        assert!(j.contains(r#""function""#));
        assert!(!j.contains(r#""content""#));
    }
}
