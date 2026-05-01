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
    /// OpenAI 2025 rename of `max_tokens` — captured so we can 400
    /// rather than silently ignore. Both clients sometimes send only
    /// this field; we reject in `reject_v1_unsupported_chat` until
    /// the chat template + handler thread it through (alias of
    /// `max_tokens`, but the precedence rule needs explicit code).
    pub max_completion_tokens: Option<u32>,
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
    /// JSON mode / structured outputs. Without grammar-constrained
    /// decoding the server cannot guarantee the requested format, so
    /// silently ignoring it would mislead callers. Captured + 400'd.
    pub response_format: Option<serde_json::Value>,
    /// HF-style penalties — would require additional logits-processor
    /// kernels (we only ship `RVLLM_REPETITION_PENALTY` today which
    /// uses a different formula). Captured + 400'd until wired.
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// `stream_options`: clients use this to request usage in the
    /// final SSE chunk. Captured + 400'd; supporting it is a small
    /// follow-up but until then we reject so callers don't think
    /// they got it.
    pub stream_options: Option<serde_json::Value>,
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
            max_completion_tokens: None,
            stop: None,
            stream: false,
            n: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream_options: None,
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
///   * user / system: `content` is required (string OR parts array).
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
    pub content: Option<ChatContent>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Render `content` as a plain text string for the chat template.
    /// `None` and unsupported part shapes produce an empty string;
    /// content-part arrays are joined with single spaces between
    /// successive text parts (matching the typical OpenAI client
    /// behaviour). Use this anywhere the downstream consumer needs
    /// `&str` or `String`, not the structured form.
    pub fn content_text(&self) -> String {
        self.content.as_ref().map(ChatContent::to_text).unwrap_or_default()
    }
}

/// `content` field of an inbound chat message. OpenAI accepts either
/// a plain string or an array of typed parts. Today we honour the
/// `text` part type only; other types (image_url, input_audio, …)
/// are rejected at validation time so callers get a clear 400 rather
/// than a silent semantic loss.
#[derive(Debug)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

impl serde::Serialize for ChatContent {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ChatContent::Text(t) => s.serialize_str(t),
            ChatContent::Parts(parts) => parts.serialize(s),
        }
    }
}

impl From<String> for ChatContent {
    fn from(s: String) -> Self {
        ChatContent::Text(s)
    }
}

impl From<&str> for ChatContent {
    fn from(s: &str) -> Self {
        ChatContent::Text(s.to_string())
    }
}

impl ChatContent {
    /// Flatten to a plain text string. Multiple text parts are joined
    /// with a single space — the OpenAI ref clients do the same.
    /// Non-text parts are skipped here; the request validator (called
    /// before tokenisation) is responsible for rejecting them with a
    /// 400, so this fallback only fires on already-validated input.
    pub fn to_text(&self) -> String {
        match self {
            ChatContent::Text(s) => s.clone(),
            ChatContent::Parts(parts) => {
                let mut out = String::new();
                for p in parts {
                    if let ChatContentPart::Text { text } = p {
                        if !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str(text);
                    }
                }
                out
            }
        }
    }

    /// Walk the parts array (if any) and return the first part type
    /// we cannot render as text. Used by the request validator to
    /// 400 cleanly on `image_url` etc. Returns `None` for plain-string
    /// content or all-text parts.
    pub fn first_unsupported_part_type(&self) -> Option<&str> {
        match self {
            ChatContent::Text(_) => None,
            ChatContent::Parts(parts) => parts.iter().find_map(|p| match p {
                ChatContentPart::Text { .. } => None,
                ChatContentPart::Other { kind } => Some(kind.as_str()),
            }),
        }
    }
}

// Custom Deserialize: serde's `#[serde(untagged)]` would accept the
// String variant fine, but every Parts shape must dispatch through the
// part-level `type` discriminator. We hand-write the visitor so that
// (a) plain JSON strings hit ChatContent::Text directly and (b) any
// JSON array round-trips via ChatContentPart's typed enum below.
impl<'de> Deserialize<'de> for ChatContent {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        match v {
            serde_json::Value::String(s) => Ok(ChatContent::Text(s)),
            serde_json::Value::Array(_) => {
                let parts: Vec<ChatContentPart> =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ChatContent::Parts(parts))
            }
            serde_json::Value::Null => Ok(ChatContent::Text(String::new())),
            other => Err(serde::de::Error::custom(format!(
                "`content` must be a string or an array of content parts, got {}",
                match other {
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::Object(_) => "object",
                    _ => "unknown",
                }
            ))),
        }
    }
}

/// One element of a content-parts array. The `text` shape is the
/// only one we render today; any other `type` is captured as
/// [`ChatContentPart::Other`] so the request validator can 400 with
/// the actual unsupported type string in the payload, rather than
/// the request failing JSON-parse with a generic "unknown variant".
#[derive(Debug)]
pub enum ChatContentPart {
    /// `{"type":"text","text":"..."}`.
    Text { text: String },
    /// `{"type":"<anything>", ...}` — kind captured for validation.
    Other { kind: String },
}

impl serde::Serialize for ChatContentPart {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ChatContentPart::Text { text } => {
                let mut m = s.serialize_map(Some(2))?;
                m.serialize_entry("type", "text")?;
                m.serialize_entry("text", text)?;
                m.end()
            }
            ChatContentPart::Other { kind } => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry("type", kind)?;
                m.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ChatContentPart {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        let kind = v
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| serde::de::Error::custom("content part is missing `type`"))?;
        match kind {
            "text" => {
                let text = v
                    .get("text")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| {
                        serde::de::Error::custom("text content part is missing `text`")
                    })?
                    .to_string();
                Ok(ChatContentPart::Text { text })
            }
            other => Ok(ChatContentPart::Other { kind: other.to_string() }),
        }
    }
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
    pub tool_calls: Option<Vec<ChatDeltaToolCall>>,
}

/// One element of `delta.tool_calls` in a streaming chunk.
///
/// Distinct from the non-streaming [`ToolCall`] because OpenAI's
/// streaming spec REQUIRES an `index` so clients can stitch
/// arguments-fragments arriving across chunks back to the right call.
/// `id` / `type` / `function` are optional in the streaming shape:
/// the first chunk per call carries `id` + `type` + `function.name`
/// and (optionally) the first slice of `function.arguments`; later
/// chunks for the same call only need `index` + the next
/// `function.arguments` slice.
///
/// We currently flush a tool-call once it is fully parsed, so all
/// three optional fields are populated and `function.arguments`
/// contains the complete JSON-string payload — fully OpenAI-compliant
/// and what every client we have seen accepts.
#[derive(Clone, Debug, Serialize)]
pub struct ChatDeltaToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<crate::openai::types::ToolCallFunction>,
}

impl ChatDeltaToolCall {
    /// Build a fully-populated streaming tool-call entry from a
    /// completed [`ToolCall`] and its position in the assistant's
    /// emitted call list (0-based).
    pub fn from_complete(call: ToolCall, index: u32) -> Self {
        Self {
            index,
            id: Some(call.id),
            kind: Some(call.kind),
            function: Some(call.function),
        }
    }
}

impl ChatDelta {
    pub fn role(role: Role) -> Self {
        Self { role: Some(role), content: None, tool_calls: None }
    }
    pub fn content<S: Into<String>>(c: S) -> Self {
        Self { role: None, content: Some(c.into()), tool_calls: None }
    }
    /// Build a `delta.tool_calls` payload from completed tool-calls.
    /// Each call is paired with its 0-based `index` as required by
    /// the OpenAI streaming spec.
    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        let deltas = calls
            .into_iter()
            .enumerate()
            .map(|(i, c)| ChatDeltaToolCall::from_complete(c, i as u32))
            .collect();
        Self { role: None, content: None, tool_calls: Some(deltas) }
    }
    pub fn done() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_string_content() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("parse");
        let m = &req.messages[0];
        assert!(matches!(m.content.as_ref(), Some(ChatContent::Text(s)) if s == "hi"));
        assert_eq!(m.content_text(), "hi");
    }

    #[test]
    fn parses_null_assistant_content() {
        // Assistant tool-call message: content explicitly null, payload
        // in tool_calls. Must round-trip without error.
        let body = r#"{"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":[]}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("parse");
        // serde maps `null` for an Option field to None; ChatContent
        // never sees the value at all here.
        let m = &req.messages[0];
        assert!(m.content.is_none() || matches!(m.content, Some(ChatContent::Text(ref s)) if s.is_empty()));
        assert_eq!(m.content_text(), "");
    }

    #[test]
    fn parses_text_part_array() {
        let body = r#"{
            "model":"m",
            "messages":[{
                "role":"user",
                "content":[{"type":"text","text":"hello"},{"type":"text","text":"world"}]
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("parse");
        let m = &req.messages[0];
        match m.content.as_ref().expect("content") {
            ChatContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(&parts[0], ChatContentPart::Text { text } if text == "hello"));
            }
            _ => panic!("expected Parts variant"),
        }
        assert_eq!(m.content_text(), "hello world");
        assert_eq!(m.content.as_ref().unwrap().first_unsupported_part_type(), None);
    }

    #[test]
    fn parses_unsupported_part_for_validator() {
        // Unknown part types must NOT fail JSON parse — the request
        // validator is responsible for emitting a clean 400 with the
        // unsupported `type` name.
        let body = r#"{
            "model":"m",
            "messages":[{
                "role":"user",
                "content":[{"type":"image_url","image_url":{"url":"http://example/x.png"}}]
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("parse");
        let bad = req.messages[0]
            .content
            .as_ref()
            .expect("content")
            .first_unsupported_part_type();
        assert_eq!(bad, Some("image_url"));
    }

    #[test]
    fn rejects_non_string_non_array_content() {
        // OpenAI rejects `{"content": 42}` etc. We mirror that at the
        // JSON-parse boundary.
        let body = r#"{"model":"m","messages":[{"role":"user","content":42}]}"#;
        let r: Result<ChatCompletionRequest, _> = serde_json::from_str(body);
        assert!(r.is_err(), "numeric content must reject at parse time");
    }

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

    #[test]
    fn stream_delta_tool_calls_carry_index() {
        use crate::openai::types::{new_tool_call_id, ToolCall, ToolCallFunction};
        let d = ChatDelta::tool_calls(vec![
            ToolCall {
                id: new_tool_call_id(),
                kind: "function",
                function: ToolCallFunction { name: "a".into(), arguments: "{}".into() },
            },
            ToolCall {
                id: new_tool_call_id(),
                kind: "function",
                function: ToolCallFunction { name: "b".into(), arguments: "{}".into() },
            },
        ]);
        let v: serde_json::Value = serde_json::to_value(&d).expect("ok");
        let calls = v["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["index"], 0);
        assert_eq!(calls[1]["index"], 1);
        // OpenAI shape: id/type/function present on the first (and only)
        // delta chunk per call when we flush a fully-parsed call.
        assert!(calls[0]["id"].is_string());
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "a");
        assert_eq!(calls[1]["function"]["name"], "b");
    }
}
