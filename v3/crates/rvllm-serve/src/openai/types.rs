//! Shared OpenAI primitives.

use serde::{Deserialize, Serialize};

/// Chat role. `system`, `user`, `assistant`, and `tool` are what the
/// API accepts; we reject anything else at parse time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Why a completion stopped. Mirrors OpenAI's `finish_reason`.
///
/// Cycle 37 (codex audit): added `Deserialize` so tests can parse the
/// JSON envelope without a separate stub type. No production handler
/// reads this field as input — the `Serialize` direction is what
/// matters at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Hit the model's natural stop (EOS token).
    Stop,
    /// Hit `max_tokens` ceiling.
    Length,
    /// Client disconnected or request timed out.
    Cancelled,
    /// Content filter (we don't run one yet, kept for forward-compat).
    ContentFilter,
    /// Response contained one or more tool calls; see `tool_calls`.
    ToolCalls,
}

/// OpenAI `tool_calls[i]` entry. Only `function`-kind calls are emitted.
#[derive(Clone, Debug, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded string per OpenAI spec (not a JSON object).
    pub arguments: String,
}

/// Generate a short tool-call id (`call_<uuid-nodashes-prefix>`).
pub fn new_tool_call_id() -> String {
    let u = uuid::Uuid::new_v4();
    let s = u.simple().to_string();
    format!("call_{}", &s[..16])
}

/// Token usage counts. Returned on the final chunk / non-stream body.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

/// `created` timestamp helper. OpenAI uses Unix seconds.
pub fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Generate an OpenAI-style completion id (`cmpl-<uuid-nodashes>`).
pub fn new_completion_id() -> String {
    let u = uuid::Uuid::new_v4();
    format!("cmpl-{}", u.simple())
}

/// Generate an OpenAI-style chat completion id (`chatcmpl-<uuid>`).
pub fn new_chat_completion_id() -> String {
    let u = uuid::Uuid::new_v4();
    format!("chatcmpl-{}", u.simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_roundtrip() {
        let j = serde_json::to_string(&Role::Assistant);
        assert_eq!(j.ok().as_deref(), Some("\"assistant\""));
        let r: Role = serde_json::from_str("\"user\"").expect("valid");
        assert_eq!(r, Role::User);
    }

    #[test]
    fn finish_reason_serializes_snake() {
        let j = serde_json::to_string(&FinishReason::ContentFilter).expect("ok");
        assert_eq!(j, "\"content_filter\"");
    }

    // === Cycle 37 finish-reason serde roundtrip tests ===

    #[test]
    fn finish_reason_all_variants_roundtrip() {
        for (variant, expected) in [
            (FinishReason::Stop, "\"stop\""),
            (FinishReason::Length, "\"length\""),
            (FinishReason::Cancelled, "\"cancelled\""),
            (FinishReason::ContentFilter, "\"content_filter\""),
            (FinishReason::ToolCalls, "\"tool_calls\""),
        ] {
            let j = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(j, expected, "serialize {variant:?}");
            let back: FinishReason =
                serde_json::from_str(expected).expect("deserialize");
            assert_eq!(back, variant, "roundtrip {variant:?}");
        }
    }

    #[test]
    fn finish_reason_unknown_string_rejected() {
        let r: Result<FinishReason, _> =
            serde_json::from_str("\"definitely_not_a_finish_reason\"");
        assert!(r.is_err(), "unknown finish_reason must fail to parse");
    }

    #[test]
    fn usage_sums() {
        let u = Usage::new(10, 3);
        assert_eq!(u.total_tokens, 13);
    }

    #[test]
    fn ids_prefixed() {
        assert!(new_completion_id().starts_with("cmpl-"));
        assert!(new_chat_completion_id().starts_with("chatcmpl-"));
    }
}
