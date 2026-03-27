//! SSE streaming chunk types matching OpenAI's streaming format.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Delta content in a streaming chat chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// A single choice within a streaming completion chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct CompletionStreamChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

/// Streaming chunk for POST /v1/completions with stream=true.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct CompletionStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
}

/// A single choice within a streaming chat completion chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatStreamChoice {
    pub delta: ChatDelta,
    pub index: usize,
    pub finish_reason: Option<String>,
}

/// Streaming chunk for POST /v1/chat/completions with stream=true.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatCompletionStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

/// Format a chunk as an SSE data line.
pub fn format_sse_data<T: Serialize>(chunk: &T) -> String {
    let json = serde_json::to_string(chunk).unwrap_or_default();
    format!("data: {}\n\n", json)
}

/// The final SSE message indicating the stream is done.
pub const SSE_DONE: &str = "data: [DONE]\n\n";

impl CompletionStreamChunk {
    /// Create a chunk with incremental text for a single choice.
    pub fn new(id: &str, model: &str, index: usize, text: &str, finish_reason: Option<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: now,
            model: model.to_string(),
            choices: vec![CompletionStreamChoice {
                text: text.to_string(),
                index,
                logprobs: None,
                finish_reason,
            }],
        }
    }
}

impl ChatCompletionStreamChunk {
    /// Create the initial chunk that sets the assistant role.
    pub fn role_chunk(id: &str, model: &str) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model.to_string(),
            choices: vec![ChatStreamChoice {
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                index: 0,
                finish_reason: None,
            }],
        }
    }

    /// Create a content delta chunk.
    pub fn content_chunk(
        id: &str,
        model: &str,
        index: usize,
        content: &str,
        finish_reason: Option<String>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model.to_string(),
            choices: vec![ChatStreamChoice {
                delta: ChatDelta {
                    role: None,
                    content: Some(content.to_string()),
                },
                index,
                finish_reason,
            }],
        }
    }

    /// Create the final chunk with finish_reason and empty delta.
    pub fn finish_chunk(id: &str, model: &str, index: usize, reason: &str) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model.to_string(),
            choices: vec![ChatStreamChoice {
                delta: ChatDelta {
                    role: None,
                    content: None,
                },
                index,
                finish_reason: Some(reason.to_string()),
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_stream_chunk_serde() {
        let chunk = CompletionStreamChunk::new("cmpl-1", "model", 0, "hello", None);
        let json = serde_json::to_string(&chunk).unwrap();
        let back: CompletionStreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "cmpl-1");
        assert_eq!(back.choices[0].text, "hello");
        assert_eq!(back.object, "text_completion");
    }

    #[test]
    fn chat_stream_role_chunk() {
        let chunk = ChatCompletionStreamChunk::role_chunk("chatcmpl-1", "model");
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("assistant"));
        assert_eq!(chunk.object, "chat.completion.chunk");
    }

    #[test]
    fn chat_stream_content_chunk() {
        let chunk =
            ChatCompletionStreamChunk::content_chunk("chatcmpl-1", "model", 0, "Hello", None);
        let json = serde_json::to_string(&chunk).unwrap();
        let back: ChatCompletionStreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.choices[0].delta.content, Some("Hello".into()));
        assert!(back.choices[0].delta.role.is_none());
    }

    #[test]
    fn chat_stream_finish_chunk() {
        let chunk = ChatCompletionStreamChunk::finish_chunk("chatcmpl-1", "model", 0, "stop");
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".into()));
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.choices[0].delta.role.is_none());
    }

    #[test]
    fn format_sse_data_produces_valid_sse() {
        let chunk = CompletionStreamChunk::new("id", "m", 0, "hi", None);
        let sse = format_sse_data(&chunk);
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
        // The JSON should be parseable
        let json_part = sse.strip_prefix("data: ").unwrap().trim();
        let _: CompletionStreamChunk = serde_json::from_str(json_part).unwrap();
    }

    #[test]
    fn sse_done_format() {
        assert_eq!(SSE_DONE, "data: [DONE]\n\n");
    }

    #[test]
    fn chat_delta_skips_none_fields() {
        let delta = ChatDelta {
            role: None,
            content: Some("hi".into()),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(!json.contains("role"));
        assert!(json.contains("content"));
    }

    #[test]
    fn chat_message_in_response() {
        let msg = crate::types::request::ChatMessage {
            role: "assistant".into(),
            content: "test".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: crate::types::request::ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "assistant");
    }
}
