use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    pub stop: Option<StopSpec>,
    pub n: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    Null(()),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub text: Option<String>,
    pub image_url: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct PreparedChat {
    pub prompt: String,
    pub max_tokens: usize,
    pub stream: bool,
    pub stop: Vec<String>,
}

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub error_type: &'static str,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            error_type: "invalid_request_error",
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
            error_type: "invalid_request_error",
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
            error_type: "server_error",
        }
    }
}

impl MessageContent {
    fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p.kind.as_deref() {
                    Some("text") | None => p.text.as_deref(),
                    Some("image_url") | Some("image") => Some(
                        "[image attached: rvLLM text path needs a vision observer/tool result to describe pixels]",
                    ),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
            MessageContent::Null(()) => String::new(),
        }
    }
}

impl StopSpec {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSpec::One(s) => vec![s],
            StopSpec::Many(v) => v,
        }
    }
}

pub fn prepare_chat_request(
    req: ChatCompletionRequest,
    served_model: &str,
    default_max_tokens: usize,
    default_system_prompt: Option<&str>,
) -> Result<PreparedChat, ApiError> {
    if req.model != served_model {
        return Err(ApiError::not_found(format!(
            "model '{}' is not served by this rvllm-server; available model is '{}'",
            req.model, served_model
        )));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid("only n=1 is supported"));
    }
    if req.temperature.unwrap_or(0.0) != 0.0 {
        return Err(ApiError::invalid(
            "only greedy decoding is supported; set temperature to 0",
        ));
    }

    let max_tokens = req.max_tokens.unwrap_or(default_max_tokens);
    if max_tokens == 0 {
        return Err(ApiError::invalid("max_tokens must be > 0"));
    }
    if req.messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }

    let messages = apply_default_system_prompt(req.messages, default_system_prompt);

    Ok(PreparedChat {
        prompt: render_gemma_chat(&messages)?,
        max_tokens,
        stream: req.stream.unwrap_or(false),
        stop: req.stop.map(StopSpec::into_vec).unwrap_or_default(),
    })
}

fn apply_default_system_prompt(
    mut messages: Vec<ChatMessage>,
    default_system_prompt: Option<&str>,
) -> Vec<ChatMessage> {
    let Some(prompt) = default_system_prompt
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return messages;
    };
    messages.insert(
        0,
        ChatMessage {
            role: "system".into(),
            content: MessageContent::Text(prompt.into()),
        },
    );
    messages
}

pub fn render_gemma_chat(messages: &[ChatMessage]) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }

    let mut out = String::new();
    let mut system = String::new();
    let mut saw_turn = false;

    for msg in messages {
        let role = msg.role.as_str();
        let text = msg.content.text();
        match role {
            "system" | "developer" => {
                append_system(&mut system, &text);
            }
            "user" => {
                let merged = if system.is_empty() {
                    text
                } else {
                    let mut s = String::new();
                    s.push_str(system.trim_end());
                    s.push_str("\n\n");
                    s.push_str(text.trim_end());
                    system.clear();
                    s
                };
                push_turn(&mut out, "user", &merged);
                saw_turn = true;
            }
            "assistant" => {
                push_turn(&mut out, "model", &text);
                saw_turn = true;
            }
            other => {
                return Err(ApiError::invalid(format!(
                    "unsupported message role for Gemma chat template: {other}"
                )));
            }
        }
    }

    if !system.is_empty() {
        push_turn(&mut out, "user", &system);
        saw_turn = true;
    }
    if !saw_turn {
        return Err(ApiError::invalid(
            "messages must include a user or assistant turn",
        ));
    }

    out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    Ok(out)
}

pub fn created_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn completion_id(created: u64) -> String {
    let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-rvllm-{created}-{n}")
}

pub fn models_response(model: &str) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "rvllm"
        }]
    })
}

pub fn completion_response(
    id: &str,
    model: &str,
    created: u64,
    content: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

pub fn stream_role_chunk(id: &str, model: &str, created: u64) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant" },
            "finish_reason": null
        }]
    })
}

pub fn stream_content_chunk(id: &str, model: &str, created: u64, content: &str) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "content": content },
            "finish_reason": null
        }]
    })
}

pub fn stream_finish_chunk(id: &str, model: &str, created: u64) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    })
}

pub fn error_response(err: &ApiError) -> Value {
    json!({
        "error": {
            "message": err.message,
            "type": err.error_type,
            "param": null,
            "code": null
        }
    })
}

pub fn stream_text_chunks(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if cur.len() >= 64 || ch == '\n' || ch == ' ' {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn append_system(dst: &mut String, text: &str) {
    if !dst.is_empty() {
        dst.push_str("\n\n");
    }
    dst.push_str(text.trim_end());
}

fn push_turn(out: &mut String, role: &str, text: &str) {
    out.push_str("<|turn>");
    out.push_str(role);
    out.push('\n');
    out.push_str(text.trim_end());
    out.push_str("<turn|>\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: MessageContent::Text(content.into()),
        }
    }

    #[test]
    fn renders_gemma_turns() {
        let prompt = render_gemma_chat(&[msg("user", "hello")]).unwrap();
        assert_eq!(
            prompt,
            "<|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn folds_system_into_first_user() {
        let prompt = render_gemma_chat(&[msg("system", "be brief"), msg("user", "hi")]).unwrap();
        assert!(prompt.contains("be brief\n\nhi"));
    }

    #[test]
    fn injects_default_system_prompt() {
        let req = ChatCompletionRequest {
            model: "served".into(),
            messages: vec![msg("system", "request system"), msg("user", "hi")],
            max_tokens: Some(8),
            temperature: Some(0.0),
            stream: None,
            stop: None,
            n: None,
        };
        let prepared = prepare_chat_request(req, "served", 16, Some("server system")).unwrap();
        assert!(prepared
            .prompt
            .contains("server system\n\nrequest system\n\nhi"));
    }
}
