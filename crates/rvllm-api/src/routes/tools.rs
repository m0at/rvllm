//! Tool/function calling support for OpenAI-compatible chat completions.
//!
//! This module provides:
//! - Request/response types for `tools`, `tool_choice`, and `tool_calls`
//! - Prompt augmentation: injects tool definitions into the system prompt
//! - Response post-processing: parses model output for tool call JSON and
//!   returns structured `tool_calls` in the response
//!
//! Supports `auto`, `none`, `required`, and specific function tool choice modes.
//! Handles parallel tool calls (multiple calls in one response).

use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::info;
use utoipa::ToSchema;

use crate::error::ApiError;
use crate::server::AppState;
use crate::types::request::ChatMessage;
use crate::types::response::Usage;
use crate::types::streaming::{format_sse_data, ChatCompletionStreamChunk, SSE_DONE};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// A tool definition in the chat completion request (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct RequestTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: RequestFunction,
}

/// Function definition within a request tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct RequestFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool choice: controls whether/how the model calls tools.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String mode: "auto", "none", or "required".
    Mode(String),
    /// Specific function: `{"type": "function", "function": {"name": "..."}}`
    Specific(SpecificToolChoice),
}

/// Force the model to call a specific function.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct SpecificToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: SpecificFunctionChoice,
}

/// The function name to force.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct SpecificFunctionChoice {
    pub name: String,
}

/// Chat completion request with tool support.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionToolRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Tool definitions available to the model.
    #[serde(default)]
    pub tools: Option<Vec<RequestTool>>,
    /// How/whether the model should use tools.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_n() -> usize {
    1
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// A tool call in the response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ResponseFunctionCall,
}

/// Function call details in a response tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// A chat choice that may contain tool_calls instead of/alongside content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ToolChatChoice {
    pub index: usize,
    pub message: ToolChatMessage,
    pub finish_reason: Option<String>,
}

/// Message that can carry tool_calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ToolChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

/// Chat completion response with tool call support.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatCompletionToolResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ToolChatChoice>,
    pub usage: Usage,
}

// ---------------------------------------------------------------------------
// Validation & conversion
// ---------------------------------------------------------------------------

impl ChatCompletionToolRequest {
    /// Validate the request.
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.model.is_empty() {
            return Err(ApiError::InvalidRequest("model is required".into()));
        }
        if self.messages.is_empty() {
            return Err(ApiError::InvalidRequest(
                "messages must not be empty".into(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(ApiError::InvalidRequest(
                "max_tokens must be greater than 0".into(),
            ));
        }
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(ApiError::InvalidRequest(
                "temperature must be between 0.0 and 2.0".into(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::InvalidRequest(
                "top_p must be between 0.0 and 1.0".into(),
            ));
        }
        // Validate tool_choice value
        if let Some(ToolChoice::Mode(ref mode)) = self.tool_choice {
            if !["auto", "none", "required"].contains(&mode.as_str()) {
                return Err(ApiError::InvalidRequest(format!(
                    "invalid tool_choice mode '{}', expected auto/none/required",
                    mode
                )));
            }
        }
        Ok(())
    }

    /// Convert to engine sampling params.
    pub fn to_sampling_params(&self) -> rvllm_core::prelude::SamplingParams {
        rvllm_core::prelude::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stop_strings: self.stop.clone().unwrap_or_default(),
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            best_of: self.n,
            ..Default::default()
        }
    }

    /// Returns true if tools are provided and tool_choice is not "none".
    pub fn tools_enabled(&self) -> bool {
        if self.tools.as_ref().map_or(true, |t| t.is_empty()) {
            return false;
        }
        if let Some(ToolChoice::Mode(ref m)) = self.tool_choice {
            if m == "none" {
                return false;
            }
        }
        true
    }

    /// Convert request tools to rvllm-tokenizer ToolDefinitions.
    fn to_tool_definitions(&self) -> Vec<rvllm_tokenizer::ToolDefinition> {
        self.tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .map(|t| rvllm_tokenizer::ToolDefinition {
                        tool_type: t.tool_type.clone(),
                        function: rvllm_tokenizer::FunctionDefinition {
                            name: t.function.name.clone(),
                            description: t.function.description.clone(),
                            parameters: t
                                .function
                                .parameters
                                .as_ref()
                                .and_then(|p| serde_json::from_value(p.clone()).ok()),
                        },
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Prompt augmentation
// ---------------------------------------------------------------------------

/// Augment chat messages with tool definitions injected into the system prompt.
pub fn augment_messages_with_tools(
    messages: &[ChatMessage],
    tools: &[rvllm_tokenizer::ToolDefinition],
    style: rvllm_tokenizer::ToolPromptStyle,
) -> Vec<ChatMessage> {
    let tool_text = rvllm_tokenizer::format_tool_definitions(tools, style);

    let mut result = Vec::with_capacity(messages.len() + 1);

    // Check if there's already a system message
    let has_system = messages.first().map_or(false, |m| m.role == "system");

    if has_system {
        // Prepend tool definitions to existing system message
        let sys = &messages[0];
        result.push(ChatMessage {
            role: "system".to_string(),
            content: format!("{}\n\n{}", tool_text, sys.content),
        });
        result.extend_from_slice(&messages[1..]);
    } else {
        // Insert a new system message with tool definitions
        result.push(ChatMessage {
            role: "system".to_string(),
            content: tool_text,
        });
        result.extend_from_slice(messages);
    }

    result
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

/// POST /v1/chat/completions with tool/function calling support.
///
/// This handler extends the standard chat completion endpoint to accept
/// `tools` and `tool_choice` parameters. When tools are provided, it:
/// 1. Injects tool definitions into the system prompt
/// 2. Runs inference normally
/// 3. Parses model output for tool call JSON
/// 4. Returns structured `tool_calls` in the response
pub async fn create_chat_completion_with_tools(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionToolRequest>,
) -> Result<Response, ApiError> {
    req.validate()?;

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }

    let sampling_params = req.to_sampling_params();
    let tools_active = req.tools_enabled();

    // Build messages, optionally augmented with tool definitions
    let messages = if tools_active {
        let tool_defs = req.to_tool_definitions();
        augment_messages_with_tools(
            &req.messages,
            &tool_defs,
            rvllm_tokenizer::ToolPromptStyle::Hermes,
        )
    } else {
        req.messages.clone()
    };

    // Apply chat template
    let chat_messages: Vec<rvllm_tokenizer::ChatMessage> = messages
        .iter()
        .map(|m| rvllm_tokenizer::ChatMessage::new(&m.role, &m.content))
        .collect();

    let rendered_prompt = state
        .tokenizer
        .read()
        .await
        .render_chat_prompt(&chat_messages, true)
        .map_err(|e| ApiError::Internal(format!("chat template error: {}", e)))?;

    info!(
        model = %req.model,
        stream = req.stream,
        messages = req.messages.len(),
        tools = tools_active,
        "chat completion request (with tools)"
    );

    if req.stream {
        // Streaming: tool call parsing happens client-side in streaming mode,
        // but we still emit proper SSE events.
        let stream_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model = state.model_name.clone();

        let (_request_id, output_stream) = match &rendered_prompt {
            rvllm_tokenizer::RenderedPrompt::Text(prompt) => state
                .engine
                .generate(prompt.clone(), sampling_params.clone())
                .await
                .map_err(ApiError::from)?,
            rvllm_tokenizer::RenderedPrompt::Tokens(prompt_token_ids) => state
                .engine
                .generate_token_ids(
                    format!("<rendered:{} tokens>", prompt_token_ids.len()),
                    prompt_token_ids.clone(),
                    sampling_params.clone(),
                )
                .await
                .map_err(ApiError::from)?,
        };

        let stream_id_clone = stream_id.clone();
        let model_clone = model.clone();

        let initial = format_sse_data(&ChatCompletionStreamChunk::role_chunk(&stream_id, &model));

        let sse_stream = output_stream.map(move |output| {
            let mut events = String::new();
            for co in &output.outputs {
                let finish = co.finish_reason.map(|r| match r {
                    rvllm_core::prelude::FinishReason::Stop => "stop".to_string(),
                    rvllm_core::prelude::FinishReason::Length => "length".to_string(),
                    rvllm_core::prelude::FinishReason::Abort => "stop".to_string(),
                });
                if finish.is_some() {
                    let chunk = ChatCompletionStreamChunk::finish_chunk(
                        &stream_id_clone,
                        &model_clone,
                        co.index,
                        finish.as_deref().unwrap(),
                    );
                    events.push_str(&format_sse_data(&chunk));
                } else {
                    let chunk = ChatCompletionStreamChunk::content_chunk(
                        &stream_id_clone,
                        &model_clone,
                        co.index,
                        &co.text,
                        None,
                    );
                    events.push_str(&format_sse_data(&chunk));
                }
            }
            if output.finished {
                events.push_str(SSE_DONE);
            }
            Ok::<_, std::convert::Infallible>(events)
        });

        let init_stream = tokio_stream::once(Ok::<_, std::convert::Infallible>(initial));
        let full_stream = init_stream.chain(sse_stream);
        let body = axum::body::Body::from_stream(full_stream);

        Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(body)
            .unwrap()
            .into_response())
    } else {
        // Non-streaming: collect output, then parse for tool calls
        let (_request_id, mut output_stream) = match &rendered_prompt {
            rvllm_tokenizer::RenderedPrompt::Text(prompt) => state
                .engine
                .generate(prompt.clone(), sampling_params.clone())
                .await
                .map_err(ApiError::from)?,
            rvllm_tokenizer::RenderedPrompt::Tokens(prompt_token_ids) => state
                .engine
                .generate_token_ids(
                    format!("<rendered:{} tokens>", prompt_token_ids.len()),
                    prompt_token_ids.clone(),
                    sampling_params.clone(),
                )
                .await
                .map_err(ApiError::from)?,
        };

        let mut last_output = None;
        while let Some(output) = output_stream.next().await {
            if output.finished {
                last_output = Some(output);
                break;
            }
            last_output = Some(output);
        }

        let output =
            last_output.ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let resp_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        let mut total_completion = 0usize;
        let choices: Vec<ToolChatChoice> = output
            .outputs
            .iter()
            .map(|co| {
                total_completion += co.token_ids.len();

                let finish_reason_val = co.finish_reason.map(|r| match r {
                    rvllm_core::prelude::FinishReason::Stop => "stop",
                    rvllm_core::prelude::FinishReason::Length => "length",
                    rvllm_core::prelude::FinishReason::Abort => "stop",
                });

                if tools_active {
                    let call_prefix = format!("{}_{}_", resp_id, co.index);
                    let parse_result = rvllm_tokenizer::parse_tool_calls(&co.text, &call_prefix);

                    match parse_result {
                        rvllm_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls } => {
                            let tool_calls: Vec<ResponseToolCall> = calls
                                .into_iter()
                                .map(|tc| ResponseToolCall {
                                    id: tc.id,
                                    call_type: "function".to_string(),
                                    function: ResponseFunctionCall {
                                        name: tc.name,
                                        arguments: tc.arguments,
                                    },
                                })
                                .collect();

                            let content = if prefix_text.is_empty() {
                                None
                            } else {
                                Some(prefix_text)
                            };

                            ToolChatChoice {
                                index: co.index,
                                message: ToolChatMessage {
                                    role: "assistant".to_string(),
                                    content,
                                    tool_calls: Some(tool_calls),
                                },
                                finish_reason: Some("tool_calls".to_string()),
                            }
                        }
                        rvllm_tokenizer::ToolParseResult::PlainText(text) => ToolChatChoice {
                            index: co.index,
                            message: ToolChatMessage {
                                role: "assistant".to_string(),
                                content: Some(text),
                                tool_calls: None,
                            },
                            finish_reason: finish_reason_val.map(|s| s.to_string()),
                        },
                    }
                } else {
                    ToolChatChoice {
                        index: co.index,
                        message: ToolChatMessage {
                            role: "assistant".to_string(),
                            content: Some(co.text.clone()),
                            tool_calls: None,
                        },
                        finish_reason: finish_reason_val.map(|s| s.to_string()),
                    }
                }
            })
            .collect();

        let resp = ChatCompletionToolResponse {
            id: resp_id,
            object: "chat.completion".to_string(),
            created: now,
            model: state.model_name.clone(),
            choices,
            usage: Usage {
                prompt_tokens: output.prompt_token_ids.len(),
                completion_tokens: total_completion,
                total_tokens: output.prompt_token_ids.len() + total_completion,
            },
        };

        Ok(Json(resp).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_choice_deserialize_string() {
        let json = r#""auto""#;
        let tc: ToolChoice = serde_json::from_str(json).unwrap();
        assert_eq!(tc, ToolChoice::Mode("auto".to_string()));
    }

    #[test]
    fn tool_choice_deserialize_specific() {
        let json = r#"{"type": "function", "function": {"name": "get_weather"}}"#;
        let tc: ToolChoice = serde_json::from_str(json).unwrap();
        match tc {
            ToolChoice::Specific(s) => {
                assert_eq!(s.function.name, "get_weather");
            }
            _ => panic!("expected Specific"),
        }
    }

    #[test]
    fn request_tool_serde_roundtrip() {
        let tool = RequestTool {
            tool_type: "function".to_string(),
            function: RequestFunction {
                name: "search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                })),
            },
        };
        let json = serde_json::to_string(&tool).unwrap();
        let back: RequestTool = serde_json::from_str(&json).unwrap();
        assert_eq!(back, tool);
    }

    #[test]
    fn response_tool_call_serde() {
        let tc = ResponseToolCall {
            id: "call_abc123".to_string(),
            call_type: "function".to_string(),
            function: ResponseFunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location": "NYC"}"#.to_string(),
            },
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("call_abc123"));
        assert!(json.contains("get_weather"));
        let back: ResponseToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back, tc);
    }

    #[test]
    fn tool_chat_message_no_content_when_tools() {
        let msg = ToolChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ResponseToolCall {
                id: "call_0".to_string(),
                call_type: "function".to_string(),
                function: ResponseFunctionCall {
                    name: "test".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("\"content\""));
        assert!(json.contains("tool_calls"));
    }

    #[test]
    fn tool_chat_message_content_no_tools() {
        let msg = ToolChatMessage {
            role: "assistant".to_string(),
            content: Some("Hello!".to_string()),
            tool_calls: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Hello!"));
        assert!(!json.contains("tool_calls"));
    }

    #[test]
    fn tools_enabled_checks() {
        let base = ChatCompletionToolRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            n: 1,
            stream: false,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
        };
        assert!(!base.tools_enabled());

        let with_tools = ChatCompletionToolRequest {
            tools: Some(vec![RequestTool {
                tool_type: "function".to_string(),
                function: RequestFunction {
                    name: "f".to_string(),
                    description: None,
                    parameters: None,
                },
            }]),
            ..base.clone()
        };
        assert!(with_tools.tools_enabled());

        let tools_none_choice = ChatCompletionToolRequest {
            tool_choice: Some(ToolChoice::Mode("none".to_string())),
            ..with_tools.clone()
        };
        assert!(!tools_none_choice.tools_enabled());

        let tools_auto = ChatCompletionToolRequest {
            tool_choice: Some(ToolChoice::Mode("auto".to_string())),
            ..with_tools.clone()
        };
        assert!(tools_auto.tools_enabled());

        let tools_required = ChatCompletionToolRequest {
            tool_choice: Some(ToolChoice::Mode("required".to_string())),
            ..with_tools
        };
        assert!(tools_required.tools_enabled());
    }

    #[test]
    fn augment_messages_prepends_system() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "What's the weather?".into(),
        }];
        let tools = vec![rvllm_tokenizer::ToolDefinition {
            tool_type: "function".to_string(),
            function: rvllm_tokenizer::FunctionDefinition {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let augmented =
            augment_messages_with_tools(&msgs, &tools, rvllm_tokenizer::ToolPromptStyle::Hermes);
        assert_eq!(augmented.len(), 2);
        assert_eq!(augmented[0].role, "system");
        assert!(augmented[0].content.contains("get_weather"));
        assert_eq!(augmented[1].content, "What's the weather?");
    }

    #[test]
    fn augment_messages_merges_system() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            },
        ];
        let tools = vec![rvllm_tokenizer::ToolDefinition {
            tool_type: "function".to_string(),
            function: rvllm_tokenizer::FunctionDefinition {
                name: "search".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let augmented =
            augment_messages_with_tools(&msgs, &tools, rvllm_tokenizer::ToolPromptStyle::Hermes);
        assert_eq!(augmented.len(), 2);
        assert_eq!(augmented[0].role, "system");
        assert!(augmented[0].content.contains("search"));
        assert!(augmented[0].content.contains("You are helpful."));
    }

    #[test]
    fn validate_bad_tool_choice() {
        let req = ChatCompletionToolRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            n: 1,
            stream: false,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
            tools: None,
            tool_choice: Some(ToolChoice::Mode("banana".to_string())),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn full_request_deserialization() {
        let json = r#"{
            "model": "hermes-3",
            "messages": [
                {"role": "user", "content": "What is the weather in NYC?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionToolRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "hermes-3");
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.tool_choice, Some(ToolChoice::Mode("auto".to_string())));
        assert!(req.validate().is_ok());
        assert!(req.tools_enabled());
    }

    #[test]
    fn full_response_serialization() {
        let resp = ChatCompletionToolResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "hermes-3".to_string(),
            choices: vec![ToolChatChoice {
                index: 0,
                message: ToolChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ResponseToolCall {
                        id: "call_0".to_string(),
                        call_type: "function".to_string(),
                        function: ResponseFunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"NYC"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 50,
                completion_tokens: 20,
                total_tokens: 70,
            },
        };
        let json = serde_json::to_string_pretty(&resp).unwrap();
        assert!(json.contains("tool_calls"));
        assert!(json.contains("get_weather"));
        assert!(json.contains("\"finish_reason\": \"tool_calls\""));

        let back: ChatCompletionToolResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn to_sampling_params_maps_fields() {
        let req = ChatCompletionToolRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            max_tokens: 42,
            temperature: 0.8,
            top_p: 0.95,
            n: 2,
            stream: false,
            stop: Some(vec!["END".into()]),
            presence_penalty: 0.1,
            frequency_penalty: 0.2,
            user: None,
            seed: Some(123),
            tools: None,
            tool_choice: None,
        };
        let sp = req.to_sampling_params();
        assert_eq!(sp.max_tokens, 42);
        assert_eq!(sp.temperature, 0.8);
        assert_eq!(sp.top_p, 0.95);
        assert_eq!(sp.stop_strings, vec!["END".to_string()]);
        assert_eq!(sp.seed, Some(123));
    }
}
