//! Chat completion endpoint: POST /v1/chat/completions

use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::StreamExt;
use tracing::info;

use crate::error::ApiError;
use crate::routes::tools::{augment_messages_with_tools, ToolChoice};
use crate::server::AppState;
use crate::types::request::ChatCompletionRequest;
use crate::types::response::ChatCompletionResponse;
use crate::types::streaming::{format_sse_data, ChatCompletionStreamChunk, SSE_DONE};

/// POST /v1/chat/completions -- chat completion (streaming or non-streaming).
pub async fn create_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    req.validate()?;

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }

    let sampling_params = req.to_sampling_params();

    // Check if tools are active
    let tools_active = req.tools.as_ref().map_or(false, |t| !t.is_empty())
        && !matches!(req.tool_choice.as_ref(), Some(ToolChoice::Mode(m)) if m == "none");

    // Build messages, optionally augmented with tool definitions
    let messages = if tools_active {
        let tool_defs: Vec<rvllm_tokenizer::ToolDefinition> = req
            .tools
            .as_ref()
            .unwrap()
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
            .collect();
        augment_messages_with_tools(
            &req.messages,
            &tool_defs,
            rvllm_tokenizer::ToolPromptStyle::Hermes,
        )
    } else {
        req.messages.clone()
    };

    // Convert chat messages to a prompt via the tokenizer's chat template.
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
        "chat completion request"
    );

    if req.stream {
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

        // First event: role chunk
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

        // Prepend the initial role chunk
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
        // Non-streaming: collect all outputs until finished.
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

        if tools_active {
            // Parse output for tool calls
            let resp_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let mut total_completion = 0usize;
            let choices: Vec<crate::routes::tools::ToolChatChoice> = output
                .outputs
                .iter()
                .map(|co| {
                    total_completion += co.token_ids.len();
                    let finish_reason_val = co.finish_reason.map(|r| match r {
                        rvllm_core::prelude::FinishReason::Stop => "stop",
                        rvllm_core::prelude::FinishReason::Length => "length",
                        rvllm_core::prelude::FinishReason::Abort => "stop",
                    });
                    let call_prefix = format!("{}_{}_", resp_id, co.index);
                    let parse_result = rvllm_tokenizer::parse_tool_calls(&co.text, &call_prefix);
                    match parse_result {
                        rvllm_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls } => {
                            let tool_calls: Vec<crate::routes::tools::ResponseToolCall> = calls
                                .into_iter()
                                .map(|tc| crate::routes::tools::ResponseToolCall {
                                    id: tc.id,
                                    call_type: "function".to_string(),
                                    function: crate::routes::tools::ResponseFunctionCall {
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
                            crate::routes::tools::ToolChatChoice {
                                index: co.index,
                                message: crate::routes::tools::ToolChatMessage {
                                    role: "assistant".to_string(),
                                    content,
                                    tool_calls: Some(tool_calls),
                                },
                                finish_reason: Some("tool_calls".to_string()),
                            }
                        }
                        rvllm_tokenizer::ToolParseResult::PlainText(text) => {
                            crate::routes::tools::ToolChatChoice {
                                index: co.index,
                                message: crate::routes::tools::ToolChatMessage {
                                    role: "assistant".to_string(),
                                    content: Some(text),
                                    tool_calls: None,
                                },
                                finish_reason: finish_reason_val.map(|s| s.to_string()),
                            }
                        }
                    }
                })
                .collect();

            let resp = crate::routes::tools::ChatCompletionToolResponse {
                id: resp_id,
                object: "chat.completion".to_string(),
                created: now,
                model: state.model_name.clone(),
                choices,
                usage: crate::types::response::Usage {
                    prompt_tokens: output.prompt_token_ids.len(),
                    completion_tokens: total_completion,
                    total_tokens: output.prompt_token_ids.len() + total_completion,
                },
            };
            Ok(Json(resp).into_response())
        } else {
            let resp = ChatCompletionResponse::from_request_output(&output, &state.model_name);
            Ok(Json(resp).into_response())
        }
    }
}
