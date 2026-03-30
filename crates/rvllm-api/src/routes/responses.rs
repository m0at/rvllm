//! Responses API routes: unified text generation with stored follow-up turns.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tracing::info;

use crate::error::ApiError;
use crate::routes::tools::augment_messages_with_tools;
use crate::server::AppState;
use crate::types::request::ChatMessage;
use crate::types::responses::{
    CreateResponseRequest, ResponseFunctionCallItem, ResponseFunctionCallOutputItem,
    ResponseInputItem, ResponseInputItemsList, ResponseObject, ResponseOutputItem,
    ResponseOutputMessage, ResponseToolChoice, ResponseUsage,
};

#[derive(Debug, Clone)]
pub enum StoredConversationItem {
    Input(ResponseInputItem),
    Output(ResponseOutputItem),
}

#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub response: ResponseObject,
    pub input_items: Vec<ResponseInputItem>,
    pub conversation_items: Vec<StoredConversationItem>,
}

pub type SharedResponseStore = Arc<RwLock<HashMap<String, StoredResponse>>>;

/// POST /v1/responses -- create a unified response.
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateResponseRequest>,
) -> Result<Response, ApiError> {
    req.validate()?;

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }

    let input_items = req.normalize_input_items()?;
    let mut conversation_items = if let Some(previous_response_id) = &req.previous_response_id {
        let store = state.response_store.read().await;
        let stored = store.get(previous_response_id).ok_or_else(|| {
            ApiError::InvalidRequest(format!(
                "previous_response_id '{}' was not found or is not stored",
                previous_response_id
            ))
        })?;
        stored.conversation_items.clone()
    } else {
        Vec::new()
    };
    conversation_items.extend(
        input_items
            .iter()
            .cloned()
            .map(StoredConversationItem::Input),
    );

    let prompt_messages = render_conversation_items(&conversation_items);
    if prompt_messages.is_empty() {
        return Err(ApiError::InvalidRequest(
            "input must not be empty for /v1/responses".into(),
        ));
    }

    let mut templated_messages = prompt_messages.clone();
    if let Some(instructions) = req.instructions.clone() {
        if !instructions.is_empty() {
            templated_messages.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: instructions,
                },
            );
        }
    }

    let function_tools = req.normalize_function_tools()?;
    if req.tools_enabled() {
        let tool_defs: Vec<rvllm_tokenizer::ToolDefinition> = function_tools
            .iter()
            .map(|tool| tool.to_tool_definition())
            .collect();
        templated_messages = augment_messages_with_tools(
            &templated_messages,
            &tool_defs,
            rvllm_tokenizer::ToolPromptStyle::Hermes,
        );
    }

    let tokenizer_messages: Vec<rvllm_tokenizer::ChatMessage> = templated_messages
        .iter()
        .map(|message| rvllm_tokenizer::ChatMessage::new(&message.role, &message.content))
        .collect();

    let prompt = state
        .tokenizer
        .read()
        .await
        .apply_chat_template(&tokenizer_messages, true)
        .map_err(|e| ApiError::Internal(format!("chat template error: {}", e)))?;

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let sampling_params = req.to_sampling_params();
    let tool_choice = req.effective_tool_choice();
    let response_tools = req.tools.clone().unwrap_or_default();

    info!(
        model = %req.model,
        stream = req.stream,
        store = req.store,
        tools = req.tools_enabled(),
        previous_response = req.previous_response_id.as_deref().unwrap_or("none"),
        "responses request"
    );

    if req.stream {
        if req.tools_enabled() {
            return stream_tool_response(
                state,
                req,
                prompt,
                sampling_params,
                response_id,
                input_items,
                conversation_items,
            )
            .await;
        }

        let model = state.model_name.clone();
        let input_items_clone = input_items.clone();
        let conversation_items_clone = conversation_items.clone();
        let req_clone = req.clone();
        let response_store = state.response_store.clone();
        let response_id_clone = response_id.clone();
        let tool_choice_clone = tool_choice.clone();
        let response_tools_clone = response_tools.clone();

        let (_request_id, mut output_stream) = state
            .engine
            .generate(prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(16);

        tokio::spawn(async move {
            let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            let mut initial = ResponseObject::in_progress(
                response_id_clone.clone(),
                model.clone(),
                req_clone.instructions.clone(),
                req_clone.max_output_tokens,
                req_clone.previous_response_id.clone(),
                req_clone.store,
                req_clone.temperature,
                req_clone.top_p,
                req_clone.metadata.clone(),
                req_clone.parallel_tool_calls,
                tool_choice_clone.clone(),
                response_tools_clone.clone(),
            );

            if tx
                .send(Ok(format_sse_event(
                    "response.created",
                    &serde_json::json!({
                        "type": "response.created",
                        "response": initial,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            initial.status = "in_progress".to_string();
            if tx
                .send(Ok(format_sse_event(
                    "response.in_progress",
                    &serde_json::json!({
                        "type": "response.in_progress",
                        "response": initial,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            let mut content_open = false;
            let mut full_text = String::new();
            let mut final_output = None;

            while let Some(output) = output_stream.next().await {
                if !content_open {
                    let item = ResponseOutputMessage::in_progress(message_id.clone());
                    if tx
                        .send(Ok(format_sse_event(
                            "response.output_item.added",
                            &serde_json::json!({
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": ResponseOutputItem::Message(item),
                            }),
                        )))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    if tx
                        .send(Ok(format_sse_event(
                            "response.content_part.added",
                            &serde_json::json!({
                                "type": "response.content_part.added",
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                },
                            }),
                        )))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    content_open = true;
                }

                if let Some(choice) = output.outputs.first() {
                    let delta = diff_text(&full_text, &choice.text);
                    full_text = choice.text.clone();
                    if !delta.is_empty()
                        && tx
                            .send(Ok(format_sse_event(
                                "response.output_text.delta",
                                &serde_json::json!({
                                    "type": "response.output_text.delta",
                                    "item_id": message_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": delta,
                                }),
                            )))
                            .await
                            .is_err()
                    {
                        return;
                    }
                }

                final_output = Some(output.clone());
                if output.finished {
                    break;
                }
            }

            let Some(output) = final_output else {
                let _ = tx
                    .send(Ok(format_sse_event(
                        "response.completed",
                        &serde_json::json!({
                            "type": "response.completed",
                            "response": ResponseObject::in_progress(
                                response_id_clone.clone(),
                                model,
                                req_clone.instructions.clone(),
                                req_clone.max_output_tokens,
                                req_clone.previous_response_id.clone(),
                                req_clone.store,
                                req_clone.temperature,
                                req_clone.top_p,
                                req_clone.metadata.clone(),
                                req_clone.parallel_tool_calls,
                                tool_choice_clone,
                                response_tools_clone,
                            ),
                        }),
                    )))
                    .await;
                return;
            };

            let output_items = vec![ResponseOutputItem::Message(
                ResponseOutputMessage::completed(message_id.clone(), full_text.clone()),
            )];
            let response = ResponseObject::completed(
                response_id_clone.clone(),
                model.clone(),
                req_clone.instructions.clone(),
                req_clone.max_output_tokens,
                req_clone.previous_response_id.clone(),
                req_clone.store,
                req_clone.temperature,
                req_clone.top_p,
                req_clone.metadata.clone(),
                output_items.clone(),
                ResponseUsage::from_request_output(&output),
                req_clone.parallel_tool_calls,
                req_clone.effective_tool_choice(),
                req_clone.tools.clone().unwrap_or_default(),
            );

            if tx
                .send(Ok(format_sse_event(
                    "response.output_text.done",
                    &serde_json::json!({
                        "type": "response.output_text.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": full_text,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.content_part.done",
                    &serde_json::json!({
                        "type": "response.content_part.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": [],
                        },
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.output_item.done",
                    &serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": response.output[0],
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.completed",
                    &serde_json::json!({
                        "type": "response.completed",
                        "response": response,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            if req_clone.store {
                let mut stored_items = conversation_items_clone;
                stored_items.extend(output_items.into_iter().map(StoredConversationItem::Output));

                let mut store = response_store.write().await;
                store.insert(
                    response_id_clone,
                    StoredResponse {
                        response,
                        input_items: input_items_clone,
                        conversation_items: stored_items,
                    },
                );
            }
        });

        let body = axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
        Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(body)
            .unwrap()
            .into_response())
    } else {
        let (_request_id, mut output_stream) = state
            .engine
            .generate(prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let mut last_output = None;
        while let Some(output) = output_stream.next().await {
            last_output = Some(output.clone());
            if output.finished {
                break;
            }
        }

        let output =
            last_output.ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

        let response = response_from_output(
            &response_id,
            &state.model_name,
            &req,
            &output,
            &function_tools,
            tool_choice,
            response_tools,
        )?;

        if req.store {
            let mut stored_items = conversation_items;
            stored_items.extend(
                response
                    .output
                    .iter()
                    .cloned()
                    .map(StoredConversationItem::Output),
            );

            let mut store = state.response_store.write().await;
            store.insert(
                response_id,
                StoredResponse {
                    response: response.clone(),
                    input_items,
                    conversation_items: stored_items,
                },
            );
        }

        Ok(Json(response).into_response())
    }
}

#[derive(Debug, Clone)]
struct StreamedMessageState {
    id: String,
    text: String,
}

#[derive(Debug, Clone)]
struct StreamedFunctionCallState {
    id: String,
    call_id: String,
    name: String,
    arguments: String,
}

async fn stream_tool_response(
    state: Arc<AppState>,
    req: CreateResponseRequest,
    prompt: String,
    sampling_params: rvllm_core::prelude::SamplingParams,
    response_id: String,
    input_items: Vec<ResponseInputItem>,
    conversation_items: Vec<StoredConversationItem>,
) -> Result<Response, ApiError> {
    let model = state.model_name.clone();
    let response_store = state.response_store.clone();
    let tool_choice = req.effective_tool_choice();
    let response_tools = req.tools.clone().unwrap_or_default();

    let (_request_id, mut output_stream) = state
        .engine
        .generate(prompt, sampling_params)
        .await
        .map_err(ApiError::from)?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        let initial = ResponseObject::in_progress(
            response_id.clone(),
            model.clone(),
            req.instructions.clone(),
            req.max_output_tokens,
            req.previous_response_id.clone(),
            req.store,
            req.temperature,
            req.top_p,
            req.metadata.clone(),
            req.parallel_tool_calls,
            tool_choice.clone(),
            response_tools.clone(),
        );

        if tx
            .send(Ok(format_sse_event(
                "response.created",
                &serde_json::json!({
                    "type": "response.created",
                    "response": initial,
                }),
            )))
            .await
            .is_err()
        {
            return;
        }

        if tx
            .send(Ok(format_sse_event(
                "response.in_progress",
                &serde_json::json!({
                    "type": "response.in_progress",
                    "response": ResponseObject::in_progress(
                        response_id.clone(),
                        model.clone(),
                        req.instructions.clone(),
                        req.max_output_tokens,
                        req.previous_response_id.clone(),
                        req.store,
                        req.temperature,
                        req.top_p,
                        req.metadata.clone(),
                        req.parallel_tool_calls,
                        tool_choice.clone(),
                        response_tools.clone(),
                    ),
                }),
            )))
            .await
            .is_err()
        {
            return;
        }

        let mut full_text = String::new();
        let mut final_output = None;
        let mut prefix_state: Option<StreamedMessageState> = None;
        let mut tool_states: Vec<StreamedFunctionCallState> = Vec::new();
        let mut saw_tool_calls = false;

        while let Some(output) = output_stream.next().await {
            if let Some(choice) = output.outputs.first() {
                full_text = choice.text.clone();
                let parse_result =
                    rvllm_tokenizer::parse_tool_calls(&full_text, &format!("{response_id}_0_"));

                if let rvllm_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls } =
                    parse_result
                {
                    saw_tool_calls = true;

                    if !prefix_text.is_empty() {
                        if prefix_state.is_none() {
                            let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
                            if tx
                                .send(Ok(format_sse_event(
                                    "response.output_item.added",
                                    &serde_json::json!({
                                        "type": "response.output_item.added",
                                        "response_id": response_id,
                                        "output_index": 0,
                                        "item": ResponseOutputItem::Message(
                                            ResponseOutputMessage::in_progress(item_id.clone())
                                        ),
                                    }),
                                )))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            if tx
                                .send(Ok(format_sse_event(
                                    "response.content_part.added",
                                    &serde_json::json!({
                                        "type": "response.content_part.added",
                                        "item_id": item_id,
                                        "output_index": 0,
                                        "content_index": 0,
                                        "part": {
                                            "type": "output_text",
                                            "text": "",
                                            "annotations": [],
                                        },
                                    }),
                                )))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            prefix_state = Some(StreamedMessageState {
                                id: item_id,
                                text: String::new(),
                            });
                        }

                        if let Some(state) = prefix_state.as_mut() {
                            let delta = diff_text(&state.text, &prefix_text);
                            if !delta.is_empty()
                                && tx
                                    .send(Ok(format_sse_event(
                                        "response.output_text.delta",
                                        &serde_json::json!({
                                            "type": "response.output_text.delta",
                                            "item_id": state.id,
                                            "output_index": 0,
                                            "content_index": 0,
                                            "delta": delta,
                                        }),
                                    )))
                                    .await
                                    .is_err()
                            {
                                return;
                            }
                            state.text = prefix_text;
                        }
                    }

                    let tool_offset = usize::from(prefix_state.is_some());
                    for (index, call) in calls.into_iter().enumerate() {
                        if tool_states.len() <= index {
                            let item_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
                            let output_index = tool_offset + index;
                            if tx
                                .send(Ok(format_sse_event(
                                    "response.output_item.added",
                                    &serde_json::json!({
                                        "type": "response.output_item.added",
                                        "response_id": response_id,
                                        "output_index": output_index,
                                        "item": ResponseOutputItem::FunctionCall(
                                            ResponseFunctionCallItem::in_progress(
                                                item_id.clone(),
                                                call.id.clone(),
                                                call.name.clone(),
                                            )
                                        ),
                                    }),
                                )))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            tool_states.push(StreamedFunctionCallState {
                                id: item_id,
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: String::new(),
                            });
                        }

                        let state = &mut tool_states[index];
                        let delta = diff_text(&state.arguments, &call.arguments);
                        if !delta.is_empty()
                            && tx
                                .send(Ok(format_sse_event(
                                    "response.function_call_arguments.delta",
                                    &serde_json::json!({
                                        "type": "response.function_call_arguments.delta",
                                        "response_id": response_id,
                                        "item_id": state.id,
                                        "output_index": tool_offset + index,
                                        "delta": delta,
                                    }),
                                )))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        state.arguments = call.arguments;
                    }
                }
            }

            final_output = Some(output.clone());
            if output.finished {
                break;
            }
        }

        let Some(output) = final_output else {
            return;
        };

        let output_items = if saw_tool_calls {
            let mut items = Vec::new();

            if let Some(state) = prefix_state.as_ref() {
                if tx
                    .send(Ok(format_sse_event(
                        "response.output_text.done",
                        &serde_json::json!({
                            "type": "response.output_text.done",
                            "item_id": state.id,
                            "output_index": 0,
                            "content_index": 0,
                            "text": state.text,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                if tx
                    .send(Ok(format_sse_event(
                        "response.content_part.done",
                        &serde_json::json!({
                            "type": "response.content_part.done",
                            "item_id": state.id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": state.text,
                                "annotations": [],
                            },
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }

                let item = ResponseOutputItem::Message(ResponseOutputMessage::completed(
                    state.id.clone(),
                    state.text.clone(),
                ));
                if tx
                    .send(Ok(format_sse_event(
                        "response.output_item.done",
                        &serde_json::json!({
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item": item,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                items.push(item);
            }

            let tool_offset = usize::from(prefix_state.is_some());
            for (index, state) in tool_states.iter().enumerate() {
                let item = ResponseOutputItem::FunctionCall(ResponseFunctionCallItem::completed(
                    state.id.clone(),
                    state.call_id.clone(),
                    state.name.clone(),
                    state.arguments.clone(),
                ));
                if tx
                    .send(Ok(format_sse_event(
                        "response.function_call_arguments.done",
                        &serde_json::json!({
                            "type": "response.function_call_arguments.done",
                            "response_id": response_id,
                            "output_index": tool_offset + index,
                            "item": item,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                if tx
                    .send(Ok(format_sse_event(
                        "response.output_item.done",
                        &serde_json::json!({
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": tool_offset + index,
                            "item": item,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                items.push(item);
            }

            items
        } else {
            let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            let item = ResponseOutputItem::Message(ResponseOutputMessage::completed(
                message_id.clone(),
                full_text.clone(),
            ));

            if tx
                .send(Ok(format_sse_event(
                    "response.output_item.added",
                    &serde_json::json!({
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": 0,
                        "item": ResponseOutputItem::Message(ResponseOutputMessage::in_progress(message_id.clone())),
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.content_part.added",
                    &serde_json::json!({
                        "type": "response.content_part.added",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if !full_text.is_empty()
                && tx
                    .send(Ok(format_sse_event(
                        "response.output_text.delta",
                        &serde_json::json!({
                            "type": "response.output_text.delta",
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": full_text,
                        }),
                    )))
                    .await
                    .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.output_text.done",
                    &serde_json::json!({
                        "type": "response.output_text.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": full_text,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.content_part.done",
                    &serde_json::json!({
                        "type": "response.content_part.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": [],
                        },
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.output_item.done",
                    &serde_json::json!({
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": 0,
                        "item": item,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            vec![item]
        };

        let response = ResponseObject::completed(
            response_id.clone(),
            model.clone(),
            req.instructions.clone(),
            req.max_output_tokens,
            req.previous_response_id.clone(),
            req.store,
            req.temperature,
            req.top_p,
            req.metadata.clone(),
            output_items.clone(),
            ResponseUsage::from_request_output(&output),
            req.parallel_tool_calls,
            tool_choice,
            response_tools,
        );

        if tx
            .send(Ok(format_sse_event(
                "response.completed",
                &serde_json::json!({
                    "type": "response.completed",
                    "response": response,
                }),
            )))
            .await
            .is_err()
        {
            return;
        }

        if req.store {
            let mut stored_items = conversation_items;
            stored_items.extend(output_items.into_iter().map(StoredConversationItem::Output));

            let mut store = response_store.write().await;
            store.insert(
                response_id,
                StoredResponse {
                    response,
                    input_items,
                    conversation_items: stored_items,
                },
            );
        }
    });

    let body = axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
        .into_response())
}

/// GET /v1/responses/{response_id} -- retrieve a stored response object.
pub async fn get_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let store = state.response_store.read().await;
    let stored = store
        .get(&response_id)
        .ok_or_else(|| ApiError::NotFound(format!("response '{}' not found", response_id)))?;
    Ok(Json(stored.response.clone()))
}

/// GET /v1/responses/{response_id}/input_items -- list normalized input items.
pub async fn list_response_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let store = state.response_store.read().await;
    let stored = store
        .get(&response_id)
        .ok_or_else(|| ApiError::NotFound(format!("response '{}' not found", response_id)))?;
    Ok(Json(ResponseInputItemsList {
        object: "list".to_string(),
        data: stored.input_items.clone(),
        first_id: None,
        last_id: None,
        has_more: false,
    }))
}

fn response_from_output(
    response_id: &str,
    model: &str,
    req: &CreateResponseRequest,
    output: &rvllm_core::prelude::RequestOutput,
    function_tools: &[crate::types::responses::ResponseFunctionTool],
    tool_choice: ResponseToolChoice,
    response_tools: Vec<serde_json::Value>,
) -> Result<ResponseObject, ApiError> {
    let completion = output
        .outputs
        .first()
        .ok_or_else(|| ApiError::Internal("engine produced no completion output".into()))?;
    let output_items = response_output_items_from_text(
        response_id,
        &completion.text,
        req,
        function_tools,
        &tool_choice,
    )?;
    Ok(ResponseObject::completed(
        response_id.to_string(),
        model.to_string(),
        req.instructions.clone(),
        req.max_output_tokens,
        req.previous_response_id.clone(),
        req.store,
        req.temperature,
        req.top_p,
        req.metadata.clone(),
        output_items,
        ResponseUsage::from_request_output(output),
        req.parallel_tool_calls,
        tool_choice,
        response_tools,
    ))
}

fn response_output_items_from_text(
    response_id: &str,
    text: &str,
    req: &CreateResponseRequest,
    function_tools: &[crate::types::responses::ResponseFunctionTool],
    tool_choice: &ResponseToolChoice,
) -> Result<Vec<ResponseOutputItem>, ApiError> {
    if !req.tools_enabled() {
        return Ok(vec![ResponseOutputItem::Message(
            ResponseOutputMessage::completed(
                format!("msg_{}", uuid::Uuid::new_v4().simple()),
                text.to_string(),
            ),
        )]);
    }

    let allowed_tools: HashSet<&str> = function_tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    let call_prefix = format!("{response_id}_0_");
    match rvllm_tokenizer::parse_tool_calls(text, &call_prefix) {
        rvllm_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls } => {
            let mut output_items = Vec::new();

            if !prefix_text.is_empty() {
                output_items.push(ResponseOutputItem::Message(
                    ResponseOutputMessage::completed(
                        format!("msg_{}", uuid::Uuid::new_v4().simple()),
                        prefix_text,
                    ),
                ));
            }

            let forced_tool_name = tool_choice.forced_tool_name();
            for call in calls {
                if !allowed_tools.contains(call.name.as_str()) {
                    return Err(ApiError::Internal(format!(
                        "model emitted undeclared function '{}'",
                        call.name
                    )));
                }
                if let Some(forced_name) = forced_tool_name {
                    if call.name != forced_name {
                        return Err(ApiError::Internal(format!(
                            "model emitted function '{}' but tool_choice requires '{}'",
                            call.name, forced_name
                        )));
                    }
                }
                output_items.push(ResponseOutputItem::FunctionCall(
                    ResponseFunctionCallItem::completed(
                        format!("fc_{}", uuid::Uuid::new_v4().simple()),
                        call.id,
                        call.name,
                        call.arguments,
                    ),
                ));
            }

            Ok(output_items)
        }
        rvllm_tokenizer::ToolParseResult::PlainText(text) => {
            if matches!(tool_choice, ResponseToolChoice::Mode(mode) if mode == "required")
                || tool_choice.forced_tool_name().is_some()
            {
                return Err(ApiError::Internal(
                    "model did not emit a required function call".into(),
                ));
            }
            Ok(vec![ResponseOutputItem::Message(
                ResponseOutputMessage::completed(
                    format!("msg_{}", uuid::Uuid::new_v4().simple()),
                    text,
                ),
            )])
        }
    }
}

fn render_conversation_items(items: &[StoredConversationItem]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    let mut function_names = HashMap::new();

    for item in items {
        match item {
            StoredConversationItem::Input(ResponseInputItem::Message(message)) => {
                messages.push(message.to_chat_message());
            }
            StoredConversationItem::Input(ResponseInputItem::FunctionCallOutput(output)) => {
                let function_name = function_names.get(&output.call_id).map(String::as_str);
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: render_function_call_output(output, function_name),
                });
            }
            StoredConversationItem::Output(ResponseOutputItem::Message(message)) => {
                messages.push(message.to_chat_message());
            }
            StoredConversationItem::Output(ResponseOutputItem::FunctionCall(call)) => {
                function_names.insert(call.call_id.clone(), call.name.clone());
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: render_function_call(call),
                });
            }
        }
    }

    messages
}

fn render_function_call(call: &ResponseFunctionCallItem) -> String {
    let arguments = serde_json::from_str::<serde_json::Value>(&call.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone()));
    let body = serde_json::json!({
        "name": call.name,
        "arguments": arguments,
    });
    format!("<tool_call>{}</tool_call>", body)
}

fn render_function_call_output(
    output: &ResponseFunctionCallOutputItem,
    function_name: Option<&str>,
) -> String {
    let rendered_output = output.output_text();
    match function_name {
        Some(name) => format!(
            "Tool output for function {} (call_id {}):\n{}",
            name, output.call_id, rendered_output
        ),
        None => format!(
            "Tool output for call_id {}:\n{}",
            output.call_id, rendered_output
        ),
    }
}

fn diff_text(previous: &str, current: &str) -> String {
    if let Some(suffix) = current.strip_prefix(previous) {
        suffix.to_string()
    } else {
        current.to_string()
    }
}

fn format_sse_event<T: Serialize>(event: &str, payload: &T) -> String {
    let json = serde_json::to_string(payload).unwrap_or_default();
    format!("event: {event}\ndata: {json}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use crate::types::responses::{
        ResponseInputMessage, ResponseInputTextPart, ResponseSpecificToolChoice,
    };
    use crate::{build_router, AppState};
    use axum_test::TestServer;
    use rvllm_core::prelude::{
        CompletionOutput, FinishReason, RequestId, RequestOutput, SamplingParams,
    };
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer as HfTokenizer;
    use tokio::sync::Mutex as AsyncMutex;
    use tokio_stream::wrappers::ReceiverStream;

    struct FakeEngine {
        outputs: AsyncMutex<VecDeque<Vec<RequestOutput>>>,
        prompts: Mutex<Vec<String>>,
    }

    impl FakeEngine {
        fn new(outputs: Vec<Vec<RequestOutput>>) -> Self {
            Self {
                outputs: AsyncMutex::new(outputs.into()),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl crate::server::InferenceEngine for FakeEngine {
        async fn generate(
            &self,
            prompt: String,
            _params: SamplingParams,
        ) -> rvllm_core::prelude::Result<(RequestId, ReceiverStream<RequestOutput>)> {
            self.prompts.lock().unwrap().push(prompt);
            let maybe_outputs = self.outputs.lock().await.pop_front();
            let outputs = maybe_outputs.expect("fake engine ran out of queued outputs");
            let (tx, rx) = tokio::sync::mpsc::channel(outputs.len().max(1));
            for output in outputs {
                tx.send(output).await.unwrap();
            }
            drop(tx);
            Ok((RequestId(1), ReceiverStream::new(rx)))
        }
    }

    fn make_test_tokenizer() -> rvllm_tokenizer::Tokenizer {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("tool".to_string(), 2);
        vocab.insert("call".to_string(), 3);
        vocab.insert("weather".to_string(), 4);
        vocab.insert("time".to_string(), 5);
        vocab.insert("[UNK]".to_string(), 6);

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut hf = HfTokenizer::new(bpe);
        hf.with_pre_tokenizer(Some(Whitespace {}));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        hf.save(&path, false).unwrap();
        rvllm_tokenizer::Tokenizer::from_file(&path).unwrap()
    }

    fn make_server(outputs: Vec<Vec<RequestOutput>>) -> (TestServer, Arc<FakeEngine>) {
        let engine = Arc::new(FakeEngine::new(outputs));
        let state = Arc::new(AppState::new(
            engine.clone(),
            "test".to_string(),
            make_test_tokenizer(),
        ));
        let server = TestServer::new(build_router(state)).unwrap();
        (server, engine)
    }

    fn request_output(text: &str, finished: bool) -> RequestOutput {
        RequestOutput {
            request_id: RequestId(1),
            prompt: "prompt".into(),
            prompt_token_ids: vec![1, 2, 3],
            prompt_logprobs: None,
            outputs: vec![CompletionOutput {
                index: 0,
                text: text.to_string(),
                token_ids: vec![10, 11],
                cumulative_logprob: -0.1,
                logprobs: None,
                finish_reason: finished.then_some(FinishReason::Stop),
            }],
            finished,
        }
    }

    fn parse_sse_events(body: &str) -> Vec<(String, serde_json::Value)> {
        body.split("\n\n")
            .filter_map(|chunk| {
                let mut event = None;
                let mut data = None;
                for line in chunk.lines() {
                    if let Some(value) = line.strip_prefix("event: ") {
                        event = Some(value.to_string());
                    } else if let Some(value) = line.strip_prefix("data: ") {
                        data = Some(serde_json::from_str(value).unwrap());
                    }
                }
                event.zip(data)
            })
            .collect()
    }

    fn make_tool_request(tool_choice: ResponseToolChoice) -> CreateResponseRequest {
        CreateResponseRequest {
            model: "test".into(),
            input: Some(crate::types::responses::ResponseInput::Text("hi".into())),
            instructions: None,
            max_output_tokens: Some(32),
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: Default::default(),
            background: None,
            tools: Some(vec![serde_json::json!({
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            })]),
            tool_choice: Some(tool_choice),
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        }
    }

    #[test]
    fn diff_text_returns_suffix_when_possible() {
        assert_eq!(diff_text("Hello", "Hello there"), " there");
    }

    #[test]
    fn diff_text_returns_current_when_prefix_does_not_match() {
        assert_eq!(diff_text("Hi", "Hello"), "Hello");
    }

    #[test]
    fn format_sse_event_includes_event_name() {
        let rendered = format_sse_event("response.created", &serde_json::json!({"ok": true}));
        assert!(rendered.starts_with("event: response.created\n"));
        assert!(rendered.contains("data: {\"ok\":true}\n\n"));
    }

    #[test]
    fn response_output_items_parse_function_calls() {
        let req = make_tool_request(ResponseToolChoice::Mode("auto".into()));
        let tools = req.normalize_function_tools().unwrap();
        let items = response_output_items_from_text(
            "resp_test",
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call>",
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap();

        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseOutputItem::FunctionCall(call) => {
                assert_eq!(call.name, "get_weather");
                assert!(call.arguments.contains("Boston"));
            }
            _ => panic!("expected function_call item"),
        }
    }

    #[test]
    fn response_output_items_reject_plain_text_when_required() {
        let req = make_tool_request(ResponseToolChoice::Mode("required".into()));
        let tools = req.normalize_function_tools().unwrap();
        let err = response_output_items_from_text(
            "resp_test",
            "plain text",
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("required function call"));
    }

    #[test]
    fn response_output_items_reject_wrong_forced_tool() {
        let mut req = make_tool_request(ResponseToolChoice::Specific(ResponseSpecificToolChoice {
            choice_type: "function".into(),
            name: "get_weather".into(),
        }));
        req.tools
            .as_mut()
            .unwrap()
            .push(serde_json::json!({"type": "function", "name": "get_time"}));
        let tools = req.normalize_function_tools().unwrap();
        let err = response_output_items_from_text(
            "resp_test",
            "<tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>",
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("tool_choice requires"));
    }

    #[test]
    fn render_conversation_items_preserves_tool_history() {
        let items = vec![
            StoredConversationItem::Input(ResponseInputItem::Message(ResponseInputMessage::new(
                "user",
                vec![ResponseInputTextPart::new("Weather?")],
            ))),
            StoredConversationItem::Output(ResponseOutputItem::FunctionCall(
                ResponseFunctionCallItem::completed(
                    "fc_1",
                    "call_1",
                    "get_weather",
                    "{\"location\":\"Boston\"}",
                ),
            )),
            StoredConversationItem::Input(ResponseInputItem::FunctionCallOutput(
                ResponseFunctionCallOutputItem {
                    call_id: "call_1".into(),
                    output: serde_json::json!({"temp_c": 18}),
                },
            )),
        ];

        let messages = render_conversation_items(&items);
        assert_eq!(messages.len(), 3);
        assert!(messages[1].content.contains("<tool_call>"));
        assert!(messages[2].content.contains("get_weather"));
        assert!(messages[2].content.contains("temp_c"));
    }

    #[test]
    fn render_function_call_serializes_json_arguments() {
        let call = ResponseFunctionCallItem::completed(
            "fc_1",
            "call_1",
            "get_weather",
            "{\"location\":\"Boston\"}",
        );
        let rendered = render_function_call(&call);
        assert!(rendered.contains("<tool_call>"));
        assert!(rendered.contains("\"name\":\"get_weather\""));
        assert!(rendered.contains("\"location\":\"Boston\""));
    }

    #[test]
    fn render_function_call_output_includes_call_id() {
        let rendered = render_function_call_output(
            &ResponseFunctionCallOutputItem {
                call_id: "call_123".into(),
                output: serde_json::json!({"ok": true}),
            },
            None,
        );
        assert!(rendered.contains("call_123"));
        assert!(rendered.contains("\"ok\":true"));
    }

    #[tokio::test]
    async fn create_response_route_returns_function_call_items() {
        let (server, _) = make_server(vec![vec![request_output(
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call>",
            true,
        )]]);

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "input": "What's the weather?",
                "store": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            }))
            .await;

        response.assert_status_ok();
        let body = response.json::<serde_json::Value>();
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["output"][0]["type"], "function_call");
        assert_eq!(body["output"][0]["name"], "get_weather");
        assert_eq!(body["output"][0]["status"], "completed");
    }

    #[tokio::test]
    async fn previous_response_id_replays_function_calls_and_outputs() {
        let (server, engine) = make_server(vec![
            vec![request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call>",
                true,
            )],
            vec![request_output("It is 18C and sunny.", true)],
        ]);

        let first = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "input": "What's the weather?",
                "store": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            }))
            .await;
        first.assert_status_ok();
        let first_body = first.json::<serde_json::Value>();
        let first_id = first_body["id"].as_str().unwrap().to_string();
        let call_id = first_body["output"][0]["call_id"]
            .as_str()
            .unwrap()
            .to_string();

        let second = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "previous_response_id": first_id,
                "store": true,
                "input": [{
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": {"temp_c": 18, "conditions": "sunny"},
                }],
            }))
            .await;
        second.assert_status_ok();
        let second_body = second.json::<serde_json::Value>();
        assert_eq!(second_body["output"][0]["type"], "message");
        assert_eq!(second_body["previous_response_id"], first_id);

        let prompts = engine.prompts();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[1].contains("<tool_call>"));
        assert!(prompts[1].contains("Tool output for function get_weather"));
        assert!(prompts[1].contains("\"temp_c\":18"));

        let retrieved = server.get(&format!("/v1/responses/{}", first_id)).await;
        retrieved.assert_status_ok();
        let retrieved = retrieved.json::<serde_json::Value>();
        assert_eq!(retrieved["output"][0]["type"], "function_call");

        let input_items = server
            .get(&format!(
                "/v1/responses/{}/input_items",
                second_body["id"].as_str().unwrap()
            ))
            .await;
        input_items.assert_status_ok();
        let input_items = input_items.json::<serde_json::Value>();
        assert_eq!(input_items["data"][0]["type"], "function_call_output");
        assert_eq!(input_items["data"][0]["output"]["temp_c"], 18);
    }

    #[tokio::test]
    async fn streaming_tool_responses_emit_function_call_events_and_store_output() {
        let (server, _) = make_server(vec![vec![
            request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Bos",
                false,
            ),
            request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call><tool_call>{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UT",
                false,
            ),
            request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call><tool_call>{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}</tool_call>",
                true,
            ),
        ]]);

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "input": "Call the tools you need.",
                "stream": true,
                "store": true,
                "parallel_tool_calls": true,
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": {"type": "object"},
                    }
                ],
            }))
            .await;

        response.assert_status_ok();
        let events = parse_sse_events(&response.text());
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert!(names.contains(&"response.created"));
        assert!(names.contains(&"response.in_progress"));
        assert!(names.contains(&"response.output_item.added"));
        assert!(names.contains(&"response.function_call_arguments.delta"));
        assert!(names.contains(&"response.function_call_arguments.done"));
        assert!(names.contains(&"response.output_item.done"));
        assert!(names.contains(&"response.completed"));

        let delta_count = names
            .iter()
            .filter(|name| **name == "response.function_call_arguments.delta")
            .count();
        assert!(delta_count >= 2);

        let completed = events
            .iter()
            .find(|(name, _)| name == "response.completed")
            .unwrap();
        let response_id = completed.1["response"]["id"].as_str().unwrap();
        assert_eq!(
            completed.1["response"]["output"].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            completed.1["response"]["output"][0]["type"],
            "function_call"
        );
        assert_eq!(completed.1["response"]["output"][1]["name"], "get_time");

        let stored = server.get(&format!("/v1/responses/{response_id}")).await;
        stored.assert_status_ok();
        let stored = stored.json::<serde_json::Value>();
        assert_eq!(stored["output"].as_array().unwrap().len(), 2);
        assert_eq!(stored["output"][0]["name"], "get_weather");
        assert_eq!(stored["output"][1]["name"], "get_time");
    }
}
