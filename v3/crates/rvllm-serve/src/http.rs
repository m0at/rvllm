//! axum HTTP layer.
//!
//! Endpoints:
//! - GET  /health                 -> "ok"
//! - GET  /v1/models              -> {"object":"list","data":[...]}
//! - POST /v1/chat/completions    -> OpenAI-compatible, stream or non-stream
//!
//! Concurrency: `tokio::sync::Semaphore` of size
//! `RVLLM_MAX_NUM_SEQS` caps the number of chat completions in flight
//! against the engine. Excess requests `acquire().await` and queue —
//! no 429.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use futures::stream::Stream;
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

use crate::config::ServerConfig;
use crate::openai::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, ChatMessage, ChatTemplate, ModelEntry, ModelList,
    SharedTokenizer, Usage,
};
use crate::worker::{EngineReq, TokenEvent, WorkerHandle};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub worker: WorkerHandle,
    pub tokenizer: SharedTokenizer,
    pub chat_template: Arc<ChatTemplate>,
    pub gate: Arc<Semaphore>,
}

pub fn router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(cors)
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: state.config.served_model_name.clone(),
            object: "model",
            owned_by: "solidsf",
        }],
    })
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn error_response(status: StatusCode, msg: impl Into<String>) -> Response {
    let body = serde_json::json!({
        "error": {
            "type": "invalid_request_error",
            "message": msg.into(),
        }
    });
    (status, Json(body)).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Permits cap concurrent in-flight chats at MAX_NUM_SEQS. Excess
    // requests block here until a permit is free — no 429.
    let permit = state
        .gate
        .clone()
        .acquire_owned()
        .await
        .expect("semaphore closed");

    let prompt = match state.chat_template.render(&req.messages) {
        Ok(s) => s,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, format!("chat_template: {e}"));
        }
    };

    let encoding = match state.tokenizer.encode(prompt.as_str(), false) {
        Ok(e) => e,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, format!("tokenize: {e}"));
        }
    };
    // Gemma chat template typically already injects BOS via the template;
    // we don't double-add it here.
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

    let max_tokens = req.max_tokens.unwrap_or(512);
    let temperature = req.temperature.unwrap_or(1.0);
    let top_p = req.top_p.unwrap_or(1.0);

    let stop_texts = req.stop.clone().map(|s| s.into_vec()).unwrap_or_default();
    let stop_ids: Vec<Vec<u32>> = stop_texts
        .iter()
        .filter_map(|s| {
            state
                .tokenizer
                .encode(s.as_str(), false)
                .ok()
                .map(|e| e.get_ids().to_vec())
        })
        .filter(|v| !v.is_empty())
        .collect();

    // Gemma 4 EOS: 1 (<eos>) and 107 (<end_of_turn>), with 2 (BOS) as
    // a defensive catch. Matches rvllm-eval defaults.
    let eos = vec![1u32, 2, 107];

    let (tx, rx) = mpsc::unbounded_channel::<TokenEvent>();
    let req_id = uuid::Uuid::new_v4();
    let engine_req = EngineReq {
        req_id,
        prompt_ids,
        max_tokens,
        stop: stop_ids,
        eos,
        temperature,
        top_p,
        tx,
    };
    if state.worker.tx.send(engine_req).is_err() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine worker stopped accepting requests",
        );
    }

    let id = format!("chatcmpl-{}", req_id.simple());
    let created = now_secs();
    let model_name = state.config.served_model_name.clone();
    let tokenizer = state.tokenizer.clone();
    let stop_texts_clone = stop_texts.clone();

    if req.stream {
        let sse_stream = sse_chunks(
            rx,
            id.clone(),
            created,
            model_name.clone(),
            tokenizer,
            stop_texts_clone,
            permit,
        );
        return Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    // Non-streaming: drain all tokens, decode once at the end.
    let (text, finish_reason, usage) = match drain_nonstream(rx, &state.tokenizer, &stop_texts).await
    {
        Ok(v) => v,
        Err(e) => {
            drop(permit);
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };
    drop(permit);

    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: text,
            },
            finish_reason,
        }],
        usage,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

async fn drain_nonstream(
    mut rx: mpsc::UnboundedReceiver<TokenEvent>,
    tokenizer: &tokenizers::Tokenizer,
    stop_texts: &[String],
) -> Result<(String, String, Usage), String> {
    let mut ids: Vec<u32> = Vec::new();
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;
    let mut finish_reason: String = "stop".into();
    let mut err: Option<String> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            TokenEvent::Token { id, .. } => ids.push(id),
            TokenEvent::Finish {
                reason,
                prompt_tokens: pt,
                completion_tokens: ct,
            } => {
                finish_reason = reason.to_string();
                prompt_tokens = pt;
                completion_tokens = ct;
                break;
            }
            TokenEvent::Error(e) => {
                err = Some(e);
                break;
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut text = tokenizer
        .decode(&ids, true)
        .map_err(|e| format!("detokenize: {e}"))?;
    if let Some(idx) = stop_texts
        .iter()
        .filter_map(|s| text.find(s.as_str()))
        .min()
    {
        text.truncate(idx);
        finish_reason = "stop".into();
    }
    let total_tokens = prompt_tokens + completion_tokens;
    Ok((
        text,
        finish_reason,
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        },
    ))
}

fn sse_chunks(
    mut rx: mpsc::UnboundedReceiver<TokenEvent>,
    id: String,
    created: u64,
    model: String,
    tokenizer: SharedTokenizer,
    stop_texts: Vec<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let (out_tx, out_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

    tokio::spawn(async move {
        // Initial role chunk so clients see `delta.role` first.
        let role_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".into()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = out_tx.send(Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap())));

        let mut ids: Vec<u32> = Vec::new();
        let mut emitted_len: usize = 0;
        let mut stop_hit = false;
        let mut final_reason: Option<String> = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                TokenEvent::Token { id: tok_id, .. } => {
                    ids.push(tok_id);
                    // Re-decode the full token stream and emit only
                    // the new tail. This is the simplest correct
                    // approach for byte-level / BPE-merge tokenizers
                    // where naive per-token decoding fragments
                    // multibyte characters.
                    let decoded = match tokenizer.decode(&ids, true) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = out_tx
                                .send(Ok(Event::default().data(format!("{{\"error\":\"detokenize: {e}\"}}"))));
                            break;
                        }
                    };
                    if let Some(hit) = stop_texts
                        .iter()
                        .filter_map(|s| decoded.find(s.as_str()))
                        .min()
                    {
                        let new_text = &decoded[emitted_len..hit];
                        if !new_text.is_empty() {
                            let chunk = ChatCompletionChunk {
                                id: id.clone(),
                                object: "chat.completion.chunk",
                                created,
                                model: model.clone(),
                                choices: vec![ChatChunkChoice {
                                    index: 0,
                                    delta: ChatDelta {
                                        role: None,
                                        content: Some(new_text.into()),
                                    },
                                    finish_reason: None,
                                }],
                            };
                            let _ = out_tx.send(Ok(Event::default()
                                .data(serde_json::to_string(&chunk).unwrap())));
                        }
                        stop_hit = true;
                        break;
                    }
                    if decoded.len() > emitted_len {
                        let new_text = &decoded[emitted_len..];
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: ChatDelta {
                                    role: None,
                                    content: Some(new_text.into()),
                                },
                                finish_reason: None,
                            }],
                        };
                        let _ = out_tx.send(Ok(Event::default()
                            .data(serde_json::to_string(&chunk).unwrap())));
                        emitted_len = decoded.len();
                    }
                }
                TokenEvent::Finish { reason, .. } => {
                    final_reason = Some(reason.into());
                    break;
                }
                TokenEvent::Error(e) => {
                    warn!("engine error: {e}");
                    let _ = out_tx.send(Ok(Event::default()
                        .data(format!("{{\"error\":\"{e}\"}}"))));
                    break;
                }
            }
        }

        let reason = if stop_hit {
            "stop".to_string()
        } else {
            final_reason.unwrap_or_else(|| "stop".to_string())
        };
        let last = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta::default(),
                finish_reason: Some(reason),
            }],
        };
        let _ = out_tx.send(Ok(Event::default().data(serde_json::to_string(&last).unwrap())));
        let _ = out_tx.send(Ok(Event::default().data("[DONE]")));
        drop(permit);
    });

    UnboundedReceiverStream::new(out_rx)
}

pub async fn serve(state: AppState) -> Result<(), String> {
    let addr: std::net::SocketAddr = format!("{}:{}", state.config.host, state.config.port)
        .parse()
        .map_err(|e| format!("bind addr: {e}"))?;
    info!(%addr, "rvllm-serve listening");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;
    axum::serve(listener, router(state).into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("serve: {e}"))?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown signal received");
}

