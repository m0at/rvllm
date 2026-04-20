//! Request handlers for `/v1/chat/completions` and `/v1/completions`.
//!
//! Shared flow:
//!   1. validate (model, sampling, max_tokens)
//!   2. tokenize prompt (chat template or raw string)
//!   3. submit GenerateRequest to worker
//!   4. drain worker events — aggregate (non-stream) or stream (SSE)
//!
//! SSE frames follow the OpenAI convention exactly:
//!   `data: {chunk_json}\n\n`      per token
//!   `data: [DONE]\n\n`             terminator
//!
//! The SSE path is driven by [`futures::stream::unfold`] so axum can
//! drop the response mid-stream cleanly; when that happens the
//! `cancelled` flag is flipped and the worker stops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Sse},
    Json,
};
use futures::stream::{self, Stream};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Erased SSE event stream used by the streaming handlers' return
/// types. Writing out `Pin<Box<dyn Stream ...>>` at every call site
/// is repetitive; this alias keeps the signatures readable.
type SseStream = std::pin::Pin<
    Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static>,
>;

/// Flips the `cancelled` flag on drop. Covers two paths:
///   * non-streaming: axum cancels the handler future when the client
///     disconnects mid-request → our local is dropped → flag set →
///     worker breaks out at the next boundary.
///   * normal completion: flag set after Done is already sent → the
///     worker has nothing left to notice, the set is a no-op.
///
/// Borrows the `Arc<AtomicBool>` shared with the worker; no
/// allocation on the drop path.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.0.swap(true, Ordering::Relaxed) {
            tracing::debug!("request dropped, cancellation requested");
        }
    }
}

use crate::error::{ApiError, ApiResult};
use crate::openai::chat::{
    ChatAssistantMessage, ChatChoice, ChatChunkChoice, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, ChatDelta, StopField,
};
use crate::openai::completions::{
    CompletionChoice, CompletionChunk, CompletionChunkChoice, CompletionRequest,
    CompletionResponse, PromptField,
};
use crate::openai::models::ensure_model_matches;
use crate::openai::types::{
    new_chat_completion_id, new_completion_id, unix_now_secs, FinishReason, Role, Usage,
};
use crate::router::AppState;
use crate::worker::{GenerateEvent, GenerateRequest};

// ═════════════════════════════════════════════════════════════════════
// /v1/chat/completions
// ═════════════════════════════════════════════════════════════════════

/// Axum handler. Dispatches to streaming vs non-streaming paths based
/// on `req.stream`.
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> ApiResult<ChatCompletionsResponse> {
    let request_id = Uuid::new_v4();
    let span = tracing::info_span!(
        "chat_completions",
        request_id = %request_id,
        stream = req.stream,
    );
    let _enter = span.enter();

    ensure_model_matches(&state, &req.model)?;
    reject_v1_unsupported_chat(&req)?;

    if req.messages.is_empty() {
        return Err(ApiError::invalid_param(
            "messages must be non-empty",
            "messages",
            "empty_messages",
        ));
    }

    let sampling = req.sampling_params().ensure_supported()?;
    let max_new = resolve_max_new(req.max_tokens, state.config.max_new_tokens_cap)?;
    let stop_text = req.stop.as_ref().map(|s| extract_stop(s)).unwrap_or_default();
    validate_stops(&stop_text)?;

    let prompt_ids = state.tokenizer.render_chat(&req.messages)?;
    reject_oversized_prompt(prompt_ids.len(), max_new, state.config.max_new_tokens_cap)?;

    // Channel + cancellation. Buffer 64 tokens before worker blocks —
    // enough to absorb a slow client without starving the GPU.
    let (events_tx, events_rx) = mpsc::channel::<GenerateEvent>(64);
    let cancelled = Arc::new(AtomicBool::new(false));
    // Drop guard fires on any handler exit (normal, error, client
    // disconnect mid-non-stream) and flips `cancelled`. SSE path
    // has its own Ctx::Drop, so don't arm this one on the streaming
    // branch.
    let gen_req = GenerateRequest {
        request_id,
        prompt_ids: prompt_ids.clone(),
        sampling,
        max_new_tokens: max_new,
        stop_token_ids: state.tokenizer.eos_token_ids().to_vec(),
        events_tx,
        cancelled: cancelled.clone(),
    };
    tracing::debug!(prompt_tokens = prompt_ids.len(), max_new, "submitting to worker");
    state.worker.submit(gen_req).await?;

    let model_id = state.config.model_id.clone();
    let tokenizer = state.tokenizer.clone();
    let request_timeout = state.config.request_timeout;

    if req.stream {
        Ok(ChatCompletionsResponse::Stream(chat_stream_sse(
            model_id,
            tokenizer,
            events_rx,
            cancelled,
            state.config.sse_keepalive,
            request_timeout,
        )))
    } else {
        let _cancel_guard = CancelOnDrop(cancelled.clone());
        let body = chat_collect(
            &model_id, &tokenizer, events_rx, cancelled, request_timeout,
        )
        .await?;
        Ok(ChatCompletionsResponse::Json(Json(body)))
    }
}

/// Either a JSON body (non-streaming) or an SSE response (streaming).
/// Streaming variant carries a pre-materialised `Response` — the
/// `Sse::keep_alive` return type is `Sse<KeepAliveStream<S>>` and
/// we don't want that nested generic in the handler signature.
pub enum ChatCompletionsResponse {
    Json(Json<ChatCompletionResponse>),
    Stream(axum::response::Response),
}

impl IntoResponse for ChatCompletionsResponse {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::Json(j) => j.into_response(),
            Self::Stream(r) => r,
        }
    }
}

async fn chat_collect(
    model_id: &str,
    tokenizer: &crate::tokenize::TokenizerHandle,
    mut events_rx: mpsc::Receiver<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    request_timeout: std::time::Duration,
) -> ApiResult<ChatCompletionResponse> {
    let mut token_ids: Vec<u32> = Vec::new();
    let mut finish: Option<FinishReason> = None;
    let mut usage = Usage::default();

    // Drain loop with wall-clock deadline. Deadline fires → flip
    // cancellation + return 504. Channel close (worker gone) also
    // breaks us out cleanly.
    let deadline = tokio::time::Instant::now() + request_timeout;
    loop {
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => {
                cancelled.store(true, Ordering::Relaxed);
                tracing::warn!(
                    secs = request_timeout.as_secs(),
                    "request timeout — cancelling worker",
                );
                return Err(ApiError::Timeout { secs: request_timeout.as_secs() });
            }
            ev = events_rx.recv() => match ev {
                Some(GenerateEvent::Token { id, .. }) => token_ids.push(id),
                Some(GenerateEvent::Done { finish: f, prompt_tokens, completion_tokens }) => {
                    finish = Some(f);
                    usage = Usage::new(prompt_tokens, completion_tokens);
                    break;
                }
                Some(GenerateEvent::Error(msg)) => return Err(ApiError::Internal(msg)),
                None => break, // worker channel closed
            },
        }
    }

    let text = tokenizer.decode(&token_ids)?;
    Ok(ChatCompletionResponse {
        id: new_chat_completion_id(),
        object: "chat.completion",
        created: unix_now_secs(),
        model: model_id.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatAssistantMessage { role: Role::Assistant, content: text },
            finish_reason: finish,
        }],
        usage,
    })
}

fn chat_stream_sse(
    model_id: String,
    tokenizer: crate::tokenize::TokenizerHandle,
    events_rx: mpsc::Receiver<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    keepalive: std::time::Duration,
    request_timeout: std::time::Duration,
) -> axum::response::Response {
    let id = new_chat_completion_id();
    let created = unix_now_secs();
    let model_for_chunk = model_id.clone();
    let decoder = tokenizer.stream_decoder();
    let deadline = tokio::time::Instant::now() + request_timeout;

    // State transitions:
    //   Role       — emit the first `delta.role = assistant` chunk
    //   Content    — forward incremental text deltas
    //   Finish(r)  — emit the terminal chunk with finish_reason
    //   Done       — emit "data: [DONE]\n\n"
    //   End        — stream over
    enum S {
        Role,
        Content {
            rx: mpsc::Receiver<GenerateEvent>,
            decoder: crate::tokenize::StreamDecoder,
        },
        Finish(FinishReason),
        Done,
        End,
    }

    struct Ctx {
        state: S,
        id: String,
        created: u64,
        model: String,
        cancelled: Arc<AtomicBool>,
        deadline: tokio::time::Instant,
    }

    impl Drop for Ctx {
        fn drop(&mut self) {
            // axum drops the Sse when the client disconnects. Signal
            // the worker to stop ASAP.
            self.cancelled.store(true, Ordering::Relaxed);
        }
    }

    let mut ctx = Ctx {
        state: S::Role,
        id,
        created,
        model: model_for_chunk,
        cancelled,
        deadline,
    };
    ctx.state = S::Content { rx: events_rx, decoder };
    let _ = tokenizer; // keep for clone clarity

    let stream = stream::unfold(ctx, |mut ctx| async move {
        loop {
            match std::mem::replace(&mut ctx.state, S::End) {
                S::Role => {
                    // First chunk: announce the role.
                    let chunk = ChatCompletionChunk {
                        id: ctx.id.clone(),
                        object: "chat.completion.chunk",
                        created: ctx.created,
                        model: ctx.model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta::role(Role::Assistant),
                            finish_reason: None,
                        }],
                    };
                    // Restore content state; caller must set rx/decoder before.
                    // (This arm never runs in practice — we transition to
                    // Content immediately above.)
                    let ev = sse_json(&chunk);
                    return Some((Ok(ev), ctx));
                }
                S::Content { mut rx, mut decoder } => {
                    let deadline = ctx.deadline;
                    let ev = tokio::select! {
                        biased;
                        _ = tokio::time::sleep_until(deadline) => {
                            ctx.cancelled.store(true, Ordering::Relaxed);
                            tracing::warn!("sse request timeout — cancelling worker");
                            // Emit a terminal Length chunk + [DONE].
                            ctx.state = S::Finish(FinishReason::Length);
                            continue;
                        }
                        ev = rx.recv() => ev,
                    };
                    match ev {
                        Some(GenerateEvent::Token { id, .. }) => {
                            match decoder.step(id) {
                                Ok(text) if text.is_empty() => {
                                    // Partial codepoint held back — loop.
                                    ctx.state = S::Content { rx, decoder };
                                    continue;
                                }
                                Ok(text) => {
                                    let chunk = ChatCompletionChunk {
                                        id: ctx.id.clone(),
                                        object: "chat.completion.chunk",
                                        created: ctx.created,
                                        model: ctx.model.clone(),
                                        choices: vec![ChatChunkChoice {
                                            index: 0,
                                            delta: ChatDelta::content(text),
                                            finish_reason: None,
                                        }],
                                    };
                                    ctx.state = S::Content { rx, decoder };
                                    return Some((Ok(sse_json(&chunk)), ctx));
                                }
                                Err(_e) => {
                                    // Decode failure — treat as Stop.
                                    ctx.state = S::Finish(FinishReason::Stop);
                                    continue;
                                }
                            }
                        }
                        Some(GenerateEvent::Done { finish, .. }) => {
                            ctx.state = S::Finish(finish);
                            continue;
                        }
                        Some(GenerateEvent::Error(_)) => {
                            // Error in stream — best we can do over SSE is
                            // emit a terminal finish chunk; OpenAI doesn't
                            // define an error frame. Log + finish.
                            ctx.state = S::Finish(FinishReason::Stop);
                            continue;
                        }
                        None => {
                            ctx.state = S::Finish(FinishReason::Stop);
                            continue;
                        }
                    }
                }
                S::Finish(reason) => {
                    let chunk = ChatCompletionChunk {
                        id: ctx.id.clone(),
                        object: "chat.completion.chunk",
                        created: ctx.created,
                        model: ctx.model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta::done(),
                            finish_reason: Some(reason),
                        }],
                    };
                    ctx.state = S::Done;
                    return Some((Ok(sse_json(&chunk)), ctx));
                }
                S::Done => {
                    ctx.state = S::End;
                    return Some((Ok(Event::default().data("[DONE]")), ctx));
                }
                S::End => return None,
            }
        }
    });

    { let boxed: SseStream = Box::pin(stream); Sse::new(boxed).keep_alive(axum::response::sse::KeepAlive::new().interval(keepalive)).into_response() }
}

fn sse_json<T: serde::Serialize>(value: &T) -> Event {
    // Serialize errors are impossible on our owned types; fall back
    // to a benign empty event if one somehow occurs so we can't
    // crash the stream.
    match serde_json::to_string(value) {
        Ok(s) => Event::default().data(s),
        Err(_) => Event::default().data("{}"),
    }
}

// ═════════════════════════════════════════════════════════════════════
// /v1/completions  (legacy text-completion)
// ═════════════════════════════════════════════════════════════════════

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> ApiResult<CompletionsResponse> {
    let request_id = Uuid::new_v4();
    let span = tracing::info_span!(
        "completions",
        request_id = %request_id,
        stream = req.stream,
    );
    let _enter = span.enter();

    ensure_model_matches(&state, &req.model)?;
    reject_v1_unsupported_completions(&req)?;

    let prompt_text = match &req.prompt {
        PromptField::One(s) => s.clone(),
        PromptField::Many(v) => {
            if v.len() != 1 {
                return Err(ApiError::invalid_param(
                    "batched string prompts not supported in v1 (send one request per prompt)",
                    "prompt",
                    "batch_unsupported",
                ));
            }
            v[0].clone()
        }
        PromptField::Tokens(_) | PromptField::TokensBatched(_) => {
            return Err(ApiError::invalid_param(
                "token-id prompts not supported in v1 — send strings",
                "prompt",
                "token_prompt_unsupported",
            ))
        }
    };
    if prompt_text.is_empty() {
        return Err(ApiError::invalid_param(
            "prompt must be non-empty",
            "prompt",
            "empty_prompt",
        ));
    }

    let sampling = req.sampling_params().ensure_supported()?;
    let max_new = resolve_max_new(req.max_tokens, state.config.max_new_tokens_cap)?;

    let prompt_ids = state.tokenizer.encode(&prompt_text)?;
    reject_oversized_prompt(prompt_ids.len(), max_new, state.config.max_new_tokens_cap)?;

    let (events_tx, events_rx) = mpsc::channel::<GenerateEvent>(64);
    let cancelled = Arc::new(AtomicBool::new(false));

    let gen_req = GenerateRequest {
        request_id,
        prompt_ids: prompt_ids.clone(),
        sampling,
        max_new_tokens: max_new,
        stop_token_ids: state.tokenizer.eos_token_ids().to_vec(),
        events_tx,
        cancelled: cancelled.clone(),
    };
    tracing::debug!(prompt_tokens = prompt_ids.len(), max_new, "submitting to worker");
    state.worker.submit(gen_req).await?;

    let model_id = state.config.model_id.clone();
    let tokenizer = state.tokenizer.clone();
    let request_timeout = state.config.request_timeout;

    if req.stream {
        Ok(CompletionsResponse::Stream(completion_stream_sse(
            model_id,
            tokenizer,
            events_rx,
            cancelled,
            state.config.sse_keepalive,
            request_timeout,
        )))
    } else {
        let _cancel_guard = CancelOnDrop(cancelled.clone());
        let body = completion_collect(
            &model_id,
            &tokenizer,
            events_rx,
            cancelled,
            request_timeout,
        )
        .await?;
        Ok(CompletionsResponse::Json(Json(body)))
    }
}

pub enum CompletionsResponse {
    Json(Json<CompletionResponse>),
    Stream(axum::response::Response),
}

impl IntoResponse for CompletionsResponse {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::Json(j) => j.into_response(),
            Self::Stream(r) => r,
        }
    }
}

async fn completion_collect(
    model_id: &str,
    tokenizer: &crate::tokenize::TokenizerHandle,
    mut events_rx: mpsc::Receiver<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    request_timeout: std::time::Duration,
) -> ApiResult<CompletionResponse> {
    let mut token_ids: Vec<u32> = Vec::new();
    let mut finish: Option<FinishReason> = None;
    let mut usage = Usage::default();

    let deadline = tokio::time::Instant::now() + request_timeout;
    loop {
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => {
                cancelled.store(true, Ordering::Relaxed);
                tracing::warn!(
                    secs = request_timeout.as_secs(),
                    "request timeout — cancelling worker",
                );
                return Err(ApiError::Timeout { secs: request_timeout.as_secs() });
            }
            ev = events_rx.recv() => match ev {
                Some(GenerateEvent::Token { id, .. }) => token_ids.push(id),
                Some(GenerateEvent::Done { finish: f, prompt_tokens, completion_tokens }) => {
                    finish = Some(f);
                    usage = Usage::new(prompt_tokens, completion_tokens);
                    break;
                }
                Some(GenerateEvent::Error(msg)) => return Err(ApiError::Internal(msg)),
                None => break,
            },
        }
    }
    let text = tokenizer.decode(&token_ids)?;
    Ok(CompletionResponse {
        id: new_completion_id(),
        object: "text_completion",
        created: unix_now_secs(),
        model: model_id.to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: finish,
            logprobs: None,
        }],
        usage,
    })
}

fn completion_stream_sse(
    model_id: String,
    tokenizer: crate::tokenize::TokenizerHandle,
    events_rx: mpsc::Receiver<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    keepalive: std::time::Duration,
    request_timeout: std::time::Duration,
) -> axum::response::Response {
    let id = new_completion_id();
    let created = unix_now_secs();
    let deadline = tokio::time::Instant::now() + request_timeout;

    enum S {
        Content {
            rx: mpsc::Receiver<GenerateEvent>,
            decoder: crate::tokenize::StreamDecoder,
        },
        Finish(FinishReason),
        Done,
        End,
    }

    struct Ctx {
        state: S,
        id: String,
        created: u64,
        model: String,
        cancelled: Arc<AtomicBool>,
        deadline: tokio::time::Instant,
    }

    impl Drop for Ctx {
        fn drop(&mut self) {
            self.cancelled.store(true, Ordering::Relaxed);
        }
    }

    let ctx = Ctx {
        state: S::Content { rx: events_rx, decoder: tokenizer.stream_decoder() },
        id,
        created,
        model: model_id,
        cancelled,
        deadline,
    };

    let stream = stream::unfold(ctx, |mut ctx| async move {
        loop {
            match std::mem::replace(&mut ctx.state, S::End) {
                S::Content { mut rx, mut decoder } => {
                    let deadline = ctx.deadline;
                    let ev = tokio::select! {
                        biased;
                        _ = tokio::time::sleep_until(deadline) => {
                            ctx.cancelled.store(true, Ordering::Relaxed);
                            tracing::warn!("sse request timeout — cancelling worker");
                            ctx.state = S::Finish(FinishReason::Length);
                            continue;
                        }
                        ev = rx.recv() => ev,
                    };
                    match ev {
                    Some(GenerateEvent::Token { id, .. }) => match decoder.step(id) {
                        Ok(text) if text.is_empty() => {
                            ctx.state = S::Content { rx, decoder };
                            continue;
                        }
                        Ok(text) => {
                            let chunk = CompletionChunk {
                                id: ctx.id.clone(),
                                object: "text_completion",
                                created: ctx.created,
                                model: ctx.model.clone(),
                                choices: vec![CompletionChunkChoice {
                                    index: 0,
                                    text,
                                    finish_reason: None,
                                    logprobs: None,
                                }],
                            };
                            ctx.state = S::Content { rx, decoder };
                            return Some((Ok(sse_json(&chunk)), ctx));
                        }
                        Err(_) => {
                            ctx.state = S::Finish(FinishReason::Stop);
                            continue;
                        }
                    },
                    Some(GenerateEvent::Done { finish, .. }) => {
                        ctx.state = S::Finish(finish);
                        continue;
                    }
                    Some(GenerateEvent::Error(_)) | None => {
                        ctx.state = S::Finish(FinishReason::Stop);
                        continue;
                    }
                    }
                }
                S::Finish(reason) => {
                    let chunk = CompletionChunk {
                        id: ctx.id.clone(),
                        object: "text_completion",
                        created: ctx.created,
                        model: ctx.model.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(reason),
                            logprobs: None,
                        }],
                    };
                    ctx.state = S::Done;
                    return Some((Ok(sse_json(&chunk)), ctx));
                }
                S::Done => {
                    ctx.state = S::End;
                    return Some((Ok(Event::default().data("[DONE]")), ctx));
                }
                S::End => return None,
            }
        }
    });

    { let boxed: SseStream = Box::pin(stream); Sse::new(boxed).keep_alive(axum::response::sse::KeepAlive::new().interval(keepalive)).into_response() }
}

// ═════════════════════════════════════════════════════════════════════
// Helpers shared by both handlers
// ═════════════════════════════════════════════════════════════════════

fn reject_v1_unsupported_chat(req: &ChatCompletionRequest) -> ApiResult<()> {
    if req.n.is_some_and(|n| n != 1) {
        return Err(ApiError::invalid_param(
            "n must be 1 (multi-candidate sampling not yet supported)",
            "n",
            "multi_candidate_unsupported",
        ));
    }
    if req.logprobs.is_some_and(|b| b) || req.top_logprobs.is_some() {
        return Err(ApiError::invalid_param(
            "logprobs not yet supported",
            "logprobs",
            "logprobs_unsupported",
        ));
    }
    if req.logit_bias.is_some() {
        return Err(ApiError::invalid_param(
            "logit_bias not yet supported",
            "logit_bias",
            "logit_bias_unsupported",
        ));
    }
    if req.tools.is_some() || req.tool_choice.is_some() {
        return Err(ApiError::invalid_param(
            "tools / tool_choice not yet supported",
            "tools",
            "tools_unsupported",
        ));
    }
    Ok(())
}

fn reject_v1_unsupported_completions(req: &CompletionRequest) -> ApiResult<()> {
    if req.n.is_some_and(|n| n != 1) {
        return Err(ApiError::invalid_param(
            "n must be 1",
            "n",
            "multi_candidate_unsupported",
        ));
    }
    if req.logprobs.is_some() {
        return Err(ApiError::invalid_param(
            "logprobs not yet supported",
            "logprobs",
            "logprobs_unsupported",
        ));
    }
    if req.logit_bias.is_some() {
        return Err(ApiError::invalid_param(
            "logit_bias not yet supported",
            "logit_bias",
            "logit_bias_unsupported",
        ));
    }
    if req.echo.unwrap_or(false) {
        return Err(ApiError::invalid_param(
            "echo=true not yet supported",
            "echo",
            "echo_unsupported",
        ));
    }
    if req.suffix.is_some() {
        return Err(ApiError::invalid_param(
            "suffix (FIM) not yet supported",
            "suffix",
            "suffix_unsupported",
        ));
    }
    Ok(())
}

fn resolve_max_new(requested: Option<u32>, cap: u32) -> ApiResult<u32> {
    let m = requested.unwrap_or(cap.min(1024));
    if m == 0 {
        return Err(ApiError::invalid_param(
            "max_tokens must be > 0",
            "max_tokens",
            "invalid_max_tokens",
        ));
    }
    if m > cap {
        return Err(ApiError::invalid_param(
            format!("max_tokens {m} exceeds server cap {cap}"),
            "max_tokens",
            "max_tokens_too_large",
        ));
    }
    Ok(m)
}

fn reject_oversized_prompt(
    prompt_len: usize,
    max_new: u32,
    _cap: u32,
) -> ApiResult<()> {
    // Cheap upper bound — model max context is enforced downstream
    // by the runtime (block_tables). We refuse wildly-large prompts
    // here so we never load the worker for a request that will fail.
    const SOFT_MAX_PROMPT: usize = 200_000;
    if prompt_len > SOFT_MAX_PROMPT {
        return Err(ApiError::invalid_param(
            format!("prompt is {prompt_len} tokens — soft limit is {SOFT_MAX_PROMPT}"),
            "messages",
            "prompt_too_large",
        ));
    }
    if prompt_len as u64 + max_new as u64 > 262_144 {
        return Err(ApiError::invalid_param(
            "prompt + max_tokens exceeds model max context (256k)",
            "max_tokens",
            "context_too_large",
        ));
    }
    Ok(())
}

fn extract_stop(s: &StopField) -> Vec<String> {
    match s {
        StopField::One(s) => {
            if s.is_empty() { vec![] } else { vec![s.clone()] }
        }
        StopField::Many(v) => v.iter().filter(|x| !x.is_empty()).cloned().collect(),
    }
}

fn validate_stops(stops: &[String]) -> ApiResult<()> {
    if stops.len() > 4 {
        return Err(ApiError::invalid_param(
            "stop can have at most 4 strings",
            "stop",
            "too_many_stops",
        ));
    }
    Ok(())
}
