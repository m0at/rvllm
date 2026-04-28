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

// Request/response dump hook (env-gated via `RVLLM_DUMP_REQUEST_DIR`).
// Writes per-request JSON files capturing the full request + generated
// token prefix so the failing prompt can be replayed byte-for-byte
// outside of the original caller (zeroclaw / client). Introduced to
// close the instrumentation gap during NVFP4 quality investigation.
// See v3/GB10_SPEC.md (if present) or commit message for the matching
// cap-sweep experiment design.
use std::sync::OnceLock;
static RVLLM_DUMP_DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();

fn dump_dir() -> Option<&'static std::path::Path> {
    RVLLM_DUMP_DIR
        .get_or_init(|| {
            std::env::var("RVLLM_DUMP_REQUEST_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .map(std::path::PathBuf::from)
        })
        .as_deref()
}

fn dump_write(request_id: &Uuid, suffix: &str, body: &serde_json::Value) {
    let Some(dir) = dump_dir() else { return };
    if !dir.exists() {
        let _ = std::fs::create_dir_all(dir);
    }
    let path = dir.join(format!("{}_{}.json", request_id, suffix));
    match serde_json::to_vec_pretty(body) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, &bytes) {
                tracing::warn!(?path, error = %e, "dump write failed");
            }
        }
        Err(e) => tracing::warn!(error = %e, "dump serialize failed"),
    }
}

// === SLASH-COMMAND SHORT-CIRCUIT helpers ===
// `/new`, `/clear`, `/reset` are treated as session-control no-ops at
// the HTTP boundary. They never tokenise, never reach the cuda worker,
// never burn the NVFP4 shadow latch — important because clients
// (zeroclaw, rusty-dashboard) historically forward `/new` as a real
// 3057-token prompt that fires shadow allocation + dump on a request
// the operator did NOT mean to instrument.

/// Pick the canned reply for a slash-command-only chat request, or
/// `None` if the request is real work. Compares the LAST user message,
/// trimmed; case-sensitive on the command word; whole-message match
/// only (embedded "/new" in a longer prompt is NOT a slash command).
fn slash_command_reply(req: &crate::openai::chat::ChatCompletionRequest) -> Option<&'static str> {
    let last_user = req.messages.iter().rev()
        .find(|m| m.role == Role::User)?
        .content.as_deref()?;
    match last_user.trim() {
        "/new"   => Some("Session reset."),
        "/clear" => Some("Session cleared."),
        "/reset" => Some("Session reset."),
        _ => None,
    }
}

/// One-shot SSE body for the streaming case of a slash-command short-
/// circuit. Emits a single ChatCompletionChunk with the canned reply
/// + finish_reason=stop, then `[DONE]`.
fn slash_command_sse(
    body: ChatCompletionResponse,
    request_id: Uuid,
) -> Sse<SseStream> {
    use futures::stream;
    let chunk_id = body.id.clone();
    let model = body.model.clone();
    let created = body.created;
    let content = body.choices.first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();
    let chunk = ChatCompletionChunk {
        id: chunk_id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some(Role::Assistant),
                content: Some(content),
                tool_calls: None,
            },
            finish_reason: Some(FinishReason::Stop),
        }],
    };
    let _ = request_id;
    let events: SseStream = Box::pin(stream::iter(vec![
        Ok(sse_json(&chunk)),
        Ok(Event::default().data("[DONE]")),
    ]));
    Sse::new(events)
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
    new_chat_completion_id, new_completion_id, new_tool_call_id, unix_now_secs, FinishReason,
    Role, ToolCall, ToolCallFunction, Usage,
};
use crate::tool_parser::{
    parse_gemma4_tool_calls, strip_tool_markup,
    THOUGHT_BLOCK_OPENERS, TOOL_CALL_OPENER,
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
    _headers: axum::http::HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> ApiResult<ChatCompletionsResponse> {
    let request_id = Uuid::new_v4();
    let span = tracing::info_span!(
        "chat_completions",
        request_id = %request_id,
        stream = req.stream,
    );
    let _enter = span.enter();

    // Request dump (no-op unless RVLLM_DUMP_REQUEST_DIR is set). The
    // request body is already owned (from the `Json(req)` extractor);
    // reserialise selected fields into a stable shape for replay.
    dump_write(&request_id, "request", &serde_json::json!({
        "model": &req.model,
        "messages": &req.messages,
        "tools": &req.tools,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
        "stop": &req.stop,
    }));

    ensure_model_matches(&state, &req.model)?;
    reject_v1_unsupported_chat(&req)?;

    if req.messages.is_empty() {
        return Err(ApiError::invalid_param(
            "messages must be non-empty",
            "messages",
            "empty_messages",
        ));
    }

    // === SLASH-COMMAND SHORT-CIRCUIT ===
    // When a client sends a session-control command like `/new` or
    // `/clear` as a chat message, treat it as a no-op at the rvllm
    // boundary: return a canned assistant reply, never tokenise, never
    // reach the worker, never allocate shadow KV, never burn the
    // diagnostic latch. Comparison is case-sensitive and the command
    // must be the entire user message after trim — embedding "/new"
    // inside a real prompt does NOT trigger.
    if let Some(reply) = slash_command_reply(&req) {
        tracing::info!(
            request_id = %request_id,
            "slash-command short-circuit (no GPU work, no shadow burn)"
        );
        let body = ChatCompletionResponse {
            id: new_chat_completion_id(),
            object: "chat.completion",
            created: unix_now_secs(),
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatAssistantMessage {
                    role: Role::Assistant,
                    content: Some(reply.into()),
                    tool_calls: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage::new(0, 0),
        };
        if req.stream {
            return Ok(ChatCompletionsResponse::Stream(
                slash_command_sse(body, request_id).into_response()
            ));
        }
        return Ok(ChatCompletionsResponse::Json(Json(body)));
    }

    let sampling = req.sampling_params().ensure_supported()?;
    let max_new = resolve_max_new(req.max_tokens, state.config.max_new_tokens_cap)?;
    let stop_text = req.stop.as_ref().map(|s| extract_stop(s)).unwrap_or_default();
    validate_stops(&stop_text)?;
    // Cycle 37 P1 (codex audit): cycle-36 reject of stream+stop fired
    // AFTER tokenization + worker submit, leaking GPU work for a
    // request we then 400. Lift the rejection above the tokenize/submit
    // sequence so it's a free-cost validation error.
    if req.stream && !stop_text.is_empty() {
        return Err(ApiError::invalid_param(
            "`stop` is not yet supported with stream=true (would be \
             silently ignored). Use stream=false, or wait for \
             streaming-truncation support.",
            "stop",
            "stop_with_stream_unsupported",
        ));
    }

    let prompt_ids = state.tokenizer.render_chat(&req.messages, req.tools.as_ref())?;
    reject_oversized_prompt(prompt_ids.len(), max_new, state.config.max_new_tokens_cap)?;

    // Dump prompt_tokens count + first 32 IDs for the request. The
    // full prompt_ids are stored in the `_request.json` dump's
    // `messages` array, reconstructable via `render_chat()`.
    dump_write(&request_id, "prompt_tokens", &serde_json::json!({
        "prompt_tokens": prompt_ids.len(),
        "first_32_prompt_ids": &prompt_ids[..prompt_ids.len().min(32)],
        "last_32_prompt_ids": &prompt_ids[prompt_ids.len().saturating_sub(32)..],
    }));

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
        // Cycle 37: stream+stop rejection lifted to pre-tokenize above.
        Ok(ChatCompletionsResponse::Stream(chat_stream_sse(
            model_id,
            tokenizer,
            events_rx,
            cancelled,
            state.config.sse_keepalive,
            request_timeout,
            request_id,
        )))
    } else {
        let _cancel_guard = CancelOnDrop(cancelled.clone());
        let body = chat_collect(
            &model_id, &tokenizer, events_rx, cancelled, request_timeout,
            request_id, &stop_text,
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

#[allow(unused_assignments)] // initial `let mut finish/usage = ...` overwritten before read in loop body
async fn chat_collect(
    model_id: &str,
    tokenizer: &crate::tokenize::TokenizerHandle,
    mut events_rx: mpsc::Receiver<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    request_timeout: std::time::Duration,
    request_id: Uuid,
    stop_text: &[String],
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
                // Cycle 34 P0 fix (codex bug #2): worker channel closing
                // without a Done event is a worker crash, not a clean
                // completion. Returning Ok with finish=None lets clients
                // mistake a CUDA fault for a normal short response. Surface
                // it as a 500 instead.
                None => return Err(ApiError::Internal(
                    "worker channel closed before Done event".into(),
                )),
            },
        }
    }

    // Decode with special tokens preserved so the tool_parser + thought-
    // strip can match `<|tool_call>...<tool_call|>` and
    // `<|channel>thought\n...<channel|>` markers verbatim.
    // `shape_assistant_message` produces the user-visible content view.
    let mut text = tokenizer.decode_raw(&token_ids)?;
    // Cycle 34 P0 fix (codex bug #1): user-supplied stop strings.
    if apply_stop_truncation(&mut text, stop_text) {
        finish = Some(FinishReason::Stop);
    }

    // Response dump (no-op unless RVLLM_DUMP_REQUEST_DIR set). Captures
    // the generated token prefix for collapse-index analysis on the
    // cap-sweep experiment (32/128/256/512/1024 under NVFP4-MSE vs FP8).
    // Note `text` is the RAW decoded output with special tokens kept —
    // the user-visible content is derived by `shape_assistant_message`
    // below.
    let prefix_ids = &token_ids[..token_ids.len().min(32)];
    let prefix_text = tokenizer.decode_raw(prefix_ids).unwrap_or_default();
    dump_write(&request_id, "response", &serde_json::json!({
        "first_32_token_ids": prefix_ids,
        "first_32_decoded_raw": prefix_text,
        "total_generated_tokens": token_ids.len(),
        "finish_reason": finish,
        "usage_prompt_tokens": usage.prompt_tokens,
        "usage_completion_tokens": usage.completion_tokens,
        "full_raw_text": &text,
    }));

    let (message, finish_reason) = shape_assistant_message(text, finish);
    Ok(ChatCompletionResponse {
        id: new_chat_completion_id(),
        object: "chat.completion",
        created: unix_now_secs(),
        model: model_id.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage,
    })
}

/// Gemma 4 emits tool calls as plain text inside the reply (see
/// `tool_parser`). If the decoded text contains any `<|tool_call>...`
/// markup, hoist it into OpenAI's `tool_calls` shape and flip
/// `finish_reason` to `ToolCalls`. Otherwise return the text verbatim.
/// Truncate `text` at the FIRST occurrence of ANY stop string.
/// Returns true when truncation happened (caller should set
/// `finish_reason = Stop`). Empty stop list is a no-op.
///
/// Pure function: no I/O, no allocations beyond the find. Safe to
/// unit test without a tokenizer or worker.
pub(crate) fn apply_stop_truncation(text: &mut String, stops: &[String]) -> bool {
    if stops.is_empty() {
        return false;
    }
    let mut earliest: Option<usize> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s.as_str()) {
            earliest = Some(earliest.map_or(pos, |e| e.min(pos)));
        }
    }
    match earliest {
        Some(pos) => {
            text.truncate(pos);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod stop_truncation_tests {
    use super::apply_stop_truncation;

    #[test]
    fn empty_stops_is_noop() {
        let mut t = "hello world".to_string();
        assert!(!apply_stop_truncation(&mut t, &[]));
        assert_eq!(t, "hello world");
    }

    #[test]
    fn single_stop_truncates() {
        let mut t = "hello STOP world".to_string();
        let stops = vec!["STOP".to_string()];
        assert!(apply_stop_truncation(&mut t, &stops));
        assert_eq!(t, "hello ");
    }

    #[test]
    fn earliest_of_multiple_stops_wins() {
        let mut t = "a B c D e".to_string();
        let stops = vec!["D".to_string(), "B".to_string()];
        assert!(apply_stop_truncation(&mut t, &stops));
        assert_eq!(t, "a "); // B comes before D
    }

    #[test]
    fn no_match_leaves_text_intact() {
        let mut t = "no stop here".to_string();
        let stops = vec!["XYZ".to_string()];
        assert!(!apply_stop_truncation(&mut t, &stops));
        assert_eq!(t, "no stop here");
    }

    #[test]
    fn empty_stop_string_is_skipped() {
        // An empty needle would match at index 0 and truncate everything.
        // Sanity: validate_stops should reject these upstream, but the
        // truncator is defensive — still skip empty needles.
        let mut t = "keep this".to_string();
        let stops = vec!["".to_string()];
        assert!(!apply_stop_truncation(&mut t, &stops));
        assert_eq!(t, "keep this");
    }

    #[test]
    fn utf8_stop_string_works() {
        let mut t = "Es regnet in München.".to_string();
        let stops = vec!["München".to_string()];
        assert!(apply_stop_truncation(&mut t, &stops));
        assert_eq!(t, "Es regnet in ");
    }

    #[test]
    fn stop_at_index_zero_yields_empty() {
        let mut t = "STOP_at_start".to_string();
        let stops = vec!["STOP".to_string()];
        assert!(apply_stop_truncation(&mut t, &stops));
        assert_eq!(t, "");
    }
}

fn shape_assistant_message(
    text: String,
    model_finish: Option<FinishReason>,
) -> (ChatAssistantMessage, Option<FinishReason>) {
    let parsed = parse_gemma4_tool_calls(&text);
    if parsed.is_empty() {
        // No tool call — but the raw decode still contains reasoning /
        // control markup (`<|channel>thought\n…<channel|>`, stray
        // `<turn|>`, etc.) that must be stripped from user content.
        let cleaned = strip_tool_markup(&text);
        let content = if cleaned.is_empty() { None } else { Some(cleaned) };
        return (
            ChatAssistantMessage { role: Role::Assistant, content, tool_calls: None },
            model_finish,
        );
    }
    let calls = parsed
        .into_iter()
        .map(|p| ToolCall {
            id: new_tool_call_id(),
            kind: "function",
            function: ToolCallFunction { name: p.name, arguments: p.arguments },
        })
        .collect();
    // Preserve any prose the model emitted before the first call so
    // clients that show reasoning alongside the call still get it.
    let prefix = strip_tool_markup(&text);
    let content = if prefix.is_empty() { None } else { Some(prefix) };
    (
        ChatAssistantMessage { role: Role::Assistant, content, tool_calls: Some(calls) },
        Some(FinishReason::ToolCalls),
    )
}

fn chat_stream_sse(
    model_id: String,
    tokenizer: crate::tokenize::TokenizerHandle,
    events_rx: mpsc::Receiver<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    keepalive: std::time::Duration,
    request_timeout: std::time::Duration,
    request_id: Uuid,
) -> axum::response::Response {
    let id = new_chat_completion_id();
    let created = unix_now_secs();
    let model_for_chunk = model_id.clone();
    // RAW decoder — preserves `<|tool_call>` / `<|channel>` / etc so
    // `strip_tool_markup` and the tool parser can match them. Pre-fix
    // we used `stream_decoder()` (skip_special_tokens=true), which
    // dropped the markers before they could be recognised, leaking
    // thought-channel prose to the client and producing different
    // visible text than the non-streaming path.
    let decoder = tokenizer.stream_decoder_raw();
    let deadline = tokio::time::Instant::now() + request_timeout;

    // State transitions:
    //   Role       — emit the first `delta.role = assistant` chunk
    //                (carries the same content fields so the next loop
    //                iteration can transition to Content without
    //                re-allocating).
    //   Content    — forward incremental visible-text deltas. `accum`
    //                is the RAW decoded run with control markers
    //                preserved; `strip_tool_markup` is applied to a
    //                safe prefix on each token to compute the visible
    //                view, and only its new suffix ships as content.
    //   FlushToolCalls — emit the parsed tool_calls as one delta.
    //   Finish(r)  — emit the terminal chunk with finish_reason.
    //   Done       — emit "data: [DONE]\n\n".
    //   End        — stream over.
    //
    // Per-request disk dump (gated by RVLLM_DUMP_REQUEST_DIR) ships at
    // GenerateEvent::Done so the response file pairs with the
    // request file written by the parent handler.
    enum S {
        Role {
            rx: mpsc::Receiver<GenerateEvent>,
            decoder: crate::tokenize::StreamDecoder,
            accum: String,
            emitted: usize,
            in_tool: bool,
            token_ids: Vec<u32>,
            streamed_visible: String,
        },
        Content {
            rx: mpsc::Receiver<GenerateEvent>,
            decoder: crate::tokenize::StreamDecoder,
            /// Full RAW decoded text so far (markers preserved).
            accum: String,
            /// Bytes of the *visible* (post-strip) text already flushed.
            emitted: usize,
            /// Latched once we see the tool-call opener — from then on
            /// nothing more goes out as content.
            in_tool: bool,
            /// Captured for the response dump.
            token_ids: Vec<u32>,
            streamed_visible: String,
        },
        FlushToolCalls(Vec<ToolCall>),
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
        request_id: Uuid,
    }

    impl Drop for Ctx {
        fn drop(&mut self) {
            // axum drops the Sse when the client disconnects. Signal
            // the worker to stop ASAP.
            self.cancelled.store(true, Ordering::Relaxed);
        }
    }

    let ctx = Ctx {
        state: S::Role {
            rx: events_rx,
            decoder,
            accum: String::new(),
            emitted: 0,
            in_tool: false,
            token_ids: Vec::new(),
            streamed_visible: String::new(),
        },
        id,
        created,
        model: model_for_chunk,
        cancelled,
        deadline,
        request_id,
    };
    let _ = tokenizer; // keep for clone clarity

    let stream = stream::unfold(ctx, |mut ctx| async move {
        loop {
            match std::mem::replace(&mut ctx.state, S::End) {
                S::Role { rx, decoder, accum, emitted, in_tool, token_ids, streamed_visible } => {
                    // First chunk: announce the role. Then transition
                    // to Content with the same buffers so the next
                    // loop iteration can attach them.
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
                    ctx.state = S::Content {
                        rx, decoder, accum, emitted, in_tool, token_ids, streamed_visible,
                    };
                    return Some((Ok(sse_json(&chunk)), ctx));
                }
                S::Content {
                    mut rx, mut decoder, mut accum, mut emitted, mut in_tool,
                    mut token_ids, mut streamed_visible,
                } => {
                    let deadline = ctx.deadline;
                    let ev = tokio::select! {
                        biased;
                        _ = tokio::time::sleep_until(deadline) => {
                            ctx.cancelled.store(true, Ordering::Relaxed);
                            tracing::warn!("sse request timeout — cancelling worker");
                            // Cycle 36 P1 (codex audit): timeout was reported
                            // as FinishReason::Length, indistinguishable from
                            // hitting max_tokens. Now Cancelled — clients can
                            // tell wall-clock-timeout apart from natural stop.
                            ctx.state = S::Finish(FinishReason::Cancelled);
                            continue;
                        }
                        ev = rx.recv() => ev,
                    };
                    match ev {
                        Some(GenerateEvent::Token { id, .. }) => {
                            token_ids.push(id);
                            match decoder.step(id) {
                                Ok(text) if text.is_empty() => {
                                    ctx.state = S::Content {
                                        rx, decoder, accum, emitted, in_tool,
                                        token_ids, streamed_visible,
                                    };
                                    continue;
                                }
                                Ok(text) => {
                                    accum.push_str(&text);
                                    // Latch in_tool the moment a tool-call
                                    // marker appears anywhere in accum.
                                    if !in_tool
                                        && (accum.contains(TOOL_CALL_OPENER)
                                            || find_anchored_call(&accum, 0)
                                                .map(|p| accum[p + 5..].contains('{'))
                                                .unwrap_or(false))
                                    {
                                        in_tool = true;
                                    }
                                    // Compute the safe RAW prefix (holds back
                                    // anything that could start an unclosed
                                    // marker), then strip control markup to
                                    // get the visible view, then ship only
                                    // the new visible suffix.
                                    let safe_end = safe_content_emit_end(
                                        &accum, 0, in_tool,
                                    );
                                    if safe_end > 0 {
                                        let visible = strip_tool_markup(&accum[..safe_end]);
                                        if visible.len() > emitted {
                                            let chunk_text =
                                                visible[emitted..].to_string();
                                            emitted = visible.len();
                                            streamed_visible.push_str(&chunk_text);
                                            let chunk = ChatCompletionChunk {
                                                id: ctx.id.clone(),
                                                object: "chat.completion.chunk",
                                                created: ctx.created,
                                                model: ctx.model.clone(),
                                                choices: vec![ChatChunkChoice {
                                                    index: 0,
                                                    delta: ChatDelta::content(chunk_text),
                                                    finish_reason: None,
                                                }],
                                            };
                                            ctx.state = S::Content {
                                                rx, decoder, accum, emitted, in_tool,
                                                token_ids, streamed_visible,
                                            };
                                            return Some((Ok(sse_json(&chunk)), ctx));
                                        }
                                    }
                                    ctx.state = S::Content {
                                        rx, decoder, accum, emitted, in_tool,
                                        token_ids, streamed_visible,
                                    };
                                    continue;
                                }
                                Err(e) => {
                                    // Cycle 36 P1 (codex audit): decoder
                                    // step errors were silently mapped to
                                    // FinishReason::Stop, hiding tokenizer
                                    // failures. Log + Cancelled instead.
                                    tracing::error!(error = ?e,
                                        "chat SSE decoder step failed — terminating stream");
                                    ctx.state = S::Finish(FinishReason::Cancelled);
                                    continue;
                                }
                            }
                        }
                        Some(GenerateEvent::Done { finish, prompt_tokens, completion_tokens }) => {
                            // Dump streaming response (no-op unless
                            // RVLLM_DUMP_REQUEST_DIR set). Pairs with
                            // the request dump written at handler entry.
                            let prefix_n = token_ids.len().min(32);
                            dump_write(&ctx.request_id, "response_stream", &serde_json::json!({
                                "first_32_token_ids": &token_ids[..prefix_n],
                                "total_generated_tokens": token_ids.len(),
                                "full_raw_text": &accum,
                                "streamed_visible_text": &streamed_visible,
                                "finish_reason": finish,
                                "usage_prompt_tokens": prompt_tokens,
                                "usage_completion_tokens": completion_tokens,
                            }));

                            // At Done, inspect the full buffer: if it
                            // carries tool-call markup, emit one final
                            // `tool_calls` delta and flip finish to
                            // ToolCalls. Otherwise drain any visible
                            // suffix held back by safe_content_emit_end.
                            let parsed = parse_gemma4_tool_calls(&accum);
                            if !parsed.is_empty() {
                                let calls = parsed
                                    .into_iter()
                                    .map(|p| ToolCall {
                                        id: new_tool_call_id(),
                                        kind: "function",
                                        function: ToolCallFunction {
                                            name: p.name,
                                            arguments: p.arguments,
                                        },
                                    })
                                    .collect();
                                ctx.state = S::FlushToolCalls(calls);
                                continue;
                            }
                            // No tool call — drain residue. Apply
                            // strip_tool_markup to the FULL accum (no
                            // tail-keep at finalisation) and emit the
                            // suffix beyond what was already streamed.
                            let visible_full = strip_tool_markup(&accum);
                            if visible_full.len() > emitted {
                                let tail = visible_full[emitted..].to_string();
                                let chunk = ChatCompletionChunk {
                                    id: ctx.id.clone(),
                                    object: "chat.completion.chunk",
                                    created: ctx.created,
                                    model: ctx.model.clone(),
                                    choices: vec![ChatChunkChoice {
                                        index: 0,
                                        delta: ChatDelta::content(tail),
                                        finish_reason: None,
                                    }],
                                };
                                ctx.state = S::Finish(finish);
                                return Some((Ok(sse_json(&chunk)), ctx));
                            }
                            ctx.state = S::Finish(finish);
                            continue;
                        }
                        // Cycle 34 P0 fix (codex bug #2): SSE was mapping
                        // worker error AND channel-close to FinishReason::Stop —
                        // a CUDA crash looked like a clean completion to the
                        // client. Now we log loudly + emit an error-flavored
                        // SSE sentinel chunk so operators can see the failure
                        // in journalctl AND clients can detect via the
                        // non-`stop` finish_reason.
                        Some(GenerateEvent::Error(msg)) => {
                            tracing::error!(error = %msg, "SSE worker error event — terminating stream");
                            ctx.state = S::Finish(FinishReason::Cancelled);
                            continue;
                        }
                        None => {
                            tracing::error!("SSE worker channel closed without Done — terminating stream");
                            ctx.state = S::Finish(FinishReason::Cancelled);
                            continue;
                        }
                    }
                }
                S::FlushToolCalls(calls) => {
                    let chunk = ChatCompletionChunk {
                        id: ctx.id.clone(),
                        object: "chat.completion.chunk",
                        created: ctx.created,
                        model: ctx.model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta::tool_calls(calls),
                            finish_reason: None,
                        }],
                    };
                    ctx.state = S::Finish(FinishReason::ToolCalls);
                    return Some((Ok(sse_json(&chunk)), ctx));
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

/// How many bytes of the raw `accum` can safely leave the server as
/// SSE `delta.content` after `strip_tool_markup` is applied.
///
/// Rules:
///   * `in_tool == true` — hold everything; the FlushToolCalls stage
///     will emit the parsed calls.
///   * otherwise — emit up to the earliest byte that could start a
///     Gemma 4 control marker WHOSE CLOSER HASN'T ARRIVED YET:
///       - tier-1 tool call: `<|tool_call>` opener.
///       - thought blocks: any of `THOUGHT_BLOCK_OPENERS` whose
///         `<channel|>` / `<turn|>` closer hasn't appeared in `accum`.
///         Holding here is what stops the streaming path from leaking
///         thought-channel prose: `strip_tool_markup` on a span that
///         contains an open `<|channel>` without its closer drops
///         only the opener and keeps the prose, which used to slip
///         through to the client. By holding the cursor at the opener
///         until the closer arrives, the eventual strip removes the
///         WHOLE block in one shot.
///       - tier-2: a word-anchored `call:` sequence.
///     Also hold back the final `max(opener.len()) - 1` bytes in case
///     a marker is straddling chunks. Split on UTF-8 boundaries.
fn safe_content_emit_end(accum: &str, emitted: usize, in_tool: bool) -> usize {
    if in_tool {
        return emitted;
    }
    // Earliest anchored tier-2 marker in the unemitted region.
    let tier2 = find_anchored_call(accum, emitted);
    // Earliest tier-1 wrapper start (tool-call opener — always hold,
    // even if the closer is already present, so we can detect it as a
    // real call at Done time and emit `tool_calls` instead of content).
    let tier1 = accum[emitted..]
        .find(TOOL_CALL_OPENER)
        .map(|r| emitted + r);
    // Earliest UNCLOSED thought-block opener. If the opener IS closed,
    // strip_tool_markup will collapse the whole block — safe to emit
    // through. If it's NOT closed, we must hold back until the closer
    // arrives so the inner prose doesn't leak.
    let thought = THOUGHT_BLOCK_OPENERS
        .iter()
        .filter_map(|&op| {
            let pos = accum[emitted..].find(op).map(|r| emitted + r)?;
            let after_opener = pos + op.len();
            // Closer search: prefer `<channel|>`, accept `<turn|>` too
            // (matches strip_tool_markup behaviour).
            let tail = &accum[after_opener..];
            let has_close = tail.contains("<channel|>") || tail.contains("<turn|>");
            if has_close { None } else { Some(pos) }
        })
        .min();
    let earliest = [tier1, tier2, thought]
        .into_iter()
        .flatten()
        .min();

    let max_opener_len = THOUGHT_BLOCK_OPENERS
        .iter()
        .map(|s| s.len())
        .chain(std::iter::once(TOOL_CALL_OPENER.len()))
        .max()
        .unwrap_or(0);
    let ceiling_by_tail = {
        let keep_back = max_opener_len.saturating_sub(1); // >= "call:".len() = 5
        accum.len().saturating_sub(keep_back)
    };
    let mut cand = match earliest {
        Some(p) => p.min(ceiling_by_tail),
        None => ceiling_by_tail,
    };
    if cand < emitted {
        return emitted;
    }
    // If `cand` falls INSIDE a closed thought block — opener arrived,
    // closer arrived, but `ceiling_by_tail` happens to cut between
    // them — push `cand` to the byte after the closer so the slice
    // `accum[..cand]` contains the entire block. Otherwise
    // `strip_tool_markup` sees an opener with no closer in the slice,
    // strips only the opener, and leaks the inner prose ("thought\n
    // <channel|>In Bern...") to the streaming client.
    //
    // Verified failure mode (real Telegram traffic, 2026-04-25):
    //   accum = "<|channel>thought\nIn Bern ist es aktuell 1나°C.<channel|>..."
    //   ceiling_by_tail cuts at byte ~35 → strip_tool_markup sees
    //   "<|channel>thought\nIn Bern..." → strips opener only, leaks
    //   "thought\nIn Bern..." to user.
    for &op in THOUGHT_BLOCK_OPENERS {
        let mut search_from = emitted;
        while let Some(rel) = accum[search_from..].find(op) {
            let opener_pos = search_from + rel;
            let after_opener = opener_pos + op.len();
            let tail = &accum[after_opener..];
            let close_at = ["<channel|>", "<turn|>"]
                .iter()
                .filter_map(|c| tail.find(c).map(|r| after_opener + r + c.len()))
                .min();
            match close_at {
                Some(closer_end) => {
                    // Block is closed in `accum`. If `cand` lands
                    // inside, extend to the byte AFTER the closer.
                    if cand >= opener_pos && cand < closer_end {
                        cand = closer_end;
                    }
                    // Continue scanning past this block — there may
                    // be additional thought blocks after it.
                    search_from = closer_end;
                }
                None => {
                    // Unclosed; the `earliest` upper bound already
                    // capped `cand` at the opener_pos. Stop scanning
                    // for this opener type.
                    break;
                }
            }
        }
    }
    while cand > emitted && !accum.is_char_boundary(cand) {
        cand -= 1;
    }
    cand
}

/// Return the byte offset of the earliest `call:` in `accum` at or after
/// `from` that is anchored at start-of-string or directly after
/// whitespace. Returns `None` if no anchored marker exists.
fn find_anchored_call(accum: &str, from: usize) -> Option<usize> {
    let bytes = accum.as_bytes();
    let mut i = from;
    while i + 5 <= bytes.len() {
        if &bytes[i..i + 5] == b"call:" {
            let anchored = i == 0 || (bytes[i - 1] as char).is_whitespace();
            if anchored {
                return Some(i);
            }
        }
        i += 1;
    }
    None
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
    _headers: axum::http::HeaderMap,
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
    // Cycle 34 P0 (codex bug #1): completions did not even parse `stop`.
    // Mirror chat-handler validation + post-decode truncation below.
    let stop_text = req.stop.as_ref().map(|s| extract_stop(s)).unwrap_or_default();
    validate_stops(&stop_text)?;
    // Cycle 37 P1 (codex audit): mirror the chat-handler fix — reject
    // stream+stop BEFORE tokenization so it's a zero-GPU-cost 400.
    if req.stream && !stop_text.is_empty() {
        return Err(ApiError::invalid_param(
            "`stop` is not yet supported with stream=true (would be \
             silently ignored). Use stream=false, or wait for \
             streaming-truncation support.",
            "stop",
            "stop_with_stream_unsupported",
        ));
    }

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
        // Cycle 37: stream+stop rejection lifted to pre-tokenize above.
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
            &stop_text,
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

#[allow(unused_assignments)] // initial `let mut finish/usage = ...` overwritten before read in loop body
async fn completion_collect(
    model_id: &str,
    tokenizer: &crate::tokenize::TokenizerHandle,
    mut events_rx: mpsc::Receiver<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    request_timeout: std::time::Duration,
    stop_text: &[String],
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
                // Cycle 34 P0 fix (codex bug #2): same fix as the chat
                // handler — channel close before Done is a worker fault,
                // not a successful empty completion.
                None => return Err(ApiError::Internal(
                    "worker channel closed before Done event".into(),
                )),
            },
        }
    }
    let mut text = tokenizer.decode(&token_ids)?;
    // Cycle 34 P0 (codex bug #1): post-decode stop-string truncation.
    if apply_stop_truncation(&mut text, stop_text) {
        finish = Some(FinishReason::Stop);
    }
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
                            // Cycle 36 P1 (codex audit): timeout was reported
                            // as FinishReason::Length, indistinguishable from
                            // hitting max_tokens. Now Cancelled — clients can
                            // tell wall-clock-timeout apart from natural stop.
                            ctx.state = S::Finish(FinishReason::Cancelled);
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
                        Err(e) => {
                            // Cycle 36 P1 (codex audit): completion SSE
                            // decoder errors were also silently Stop. Now
                            // Cancelled with log to match chat SSE.
                            tracing::error!(error = ?e,
                                "completion SSE decoder step failed — terminating stream");
                            ctx.state = S::Finish(FinishReason::Cancelled);
                            continue;
                        }
                    },
                    Some(GenerateEvent::Done { finish, .. }) => {
                        ctx.state = S::Finish(finish);
                        continue;
                    }
                    // Cycle 35 P1 fix (codex bug #2): completion SSE was
                    // mapping worker errors AND channel close to
                    // FinishReason::Stop — mirror the cycle 34 chat-SSE
                    // fix so CUDA faults are not silently clean here either.
                    Some(GenerateEvent::Error(msg)) => {
                        tracing::error!(error = %msg, "completion SSE worker error event — terminating stream");
                        ctx.state = S::Finish(FinishReason::Cancelled);
                        continue;
                    }
                    None => {
                        tracing::error!("completion SSE worker channel closed without Done — terminating stream");
                        ctx.state = S::Finish(FinishReason::Cancelled);
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
    // `tools` is threaded into the Gemma 4 chat template so the model
    // emits native `<|tool_call>call:NAME{...}<tool_call|>` blocks,
    // which `tool_parser` extracts back into OpenAI `tool_calls`.
    //
    // Cycle 35 P1 fix (codex bug #4): the runtime can express
    // "auto" / "none" / "required" via the chat template, but it
    // CANNOT enforce a specific {"type":"function","function":{"name":...}}
    // tool_choice — there's no logit-biasing/grammar-constrained
    // generation in this build. Previously we silently downgraded to
    // "auto" and just logged a one-shot warning. Now we reject the
    // unsupported form so callers don't get incorrect tool dispatch.
    // String forms still pass through.
    if let Some(choice) = &req.tool_choice {
        // Cycle 37 P1 (codex audit): "none" / "required" were silently
        // accepted but not enforced — `tools` was still rendered into
        // the chat template either way, so "none" could still produce
        // tool calls and "required" was indistinguishable from "auto".
        // Reject the unenforced forms (none / required) too, until the
        // template path can honour them properly. "auto" stays the
        // only honoured value.
        let s = choice.as_str();
        match s {
            Some("auto") => {}
            Some("none") => {
                return Err(ApiError::invalid_param(
                    "tool_choice=\"none\" is not enforced (`tools` would \
                     still be rendered into the prompt). Omit `tools` \
                     entirely to disable tool calls.",
                    "tool_choice",
                    "tool_choice_none_unenforced",
                ));
            }
            Some("required") => {
                return Err(ApiError::invalid_param(
                    "tool_choice=\"required\" is not enforced \
                     (no constrained-decoding path); the model still \
                     decides whether to emit a call. Use \"auto\" and \
                     accept a non-tool response.",
                    "tool_choice",
                    "tool_choice_required_unenforced",
                ));
            }
            Some(_) => {
                return Err(ApiError::invalid_param(
                    "tool_choice must be the string \"auto\" — \
                     specific-function and other forms are not yet \
                     supported (no constrained-decoding path).",
                    "tool_choice",
                    "tool_choice_unsupported_string",
                ));
            }
            None => {
                // Object form (specific function) was previously here.
                return Err(ApiError::invalid_param(
                    "tool_choice with a specific function name is not yet \
                     supported (no constrained-decoding path); use \"auto\".",
                    "tool_choice",
                    "tool_choice_specific_unsupported",
                ));
            }
        }
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

#[cfg(test)]
mod shape_assistant_message_tests {
    //! Coverage for `shape_assistant_message`, which turns the raw
    //! token-stream text (decoded WITH special tokens preserved) into the
    //! OpenAI `ChatAssistantMessage` + `finish_reason` pair.
    //!
    //! Inputs in every test below are **verbatim raw-decoded shapes** we
    //! saw Gemma 4 produce against this server — the comments name the
    //! failure mode that test locks down. Keep them realistic; adding a
    //! cleaned-up synthetic string here would mask the bugs these tests
    //! exist to prevent.
    use super::*;
    use crate::openai::types::{FinishReason, Role};
    use serde_json::Value;
    fn parse_args(msg: &ChatAssistantMessage) -> Value {
        let tc = msg.tool_calls.as_ref().expect("tool_calls present");
        serde_json::from_str(&tc[0].function.arguments).expect("arguments is valid JSON")
    }
    /// Happy path: model emits a single `<|tool_call>call:NAME{k:"v"}<tool_call|>`
    /// block with no surrounding prose. OpenAI response should carry one
    /// tool_call, no content, `finish_reason=tool_calls`.
    #[test]
    fn single_bare_tool_call_becomes_structured() {
        let raw = "<|tool_call>call:weather{city:<|\"|>Bern<|\"|>}<tool_call|>";
        let (msg, finish) = shape_assistant_message(raw.into(), None);
        assert_eq!(finish, Some(FinishReason::ToolCalls));
        assert!(matches!(msg.role, Role::Assistant));
        assert!(msg.content.is_none(), "content should be empty, got {:?}", msg.content);
        let tc = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "weather");
        assert_eq!(parse_args(&msg), serde_json::json!({"city": "Bern"}));
    }
    /// Gemma 4 routinely wraps its pre-answer in `<|channel>thought\n…<channel|>`.
    /// None of the reasoning prose (incl. hallucinated numbers) must leak into
    /// `content`. This is the bug that reached Telegram as literal
    /// `thought\n<thought\nDas Wetter ist bewölkt, 11°C` before we stripped it.
    #[test]
    fn thought_channel_does_not_leak_into_content() {
        let raw = "<|channel>thought\nMaybe rainy, ~11°C.<channel|>\
                   <|tool_call>call:weather{city:<|\"|>Bern<|\"|>}<tool_call|>";
        let (msg, finish) = shape_assistant_message(raw.into(), None);
        assert_eq!(finish, Some(FinishReason::ToolCalls));
        assert!(msg.content.is_none(), "thought block must not reach content, got {:?}", msg.content);
        assert_eq!(msg.tool_calls.as_ref().unwrap()[0].function.name, "weather");
    }
    /// The `<|tool_response>thought\n…<channel|>` block is another shape the
    /// model puts its internal draft in. Must be dropped wholesale.
    #[test]
    fn tool_response_channel_does_not_leak_into_content() {
        let raw = "<|tool_response>thought\ndraft answer goes here<channel|>\
                   <|tool_call>call:weather{city:<|\"|>Bern<|\"|>}<tool_call|>";
        let (msg, _) = shape_assistant_message(raw.into(), None);
        assert!(msg.content.is_none(), "got {:?}", msg.content);
    }
    /// The literal `<thought …<channel|>` fragment is what Gemma emits when
    /// it hallucinates a channel opener without the leading pipe. It's NOT a
    /// special token and survives skip-specials decoding — the parser still
    /// has to strip it.
    #[test]
    fn hallucinated_thought_fragment_is_stripped() {
        let raw = "<thought\nmaybe 14°C<channel|>\
                   <|tool_call>call:weather{city:<|\"|>Bern<|\"|>}<tool_call|>";
        let (msg, _) = shape_assistant_message(raw.into(), None);
        assert!(msg.content.is_none(), "got {:?}", msg.content);
    }
    /// After the tool round-trip the model usually replies in plain German
    /// prose terminated by `<turn|>`. Content must preserve the answer
    /// verbatim (including ö) and drop the trailing turn marker. This is
    /// the check that confirms the thought-strip does NOT over-reach.
    #[test]
    fn final_answer_after_tool_round_trip_keeps_umlauts() {
        let raw = "<|channel>thought\n<channel|>\
                   Das Wetter in Bern ist heute bewölkt mit einer Temperatur von 14°C.\
                   <turn|>";
        let (msg, finish) = shape_assistant_message(raw.into(), Some(FinishReason::Stop));
        assert_eq!(finish, Some(FinishReason::Stop));
        assert!(msg.tool_calls.is_none());
        assert_eq!(
            msg.content.as_deref(),
            Some("Das Wetter in Bern ist heute bewölkt mit einer Temperatur von 14°C."),
        );
    }
    /// Plain-text reply with zero markup — baseline sanity. Model's finish
    /// reason passes through unchanged; content is the text verbatim.
    #[test]
    fn plain_text_reply_is_pass_through() {
        let raw = "Paris ist die Hauptstadt von Frankreich.";
        let (msg, finish) = shape_assistant_message(raw.into(), Some(FinishReason::Stop));
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(msg.content.as_deref(), Some(raw));
        assert!(msg.tool_calls.is_none());
    }
    /// The tokenizer sometimes strips `<|tool_call>` / `<tool_call|>` markers
    /// (depends on vocab config) — the parser must still find the call in
    /// the bare `call:NAME{...}` residue. This is tier-2 of the parser;
    /// guarding it at the shape layer is cheap insurance.
    #[test]
    fn bare_call_tier2_still_produces_structured_call() {
        let raw = "call:weather{city:<|\"|>Bern<|\"|>}";
        let (msg, finish) = shape_assistant_message(raw.into(), None);
        assert_eq!(finish, Some(FinishReason::ToolCalls));
        let tc = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].function.name, "weather");
        assert_eq!(parse_args(&msg), serde_json::json!({"city": "Bern"}));
    }
    /// Model emits two sequential calls (rare but real — batch enumeration).
    /// Both must survive.
    #[test]
    fn multiple_tool_calls_survive() {
        let raw = "<|tool_call>call:a{x:<|\"|>1<|\"|>}<tool_call|>\
                   <|tool_call>call:b{y:<|\"|>2<|\"|>}<tool_call|>";
        let (msg, _) = shape_assistant_message(raw.into(), None);
        let tc = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 2);
        assert_eq!(tc[0].function.name, "a");
        assert_eq!(tc[1].function.name, "b");
    }
    /// Empty generation (model produced zero usable tokens) — should NOT
    /// crash and NOT invent a tool call.
    #[test]
    fn empty_generation_does_not_invent_tool_calls() {
        let (msg, finish) = shape_assistant_message(String::new(), Some(FinishReason::Stop));
        assert_eq!(finish, Some(FinishReason::Stop));
        assert!(msg.tool_calls.is_none());
        assert!(msg.content.is_none());
    }
    /// The stray-marker sweep must NOT eat `<turn|>` tokens that are
    /// *inside* user content (e.g. when the user asked about "A<turn|>B"
    /// — unlikely but possible). Conservative check: alphanumeric body only.
    #[test]
    fn stray_sweep_leaves_non_token_brackets_intact() {
        let raw = "Winkel < 90° hier.";
        let (msg, _) = shape_assistant_message(raw.into(), Some(FinishReason::Stop));
        assert_eq!(msg.content.as_deref(), Some("Winkel < 90° hier."));
    }
    /// Regression for the very first bug that kicked off this whole
    /// stack: zeroclaw saw `brain(action="web_fetch", name="...")` emitted
    /// as plain text. That shape is *not* a Gemma-4 tool call — it's the
    /// Python-kwargs-style a model improvises when it gets no
    /// native-format hint. Shape layer should NOT claim a tool call here;
    /// returning it as content keeps the downstream text-parser (zeroclaw's
    /// perl/func-call regex set) free to try. Catching this wrong would
    /// produce a tool_call with name `brain` and empty args, which is the
    /// exact failure that broke production.
    #[test]
    fn python_kwargs_style_is_not_misparsed_as_native_call() {
        let raw = "brain(action=\"web_fetch\", name=\"https://example.com\")";
        let (msg, finish) = shape_assistant_message(raw.into(), Some(FinishReason::Stop));
        assert!(msg.tool_calls.is_none(), "must not be misparsed, got {:?}", msg.tool_calls);
        assert_eq!(msg.content.as_deref(), Some(raw));
        assert_eq!(finish, Some(FinishReason::Stop));
    }
}
