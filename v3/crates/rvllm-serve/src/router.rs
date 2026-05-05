//! axum Router construction + shared AppState.
//!
//! Endpoints in v1 (phase 4-6):
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/chat/completions
//!   POST /v1/completions
//!
//! Phase 1 ships route stubs returning 501 Not Implemented so
//! `cargo check` has everything wired, integration tests can hit
//! the paths, and handler bodies land in later phases.

use std::sync::Arc;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, DefaultOnResponse, TraceLayer};

use crate::config::ServerConfig;
use crate::tokenize::TokenizerHandle;
use crate::worker::WorkerHandle;

/// Shared state every handler receives via `State<AppState>`.
///
/// Clone is cheap (`Arc` inside).
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub tokenizer: TokenizerHandle,
    pub worker: WorkerHandle,
    /// Unix secs when the server started — used as the `created`
    /// timestamp on the `/v1/models` payload.
    pub started_at: u64,
    /// Which vision-tower architecture is loaded. Detected from the
    /// model dir's `config.json` at startup, NOT from the operator-
    /// supplied `--model-id` (which might be a public alias). The
    /// vision-token predictor and chat-template image-pad token
    /// dispatch off this. Codex review #2 (round 4) caught the
    /// model-id heuristic as a real correctness risk.
    pub vision_arch: VisionArch,
}

/// Which vision tower the worker has loaded. Set once at startup
/// from `config.json` inspection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionArch {
    /// Qwen 3.6 35B-A3B (Qwen3VL-style ViT, full-head rotary, Qwen
    /// PatchMerger). `<|image_pad|>` = 248056, predictor =
    /// `predict_qwen_num_tokens`.
    Qwen36,
    /// Gemma 4 31B (SigLIP-style ViT, multidimensional rotary,
    /// `Gemma4MultimodalEmbedder`). `<|image|>` = 258880, predictor
    /// = `predict_gemma_num_tokens`.
    Gemma4,
}

/// Build the top-level router with all endpoints attached.
///
/// Wraps every request in a `TraceLayer` so `tracing::info!` spans
/// emit one line per request with method + path + status + latency,
/// plus a sub-span inside each handler carrying the generated
/// `request_id`. Failures go through `DefaultOnFailure` at `ERROR`
/// so a missing model → 404 emits a single warn line and CUDA
/// launch failures → 500 emit error with the `request_id` attached.
pub fn build_router(state: AppState) -> Router {
    let trace = TraceLayer::new_for_http()
        .make_span_with(
            DefaultMakeSpan::new().level(tracing::Level::INFO).include_headers(false),
        )
        .on_response(DefaultOnResponse::new().level(tracing::Level::INFO))
        .on_failure(DefaultOnFailure::new().level(tracing::Level::ERROR));

    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(crate::openai::models::list_models))
        .route(
            "/v1/chat/completions",
            post(crate::openai::handlers::chat_completions),
        )
        .route(
            "/v1/completions",
            post(crate::openai::handlers::completions),
        )
        // OpenAI's 2025 "Responses API". Different request/response
        // shape from `/v1/chat/completions` (conversation-state,
        // different streaming-event taxonomy, server-side reasoning
        // tokens). We do not implement the full surface yet, but
        // routing it explicitly to a 501-with-guidance handler is
        // strictly better than the default 404 — the OpenAI Python
        // SDK's `client.responses.create(...)` users get a clear
        // server-side message pointing them at `/v1/chat/completions`
        // instead of a generic "endpoint not found".
        //
        // The 501 is intentional, not a bug. Test suites that probe
        // OpenAI compat will see `/v1/responses` as not-implemented;
        // responses-API support is its own project (~500 LOC) and
        // out of scope for this server's current feature set.
        // Mark Responses-API tests as skipped or expected-501.
        .route("/v1/responses", post(responses_unsupported))
        // Hard cap on request-body size before axum buffers the JSON
        // into RAM. Without this a single client can post a 1 GiB
        // data: URI and burn memory before any vision-admission
        // limits kick in. 80 MiB matches the per-request multimodal
        // ceiling (RVLLM_VISION_MAX_TOTAL_BYTES default 64 MiB +
        // generous JSON overhead for base64-encoding + chat history)
        // and is independently overridable via
        // `RVLLM_HTTP_BODY_LIMIT_BYTES`. Returns 413 Payload Too Large.
        .layer({
            let limit_bytes: usize = std::env::var("RVLLM_HTTP_BODY_LIMIT_BYTES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(80 * 1024 * 1024);
            DefaultBodyLimit::max(limit_bytes)
        })
        .layer(trace)
        .with_state(state)
}

async fn health(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    // Liveness: report 503 with `status:"unavailable"` if the worker
    // thread has dropped its end of the submit channel (CUDA crash,
    // panic, clean shutdown). The previous always-200 `{"status":"ok"}`
    // stayed green even after a worker exit while every generate
    // request would 500 — turning health checks into a false signal
    // for orchestrators / load balancers.
    if state.worker.is_alive() {
        (
            axum::http::StatusCode::OK,
            Json(json!({ "status": "ok" })),
        )
    } else {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unavailable",
                "reason": "worker submit channel closed; the inference \
                          thread has exited and generate requests will fail",
            })),
        )
    }
}

async fn responses_unsupported() -> (axum::http::StatusCode, Json<serde_json::Value>) {
    // OpenAI-shaped error envelope — same fields the rest of the
    // server uses so SDKs that surface `error.message` show the
    // same path on both endpoints.
    let body = json!({
        "error": {
            "message":
                "The Responses API (`/v1/responses`) is not implemented yet on \
                 this server. Use `/v1/chat/completions` instead — it covers \
                 the same generation surface and is fully OpenAI-compatible \
                 for chat, tools, and streaming.",
            "type":  "invalid_request_error",
            "code":  "responses_api_unsupported",
            "param": serde_json::Value::Null,
        }
    });
    (axum::http::StatusCode::NOT_IMPLEMENTED, Json(body))
}
