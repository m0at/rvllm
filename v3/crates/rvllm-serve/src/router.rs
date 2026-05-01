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
        .route("/v1/responses", post(responses_unsupported))
        .layer(trace)
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
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
