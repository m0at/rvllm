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
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;

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
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(crate::openai::models::list_models))
        .route("/v1/chat/completions", post(chat_completions_stub))
        .route("/v1/completions", post(completions_stub))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

/// Phase-4 stub — replace with the real handler once the worker +
/// tokenizer paths are wired end-to-end.
async fn chat_completions_stub() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "POST /v1/chat/completions not yet implemented (phase 4)",
                "type": "server_error",
                "code": "not_implemented"
            }
        })),
    )
}

async fn completions_stub() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "POST /v1/completions not yet implemented (phase 6)",
                "type": "server_error",
                "code": "not_implemented"
            }
        })),
    )
}
