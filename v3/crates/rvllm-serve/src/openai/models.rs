//! `GET /v1/models` — lists the single model this server hosts.

use axum::{extract::State, Json};
use serde::Serialize;

use crate::openai::types::unix_now_secs;
use crate::router::AppState;

#[derive(Serialize)]
pub struct ModelsList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

/// Handler. Returns exactly one model — the one we loaded.
pub async fn list_models(State(state): State<AppState>) -> Json<ModelsList> {
    Json(ModelsList {
        object: "list",
        data: vec![ModelInfo {
            id: state.config.model_id.clone(),
            object: "model",
            created: state.started_at,
            owned_by: "rvllm",
        }],
    })
}

/// Convenience: check the caller's requested model against what we
/// serve. Used by chat + completions handlers before they bother the
/// worker.
pub fn ensure_model_matches(
    state: &AppState,
    requested: &str,
) -> Result<(), crate::error::ApiError> {
    if requested == state.config.model_id {
        Ok(())
    } else {
        Err(crate::error::ApiError::ModelNotFound(requested.to_string()))
    }
}
