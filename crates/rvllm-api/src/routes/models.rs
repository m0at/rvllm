//! Model listing endpoint.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::server::AppState;
use crate::types::response::{ModelListResponse, ModelObject};

/// GET /v1/models -- list available models.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "vllm-rs".to_string(),
        }],
    })
}
