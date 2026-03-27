//! Completion endpoint: POST /v1/completions

use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::StreamExt;
use tracing::info;

use crate::error::ApiError;
use crate::server::AppState;
use crate::types::request::CompletionRequest;
use crate::types::response::CompletionResponse;
use crate::types::streaming::{format_sse_data, CompletionStreamChunk, SSE_DONE};

/// POST /v1/completions -- text completion (streaming or non-streaming).
pub async fn create_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    req.validate()?;

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }

    let sampling_params = req.to_sampling_params();

    info!(
        model = %req.model,
        stream = req.stream,
        max_tokens = req.max_tokens,
        "completion request"
    );

    if req.stream {
        let stream_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model = state.model_name.clone();

        let (_request_id, output_stream) = state
            .engine
            .generate(req.prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let sse_stream = output_stream.map(move |output| {
            let mut events = String::new();
            for co in &output.outputs {
                let finish = co.finish_reason.map(|r| match r {
                    rvllm_core::prelude::FinishReason::Stop => "stop".to_string(),
                    rvllm_core::prelude::FinishReason::Length => "length".to_string(),
                    rvllm_core::prelude::FinishReason::Abort => "stop".to_string(),
                });
                let chunk = CompletionStreamChunk::new(
                    &stream_id,
                    &model,
                    co.index,
                    &co.text,
                    finish,
                );
                events.push_str(&format_sse_data(&chunk));
            }
            if output.finished {
                events.push_str(SSE_DONE);
            }
            Ok::<_, std::convert::Infallible>(events)
        });

        let body = axum::body::Body::from_stream(sse_stream);
        Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(body)
            .unwrap()
            .into_response())
    } else {
        // Non-streaming: collect all outputs from the stream until finished.
        let (_request_id, mut output_stream) = state
            .engine
            .generate(req.prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let mut last_output = None;
        while let Some(output) = output_stream.next().await {
            if output.finished {
                last_output = Some(output);
                break;
            }
            last_output = Some(output);
        }

        let output = last_output.ok_or_else(|| {
            ApiError::Internal("engine produced no output".into())
        })?;

        let resp = CompletionResponse::from_request_output(&output, &state.model_name);
        Ok(Json(resp).into_response())
    }
}
