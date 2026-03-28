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

    if req.use_beam_search && !state.engine.supports_beam_search() {
        return Err(ApiError::InvalidRequest(
            "beam search is not supported by the active inference backend".into(),
        ));
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
                let chunk =
                    CompletionStreamChunk::new(&stream_id, &model, co.index, &co.text, finish);
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

        let output =
            last_output.ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

        let resp = CompletionResponse::from_request_output(&output, &state.model_name);
        Ok(Json(resp).into_response())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum_test::TestServer;

    use super::*;
    use crate::test_support::{make_finished_output, make_test_tokenizer, RecordingEngine};

    #[tokio::test]
    async fn route_accepts_beam_search_completion_request() {
        let engine = RecordingEngine::new(make_finished_output(&["beam result"], true), true);
        let state = Arc::new(crate::server::AppState::new(
            engine.clone(),
            "m".to_string(),
            make_test_tokenizer(),
        ));
        let server = TestServer::new(crate::server::build_router(state)).unwrap();

        let response = server
            .post("/v1/completions")
            .json(&CompletionRequest {
                model: "m".into(),
                prompt: "hello".into(),
                max_tokens: 16,
                temperature: 0.0,
                top_p: 1.0,
                n: 2,
                stream: false,
                stop: None,
                logprobs: None,
                echo: false,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                user: None,
                seed: None,
                best_of: Some(3),
                use_beam_search: true,
                length_penalty: 0.5,
                early_stopping: true,
            })
            .await;

        response.assert_status_ok();
        assert_eq!(
            response.json::<serde_json::Value>()["choices"][0]["text"],
            "beam result"
        );

        let calls = engine.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].prompt, "hello");
        assert_eq!(calls[0].params.best_of, 3);
        assert!(calls[0].params.use_beam_search);
        assert_eq!(calls[0].params.length_penalty, 0.5);
        assert!(calls[0].params.early_stopping);
    }

    #[tokio::test]
    async fn route_keeps_best_of_n_distinct_from_beam_search() {
        let engine = RecordingEngine::new(make_finished_output(&["first", "second"], true), true);
        let state = Arc::new(crate::server::AppState::new(
            engine.clone(),
            "m".to_string(),
            make_test_tokenizer(),
        ));
        let server = TestServer::new(crate::server::build_router(state)).unwrap();

        let response = server
            .post("/v1/completions")
            .json(&CompletionRequest {
                model: "m".into(),
                prompt: "hello".into(),
                max_tokens: 16,
                temperature: 0.8,
                top_p: 0.9,
                n: 2,
                stream: false,
                stop: None,
                logprobs: None,
                echo: false,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                user: None,
                seed: None,
                best_of: None,
                use_beam_search: false,
                length_penalty: 1.0,
                early_stopping: false,
            })
            .await;

        response.assert_status_ok();
        assert_eq!(
            response.json::<serde_json::Value>()["choices"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let calls = engine.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].params.best_of, 2);
        assert!(!calls[0].params.use_beam_search);
    }
}
