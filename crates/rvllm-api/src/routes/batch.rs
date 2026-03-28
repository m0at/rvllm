//! OpenAI Batch API endpoints.
//!
//! - `POST /v1/batches` -- submit a batch of requests as JSONL
//! - `GET /v1/batches/{id}` -- check batch status
//! - `GET /v1/batches/{id}/output` -- retrieve batch results
//! - `POST /v1/batches/{id}/cancel` -- cancel a running batch

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tracing::{error, info};

use crate::error::ApiError;
use crate::server::AppState;
use crate::types::request::{ChatCompletionRequest, CompletionRequest};
use crate::types::response::{ChatCompletionResponse, CompletionResponse};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Status of a batch job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    /// Batch has been accepted and is validating input.
    Validating,
    /// Batch is queued but not yet processing.
    InProgress,
    /// Batch completed successfully.
    Completed,
    /// Batch failed.
    Failed,
    /// Batch was cancelled by the user.
    Cancelled,
}

/// A single line in the JSONL input file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestLine {
    /// Caller-provided identifier for this request within the batch.
    pub custom_id: String,
    /// HTTP method (always POST for inference).
    pub method: String,
    /// Target URL path, e.g. "/v1/completions" or "/v1/chat/completions".
    pub url: String,
    /// The request body -- a raw JSON value parsed later based on `url`.
    pub body: serde_json::Value,
}

/// A single line in the JSONL output file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponseLine {
    /// Matches `custom_id` from the request line.
    pub custom_id: String,
    /// HTTP status code of the individual request.
    pub status_code: u16,
    /// The response body (completion or chat completion response).
    pub body: serde_json::Value,
    /// Error detail if the individual request failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<BatchResponseError>,
}

/// Error detail for a failed request within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponseError {
    pub code: String,
    pub message: String,
}

/// Metadata for a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchObject {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub input_file_id: Option<String>,
    pub status: BatchStatus,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled_at: Option<u64>,
    pub request_counts: BatchRequestCounts,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_file_id: Option<String>,
}

/// Counts of requests in various states.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
}

/// Request body for `POST /v1/batches`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateBatchRequest {
    /// JSONL content: each line is a `BatchRequestLine`.
    pub input: String,
    /// The endpoint all requests target (e.g. "/v1/completions").
    pub endpoint: String,
    /// Completion window hint (e.g. "24h"). Accepted but not enforced.
    #[serde(default)]
    pub completion_window: Option<String>,
}

// ---------------------------------------------------------------------------
// Shared batch store
// ---------------------------------------------------------------------------

/// In-memory store for batch jobs, wrapped in a RwLock for concurrent access.
/// Results are also persisted to disk as JSONL when the batch completes.
pub struct BatchStore {
    batches: HashMap<String, BatchObject>,
    results: HashMap<String, Vec<BatchResponseLine>>,
    /// Directory where output JSONL files are written.
    output_dir: PathBuf,
    /// Cancellation tokens keyed by batch id.
    cancel_tokens: HashMap<String, tokio::sync::watch::Sender<bool>>,
}

impl BatchStore {
    /// Create a new store. `output_dir` is created if it does not exist.
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            batches: HashMap::new(),
            results: HashMap::new(),
            output_dir,
            cancel_tokens: HashMap::new(),
        }
    }
}

/// Extension type added to [`AppState`] for batch support.
pub type SharedBatchStore = Arc<RwLock<BatchStore>>;

/// Create a [`SharedBatchStore`] with a configurable output directory.
pub fn create_batch_store(output_dir: Option<PathBuf>) -> SharedBatchStore {
    let dir = output_dir.unwrap_or_else(|| {
        std::env::var("VLLM_BATCH_OUTPUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/vllm-rs-batches"))
    });
    Arc::new(RwLock::new(BatchStore::new(dir)))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/batches -- submit a batch of requests as JSONL.
pub async fn create_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateBatchRequest>,
) -> Result<Json<BatchObject>, ApiError> {
    let store = state
        .batch_store
        .as_ref()
        .ok_or_else(|| ApiError::Internal("batch API not enabled".into()))?;

    // Parse JSONL input lines.
    let lines: Vec<BatchRequestLine> = req
        .input
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str::<BatchRequestLine>(l).map_err(|e| {
                ApiError::InvalidRequest(format!("invalid JSONL at line {}: {}", i + 1, e))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if lines.is_empty() {
        return Err(ApiError::InvalidRequest("input JSONL is empty".into()));
    }

    let batch_id = format!("batch_{}", uuid::Uuid::new_v4());
    let now = now_secs();

    let batch = BatchObject {
        id: batch_id.clone(),
        object: "batch".to_string(),
        endpoint: req.endpoint.clone(),
        input_file_id: None,
        status: BatchStatus::InProgress,
        created_at: now,
        completed_at: None,
        failed_at: None,
        cancelled_at: None,
        request_counts: BatchRequestCounts {
            total: lines.len(),
            completed: 0,
            failed: 0,
        },
        output_file_id: None,
        error_file_id: None,
    };

    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

    {
        let mut s = store.write().await;
        s.batches.insert(batch_id.clone(), batch.clone());
        s.results.insert(batch_id.clone(), Vec::new());
        s.cancel_tokens.insert(batch_id.clone(), cancel_tx);
    }

    // Spawn background processing.
    let engine = state.engine.clone();
    let model_name = state.model_name.clone();
    let store_clone = Arc::clone(store);
    let bid = batch_id.clone();

    tokio::spawn(async move {
        process_batch(engine, model_name, store_clone, bid, lines, cancel_rx).await;
    });

    info!(batch_id = %batch_id, total = batch.request_counts.total, "batch created");
    Ok(Json(batch))
}

/// GET /v1/batches/{batch_id} -- check batch status.
pub async fn get_batch(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<Json<BatchObject>, ApiError> {
    let store = state
        .batch_store
        .as_ref()
        .ok_or_else(|| ApiError::Internal("batch API not enabled".into()))?;

    let s = store.read().await;
    let batch = s
        .batches
        .get(&batch_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("batch '{}' not found", batch_id)))?;

    Ok(Json(batch))
}

/// GET /v1/batches/{batch_id}/output -- retrieve results as JSONL.
pub async fn get_batch_output(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<axum::response::Response, ApiError> {
    let store = state
        .batch_store
        .as_ref()
        .ok_or_else(|| ApiError::Internal("batch API not enabled".into()))?;

    let s = store.read().await;
    let batch = s
        .batches
        .get(&batch_id)
        .ok_or_else(|| ApiError::NotFound(format!("batch '{}' not found", batch_id)))?;

    if batch.status != BatchStatus::Completed
        && batch.status != BatchStatus::Failed
        && batch.status != BatchStatus::Cancelled
    {
        return Err(ApiError::InvalidRequest(format!(
            "batch '{}' is still {:?}, results not yet available",
            batch_id, batch.status
        )));
    }

    let results = s.results.get(&batch_id).cloned().unwrap_or_default();
    drop(s);

    let mut jsonl = String::new();
    for line in &results {
        jsonl.push_str(
            &serde_json::to_string(line)
                .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.into()),
        );
        jsonl.push('\n');
    }

    Ok(axum::response::Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "application/jsonl")
        .body(axum::body::Body::from(jsonl))
        .unwrap())
}

/// POST /v1/batches/{batch_id}/cancel -- cancel a running batch.
pub async fn cancel_batch(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<Json<BatchObject>, ApiError> {
    let store = state
        .batch_store
        .as_ref()
        .ok_or_else(|| ApiError::Internal("batch API not enabled".into()))?;

    let mut s = store.write().await;
    let batch = s
        .batches
        .get(&batch_id)
        .ok_or_else(|| ApiError::NotFound(format!("batch '{}' not found", batch_id)))?;

    if batch.status != BatchStatus::InProgress && batch.status != BatchStatus::Validating {
        return Err(ApiError::InvalidRequest(format!(
            "batch '{}' cannot be cancelled (status: {:?})",
            batch_id, batch.status
        )));
    }

    // Signal cancellation to the background task.
    if let Some(tx) = s.cancel_tokens.get(&batch_id) {
        let _ = tx.send(true);
    }

    let batch = s.batches.get_mut(&batch_id).unwrap();
    batch.status = BatchStatus::Cancelled;
    batch.cancelled_at = Some(now_secs());
    let result = batch.clone();

    info!(batch_id = %batch_id, "batch cancelled");
    Ok(Json(result))
}

// ---------------------------------------------------------------------------
// Background processing
// ---------------------------------------------------------------------------

/// Process all requests in a batch, updating the store as each completes.
async fn process_batch(
    engine: Arc<dyn crate::server::InferenceEngine>,
    model_name: String,
    store: SharedBatchStore,
    batch_id: String,
    lines: Vec<BatchRequestLine>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
) {
    let total = lines.len();
    let mut completed = 0usize;
    let mut failed = 0usize;

    for line in &lines {
        // Check cancellation before each request.
        if *cancel_rx.borrow() {
            info!(batch_id = %batch_id, "batch processing cancelled");
            break;
        }

        let result = process_single_request(&engine, &model_name, line).await;

        let response_line = match result {
            Ok(body) => BatchResponseLine {
                custom_id: line.custom_id.clone(),
                status_code: 200,
                body,
                error: None,
            },
            Err(e) => {
                failed += 1;
                BatchResponseLine {
                    custom_id: line.custom_id.clone(),
                    status_code: 400,
                    body: serde_json::Value::Null,
                    error: Some(BatchResponseError {
                        code: "request_failed".into(),
                        message: e.to_string(),
                    }),
                }
            }
        };

        if response_line.error.is_none() {
            completed += 1;
        }

        // Update store.
        {
            let mut s = store.write().await;
            if let Some(results) = s.results.get_mut(&batch_id) {
                results.push(response_line);
            }
            if let Some(batch) = s.batches.get_mut(&batch_id) {
                batch.request_counts.completed = completed;
                batch.request_counts.failed = failed;
            }
        }
    }

    // Finalize.
    let mut s = store.write().await;
    if let Some(batch) = s.batches.get_mut(&batch_id) {
        if batch.status == BatchStatus::Cancelled {
            // Already cancelled, leave as is.
        } else if failed == total {
            batch.status = BatchStatus::Failed;
            batch.failed_at = Some(now_secs());
        } else {
            batch.status = BatchStatus::Completed;
            batch.completed_at = Some(now_secs());
        }
    }

    // Persist results to disk.
    let output_dir = s.output_dir.clone();
    let results = s.results.get(&batch_id).cloned().unwrap_or_default();

    // Update output file path.
    let output_path = output_dir.join(format!("{}.jsonl", batch_id));
    if let Some(batch) = s.batches.get_mut(&batch_id) {
        batch.output_file_id = Some(output_path.display().to_string());
    }

    // Remove cancellation token -- batch is done.
    s.cancel_tokens.remove(&batch_id);
    drop(s);

    // Write to disk (best-effort).
    if let Err(e) = persist_results(&output_path, &results).await {
        error!(batch_id = %batch_id, error = %e, "failed to persist batch results");
    } else {
        info!(
            batch_id = %batch_id,
            completed = completed,
            failed = failed,
            total = total,
            path = %output_path.display(),
            "batch results persisted"
        );
    }
}

/// Process a single request line through the inference engine.
async fn process_single_request(
    engine: &Arc<dyn crate::server::InferenceEngine>,
    model_name: &str,
    line: &BatchRequestLine,
) -> Result<serde_json::Value, ApiError> {
    match line.url.as_str() {
        "/v1/completions" => {
            let req: CompletionRequest =
                serde_json::from_value(line.body.clone()).map_err(|e| {
                    ApiError::InvalidRequest(format!(
                        "invalid completion request for '{}': {}",
                        line.custom_id, e
                    ))
                })?;
            req.validate()?;

            if req.model != model_name {
                return Err(ApiError::ModelNotFound(format!(
                    "model '{}' not found, available: {}",
                    req.model, model_name
                )));
            }
            if req.use_beam_search && !engine.supports_beam_search() {
                return Err(ApiError::InvalidRequest(
                    "beam search is not supported by the active inference backend".into(),
                ));
            }

            let sampling_params = req.to_sampling_params();
            let (_rid, mut stream) = engine
                .generate(req.prompt, sampling_params)
                .await
                .map_err(ApiError::from)?;

            let mut last = None;
            while let Some(output) = stream.next().await {
                if output.finished {
                    last = Some(output);
                    break;
                }
                last = Some(output);
            }
            let output =
                last.ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

            let resp = CompletionResponse::from_request_output(&output, model_name);
            serde_json::to_value(&resp).map_err(|e| ApiError::Internal(e.to_string()))
        }
        "/v1/chat/completions" => {
            let req: ChatCompletionRequest =
                serde_json::from_value(line.body.clone()).map_err(|e| {
                    ApiError::InvalidRequest(format!(
                        "invalid chat completion request for '{}': {}",
                        line.custom_id, e
                    ))
                })?;
            req.validate()?;

            if req.model != model_name {
                return Err(ApiError::ModelNotFound(format!(
                    "model '{}' not found, available: {}",
                    req.model, model_name
                )));
            }
            if req.use_beam_search && !engine.supports_beam_search() {
                return Err(ApiError::InvalidRequest(
                    "beam search is not supported by the active inference backend".into(),
                ));
            }

            let sampling_params = req.to_sampling_params();
            // Build prompt from messages -- for batch we just concatenate content
            // since we don't have tokenizer access here. Real usage would apply
            // chat template.
            let prompt = req
                .messages
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n");

            let (_rid, mut stream) = engine
                .generate(prompt, sampling_params)
                .await
                .map_err(ApiError::from)?;

            let mut last = None;
            while let Some(output) = stream.next().await {
                if output.finished {
                    last = Some(output);
                    break;
                }
                last = Some(output);
            }
            let output =
                last.ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

            let resp = ChatCompletionResponse::from_request_output(&output, model_name);
            serde_json::to_value(&resp).map_err(|e| ApiError::Internal(e.to_string()))
        }
        other => Err(ApiError::InvalidRequest(format!(
            "unsupported endpoint '{}' for custom_id '{}'",
            other, line.custom_id
        ))),
    }
}

/// Write batch results to a JSONL file on disk.
async fn persist_results(
    path: &std::path::Path,
    results: &[BatchResponseLine],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut content = String::new();
    for line in results {
        if let Ok(json) = serde_json::to_string(line) {
            content.push_str(&json);
            content.push('\n');
        }
    }
    tokio::fs::write(path, content.as_bytes()).await
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_status_serde() {
        let s = BatchStatus::InProgress;
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#""in_progress""#);
        let back: BatchStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, BatchStatus::InProgress);
    }

    #[test]
    fn batch_request_line_parse() {
        let line = r#"{"custom_id":"req-1","method":"POST","url":"/v1/completions","body":{"model":"m","prompt":"hello","max_tokens":10}}"#;
        let parsed: BatchRequestLine = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.custom_id, "req-1");
        assert_eq!(parsed.url, "/v1/completions");
    }

    #[test]
    fn batch_response_line_serde() {
        let line = BatchResponseLine {
            custom_id: "req-1".into(),
            status_code: 200,
            body: serde_json::json!({"text": "hello"}),
            error: None,
        };
        let json = serde_json::to_string(&line).unwrap();
        assert!(json.contains("req-1"));
        let back: BatchResponseLine = serde_json::from_str(&json).unwrap();
        assert_eq!(back.custom_id, "req-1");
        assert_eq!(back.status_code, 200);
    }

    #[test]
    fn batch_response_line_with_error() {
        let line = BatchResponseLine {
            custom_id: "req-2".into(),
            status_code: 400,
            body: serde_json::Value::Null,
            error: Some(BatchResponseError {
                code: "invalid_request".into(),
                message: "bad prompt".into(),
            }),
        };
        let json = serde_json::to_string(&line).unwrap();
        assert!(json.contains("bad prompt"));
    }

    #[test]
    fn batch_object_serde() {
        let batch = BatchObject {
            id: "batch_123".into(),
            object: "batch".into(),
            endpoint: "/v1/completions".into(),
            input_file_id: None,
            status: BatchStatus::Completed,
            created_at: 1000,
            completed_at: Some(2000),
            failed_at: None,
            cancelled_at: None,
            request_counts: BatchRequestCounts {
                total: 5,
                completed: 4,
                failed: 1,
            },
            output_file_id: Some("/tmp/batch_123.jsonl".into()),
            error_file_id: None,
        };
        let json = serde_json::to_string(&batch).unwrap();
        let back: BatchObject = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "batch_123");
        assert_eq!(back.status, BatchStatus::Completed);
        assert_eq!(back.request_counts.total, 5);
    }

    #[test]
    fn create_batch_request_serde() {
        let req = CreateBatchRequest {
            input: r#"{"custom_id":"r1","method":"POST","url":"/v1/completions","body":{}}"#.into(),
            endpoint: "/v1/completions".into(),
            completion_window: Some("24h".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CreateBatchRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.endpoint, "/v1/completions");
    }

    #[test]
    fn batch_request_counts_default() {
        let c = BatchRequestCounts::default();
        assert_eq!(c.total, 0);
        assert_eq!(c.completed, 0);
        assert_eq!(c.failed, 0);
    }

    #[test]
    fn batch_status_all_variants() {
        for (variant, expected) in [
            (BatchStatus::Validating, r#""validating""#),
            (BatchStatus::InProgress, r#""in_progress""#),
            (BatchStatus::Completed, r#""completed""#),
            (BatchStatus::Failed, r#""failed""#),
            (BatchStatus::Cancelled, r#""cancelled""#),
        ] {
            assert_eq!(serde_json::to_string(&variant).unwrap(), expected);
        }
    }

    #[test]
    fn batch_store_new() {
        let store = BatchStore::new(PathBuf::from("/tmp/test-batches"));
        assert!(store.batches.is_empty());
        assert!(store.results.is_empty());
        assert_eq!(store.output_dir, PathBuf::from("/tmp/test-batches"));
    }
}
