//! HTTP server setup, AppState, router construction, and graceful shutdown.
//!
//! When compiled with the `cuda` feature and a CUDA-capable GPU is detected,
//! the server uses the GPU-accelerated AsyncGpuLLMEngine. Otherwise it falls
//! back to the mock executor via AsyncLLMEngine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use rvllm_config::EngineConfig;
use rvllm_core::prelude::RequestId;
use rvllm_engine::AsyncLLMEngine;
use rvllm_engine::{ExecutorAdapter, ExecutorConfig};
use rvllm_tokenizer::Tokenizer;

use crate::routes;

// ------------------------------------------------------------------
// Engine trait object for unified API
// ------------------------------------------------------------------

/// Trait abstracting over AsyncLLMEngine and AsyncGpuLLMEngine so the
/// AppState can hold either one.
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn generate(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )>;

    fn supports_beam_search(&self) -> bool;
}

#[async_trait::async_trait]
impl InferenceEngine for AsyncLLMEngine {
    async fn generate(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        self.generate(prompt, params).await
    }

    fn supports_beam_search(&self) -> bool {
        true
    }
}

#[cfg(feature = "cuda")]
#[async_trait::async_trait]
impl InferenceEngine for rvllm_engine::AsyncGpuLLMEngine {
    async fn generate(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        self.generate(prompt, params).await
    }

    fn supports_beam_search(&self) -> bool {
        true
    }
}

/// Shared application state available to all route handlers.
pub struct AppState {
    pub engine: Arc<dyn InferenceEngine>,
    pub model_name: String,
    pub tokenizer: Arc<RwLock<Tokenizer>>,
    /// Batch job store (None if batch API is not enabled).
    pub batch_store: Option<crate::routes::batch::SharedBatchStore>,
    /// Stored response objects for Responses API follow-up turns and retrieval.
    pub response_store: crate::routes::responses::SharedResponseStore,
    next_id: AtomicU64,
}

impl AppState {
    pub fn new(engine: Arc<dyn InferenceEngine>, model_name: String, tokenizer: Tokenizer) -> Self {
        Self {
            engine,
            model_name,
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            batch_store: Some(crate::routes::batch::create_batch_store(None)),
            response_store: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/v1/completions",
            post(routes::completions::create_completion),
        )
        .route(
            "/v1/chat/completions",
            post(routes::chat::create_chat_completion),
        )
        .route("/v1/responses", post(routes::responses::create_response))
        .route(
            "/v1/responses/:response_id",
            get(routes::responses::get_response),
        )
        .route(
            "/v1/responses/:response_id/input_items",
            get(routes::responses::list_response_input_items),
        )
        .route(
            "/v1/embeddings",
            post(routes::embeddings::create_embeddings),
        )
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/batches", post(routes::batch::create_batch))
        .route("/v1/batches/:batch_id", get(routes::batch::get_batch))
        .route(
            "/v1/batches/:batch_id/output",
            get(routes::batch::get_batch_output),
        )
        .route(
            "/v1/batches/:batch_id/cancel",
            post(routes::batch::cancel_batch),
        )
        .route(
            "/v1/chat/completions/tools",
            post(routes::tools::create_chat_completion_with_tools),
        )
        .route("/health", get(routes::health::health_check))
        .route("/metrics", get(metrics_placeholder))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn metrics_placeholder() -> &'static str {
    "# vllm-rs metrics endpoint\n"
}

fn cuda_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        let devices = rvllm_gpu::prelude::list_devices();
        if devices.is_empty() {
            info!("cuda feature enabled but no CUDA devices found");
            return false;
        }
        for dev in &devices {
            info!(
                id = dev.id, name = %dev.name,
                memory_gb = dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                "CUDA device available"
            );
        }
        true
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub async fn serve(config: EngineConfig) -> rvllm_core::prelude::Result<()> {
    let model_name = config.model.model_path.clone();
    let tokenizer_path = config
        .model
        .tokenizer_path
        .clone()
        .unwrap_or_else(|| config.model.model_path.clone());

    info!(model = %model_name, "initializing engine");

    let tokenizer = Tokenizer::from_pretrained(&tokenizer_path)?;

    let engine: Arc<dyn InferenceEngine> = if cuda_gpu_available() {
        info!("GPU detected, creating AsyncGpuLLMEngine for real inference");
        create_gpu_engine(config).await?
    } else {
        info!("no GPU detected, creating mock engine");
        Arc::new(create_engine(config, &tokenizer_path)?)
    };

    let state = Arc::new(AppState::new(engine, model_name, tokenizer));
    let app = build_router(state);

    let host = std::env::var("VLLM_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("VLLM_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8000);
    let addr = format!("{host}:{port}");
    info!(addr = %addr, "starting API server");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(rvllm_core::prelude::LLMError::IoError)?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(rvllm_core::prelude::LLMError::IoError)?;

    info!("server shut down gracefully");
    Ok(())
}

/// Create the real GPU engine using AsyncGpuLLMEngine.
#[cfg(feature = "cuda")]
async fn create_gpu_engine(
    config: EngineConfig,
) -> rvllm_core::prelude::Result<Arc<dyn InferenceEngine>> {
    let engine = rvllm_engine::AsyncGpuLLMEngine::new(config).await?;
    Ok(Arc::new(engine))
}

#[cfg(not(feature = "cuda"))]
async fn create_gpu_engine(
    _config: EngineConfig,
) -> rvllm_core::prelude::Result<Arc<dyn InferenceEngine>> {
    Err(rvllm_core::prelude::LLMError::GpuError(
        "CUDA not available".into(),
    ))
}

fn create_engine(
    config: EngineConfig,
    tokenizer_path: &str,
) -> rvllm_core::prelude::Result<AsyncLLMEngine> {
    let tokenizer = Tokenizer::from_pretrained(tokenizer_path)?;

    let executor_config = ExecutorConfig {
        num_gpus: config.parallel.tensor_parallel_size,
        model_name: config.model.model_path.clone(),
        block_size: config.cache.block_size,
        gpu_memory_utilization: config.cache.gpu_memory_utilization,
        tensor_parallel_size: config.parallel.tensor_parallel_size,
        pipeline_parallel_size: config.parallel.pipeline_parallel_size,
    };
    let rt = tokio::runtime::Handle::current();
    let executor = ExecutorAdapter::from_config(executor_config, rt).map_err(|e| {
        rvllm_core::prelude::LLMError::ConfigError(format!("failed to create executor: {}", e))
    })?;

    let scheduler = Box::new(PlaceholderScheduler::new());

    let engine = AsyncLLMEngine::new(config, Box::new(executor), scheduler, tokenizer)?;
    Ok(engine)
}

struct PlaceholderScheduler {
    groups: Vec<rvllm_sequence::SequenceGroup>,
}

impl PlaceholderScheduler {
    fn new() -> Self {
        Self { groups: Vec::new() }
    }
}

impl rvllm_engine::Scheduler for PlaceholderScheduler {
    fn add_seq_group(&mut self, seq_group: rvllm_sequence::SequenceGroup) {
        self.groups.push(seq_group);
    }

    fn abort_seq_group(&mut self, request_id: &RequestId) {
        self.groups.retain(|g| g.request_id != *request_id);
    }

    fn schedule(&mut self) -> rvllm_engine::SchedulerOutputs {
        let groups = self.groups.clone();
        self.groups.retain(|g| !g.is_finished());
        let num_tokens = groups
            .iter()
            .flat_map(|g| g.get_seqs())
            .map(|s| s.num_new_tokens().max(1))
            .sum();
        rvllm_engine::SchedulerOutputs {
            scheduled_seq_groups: groups,
            num_batched_tokens: num_tokens,
            preempted: false,
        }
    }

    fn has_unfinished_seqs(&self) -> bool {
        !self.groups.is_empty()
    }
    fn get_num_unfinished_seq_groups(&self) -> usize {
        self.groups.len()
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { info!("received Ctrl+C, shutting down"); }
        _ = terminate => { info!("received SIGTERM, shutting down"); }
    }
}
