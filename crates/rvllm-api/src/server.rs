//! HTTP server setup, AppState, router construction, and graceful shutdown.
//!
//! The server requires a CUDA-capable GPU and uses AsyncGpuLLMEngine for real
//! inference. It fails fast instead of falling back to a mock executor.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cfg(not(feature = "cuda"))]
use tokenizers::models::bpe::BPE;
#[cfg(not(feature = "cuda"))]
use tokenizers::pre_tokenizers::whitespace::Whitespace;
#[cfg(not(feature = "cuda"))]
use tokenizers::Tokenizer as HfTokenizer;

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use rvllm_config::EngineConfig;
use rvllm_core::prelude::RequestId;
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

    async fn generate_with_mode(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
        emit_intermediate: bool,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        let _ = emit_intermediate;
        self.generate(prompt, params).await
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

    async fn generate_with_mode(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
        emit_intermediate: bool,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        self.generate_with_mode(prompt, params, emit_intermediate).await
    }
}

#[async_trait::async_trait]
impl InferenceEngine for rvllm_engine::AsyncLLMEngine {
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
    /// Stored conversation state for Responses API conversation-based turns.
    pub conversation_store: crate::routes::responses::SharedConversationStore,
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
            conversation_store: Arc::new(RwLock::new(HashMap::new())),
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

#[cfg(not(feature = "cuda"))]
struct MockScheduler {
    groups: Vec<rvllm_sequence::SequenceGroup>,
}

#[cfg(not(feature = "cuda"))]
impl MockScheduler {
    fn new() -> Self {
        Self { groups: Vec::new() }
    }
}

#[cfg(not(feature = "cuda"))]
impl rvllm_engine::Scheduler for MockScheduler {
    fn add_seq_group(&mut self, seq_group: rvllm_sequence::SequenceGroup) {
        self.groups.push(seq_group);
    }

    fn abort_seq_group(&mut self, request_id: &RequestId) {
        self.groups.retain(|g| g.request_id != *request_id);
    }

    fn schedule(&mut self) -> rvllm_engine::SchedulerOutputs {
        let groups = self.groups.clone();
        self.groups.retain(|g| !g.is_finished());
        let num_tokens = groups.iter().flat_map(|g| g.get_seqs()).map(|s| s.num_new_tokens().max(1)).sum();
        rvllm_engine::SchedulerOutputs { scheduled_seq_groups: groups, num_batched_tokens: num_tokens, preempted: false }
    }

    fn has_unfinished_seqs(&self) -> bool { !self.groups.is_empty() }
    fn get_num_unfinished_seq_groups(&self) -> usize { self.groups.len() }
}

#[cfg(not(feature = "cuda"))]
struct MockExecutor { calls: usize }

#[cfg(not(feature = "cuda"))]
impl MockExecutor { fn new() -> Self { Self { calls: 0 } } }

#[cfg(not(feature = "cuda"))]
impl rvllm_engine::Executor for MockExecutor {
    fn execute_model(&mut self, input: rvllm_engine::ExecutorInput) -> rvllm_core::prelude::Result<Vec<rvllm_engine::SamplerOutput>> {
        self.calls += 1;
        let mut outputs = Vec::new();
        for meta in &input.seq_group_metadata {
            for &seq_id in meta.seq_data.keys() {
                outputs.push(rvllm_engine::SamplerOutput { seq_id, token_id: if self.calls >= 8 { 0 } else { 1 }, logprob: -0.5, top_logprobs: None });
            }
        }
        Ok(outputs)
    }
}

#[cfg(not(feature = "cuda"))]
fn build_mock_tokenizer() -> rvllm_tokenizer::Tokenizer {
    let mut vocab = std::collections::HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert(" ".to_string(), 2);
    vocab.insert("!".to_string(), 3);
    vocab.insert("[UNK]".to_string(), 4);
    let bpe = BPE::builder().vocab_and_merges(vocab, vec![]).unk_token("[UNK]".to_string()).build().unwrap();
    let mut hf = HfTokenizer::new(bpe);
    hf.with_pre_tokenizer(Some(Whitespace {}));
    let path = std::env::temp_dir().join(format!("rvllm-mock-tokenizer-{}.json", std::process::id()));
    hf.save(&path, false).unwrap();
    let tok = rvllm_tokenizer::Tokenizer::from_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    tok
}

pub async fn serve(config: EngineConfig) -> rvllm_core::prelude::Result<()> {
    let model_name = config.model.model_path.clone();
    let tokenizer_path = config
        .model
        .tokenizer_path
        .clone()
        .unwrap_or_else(|| config.model.model_path.clone());

    info!(model = %model_name, "initializing engine");

    let has_cuda = cuda_gpu_available();
    let mock_mode = model_name == "mock-model" && !has_cuda;

    let tokenizer = if mock_mode {
        #[cfg(not(feature = "cuda"))]
        {
            build_mock_tokenizer()
        }
        #[cfg(feature = "cuda")]
        {
            Tokenizer::from_pretrained(&tokenizer_path)?
        }
    } else {
        Tokenizer::from_pretrained(&tokenizer_path)?
    };

    let engine: Arc<dyn InferenceEngine> = if has_cuda {
        info!("GPU detected, creating AsyncGpuLLMEngine for real inference");
        create_gpu_engine(config).await?
    } else {
        #[cfg(not(feature = "cuda"))]
        {
            if !mock_mode {
                return Err(rvllm_core::prelude::LLMError::GpuError(
                    "no CUDA GPU detected; non-CUDA server mode currently supports mock-model only".into(),
                ));
            }
            info!("starting mock AsyncLLMEngine for non-CUDA smoke/integration flows");
            Arc::new(rvllm_engine::AsyncLLMEngine::new(
                config,
                Box::new(MockExecutor::new()),
                Box::new(MockScheduler::new()),
                build_mock_tokenizer(),
            )?)
        }
        #[cfg(feature = "cuda")]
        {
            return Err(rvllm_core::prelude::LLMError::GpuError(
                "no CUDA GPU detected; refusing to start non-GPU server backend".into(),
            ));
        }
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
