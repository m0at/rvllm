//! `rvllm-server` binary — thin entrypoint that wires CLI args into
//! [`rvllm_serve::ServerConfig`], starts the worker, builds the axum
//! router, and blocks on the HTTP listener.

// main.rs does need a handful of unwrap-on-startup paths (CLI parse,
// listener bind). The library crate itself stays `deny(unwrap_used)`.
#![allow(clippy::unwrap_used)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use rvllm_serve::{build_router, AppState, ServerConfig};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "rvllm-server — OpenAI-compatible inference server"
)]
struct Cli {
    /// Path to the HF model directory (must contain config.json,
    /// tokenizer.json, safetensors shards).
    #[arg(long, env = "RVLLM_MODEL_DIR")]
    model_dir: PathBuf,

    /// Model name advertised on /v1/models. Defaults to the model_dir
    /// basename.
    #[arg(long, env = "RVLLM_MODEL_ID")]
    model_id: Option<String>,

    /// Bind address.
    #[arg(long, env = "RVLLM_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Max queued + in-flight requests before 429.
    #[arg(long, env = "RVLLM_QUEUE_DEPTH", default_value_t = 8)]
    max_queue_depth: usize,

    /// Per-request hard cap on `max_tokens`.
    #[arg(long, env = "RVLLM_MAX_TOKENS_CAP", default_value_t = 4096)]
    max_new_tokens_cap: u32,

    /// Per-request timeout, seconds.
    #[arg(long, env = "RVLLM_REQUEST_TIMEOUT_SECS", default_value_t = 300)]
    request_timeout_secs: u64,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow_compat::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,rvllm_serve=debug")),
        )
        .init();

    let cli = Cli::parse();

    let model_id = cli.model_id.unwrap_or_else(|| {
        cli.model_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "rvllm".into())
    });

    let config = ServerConfig {
        bind: cli.bind,
        model_dir: cli.model_dir.clone(),
        model_id,
        max_queue_depth: cli.max_queue_depth,
        max_new_tokens_cap: cli.max_new_tokens_cap,
        request_timeout: Duration::from_secs(cli.request_timeout_secs),
        ..ServerConfig::default()
    };
    config.validate().map_err(|e| anyhow_compat::err(format!("config: {e}")))?;
    let config = Arc::new(config);

    // Tokenizer + worker.
    let tokenizer = rvllm_serve::tokenize::TokenizerHandle::load(&cli.model_dir)
        .map_err(|e| anyhow_compat::err(format!("tokenizer: {e:?}")))?;

    // Phase 1 wires the mock worker. Phase 2+ will gate on feature
    // `cuda` to spawn the real Gemma4Bringup-backed worker.
    let (worker, _join) = rvllm_serve::worker::spawn_mock_worker(config.max_queue_depth);

    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let state = AppState { config: config.clone(), tokenizer, worker, started_at };
    let router = build_router(state);

    tracing::info!(bind = %config.bind, model = %config.model_id, "rvllm-server listening");
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .map_err(|e| anyhow_compat::err(format!("bind {}: {e}", config.bind)))?;

    // Graceful shutdown on Ctrl-C.
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| anyhow_compat::err(format!("axum serve: {e}")))?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .ok()
            .and_then(|mut s| futures::executor::block_on(async move { s.recv().await }));
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("ctrl-c received, shutting down"),
        _ = term => tracing::info!("SIGTERM received, shutting down"),
    }
}

/// Tiny inline `anyhow`-like error wrapper. Avoids adding `anyhow`
/// as a dep just for main.rs.
mod anyhow_compat {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
    pub fn err<S: Into<String>>(s: S) -> Box<dyn std::error::Error + Send + Sync> {
        s.into().into()
    }
}
