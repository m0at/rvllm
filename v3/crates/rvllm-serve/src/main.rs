//! `rvllm-server` binary — thin entrypoint that wires CLI args into
//! [`rvllm_serve::ServerConfig`], starts the worker, builds the axum
//! router, and blocks on the HTTP listener.
//!
//! The worker backing the HTTP loop is chosen at compile time:
//!   * default build (no features): mock worker (no CUDA), emits
//!     canned tokens. Useful for HTTP/tokenizer/CI dev.
//!   * `--features cuda` / `--features gb10`: real worker backed by
//!     `Gemma4Bringup::run_generate` on a dedicated OS thread.

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

    /// Directory with per-arch kernel PTX + manifest.json. Required
    /// under `--features cuda`; ignored with the default mock worker.
    #[arg(long, env = "RVLLM_KERNELS_DIR")]
    kernels_dir: Option<PathBuf>,

    /// CUTLASS SM90 `.so`. Optional on sm_121 (never opened there).
    #[arg(long, env = "RVLLM_CUTLASS_SO")]
    cutlass_so: Option<PathBuf>,

    /// FA3 SM90 `.so`. Optional on sm_121 (never opened there).
    #[arg(long, env = "RVLLM_FA3_SO")]
    fa3_so: Option<PathBuf>,

    /// CUTLASS autotune policy JSON. Optional on sm_121 (a minimal
    /// placeholder is generated in `--kernels-dir` when missing).
    #[arg(long, env = "RVLLM_POLICY")]
    policy_json: Option<PathBuf>,

    /// Arena size in GiB for the CUDA worker. Gemma 4 31B fp8 needs
    /// >= 35 GiB. Ignored with the mock worker.
    #[arg(long, env = "RVLLM_ARENA_GB", default_value_t = 40)]
    arena_gb: u64,

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

    /// Graceful shutdown drain window, seconds. After the shutdown
    /// signal this is the max time we wait for in-flight requests
    /// before forcing process exit.
    #[arg(long, env = "RVLLM_SHUTDOWN_DRAIN_SECS", default_value_t = 30)]
    shutdown_drain_secs: u64,
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

    let model_id = cli.model_id.clone().unwrap_or_else(|| {
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
        shutdown_drain_timeout: Duration::from_secs(cli.shutdown_drain_secs),
        ..ServerConfig::default()
    };
    config.validate().map_err(|e| anyhow_compat::err(format!("config: {e}")))?;
    let config = Arc::new(config);

    let tokenizer = rvllm_serve::tokenize::TokenizerHandle::load(&cli.model_dir)
        .map_err(|e| anyhow_compat::err(format!("tokenizer: {e:?}")))?;

    let (worker, _join) = spawn_worker(&cli, config.max_queue_depth).await?;

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

    let drain_timeout = config.shutdown_drain_timeout;
    let serve = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal());

    tokio::select! {
        res = serve => {
            res.map_err(|e| anyhow_compat::err(format!("axum serve: {e}")))?;
        }
        _ = forced_drain_watchdog(drain_timeout) => {
            tracing::warn!(
                drain_timeout_secs = drain_timeout.as_secs(),
                "drain timeout exceeded — forcing shutdown",
            );
        }
    }

    Ok(())
}

async fn forced_drain_watchdog(drain_timeout: std::time::Duration) {
    shutdown_signal().await;
    tokio::time::sleep(drain_timeout).await;
}

#[cfg(feature = "cuda")]
async fn spawn_worker(
    cli: &Cli,
    queue_depth: usize,
) -> anyhow_compat::Result<(
    rvllm_serve::WorkerHandle,
    std::thread::JoinHandle<()>,
)> {
    use rvllm_serve::cuda_worker;

    let kernels_dir = cli
        .kernels_dir
        .clone()
        .ok_or_else(|| anyhow_compat::err("--kernels-dir required under --features cuda"))?;
    let paths = cuda_worker::resolve_paths(
        cli.model_dir.clone(),
        kernels_dir,
        cli.cutlass_so.clone(),
        cli.fa3_so.clone(),
        cli.policy_json.clone(),
    )
    .map_err(|e| anyhow_compat::err(format!("paths: {e:?}")))?;

    tracing::info!(
        arena_gb = cli.arena_gb,
        "starting cuda worker — Gemma4Bringup::load takes ~20 s for Gemma 4 31B",
    );
    let cfg = cuda_worker::CudaWorkerConfig {
        paths,
        arena_bytes: (cli.arena_gb as usize) * 1024 * 1024 * 1024,
        queue_depth,
    };
    cuda_worker::spawn_cuda_worker(cfg)
        .await
        .map_err(|e| anyhow_compat::err(format!("cuda worker: {e:?}")))
}

#[cfg(not(feature = "cuda"))]
async fn spawn_worker(
    _cli: &Cli,
    queue_depth: usize,
) -> anyhow_compat::Result<(
    rvllm_serve::WorkerHandle,
    std::thread::JoinHandle<()>,
)> {
    tracing::warn!(
        "built without --features cuda — using mock worker (canned tokens, no real inference)",
    );
    Ok(rvllm_serve::worker::spawn_mock_worker(queue_depth))
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.ok(); };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("ctrl-c received, shutting down"),
        _ = term => tracing::info!("SIGTERM received, shutting down"),
    }
}

mod anyhow_compat {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
    pub fn err<S: Into<String>>(s: S) -> Box<dyn std::error::Error + Send + Sync> {
        s.into().into()
    }
}
