//! Server configuration. Populated from CLI flags + env vars in
//! `main.rs`; plain data beyond that. Kept in a separate module so
//! integration tests can construct configs directly without going
//! through `clap`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Top-level server configuration.
///
/// All fields are public so `main.rs` and tests can populate them
/// directly. Invariants are checked in [`ServerConfig::validate`];
/// callers must call it before handing the config to the router.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// HTTP bind address. Default `127.0.0.1:8080`.
    pub bind: SocketAddr,
    /// Directory containing the HF model artefacts (config.json,
    /// tokenizer.json, safetensors shards). Required.
    pub model_dir: PathBuf,
    /// Model name advertised on `/v1/models` and echoed in response
    /// bodies. Defaults to the directory's last component.
    pub model_id: String,
    /// Max number of in-flight + queued generate requests. Full =
    /// admission returns 429. The worker pulls one request out of the
    /// queue and processes it on its dedicated thread; the channel
    /// buffer + admission permits are both sized to `max_queue_depth`,
    /// so depth=1 is strictly serial (1 permit + 1 channel slot,
    /// dequeued one at a time). Validate() only rejects 0.
    pub max_queue_depth: usize,
    /// Hard upper bound on `max_tokens` a request may ask for. Prevents
    /// a single client from pinning the worker.
    pub max_new_tokens_cap: u32,
    /// Per-request hard wall-clock cap. Worker aborts past this.
    pub request_timeout: Duration,
    /// SSE keep-alive interval. Proxies drop idle TCP.
    pub sse_keepalive: Duration,
    /// After the shutdown signal, wait at most this long for
    /// in-flight requests to finish before forcing the server down.
    pub shutdown_drain_timeout: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
            model_dir: PathBuf::new(),
            model_id: String::from("rvllm"),
            max_queue_depth: 8,
            max_new_tokens_cap: 4096,
            request_timeout: Duration::from_secs(300),
            sse_keepalive: Duration::from_secs(15),
            shutdown_drain_timeout: Duration::from_secs(30),
        }
    }
}

impl ServerConfig {
    /// Check invariants after population. Called once at startup.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_dir.as_os_str().is_empty() {
            return Err(ConfigError::MissingModelDir);
        }
        if !self.model_dir.is_dir() {
            return Err(ConfigError::ModelDirMissing(self.model_dir.clone()));
        }
        if self.max_queue_depth == 0 {
            return Err(ConfigError::InvalidQueueDepth);
        }
        // Both the cuda_worker and the mock spawn use
        // `mpsc::channel(queue_depth.max(1))` and the same number of
        // admission permits. With depth=1 that's exactly:
        //   1 admission permit + 1 channel slot + serial dequeue
        // = strictly serial, which is what the doc says it is. The
        // earlier ≥2 floor was a leftover from before the
        // admission/channel arithmetic was harmonised.
        if self.max_new_tokens_cap == 0 {
            return Err(ConfigError::InvalidMaxTokens);
        }
        if self.model_id.is_empty() {
            return Err(ConfigError::InvalidModelId);
        }
        // `request_timeout=0` was accepted before — clap parses any
        // u64 from `--request-timeout-secs` /
        // `RVLLM_REQUEST_TIMEOUT_SECS`. The handler uses it to build
        // `tokio::time::Instant::now() + request_timeout` as the
        // per-request deadline, so a 0-duration deadline rejects
        // every preprocess step (or hits an immediate timeout
        // post-submit) and the server is unusable. Refuse at
        // startup with a clear error instead of staying up and
        // 4xx-ing every request.
        if self.request_timeout.is_zero() {
            return Err(ConfigError::InvalidRequestTimeout);
        }
        Ok(())
    }
}

/// Config validation errors. Distinct from [`crate::error::ApiError`]
/// because these fire at startup, not in the request path.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("--model-dir is required")]
    MissingModelDir,
    #[error("model dir does not exist or is not a directory: {0}")]
    ModelDirMissing(PathBuf),
    #[error("--max-queue-depth must be > 0")]
    InvalidQueueDepth,
    #[error("--max-new-tokens-cap must be > 0")]
    InvalidMaxTokens,
    #[error("--model-id must be non-empty")]
    InvalidModelId,
    #[error("--request-timeout-secs must be > 0 (zero deadlines reject every request)")]
    InvalidRequestTimeout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_sane_values() {
        let c = ServerConfig::default();
        assert_eq!(c.max_queue_depth, 8);
        assert!(c.request_timeout >= Duration::from_secs(60));
    }

    #[test]
    fn validate_rejects_empty_model_dir() {
        let c = ServerConfig::default();
        assert!(matches!(c.validate(), Err(ConfigError::MissingModelDir)));
    }

    #[test]
    fn validate_accepts_queue_depth_one() {
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here,
            model_id: "test".into(),
            max_queue_depth: 1,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(60),
            ..ServerConfig::default()
        };
        assert!(c.validate().is_ok(), "depth=1 should be allowed (strictly serial)");
    }

    #[test]
    fn validate_rejects_zero_request_timeout() {
        // Use std::env::current_dir() as a guaranteed-existing dir so
        // the model-dir check passes; we only want to exercise the
        // request_timeout branch.
        let here = std::env::current_dir().expect("cwd");
        let c = ServerConfig {
            model_dir: here,
            model_id: "test".into(),
            max_queue_depth: 2,
            max_new_tokens_cap: 32,
            request_timeout: Duration::from_secs(0),
            ..ServerConfig::default()
        };
        assert!(matches!(c.validate(), Err(ConfigError::InvalidRequestTimeout)));
    }
}
