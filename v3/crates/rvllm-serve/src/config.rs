//! Server configuration. Populated from CLI flags + env vars in
//! `main.rs`; plain data beyond that. Kept in a separate module so
//! integration tests can construct configs directly without going
//! through `clap`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Which model family the worker should bring up.
///
/// `Auto` (default) defers to `crate::family::resolve_model_family`,
/// which inspects the model dir's `config.json`. Explicit values take
/// precedence and require the dir to actually match — mismatch is a
/// startup error rather than silent fall-through. The CLI flag
/// `--model-family` and env var `RVLLM_MODEL_FAMILY` map onto this
/// enum directly (kebab-case lowercase).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ModelFamily {
    #[default]
    Auto,
    Qwen36,
    Gemma4,
    Mistral35,
}

impl ModelFamily {
    /// Parse the CLI / env-var spelling. Accepts the canonical
    /// kebab-case ids plus a couple of obvious aliases. Unknown values
    /// produce a typed error so the operator gets a clear startup
    /// rejection instead of a silent default.
    pub fn parse(s: &str) -> Result<Self, ModelFamilyParseError> {
        let norm = s.trim().to_ascii_lowercase();
        match norm.as_str() {
            "auto" | "" => Ok(ModelFamily::Auto),
            "qwen36" | "qwen-36" | "qwen3.6" | "qwen-3.6" => Ok(ModelFamily::Qwen36),
            "gemma4" | "gemma-4" => Ok(ModelFamily::Gemma4),
            "mistral35" | "mistral-35" | "mistral3.5" | "mistral-3.5" => {
                Ok(ModelFamily::Mistral35)
            }
            other => Err(ModelFamilyParseError(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ModelFamily::Auto => "auto",
            ModelFamily::Qwen36 => "qwen36",
            ModelFamily::Gemma4 => "gemma4",
            ModelFamily::Mistral35 => "mistral35",
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("--model-family must be one of auto|qwen36|gemma4|mistral35 (got: {0:?})")]
pub struct ModelFamilyParseError(pub String);

impl std::str::FromStr for ModelFamily {
    type Err = ModelFamilyParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

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
    /// Operator-supplied model family. `Auto` defers to
    /// config-marker detection. Anything else asserts the model
    /// matches and fails on mismatch.
    pub model_family: ModelFamily,
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
            model_family: ModelFamily::Auto,
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
