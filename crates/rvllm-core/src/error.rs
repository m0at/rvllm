//! Error hierarchy for vllm-rs.

use thiserror::Error;

/// Top-level error type covering all vllm-rs failure modes.
#[derive(Debug, Error)]
pub enum LLMError {
    /// Invalid or missing configuration.
    #[error("config error: {0}")]
    ConfigError(String),

    /// GPU device or driver failure.
    #[error("gpu error: {0}")]
    GpuError(String),

    /// Out of memory or allocation failure.
    #[error("memory error: {0}")]
    MemoryError(String),

    /// Tokenizer load or encode/decode failure.
    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    /// Model load or forward-pass failure.
    #[error("model error: {0}")]
    ModelError(String),

    /// Scheduler policy violation.
    #[error("scheduler error: {0}")]
    SchedulerError(String),

    /// Sampling or logit-processing failure.
    #[error("sampling error: {0}")]
    SamplingError(String),

    /// Underlying I/O error.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization / deserialization error.
    #[error("serialization error: {0}")]
    SerializationError(String),
}

/// Convenience alias used throughout vllm-rs.
pub type Result<T> = std::result::Result<T, LLMError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = LLMError::ConfigError("bad value".into());
        assert_eq!(e.to_string(), "config error: bad value");
    }

    #[test]
    fn io_error_converts() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let e: LLMError = io.into();
        assert!(matches!(e, LLMError::IoError(_)));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LLMError>();
    }
}
