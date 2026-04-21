use thiserror::Error;

#[derive(Debug, Error)]
pub enum RlmError {
    #[error("language model client is not configured")]
    MissingClient,
    #[error("environment is not configured")]
    MissingEnvironment,
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("client error: {0}")]
    Client(String),
    #[error("environment error: {0}")]
    Environment(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, RlmError>;
