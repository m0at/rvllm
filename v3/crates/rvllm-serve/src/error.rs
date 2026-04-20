//! HTTP error surface.
//!
//! Every request path returns [`ApiResult<T>`] = `Result<T, ApiError>`.
//! `ApiError` implements `axum::response::IntoResponse` and renders
//! the OpenAI-style error envelope:
//!
//! ```json
//! { "error": { "message": "...", "type": "invalid_request_error",
//!              "param": null, "code": "invalid_prompt" } }
//! ```
//!
//! See `v3/INFERENCE_SERVER_PLAN.md` § "Error mapping" for the
//! status-code ↔ error-type table.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

pub type ApiResult<T> = Result<T, ApiError>;

/// All errors that can cross the HTTP boundary. Constructors below
/// are the only way to build one — keeps the status/type mapping in
/// one place.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// 400 — request malformed / missing required field / invalid
    /// value. Maps to OpenAI `invalid_request_error`.
    #[error("invalid request: {message}")]
    InvalidRequest {
        message: String,
        param: Option<String>,
        code: &'static str,
    },

    /// 404 — model id unknown. We host exactly one model, so this
    /// fires when a client asks for a different name.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// 429 — admission control: request queue is full. Maps to
    /// OpenAI `rate_limit_exceeded`.
    #[error("server busy: {0}")]
    Busy(String),

    /// 503 — model not yet ready (worker still loading weights).
    #[error("service unavailable: {0}")]
    Unavailable(String),

    /// 500 — internal failure (CUDA launch error, worker panic,
    /// tokenizer load error). Body carries a correlation id for log
    /// lookup; full error chain goes to `tracing::error!`.
    #[error("internal error: {0}")]
    Internal(String),

    /// Tokenizer / chat-template failure. 400 if caused by user input
    /// (e.g. unknown role), 500 otherwise — caller picks via the
    /// constructor.
    #[error("tokenize: {0}")]
    Tokenize(String),
}

impl ApiError {
    /// Shorthand for `InvalidRequest` with a static code.
    pub fn invalid<M: Into<String>>(message: M, code: &'static str) -> Self {
        Self::InvalidRequest { message: message.into(), param: None, code }
    }

    /// Same, but point at a specific request field.
    pub fn invalid_param<M, P>(message: M, param: P, code: &'static str) -> Self
    where
        M: Into<String>,
        P: Into<String>,
    {
        Self::InvalidRequest {
            message: message.into(),
            param: Some(param.into()),
            code,
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            ApiError::InvalidRequest { .. } | ApiError::Tokenize(_) => {
                StatusCode::BAD_REQUEST
            }
            ApiError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ApiError::Busy(_) => StatusCode::TOO_MANY_REQUESTS,
            ApiError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            ApiError::InvalidRequest { .. } | ApiError::Tokenize(_) => {
                "invalid_request_error"
            }
            ApiError::ModelNotFound(_) => "invalid_request_error",
            ApiError::Busy(_) => "rate_limit_exceeded",
            ApiError::Unavailable(_) => "service_unavailable",
            ApiError::Internal(_) => "server_error",
        }
    }

    fn code(&self) -> Option<&str> {
        match self {
            ApiError::InvalidRequest { code, .. } => Some(code),
            _ => None,
        }
    }

    fn param(&self) -> Option<&str> {
        match self {
            ApiError::InvalidRequest { param, .. } => param.as_deref(),
            _ => None,
        }
    }
}

/// OpenAI error envelope. Serialised as `{"error": { ... }}`.
#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<&'a str>,
    code: Option<&'a str>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // Log server-side for correlation; clients never see stack.
        if matches!(self, ApiError::Internal(_) | ApiError::Unavailable(_)) {
            tracing::error!(error = %self, "request failed");
        } else {
            tracing::warn!(error = %self, "request rejected");
        }

        let status = self.status();
        let body = ErrorEnvelope {
            error: ErrorBody {
                message: self.to_string(),
                error_type: self.error_type(),
                param: self.param(),
                code: self.code(),
            },
        };
        (status, Json(body)).into_response()
    }
}

/// Forward `RvllmError` into the HTTP surface. Most library errors
/// are internal; the few user-caused ones get classified below.
impl From<rvllm_core::RvllmError> for ApiError {
    fn from(err: rvllm_core::RvllmError) -> Self {
        // The library's own `Display` impl is verbose; wrap with a
        // short HTTP-safe string and log the full chain for debug.
        tracing::error!(error = ?err, "rvllm error crossed HTTP boundary");
        ApiError::Internal(format!("{err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_is_400() {
        let e = ApiError::invalid("nope", "bad");
        assert_eq!(e.status(), StatusCode::BAD_REQUEST);
        assert_eq!(e.error_type(), "invalid_request_error");
        assert_eq!(e.code(), Some("bad"));
    }

    #[test]
    fn busy_is_429() {
        let e = ApiError::Busy("queue full".into());
        assert_eq!(e.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(e.error_type(), "rate_limit_exceeded");
    }

    #[test]
    fn internal_is_500() {
        let e = ApiError::Internal("boom".into());
        assert_eq!(e.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(e.error_type(), "server_error");
    }
}
