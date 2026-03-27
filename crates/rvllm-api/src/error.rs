//! OpenAI-compatible error responses and API error type.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

/// OpenAI-compatible error envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiErrorResponse {
    pub error: ApiErrorDetail,
}

/// Inner detail of an OpenAI-compatible error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

/// API-layer error type.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("engine error: {0}")]
    EngineError(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("not found: {0}")]
    NotFound(String),
}

impl From<rvllm_core::prelude::LLMError> for ApiError {
    fn from(e: rvllm_core::prelude::LLMError) -> Self {
        ApiError::EngineError(e.to_string())
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        ApiError::InvalidRequest(e.to_string())
    }
}

impl ApiError {
    fn status_code(&self) -> StatusCode {
        match self {
            ApiError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ApiError::EngineError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
        }
    }

    fn error_type(&self) -> &str {
        match self {
            ApiError::InvalidRequest(_) => "invalid_request_error",
            ApiError::ModelNotFound(_) => "invalid_request_error",
            ApiError::EngineError(_) => "server_error",
            ApiError::Internal(_) => "server_error",
            ApiError::NotFound(_) => "invalid_request_error",
        }
    }

    fn code(&self) -> &str {
        match self {
            ApiError::InvalidRequest(_) => "invalid_request",
            ApiError::ModelNotFound(_) => "model_not_found",
            ApiError::EngineError(_) => "engine_error",
            ApiError::Internal(_) => "internal_error",
            ApiError::NotFound(_) => "not_found",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ApiErrorResponse {
            error: ApiErrorDetail {
                message: self.to_string(),
                error_type: self.error_type().to_string(),
                code: self.code().to_string(),
            },
        };
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_response_serialization() {
        let resp = ApiErrorResponse {
            error: ApiErrorDetail {
                message: "bad param".into(),
                error_type: "invalid_request_error".into(),
                code: "invalid_request".into(),
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("bad param"));
        assert!(json.contains("invalid_request_error"));

        let back: ApiErrorResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn error_status_codes() {
        assert_eq!(
            ApiError::InvalidRequest("x".into()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ApiError::ModelNotFound("x".into()).status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiError::EngineError("x".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            ApiError::Internal("x".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn error_types_match_openai() {
        assert_eq!(
            ApiError::InvalidRequest("".into()).error_type(),
            "invalid_request_error"
        );
        assert_eq!(ApiError::EngineError("".into()).error_type(), "server_error");
    }

    #[test]
    fn from_llm_error() {
        let llm = rvllm_core::prelude::LLMError::ConfigError("oops".into());
        let api: ApiError = llm.into();
        assert!(matches!(api, ApiError::EngineError(_)));
    }

    #[test]
    fn from_serde_error() {
        let e = serde_json::from_str::<serde_json::Value>("{{invalid").unwrap_err();
        let api: ApiError = e.into();
        assert!(matches!(api, ApiError::InvalidRequest(_)));
    }
}
