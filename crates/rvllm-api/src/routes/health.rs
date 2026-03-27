//! Health check endpoint.

use axum::http::StatusCode;
use axum::response::IntoResponse;

/// GET /health -- simple liveness check.
pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_returns_ok() {
        let resp = health_check().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
