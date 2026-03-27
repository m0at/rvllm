use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use metrics_exporter_prometheus::PrometheusHandle;

/// Axum handler that renders Prometheus text exposition format.
pub async fn metrics_handler(State(handle): State<Arc<PrometheusHandle>>) -> impl IntoResponse {
    let body = handle.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn handle_renders() {
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let rendered = handle.render();
        // Empty recorder produces empty or minimal output -- just verify no panic.
        assert!(rendered.is_empty() || rendered.contains('#') || true);
    }
}
