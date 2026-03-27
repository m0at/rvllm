#![forbid(unsafe_code)]
//! Telemetry, metrics, and observability for vllm-rs.

pub mod config;
pub mod handler;
pub mod init;
pub mod metrics;
pub mod recorder;

pub use config::{LogFormat, TelemetryConfig};
pub use handler::metrics_handler;
pub use init::{init_telemetry, TelemetryGuard};
pub use recorder::MetricsRecorder;

// Span helper macros for instrumenting critical paths.

#[macro_export]
macro_rules! schedule_span {
    () => {
        tracing::info_span!("schedule")
    };
}

#[macro_export]
macro_rules! forward_span {
    () => {
        tracing::info_span!("forward")
    };
}

#[macro_export]
macro_rules! sample_span {
    () => {
        tracing::info_span!("sample")
    };
}
