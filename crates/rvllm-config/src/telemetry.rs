//! Telemetry / observability configuration.

use serde::{Deserialize, Serialize};

/// Configuration for metrics, tracing, and logging.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelemetryConfig {
    /// Whether telemetry collection is enabled.
    pub enabled: bool,
    /// Port to expose Prometheus metrics on (None = disabled).
    pub prometheus_port: Option<u16>,
    /// OTLP gRPC endpoint for distributed traces (None = disabled).
    pub otlp_endpoint: Option<String>,
    /// Minimum log level (e.g. "info", "debug", "warn").
    pub log_level: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prometheus_port: None,
            otlp_endpoint: None,
            log_level: "info".into(),
        }
    }
}

impl TelemetryConfig {
    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> TelemetryConfigBuilder {
        TelemetryConfigBuilder::default()
    }
}

/// Builder for [`TelemetryConfig`].
#[derive(Debug, Default)]
pub struct TelemetryConfigBuilder(TelemetryConfig);

impl TelemetryConfigBuilder {
    /// Enable or disable telemetry.
    pub fn enabled(mut self, v: bool) -> Self {
        self.0.enabled = v;
        self
    }

    /// Set Prometheus port.
    pub fn prometheus_port(mut self, v: u16) -> Self {
        self.0.prometheus_port = Some(v);
        self
    }

    /// Set OTLP endpoint.
    pub fn otlp_endpoint(mut self, v: impl Into<String>) -> Self {
        self.0.otlp_endpoint = Some(v.into());
        self
    }

    /// Set log level.
    pub fn log_level(mut self, v: impl Into<String>) -> Self {
        self.0.log_level = v.into();
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> TelemetryConfig {
        self.0
    }
}
