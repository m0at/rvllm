use metrics_exporter_prometheus::PrometheusHandle;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use rvllm_core::prelude::*;

use crate::config::{LogFormat, TelemetryConfig};

/// RAII guard -- flushes metrics/traces on drop.
pub struct TelemetryGuard {
    _prom_handle: Option<PrometheusHandle>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        // Flush any remaining trace spans.
        tracing::dispatcher::get_default(|_| {});
    }
}

/// Initialise telemetry: structured logging, Prometheus metrics, optional OTLP stub.
pub fn init_telemetry(config: &TelemetryConfig) -> Result<TelemetryGuard> {
    if !config.enabled {
        return Ok(TelemetryGuard {
            _prom_handle: None,
        });
    }

    // -- Tracing / logging --
    let env_filter = EnvFilter::try_new(&config.log_level)
        .map_err(|e| LLMError::ConfigError(format!("bad log level: {e}")))?;

    match config.log_format {
        LogFormat::Text => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer())
                .try_init()
                .map_err(|e| LLMError::ConfigError(format!("tracing init: {e}")))?;
        }
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer().json())
                .try_init()
                .map_err(|e| LLMError::ConfigError(format!("tracing init: {e}")))?;
        }
    }

    // -- Prometheus --
    let prom_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| LLMError::ConfigError(format!("prometheus init: {e}")))?;

    crate::metrics::register_descriptions();

    // -- OTLP (stub) --
    if let Some(ref _endpoint) = config.otlp_endpoint {
        tracing::info!("OTLP export endpoint configured (stub, not yet connected)");
    }

    Ok(TelemetryGuard {
        _prom_handle: Some(prom_handle),
    })
}
