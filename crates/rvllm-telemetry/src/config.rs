use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    Text,
    Json,
}

impl Default for LogFormat {
    fn default() -> Self {
        Self::Text
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub enabled: bool,
    #[serde(default = "default_prometheus_port")]
    pub prometheus_port: Option<u16>,
    pub otlp_endpoint: Option<String>,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub log_format: LogFormat,
}

fn default_prometheus_port() -> Option<u16> {
    Some(9090)
}

fn default_log_level() -> String {
    "info".into()
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prometheus_port: Some(9090),
            otlp_endpoint: None,
            log_level: "info".into(),
            log_format: LogFormat::Text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = TelemetryConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.prometheus_port, Some(9090));
        assert!(cfg.otlp_endpoint.is_none());
        assert_eq!(cfg.log_level, "info");
        assert_eq!(cfg.log_format, LogFormat::Text);
    }
}
