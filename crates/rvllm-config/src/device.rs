//! Device configuration.

use serde::{Deserialize, Serialize};

/// Which device family to target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceConfig {
    /// Device string: "cuda", "cpu", "metal", etc.
    pub device: String,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device: "cuda".into(),
        }
    }
}

impl DeviceConfig {
    /// Returns true when targeting a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(self.device.as_str(), "cuda" | "metal")
    }

    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> DeviceConfigBuilder {
        DeviceConfigBuilder::default()
    }
}

/// Builder for [`DeviceConfig`].
#[derive(Debug, Default)]
pub struct DeviceConfigBuilder(DeviceConfig);

impl DeviceConfigBuilder {
    /// Set device string.
    pub fn device(mut self, v: impl Into<String>) -> Self {
        self.0.device = v.into();
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> DeviceConfig {
        self.0
    }
}
