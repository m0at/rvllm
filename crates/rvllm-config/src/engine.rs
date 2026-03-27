//! Top-level engine configuration that composes all sub-configs.

use serde::{Deserialize, Serialize};

use crate::cache::CacheConfigImpl;
use crate::device::DeviceConfig;
use crate::model::ModelConfigImpl;
use crate::parallel::ParallelConfigImpl;
use crate::scheduler::SchedulerConfigImpl;
use crate::telemetry::TelemetryConfig;

/// Top-level configuration composing every subsystem.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineConfig {
    /// Model weights and tokenizer settings.
    pub model: ModelConfigImpl,
    /// KV-cache memory budget.
    pub cache: CacheConfigImpl,
    /// Scheduler batch limits and preemption policy.
    pub scheduler: SchedulerConfigImpl,
    /// Tensor / pipeline parallelism.
    pub parallel: ParallelConfigImpl,
    /// Target device.
    pub device: DeviceConfig,
    /// Observability settings.
    pub telemetry: TelemetryConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelConfigImpl::default(),
            cache: CacheConfigImpl::default(),
            scheduler: SchedulerConfigImpl::default(),
            parallel: ParallelConfigImpl::default(),
            device: DeviceConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

impl EngineConfig {
    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::default()
    }
}

/// Builder for [`EngineConfig`].
#[derive(Debug, Default)]
pub struct EngineConfigBuilder(EngineConfig);

impl EngineConfigBuilder {
    /// Set model config.
    pub fn model(mut self, v: ModelConfigImpl) -> Self {
        self.0.model = v;
        self
    }

    /// Set cache config.
    pub fn cache(mut self, v: CacheConfigImpl) -> Self {
        self.0.cache = v;
        self
    }

    /// Set scheduler config.
    pub fn scheduler(mut self, v: SchedulerConfigImpl) -> Self {
        self.0.scheduler = v;
        self
    }

    /// Set parallel config.
    pub fn parallel(mut self, v: ParallelConfigImpl) -> Self {
        self.0.parallel = v;
        self
    }

    /// Set device config.
    pub fn device(mut self, v: DeviceConfig) -> Self {
        self.0.device = v;
        self
    }

    /// Set telemetry config.
    pub fn telemetry(mut self, v: TelemetryConfig) -> Self {
        self.0.telemetry = v;
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> EngineConfig {
        self.0
    }
}
