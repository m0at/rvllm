//! Parallelism configuration.

use serde::{Deserialize, Serialize};

/// Configuration for tensor and pipeline parallelism.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParallelConfigImpl {
    /// Number of GPUs for tensor parallelism.
    pub tensor_parallel_size: usize,
    /// Number of pipeline stages.
    pub pipeline_parallel_size: usize,
}

impl Default for ParallelConfigImpl {
    fn default() -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
        }
    }
}

impl ParallelConfigImpl {
    /// Total number of GPUs required (tp * pp).
    pub fn world_size(&self) -> usize {
        self.tensor_parallel_size * self.pipeline_parallel_size
    }

    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> ParallelConfigBuilder {
        ParallelConfigBuilder::default()
    }
}

/// Builder for [`ParallelConfigImpl`].
#[derive(Debug, Default)]
pub struct ParallelConfigBuilder(ParallelConfigImpl);

impl ParallelConfigBuilder {
    /// Set tensor parallel size.
    pub fn tensor_parallel_size(mut self, v: usize) -> Self {
        self.0.tensor_parallel_size = v;
        self
    }

    /// Set pipeline parallel size.
    pub fn pipeline_parallel_size(mut self, v: usize) -> Self {
        self.0.pipeline_parallel_size = v;
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> ParallelConfigImpl {
        self.0
    }
}
