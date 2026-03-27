//! Scheduler configuration.

use serde::{Deserialize, Serialize};

/// Strategy the scheduler uses when a running sequence must be evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreemptionMode {
    /// Swap KV blocks to CPU.
    Swap,
    /// Discard KV and recompute on resume.
    Recompute,
}

impl Default for PreemptionMode {
    fn default() -> Self {
        Self::Recompute
    }
}

impl std::fmt::Display for PreemptionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Swap => write!(f, "swap"),
            Self::Recompute => write!(f, "recompute"),
        }
    }
}

impl std::str::FromStr for PreemptionMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "swap" => Ok(Self::Swap),
            "recompute" => Ok(Self::Recompute),
            other => Err(format!("unknown preemption mode: {other}")),
        }
    }
}

/// Configuration for the continuous-batching scheduler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchedulerConfigImpl {
    /// Maximum number of sequences that can run concurrently.
    pub max_num_seqs: usize,
    /// Maximum number of tokens in a single batch (prefill + decode).
    pub max_num_batched_tokens: usize,
    /// Maximum padding tokens allowed in a batch.
    pub max_paddings: usize,
    /// Preemption strategy.
    pub preemption_mode: PreemptionMode,
}

impl Default for SchedulerConfigImpl {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 2048,
            max_paddings: 256,
            preemption_mode: PreemptionMode::default(),
        }
    }
}

impl SchedulerConfigImpl {
    /// Create a new builder for tests and programmatic construction.
    pub fn builder() -> SchedulerConfigBuilder {
        SchedulerConfigBuilder::default()
    }
}

/// Builder for [`SchedulerConfigImpl`].
#[derive(Debug, Default)]
pub struct SchedulerConfigBuilder(SchedulerConfigImpl);

impl SchedulerConfigBuilder {
    /// Set max number of sequences.
    pub fn max_num_seqs(mut self, v: usize) -> Self {
        self.0.max_num_seqs = v;
        self
    }

    /// Set max batched tokens.
    pub fn max_num_batched_tokens(mut self, v: usize) -> Self {
        self.0.max_num_batched_tokens = v;
        self
    }

    /// Set max paddings.
    pub fn max_paddings(mut self, v: usize) -> Self {
        self.0.max_paddings = v;
        self
    }

    /// Set preemption mode.
    pub fn preemption_mode(mut self, v: PreemptionMode) -> Self {
        self.0.preemption_mode = v;
        self
    }

    /// Consume the builder and return the config.
    pub fn build(self) -> SchedulerConfigImpl {
        self.0
    }
}
