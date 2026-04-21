use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{BackendKind, EnvironmentKind};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RlmConfig {
    pub backend: BackendKind,
    pub backend_kwargs: BTreeMap<String, Value>,
    pub environment: EnvironmentKind,
    pub environment_kwargs: BTreeMap<String, Value>,
    pub model_name: Option<String>,
    pub depth: u32,
    pub max_depth: u32,
    pub max_iterations: u32,
    pub max_budget_usd: Option<f64>,
    pub max_timeout_secs: Option<f64>,
    pub max_tokens: Option<u64>,
    pub max_errors: Option<u32>,
    pub other_backends: Vec<BackendKind>,
    pub persistent: bool,
    pub compaction: bool,
    pub compaction_threshold_pct: f32,
    pub max_concurrent_subcalls: usize,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::OpenAi,
            backend_kwargs: BTreeMap::new(),
            environment: EnvironmentKind::Local,
            environment_kwargs: BTreeMap::new(),
            model_name: None,
            depth: 0,
            max_depth: 1,
            max_iterations: 30,
            max_budget_usd: None,
            max_timeout_secs: None,
            max_tokens: None,
            max_errors: None,
            other_backends: Vec::new(),
            persistent: false,
            compaction: false,
            compaction_threshold_pct: 0.85,
            max_concurrent_subcalls: 4,
        }
    }
}
