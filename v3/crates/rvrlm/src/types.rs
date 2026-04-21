use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Rvllm,
    OpenAi,
    Portkey,
    OpenRouter,
    Vercel,
    Vllm,
    Anthropic,
    AzureOpenAi,
    Gemini,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentKind {
    Local,
    Docker,
    Modal,
    Prime,
    Daytona,
    E2b,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Text(String),
    Structured(Value),
}

impl Prompt {
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Structured(value) => value.to_string(),
        }
    }

    pub fn as_value(&self) -> Value {
        match self {
            Self::Text(text) => Value::String(text.clone()),
            Self::Structured(value) => value.clone(),
        }
    }
}

impl From<String> for Prompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for Prompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<Value> for Prompt {
    fn from(value: Value) -> Self {
        Self::Structured(value)
    }
}

impl Default for Prompt {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelUsageSummary {
    pub total_calls: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageSummary {
    pub model_usage_summaries: BTreeMap<String, ModelUsageSummary>,
}

impl UsageSummary {
    pub fn total_cost(&self) -> Option<f64> {
        let mut total = 0.0;
        let mut saw_cost = false;
        for summary in self.model_usage_summaries.values() {
            if let Some(cost) = summary.total_cost {
                total += cost;
                saw_cost = true;
            }
        }
        if saw_cost {
            Some(total)
        } else {
            None
        }
    }

    pub fn total_input_tokens(&self) -> u64 {
        self.model_usage_summaries
            .values()
            .map(|summary| summary.total_input_tokens)
            .sum()
    }

    pub fn total_output_tokens(&self) -> u64 {
        self.model_usage_summaries
            .values()
            .map(|summary| summary.total_output_tokens)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlmChatCompletion {
    pub root_model: String,
    pub prompt: Prompt,
    pub response: String,
    pub usage_summary: UsageSummary,
    pub execution_time_secs: f64,
    pub metadata: Option<Value>,
}

pub type ChatCompletion = RlmChatCompletion;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerplexitySummary {
    pub root_model: String,
    pub prompt: Prompt,
    pub perplexity: f64,
    pub evaluated_tokens: usize,
    pub total_nll: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplResult {
    pub stdout: String,
    pub stderr: String,
    pub locals: BTreeMap<String, Value>,
    pub execution_time_secs: f64,
    pub llm_calls: Vec<RlmChatCompletion>,
    pub final_answer: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeBlock {
    pub code: String,
    pub result: ReplResult,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlmIteration {
    pub prompt: Prompt,
    pub response: String,
    pub code_blocks: Vec<CodeBlock>,
    pub final_answer: Option<String>,
    pub iteration_time_secs: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RlmMetadata {
    pub root_model: String,
    pub max_depth: u32,
    pub max_iterations: u32,
    pub backend: BackendKind,
    pub backend_kwargs: BTreeMap<String, Value>,
    pub environment_type: EnvironmentKind,
    pub environment_kwargs: BTreeMap<String, Value>,
    pub other_backends: Vec<BackendKind>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryMetadata {
    pub context_lengths: Vec<usize>,
}
