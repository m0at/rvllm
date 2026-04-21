use std::collections::BTreeMap;
use std::collections::VecDeque;

use crate::error::Result;
use crate::types::{ModelUsageSummary, PerplexitySummary, Prompt, UsageSummary};
use crate::utils::token_utils::prompt_token_count;

pub trait LanguageModel {
    fn model_name(&self) -> &str;
    fn completion(&mut self, prompt: &Prompt) -> Result<String>;
    fn fork(&self) -> Result<Box<dyn LanguageModel>>;
    fn perplexity(&mut self, _prompt: &Prompt) -> Result<PerplexitySummary> {
        Err(crate::error::RlmError::Unsupported(format!(
            "perplexity is not supported for model `{}`",
            self.model_name()
        )))
    }
    fn usage_summary(&self) -> UsageSummary;
    fn last_usage(&self) -> Option<ModelUsageSummary>;
}

pub struct StubLanguageModel {
    model_name: String,
    response: String,
    total_calls: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    last_usage: Option<ModelUsageSummary>,
}

impl StubLanguageModel {
    pub fn new(model_name: impl Into<String>, response: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            response: response.into(),
            total_calls: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            last_usage: None,
        }
    }
}

impl LanguageModel for StubLanguageModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn completion(&mut self, prompt: &Prompt) -> Result<String> {
        let input_tokens = prompt_token_count(prompt) as u64;
        let output_tokens = self.response.split_whitespace().count() as u64;

        self.total_calls += 1;
        self.total_input_tokens += input_tokens;
        self.total_output_tokens += output_tokens;
        self.last_usage = Some(ModelUsageSummary {
            total_calls: 1,
            total_input_tokens: input_tokens,
            total_output_tokens: output_tokens,
            total_cost: None,
        });

        Ok(self.response.clone())
    }

    fn fork(&self) -> Result<Box<dyn LanguageModel>> {
        Ok(Box::new(Self::new(
            self.model_name.clone(),
            self.response.clone(),
        )))
    }

    fn usage_summary(&self) -> UsageSummary {
        let mut summaries = BTreeMap::new();
        summaries.insert(
            self.model_name.clone(),
            ModelUsageSummary {
                total_calls: self.total_calls,
                total_input_tokens: self.total_input_tokens,
                total_output_tokens: self.total_output_tokens,
                total_cost: None,
            },
        );
        UsageSummary {
            model_usage_summaries: summaries,
        }
    }

    fn last_usage(&self) -> Option<ModelUsageSummary> {
        self.last_usage.clone()
    }
}

pub struct ScriptedLanguageModel {
    model_name: String,
    responses: VecDeque<String>,
    prompts: Vec<String>,
}

impl ScriptedLanguageModel {
    pub fn new(model_name: impl Into<String>, responses: &[&str]) -> Self {
        Self {
            model_name: model_name.into(),
            responses: responses.iter().map(|item| (*item).to_owned()).collect(),
            prompts: Vec::new(),
        }
    }

    pub fn seen_prompts(&self) -> &[String] {
        &self.prompts
    }
}

impl LanguageModel for ScriptedLanguageModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn completion(&mut self, prompt: &Prompt) -> Result<String> {
        self.prompts.push(prompt.as_text());
        self.responses.pop_front().ok_or_else(|| {
            crate::error::RlmError::Client("scripted model ran out of responses".to_owned())
        })
    }

    fn fork(&self) -> Result<Box<dyn LanguageModel>> {
        Ok(Box::new(Self {
            model_name: self.model_name.clone(),
            responses: self.responses.clone(),
            prompts: Vec::new(),
        }))
    }

    fn usage_summary(&self) -> UsageSummary {
        UsageSummary::default()
    }

    fn last_usage(&self) -> Option<ModelUsageSummary> {
        None
    }
}
