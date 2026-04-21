use std::time::Instant;

use tracing::debug;

use crate::clients::LanguageModel;
use crate::config::RlmConfig;
use crate::environments::{Environment, ExecutionCallbacks, HostCall};
use crate::error::{Result, RlmError};
use crate::logger::TrajectoryLogger;
use crate::types::{
    CodeBlock, ModelUsageSummary, PerplexitySummary, Prompt, RlmChatCompletion, RlmIteration,
    RlmMetadata, UsageSummary,
};
use crate::utils::parsing::{find_code_blocks, find_final_answer};
use crate::utils::prompts::build_iteration_prompt;

pub struct RlmBuilder {
    config: RlmConfig,
    client: Option<Box<dyn LanguageModel>>,
    environment: Option<Box<dyn Environment>>,
    logger: Option<TrajectoryLogger>,
}

impl Default for RlmBuilder {
    fn default() -> Self {
        Self {
            config: RlmConfig::default(),
            client: None,
            environment: None,
            logger: None,
        }
    }
}

impl RlmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn config(mut self, config: RlmConfig) -> Self {
        self.config = config;
        self
    }

    pub fn client<LM>(mut self, client: LM) -> Self
    where
        LM: LanguageModel + 'static,
    {
        self.client = Some(Box::new(client));
        self
    }

    pub fn boxed_client(mut self, client: Box<dyn LanguageModel>) -> Self {
        self.client = Some(client);
        self
    }

    pub fn environment<E>(mut self, environment: E) -> Self
    where
        E: Environment + 'static,
    {
        self.environment = Some(Box::new(environment));
        self
    }

    pub fn boxed_environment(mut self, environment: Box<dyn Environment>) -> Self {
        self.environment = Some(environment);
        self
    }

    pub fn logger(mut self, logger: TrajectoryLogger) -> Self {
        self.logger = Some(logger);
        self
    }

    pub fn build(self) -> Result<Rlm> {
        let client = self.client.ok_or(RlmError::MissingClient)?;
        let environment = self.environment.ok_or(RlmError::MissingEnvironment)?;

        Ok(Rlm {
            config: self.config,
            client,
            environment,
            logger: self.logger,
        })
    }
}

pub struct Rlm {
    config: RlmConfig,
    client: Box<dyn LanguageModel>,
    environment: Box<dyn Environment>,
    logger: Option<TrajectoryLogger>,
}

impl Rlm {
    pub fn builder() -> RlmBuilder {
        RlmBuilder::new()
    }

    #[cfg(feature = "cuda")]
    pub fn from_rvllm_cuda(config: crate::clients::RvllmCudaConfig) -> Result<Self> {
        let mut rlm_config = RlmConfig::default();
        rlm_config.backend = crate::types::BackendKind::Rvllm;
        rlm_config.environment = crate::types::EnvironmentKind::Local;
        rlm_config.model_name = config.model_name.clone();

        Rlm::builder()
            .config(rlm_config)
            .client(crate::clients::RvllmCudaClient::from_config(config)?)
            .environment(crate::environments::LocalEnvironment::default())
            .build()
    }

    pub fn config(&self) -> &RlmConfig {
        &self.config
    }

    pub fn completion(&mut self, prompt: impl Into<Prompt>) -> Result<RlmChatCompletion> {
        let prompt = prompt.into();
        self.environment.setup()?;
        self.environment.load_context(prompt.clone())?;

        let metadata = self.metadata();
        if let Some(logger) = self.logger.as_mut() {
            logger.log_metadata(metadata)?;
        }

        let started = Instant::now();
        let mut previous_response: Option<String> = None;
        let mut executed_blocks: Vec<CodeBlock> = Vec::new();
        let mut final_response: Option<String> = None;

        for _ in 0..self.config.max_iterations {
            let iteration_prompt =
                build_iteration_prompt(&prompt, previous_response.as_deref(), &executed_blocks);
            let iteration_started = Instant::now();
            let response = self
                .client
                .completion(&Prompt::from(iteration_prompt.clone()))?;
            let final_answer = find_final_answer(&response);

            let mut subcall_client = self.client.fork()?;
            let subcall_environment = self.environment.fork()?;
            let parent_config = self.config.clone();
            let mut host_call = |call: HostCall| -> Result<RlmChatCompletion> {
                match call {
                    HostCall::Llm(prompt) => {
                        let completion = call_model_once(self.client.as_mut(), prompt)?;
                        subcall_client = self.client.fork()?;
                        Ok(completion)
                    }
                    HostCall::Rlm(prompt) => run_subcall(
                        &parent_config,
                        subcall_client.fork()?,
                        subcall_environment.fork()?,
                        prompt,
                    ),
                }
            };

            let code_blocks = find_code_blocks(&response)
                .into_iter()
                .map(|code| {
                    let result = self.environment.execute_code(
                        &code,
                        &mut ExecutionCallbacks {
                            host_call: &mut host_call,
                        },
                    )?;
                    Ok(CodeBlock { code, result })
                })
                .collect::<Result<Vec<_>>>()?;

            let had_code_blocks = !code_blocks.is_empty();
            if let Some(logger) = self.logger.as_mut() {
                logger.log_iteration(RlmIteration {
                    prompt: Prompt::from(iteration_prompt),
                    response: response.clone(),
                    code_blocks: code_blocks.clone(),
                    final_answer: final_answer.clone(),
                    iteration_time_secs: Some(iteration_started.elapsed().as_secs_f64()),
                })?;
            }

            executed_blocks.extend(code_blocks);
            if let Some(answer) = final_answer {
                final_response = Some(answer);
                break;
            }
            if !had_code_blocks {
                final_response = Some(response);
                break;
            }
            previous_response = Some(response);
        }

        let response = final_response.ok_or_else(|| {
            RlmError::Unsupported("rvRLM exhausted iterations without a final answer".to_owned())
        })?;
        let execution_time_secs = started.elapsed().as_secs_f64();
        debug!(
            model = self.client.model_name(),
            elapsed_secs = execution_time_secs,
            "rvRLM completion"
        );

        let metadata = if let Some(logger) = self.logger.as_ref() {
            Some(serde_json::to_value(logger.snapshot())?)
        } else {
            None
        };

        Ok(RlmChatCompletion {
            root_model: self.client.model_name().to_owned(),
            prompt,
            response,
            usage_summary: aggregate_usage(self.client.usage_summary(), &executed_blocks),
            execution_time_secs,
            metadata,
        })
    }

    pub fn perplexity(&mut self, prompt: impl Into<Prompt>) -> Result<PerplexitySummary> {
        self.client.perplexity(&prompt.into())
    }

    fn metadata(&self) -> RlmMetadata {
        RlmMetadata {
            root_model: self
                .config
                .model_name
                .clone()
                .unwrap_or_else(|| self.client.model_name().to_owned()),
            max_depth: self.config.max_depth,
            max_iterations: self.config.max_iterations,
            backend: self.config.backend,
            backend_kwargs: self.config.backend_kwargs.clone(),
            environment_type: self.config.environment,
            environment_kwargs: self.config.environment_kwargs.clone(),
            other_backends: self.config.other_backends.clone(),
        }
    }
}

pub type RLM = Rlm;

fn run_subcall(
    parent_config: &RlmConfig,
    client: Box<dyn LanguageModel>,
    environment: Box<dyn Environment>,
    prompt: Prompt,
) -> Result<RlmChatCompletion> {
    if parent_config.depth >= parent_config.max_depth {
        return Err(RlmError::Unsupported(format!(
            "rvRLM max recursion depth {} reached",
            parent_config.max_depth
        )));
    }

    let mut child_config = parent_config.clone();
    child_config.depth += 1;

    Rlm::builder()
        .config(child_config)
        .boxed_client(client)
        .boxed_environment(environment)
        .build()?
        .completion(prompt)
}

fn call_model_once(client: &mut dyn LanguageModel, prompt: Prompt) -> Result<RlmChatCompletion> {
    let started = Instant::now();
    let response = client.completion(&prompt)?;

    Ok(RlmChatCompletion {
        root_model: client.model_name().to_owned(),
        prompt,
        response,
        usage_summary: usage_from_last_call(client.model_name(), client.last_usage()),
        execution_time_secs: started.elapsed().as_secs_f64(),
        metadata: None,
    })
}

fn usage_from_last_call(model_name: &str, usage: Option<ModelUsageSummary>) -> UsageSummary {
    let mut summaries = std::collections::BTreeMap::new();
    if let Some(summary) = usage {
        summaries.insert(model_name.to_owned(), summary);
    }
    UsageSummary {
        model_usage_summaries: summaries,
    }
}

fn aggregate_usage(base: UsageSummary, executed_blocks: &[CodeBlock]) -> UsageSummary {
    let mut merged = base;
    for block in executed_blocks {
        for call in &block.result.llm_calls {
            merge_usage(&mut merged, &call.usage_summary);
        }
    }
    merged
}

fn merge_usage(target: &mut UsageSummary, source: &UsageSummary) {
    for (model, summary) in &source.model_usage_summaries {
        let entry = target
            .model_usage_summaries
            .entry(model.clone())
            .or_default();
        entry.total_calls += summary.total_calls;
        entry.total_input_tokens += summary.total_input_tokens;
        entry.total_output_tokens += summary.total_output_tokens;
        entry.total_cost = match (entry.total_cost, summary.total_cost) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use serde_json::Value;

    use crate::clients::{LanguageModel, ScriptedLanguageModel, StubLanguageModel};
    use crate::environments::{Environment, LocalEnvironment};
    use crate::error::Result;
    use crate::types::{
        EnvironmentKind, ModelUsageSummary, PerplexitySummary, Prompt, ReplResult, UsageSummary,
    };

    use super::Rlm;

    struct ForkAwareLanguageModel {
        model_name: String,
        root_responses: VecDeque<String>,
        child_responses: Vec<String>,
    }

    impl ForkAwareLanguageModel {
        fn new(root_responses: &[&str], child_responses: &[&str]) -> Self {
            Self {
                model_name: "fork-aware".to_owned(),
                root_responses: root_responses
                    .iter()
                    .map(|item| (*item).to_owned())
                    .collect(),
                child_responses: child_responses
                    .iter()
                    .map(|item| (*item).to_owned())
                    .collect(),
            }
        }
    }

    impl LanguageModel for ForkAwareLanguageModel {
        fn model_name(&self) -> &str {
            &self.model_name
        }

        fn completion(&mut self, _prompt: &Prompt) -> Result<String> {
            self.root_responses.pop_front().ok_or_else(|| {
                crate::error::RlmError::Client("fork-aware model ran out of responses".to_owned())
            })
        }

        fn fork(&self) -> Result<Box<dyn LanguageModel>> {
            Ok(Box::new(ScriptedLanguageModel::new(
                self.model_name.clone(),
                &self
                    .child_responses
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )))
        }

        fn usage_summary(&self) -> UsageSummary {
            UsageSummary::default()
        }

        fn last_usage(&self) -> Option<ModelUsageSummary> {
            None
        }
    }

    struct ScriptedEnvironment {
        executed: Vec<String>,
    }

    impl ScriptedEnvironment {
        fn new() -> Self {
            Self {
                executed: Vec::new(),
            }
        }
    }

    impl Environment for ScriptedEnvironment {
        fn kind(&self) -> EnvironmentKind {
            EnvironmentKind::Local
        }

        fn fork(&self) -> Result<Box<dyn Environment>> {
            Ok(Box::new(Self::new()))
        }

        fn setup(&mut self) -> Result<()> {
            Ok(())
        }

        fn load_context(&mut self, _context: Prompt) -> Result<()> {
            Ok(())
        }

        fn execute_code(
            &mut self,
            code: &str,
            _callbacks: &mut crate::environments::ExecutionCallbacks<'_>,
        ) -> Result<ReplResult> {
            self.executed.push(code.to_owned());
            Ok(ReplResult {
                stdout: "2\n".to_owned(),
                stderr: String::new(),
                locals: std::iter::once(("answer".to_owned(), Value::from(2))).collect(),
                execution_time_secs: 0.001,
                llm_calls: Vec::new(),
                final_answer: None,
            })
        }
    }

    #[test]
    fn stub_completion_compiles_end_to_end() {
        let mut rlm = Rlm::builder()
            .client(StubLanguageModel::new("gpt-5-nano", "stub response"))
            .environment(LocalEnvironment::default())
            .build()
            .expect("builder should succeed");

        let completion = rlm.completion("hello recursive world").expect("completion");
        assert_eq!(completion.root_model, "gpt-5-nano");
        assert_eq!(completion.response, "stub response");
        assert!(completion.usage_summary.total_input_tokens() >= 3);
    }

    #[test]
    fn executes_code_and_returns_final_answer() {
        let model = ScriptedLanguageModel::new(
            "scripted",
            &[
                "```python\nprint(1 + 1)\n```",
                "FINAL_ANSWER: the answer is 2",
            ],
        );
        let mut rlm = Rlm::builder()
            .client(model)
            .environment(ScriptedEnvironment::new())
            .build()
            .expect("builder should succeed");

        let completion = rlm.completion("compute 1 + 1").expect("completion");
        assert_eq!(completion.root_model, "scripted");
        assert_eq!(completion.response, "the answer is 2");
    }

    #[test]
    fn recursive_subcall_runs_through_native_engine() {
        let model = ForkAwareLanguageModel::new(
            &[
                "```python\nFINAL_VAR = rlm_query('solve child task')\n```",
                "FINAL_ANSWER: child finished",
            ],
            &["FINAL_ANSWER: child result"],
        );
        let mut rlm = Rlm::builder()
            .config(crate::config::RlmConfig {
                max_depth: 2,
                ..crate::config::RlmConfig::default()
            })
            .client(model)
            .environment(LocalEnvironment::default())
            .build()
            .expect("builder should succeed");

        let completion = rlm
            .completion("run the child engine")
            .expect("completion should succeed");

        assert_eq!(completion.response, "child finished");
    }

    #[test]
    fn perplexity_forwards_to_client() {
        struct PplModel;

        impl LanguageModel for PplModel {
            fn model_name(&self) -> &str {
                "ppl-model"
            }

            fn completion(&mut self, _prompt: &Prompt) -> Result<String> {
                Ok("unused".to_owned())
            }

            fn fork(&self) -> Result<Box<dyn LanguageModel>> {
                Ok(Box::new(Self))
            }

            fn perplexity(&mut self, prompt: &Prompt) -> Result<PerplexitySummary> {
                Ok(PerplexitySummary {
                    root_model: self.model_name().to_owned(),
                    prompt: prompt.clone(),
                    perplexity: 1.5,
                    evaluated_tokens: 12,
                    total_nll: 4.86,
                })
            }

            fn usage_summary(&self) -> UsageSummary {
                UsageSummary::default()
            }

            fn last_usage(&self) -> Option<ModelUsageSummary> {
                None
            }
        }

        let mut rlm = Rlm::builder()
            .client(PplModel)
            .environment(LocalEnvironment::default())
            .build()
            .expect("builder should succeed");

        let summary = rlm.perplexity("bench me").expect("perplexity should work");
        assert_eq!(summary.root_model, "ppl-model");
        assert_eq!(summary.perplexity, 1.5);
        assert_eq!(summary.prompt, Prompt::from("bench me"));
    }
}
