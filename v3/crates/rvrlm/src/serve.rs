use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::Rlm;
use crate::error::Result;
use crate::types::{EnvironmentKind, PerplexitySummary, Prompt, RlmChatCompletion};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ServeRequest {
    pub prompt: Prompt,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeResponse {
    pub completion: RlmChatCompletion,
    pub environment: EnvironmentKind,
}

pub struct ServeService {
    engine: Rlm,
}

impl ServeService {
    pub fn new(engine: Rlm) -> Self {
        Self { engine }
    }

    #[cfg(feature = "cuda")]
    pub fn from_rvllm_cuda(config: crate::clients::RvllmCudaConfig) -> Result<Self> {
        Ok(Self::new(Rlm::from_rvllm_cuda(config)?))
    }

    pub fn complete(&mut self, request: ServeRequest) -> Result<ServeResponse> {
        let completion = self.engine.completion(request.prompt)?;
        Ok(ServeResponse {
            completion,
            environment: self.engine.config().environment,
        })
    }

    pub fn perplexity(&mut self, request: ServeRequest) -> Result<PerplexitySummary> {
        self.engine.perplexity(request.prompt)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::clients::StubLanguageModel;
    use crate::environments::LocalEnvironment;
    use crate::types::{EnvironmentKind, Prompt};

    use super::{ServeRequest, ServeService};

    #[test]
    fn serve_service_runs_stub_completion() {
        let engine = crate::core::Rlm::builder()
            .client(StubLanguageModel::new("stub-model", "recursive answer"))
            .environment(LocalEnvironment::default())
            .build()
            .expect("builder should succeed");
        let mut service = ServeService::new(engine);

        let response = service
            .complete(ServeRequest {
                prompt: Prompt::from("trace this request"),
                ..ServeRequest::default()
            })
            .expect("completion should succeed");

        assert_eq!(response.environment, EnvironmentKind::Local);
        assert_eq!(response.completion.root_model, "stub-model");
        assert_eq!(response.completion.response, "recursive answer");
        assert!(response.completion.usage_summary.total_input_tokens() >= 3);
        assert_eq!(response.completion.usage_summary.total_output_tokens(), 2);
    }

    #[test]
    fn serve_service_preserves_structured_prompt() {
        let engine = crate::core::Rlm::builder()
            .client(StubLanguageModel::new("stub-model", "ok"))
            .environment(LocalEnvironment::default())
            .build()
            .expect("builder should succeed");
        let mut service = ServeService::new(engine);
        let prompt = Prompt::Structured(json!({
            "task": "summarize",
            "messages": ["a", "b"],
        }));

        let response = service
            .complete(ServeRequest {
                prompt: prompt.clone(),
                ..ServeRequest::default()
            })
            .expect("completion should succeed");

        assert_eq!(response.completion.prompt, prompt);
    }
}
