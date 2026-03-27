//! OpenAI-compatible request types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::ApiError;

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// POST /v1/completions request body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub logprobs: Option<usize>,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// POST /v1/chat/completions request body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Option<Vec<crate::routes::tools::RequestTool>>,
    #[serde(default)]
    pub tool_choice: Option<crate::routes::tools::ToolChoice>,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_n() -> usize {
    1
}

impl CompletionRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.model.is_empty() {
            return Err(ApiError::InvalidRequest("model is required".into()));
        }
        if self.prompt.is_empty() {
            return Err(ApiError::InvalidRequest("prompt is required".into()));
        }
        if self.max_tokens == 0 {
            return Err(ApiError::InvalidRequest(
                "max_tokens must be greater than 0".into(),
            ));
        }
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(ApiError::InvalidRequest(
                "temperature must be between 0.0 and 2.0".into(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::InvalidRequest(
                "top_p must be between 0.0 and 1.0".into(),
            ));
        }
        if self.n == 0 {
            return Err(ApiError::InvalidRequest(
                "n must be greater than 0".into(),
            ));
        }
        Ok(())
    }

    pub fn to_sampling_params(&self) -> rvllm_core::prelude::SamplingParams {
        rvllm_core::prelude::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stop_strings: self.stop.clone().unwrap_or_default(),
            logprobs: self.logprobs,
            echo: self.echo,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            best_of: self.n,
            ..Default::default()
        }
    }
}

impl ChatCompletionRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.model.is_empty() {
            return Err(ApiError::InvalidRequest("model is required".into()));
        }
        if self.messages.is_empty() {
            return Err(ApiError::InvalidRequest(
                "messages must not be empty".into(),
            ));
        }
        for (i, msg) in self.messages.iter().enumerate() {
            if msg.role.is_empty() {
                return Err(ApiError::InvalidRequest(format!(
                    "messages[{}].role is required",
                    i
                )));
            }
        }
        if self.max_tokens == 0 {
            return Err(ApiError::InvalidRequest(
                "max_tokens must be greater than 0".into(),
            ));
        }
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(ApiError::InvalidRequest(
                "temperature must be between 0.0 and 2.0".into(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::InvalidRequest(
                "top_p must be between 0.0 and 1.0".into(),
            ));
        }
        if self.n == 0 {
            return Err(ApiError::InvalidRequest(
                "n must be greater than 0".into(),
            ));
        }
        Ok(())
    }

    pub fn to_sampling_params(&self) -> rvllm_core::prelude::SamplingParams {
        rvllm_core::prelude::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stop_strings: self.stop.clone().unwrap_or_default(),
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            best_of: self.n,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_request_serde_roundtrip() {
        let req = CompletionRequest {
            model: "gpt-3.5-turbo".into(),
            prompt: "Hello".into(),
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: Some(vec!["\n".into()]),
            logprobs: Some(5),
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "gpt-3.5-turbo");
        assert_eq!(back.max_tokens, 100);
    }

    #[test]
    fn chat_request_serde_roundtrip() {
        let req = ChatCompletionRequest {
            model: "gpt-4".into(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: "You are helpful.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Hello".into(),
                },
            ],
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            n: 1,
            stream: true,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "gpt-4");
        assert_eq!(back.messages.len(), 2);
        assert!(back.stream);
    }

    #[test]
    fn completion_request_defaults() {
        let json = r#"{"model":"m","prompt":"p"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.top_p, 1.0);
        assert_eq!(req.n, 1);
        assert!(!req.stream);
    }

    #[test]
    fn completion_validate_ok() {
        let req = CompletionRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            logprobs: None,
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn completion_validate_empty_model() {
        let req = CompletionRequest {
            model: "".into(),
            prompt: "p".into(),
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            logprobs: None,
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn completion_validate_bad_temperature() {
        let req = CompletionRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 10,
            temperature: 3.0,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            logprobs: None,
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn chat_validate_empty_messages() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![],
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn chat_validate_empty_role() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "".into(),
                content: "hi".into(),
            }],
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn to_sampling_params_maps_fields() {
        let req = CompletionRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 42,
            temperature: 0.8,
            top_p: 0.95,
            n: 2,
            stream: false,
            stop: Some(vec!["END".into()]),
            logprobs: Some(3),
            echo: true,
            presence_penalty: 0.1,
            frequency_penalty: 0.2,
            user: None,
            seed: Some(123),
        };
        let sp = req.to_sampling_params();
        assert_eq!(sp.max_tokens, 42);
        assert_eq!(sp.temperature, 0.8);
        assert_eq!(sp.top_p, 0.95);
        assert_eq!(sp.stop_strings, vec!["END".to_string()]);
        assert_eq!(sp.logprobs, Some(3));
        assert!(sp.echo);
        assert_eq!(sp.seed, Some(123));
        assert_eq!(sp.best_of, 2);
    }

    #[test]
    fn chat_message_serde() {
        let msg = ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }
}
