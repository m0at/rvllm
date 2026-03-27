//! OpenAI-compatible response types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::request::ChatMessage;

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// A single completion choice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

/// Response for POST /v1/completions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// A single chat completion choice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub index: usize,
    pub finish_reason: Option<String>,
}

/// Response for POST /v1/chat/completions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Model info returned by GET /v1/models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Response for GET /v1/models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

fn finish_reason_string(reason: Option<rvllm_core::prelude::FinishReason>) -> Option<String> {
    reason.map(|r| match r {
        rvllm_core::prelude::FinishReason::Stop => "stop".to_string(),
        rvllm_core::prelude::FinishReason::Length => "length".to_string(),
        rvllm_core::prelude::FinishReason::Abort => "stop".to_string(),
    })
}

impl CompletionResponse {
    /// Build a CompletionResponse from engine output.
    pub fn from_request_output(
        output: &rvllm_core::prelude::RequestOutput,
        model: &str,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut total_completion = 0usize;
        let choices: Vec<CompletionChoice> = output
            .outputs
            .iter()
            .map(|co| {
                total_completion += co.token_ids.len();
                let logprobs_val = co.logprobs.as_ref().map(|lps| {
                    serde_json::json!({
                        "tokens": co.token_ids,
                        "token_logprobs": lps.iter().map(|pos| {
                            pos.first().map(|&(_, lp)| lp).unwrap_or(f32::NEG_INFINITY)
                        }).collect::<Vec<f32>>(),
                        "top_logprobs": lps.iter().map(|pos| {
                            pos.iter()
                                .map(|&(tid, lp)| (tid.to_string(), lp))
                                .collect::<std::collections::HashMap<String, f32>>()
                        }).collect::<Vec<_>>(),
                    })
                });
                CompletionChoice {
                    text: co.text.clone(),
                    index: co.index,
                    logprobs: logprobs_val,
                    finish_reason: finish_reason_string(co.finish_reason),
                }
            })
            .collect();

        CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: now,
            model: model.to_string(),
            choices,
            usage: Usage {
                prompt_tokens: output.prompt_token_ids.len(),
                completion_tokens: total_completion,
                total_tokens: output.prompt_token_ids.len() + total_completion,
            },
        }
    }
}

impl ChatCompletionResponse {
    /// Build a ChatCompletionResponse from engine output.
    pub fn from_request_output(
        output: &rvllm_core::prelude::RequestOutput,
        model: &str,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut total_completion = 0usize;
        let choices: Vec<ChatChoice> = output
            .outputs
            .iter()
            .map(|co| {
                total_completion += co.token_ids.len();
                ChatChoice {
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: co.text.clone(),
                    },
                    index: co.index,
                    finish_reason: finish_reason_string(co.finish_reason),
                }
            })
            .collect();

        ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: now,
            model: model.to_string(),
            choices,
            usage: Usage {
                prompt_tokens: output.prompt_token_ids.len(),
                completion_tokens: total_completion,
                total_tokens: output.prompt_token_ids.len() + total_completion,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::prelude::{CompletionOutput, FinishReason, RequestId, RequestOutput};

    fn sample_output() -> RequestOutput {
        RequestOutput {
            request_id: RequestId(1),
            prompt: "Hello".into(),
            prompt_token_ids: vec![1, 2, 3],
            prompt_logprobs: None,
            outputs: vec![CompletionOutput {
                index: 0,
                text: " world".into(),
                token_ids: vec![4, 5],
                cumulative_logprob: -0.5,
                logprobs: None,
                finish_reason: Some(FinishReason::Stop),
            }],
            finished: true,
        }
    }

    #[test]
    fn completion_response_serde() {
        let resp = CompletionResponse {
            id: "cmpl-test".into(),
            object: "text_completion".into(),
            created: 1234567890,
            model: "test-model".into(),
            choices: vec![CompletionChoice {
                text: "hello".into(),
                index: 0,
                logprobs: None,
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 1,
                total_tokens: 4,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: CompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn chat_response_serde() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion".into(),
            created: 1234567890,
            model: "test-model".into(),
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "hi".into(),
                },
                index: 0,
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: ChatCompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn from_request_output_completion() {
        let output = sample_output();
        let resp = CompletionResponse::from_request_output(&output, "my-model");
        assert!(resp.id.starts_with("cmpl-"));
        assert_eq!(resp.object, "text_completion");
        assert_eq!(resp.model, "my-model");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].text, " world");
        assert_eq!(resp.choices[0].finish_reason, Some("stop".into()));
        assert_eq!(resp.usage.prompt_tokens, 3);
        assert_eq!(resp.usage.completion_tokens, 2);
        assert_eq!(resp.usage.total_tokens, 5);
    }

    #[test]
    fn from_request_output_chat() {
        let output = sample_output();
        let resp = ChatCompletionResponse::from_request_output(&output, "my-model");
        assert!(resp.id.starts_with("chatcmpl-"));
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.role, "assistant");
        assert_eq!(resp.choices[0].message.content, " world");
    }

    #[test]
    fn usage_serde() {
        let u = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        let json = serde_json::to_string(&u).unwrap();
        let back: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, u);
    }

    #[test]
    fn model_list_response_serde() {
        let resp = ModelListResponse {
            object: "list".into(),
            data: vec![ModelObject {
                id: "gpt-4".into(),
                object: "model".into(),
                created: 123,
                owned_by: "org".into(),
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: ModelListResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn finish_reason_mapping() {
        assert_eq!(
            finish_reason_string(Some(FinishReason::Stop)),
            Some("stop".into())
        );
        assert_eq!(
            finish_reason_string(Some(FinishReason::Length)),
            Some("length".into())
        );
        assert_eq!(
            finish_reason_string(Some(FinishReason::Abort)),
            Some("stop".into())
        );
        assert_eq!(finish_reason_string(None), None);
    }
}
