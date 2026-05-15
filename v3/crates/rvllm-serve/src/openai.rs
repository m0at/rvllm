//! OpenAI-compatible wire types + chat-template rendering.
//!
//! We deliberately stay close to vLLM's emitted shape so existing
//! clients (openai SDK, langchain, etc.) work unchanged. The
//! per-token streaming chunk carries `choices[0].delta.content`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Incoming /v1/chat/completions request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// vLLM/OpenAI accept either a single string or an array; we accept
    /// both transparently.
    #[serde(default)]
    pub stop: Option<StopField>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

/// Non-streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// One streaming chunk (one per token, plus a final one carrying
/// finish_reason).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// /v1/models list entry.
#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str, // "model"
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str, // "list"
    pub data: Vec<ModelEntry>,
}

/// Renders chat messages -> a single prompt string using the model's
/// `chat_template.jinja`. Mirrors HuggingFace's
/// `tokenizer.apply_chat_template(..., add_generation_prompt=true)`.
#[derive(Debug)]
pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
    template_name: &'static str,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Load chat_template.jinja + tokenizer_config.json's special
    /// tokens from a HF model directory. No fallback: a missing
    /// chat_template aborts startup.
    pub fn from_model_dir(dir: &std::path::Path) -> Result<Self, String> {
        let tmpl_path = dir.join("chat_template.jinja");
        let tmpl_source = match std::fs::read_to_string(&tmpl_path) {
            Ok(s) => s,
            Err(_) => {
                // Some HF repos store the template inside
                // `tokenizer_config.json` under `chat_template`.
                let tok_cfg = dir.join("tokenizer_config.json");
                let cfg_str = std::fs::read_to_string(&tok_cfg).map_err(|e| {
                    format!(
                        "neither {} nor {} present: {e}",
                        tmpl_path.display(),
                        tok_cfg.display()
                    )
                })?;
                let v: serde_json::Value = serde_json::from_str(&cfg_str)
                    .map_err(|e| format!("parse {}: {e}", tok_cfg.display()))?;
                v.get("chat_template")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| {
                        format!(
                            "no chat_template.jinja and no \"chat_template\" key in {}",
                            tok_cfg.display()
                        )
                    })?
                    .to_string()
            }
        };

        let (bos, eos) = load_special_tokens(dir).unwrap_or_default();

        let mut env = minijinja::Environment::new();
        env.add_template_owned("chat", tmpl_source)
            .map_err(|e| format!("chat_template compile: {e}"))?;
        Ok(ChatTemplate {
            env,
            template_name: "chat",
            bos_token: bos,
            eos_token: eos,
        })
    }

    /// Build a ChatTemplate directly from a template source string +
    /// special tokens. Used by the dry-run path so off-GPU smoke
    /// tests work without a HF model dir on disk.
    pub fn from_source(
        source: String,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self, String> {
        let mut env = minijinja::Environment::new();
        env.add_template_owned("chat", source)
            .map_err(|e| format!("chat_template compile: {e}"))?;
        Ok(ChatTemplate {
            env,
            template_name: "chat",
            bos_token,
            eos_token,
        })
    }

    pub fn render(&self, messages: &[ChatMessage]) -> Result<String, String> {
        let tmpl = self
            .env
            .get_template(self.template_name)
            .map_err(|e| format!("chat_template: {e}"))?;
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        let ctx = minijinja::context! {
            messages => msgs,
            add_generation_prompt => true,
            bos_token => self.bos_token.as_str(),
            eos_token => self.eos_token.as_str(),
        };
        tmpl.render(ctx)
            .map_err(|e| format!("chat_template render: {e}"))
    }
}

fn load_special_tokens(dir: &std::path::Path) -> Option<(String, String)> {
    let p = dir.join("tokenizer_config.json");
    let s = std::fs::read_to_string(&p).ok()?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    let bos = v
        .get("bos_token")
        .and_then(extract_token_str)
        .unwrap_or_default();
    let eos = v
        .get("eos_token")
        .and_then(extract_token_str)
        .unwrap_or_default();
    Some((bos, eos))
}

fn extract_token_str(v: &serde_json::Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    v.get("content").and_then(|c| c.as_str()).map(|s| s.to_string())
}

/// SharedTokenizer is an Arc-wrapped HF tokenizer (cheap to clone for
/// per-request use).
pub type SharedTokenizer = Arc<tokenizers::Tokenizer>;
