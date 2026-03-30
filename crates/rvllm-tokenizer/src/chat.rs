//! Chat message types and simple template rendering.

use openai_harmony::chat::{
    Conversation as HarmonyConversation, Message as HarmonyMessage, Role as HarmonyRole,
};
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};
use rvllm_core::prelude::{LLMError, Result, TokenId};

/// Role in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System.as_str(), content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User.as_str(), content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant.as_str(), content)
    }
}

/// Apply a ChatML-style template to messages.
/// This is the default fallback when no Jinja2 template is available.
pub(crate) fn apply_chatml(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(LLMError::TokenizerError("empty message list".into()));
    }

    let mut out = String::new();
    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    Ok(out)
}

pub(crate) fn apply_harmony(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<Vec<TokenId>> {
    if messages.is_empty() {
        return Err(LLMError::TokenizerError("empty message list".into()));
    }

    let harmony_messages = messages
        .iter()
        .map(|msg| {
            let role = HarmonyRole::try_from(msg.role.as_str()).map_err(|_| {
                LLMError::TokenizerError(format!(
                    "unsupported Harmony chat role '{}'",
                    msg.role
                ))
            })?;
            Ok(HarmonyMessage::from_role_and_content(role, msg.content.clone()))
        })
        .collect::<Result<Vec<_>>>()?;

    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
        .map_err(|e| LLMError::TokenizerError(format!("failed to load Harmony encoding: {e}")))?;
    let convo = HarmonyConversation::from_messages(harmony_messages);

    let tokens = if add_generation_prompt {
        encoding.render_conversation_for_completion(&convo, HarmonyRole::Assistant, None)
    } else {
        encoding.render_conversation(&convo, None)
    }
    .map_err(|e| LLMError::TokenizerError(format!("failed to render Harmony prompt: {e}")))?;

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_basic() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let result = apply_chatml(&msgs, true).unwrap();
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_no_generation_prompt() {
        let msgs = vec![ChatMessage::user("Hi")];
        let result = apply_chatml(&msgs, false).unwrap();
        assert!(!result.contains("assistant"));
    }

    #[test]
    fn chatml_empty_errors() {
        assert!(apply_chatml(&[], true).is_err());
    }

    #[test]
    fn harmony_basic() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let result = apply_harmony(&msgs, true).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn chat_role_display() {
        assert_eq!(ChatRole::System.to_string(), "system");
        assert_eq!(ChatRole::User.to_string(), "user");
        assert_eq!(ChatRole::Assistant.to_string(), "assistant");
    }

    #[test]
    fn chat_message_constructors() {
        let m = ChatMessage::system("test");
        assert_eq!(m.role, "system");
        assert_eq!(m.content, "test");

        let m = ChatMessage::user("hello");
        assert_eq!(m.role, "user");

        let m = ChatMessage::assistant("hi");
        assert_eq!(m.role, "assistant");
    }
}
