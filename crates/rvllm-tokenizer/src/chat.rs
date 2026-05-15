//! Chat message types and simple template rendering.

use rvllm_core::prelude::{LLMError, Result};

/// Chat template variant to use when formatting messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplate {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>\n`
    #[default]
    ChatML,
    /// Harmony (GPT-OSS): `<|im_start|>role<|im_sep|>\ncontent\n<|im_end|>\n`
    Harmony,
    /// Gemma 4 turn format: `<|turn>user\n...<turn|>\n<|turn>model\n...`
    Gemma4,
}

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

/// Apply a Harmony-style template to messages (used by GPT-OSS family).
/// Format: `<|im_start|>role<|im_sep|>\ncontent\n<|im_end|>\n`
pub(crate) fn apply_harmony(
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
        out.push_str("<|im_sep|>\n");
        out.push_str(&msg.content);
        out.push_str("\n<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant<|im_sep|>\n");
    }
    Ok(out)
}

/// Apply the Gemma 4 chat template for text-only OpenAI chat messages.
pub(crate) fn apply_gemma4(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(LLMError::TokenizerError("empty message list".into()));
    }

    let mut out = String::from("<bos>");
    let mut start = 0usize;
    if matches!(messages[0].role.as_str(), "system" | "developer") {
        out.push_str("<|turn>system\n");
        out.push_str(messages[0].content.trim());
        out.push_str("<turn|>\n");
        start = 1;
    }

    for msg in &messages[start..] {
        if msg.role == "tool" {
            continue;
        }
        let role = if msg.role == "assistant" {
            "model"
        } else {
            msg.role.as_str()
        };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(msg.content.trim());
        out.push_str("<turn|>\n");
    }

    if add_generation_prompt {
        out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    }

    Ok(out)
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
    fn chat_role_display() {
        assert_eq!(ChatRole::System.to_string(), "system");
        assert_eq!(ChatRole::User.to_string(), "user");
        assert_eq!(ChatRole::Assistant.to_string(), "assistant");
    }

    #[test]
    fn harmony_basic() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let result = apply_harmony(&msgs, true).unwrap();
        assert!(result.contains("<|im_start|>system<|im_sep|>\nYou are helpful.\n<|im_end|>"));
        assert!(result.contains("<|im_start|>user<|im_sep|>\nHello\n<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant<|im_sep|>\n"));
    }

    #[test]
    fn harmony_no_generation_prompt() {
        let msgs = vec![ChatMessage::user("Hi")];
        let result = apply_harmony(&msgs, false).unwrap();
        assert!(!result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn harmony_empty_errors() {
        assert!(apply_harmony(&[], true).is_err());
    }

    #[test]
    fn gemma4_basic() {
        let msgs = vec![ChatMessage::user("Hello")];
        let result = apply_gemma4(&msgs, true).unwrap();
        assert_eq!(
            result,
            "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn gemma4_system() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi"),
        ];
        let result = apply_gemma4(&msgs, true).unwrap();
        assert!(result.starts_with("<bos><|turn>system\nYou are helpful.<turn|>\n"));
        assert!(result.contains("<|turn>user\nHi<turn|>\n"));
        assert!(result.ends_with("<|channel>thought\n<channel|>"));
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
