//! Chat message types and simple template rendering.

use rvllm_core::prelude::{LLMError, Result, TokenId};
use tokenizers::Tokenizer as HfTokenizer;

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
    tokenizer: &HfTokenizer,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<Vec<TokenId>> {
    if messages.is_empty() {
        return Err(LLMError::TokenizerError("empty message list".into()));
    }

    let mut out = Vec::new();
    for msg in messages {
        validate_harmony_role(&msg.role)?;
        push_special_token(tokenizer, &mut out, "<|start|>")?;
        extend_encoded_text(tokenizer, &mut out, &msg.role)?;
        push_special_token(tokenizer, &mut out, "<|message|>")?;
        extend_encoded_text(tokenizer, &mut out, &msg.content)?;
        push_special_token(tokenizer, &mut out, "<|end|>")?;
    }

    if add_generation_prompt {
        push_special_token(tokenizer, &mut out, "<|start|>")?;
        extend_encoded_text(tokenizer, &mut out, ChatRole::Assistant.as_str())?;
    }

    Ok(out)
}

fn validate_harmony_role(role: &str) -> Result<()> {
    match role {
        "system" | "user" | "assistant" | "developer" | "tool" => Ok(()),
        other => Err(LLMError::TokenizerError(format!(
            "unsupported Harmony chat role '{}'",
            other
        ))),
    }
}

fn push_special_token(
    tokenizer: &HfTokenizer,
    out: &mut Vec<TokenId>,
    token: &str,
) -> Result<()> {
    let token_id = tokenizer.token_to_id(token).ok_or_else(|| {
        LLMError::TokenizerError(format!("missing Harmony special token '{}'", token))
    })?;
    out.push(token_id);
    Ok(())
}

fn extend_encoded_text(tokenizer: &HfTokenizer, out: &mut Vec<TokenId>, text: &str) -> Result<()> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| LLMError::TokenizerError(format!("Harmony encode failed: {e}")))?;
    out.extend_from_slice(encoding.get_ids());
    Ok(())
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
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::AddedToken;

        let mut vocab = std::collections::HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("system".to_string(), 1);
        vocab.insert("user".to_string(), 2);
        vocab.insert("assistant".to_string(), 3);
        vocab.insert("You".to_string(), 4);
        vocab.insert("are".to_string(), 5);
        vocab.insert("helpful.".to_string(), 6);
        vocab.insert("Hello".to_string(), 7);

        let wp = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = HfTokenizer::new(wp);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|start|>", true),
            AddedToken::from("<|message|>", true),
            AddedToken::from("<|end|>", true),
        ]);

        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let result = apply_harmony(&tokenizer, &msgs, true).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0], tokenizer.token_to_id("<|start|>").unwrap());
    }

    #[test]
    fn harmony_rejects_unknown_roles() {
        use tokenizers::models::wordpiece::WordPiece;

        let wp = WordPiece::builder()
            .vocab(std::collections::HashMap::from([("[UNK]".to_string(), 0)]))
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let tokenizer = HfTokenizer::new(wp);
        let result = apply_harmony(&tokenizer, &[ChatMessage::new("invalid", "hi")], false);
        assert!(result.is_err());
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
