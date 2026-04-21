use crate::types::Prompt;

pub fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

pub fn prompt_token_count(prompt: &Prompt) -> usize {
    estimate_tokens(&prompt.as_text())
}
