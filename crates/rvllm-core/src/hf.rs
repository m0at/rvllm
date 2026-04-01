//! Hugging Face auth helpers shared across crates.

/// Preferred Hugging Face auth token environment variable.
pub const HF_TOKEN_ENV: &str = "HF_TOKEN";
/// Legacy Hugging Face auth token environment variable still seen in the wild.
pub const LEGACY_HF_TOKEN_ENV: &str = "HUGGING_FACE_HUB_TOKEN";

const GATED_MODEL_HINT: &str =
    "If this is a gated Hugging Face model (for example Meta Llama), run `hf auth login` or set `HF_TOKEN`.";

fn first_non_empty_token(tokens: impl IntoIterator<Item = Option<String>>) -> Option<String> {
    tokens
        .into_iter()
        .flatten()
        .map(|token| token.trim().to_string())
        .find(|token| !token.is_empty())
}

/// Return the first non-empty Hugging Face auth token found in the environment.
pub fn hf_token_from_env() -> Option<String> {
    first_non_empty_token(
        [HF_TOKEN_ENV, LEGACY_HF_TOKEN_ENV]
            .into_iter()
            .map(|name| std::env::var(name).ok()),
    )
}

/// Return a user-facing auth hint when an error message looks like a gated/private
/// Hugging Face access failure.
pub fn hf_auth_hint(error: &str) -> Option<&'static str> {
    let error = error.to_ascii_lowercase();
    let auth_markers = [
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "authentication",
        "authorization",
        "gated",
    ];

    auth_markers
        .iter()
        .any(|marker| error.contains(marker))
        .then_some(GATED_MODEL_HINT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_hint_detects_gated_errors() {
        assert!(hf_auth_hint("HTTP status client error (401 Unauthorized)").is_some());
        assert!(hf_auth_hint("403 Forbidden for gated repo").is_some());
    }

    #[test]
    fn auth_hint_ignores_unrelated_errors() {
        assert!(hf_auth_hint("failed to parse config.json").is_none());
    }

    #[test]
    fn skips_empty_primary_token_env() {
        assert_eq!(
            first_non_empty_token([Some("".into()), Some("legacy-token".into())]).as_deref(),
            Some("legacy-token")
        );
    }
}
