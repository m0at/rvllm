//! Stop condition checking for generated sequences.

use rvllm_core::prelude::{FinishReason, SamplingParams, TokenId};

/// Checks whether a sequence should stop generating.
pub struct StopChecker;

impl StopChecker {
    /// Check if generation should stop, returning the finish reason if so.
    ///
    /// Checks in order:
    /// 1. EOS token produced
    /// 2. Max tokens reached
    /// 3. Stop string found in decoded text
    pub fn check_stop(
        text: &str,
        output_token_ids: &[TokenId],
        params: &SamplingParams,
        eos_token_id: Option<TokenId>,
    ) -> Option<FinishReason> {
        // Check EOS token
        if let Some(eos) = eos_token_id {
            if output_token_ids.last().copied() == Some(eos) {
                return Some(FinishReason::Stop);
            }
        }

        // Check max tokens
        if output_token_ids.len() >= params.max_tokens {
            return Some(FinishReason::Length);
        }

        // Check stop strings
        for stop in &params.stop_strings {
            if text.contains(stop.as_str()) {
                return Some(FinishReason::Stop);
            }
        }

        None
    }

    /// Truncate generated text at the first occurrence of any stop string,
    /// returning the truncated text and whether truncation occurred.
    pub fn truncate_at_stop(text: &str, stop_strings: &[String]) -> (String, bool) {
        let mut earliest_pos = text.len();
        let mut found = false;

        for stop in stop_strings {
            if let Some(pos) = text.find(stop.as_str()) {
                if pos < earliest_pos {
                    earliest_pos = pos;
                    found = true;
                }
            }
        }

        if found {
            (text[..earliest_pos].to_string(), true)
        } else {
            (text.to_string(), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> SamplingParams {
        SamplingParams::default()
    }

    #[test]
    fn no_stop_when_empty() {
        let result = StopChecker::check_stop("", &[], &default_params(), None);
        assert!(result.is_none());
    }

    #[test]
    fn stop_on_eos_token() {
        let result = StopChecker::check_stop("hello", &[1, 2, 99], &default_params(), Some(99));
        assert_eq!(result, Some(FinishReason::Stop));
    }

    #[test]
    fn no_stop_without_eos() {
        let result = StopChecker::check_stop("hello", &[1, 2, 3], &default_params(), Some(99));
        assert!(result.is_none());
    }

    #[test]
    fn stop_on_max_tokens() {
        let mut params = default_params();
        params.max_tokens = 3;
        let result = StopChecker::check_stop("abc", &[1, 2, 3], &params, None);
        assert_eq!(result, Some(FinishReason::Length));
    }

    #[test]
    fn stop_on_stop_string() {
        let mut params = default_params();
        params.stop_strings = vec!["<|end|>".to_string()];
        let result =
            StopChecker::check_stop("hello world<|end|>more", &[1, 2], &params, None);
        assert_eq!(result, Some(FinishReason::Stop));
    }

    #[test]
    fn eos_takes_priority_over_max_tokens() {
        let mut params = default_params();
        params.max_tokens = 3;
        // 3 tokens with last being eos -- eos checked first
        let result = StopChecker::check_stop("abc", &[1, 2, 99], &params, Some(99));
        assert_eq!(result, Some(FinishReason::Stop));
    }

    #[test]
    fn truncate_at_stop_basic() {
        let stops = vec!["<stop>".to_string()];
        let (text, truncated) = StopChecker::truncate_at_stop("hello<stop>world", &stops);
        assert!(truncated);
        assert_eq!(text, "hello");
    }

    #[test]
    fn truncate_at_stop_no_match() {
        let stops = vec!["<stop>".to_string()];
        let (text, truncated) = StopChecker::truncate_at_stop("hello world", &stops);
        assert!(!truncated);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn truncate_picks_earliest_stop() {
        let stops = vec!["<a>".to_string(), "<b>".to_string()];
        let (text, truncated) =
            StopChecker::truncate_at_stop("prefix<b>middle<a>suffix", &stops);
        assert!(truncated);
        assert_eq!(text, "prefix");
    }

    #[test]
    fn truncate_empty_stops() {
        let stops: Vec<String> = vec![];
        let (text, truncated) = StopChecker::truncate_at_stop("hello", &stops);
        assert!(!truncated);
        assert_eq!(text, "hello");
    }

    #[test]
    fn stop_on_multiple_stop_strings() {
        let mut params = default_params();
        params.stop_strings = vec!["END".to_string(), "STOP".to_string()];
        let result = StopChecker::check_stop("text then STOP here", &[1], &params, None);
        assert_eq!(result, Some(FinishReason::Stop));
    }
}
