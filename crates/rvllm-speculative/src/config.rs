//! Configuration for speculative decoding.

/// Configuration for draft-model-based speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Path to the draft (smaller/faster) model used for speculation.
    pub draft_model_path: String,
    /// Number of speculative tokens to generate per step (K, typically 3-5).
    pub num_speculative_tokens: usize,
    /// Threshold for modified rejection sampling acceptance.
    pub acceptance_threshold: f32,
    /// Whether speculative decoding is enabled.
    pub enabled: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_model_path: String::new(),
            num_speculative_tokens: 3,
            acceptance_threshold: 1.0,
            enabled: false,
        }
    }
}

impl SpeculativeConfig {
    pub fn new(draft_model_path: String, num_speculative_tokens: usize) -> Self {
        Self {
            draft_model_path,
            num_speculative_tokens,
            acceptance_threshold: 1.0,
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = SpeculativeConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.num_speculative_tokens, 3);
        assert_eq!(cfg.acceptance_threshold, 1.0);
        assert!(cfg.draft_model_path.is_empty());
    }

    #[test]
    fn new_config() {
        let cfg = SpeculativeConfig::new("/models/draft".into(), 5);
        assert!(cfg.enabled);
        assert_eq!(cfg.num_speculative_tokens, 5);
        assert_eq!(cfg.draft_model_path, "/models/draft");
    }
}
