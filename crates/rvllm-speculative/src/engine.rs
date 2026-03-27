//! Speculative decoding engine wrapping AsyncLLMEngine with draft+verify loop.

use std::collections::VecDeque;

use rvllm_core::prelude::{LLMError, RequestOutput, Result, TokenId};

use crate::config::SpeculativeConfig;
use crate::draft::DraftModelRunner;
use crate::verification::{verify_tokens, VerificationResult};

/// Metrics tracked by the speculative engine.
#[derive(Debug, Clone)]
pub struct SpeculativeMetrics {
    /// Total draft tokens proposed.
    pub total_draft_tokens: u64,
    /// Total draft tokens accepted by the target model.
    pub total_accepted_tokens: u64,
    /// Total bonus tokens generated.
    pub total_bonus_tokens: u64,
    /// Total verification steps executed.
    pub total_steps: u64,
}

impl SpeculativeMetrics {
    pub fn new() -> Self {
        Self {
            total_draft_tokens: 0,
            total_accepted_tokens: 0,
            total_bonus_tokens: 0,
            total_steps: 0,
        }
    }

    /// Rolling acceptance rate: accepted / proposed.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_accepted_tokens as f64 / self.total_draft_tokens as f64
    }

    /// Average tokens produced per step (accepted + bonus) vs 1 without speculation.
    pub fn speedup_ratio(&self) -> f64 {
        if self.total_steps == 0 {
            return 1.0;
        }
        let total_produced = self.total_accepted_tokens + self.total_bonus_tokens;
        total_produced as f64 / self.total_steps as f64
    }
}

impl Default for SpeculativeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Wraps an AsyncLLMEngine with a draft+verify speculative decoding loop.
///
/// The engine type is generic to avoid hard coupling to a concrete engine
/// implementation during development.
pub struct SpeculativeEngine<E> {
    config: SpeculativeConfig,
    draft_runner: DraftModelRunner,
    engine: E,
    metrics: SpeculativeMetrics,
    /// Buffer for outputs from completed verification steps.
    #[allow(dead_code)]
    pending_outputs: VecDeque<RequestOutput>,
}

impl<E> SpeculativeEngine<E> {
    /// Create a new speculative engine wrapping the given target engine.
    pub fn new(config: SpeculativeConfig, engine: E) -> Result<Self> {
        let draft_runner = DraftModelRunner::new(config.clone())?;
        tracing::info!(
            k = config.num_speculative_tokens,
            threshold = config.acceptance_threshold,
            "speculative engine initialized"
        );
        Ok(Self {
            config,
            draft_runner,
            engine,
            metrics: SpeculativeMetrics::new(),
            pending_outputs: VecDeque::new(),
        })
    }

    /// Execute one speculative decode step:
    /// 1. Generate K draft tokens
    /// 2. Run target model on all K+1 positions
    /// 3. Verify and accept/reject
    ///
    /// Returns accepted outputs. The actual target model forward pass is
    /// delegated to the wrapped engine; this method orchestrates the loop.
    pub fn step_with_probs(
        &mut self,
        input_tokens: &[TokenId],
        target_probs_fn: impl Fn(&[TokenId]) -> Vec<Vec<f32>>,
    ) -> Result<VerificationResult> {
        if !self.config.enabled {
            return Err(LLMError::ConfigError(
                "speculative decoding is not enabled".into(),
            ));
        }

        let k = self.config.num_speculative_tokens;

        // 1. Generate draft tokens
        let drafts = self.draft_runner.generate_draft_tokens(input_tokens, k)?;
        let draft_token_ids: Vec<TokenId> = drafts.iter().map(|d| d.token_id).collect();
        let draft_probs: Vec<Vec<f32>> = drafts.iter().map(|d| d.draft_probs.clone()).collect();

        // 2. Build verification input and get target probabilities
        let mut verify_input = input_tokens.to_vec();
        verify_input.extend_from_slice(&draft_token_ids);
        let target_probs = target_probs_fn(&verify_input);

        // 3. Verify
        let result = verify_tokens(&draft_probs, &target_probs, &draft_token_ids);

        // Update metrics
        self.metrics.total_draft_tokens += k as u64;
        self.metrics.total_accepted_tokens += result.num_accepted as u64;
        if result.bonus_token.is_some() {
            self.metrics.total_bonus_tokens += 1;
        }
        self.metrics.total_steps += 1;

        tracing::debug!(
            accepted = result.num_accepted,
            bonus = result.bonus_token.is_some(),
            rate = %self.metrics.acceptance_rate(),
            "verification complete"
        );

        Ok(result)
    }

    pub fn metrics(&self) -> &SpeculativeMetrics {
        &self.metrics
    }

    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SpeculativeConfig {
        SpeculativeConfig::new("/models/draft".into(), 3)
    }

    /// Dummy engine for testing.
    struct MockEngine;

    #[test]
    fn metrics_default() {
        let m = SpeculativeMetrics::new();
        assert_eq!(m.acceptance_rate(), 0.0);
        assert_eq!(m.speedup_ratio(), 1.0);
    }

    #[test]
    fn metrics_tracking() {
        let mut m = SpeculativeMetrics::new();
        m.total_draft_tokens = 10;
        m.total_accepted_tokens = 7;
        m.total_bonus_tokens = 3;
        m.total_steps = 2;
        assert!((m.acceptance_rate() - 0.7).abs() < 1e-6);
        // speedup = (7+3)/2 = 5.0
        assert!((m.speedup_ratio() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn engine_disabled_errors() {
        let mut cfg = test_config();
        cfg.enabled = false;
        let mut engine = SpeculativeEngine::new(cfg, MockEngine).unwrap();
        let result = engine.step_with_probs(&[1, 2, 3], |_| vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn engine_step_self_speculation() {
        let cfg = test_config();
        let k = cfg.num_speculative_tokens;
        let mut engine = SpeculativeEngine::new(cfg, MockEngine).unwrap();

        // Target returns same probs as draft (self-speculation)
        let draft_runner_ref = &engine.draft_runner;
        let vocab_size = draft_runner_ref.vocab_size();

        let result = engine.step_with_probs(&[100, 200], |_input| {
            // Reconstruct what the draft model would produce
            let mut probs_vec = Vec::new();
            let mut ctx = vec![100u32, 200];
            for i in 0..3 {
                let mut probs = vec![0.0f32; vocab_size];
                let last = *ctx.last().unwrap();
                let selected = ((last as usize + i + 1) % vocab_size) as u32;
                probs[selected as usize] = 1.0;
                ctx.push(selected);
                probs_vec.push(probs);
            }
            probs_vec
        }).unwrap();

        assert_eq!(result.num_accepted, k);
        assert_eq!(engine.metrics().total_steps, 1);
        assert_eq!(engine.metrics().total_draft_tokens, k as u64);
        assert_eq!(engine.metrics().total_accepted_tokens, k as u64);
        assert!((engine.metrics().acceptance_rate() - 1.0).abs() < 1e-6);
    }
}
