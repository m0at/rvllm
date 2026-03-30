//! Speculative decoding engine wrapping a target model with draft+verify loop.
//!
//! The `SpeculativeEngine` orchestrates the full speculative decoding algorithm:
//!   1. Draft phase: generate K tokens with the draft model
//!   2. Verify phase: run the target model on all K candidates (single forward pass)
//!   3. Accept/reject via modified rejection sampling
//!   4. Emit accepted tokens + bonus token, advance sequence state
//!
//! The target model is abstracted via the `TargetModel` trait so the engine
//! is decoupled from GPU execution details.

use std::collections::VecDeque;

use rvllm_core::prelude::{LLMError, RequestOutput, Result, TokenId};

use crate::config::SpeculativeConfig;
use crate::draft::{DraftModel, DraftModelRunner};
use crate::verification::{verify_tokens, VerificationResult};

/// Trait abstracting the target (large) model's forward pass.
///
/// The speculative engine calls `forward_verify` with the full token sequence
/// (context + K draft tokens) and expects back K+1 probability distributions:
///   - positions 0..K: target probabilities at each draft token position
///   - position K: target probability for the next token after all K drafts
///
/// Implementations should run a single batched forward pass treating the K
/// draft tokens as a prefill chunk (all positions computed in parallel).
pub trait TargetModel: Send {
    /// Run the target model on `tokens` and return probability distributions
    /// for the last `num_verify_positions` positions.
    ///
    /// `tokens` = original context ++ K draft tokens.
    /// `num_verify_positions` = K + 1 (K draft positions + 1 bonus position).
    ///
    /// Returns `num_verify_positions` probability vectors, one per position.
    fn forward_verify(
        &mut self,
        tokens: &[TokenId],
        num_verify_positions: usize,
    ) -> Result<Vec<Vec<f32>>>;

    /// Vocabulary size of the target model.
    fn vocab_size(&self) -> usize;
}

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

/// Output of a single speculative decode step.
#[derive(Debug, Clone)]
pub struct SpecStepOutput {
    /// All tokens produced this step: accepted draft tokens + bonus token.
    pub tokens: Vec<TokenId>,
    /// How many of the K draft tokens were accepted.
    pub num_accepted: usize,
    /// The bonus token (always present unless input was empty).
    pub bonus_token: Option<TokenId>,
    /// The underlying verification result.
    pub verification: VerificationResult,
}

/// Wraps a target model with a draft+verify speculative decoding loop.
///
/// Generic over `T: TargetModel` for the large model and uses the `DraftModel`
/// trait for the draft model. A closure-based `step_with_probs` is also
/// available for testing without a full TargetModel implementation.
pub struct SpeculativeEngine<T> {
    config: SpeculativeConfig,
    draft: Box<dyn DraftModel>,
    target: T,
    metrics: SpeculativeMetrics,
    #[allow(dead_code)]
    pending_outputs: VecDeque<RequestOutput>,
}

impl<T: TargetModel> SpeculativeEngine<T> {
    /// Create a new speculative engine.
    pub fn with_draft(
        config: SpeculativeConfig,
        target: T,
        draft: Box<dyn DraftModel>,
    ) -> Result<Self> {
        tracing::info!(
            k = config.num_speculative_tokens,
            threshold = config.acceptance_threshold,
            "speculative engine initialized"
        );
        Ok(Self {
            config,
            draft,
            target,
            metrics: SpeculativeMetrics::new(),
            pending_outputs: VecDeque::new(),
        })
    }

    /// Execute one speculative decode step using the TargetModel trait.
    ///
    /// 1. Generate K draft tokens from the draft model
    /// 2. Build verification input: context + K draft tokens
    /// 3. Run target model forward pass for K+1 positions
    /// 4. Verify via speculative sampling
    /// 5. Return accepted tokens + bonus
    pub fn step(&mut self, context: &[TokenId]) -> Result<SpecStepOutput> {
        if !self.config.enabled {
            return Err(LLMError::ConfigError(
                "speculative decoding is not enabled".into(),
            ));
        }

        let k = self.config.num_speculative_tokens;

        // 1. Draft phase
        let drafts = self.draft.generate(context, k)?;
        let draft_token_ids: Vec<TokenId> = drafts.iter().map(|d| d.token_id).collect();
        let draft_probs: Vec<Vec<f32>> = drafts.iter().map(|d| d.draft_probs.clone()).collect();

        // 2. Build verification input: context + K draft tokens
        let mut verify_input = context.to_vec();
        verify_input.extend_from_slice(&draft_token_ids);

        // 3. Target forward pass: get K+1 probability distributions
        let target_probs = self.target.forward_verify(&verify_input, k + 1)?;

        if target_probs.len() != k + 1 {
            return Err(LLMError::ModelError(format!(
                "target model returned {} distributions, expected {}",
                target_probs.len(),
                k + 1
            )));
        }

        // 4. Verify
        let result = verify_tokens(&draft_probs, &target_probs, &draft_token_ids);

        // 5. Build output: accepted tokens + bonus
        let mut tokens = result.accepted_tokens.clone();
        if let Some(bonus) = result.bonus_token {
            tokens.push(bonus);
        }

        // Update metrics
        self.metrics.total_draft_tokens += k as u64;
        self.metrics.total_accepted_tokens += result.num_accepted as u64;
        if result.bonus_token.is_some() {
            self.metrics.total_bonus_tokens += 1;
        }
        self.metrics.total_steps += 1;

        tracing::debug!(
            accepted = result.num_accepted,
            total_tokens = tokens.len(),
            bonus = result.bonus_token.is_some(),
            rate = %self.metrics.acceptance_rate(),
            speedup = %self.metrics.speedup_ratio(),
            "speculative step complete"
        );

        Ok(SpecStepOutput {
            num_accepted: result.num_accepted,
            bonus_token: result.bonus_token,
            verification: result,
            tokens,
        })
    }

    /// Run speculative decoding loop until `max_tokens` are generated or
    /// `is_done` returns true.
    ///
    /// Each iteration does one speculative step (producing 1..K+1 tokens),
    /// appends accepted tokens to the growing context, and checks termination.
    pub fn generate(
        &mut self,
        prompt: &[TokenId],
        max_tokens: usize,
        is_done: impl Fn(TokenId) -> bool,
    ) -> Result<Vec<TokenId>> {
        if !self.config.enabled {
            return Err(LLMError::ConfigError(
                "speculative decoding is not enabled".into(),
            ));
        }

        let mut context = prompt.to_vec();
        let mut output_tokens = Vec::new();

        while output_tokens.len() < max_tokens {
            let step = self.step(&context)?;

            for &tok in &step.tokens {
                if output_tokens.len() >= max_tokens {
                    break;
                }
                output_tokens.push(tok);
                context.push(tok);

                if is_done(tok) {
                    return Ok(output_tokens);
                }
            }

            // If step produced no tokens, something is wrong
            if step.tokens.is_empty() {
                break;
            }
        }

        Ok(output_tokens)
    }

    pub fn metrics(&self) -> &SpeculativeMetrics {
        &self.metrics
    }

    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    pub fn target(&self) -> &T {
        &self.target
    }

    pub fn target_mut(&mut self) -> &mut T {
        &mut self.target
    }
}

/// Convenience constructor using the default DraftModelRunner.
impl<T: TargetModel> SpeculativeEngine<T> {
    pub fn new(config: SpeculativeConfig, target: T) -> Result<Self> {
        let draft = Box::new(DraftModelRunner::new(config.clone())?);
        Self::with_draft(config, target, draft)
    }
}

/// Closure-based step for testing (no TargetModel required).
///
/// This is separate from the trait-based engine and kept for backward
/// compatibility with the original test harness.
pub fn speculative_step_with_probs(
    draft: &dyn DraftModel,
    input_tokens: &[TokenId],
    k: usize,
    target_probs_fn: impl Fn(&[TokenId]) -> Vec<Vec<f32>>,
) -> Result<VerificationResult> {
    let drafts = draft.generate(input_tokens, k)?;
    let draft_token_ids: Vec<TokenId> = drafts.iter().map(|d| d.token_id).collect();
    let draft_probs: Vec<Vec<f32>> = drafts.iter().map(|d| d.draft_probs.clone()).collect();

    let mut verify_input = input_tokens.to_vec();
    verify_input.extend_from_slice(&draft_token_ids);
    let target_probs = target_probs_fn(&verify_input);

    Ok(verify_tokens(&draft_probs, &target_probs, &draft_token_ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpeculativeConfig;

    fn test_config() -> SpeculativeConfig {
        SpeculativeConfig::new("/models/draft".into(), 3)
    }

    /// Mock target model that returns the same distributions as the draft
    /// (self-speculation), guaranteeing 100% acceptance.
    struct SelfSpecTarget {
        vocab_size: usize,
    }

    impl TargetModel for SelfSpecTarget {
        fn forward_verify(
            &mut self,
            tokens: &[TokenId],
            num_verify_positions: usize,
        ) -> Result<Vec<Vec<f32>>> {
            // Reconstruct what the draft model would produce: the draft model
            // is deterministic based on (last_context_token + position_index + 1) % vocab.
            // The verification positions start after the original context.
            let context_len = tokens.len() - (num_verify_positions - 1);
            let mut probs_vec = Vec::with_capacity(num_verify_positions);
            let mut ctx: Vec<TokenId> = tokens[..context_len].to_vec();

            for i in 0..num_verify_positions {
                let mut probs = vec![0.0f32; self.vocab_size];
                let last = *ctx.last().unwrap_or(&0);
                let selected = ((last as usize + i + 1) % self.vocab_size) as TokenId;
                probs[selected as usize] = 1.0;
                ctx.push(selected);
                probs_vec.push(probs);
            }
            Ok(probs_vec)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    /// Mock target that disagrees with draft at every position.
    struct DisagreeTarget {
        vocab_size: usize,
    }

    impl TargetModel for DisagreeTarget {
        fn forward_verify(
            &mut self,
            _tokens: &[TokenId],
            num_verify_positions: usize,
        ) -> Result<Vec<Vec<f32>>> {
            // Always put mass on token 0 (draft never picks token 0 for our test inputs)
            let mut probs_vec = Vec::with_capacity(num_verify_positions);
            for _ in 0..num_verify_positions {
                let mut probs = vec![0.0f32; self.vocab_size];
                probs[0] = 1.0;
                probs_vec.push(probs);
            }
            Ok(probs_vec)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

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
        assert!((m.speedup_ratio() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn engine_disabled_errors() {
        let mut cfg = test_config();
        cfg.enabled = false;
        let target = SelfSpecTarget { vocab_size: 32000 };
        let mut engine = SpeculativeEngine::new(cfg, target).unwrap();
        assert!(engine.step(&[1, 2, 3]).is_err());
    }

    #[test]
    fn engine_self_speculation_accepts_all() {
        let cfg = test_config();
        let k = cfg.num_speculative_tokens;
        let target = SelfSpecTarget { vocab_size: 32000 };
        let mut engine = SpeculativeEngine::new(cfg, target).unwrap();

        let result = engine.step(&[100, 200]).unwrap();

        assert_eq!(result.num_accepted, k);
        // K accepted + 1 bonus = K+1 tokens
        assert_eq!(result.tokens.len(), k + 1);
        assert!(result.bonus_token.is_some());
        assert_eq!(engine.metrics().total_steps, 1);
        assert_eq!(engine.metrics().total_draft_tokens, k as u64);
        assert_eq!(engine.metrics().total_accepted_tokens, k as u64);
        assert!((engine.metrics().acceptance_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn engine_disagree_rejects_all() {
        let cfg = test_config();
        let target = DisagreeTarget { vocab_size: 32000 };
        let mut engine = SpeculativeEngine::new(cfg, target).unwrap();

        let result = engine.step(&[100, 200]).unwrap();

        // All drafts rejected, but we still get 1 bonus token
        assert_eq!(result.num_accepted, 0);
        assert_eq!(result.tokens.len(), 1);
        assert!(result.bonus_token.is_some());
        // Bonus should be token 0 (where DisagreeTarget puts all mass)
        assert_eq!(result.bonus_token.unwrap(), 0);
    }

    #[test]
    fn generate_with_eos() {
        let cfg = SpeculativeConfig::new("/models/draft".into(), 3);
        let target = SelfSpecTarget { vocab_size: 32000 };
        let mut engine = SpeculativeEngine::new(cfg, target).unwrap();

        // Use token 0 as EOS. The self-spec target won't produce 0 for
        // our inputs, so this should just hit max_tokens.
        let result = engine.generate(&[100, 200], 10, |tok| tok == 0).unwrap();

        assert!(!result.is_empty());
        assert!(result.len() <= 10);
    }

    #[test]
    fn generate_respects_max_tokens() {
        let cfg = SpeculativeConfig::new("/models/draft".into(), 5);
        let target = SelfSpecTarget { vocab_size: 32000 };
        let mut engine = SpeculativeEngine::new(cfg, target).unwrap();

        let result = engine
            .generate(&[100, 200], 8, |_| false)
            .unwrap();

        assert!(result.len() <= 8);
    }

    #[test]
    fn speedup_ratio_with_self_speculation() {
        let cfg = SpeculativeConfig::new("/models/draft".into(), 4);
        let k = cfg.num_speculative_tokens;
        let target = SelfSpecTarget { vocab_size: 32000 };
        let mut engine = SpeculativeEngine::new(cfg, target).unwrap();

        // Run a few steps
        for _ in 0..5 {
            let _ = engine.step(&[100, 200]);
        }

        // With self-speculation: every step accepts K + 1 bonus = K+1 tokens
        // speedup = (K*5 + 5) / 5 = K + 1
        let expected = (k + 1) as f64;
        assert!(
            (engine.metrics().speedup_ratio() - expected).abs() < 0.1,
            "expected speedup ~{expected}, got {}",
            engine.metrics().speedup_ratio()
        );
    }
}
