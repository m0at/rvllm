//! Core sampler: orchestrates logit processing and token selection.

use std::collections::HashMap;

use rand::Rng;
use tracing::trace;
use rvllm_core::prelude::{LLMError, Result, SamplingParams, TokenId};

use crate::logit_processors;
use crate::math;

/// Output of a single sampling step.
#[derive(Debug, Clone)]
pub struct SamplerOutput {
    /// Selected token.
    pub token_id: TokenId,
    /// Log-probability of the selected token.
    pub logprob: f32,
    /// Top log-probabilities for this position (if requested).
    pub top_logprobs: Vec<(TokenId, f32)>,
}

/// Stateless sampler that converts logits into token selections.
#[derive(Debug, Default)]
pub struct Sampler;

impl Sampler {
    pub fn new() -> Self {
        Self
    }

    /// Sample a single token from logits given sampling parameters.
    pub fn sample(
        &self,
        logits: &[f32],
        vocab_size: usize,
        params: &SamplingParams,
        past_tokens: &[TokenId],
        rng: &mut impl Rng,
    ) -> Result<SamplerOutput> {
        if logits.len() != vocab_size {
            return Err(LLMError::SamplingError(format!(
                "logits length {} != vocab_size {}",
                logits.len(),
                vocab_size
            )));
        }
        if vocab_size == 0 {
            return Err(LLMError::SamplingError("empty vocabulary".into()));
        }

        let mut work = logits.to_vec();

        // 1. Repetition penalty
        if params.repetition_penalty != 1.0 {
            logit_processors::apply_repetition_penalty(
                &mut work,
                past_tokens,
                params.repetition_penalty,
            );
        }

        // 2. Frequency / presence penalty
        if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
            let counts = build_token_counts(past_tokens);
            logit_processors::apply_frequency_presence_penalty(
                &mut work,
                &counts,
                params.frequency_penalty,
                params.presence_penalty,
            );
        }

        // Greedy: skip temperature/top-k/top-p entirely.
        let is_greedy = params.temperature == 0.0;

        if !is_greedy {
            // 3. Temperature
            logit_processors::apply_temperature(&mut work, params.temperature);

            // 4. Top-k
            logit_processors::apply_top_k(&mut work, params.top_k);

            // 5. Top-p
            logit_processors::apply_top_p(&mut work, params.top_p);

            // 6. Min-p
            logit_processors::apply_min_p(&mut work, params.min_p);
        }

        // Collect top logprobs before sampling (from original processed logits).
        let top_lp = match params.logprobs {
            Some(n) if n > 0 => math::top_logprobs(&work, n),
            _ => Vec::new(),
        };

        // Sample
        let token_id = if is_greedy {
            math::greedy_sample(&work)
        } else {
            let probs = math::softmax(&work);
            math::multinomial_sample(&probs, rng)
        };

        // Compute log-prob of selected token.
        let log_probs = math::log_softmax(&work);
        let logprob = log_probs[token_id as usize];

        trace!(token_id, logprob, "sampled token");

        Ok(SamplerOutput {
            token_id,
            logprob,
            top_logprobs: top_lp,
        })
    }
}

/// Build a token -> count map from a sequence of past tokens.
fn build_token_counts(tokens: &[TokenId]) -> HashMap<TokenId, usize> {
    let mut counts = HashMap::new();
    for &t in tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn default_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn greedy_sampling() {
        let sampler = Sampler::new();
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = default_rng();
        let out = sampler.sample(&logits, 4, &params, &[], &mut rng).unwrap();
        assert_eq!(out.token_id, 1);
    }

    #[test]
    fn deterministic_seeded_sampling() {
        let sampler = Sampler::new();
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };

        let mut rng1 = ChaCha8Rng::seed_from_u64(999);
        let mut rng2 = ChaCha8Rng::seed_from_u64(999);

        let out1 = sampler.sample(&logits, 5, &params, &[], &mut rng1).unwrap();
        let out2 = sampler.sample(&logits, 5, &params, &[], &mut rng2).unwrap();

        assert_eq!(out1.token_id, out2.token_id);
        assert_eq!(out1.logprob, out2.logprob);
    }

    #[test]
    fn different_seeds_differ() {
        let sampler = Sampler::new();
        // Uniform logits -> different seeds should (almost certainly) produce
        // different sequences over many draws.
        let logits = vec![0.0; 1000];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };

        let mut rng1 = ChaCha8Rng::seed_from_u64(1);
        let mut rng2 = ChaCha8Rng::seed_from_u64(2);

        let mut same = true;
        for _ in 0..20 {
            let o1 = sampler.sample(&logits, 1000, &params, &[], &mut rng1).unwrap();
            let o2 = sampler.sample(&logits, 1000, &params, &[], &mut rng2).unwrap();
            if o1.token_id != o2.token_id {
                same = false;
                break;
            }
        }
        assert!(!same, "different seeds should produce different outputs");
    }

    #[test]
    fn vocab_size_mismatch() {
        let sampler = Sampler::new();
        let logits = vec![1.0, 2.0];
        let params = SamplingParams::default();
        let mut rng = default_rng();
        let result = sampler.sample(&logits, 5, &params, &[], &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn empty_vocab() {
        let sampler = Sampler::new();
        let params = SamplingParams::default();
        let mut rng = default_rng();
        let result = sampler.sample(&[], 0, &params, &[], &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn logprobs_returned() {
        let sampler = Sampler::new();
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            logprobs: Some(3),
            ..Default::default()
        };
        let mut rng = default_rng();
        let out = sampler.sample(&logits, 4, &params, &[], &mut rng).unwrap();
        assert_eq!(out.top_logprobs.len(), 3);
        // First top logprob should be token 1 (highest logit).
        assert_eq!(out.top_logprobs[0].0, 1);
    }

    #[test]
    fn repetition_penalty_reduces_repeated() {
        let sampler = Sampler::new();
        // Token 1 has highest logit, but we've seen it many times.
        let logits = vec![3.0, 4.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 100.0,
            ..Default::default()
        };
        let mut rng = default_rng();
        let out = sampler
            .sample(&logits, 3, &params, &[1, 1, 1], &mut rng)
            .unwrap();
        // With heavy penalty, token 1 should no longer win.
        assert_ne!(out.token_id, 1);
    }

    #[test]
    fn top_k_constrains_output() {
        let sampler = Sampler::new();
        // Only token 2 (logit=10) should survive top_k=1.
        let logits = vec![1.0, 1.0, 10.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        let mut rng = default_rng();
        for _ in 0..20 {
            let out = sampler.sample(&logits, 4, &params, &[], &mut rng).unwrap();
            assert_eq!(out.token_id, 2);
        }
    }

    #[test]
    fn top_p_constrains_output() {
        let sampler = Sampler::new();
        // Token 2 has overwhelming probability with logit=20.
        let logits = vec![0.0, 0.0, 20.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        };
        let mut rng = default_rng();
        for _ in 0..20 {
            let out = sampler.sample(&logits, 4, &params, &[], &mut rng).unwrap();
            assert_eq!(out.token_id, 2);
        }
    }
}
