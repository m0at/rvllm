//! Speculative sampling verification algorithm.
//!
//! Implements the core accept/reject loop from "Fast Inference from
//! Transformers via Speculative Decoding" (Leviathan et al., 2023).

use rand::Rng;
use rvllm_core::prelude::TokenId;

/// Result of verifying draft tokens against target model probabilities.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Tokens accepted through the verification process.
    pub accepted_tokens: Vec<TokenId>,
    /// Number of draft tokens that were accepted.
    pub num_accepted: usize,
    /// Bonus token sampled from the target model at the first rejection
    /// point (or after all tokens accepted).
    pub bonus_token: Option<TokenId>,
}

/// Verify draft tokens using the speculative sampling algorithm.
///
/// For each position i:
///   - Accept draft token if target_prob[token] >= draft_prob[token]
///     (always accept, no randomness needed for this condition)
///   - Otherwise accept with probability target_prob[token] / draft_prob[token]
///   - On rejection: resample from max(0, target - draft) normalized
///   - Always produce one bonus token from the target distribution at the
///     first rejected position (or position K if all accepted)
///
/// # Arguments
/// - `draft_probs`: per-position probability distributions from the draft model
/// - `target_probs`: per-position probability distributions from the target model
/// - `draft_tokens`: the token ids chosen by the draft model at each position
///
/// # Panics
/// Panics if the three slices have different lengths or if any distribution is empty.
pub fn verify_tokens(
    draft_probs: &[Vec<f32>],
    target_probs: &[Vec<f32>],
    draft_tokens: &[TokenId],
) -> VerificationResult {
    verify_tokens_with_rng(draft_probs, target_probs, draft_tokens, &mut rand::thread_rng())
}

/// Deterministic version for testing: accepts an explicit RNG.
pub fn verify_tokens_with_rng<R: Rng>(
    draft_probs: &[Vec<f32>],
    target_probs: &[Vec<f32>],
    draft_tokens: &[TokenId],
    rng: &mut R,
) -> VerificationResult {
    assert_eq!(draft_probs.len(), target_probs.len());
    assert_eq!(draft_probs.len(), draft_tokens.len());

    let k = draft_tokens.len();
    let mut accepted_tokens = Vec::with_capacity(k);

    for i in 0..k {
        let token = draft_tokens[i] as usize;
        let dp = draft_probs[i][token];
        let tp = target_probs[i][token];

        // Modified rejection sampling: accept if r < min(1, tp/dp)
        let accept_prob = if dp <= 0.0 {
            if tp > 0.0 { 1.0 } else { 0.0 }
        } else {
            (tp / dp).min(1.0)
        };

        let r: f32 = rng.gen();
        if r < accept_prob {
            accepted_tokens.push(draft_tokens[i]);
        } else {
            // Rejection: resample from adjusted distribution max(0, target - draft)
            let bonus = sample_adjusted_distribution(
                &target_probs[i],
                &draft_probs[i],
                rng,
            );
            let num_accepted = accepted_tokens.len();
            return VerificationResult {
                accepted_tokens,
                num_accepted,
                bonus_token: Some(bonus),
            };
        }
    }

    // All K tokens accepted: sample bonus token from target at position K
    // (the target model evaluated K+1 positions, use the last one).
    // If we have target_probs for all K positions, use the last one as the
    // bonus distribution. In practice the engine provides K+1 target distributions.
    let bonus = if !target_probs.is_empty() {
        let last_target = &target_probs[k - 1];
        Some(sample_from_distribution(last_target, rng))
    } else {
        None
    };

    let num_accepted = accepted_tokens.len();
    VerificationResult {
        accepted_tokens,
        num_accepted,
        bonus_token: bonus,
    }
}

/// Sample from max(0, target_probs - draft_probs), normalized.
fn sample_adjusted_distribution<R: Rng>(
    target_probs: &[f32],
    draft_probs: &[f32],
    rng: &mut R,
) -> TokenId {
    assert_eq!(target_probs.len(), draft_probs.len());

    let adjusted: Vec<f32> = target_probs
        .iter()
        .zip(draft_probs.iter())
        .map(|(&t, &d)| (t - d).max(0.0))
        .collect();

    let sum: f32 = adjusted.iter().sum();
    if sum <= 0.0 {
        // Fallback: sample uniformly (degenerate case)
        return rng.gen_range(0..target_probs.len() as u32);
    }

    sample_from_unnormalized(&adjusted, sum, rng)
}

/// Sample a token from a probability distribution.
fn sample_from_distribution<R: Rng>(probs: &[f32], rng: &mut R) -> TokenId {
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return rng.gen_range(0..probs.len() as u32);
    }
    sample_from_unnormalized(probs, sum, rng)
}

/// Sample from an unnormalized distribution with known sum.
fn sample_from_unnormalized<R: Rng>(weights: &[f32], sum: f32, rng: &mut R) -> TokenId {
    let r: f32 = rng.gen::<f32>() * sum;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if r < cumulative {
            return i as TokenId;
        }
    }
    // Numerical edge case: return last token
    (weights.len() - 1) as TokenId
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    /// When draft == target distributions, acceptance rate should be 1.0
    /// (self-speculation).
    #[test]
    fn self_speculation_accepts_all() {
        // Run many trials to be sure
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let vocab = 10;
            let k = 5;

            // Create identical distributions for draft and target
            let mut draft_probs = Vec::new();
            let mut target_probs = Vec::new();
            let mut draft_tokens = Vec::new();

            for i in 0..k {
                let mut probs = vec![0.0f32; vocab];
                // Put all mass on one token (deterministic)
                let token = (i % vocab) as TokenId;
                probs[token as usize] = 1.0;
                draft_probs.push(probs.clone());
                target_probs.push(probs);
                draft_tokens.push(token);
            }

            let result = verify_tokens_with_rng(
                &draft_probs,
                &target_probs,
                &draft_tokens,
                &mut rng,
            );

            assert_eq!(
                result.num_accepted, k,
                "seed={seed}: expected all {k} accepted, got {}",
                result.num_accepted
            );
            assert_eq!(result.accepted_tokens, draft_tokens);
        }
    }

    /// When target has zero probability for draft tokens, all should be rejected.
    #[test]
    fn zero_target_rejects_all() {
        let mut rng = make_rng();
        let vocab = 4;
        let k = 3;

        let mut draft_probs = Vec::new();
        let mut target_probs = Vec::new();
        let mut draft_tokens = Vec::new();

        for i in 0..k {
            let draft_tok = i as TokenId;
            let mut dp = vec![0.0f32; vocab];
            dp[draft_tok as usize] = 1.0;

            // Target puts all mass on a DIFFERENT token
            let target_tok = ((i + 1) % vocab) as TokenId;
            let mut tp = vec![0.0f32; vocab];
            tp[target_tok as usize] = 1.0;

            draft_probs.push(dp);
            target_probs.push(tp);
            draft_tokens.push(draft_tok);
        }

        let result = verify_tokens_with_rng(
            &draft_probs,
            &target_probs,
            &draft_tokens,
            &mut rng,
        );

        // First token should be rejected since target_prob[draft_token] == 0
        assert_eq!(result.num_accepted, 0);
        assert!(result.accepted_tokens.is_empty());
        assert!(result.bonus_token.is_some());
        // Bonus should come from adjusted dist, which equals target dist here
        // since draft has 0 at those positions. The bonus should be token 1.
        assert_eq!(result.bonus_token.unwrap(), 1);
    }

    /// Test partial acceptance: target agrees on some but not all tokens.
    #[test]
    fn partial_acceptance() {
        let mut rng = StdRng::seed_from_u64(999);

        // Position 0: identical distributions -> accept
        // Position 1: different distributions -> may reject
        let dp0 = vec![0.0, 1.0, 0.0, 0.0];
        let tp0 = vec![0.0, 1.0, 0.0, 0.0];

        let dp1 = vec![1.0, 0.0, 0.0, 0.0]; // draft says token 0
        let tp1 = vec![0.0, 0.0, 1.0, 0.0]; // target says token 2

        let draft_probs = vec![dp0, dp1];
        let target_probs = vec![tp0, tp1];
        let draft_tokens = vec![1, 0];

        let result = verify_tokens_with_rng(
            &draft_probs,
            &target_probs,
            &draft_tokens,
            &mut rng,
        );

        // First should be accepted, second rejected
        assert_eq!(result.num_accepted, 1);
        assert_eq!(result.accepted_tokens, vec![1]);
        assert!(result.bonus_token.is_some());
        // Bonus from adjusted(target - draft) at pos 1 = max(0, [0,0,1,0] - [1,0,0,0]) = [0,0,1,0] -> token 2
        assert_eq!(result.bonus_token.unwrap(), 2);
    }

    /// Bonus token is produced even when all tokens are accepted.
    #[test]
    fn bonus_token_on_full_accept() {
        for seed in 0..50 {
            let mut rng = StdRng::seed_from_u64(seed);
            let vocab = 4;
            let k = 2;

            let mut draft_probs = Vec::new();
            let mut target_probs = Vec::new();
            let mut draft_tokens = Vec::new();

            for i in 0..k {
                let tok = i as TokenId;
                let mut probs = vec![0.0f32; vocab];
                probs[tok as usize] = 1.0;
                draft_probs.push(probs.clone());
                target_probs.push(probs);
                draft_tokens.push(tok);
            }

            let result = verify_tokens_with_rng(
                &draft_probs,
                &target_probs,
                &draft_tokens,
                &mut rng,
            );

            assert_eq!(result.num_accepted, k);
            assert!(result.bonus_token.is_some());
        }
    }

    /// Probabilistic acceptance: when target has lower probability,
    /// some tokens should still be accepted proportionally.
    #[test]
    fn probabilistic_acceptance_rate() {
        let trials = 10000;
        let mut accepted = 0;

        for seed in 0..trials {
            let mut rng = StdRng::seed_from_u64(seed);

            // Draft: token 0 with prob 0.8, token 1 with prob 0.2
            let dp = vec![0.8, 0.2];
            // Target: token 0 with prob 0.4, token 1 with prob 0.6
            let tp = vec![0.4, 0.6];

            let result = verify_tokens_with_rng(
                &[dp],
                &[tp],
                &[0], // draft chose token 0
                &mut rng,
            );

            if result.num_accepted == 1 {
                accepted += 1;
            }
        }

        // Acceptance rate should be ~0.4/0.8 = 0.5
        let rate = accepted as f64 / trials as f64;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "expected acceptance rate ~0.5, got {rate}"
        );
    }

    #[test]
    fn empty_input() {
        let mut rng = make_rng();
        let result = verify_tokens_with_rng(&[], &[], &[], &mut rng);
        assert_eq!(result.num_accepted, 0);
        assert!(result.accepted_tokens.is_empty());
        assert!(result.bonus_token.is_none());
    }
}
