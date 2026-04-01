//! Numerical primitives: softmax, log-softmax, top-logprobs extraction.
//! All hot paths use Zig SIMD (rvllm_zig).

use rvllm_core::prelude::TokenId;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// -- softmax ----------------------------------------------------------------

/// Numerically stable softmax over logits.
#[inline]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; logits.len()];
    softmax_into(logits, &mut out);
    out
}

/// Softmax writing into a caller-provided buffer.
#[inline]
pub fn softmax_into(logits: &[f32], out: &mut [f32]) {
    debug_assert!(out.len() >= logits.len());
    if logits.is_empty() {
        return;
    }
    rvllm_zig::softmax_into(logits, out);
}

// -- log_softmax ------------------------------------------------------------

/// Numerically stable log-softmax over logits.
#[inline]
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; logits.len()];
    log_softmax_into(logits, &mut out);
    out
}

/// Log-softmax writing into a caller-provided buffer.
#[inline]
pub fn log_softmax_into(logits: &[f32], out: &mut [f32]) {
    debug_assert!(out.len() >= logits.len());
    if logits.is_empty() {
        return;
    }
    rvllm_zig::log_softmax_into(logits, out);
}

// -- greedy_sample (argmax) -------------------------------------------------

/// Greedy (argmax) sampling: pick the token with the highest logit.
#[inline]
pub fn greedy_sample(logits: &[f32]) -> TokenId {
    if logits.is_empty() {
        return 0;
    }
    rvllm_zig::argmax(logits)
}

// -- multinomial_sample -----------------------------------------------------

/// Multinomial sampling: draw one token from a probability distribution.
#[inline]
pub fn multinomial_sample(probs: &[f32], rng: &mut impl rand::Rng) -> TokenId {
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as TokenId;
        }
    }
    (probs.len().saturating_sub(1)) as TokenId
}

// -- top_logprobs -----------------------------------------------------------

#[derive(PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Return the top-n (token_id, log_prob) pairs sorted by descending log-prob.
#[inline]
pub fn top_logprobs(logits: &[f32], n: usize) -> Vec<(TokenId, f32)> {
    if n == 0 || logits.is_empty() {
        return Vec::new();
    }

    let mut log_probs = vec![0.0f32; logits.len()];
    log_softmax_into(logits, &mut log_probs);

    let actual_n = n.min(logits.len());

    let mut heap: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(actual_n + 1);

    for (i, &lp) in log_probs.iter().enumerate() {
        if heap.len() < actual_n {
            heap.push(Reverse((OrdF32(lp), i as u32)));
        } else if let Some(&Reverse((OrdF32(min_val), _))) = heap.peek() {
            if lp > min_val {
                heap.pop();
                heap.push(Reverse((OrdF32(lp), i as u32)));
            }
        }
    }

    let mut result: Vec<(TokenId, f32)> = heap
        .into_iter()
        .map(|Reverse((OrdF32(lp), id))| (id, lp))
        .collect();
    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_uniform() {
        let logits = vec![0.0; 4];
        let probs = softmax(&logits);
        for p in &probs {
            assert!((*p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_large_values() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn log_softmax_matches_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let lsm = log_softmax(&logits);
        let sm = softmax(&logits);
        for (lp, p) in lsm.iter().zip(sm.iter()) {
            assert!((lp.exp() - p).abs() < 1e-6);
        }
    }

    #[test]
    fn log_softmax_empty() {
        assert!(log_softmax(&[]).is_empty());
    }

    #[test]
    fn greedy_sample_picks_max() {
        let logits = vec![0.1, 0.3, 0.9, 0.2];
        assert_eq!(greedy_sample(&logits), 2);
    }

    #[test]
    fn greedy_sample_single() {
        assert_eq!(greedy_sample(&[5.0]), 0);
    }

    #[test]
    fn multinomial_sample_deterministic() {
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..10 {
            assert_eq!(multinomial_sample(&probs, &mut rng), 2);
        }
    }

    #[test]
    fn multinomial_sample_distribution() {
        let probs = vec![0.5, 0.5];
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut counts = [0u32; 2];
        let n = 10_000;
        for _ in 0..n {
            let t = multinomial_sample(&probs, &mut rng) as usize;
            counts[t] += 1;
        }
        for c in &counts {
            assert!((*c as f32 / n as f32 - 0.5).abs() < 0.05);
        }
    }

    #[test]
    fn top_logprobs_basic() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let top = top_logprobs(&logits, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 2);
    }

    #[test]
    fn top_logprobs_n_larger_than_vocab() {
        let logits = vec![1.0, 2.0];
        let top = top_logprobs(&logits, 5);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn top_logprobs_zero() {
        let logits = vec![1.0, 2.0, 3.0];
        let top = top_logprobs(&logits, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn softmax_into_matches_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected = softmax(&logits);
        let mut out = vec![0.0f32; logits.len()];
        softmax_into(&logits, &mut out);
        for (a, b) in expected.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn log_softmax_into_matches_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected = log_softmax(&logits);
        let mut out = vec![0.0f32; logits.len()];
        log_softmax_into(&logits, &mut out);
        for (a, b) in expected.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn greedy_sample_large() {
        let mut logits = vec![0.0f32; 1024];
        logits[777] = 99.0;
        assert_eq!(greedy_sample(&logits), 777);
    }

    #[test]
    fn top_logprobs_large() {
        let mut logits = vec![0.0f32; 1024];
        logits[100] = 10.0;
        logits[200] = 9.0;
        logits[300] = 8.0;
        let top = top_logprobs(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 100);
        assert_eq!(top[1].0, 200);
        assert_eq!(top[2].0, 300);
    }
}
