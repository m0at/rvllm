//! In-place logit processors: temperature, top-k, top-p, min-p,
//! repetition penalty, and frequency/presence penalty.

use rvllm_core::prelude::TokenId;
use std::collections::HashMap;

/// Scale logits by temperature. temperature=0 is handled upstream as greedy.
#[inline]
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || temperature == 1.0 {
        return;
    }
    rvllm_zig::temperature_scale(logits, 1.0 / temperature);
}

/// Keep only the top-k highest logits; set the rest to -inf.
/// k=0 means no filtering.
/// Uses O(n) quickselect on indices to find threshold without cloning logits.
#[inline]
pub fn apply_top_k(logits: &mut [f32], k: u32) {
    let k = k as usize;
    if k == 0 || k >= logits.len() {
        return;
    }
    // Quickselect on index array to find the k-th largest value.
    let mut indices: Vec<u32> = (0..logits.len() as u32).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[indices[k - 1] as usize];

    let count_above = logits.iter().filter(|&&v| v >= threshold).count();

    if count_above <= k {
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    } else {
        let mut kept = 0;
        for l in logits.iter_mut() {
            if *l > threshold {
                kept += 1;
            } else if *l == threshold && kept < k {
                kept += 1;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

/// Nucleus (top-p) sampling filter. Sort by descending probability, accumulate
/// until the cumulative probability exceeds p, then mask out the rest.
/// Uses index sort to avoid allocating a separate (index, prob) vec.
#[inline]
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 || p <= 0.0 {
        return;
    }
    let n = logits.len();
    let probs = crate::math::softmax(logits);

    // Sort indices by descending probability.
    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Walk sorted indices, accumulate until we hit p.
    let mut cumulative = 0.0f32;
    let mut cutoff = n; // everything past this index in sorted order gets masked
    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx as usize];
        if cumulative >= p {
            cutoff = rank + 1;
            break;
        }
    }

    // Mask out everything not in the top-p nucleus.
    for &idx in &indices[cutoff..] {
        logits[idx as usize] = f32::NEG_INFINITY;
    }
}

/// Min-p filter: discard tokens whose probability is below min_p * max_prob.
#[inline]
pub fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || min_p >= 1.0 {
        return;
    }
    let probs = crate::math::softmax(logits);
    let max_prob = probs.iter().cloned().fold(0.0_f32, f32::max);
    let threshold = min_p * max_prob;
    for (i, l) in logits.iter_mut().enumerate() {
        if probs[i] < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Repetition penalty: divide logits of previously-seen tokens by penalty (if
/// positive) or multiply (if negative). penalty=1.0 means no change.
#[inline]
pub fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[TokenId], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &tok in past_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Frequency and presence penalty applied to token counts.
///   logit -= freq_penalty * count + presence_penalty * (count > 0)
#[inline]
pub fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    token_counts: &HashMap<TokenId, usize>,
    freq_penalty: f32,
    presence_penalty: f32,
) {
    if freq_penalty == 0.0 && presence_penalty == 0.0 {
        return;
    }
    for (&tok, &count) in token_counts {
        let idx = tok as usize;
        if idx < logits.len() && count > 0 {
            logits[idx] -= freq_penalty * count as f32 + presence_penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- temperature ---

    #[test]
    fn temperature_identity() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn temperature_zero_no_change() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_temperature(&mut logits, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn temperature_half_doubles_logits() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 0.5);
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_two_halves_logits() {
        let mut logits = vec![2.0, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    // --- top-k ---

    #[test]
    fn top_k_zero_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_k(&mut logits, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn top_k_larger_than_vocab_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_k(&mut logits, 10);
        assert_eq!(logits, original);
    }

    #[test]
    fn top_k_filters_correctly() {
        let mut logits = vec![1.0, 4.0, 2.0, 3.0];
        apply_top_k(&mut logits, 2);
        assert_eq!(logits[0], f32::NEG_INFINITY); // token 0 (1.0) removed
        assert_eq!(logits[1], 4.0); // kept
        assert_eq!(logits[2], f32::NEG_INFINITY); // token 2 (2.0) removed
        assert_eq!(logits[3], 3.0); // kept
    }

    #[test]
    fn top_k_one() {
        let mut logits = vec![1.0, 5.0, 3.0];
        apply_top_k(&mut logits, 1);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    // --- top-p ---

    #[test]
    fn top_p_one_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn top_p_filters() {
        // softmax([0, 0, 10]) ~= [~0, ~0, ~1.0]
        let mut logits = vec![0.0, 0.0, 10.0];
        apply_top_p(&mut logits, 0.5);
        // Token 2 dominates, should be the only one kept.
        assert!(logits[0] == f32::NEG_INFINITY);
        assert!(logits[1] == f32::NEG_INFINITY);
        assert!(logits[2] > 0.0);
    }

    #[test]
    fn top_p_keeps_at_least_one() {
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        apply_top_p(&mut logits, 0.01);
        let kept = logits.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        assert!(kept >= 1);
    }

    // --- min-p ---

    #[test]
    fn min_p_zero_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_min_p(&mut logits, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn min_p_filters_low() {
        // softmax([0, 0, 10]) -> token 2 dominates.
        // min_p = 0.5 means threshold = 0.5 * max_prob ~ 0.5.
        // Tokens 0 and 1 have prob near 0, so they get filtered.
        let mut logits = vec![0.0, 0.0, 10.0];
        apply_min_p(&mut logits, 0.5);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert!(logits[2] > 0.0);
    }

    // --- repetition penalty ---

    #[test]
    fn repetition_penalty_identity() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn repetition_penalty_positive_logits() {
        let mut logits = vec![2.0, 4.0, 6.0];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!((logits[0] - 2.0).abs() < 1e-6); // unchanged
        assert!((logits[1] - 2.0).abs() < 1e-6); // 4.0 / 2.0
        assert!((logits[2] - 6.0).abs() < 1e-6); // unchanged
    }

    #[test]
    fn repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, -4.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert!((logits[0] - (-4.0)).abs() < 1e-6); // -2.0 * 2.0
        assert!((logits[1] - (-8.0)).abs() < 1e-6); // -4.0 * 2.0
        assert!((logits[2] - 1.0).abs() < 1e-6); // unchanged
    }

    #[test]
    fn repetition_penalty_out_of_range_token() {
        let mut logits = vec![1.0, 2.0];
        apply_repetition_penalty(&mut logits, &[999], 2.0);
        assert_eq!(logits, vec![1.0, 2.0]);
    }

    // --- frequency / presence penalty ---

    #[test]
    fn freq_pres_penalty_zero_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        let counts = HashMap::from([(0, 5), (1, 3)]);
        apply_frequency_presence_penalty(&mut logits, &counts, 0.0, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn frequency_penalty_applies() {
        let mut logits = vec![5.0, 5.0, 5.0];
        let counts = HashMap::from([(0, 3_usize), (2, 1_usize)]);
        apply_frequency_presence_penalty(&mut logits, &counts, 1.0, 0.0);
        assert!((logits[0] - 2.0).abs() < 1e-6); // 5 - 1*3
        assert!((logits[1] - 5.0).abs() < 1e-6); // unchanged
        assert!((logits[2] - 4.0).abs() < 1e-6); // 5 - 1*1
    }

    #[test]
    fn presence_penalty_applies() {
        let mut logits = vec![5.0, 5.0, 5.0];
        let counts = HashMap::from([(0, 3_usize), (2, 1_usize)]);
        apply_frequency_presence_penalty(&mut logits, &counts, 0.0, 2.0);
        assert!((logits[0] - 3.0).abs() < 1e-6); // 5 - 2
        assert!((logits[1] - 5.0).abs() < 1e-6); // unchanged
        assert!((logits[2] - 3.0).abs() < 1e-6); // 5 - 2
    }

    #[test]
    fn combined_freq_pres_penalty() {
        let mut logits = vec![10.0, 10.0];
        let counts = HashMap::from([(0, 4_usize)]);
        apply_frequency_presence_penalty(&mut logits, &counts, 0.5, 1.0);
        // logits[0] -= 0.5*4 + 1.0 = 3.0 => 7.0
        assert!((logits[0] - 7.0).abs() < 1e-6);
        assert!((logits[1] - 10.0).abs() < 1e-6);
    }
}
