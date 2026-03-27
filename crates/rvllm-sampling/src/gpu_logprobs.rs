//! GPU logprobs: compute log_softmax and extract top-N logprobs from logits
//! produced by the GPU forward pass.
//!
//! This module operates on CPU-side logit buffers after DtoH transfer.
//! The GPU forward pass produces a flat `[batch_size, vocab_size]` logit tensor;
//! we compute log-softmax per position and extract the requested top-N entries
//! for populating `CompletionOutput.logprobs`.

use tracing::trace;
use rvllm_core::prelude::{LogProb, TokenId};

use crate::math;

/// Per-position logprob result: the selected token's logprob plus optional top-N.
#[derive(Debug, Clone)]
pub struct PositionLogprobs {
    /// Log-probability of the token that was actually sampled/selected.
    pub token_logprob: LogProb,
    /// Top-N (token_id, log_prob) pairs sorted by descending log-prob.
    /// Empty if logprobs were not requested.
    pub top_logprobs: Vec<(TokenId, LogProb)>,
}

/// Compute log-softmax over a single position's logits and extract the
/// selected token's logprob plus optional top-N logprobs.
///
/// # Arguments
/// * `logits` - Raw logit slice for one token position (length = vocab_size).
/// * `selected_token` - The token that was sampled at this position.
/// * `num_top` - How many top logprobs to return. Pass 0 or None to skip.
pub fn compute_position_logprobs(
    logits: &[f32],
    selected_token: TokenId,
    num_top: Option<usize>,
) -> PositionLogprobs {
    if logits.is_empty() {
        return PositionLogprobs {
            token_logprob: f32::NEG_INFINITY,
            top_logprobs: Vec::new(),
        };
    }

    let log_probs = math::log_softmax(logits);
    let token_logprob = log_probs
        .get(selected_token as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY);

    let top_logprobs = match num_top {
        Some(n) if n > 0 => math::top_logprobs(logits, n),
        _ => Vec::new(),
    };

    PositionLogprobs {
        token_logprob,
        top_logprobs,
    }
}

/// Compute logprobs for a batch of positions from a flat logits buffer.
///
/// Each entry in `positions` is `(offset_in_logits, selected_token)` where
/// `offset_in_logits` is the starting index into the flat logits buffer for
/// that position's vocab-sized slice.
///
/// # Arguments
/// * `logits` - Flat logits buffer `[total_positions, vocab_size]`.
/// * `vocab_size` - Size of the vocabulary dimension.
/// * `positions` - Per-position `(logit_offset, selected_token_id)`.
/// * `num_top` - How many top logprobs to return per position.
pub fn compute_batch_logprobs(
    logits: &[f32],
    vocab_size: usize,
    positions: &[(usize, TokenId)],
    num_top: Option<usize>,
) -> Vec<PositionLogprobs> {
    trace!(
        num_positions = positions.len(),
        vocab_size,
        "compute_batch_logprobs"
    );

    positions
        .iter()
        .map(|&(offset, token_id)| {
            let end = offset + vocab_size;
            if end <= logits.len() {
                compute_position_logprobs(&logits[offset..end], token_id, num_top)
            } else {
                PositionLogprobs {
                    token_logprob: f32::NEG_INFINITY,
                    top_logprobs: Vec::new(),
                }
            }
        })
        .collect()
}

/// Compute logprobs for prompt tokens when `echo: true` is requested.
///
/// Given the full logits from a prefill pass `[num_prompt_tokens, vocab_size]`
/// and the prompt token IDs, computes the logprob of each prompt token at its
/// position. For position 0 there is no preceding context so we use the logits
/// from position 0 to score token at position 1, etc. The first token gets
/// `NEG_INFINITY` since it has no conditioning logits.
///
/// # Arguments
/// * `logits` - Flat logits from prefill, shape `[num_prompt_tokens, vocab_size]`.
/// * `vocab_size` - Vocabulary size.
/// * `prompt_tokens` - The prompt token IDs.
/// * `num_top` - How many top logprobs per position.
pub fn compute_prompt_logprobs(
    logits: &[f32],
    vocab_size: usize,
    prompt_tokens: &[TokenId],
    num_top: Option<usize>,
) -> Vec<PositionLogprobs> {
    if prompt_tokens.is_empty() || vocab_size == 0 {
        return Vec::new();
    }

    trace!(
        num_prompt_tokens = prompt_tokens.len(),
        vocab_size,
        "compute_prompt_logprobs"
    );

    let mut results = Vec::with_capacity(prompt_tokens.len());

    // Position 0: no conditioning context, logprob is undefined
    results.push(PositionLogprobs {
        token_logprob: f32::NEG_INFINITY,
        top_logprobs: Vec::new(),
    });

    // For positions 1..n, use logits[i-1] to score prompt_tokens[i]
    // because logits[i-1] gives P(token | prefix[..i])
    for i in 1..prompt_tokens.len() {
        let logit_offset = (i - 1) * vocab_size;
        let logit_end = logit_offset + vocab_size;
        if logit_end <= logits.len() {
            results.push(compute_position_logprobs(
                &logits[logit_offset..logit_end],
                prompt_tokens[i],
                num_top,
            ));
        } else {
            results.push(PositionLogprobs {
                token_logprob: f32::NEG_INFINITY,
                top_logprobs: Vec::new(),
            });
        }
    }

    results
}

/// Convert a list of `PositionLogprobs` into the format stored in
/// `CompletionOutput.logprobs`: `Vec<Vec<(TokenId, LogProb)>>`.
pub fn logprobs_to_output_format(
    positions: &[PositionLogprobs],
) -> Vec<Vec<(TokenId, LogProb)>> {
    positions
        .iter()
        .map(|p| p.top_logprobs.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_position_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let result = compute_position_logprobs(&logits, 1, Some(2));
        // Token 1 has the highest logit, so its logprob should be closest to 0
        assert!(result.token_logprob > -1.0);
        assert_eq!(result.top_logprobs.len(), 2);
        assert_eq!(result.top_logprobs[0].0, 1); // highest
    }

    #[test]
    fn single_position_no_top() {
        let logits = vec![1.0, 5.0, 3.0];
        let result = compute_position_logprobs(&logits, 0, None);
        assert!(result.token_logprob.is_finite());
        assert!(result.top_logprobs.is_empty());
    }

    #[test]
    fn single_position_zero_top() {
        let logits = vec![1.0, 2.0];
        let result = compute_position_logprobs(&logits, 0, Some(0));
        assert!(result.top_logprobs.is_empty());
    }

    #[test]
    fn empty_logits() {
        let result = compute_position_logprobs(&[], 0, Some(5));
        assert_eq!(result.token_logprob, f32::NEG_INFINITY);
        assert!(result.top_logprobs.is_empty());
    }

    #[test]
    fn batch_logprobs_basic() {
        let vocab = 4;
        let logits = vec![
            1.0, 5.0, 2.0, 0.5, // position 0
            0.1, 0.2, 0.3, 9.0, // position 1
        ];
        let positions = vec![(0, 1u32), (4, 3u32)];
        let results = compute_batch_logprobs(&logits, vocab, &positions, Some(2));
        assert_eq!(results.len(), 2);
        assert!(results[0].token_logprob > -1.0); // token 1 is argmax
        assert!(results[1].token_logprob > -1.0); // token 3 is argmax
        assert_eq!(results[0].top_logprobs[0].0, 1);
        assert_eq!(results[1].top_logprobs[0].0, 3);
    }

    #[test]
    fn batch_logprobs_out_of_bounds() {
        let logits = vec![1.0, 2.0, 3.0];
        let positions = vec![(0, 2u32), (100, 0u32)]; // second is OOB
        let results = compute_batch_logprobs(&logits, 3, &positions, Some(1));
        assert_eq!(results.len(), 2);
        assert!(results[0].token_logprob.is_finite());
        assert_eq!(results[1].token_logprob, f32::NEG_INFINITY);
    }

    #[test]
    fn prompt_logprobs_basic() {
        let vocab = 4;
        // 3 prompt tokens -> need logits for 3 positions
        let logits = vec![
            1.0, 5.0, 2.0, 0.5, // logits at position 0
            0.1, 0.2, 8.0, 0.3, // logits at position 1
            3.0, 1.0, 1.0, 1.0, // logits at position 2
        ];
        let prompt_tokens = vec![10, 1, 2]; // token IDs

        let results = compute_prompt_logprobs(&logits, vocab, &prompt_tokens, Some(2));
        assert_eq!(results.len(), 3);

        // Position 0: no conditioning, should be NEG_INFINITY
        assert_eq!(results[0].token_logprob, f32::NEG_INFINITY);
        assert!(results[0].top_logprobs.is_empty());

        // Position 1: uses logits[0] to score token 1 (which has logit 5.0, the max)
        assert!(results[1].token_logprob > -1.0);
        assert_eq!(results[1].top_logprobs.len(), 2);

        // Position 2: uses logits[1] to score token 2 (which has logit 8.0, the max)
        assert!(results[2].token_logprob > -1.0);
    }

    #[test]
    fn prompt_logprobs_empty() {
        let results = compute_prompt_logprobs(&[], 4, &[], Some(2));
        assert!(results.is_empty());
    }

    #[test]
    fn prompt_logprobs_single_token() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let results = compute_prompt_logprobs(&logits, 4, &[2], Some(2));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].token_logprob, f32::NEG_INFINITY);
    }

    #[test]
    fn logprobs_to_output_format_roundtrip() {
        let positions = vec![
            PositionLogprobs {
                token_logprob: -0.5,
                top_logprobs: vec![(1, -0.5), (2, -1.0)],
            },
            PositionLogprobs {
                token_logprob: -0.3,
                top_logprobs: vec![(3, -0.3)],
            },
        ];
        let output = logprobs_to_output_format(&positions);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 2);
        assert_eq!(output[1].len(), 1);
        assert_eq!(output[0][0].0, 1);
    }

    #[test]
    fn logprobs_values_are_negative() {
        // All logprobs from log_softmax must be <= 0
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_position_logprobs(&logits, 4, Some(5));
        assert!(result.token_logprob <= 0.0);
        for &(_, lp) in &result.top_logprobs {
            assert!(lp <= 0.0);
        }
    }

    #[test]
    fn logprobs_sum_to_one_approximately() {
        // exp(logprobs) should sum to 1.0
        let logits = vec![1.0, 2.0, 3.0];
        let result = compute_position_logprobs(&logits, 0, Some(3));
        let sum: f32 = result.top_logprobs.iter().map(|(_, lp)| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn selected_token_out_of_range() {
        let logits = vec![1.0, 2.0];
        let result = compute_position_logprobs(&logits, 999, Some(2));
        assert_eq!(result.token_logprob, f32::NEG_INFINITY);
        assert_eq!(result.top_logprobs.len(), 2);
    }
}
