//! GPU sampling bridge: sample from logits transferred back from GPU.
//!
//! Logits are small enough (batch_size * vocab_size * 4 bytes) to transfer
//! back to CPU, so we reuse the existing optimized Rust sampler rather than
//! running a separate CUDA kernel.

use rand::rngs::StdRng;
use tracing::trace;
use rvllm_core::prelude::{LLMError, Result, SamplingParams, TokenId};

use crate::batch::sample_batch;
use crate::sampler::SamplerOutput;

/// Sample tokens from a flat logits buffer returned by GPU forward pass.
///
/// # Arguments
/// * `logits` - Flat logits vector of shape `[batch_size, vocab_size]`,
///   transferred from GPU to CPU. Length must equal `batch_size * vocab_size`.
/// * `vocab_size` - Vocabulary size (columns in the logits matrix).
/// * `params` - Per-sequence sampling parameters. Length = batch_size.
/// * `past_tokens` - Per-sequence previously generated tokens (for penalties).
/// * `rngs` - Per-sequence RNGs for deterministic sampling.
///
/// # Returns
/// One `SamplerOutput` per sequence in the batch.
pub fn sample_from_gpu_logits(
    logits: Vec<f32>,
    vocab_size: usize,
    params: &[&SamplingParams],
    past_tokens: &[&[TokenId]],
    rngs: &mut [StdRng],
) -> Result<Vec<SamplerOutput>> {
    let batch_size = params.len();

    if vocab_size == 0 {
        return Err(LLMError::SamplingError("vocab_size is 0".into()));
    }

    let expected_len = batch_size * vocab_size;
    if logits.len() != expected_len {
        return Err(LLMError::SamplingError(format!(
            "logits length {} != batch_size({}) * vocab_size({})",
            logits.len(),
            batch_size,
            vocab_size,
        )));
    }

    if past_tokens.len() != batch_size || rngs.len() != batch_size {
        return Err(LLMError::SamplingError(format!(
            "batch size mismatch: params={}, past_tokens={}, rngs={}",
            batch_size,
            past_tokens.len(),
            rngs.len(),
        )));
    }

    if batch_size == 0 {
        return Ok(Vec::new());
    }

    trace!(batch_size, vocab_size, "sample_from_gpu_logits");

    // Split flat logits into per-sequence slices. Each is a contiguous
    // vocab_size chunk -- we collect into Vec<Vec<f32>> to match the
    // sample_batch API.
    let logits_batch: Vec<Vec<f32>> = logits
        .chunks_exact(vocab_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    sample_batch(&logits_batch, params, past_tokens, rngs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_rngs(n: usize) -> Vec<StdRng> {
        (0..n).map(|i| StdRng::seed_from_u64(i as u64 + 42)).collect()
    }

    #[test]
    fn basic_greedy() {
        let vocab_size = 4;
        // Batch of 2: first sequence peaks at token 1, second at token 3
        let logits = vec![
            1.0, 5.0, 2.0, 0.5, // seq 0 -> argmax = 1
            0.1, 0.2, 0.3, 9.0, // seq 1 -> argmax = 3
        ];
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p, &p];
        let past: Vec<&[TokenId]> = vec![&[], &[]];
        let mut rngs = make_rngs(2);

        let out = sample_from_gpu_logits(logits, vocab_size, &params, &past, &mut rngs).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].token_id, 1);
        assert_eq!(out[1].token_id, 3);
    }

    #[test]
    fn single_sequence() {
        let vocab_size = 3;
        let logits = vec![0.0, 0.0, 10.0];
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p];
        let past: Vec<&[TokenId]> = vec![&[]];
        let mut rngs = make_rngs(1);

        let out = sample_from_gpu_logits(logits, vocab_size, &params, &past, &mut rngs).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, 2);
    }

    #[test]
    fn empty_batch() {
        let out = sample_from_gpu_logits(vec![], 32000, &[], &[], &mut []).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn logits_length_mismatch() {
        let p = SamplingParams::default();
        let params: Vec<&SamplingParams> = vec![&p];
        let past: Vec<&[TokenId]> = vec![&[]];
        let mut rngs = make_rngs(1);

        // 5 logits but vocab_size=4 and batch=1 expects 4
        let result = sample_from_gpu_logits(vec![1.0; 5], 4, &params, &past, &mut rngs);
        assert!(result.is_err());
    }

    #[test]
    fn zero_vocab_size() {
        let result = sample_from_gpu_logits(vec![], 0, &[], &[], &mut []);
        assert!(result.is_err());
    }

    #[test]
    fn batch_size_mismatch_past_tokens() {
        let p = SamplingParams::default();
        let params: Vec<&SamplingParams> = vec![&p, &p];
        let past: Vec<&[TokenId]> = vec![&[]]; // only 1, but batch=2
        let mut rngs = make_rngs(2);

        let result = sample_from_gpu_logits(vec![0.0; 8], 4, &params, &past, &mut rngs);
        assert!(result.is_err());
    }

    #[test]
    fn batch_size_mismatch_rngs() {
        let p = SamplingParams::default();
        let params: Vec<&SamplingParams> = vec![&p, &p];
        let past: Vec<&[TokenId]> = vec![&[], &[]];
        let mut rngs = make_rngs(1); // only 1 rng for batch=2

        let result = sample_from_gpu_logits(vec![0.0; 8], 4, &params, &past, &mut rngs);
        assert!(result.is_err());
    }

    #[test]
    fn deterministic_with_same_seeds() {
        let vocab_size = 5;
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // uniform
        let p = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p];
        let past: Vec<&[TokenId]> = vec![&[]];

        let mut rngs1 = vec![StdRng::seed_from_u64(999)];
        let mut rngs2 = vec![StdRng::seed_from_u64(999)];

        let out1 =
            sample_from_gpu_logits(logits.clone(), vocab_size, &params, &past, &mut rngs1).unwrap();
        let out2 =
            sample_from_gpu_logits(logits, vocab_size, &params, &past, &mut rngs2).unwrap();

        assert_eq!(out1[0].token_id, out2[0].token_id);
    }

    #[test]
    fn with_repetition_penalty() {
        let vocab_size = 3;
        // Token 1 has highest logit, but heavy repetition penalty on it
        let logits = vec![3.0, 4.0, 3.0];
        let p = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 100.0,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p];
        let past: Vec<&[TokenId]> = vec![&[1, 1, 1]];
        let mut rngs = make_rngs(1);

        let out = sample_from_gpu_logits(logits, vocab_size, &params, &past, &mut rngs).unwrap();
        assert_ne!(out[0].token_id, 1);
    }

    #[test]
    fn large_batch() {
        let vocab_size = 128;
        let batch_size = 16;
        let mut logits = vec![0.0f32; batch_size * vocab_size];

        // Each sequence has a different argmax
        for i in 0..batch_size {
            logits[i * vocab_size + i] = 10.0;
        }

        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p; batch_size];
        let past: Vec<&[TokenId]> = vec![&[]; batch_size];
        let mut rngs = make_rngs(batch_size);

        let out = sample_from_gpu_logits(logits, vocab_size, &params, &past, &mut rngs).unwrap();
        assert_eq!(out.len(), batch_size);
        for i in 0..batch_size {
            assert_eq!(out[i].token_id, i as TokenId);
        }
    }
}
