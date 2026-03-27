//! Batched sampling: process multiple sequences in one call.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use tracing::trace;
use rvllm_core::prelude::{LLMError, Result, SamplingParams};

use crate::sampler::{Sampler, SamplerOutput};

/// Minimum batch size to justify thread-pool overhead.
const PAR_THRESHOLD: usize = 4;

/// Sample tokens for a batch of sequences.
///
/// For small batches (< 4), runs sequentially to avoid thread-pool overhead.
/// For larger batches, fans out across Rayon's thread pool.
pub fn sample_batch(
    logits_batch: &[Vec<f32>],
    params: &[&SamplingParams],
    past_tokens: &[&[u32]],
    rngs: &mut [StdRng],
) -> Result<Vec<SamplerOutput>> {
    let batch_size = logits_batch.len();
    if params.len() != batch_size || past_tokens.len() != batch_size || rngs.len() != batch_size {
        return Err(LLMError::SamplingError(format!(
            "batch size mismatch: logits={}, params={}, past_tokens={}, rngs={}",
            batch_size,
            params.len(),
            past_tokens.len(),
            rngs.len(),
        )));
    }

    if batch_size < PAR_THRESHOLD {
        trace!(batch_size, "sample_batch sequential");
        sample_batch_sequential(logits_batch, params, past_tokens, rngs)
    } else {
        trace!(batch_size, "sample_batch parallel");
        sample_batch_parallel_inner(logits_batch, params, past_tokens, rngs)
    }
}

/// Sequential path -- no thread-pool overhead.
fn sample_batch_sequential(
    logits_batch: &[Vec<f32>],
    params: &[&SamplingParams],
    past_tokens: &[&[u32]],
    rngs: &mut [StdRng],
) -> Result<Vec<SamplerOutput>> {
    let sampler = Sampler::new();
    let mut outputs = Vec::with_capacity(logits_batch.len());
    for i in 0..logits_batch.len() {
        let vocab_size = logits_batch[i].len();
        outputs.push(sampler.sample(
            &logits_batch[i],
            vocab_size,
            params[i],
            past_tokens[i],
            &mut rngs[i],
        )?);
    }
    Ok(outputs)
}

/// Parallel path -- each sequence processed on its own Rayon task.
fn sample_batch_parallel_inner(
    logits_batch: &[Vec<f32>],
    params: &[&SamplingParams],
    past_tokens: &[&[u32]],
    rngs: &mut [StdRng],
) -> Result<Vec<SamplerOutput>> {
    let results: Vec<Result<SamplerOutput>> = logits_batch
        .par_iter()
        .zip(params.par_iter())
        .zip(past_tokens.par_iter())
        .zip(rngs.par_iter_mut())
        .map(|(((logits, params), past), rng)| {
            let sampler = Sampler::new();
            let vocab_size = logits.len();
            sampler.sample(logits, vocab_size, params, past, rng)
        })
        .collect();

    results.into_iter().collect()
}

/// Explicitly parallel entry point -- always uses Rayon regardless of batch size.
/// Useful when the caller knows the batch is large or wants to control dispatch.
pub fn sample_batch_parallel(
    logits_batch: &[Vec<f32>],
    params: &[&SamplingParams],
    past_tokens: &[&[u32]],
    rngs: &mut [StdRng],
) -> Result<Vec<SamplerOutput>> {
    let batch_size = logits_batch.len();
    if params.len() != batch_size || past_tokens.len() != batch_size || rngs.len() != batch_size {
        return Err(LLMError::SamplingError(format!(
            "batch size mismatch: logits={}, params={}, past_tokens={}, rngs={}",
            batch_size,
            params.len(),
            past_tokens.len(),
            rngs.len(),
        )));
    }

    trace!(batch_size, "sample_batch_parallel");

    if batch_size == 0 {
        return Ok(vec![]);
    }

    sample_batch_parallel_inner(logits_batch, params, past_tokens, rngs)
}

/// Create a deterministic RNG from an optional seed. If seed is None,
/// falls back to a thread-level random seed.
pub fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rvllm_core::prelude::SamplingParams;

    #[test]
    fn batch_sample_basic() {
        let logits1 = vec![1.0, 5.0, 2.0];
        let logits2 = vec![3.0, 1.0, 8.0];
        let logits_batch = vec![logits1, logits2];

        let p1 = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let p2 = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p1, &p2];
        let past: Vec<&[u32]> = vec![&[], &[]];
        let mut rngs = vec![
            StdRng::seed_from_u64(1),
            StdRng::seed_from_u64(2),
        ];

        let outputs = sample_batch(&logits_batch, &params, &past, &mut rngs).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].token_id, 1); // greedy -> max at index 1
        assert_eq!(outputs[1].token_id, 2); // greedy -> max at index 2
    }

    #[test]
    fn batch_size_mismatch() {
        let logits_batch = vec![vec![1.0, 2.0]];
        let p = SamplingParams::default();
        let params: Vec<&SamplingParams> = vec![&p, &p]; // mismatch
        let past: Vec<&[u32]> = vec![&[]];
        let mut rngs = vec![StdRng::seed_from_u64(1)];

        let result = sample_batch(&logits_batch, &params, &past, &mut rngs);
        assert!(result.is_err());
    }

    #[test]
    fn batch_determinism() {
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let logits_batch = vec![logits.clone(), logits.clone()];
        let p = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p, &p];
        let past: Vec<&[u32]> = vec![&[], &[]];

        // Run twice with same seeds.
        let mut rngs1 = vec![StdRng::seed_from_u64(42), StdRng::seed_from_u64(43)];
        let mut rngs2 = vec![StdRng::seed_from_u64(42), StdRng::seed_from_u64(43)];

        let out1 = sample_batch(&logits_batch, &params, &past, &mut rngs1).unwrap();
        let out2 = sample_batch(&logits_batch, &params, &past, &mut rngs2).unwrap();

        assert_eq!(out1[0].token_id, out2[0].token_id);
        assert_eq!(out1[1].token_id, out2[1].token_id);
    }

    #[test]
    fn make_rng_deterministic() {
        let mut r1 = make_rng(Some(12345));
        let mut r2 = make_rng(Some(12345));
        use rand::Rng;
        let v1: u64 = r1.gen();
        let v2: u64 = r2.gen();
        assert_eq!(v1, v2);
    }

    #[test]
    fn make_rng_entropy() {
        // Just ensure it doesn't panic.
        let _r = make_rng(None);
    }

    #[test]
    fn batch_empty() {
        let outputs = sample_batch(&[], &[], &[], &mut []).unwrap();
        assert!(outputs.is_empty());
    }

    #[test]
    fn parallel_matches_sequential() {
        // Build a batch large enough to exercise the parallel path,
        // then verify results match the sequential path exactly.
        let batch_size = 16;
        let vocab = 128;

        // Deterministic logits per sequence.
        let logits_batch: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| (0..vocab).map(|v| ((i * vocab + v) as f32).sin()).collect())
            .collect();

        let p = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            ..Default::default()
        };
        let params: Vec<&SamplingParams> = vec![&p; batch_size];
        let past: Vec<&[u32]> = vec![&[]; batch_size];

        // Sequential run.
        let mut rngs_seq: Vec<StdRng> =
            (0..batch_size).map(|i| StdRng::seed_from_u64(i as u64 + 100)).collect();
        let seq_out = sample_batch_sequential(&logits_batch, &params, &past, &mut rngs_seq).unwrap();

        // Parallel run (same seeds).
        let mut rngs_par: Vec<StdRng> =
            (0..batch_size).map(|i| StdRng::seed_from_u64(i as u64 + 100)).collect();
        let par_out = sample_batch_parallel(&logits_batch, &params, &past, &mut rngs_par).unwrap();

        for i in 0..batch_size {
            assert_eq!(
                seq_out[i].token_id, par_out[i].token_id,
                "mismatch at sequence {}: seq={} par={}",
                i, seq_out[i].token_id, par_out[i].token_id
            );
            assert!(
                (seq_out[i].logprob - par_out[i].logprob).abs() < 1e-6,
                "logprob mismatch at sequence {}",
                i
            );
        }
    }

    #[test]
    fn parallel_explicit_empty() {
        let outputs = sample_batch_parallel(&[], &[], &[], &mut []).unwrap();
        assert!(outputs.is_empty());
    }

    #[test]
    fn parallel_mismatch() {
        let logits_batch = vec![vec![1.0, 2.0]];
        let p = SamplingParams::default();
        let params: Vec<&SamplingParams> = vec![&p, &p];
        let past: Vec<&[u32]> = vec![&[]];
        let mut rngs = vec![StdRng::seed_from_u64(1)];

        let result = sample_batch_parallel(&logits_batch, &params, &past, &mut rngs);
        assert!(result.is_err());
    }
}
