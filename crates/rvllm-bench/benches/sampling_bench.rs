use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Inline implementations (match crate code, avoid cross-crate wiring)
// ---------------------------------------------------------------------------

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    let inv = 1.0 / sum;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&x| x - lse).collect()
}

fn apply_temperature(logits: &mut [f32], temperature: f32) {
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}

fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut vals: Vec<f32> = logits.to_vec();
    vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = vals[k - 1];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(logits: &mut [f32], p: f32) {
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0;
    let mut keep = vec![false; logits.len()];
    for (idx, prob) in &indexed {
        keep[*idx] = true;
        cumulative += prob;
        if cumulative >= p {
            break;
        }
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn apply_min_p(logits: &mut [f32], min_p: f32) {
    let probs = softmax(logits);
    let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_prob * min_p;
    for (i, l) in logits.iter_mut().enumerate() {
        if probs[i] < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
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

fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    token_counts: &HashMap<u32, usize>,
    freq_penalty: f32,
    presence_penalty: f32,
) {
    for (&tok, &count) in token_counts {
        let idx = tok as usize;
        if idx < logits.len() && count > 0 {
            logits[idx] -= freq_penalty * count as f32 + presence_penalty;
        }
    }
}

fn apply_combined_penalties(
    logits: &mut [f32],
    past_tokens: &[u32],
    token_counts: &HashMap<u32, usize>,
    rep_penalty: f32,
    freq_penalty: f32,
    presence_penalty: f32,
) {
    apply_repetition_penalty(logits, past_tokens, rep_penalty);
    apply_frequency_presence_penalty(logits, token_counts, freq_penalty, presence_penalty);
}

fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn multinomial_sample(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn top_logprobs(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let log_probs = log_softmax(logits);
    let mut indexed: Vec<(u32, f32)> = log_probs
        .into_iter()
        .enumerate()
        .map(|(i, lp)| (i as u32, lp))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(n);
    indexed
}

fn dequantize_q4_0(data: &[u8], scales: &[f32], total: usize) -> Vec<f32> {
    let group_size = 32usize;
    let num_groups = (total + group_size - 1) / group_size;
    let mut output = Vec::with_capacity(total);
    for g in 0..num_groups {
        let scale = scales[g];
        let start = g * group_size;
        let end = (start + group_size).min(total);
        let byte_offset = g * (group_size / 2);
        for i in start..end {
            let local = i - start;
            let byte_idx = byte_offset + local / 2;
            let nibble = if local % 2 == 0 {
                (data[byte_idx] & 0x0F) as i8
            } else {
                ((data[byte_idx] >> 4) & 0x0F) as i8
            };
            output.push((nibble as f32 - 8.0) * scale);
        }
    }
    output
}

fn full_sampling_pipeline(logits: &[f32], past: &[u32]) -> u32 {
    let mut l = logits.to_vec();
    apply_temperature(&mut l, 0.8);
    apply_top_k(&mut l, 50);
    apply_top_p(&mut l, 0.9);
    apply_repetition_penalty(&mut l, past, 1.1);
    let probs = softmax(&l);
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn random_logits(vocab_size: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..vocab_size).map(|_| rng.gen_range(-10.0..10.0f32)).collect()
}

// ---------------------------------------------------------------------------
// Original benchmarks
// ---------------------------------------------------------------------------

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    for &vocab in &[1000, 32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust", vocab), &logits, |b, l| {
            b.iter(|| softmax(black_box(l)))
        });
    }
    group.finish();
}

fn bench_softmax_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_inplace");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust", vocab), &logits, |b, l| {
            b.iter(|| {
                let mut v = l.clone();
                softmax_inplace(black_box(&mut v));
                v
            })
        });
    }
    group.finish();
}

fn bench_log_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_softmax");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust", vocab), &logits, |b, l| {
            b.iter(|| log_softmax(black_box(l)))
        });
    }
    group.finish();
}

fn bench_multinomial(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinomial");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        let probs = softmax(&logits);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust", vocab), &probs, |b, p| {
            let mut rng = ChaCha8Rng::seed_from_u64(123);
            b.iter(|| multinomial_sample(black_box(p), &mut rng))
        });
    }
    group.finish();
}

fn bench_temperature(c: &mut Criterion) {
    let mut group = c.benchmark_group("temperature");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust", vocab), &logits, |b, l| {
            b.iter(|| {
                let mut v = l.clone();
                apply_temperature(black_box(&mut v), 0.8);
                v
            })
        });
    }
    group.finish();
}

fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust_k50", vocab), &logits, |b, l| {
            b.iter(|| {
                let mut v = l.clone();
                apply_top_k(black_box(&mut v), 50);
                v
            })
        });
    }
    group.finish();
}

fn bench_top_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_p");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust_p0.9", vocab), &logits, |b, l| {
            b.iter(|| {
                let mut v = l.clone();
                apply_top_p(black_box(&mut v), 0.9);
                v
            })
        });
    }
    group.finish();
}

fn bench_min_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_p");
    let logits = random_logits(32000, 42);
    group.throughput(Throughput::Elements(32000));
    group.bench_function("rust_32k", |b| {
        b.iter(|| {
            let mut v = logits.clone();
            apply_min_p(black_box(&mut v), 0.1);
            v
        })
    });
    group.finish();
}

fn bench_repetition_penalty(c: &mut Criterion) {
    let mut group = c.benchmark_group("repetition_penalty");
    let logits = random_logits(32000, 42);
    let past: Vec<u32> = (0..500).collect();
    group.throughput(Throughput::Elements(32000));
    group.bench_function("rust_32k_500past", |b| {
        b.iter(|| {
            let mut v = logits.clone();
            apply_repetition_penalty(black_box(&mut v), black_box(&past), 1.1);
            v
        })
    });
    group.finish();
}

fn bench_repetition_penalty_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("repetition_penalty_large");
    let logits = random_logits(32000, 42);
    let past: Vec<u32> = (0..2000).collect();
    group.throughput(Throughput::Elements(32000));
    group.bench_function("rust_32k_2000past", |b| {
        b.iter(|| {
            let mut v = logits.clone();
            apply_repetition_penalty(black_box(&mut v), black_box(&past), 1.1);
            v
        })
    });
    group.finish();
}

fn bench_combined_penalties(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_penalties");
    let logits = random_logits(32000, 42);
    let past: Vec<u32> = (0..500).collect();
    let mut token_counts = HashMap::new();
    for &t in &past {
        *token_counts.entry(t).or_insert(0usize) += 1;
    }
    // Add some repeated tokens
    for t in 0..100u32 {
        *token_counts.entry(t).or_insert(0usize) += 3;
    }
    group.throughput(Throughput::Elements(32000));
    group.bench_function("rust_32k_freq_pres_rep", |b| {
        b.iter(|| {
            let mut v = logits.clone();
            apply_combined_penalties(
                black_box(&mut v),
                black_box(&past),
                black_box(&token_counts),
                1.1,
                0.5,
                0.5,
            );
            v
        })
    });
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        let past: Vec<u32> = (0..200).collect();
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(
            BenchmarkId::new("rust", vocab),
            &(logits.clone(), past.clone()),
            |b, (l, p)| b.iter(|| full_sampling_pipeline(black_box(l), black_box(p))),
        );
    }
    group.finish();
}

fn bench_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_sample");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust", vocab), &logits, |b, l| {
            b.iter(|| greedy_sample(black_box(l)))
        });
    }
    group.finish();
}

fn bench_top_logprobs(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_logprobs");
    for &vocab in &[32000, 128000] {
        let logits = random_logits(vocab, 42);
        group.throughput(Throughput::Elements(vocab as u64));
        group.bench_with_input(BenchmarkId::new("rust_top5", vocab), &logits, |b, l| {
            b.iter(|| top_logprobs(black_box(l), 5))
        });
    }
    group.finish();
}

fn bench_dequant_q4(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequant_q4_0");
    for &size in &[1_000_000, 10_000_000] {
        let num_groups = size / 32;
        let data: Vec<u8> = (0..size / 2).map(|i| (i % 256) as u8).collect();
        let scales: Vec<f32> = (0..num_groups).map(|i| (i as f32 + 1.0) * 0.001).collect();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("rust", size),
            &(data.clone(), scales.clone()),
            |b, (d, s)| b.iter(|| dequantize_q4_0(black_box(d), black_box(s), size)),
        );
    }
    group.finish();
}

fn bench_batch_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sampling");
    for &batch in &[8, 32, 64] {
        let logits_batch: Vec<Vec<f32>> =
            (0..batch).map(|i| random_logits(32000, i as u64)).collect();
        let past: Vec<Vec<u32>> = (0..batch)
            .map(|i| (0..(100 + i * 10) as u32).collect())
            .collect();
        group.throughput(Throughput::Elements((batch * 32000) as u64));
        group.bench_with_input(
            BenchmarkId::new("rust", batch),
            &(logits_batch.clone(), past.clone()),
            |b, (lb, pb)| {
                b.iter(|| {
                    for (l, p) in lb.iter().zip(pb.iter()) {
                        full_sampling_pipeline(black_box(l), black_box(p));
                    }
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Rayon parallel batch benchmarks
// ---------------------------------------------------------------------------

fn bench_batch_sampling_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sampling_parallel");
    for &batch in &[8, 32, 64] {
        let logits_batch: Vec<Vec<f32>> =
            (0..batch).map(|i| random_logits(32000, i as u64)).collect();
        let past: Vec<Vec<u32>> = (0..batch)
            .map(|i| (0..(100 + i * 10) as u32).collect())
            .collect();
        group.throughput(Throughput::Elements((batch * 32000) as u64));
        group.bench_with_input(
            BenchmarkId::new("rust_rayon", batch),
            &(logits_batch.clone(), past.clone()),
            |b, (lb, pb)| {
                b.iter(|| {
                    lb.par_iter()
                        .zip(pb.par_iter())
                        .for_each(|(l, p)| {
                            let _ = full_sampling_pipeline(black_box(l), black_box(p));
                        });
                })
            },
        );
    }
    group.finish();
}

fn bench_batch_softmax_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_softmax_parallel");
    for &batch in &[8, 32, 64] {
        let logits_batch: Vec<Vec<f32>> =
            (0..batch).map(|i| random_logits(32000, i as u64)).collect();
        group.throughput(Throughput::Elements((batch * 32000) as u64));
        group.bench_with_input(
            BenchmarkId::new("rust_rayon", batch),
            &logits_batch,
            |b, lb| {
                b.iter(|| {
                    let _results: Vec<Vec<f32>> =
                        lb.par_iter().map(|l| softmax(black_box(l))).collect();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_softmax,
    bench_softmax_inplace,
    bench_log_softmax,
    bench_multinomial,
    bench_temperature,
    bench_top_k,
    bench_top_p,
    bench_min_p,
    bench_repetition_penalty,
    bench_repetition_penalty_large,
    bench_combined_penalties,
    bench_full_pipeline,
    bench_greedy,
    bench_top_logprobs,
    bench_dequant_q4,
    bench_batch_sampling,
    bench_batch_sampling_parallel,
    bench_batch_softmax_parallel,
);
criterion_main!(benches);
