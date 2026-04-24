# B200 vs TPU v6e for rvLLM

This note compares a single NVIDIA B200 against TPU v6e slices using the measured and projected numbers in `docs/bench.html`.

## Headline

For rvLLM, the answer depends on the model regime:

| Workload | v6e-8 vs single B200 | Read |
|---|---:|---|
| E4B / 4B int8 decode | ~0.37x B200 peak | B200 is ~2.7x v6e-8 |
| 31B FP8/int8 high-batch decode | ~0.9-1.0x B200 | v6e-8 is roughly one B200 |
| Raw dense BF16 spec sheet | ~3.3x B200 | Misleading for this decode workload |

The spec-sheet FLOP comparison makes TPU v6e-8 look larger than one B200. The rvLLM benchmark behavior says something different: small-model decode is mostly memory-bandwidth and launch/fusion limited, so B200 wins hard on E4B. For 31B high-batch decode, v6e-8 lands near one B200.

## Source Numbers

From `docs/bench.html`:

| Config | B=1 tok/s | B=128 tok/s | Peak tok/s |
|---|---:|---:|---:|
| E4B on TPU v6e-4 | 78.3 | 6,298 | 16,794 |
| 26B-A4B on TPU v6e-4 | 52.9 | 4,915 | 14,899 |
| 31B on TPU v6e-4 | 44.2 | 3,853 | 9,600 |
| 31B rvLLM GPU H100 | 53 | 5,802 | 8,786 |
| 31B vLLM GPU H100 | 66.9 | 4,689 | 8,243 |

Bench assumptions:

| Item | Value |
|---|---|
| TPU slice measured | v6e-4 |
| TPU price in bench | $5.20/hr |
| TPU weights/KV | int8 weights, bf16 KV |
| GPU measured | H100 SXM 80 GB |
| GPU price in bench | $1.92/hr |
| GPU weights/KV | FP8 weights, F16 KV |
| Max context | 2048 |

The B200 E4B projection in `bench.html` uses 8.0 TB/s HBM bandwidth:

| Batch | Projected B200 int8 E4B tok/s |
|---:|---:|
| 1 | 700 |
| 8 | 5,000 |
| 64 | 30,000 |
| 128 | 50,000 |
| 256 | 72,000 |
| 512 | 90,000 |

## E4B: v6e-8 vs B200

Assumption: v6e-8 is two independent v6e-4 replicas for aggregate serving throughput. That doubles measured v6e-4 throughput. It does not mean one request gets half the latency.

| Batch | v6e-4 measured | v6e-8 aggregate | B200 projected | v6e-8 / B200 | B200 / v6e-8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 78 | 156 | 700 | 0.22x | 4.49x |
| 8 | 542 | 1,084 | 5,000 | 0.22x | 4.61x |
| 64 | 3,661 | 7,322 | 30,000 | 0.24x | 4.10x |
| 128 | 6,298 | 12,596 | 50,000 | 0.25x | 3.97x |
| 256 | 10,214 | 20,428 | 72,000 | 0.28x | 3.52x |
| 512 | 13,773 | 27,546 | 90,000 | 0.31x | 3.27x |
| peak | 16,794 | 33,588 | 90,000 | 0.37x | 2.68x |

For E4B, one B200 is roughly equivalent to 21-27 TPU v6e chips depending on batch, using the bench projection and measured v6e-4 curve.

## E4B: TPU Slice Equivalents

Using v6e-4 peak `16,794 tok/s` and B200 projected peak `90,000 tok/s`:

| TPU slice | Chips | E4B peak estimate | Single B200 equivalents |
|---|---:|---:|---:|
| v6e-8 | 8 | 33,588 tok/s | 0.37x |
| v6e-16 | 16 | 67,176 tok/s | 0.75x |
| v6e-32 | 32 | 134,352 tok/s | 1.49x |
| v6e-64 | 64 | 268,704 tok/s | 2.99x |
| v6e-128 | 128 | 537,408 tok/s | 5.97x |
| v6e-256 full pod | 256 | 1,074,816 tok/s | 11.94x |

Read this as serving capacity under replication. It is not a single-sequence latency comparison.

## 31B: v6e-8 vs B200

For 31B, the direct B200 number is not measured in `bench.html`. The estimate below starts from measured rvLLM H100 peak and scales by a rough B200/H100 factor of 2.25-2.39x. That gives:

| Config | 31B high-batch tok/s |
|---|---:|
| TPU v6e-4 measured | 9,600 |
| TPU v6e-8 replicated | 19,200 |
| rvLLM H100 measured | 8,786 |
| B200 estimated | ~19,800-21,000 |

So for 31B high-batch decode:

| TPU slice | 31B peak estimate | Single B200 equivalents |
|---|---:|---:|
| v6e-8 | 19,200 tok/s | ~0.9-1.0x |
| v6e-16 | 38,400 tok/s | ~1.8-1.9x |
| v6e-32 | 76,800 tok/s | ~3.7-3.9x |
| v6e-64 | 153,600 tok/s | ~7.3-7.8x |
| v6e-128 | 307,200 tok/s | ~14.6-15.5x |
| v6e-256 full pod | 614,400 tok/s | ~29-31x |

## Why FLOPs Mislead Here

A raw dense BF16 comparison says:

| Hardware | Dense BF16 peak |
|---|---:|
| Single B200 | ~2.25 PFLOP/s |
| TPU v6e-8 | ~7.34 PFLOP/s |

That makes v6e-8 look like ~3.3 B200s. For rvLLM decode, that is the wrong mental model.

E4B B=1 decode is a weight-streaming problem. The model is about 5 GB at int8, and each token mostly reads weights once. Arithmetic intensity is low, so HBM bandwidth and fused execution dominate. That is why the bench page projects B200 at ~500-900 tok/s practical B=1 for E4B, while measured TPU v6e-4 is ~78 tok/s.

At high batch and larger models, compute utilization improves and TPU's XLA fused loop plus aggregate bandwidth closes the gap. That is why 31B v6e-8 is roughly comparable to a B200, while E4B v6e-8 is not.

## Bottom Line

If the question is "how many B200s is a TPU v6e-8 worth for rvLLM?":

| Workload | Answer |
|---|---|
| E4B small-model serving | v6e-8 is ~0.37 B200 peak |
| E4B small-model latency | B200 is much faster per single request |
| 31B high-batch serving | v6e-8 is roughly 1 B200 |
| Full v6e-256 pod, E4B | ~12 B200s |
| Full v6e-256 pod, 31B | ~30 B200s |

