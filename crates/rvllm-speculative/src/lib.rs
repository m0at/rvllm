#![forbid(unsafe_code)]
//! Speculative decoding for vllm-rs.
//!
//! Draft-model-based speculative decoding for latency reduction. Generates K
//! tokens with a fast draft model, verifies them against the target model in
//! a single forward pass, and accepts/rejects using the speculative sampling
//! algorithm.
//!
//! The core algorithm (Leviathan et al., 2023):
//!   1. Draft phase: run a small/fast model for K steps -> K candidate tokens
//!   2. Verify phase: run the target model on all K candidates as a single
//!      prefill-style forward pass, producing K+1 probability distributions
//!   3. Accept/reject: walk positions 0..K, accepting draft tokens via modified
//!      rejection sampling. On first rejection, resample from max(0, p_target - p_draft).
//!   4. Output: all accepted tokens + 1 bonus token from the target at the
//!      rejection point (or position K+1 if all K accepted).

pub mod config;
pub mod draft;
pub mod engine;
pub mod scheduler;
pub mod verification;

pub use config::SpeculativeConfig;
pub use draft::{DraftModel, DraftModelRunner, DraftToken};
pub use engine::{SpeculativeEngine, SpeculativeMetrics, TargetModel};
pub use scheduler::{SpeculativeScheduler, SpeculativeStep};
pub use verification::{verify_tokens, verify_tokens_with_rng, VerificationResult};
