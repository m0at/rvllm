#![forbid(unsafe_code)]
//! Speculative decoding for vllm-rs.
//!
//! Draft-model-based speculative decoding for latency reduction. Generates K
//! tokens with a fast draft model, verifies them against the target model in
//! a single forward pass, and accepts/rejects using the speculative sampling
//! algorithm.

pub mod config;
pub mod draft;
pub mod engine;
pub mod scheduler;
pub mod verification;

pub use config::SpeculativeConfig;
pub use draft::{DraftModelRunner, DraftToken};
pub use engine::{SpeculativeEngine, SpeculativeMetrics};
pub use scheduler::{SpeculativeScheduler, SpeculativeStep};
pub use verification::{verify_tokens, verify_tokens_with_rng, VerificationResult};
