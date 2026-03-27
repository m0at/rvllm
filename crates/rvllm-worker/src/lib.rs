#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
//! Single-GPU worker execution context for vllm-rs.
//!
//! Provides the `Worker` type that owns a model runner, KV cache engine, and
//! GPU stream for a single device. The worker receives `WorkerInput` from the
//! executor, applies cache operations, runs the forward pass, samples tokens,
//! and returns `WorkerOutput`.

pub mod config;
pub mod graph_runner;
#[cfg(feature = "cuda")]
pub mod gpu_worker;
pub mod input;
pub mod metrics;
pub mod worker;

pub use config::WorkerConfig;
pub use graph_runner::{GraphRunner, GraphRunnerConfig};
pub use input::prepare_input;
pub use rvllm_sequence::{SequenceData, SequenceGroupMetadata};
pub use worker::{SamplerOutput, Worker, WorkerInput, WorkerOutput};
