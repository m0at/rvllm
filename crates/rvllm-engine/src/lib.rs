#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
//! Main inference engine for vllm-rs.
//!
//! Composes the scheduler, executor, and tokenizer into a synchronous
//! [`LLMEngine`] and an async [`AsyncLLMEngine`] that drives the
//! continuous-batching inference loop.
//!
//! Real dependency crate types are imported and adapted:
//! - `rvllm_executor` -- `Executor` trait, `ExecutorInput`, `SamplerOutput`, `ExecutorFactory`
//! - `rvllm_tokenizer` -- `Tokenizer` for encode/decode
//! - `rvllm_sequence` -- `Sequence`, `SequenceGroup`, `SequenceGroupMetadata`
//!
//! `ExecutorAdapter` bridges the real async `rvllm_executor::Executor` into the
//! engine's sync step loop. A `SchedulerAdapter` for `rvllm_scheduler` is
//! pending that crate's API alignment with `rvllm_sequence` types.

pub mod async_engine;
#[cfg(feature = "cuda")]
pub mod async_gpu_engine;
pub mod beam_search;
pub mod best_of_n;
pub mod engine;
#[cfg(feature = "cuda")]
pub mod gpu_engine;
pub mod gpu_metrics;
pub mod output;
pub mod stop_checker;

pub use async_engine::AsyncLLMEngine;
pub use beam_search::BeamSearchState;
pub use best_of_n::{build_best_of_n_output, select_best_of_n};
pub use engine::LLMEngine;
pub use engine::{Executor, ExecutorAdapter, Scheduler};
pub use engine::{ExecutorInput, SamplerOutput, SchedulerOutputs};
pub use output::OutputProcessor;
pub use stop_checker::StopChecker;

#[cfg(feature = "cuda")]
pub use async_gpu_engine::AsyncGpuLLMEngine;
#[cfg(feature = "cuda")]
pub use gpu_engine::GpuLLMEngine;

// Re-export real executor types for downstream convenience
pub use rvllm_executor::ExecutorConfig;
pub use rvllm_executor::ExecutorFactory;
