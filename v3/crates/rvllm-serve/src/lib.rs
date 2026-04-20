//! rvllm-serve — OpenAI-compatible HTTP inference server.
//!
//! Architecture + phase plan: `v3/INFERENCE_SERVER_PLAN.md`.
//!
//! Public surface is intentionally small: a [`config::ServerConfig`],
//! a [`router::build_router`] function that takes an [`AppState`] and
//! returns an `axum::Router`, and [`worker::WorkerHandle`] spawned via
//! [`worker::spawn_mock_worker`] (tests) or the runtime-backed worker
//! (`cuda` feature). Integration tests build the router against the
//! mock worker to cover request validation, SSE framing, and error
//! mapping without touching CUDA.

#![deny(clippy::unwrap_used)]

pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda_worker;
pub mod error;
pub mod openai;
pub mod router;
pub mod sampling;
pub mod tokenize;
pub mod worker;

pub use config::ServerConfig;
pub use error::{ApiError, ApiResult};
pub use router::{build_router, AppState};
pub use worker::{GenerateEvent, GenerateRequest, WorkerHandle};
