//! rvllm-serve: OpenAI-compatible HTTP frontend.
//!
//! Layout:
//! - `worker`  : owns the engine on a `std::thread` (CUDA stream
//!               affinity, so it cannot live on the tokio executor).
//!               Communicates with the HTTP side via mpsc channels.
//! - `openai`  : request/response wire types matching the OpenAI
//!               `/v1/chat/completions` shape, plus chat-template
//!               rendering via minijinja.
//! - `http`    : axum router + handlers.

pub mod config;
pub mod http;
pub mod openai;
pub mod worker;

pub use config::ServerConfig;
pub use worker::{EngineReq, TokenEvent, WorkerHandle};
