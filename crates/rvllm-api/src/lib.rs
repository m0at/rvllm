#![forbid(unsafe_code)]
//! OpenAI-compatible API server for vllm-rs.
//!
//! Provides an axum-based HTTP server with endpoints matching the OpenAI API:
//! - `POST /v1/completions` -- text completion
//! - `POST /v1/chat/completions` -- chat completion
//! - `POST /v1/responses` -- unified responses API
//! - `POST /v1/embeddings` -- compute embeddings
//! - `GET /v1/models` -- list available models
//! - `POST /v1/batches` -- submit a batch of requests as JSONL
//! - `GET /v1/batches/{id}` -- check batch status
//! - `GET /v1/batches/{id}/output` -- retrieve batch results
//! - `POST /v1/batches/{id}/cancel` -- cancel a running batch
//! - `GET /health` -- health check
//! - `GET /metrics` -- Prometheus exposition

pub mod error;
pub mod routes;
pub mod server;
pub mod types;

#[cfg(test)]
mod test_support;

pub use server::{build_router, serve, AppState};
