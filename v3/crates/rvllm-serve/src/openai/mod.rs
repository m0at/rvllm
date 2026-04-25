//! OpenAI-compatible request + response schemas.
//!
//! Each submodule owns one endpoint family:
//!   * [`models`]     — `GET /v1/models`
//!   * [`chat`]       — `POST /v1/chat/completions`
//!   * [`completions`] — `POST /v1/completions`
//!
//! Shared primitives (role, finish reason, usage) live in [`types`].

pub mod chat;
pub mod completions;
pub mod handlers;
pub mod models;
pub mod types;
