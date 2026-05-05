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
pub mod vision_fetch;

/// Default temperature when the client omits it. OpenAI's documented
/// default is 1.0 (stochastic). On rvllm-serve the stochastic path
/// does a per-token DtoH-copy of the full vocab and host-side sort
/// — expensive on Gemma 4's 262k vocab. Operators running a
/// throughput benchmark or deployment that does not need temperature
/// can flip the default to greedy with `RVLLM_DEFAULT_TEMPERATURE=0`.
/// Clients that explicitly send `"temperature": …` always win.
pub fn default_temperature() -> f32 {
    std::env::var("RVLLM_DEFAULT_TEMPERATURE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0)
}
