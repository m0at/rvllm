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
/// default is 1.0 (stochastic), but on rvllm-serve that triggers a
/// per-token DtoH of the full 262k-element vocab + host-side sort,
/// AND outright fails on Qwen 3.6 (its worker only supports greedy
/// today and rejected stochastic with a self-contradicting "omit
/// sampling params" hint — exactly the failing path).
///
/// We flip the default to **greedy (0.0)** so a vanilla OpenAI client
/// that omits `temperature` works on every supported model and lands
/// on the fast path. Clients that *explicitly* send any non-zero
/// `temperature` still get the documented stochastic semantics.
///
/// To restore the OpenAI-spec default-1.0 behaviour for absent
/// temperature, set `RVLLM_DEFAULT_TEMPERATURE=1.0`.
pub fn default_temperature() -> f32 {
    std::env::var("RVLLM_DEFAULT_TEMPERATURE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0)
}
