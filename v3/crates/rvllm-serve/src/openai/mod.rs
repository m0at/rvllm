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

/// Round-22 finding #1: pick the effective temperature for a request,
/// honouring **raw field presence** of the other sampling knobs.
///
/// Previously we just unwrapped `temperature.unwrap_or_else(default_…)`,
/// which silently flipped any request shaped like `{"top_p": 0.9}`
/// (no temperature) into greedy — discarding `top_p`, `top_k`, and
/// `seed`. Worse, on Qwen 3.6 the arch-vs-sampling reject
/// (`reject_unsupported_sampling_for_arch`) only inspects
/// `is_greedy()`, so that misrouted greedy request slipped past the
/// 400 the helper's error message advertised for `top_p<1`.
///
/// Resolution order:
/// 1. Caller supplied `temperature` explicitly → use it as-is.
/// 2. Caller didn't supply `temperature` but DID supply any other
///    sampling knob (`top_p`, `top_k`, `seed`) → treat as a stochastic
///    request and default to OpenAI's documented `temperature=1.0`,
///    so the supplied knobs actually take effect.
/// 3. Caller supplied nothing → fall back to the server-side default
///    (greedy unless `RVLLM_DEFAULT_TEMPERATURE` is set).
pub(crate) fn resolve_temperature(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
) -> f32 {
    if let Some(t) = temperature {
        return t;
    }
    let stochastic_signal = top_p.is_some() || top_k.is_some() || seed.is_some();
    if stochastic_signal {
        1.0
    } else {
        default_temperature()
    }
}
