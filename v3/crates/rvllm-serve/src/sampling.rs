//! Sampling parameters parsed from request bodies.
//!
//! v1 of the runtime only supports **greedy** decoding
//! (`run_generate` ends in `argmax`). We accept the full OpenAI
//! sampling surface in the request schema and **coerce** anything
//! non-greedy to greedy, with a `tracing::warn!` line on the first
//! occurrence of each distinct parameter so an operator sees it once
//! without log spam. Rationale: clients like zeroclaw and most
//! OpenAI SDKs default to `temperature=1.0` / `top_p=1.0`; rejecting
//! those with a 400 forced every caller to hard-code
//! `temperature=0`. Coercing silently surprises callers; coercing
//! with one warn-per-process is the honest middle.
//!
//! When a sampling kernel lands, this file becomes the place to route
//! to stochastic decode.

use crate::error::ApiError;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    /// OpenAI defaults: temperature 1.0, top_p 1.0, top_k unset.
    fn default() -> Self {
        Self { temperature: 1.0, top_p: 1.0, top_k: None, seed: None }
    }
}

/// Once-per-process warn latches for each non-greedy parameter.
static WARN_TEMPERATURE: OnceLock<()> = OnceLock::new();
static WARN_TOP_P:       OnceLock<()> = OnceLock::new();
static WARN_TOP_K:       OnceLock<()> = OnceLock::new();

impl SamplingParams {
    /// Coerce to greedy for the worker. Returns `GreedyParams` — the
    /// type-level proof that stochastic params never reach the
    /// runtime. Any non-default param triggers a one-shot WARN log.
    /// Negative values (temperature < 0, top_p < 0) are still a hard
    /// 400 because they're nonsensical, not just unsupported.
    pub fn ensure_supported(self) -> Result<GreedyParams, ApiError> {
        if self.temperature < 0.0 {
            return Err(ApiError::invalid_param(
                "temperature must be non-negative",
                "temperature",
                "invalid_value",
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::invalid_param(
                "top_p must be in [0, 1]",
                "top_p",
                "invalid_value",
            ));
        }
        if self.temperature != 0.0 {
            WARN_TEMPERATURE.get_or_init(|| {
                tracing::warn!(
                    temperature = self.temperature,
                    "coercing temperature to 0.0 (greedy) — sampling kernel \
                     not wired yet; this warning fires once per process"
                );
            });
        }
        if self.top_p != 1.0 {
            WARN_TOP_P.get_or_init(|| {
                tracing::warn!(
                    top_p = self.top_p,
                    "coercing top_p to 1.0 (greedy) — sampling kernel not \
                     wired yet; this warning fires once per process"
                );
            });
        }
        if self.top_k.is_some() {
            WARN_TOP_K.get_or_init(|| {
                tracing::warn!(
                    top_k = self.top_k,
                    "ignoring top_k — sampling kernel not wired yet; this \
                     warning fires once per process"
                );
            });
        }
        // `seed` is harmless with greedy decoding (deterministic anyway);
        // accept and ignore.
        Ok(GreedyParams { _seed: self.seed })
    }
}

/// Sampling parameters after validation — type-level proof that the
/// worker only ever sees greedy.
#[derive(Debug, Clone, Copy)]
pub struct GreedyParams {
    _seed: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_defaults_coerce_to_greedy() {
        // temperature=1.0 / top_p=1.0 is the OpenAI SDK default — must
        // not error, must coerce silently (one WARN log, fired once
        // per process).
        assert!(SamplingParams::default().ensure_supported().is_ok());
    }

    #[test]
    fn greedy_accepted() {
        let p = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: Some(42),
        };
        assert!(p.ensure_supported().is_ok());
    }

    #[test]
    fn non_default_top_p_accepted_and_coerced() {
        let p = SamplingParams { temperature: 0.0, top_p: 0.9, ..Default::default() };
        assert!(p.ensure_supported().is_ok());
    }

    #[test]
    fn negative_temperature_rejected() {
        let p = SamplingParams { temperature: -0.1, ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }

    #[test]
    fn out_of_range_top_p_rejected() {
        let p = SamplingParams { temperature: 0.0, top_p: 1.5, ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }
}
