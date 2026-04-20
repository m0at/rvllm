//! Sampling parameters parsed from request bodies.
//!
//! v1 of the runtime only supports **greedy** decoding
//! (`run_generate` ends in `argmax`). We accept the full OpenAI
//! sampling surface in the request schema but reject any non-default
//! value with a 400 — honest about the limitation, so callers don't
//! silently get greedy output when they asked for `temperature=1.0`.
//!
//! When a sampling kernel lands, the rejection rules below relax.

use crate::error::ApiError;

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

impl SamplingParams {
    /// Accept only parameter combinations the runtime can honour today.
    /// v1 = greedy only. Returns `ApiError::InvalidRequest` otherwise.
    pub fn ensure_supported(self) -> Result<GreedyParams, ApiError> {
        // Temperature 0.0 is "greedy" by OpenAI convention; treat
        // anything not exactly 0.0 as "wants sampling".
        if self.temperature != 0.0 {
            return Err(ApiError::invalid_param(
                "temperature must be 0 — sampling kernel not yet wired",
                "temperature",
                "sampling_unsupported",
            ));
        }
        if self.top_p != 1.0 {
            return Err(ApiError::invalid_param(
                "top_p must be 1.0 — sampling kernel not yet wired",
                "top_p",
                "sampling_unsupported",
            ));
        }
        if self.top_k.is_some() {
            return Err(ApiError::invalid_param(
                "top_k unsupported — sampling kernel not yet wired",
                "top_k",
                "sampling_unsupported",
            ));
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
    fn default_params_reject_because_temp_is_one() {
        assert!(SamplingParams::default().ensure_supported().is_err());
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
    fn non_default_top_p_rejected() {
        let p = SamplingParams { temperature: 0.0, top_p: 0.9, ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }
}
